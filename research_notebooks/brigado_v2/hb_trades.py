import json
import pandas as pd
from core.data_sources.hummingbot_database import HummingbotDatabase


class EnrichedHummingbotDatabase(HummingbotDatabase):
    """
    Extended HummingbotDatabase class that provides methods to merge
    Controllers, Executors, Orders, and TradeFills data with JSON field expansion.
    """

    def get_trade_fills(self, config_file_path=None, start_date=None, end_date=None):
        """
        Override parent method to NOT calculate cumulative PnL.
        We'll recalculate it properly grouped by controller_id in get_enriched_trade_fills.

        Returns:
            DataFrame with basic trade fills (no cumulative calculations)
        """
        float_cols = ["amount", "price", "trade_fee_in_quote"]
        query = "SELECT * FROM TradeFill"
        trade_fills = pd.read_sql_query(query, self.connection)

        # Basic conversions only
        trade_fills[float_cols] = trade_fills[float_cols] / 1e6
        trade_fills["net_amount"] = trade_fills['amount'] * trade_fills['trade_type'].apply(
            lambda x: 1 if x == 'BUY' else -1)
        trade_fills["net_amount_quote"] = trade_fills['net_amount'] * trade_fills['price']
        trade_fills["timestamp"] = pd.to_datetime(trade_fills["timestamp"], unit="ms")
        trade_fills["quote_volume"] = trade_fills["price"] * trade_fills["amount"]

        return trade_fills

    def get_enriched_trade_fills(self) -> pd.DataFrame:
        """
        Get TradeFills enriched with controller_id and accurate PnL calculations.

        Returns:
            DataFrame with trade fills including controller_id, executor metadata, and accurate PnL metrics
        """
        # Load base tables
        trade_fills = self.get_trade_fills()
        orders = self.get_orders()
        executors = self.get_executors_data()
        controllers = self.get_controller_data()

        # Expand executor custom_info to extract order_ids
        # Create a mapping of order_id -> executor_id -> controller_id
        order_mapping_records = []
        for _, executor in executors.iterrows():
            custom_info = executor['custom_info']
            if isinstance(custom_info, dict) and 'order_ids' in custom_info:
                order_ids = custom_info['order_ids']
                for order_id in order_ids:
                    order_mapping_records.append({
                        'order_id': order_id,
                        'executor_id': executor['id'],
                        'controller_id': executor['controller_id'],
                        'level_id': custom_info.get('level_id'),
                        'side': custom_info.get('side'),
                        'current_position_average_price': custom_info.get('current_position_average_price'),
                        'close_price': custom_info.get('close_price'),
                    })

        order_executor_df = pd.DataFrame(order_mapping_records)

        # Merge trade_fills with order->executor mapping
        enriched_trades = trade_fills.merge(
            order_executor_df,
            on='order_id',
            how='left'
        )

        # Sort by timestamp for accurate cumulative calculations
        enriched_trades = enriched_trades.sort_values('timestamp')

        # Merge with executors to get additional executor data
        executor_cols = ['id', 'timestamp', 'type', 'close_type', 'close_timestamp',
                         'status', 'net_pnl_pct', 'net_pnl_quote', 'cum_fees_quote',
                         'filled_amount_quote', 'is_active', 'is_trading']
        enriched_trades = enriched_trades.merge(
            executors[executor_cols].rename(columns={'id': 'executor_id', 'timestamp': 'executor_timestamp'}),
            on='executor_id',
            how='left',
            suffixes=('', '_executor')
        )

        # Expand controller config keys - optimized for pmm_mister strategy
        controller_config_expanded = []
        for _, controller in controllers.iterrows():
            config = controller['config']
            # Full PMM Mister config expansion
            config_flat = {
                'controller_id': controller['id'],
                'controller_name': config.get('controller_name'),
                'controller_type': config.get('controller_type'),
                'total_amount_quote': config.get('total_amount_quote'),
                'manual_kill_switch': config.get('manual_kill_switch'),
                'connector_name': config.get('connector_name'),
                'trading_pair': config.get('trading_pair'),
                'portfolio_allocation': config.get('portfolio_allocation'),
                'target_base_pct': config.get('target_base_pct'),
                'min_base_pct': config.get('min_base_pct'),
                'max_base_pct': config.get('max_base_pct'),
                'buy_spreads': str(config.get('buy_spreads')),  # Convert list to string
                'sell_spreads': str(config.get('sell_spreads')),  # Convert list to string
                'buy_amounts_pct': str(config.get('buy_amounts_pct')),  # Convert list to string
                'sell_amounts_pct': str(config.get('sell_amounts_pct')),  # Convert list to string
                'executor_refresh_time': config.get('executor_refresh_time'),
                'buy_cooldown_time': config.get('buy_cooldown_time'),
                'sell_cooldown_time': config.get('sell_cooldown_time'),
                'buy_position_effectivization_time': config.get('buy_position_effectivization_time'),
                'sell_position_effectivization_time': config.get('sell_position_effectivization_time'),
                'min_buy_price_distance_pct': config.get('min_buy_price_distance_pct'),
                'min_sell_price_distance_pct': config.get('min_sell_price_distance_pct'),
                'leverage': config.get('leverage'),
                'position_mode': config.get('position_mode'),
                'take_profit': config.get('take_profit'),
                'take_profit_order_type': config.get('take_profit_order_type'),
                'max_active_executors_by_level': config.get('max_active_executors_by_level'),
                'tick_mode': config.get('tick_mode'),
            }
            controller_config_expanded.append(config_flat)

        controller_config_df = pd.DataFrame(controller_config_expanded)

        # Merge with controller config
        enriched_trades = enriched_trades.merge(
            controller_config_df,
            on='controller_id',
            how='left',
            suffixes=('', '_controller')
        )

        # Recalculate PnL metrics grouped by controller_id
        # Group by controller_id, market, symbol for proper cumulative calculations
        groupers = ["controller_id", "market", "symbol"]

        # For trades without controller_id (orphans), use original grouping
        has_controller = enriched_trades['controller_id'].notna()
        orphan_trades = enriched_trades[~has_controller].copy()
        controller_trades = enriched_trades[has_controller].copy()

        # Calculate metrics for controller trades
        if not controller_trades.empty:
            controller_trades["cum_fees_in_quote"] = controller_trades.groupby(groupers)["trade_fee_in_quote"].cumsum()
            controller_trades["cum_net_amount"] = controller_trades.groupby(groupers)["net_amount"].cumsum()
            controller_trades["unrealized_trade_pnl"] = -1 * controller_trades.groupby(groupers)["net_amount_quote"].cumsum()
            controller_trades["inventory_cost"] = controller_trades["cum_net_amount"] * controller_trades["price"]
            controller_trades["realized_trade_pnl"] = controller_trades["unrealized_trade_pnl"] + controller_trades["inventory_cost"]
            controller_trades["net_realized_pnl"] = controller_trades["realized_trade_pnl"] - controller_trades["cum_fees_in_quote"]
            controller_trades["realized_pnl"] = controller_trades.groupby(groupers)["net_realized_pnl"].diff()
            controller_trades["gross_pnl"] = controller_trades.groupby(groupers)["realized_trade_pnl"].diff()
            controller_trades["trade_fee"] = controller_trades.groupby(groupers)["cum_fees_in_quote"].diff()

        # Calculate metrics for orphan trades (use original grouping)
        if not orphan_trades.empty:
            orphan_groupers = ["config_file_path", "market", "symbol"]
            orphan_trades["cum_fees_in_quote"] = orphan_trades.groupby(orphan_groupers)["trade_fee_in_quote"].cumsum()
            orphan_trades["cum_net_amount"] = orphan_trades.groupby(orphan_groupers)["net_amount"].cumsum()
            orphan_trades["unrealized_trade_pnl"] = -1 * orphan_trades.groupby(orphan_groupers)["net_amount_quote"].cumsum()
            orphan_trades["inventory_cost"] = orphan_trades["cum_net_amount"] * orphan_trades["price"]
            orphan_trades["realized_trade_pnl"] = orphan_trades["unrealized_trade_pnl"] + orphan_trades["inventory_cost"]
            orphan_trades["net_realized_pnl"] = orphan_trades["realized_trade_pnl"] - orphan_trades["cum_fees_in_quote"]
            orphan_trades["realized_pnl"] = orphan_trades.groupby(orphan_groupers)["net_realized_pnl"].diff()
            orphan_trades["gross_pnl"] = orphan_trades.groupby(orphan_groupers)["realized_trade_pnl"].diff()
            orphan_trades["trade_fee"] = orphan_trades.groupby(orphan_groupers)["cum_fees_in_quote"].diff()

        # Combine back
        enriched_trades = pd.concat([controller_trades, orphan_trades], ignore_index=True)
        enriched_trades = enriched_trades.sort_values('timestamp').reset_index(drop=True)

        return enriched_trades

    def get_order_executor_mapping(self) -> pd.DataFrame:
        """
        Create a mapping table between orders and executors via custom_info.order_ids.

        Returns:
            DataFrame with columns: order_id, executor_id, controller_id, level_id
        """
        executors = self.get_executors_data()

        mappings = []
        for _, executor in executors.iterrows():
            executor_id = executor['id']
            controller_id = executor['controller_id']
            custom_info = executor['custom_info']

            if isinstance(custom_info, dict) and 'order_ids' in custom_info:
                order_ids = custom_info['order_ids']
                for order_id in order_ids:
                    mappings.append({
                        'order_id': order_id,
                        'executor_id': executor_id,
                        'controller_id': controller_id,
                        'level_id': custom_info.get('level_id'),
                    })

        return pd.DataFrame(mappings)

    def get_executors_with_expanded_config(self) -> pd.DataFrame:
        """
        Get executors with config JSON fields expanded into separate columns.

        Returns:
            DataFrame with executors and expanded config fields
        """
        executors = self.get_executors_data()

        # Expand config fields
        config_expanded = []
        for _, executor in executors.iterrows():
            config = executor['config']
            if isinstance(config, dict):
                config_flat = {
                    'executor_id': executor['id'],
                    'config_type': config.get('type'),
                    'config_controller_id': config.get('controller_id'),
                    'config_trading_pair': config.get('trading_pair'),
                    'config_connector_name': config.get('connector_name'),
                    'config_side': config.get('side'),
                    'config_entry_price': config.get('entry_price'),
                    'config_amount': config.get('amount'),
                    'config_leverage': config.get('leverage'),
                    'config_level_id': config.get('level_id'),
                }
                config_expanded.append(config_flat)

        config_df = pd.DataFrame(config_expanded)

        # Expand custom_info fields
        custom_info_expanded = []
        for _, executor in executors.iterrows():
            custom_info = executor['custom_info']
            if isinstance(custom_info, dict):
                custom_flat = {
                    'executor_id': executor['id'],
                    'level_id': custom_info.get('level_id'),
                    'current_position_average_price': custom_info.get('current_position_average_price'),
                    'side': custom_info.get('side'),
                    'current_retries': custom_info.get('current_retries'),
                    'max_retries': custom_info.get('max_retries'),
                    'close_price': custom_info.get('close_price'),
                    'order_ids_count': len(custom_info.get('order_ids', [])),
                }
                custom_info_expanded.append(custom_flat)

        custom_info_df = pd.DataFrame(custom_info_expanded)

        # Merge with original executors
        executors_expanded = executors.merge(config_df, left_on='id', right_on='executor_id', how='left')
        executors_expanded = executors_expanded.merge(custom_info_df, left_on='id', right_on='executor_id',
                                                       how='left', suffixes=('', '_custom'))

        # Drop duplicate executor_id columns
        executors_expanded = executors_expanded.loc[:, ~executors_expanded.columns.duplicated()]

        return executors_expanded

    def get_controllers_with_expanded_config(self) -> pd.DataFrame:
        """
        Get controllers with config JSON fields expanded into separate columns.

        Returns:
            DataFrame with controllers and expanded config fields
        """
        controllers = self.get_controller_data()

        # Expand all config fields
        config_expanded = []
        for _, controller in controllers.iterrows():
            config = controller['config']
            if isinstance(config, dict):
                config_flat = {
                    'controller_id': controller['id'],
                    **config  # Expand all config keys
                }
                config_expanded.append(config_flat)

        config_df = pd.DataFrame(config_expanded)

        # Merge with original controllers
        controllers_expanded = controllers.merge(config_df, left_on='id', right_on='controller_id', how='left')
        controllers_expanded = controllers_expanded.loc[:, ~controllers_expanded.columns.duplicated()]

        return controllers_expanded


# Example usage
if __name__ == "__main__":
    # Initialize database
    db = EnrichedHummingbotDatabase(
        db_name="pmm-mister-20-1-20251223-1025-20251223-102533.sqlite",
        root_path="/Users/tomasgaudino/PycharmProjects/quants-lab",
        server_name="brigado_server"
    )

    # Get enriched trade fills with controller_id
    enriched_trades = db.get_enriched_trade_fills()
    print("Enriched Trade Fills (first 5 rows):")
    print(enriched_trades.head())
    print(f"\nShape: {enriched_trades.shape}")
    print(f"\nColumns: {enriched_trades.columns.tolist()}")

    # Get order->executor mapping
    print("\n" + "="*80)
    print("Order-Executor Mapping (first 5 rows):")
    order_executor_map = db.get_order_executor_mapping()
    print(order_executor_map.head())

    # Get executors with expanded config
    print("\n" + "="*80)
    print("Executors with Expanded Config (first 5 rows):")
    executors_expanded = db.get_executors_with_expanded_config()
    print(executors_expanded.head())

    # Get controllers with expanded config
    print("\n" + "="*80)
    print("Controllers with Expanded Config:")
    controllers_expanded = db.get_controllers_with_expanded_config()
    print(controllers_expanded.head())

    # Check controller_id coverage in trade_fills
    print("\n" + "="*80)
    print("Controller ID Coverage Analysis:")
    total_trades = len(enriched_trades)
    trades_with_controller = enriched_trades['controller_id'].notna().sum()
    print(f"Total trades: {total_trades}")
    print(f"Trades with controller_id: {trades_with_controller}")
    print(f"Coverage: {trades_with_controller/total_trades*100:.2f}%")
    enriched_trades.to_csv("trades_with_controller_id.csv", index=False)
