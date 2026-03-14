"""
Trade Enricher Module

Enriches trade fills with controller_id, executor metadata, and accurate PnL calculations.

TODO - Future Improvements:
- Add support for multiple attribution methods (FIFO, LIFO, VWAP)
- Implement position reconciliation checks
- Add trade matching validation
- Support for partial fills and amendments
"""

from typing import Optional
import pandas as pd
import numpy as np


class TradeEnricher:
    """
    Enriches trade fills with controller attribution and PnL calculations.

    Responsibilities:
    - Map trades to executors via order_id
    - Map executors to controllers
    - Calculate accurate PnL metrics grouped by controller
    - Expand controller and executor metadata

    IMPORTANT: Follows portfolio-level accounting principles from hb_calculator_instructions.md
    - Single source of truth for positions
    - Strategy attribution is non-ownership
    - PnL calculated at portfolio level, then attributed
    """

    def __init__(self):
        """Initialize trade enricher."""
        pass

    def enrich_trades(
        self,
        trade_fills: pd.DataFrame,
        orders: pd.DataFrame,
        executors: pd.DataFrame,
        controllers: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Enrich trade fills with controller_id and accurate PnL calculations.

        Args:
            trade_fills: Raw trade fills from database
            orders: Orders from database
            executors: Executors with config and custom_info
            controllers: Controllers with config

        Returns:
            Enriched trades with controller attribution and PnL metrics

        TODO:
        - Add validation for orphan trades
        - Implement configurable grouping strategies
        - Add position reconciliation warnings
        """
        # Create order->executor->controller mapping
        order_mapping = self._create_order_mapping(executors)

        # Merge trades with mapping
        enriched = trade_fills.merge(
            order_mapping,
            on='order_id',
            how='left'
        )

        # Sort by timestamp for accurate cumulative calculations
        enriched = enriched.sort_values('timestamp')

        # Merge with executor details
        enriched = self._merge_executor_details(enriched, executors)

        # Merge with controller config
        enriched = self._merge_controller_config(enriched, controllers)

        # Recalculate PnL metrics grouped by controller_id
        enriched = self._recalculate_pnl_metrics(enriched)

        return enriched

    def _create_order_mapping(self, executors: pd.DataFrame) -> pd.DataFrame:
        """
        Create mapping from order_id to executor_id to controller_id.

        CRITICAL: Deduplicates order_ids to avoid duplicate trades.
        Executors may have duplicate order_ids in their custom_info arrays.

        Args:
            executors: DataFrame with executors

        Returns:
            DataFrame with order_id, executor_id, controller_id, and metadata

        TODO:
        - Add validation for many-to-many relationships
        - Add warnings for duplicate mappings
        - Track order_id reuse across executors
        """
        order_mapping_records = []

        for _, executor in executors.iterrows():
            custom_info = executor['custom_info']
            if isinstance(custom_info, dict) and 'order_ids' in custom_info:
                # CRITICAL: Deduplicate order_ids using set to avoid duplicate trades
                order_ids = list(set(custom_info['order_ids']))
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

        return pd.DataFrame(order_mapping_records)

    def _merge_executor_details(
        self,
        enriched: pd.DataFrame,
        executors: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge executor metadata into enriched trades.

        Args:
            enriched: Trades with controller_id
            executors: Executors data

        Returns:
            Enriched trades with executor details

        TODO:
        - Add executor performance metrics (net_pnl_pct, filled_amount)
        - Add executor lifecycle metadata (status, close_type)
        """
        executor_cols = [
            'id', 'timestamp', 'type', 'close_type', 'close_timestamp',
            'status', 'net_pnl_pct', 'net_pnl_quote', 'cum_fees_quote',
            'filled_amount_quote', 'is_active', 'is_trading'
        ]

        return enriched.merge(
            executors[executor_cols].rename(
                columns={'id': 'executor_id', 'timestamp': 'executor_timestamp'}
            ),
            on='executor_id',
            how='left',
            suffixes=('', '_executor')
        )

    def _merge_controller_config(
        self,
        enriched: pd.DataFrame,
        controllers: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge controller configuration into enriched trades.

        Args:
            enriched: Trades with executor details
            controllers: Controllers with config

        Returns:
            Enriched trades with controller config

        TODO:
        - Add dynamic config expansion based on controller_type
        - Support for different strategy configs (not just PMM Mister)
        - Add config validation
        """
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
                'buy_spreads': str(config.get('buy_spreads')),
                'sell_spreads': str(config.get('sell_spreads')),
                'buy_amounts_pct': str(config.get('buy_amounts_pct')),
                'sell_amounts_pct': str(config.get('sell_amounts_pct')),
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

        return enriched.merge(
            controller_config_df,
            on='controller_id',
            how='left',
            suffixes=('', '_controller')
        )

    def _recalculate_pnl_metrics(self, enriched: pd.DataFrame) -> pd.DataFrame:
        """
        Recalculate PnL metrics grouped by controller_id.

        CRITICAL: This ensures accurate attribution when multiple controllers
        trade the same instrument on a shared portfolio.

        Args:
            enriched: Trades with controller and executor metadata

        Returns:
            Enriched trades with accurate PnL calculations

        TODO:
        - Add support for multiple grouping strategies (by controller, by pair, etc.)
        - Implement configurable PnL calculation methods
        - Add position reconciliation checks
        - Support for cross-currency PnL
        """
        # Group by controller_id, market, symbol for proper cumulative calculations
        groupers = ["controller_id", "market", "symbol"]

        # Separate controller trades from orphan trades
        has_controller = enriched['controller_id'].notna()
        orphan_trades = enriched[~has_controller].copy()
        controller_trades = enriched[has_controller].copy()

        # Calculate metrics for controller trades
        if not controller_trades.empty:
            controller_trades = self._calculate_pnl_for_group(
                controller_trades,
                groupers
            )

        # Calculate metrics for orphan trades (use original grouping)
        if not orphan_trades.empty:
            orphan_groupers = ["config_file_path", "market", "symbol"]
            orphan_trades = self._calculate_pnl_for_group(
                orphan_trades,
                orphan_groupers
            )

        # Combine back
        enriched = pd.concat([controller_trades, orphan_trades], ignore_index=True)
        enriched = enriched.sort_values('timestamp').reset_index(drop=True)

        return enriched

    def _calculate_pnl_for_group(
        self,
        trades: pd.DataFrame,
        groupers: list
    ) -> pd.DataFrame:
        """
        Calculate PnL metrics for a specific grouping.

        Implements standard inventory-based PnL calculation:
        - Cumulative position tracking
        - Unrealized PnL = -sum(net_amount_quote)
        - Realized PnL = unrealized + inventory_cost
        - Net PnL = realized - fees

        Args:
            trades: Trades to calculate PnL for
            groupers: List of columns to group by

        Returns:
            Trades with PnL calculations

        TODO:
        - Add support for different accounting methods (FIFO, LIFO, Average)
        - Add position limits validation
        - Implement cross-instrument PnL aggregation
        """
        trades["cum_fees_in_quote"] = trades.groupby(groupers)["trade_fee_in_quote"].cumsum()
        trades["cum_net_amount"] = trades.groupby(groupers)["net_amount"].cumsum()
        trades["unrealized_trade_pnl"] = -1 * trades.groupby(groupers)["net_amount_quote"].cumsum()
        trades["inventory_cost"] = trades["cum_net_amount"] * trades["price"]
        trades["realized_trade_pnl"] = trades["unrealized_trade_pnl"] + trades["inventory_cost"]
        trades["net_realized_pnl"] = trades["realized_trade_pnl"] - trades["cum_fees_in_quote"]

        # CRITICAL: Use diff() for incremental PnL, not cumulative
        trades["realized_pnl"] = trades.groupby(groupers)["net_realized_pnl"].diff()
        trades["gross_pnl"] = trades.groupby(groupers)["realized_trade_pnl"].diff()
        trades["trade_fee"] = trades.groupby(groupers)["cum_fees_in_quote"].diff()

        return trades
