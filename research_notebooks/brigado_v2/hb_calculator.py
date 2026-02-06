import asyncio
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

from core.data_sources.clob import CLOBDataSource


class HummingbotCalculator:
    """
    Calculator to generate daily performance reports from enriched trade data.
    Groups by controller_id and calculates market and bot metrics.
    """

    def __init__(self, trades_csv_path: str, root_path: str = ""):
        """
        Initialize the calculator with enriched trades data.

        Args:
            trades_csv_path: Path to the trades_with_controller_id.csv file
            root_path: Root path for the project
        """
        self.trades_df = pd.read_csv(trades_csv_path)
        self.trades_df['timestamp'] = pd.to_datetime(self.trades_df['timestamp'])
        self.root_path = root_path
        self.clob = CLOBDataSource()

    async def get_market_metrics(
        self,
        connector_name: str,
        trading_pair: str,
        start_date: datetime,
        end_date: datetime
    ) -> dict:
        """
        Get market metrics for a specific date range using daily candles.

        Args:
            connector_name: Exchange connector name (e.g., 'binance')
            trading_pair: Trading pair (e.g., 'USDT-BRL')
            start_date: Start date for metrics
            end_date: End date for metrics

        Returns:
            Dictionary with market metrics
        """
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())

        # Get daily candles - Candles object has .data attribute containing the DataFrame
        candles = await self.clob.get_candles(
            connector_name=connector_name,
            trading_pair=trading_pair,
            interval='1d',
            start_time=start_timestamp,
            end_time=end_timestamp
        )

        # Access the DataFrame through .data attribute
        df = candles.data

        if df.empty:
            return {
                'market_volume_usdt': 0.0,
                'market_price_min': 0.0,
                'market_price_max': 0.0,
                'market_price_var_pct': 0.0,
                'market_trades_count': 0
            }

        return {
            'market_volume_usdt': float(df['volume'].sum()),
            'market_price_min': float(df['low'].min()),
            'market_price_max': float(df['high'].max()),
            'market_price_var_pct': float(
                ((df['high'].max() - df['low'].min()) / df['low'].min() * 100)
                if df['low'].min() > 0 else 0
            ),
            'market_trades_count': int(df['n_trades'].sum()) if 'n_trades' in df.columns else 0
        }

    def calculate_break_even_price(
        self,
        trades: pd.DataFrame,
        trade_type: str
    ) -> float:
        """
        Calculate break-even price for buy or sell trades.

        Args:
            trades: DataFrame of trades filtered by type
            trade_type: 'BUY' or 'SELL'

        Returns:
            Break-even price
        """
        filtered = trades[trades['trade_type'] == trade_type]
        if filtered.empty or filtered['amount'].sum() == 0:
            return 0.0

        # Weighted average price by volume
        total_value = (filtered['price'] * filtered['amount']).sum()
        total_volume = filtered['amount'].sum()
        return float(total_value / total_volume)

    def calculate_position_inventory(
        self,
        trades: pd.DataFrame,
        up_to_timestamp: Optional[pd.Timestamp] = None
    ) -> float:
        """
        Calculate net base asset position from trades.

        Args:
            trades: DataFrame of trades
            up_to_timestamp: Calculate position up to this timestamp (None for all trades)

        Returns:
            Net position in base asset (positive = long, negative = short)
        """
        if up_to_timestamp is not None:
            trades = trades[trades['timestamp'] <= up_to_timestamp]

        if trades.empty:
            return 0.0

        # Net amount: BUY adds to position, SELL subtracts
        position = trades['net_amount'].sum()
        return float(position)

    def calculate_global_bot_metrics(
        self,
        daily_trades: pd.DataFrame,
        previous_day_end_position: float = 0.0
    ) -> dict:
        """
        Calculate global bot metrics for a day.

        Args:
            daily_trades: Trades for a specific day
            previous_day_end_position: Position at end of previous day

        Returns:
            Dictionary with global bot metrics
        """
        if daily_trades.empty:
            return {
                'bot_volume_base': 0.0,
                'bot_volume_quote': 0.0,
                'bot_realized_pnl': 0.0,
                'bot_unrealized_pnl': 0.0,
                'bot_trades_count': 0,
                'bot_orphan_trades_count': 0,
                'bot_market_share_pct': 0.0,
                'bot_buy_break_even': 0.0,
                'bot_sell_break_even': 0.0,
                'bot_initial_position': previous_day_end_position,
                'bot_final_position': previous_day_end_position,
            }

        # Volume metrics
        # amount = base asset volume (USDT)
        # quote_volume = quote asset volume (BRL) = price * amount
        volume_base = daily_trades['amount'].sum()
        volume_quote = daily_trades['quote_volume'].sum()

        # PnL metrics
        # Realized PnL is the net_realized_pnl from the last trade (cumulative)
        realized_pnl = daily_trades['net_realized_pnl'].iloc[-1] if 'net_realized_pnl' in daily_trades.columns else 0.0

        # Unrealized PnL: current position * last price - inventory cost
        final_position = previous_day_end_position + daily_trades['net_amount'].sum()
        last_price = daily_trades['price'].iloc[-1]
        inventory_cost = daily_trades['inventory_cost'].iloc[-1] if 'inventory_cost' in daily_trades.columns else 0.0
        unrealized_pnl = (final_position * last_price) - inventory_cost

        # Trade counts
        total_trades = len(daily_trades)
        orphan_trades = daily_trades['controller_id'].isna().sum()

        # Break-even prices
        buy_break_even = self.calculate_break_even_price(daily_trades, 'BUY')
        sell_break_even = self.calculate_break_even_price(daily_trades, 'SELL')

        return {
            'bot_volume_base': float(volume_base),
            'bot_volume_quote': float(volume_quote),
            'bot_realized_pnl': float(realized_pnl),
            'bot_unrealized_pnl': float(unrealized_pnl),
            'bot_trades_count': int(total_trades),
            'bot_orphan_trades_count': int(orphan_trades),
            'bot_market_share_pct': 0.0,  # Calculated later when market volume is available
            'bot_buy_break_even': float(buy_break_even),
            'bot_sell_break_even': float(sell_break_even),
            'bot_initial_position': float(previous_day_end_position),
            'bot_final_position': float(final_position),
        }

    def calculate_controller_metrics(
        self,
        controller_trades: pd.DataFrame,
        controller_id: str
    ) -> dict:
        """
        Calculate metrics for a specific controller.

        Args:
            controller_trades: Trades for a specific controller
            controller_id: Controller identifier

        Returns:
            Dictionary with controller-specific metrics
        """
        if controller_trades.empty:
            return {
                'controller_id': controller_id,
                'controller_volume_base': 0.0,
                'controller_volume_quote': 0.0,
                'controller_realized_pnl': 0.0,
                'controller_unrealized_pnl': 0.0,
                'controller_trades_count': 0,
                'controller_buy_break_even': 0.0,
                'controller_sell_break_even': 0.0,
            }

        # Volume metrics
        volume_base = controller_trades['amount'].sum()
        volume_quote = controller_trades['quote_volume'].sum()

        # PnL metrics
        realized_pnl = controller_trades['net_realized_pnl'].iloc[-1] if 'net_realized_pnl' in controller_trades.columns else 0.0

        # Unrealized PnL
        final_position = controller_trades['net_amount'].sum()
        last_price = controller_trades['price'].iloc[-1]
        inventory_cost = controller_trades['inventory_cost'].iloc[-1] if 'inventory_cost' in controller_trades.columns else 0.0
        unrealized_pnl = (final_position * last_price) - inventory_cost

        # Trade count
        total_trades = len(controller_trades)

        # Break-even prices
        buy_break_even = self.calculate_break_even_price(controller_trades, 'BUY')
        sell_break_even = self.calculate_break_even_price(controller_trades, 'SELL')

        return {
            'controller_id': controller_id,
            'controller_volume_base': float(volume_base),
            'controller_volume_quote': float(volume_quote),
            'controller_realized_pnl': float(realized_pnl),
            'controller_unrealized_pnl': float(unrealized_pnl),
            'controller_trades_count': int(total_trades),
            'controller_buy_break_even': float(buy_break_even),
            'controller_sell_break_even': float(sell_break_even),
        }

    async def generate_daily_report(
        self,
        output_path: str = "daily_performance_report.csv"
    ) -> pd.DataFrame:
        """
        Generate daily performance report grouped by date and controller_id.

        Args:
            output_path: Path to save the CSV report

        Returns:
            DataFrame with daily performance metrics
        """
        # Add date column
        self.trades_df['date'] = self.trades_df['timestamp'].dt.date

        # Get unique dates
        dates = sorted(self.trades_df['date'].unique())

        all_reports = []
        previous_position = 0.0

        for date in dates:
            print(f"Processing date: {date}")

            # Filter trades for this date
            daily_trades = self.trades_df[self.trades_df['date'] == date].copy()
            daily_trades = daily_trades.sort_values('timestamp')

            # Get market info
            if not daily_trades.empty:
                connector_name = daily_trades['market'].iloc[0]
                trading_pair = daily_trades['symbol'].iloc[0]

                # Get market metrics
                start_datetime = datetime.combine(date, datetime.min.time())
                end_datetime = datetime.combine(date, datetime.max.time())

                try:
                    market_metrics = await self.get_market_metrics(
                        connector_name=connector_name,
                        trading_pair=trading_pair,
                        start_date=start_datetime,
                        end_date=end_datetime
                    )
                except Exception as e:
                    print(f"Warning: Could not fetch market data for {date}: {e}")
                    market_metrics = {
                        'market_volume_usdt': 0.0,
                        'market_price_min': 0.0,
                        'market_price_max': 0.0,
                        'market_price_var_pct': 0.0,
                        'market_trades_count': 0
                    }

                # Calculate global bot metrics
                global_metrics = self.calculate_global_bot_metrics(
                    daily_trades,
                    previous_day_end_position=previous_position
                )

                # Calculate market share (using base volume - USDT)
                if market_metrics['market_volume_usdt'] > 0:
                    global_metrics['bot_market_share_pct'] = (
                        global_metrics['bot_volume_base'] / market_metrics['market_volume_usdt'] * 100
                    )

                # Get unique controllers for this day
                controllers = daily_trades['controller_id'].dropna().unique()

                # Create report for this date
                base_report = {
                    'date': date,
                    **market_metrics,
                    **global_metrics
                }

                # Add controller-specific metrics
                for controller_id in controllers:
                    controller_trades = daily_trades[
                        daily_trades['controller_id'] == controller_id
                    ].copy()

                    controller_metrics = self.calculate_controller_metrics(
                        controller_trades,
                        controller_id
                    )

                    # Merge controller metrics with base report
                    report = {**base_report, **controller_metrics}
                    all_reports.append(report)

                # If no controllers, add report with global metrics only
                if len(controllers) == 0:
                    all_reports.append(base_report)

                # Update position for next day
                previous_position = global_metrics['bot_final_position']

        # Create DataFrame
        report_df = pd.DataFrame(all_reports)

        # Sort columns for better readability
        column_order = [
            'date', 'controller_id',
            # Market metrics
            'market_volume_usdt', 'market_price_min', 'market_price_max',
            'market_price_var_pct', 'market_trades_count',
            # Global bot metrics
            'bot_volume_base', 'bot_volume_quote', 'bot_realized_pnl',
            'bot_unrealized_pnl', 'bot_trades_count', 'bot_orphan_trades_count',
            'bot_market_share_pct', 'bot_buy_break_even', 'bot_sell_break_even',
            'bot_initial_position', 'bot_final_position',
            # Controller metrics
            'controller_volume_base', 'controller_volume_quote',
            'controller_realized_pnl', 'controller_unrealized_pnl',
            'controller_trades_count', 'controller_buy_break_even',
            'controller_sell_break_even',
        ]

        # Reorder columns (keep any extra columns at the end)
        existing_cols = [col for col in column_order if col in report_df.columns]
        extra_cols = [col for col in report_df.columns if col not in column_order]
        report_df = report_df[existing_cols + extra_cols]

        # Save to CSV
        report_df.to_csv(output_path, index=False)
        print(f"Report saved to: {output_path}")

        return report_df


# Example usage
async def main():
    # Initialize calculator
    calculator = HummingbotCalculator(
        trades_csv_path="trades_with_controller_id.csv",
        root_path="/Users/tomasgaudino/PycharmProjects/quants-lab"
    )

    # Generate daily report
    report_df = await calculator.generate_daily_report(
        output_path="daily_performance_report.csv"
    )

    print("\n" + "="*80)
    print("Daily Performance Report Summary:")
    print("="*80)
    print(report_df.head())
    print(f"\nTotal days: {len(report_df['date'].unique())}")
    print(f"Total controllers: {report_df['controller_id'].nunique()}")


if __name__ == "__main__":
    asyncio.run(main())
