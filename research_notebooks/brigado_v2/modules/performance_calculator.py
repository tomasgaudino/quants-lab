"""
Performance Calculator Module

Calculates daily performance metrics including market data, bot performance, and controller attribution.

TODO - Future Improvements:
- Add intraday performance metrics (hourly, 15min)
- Implement real-time performance tracking
- Add benchmark comparisons (market returns, passive strategies)
- Support for multiple trading pairs aggregation
- Add risk-adjusted metrics (Sharpe, Sortino, Max Drawdown)
"""

import asyncio
from datetime import datetime
from typing import Optional, Dict
import pandas as pd
import numpy as np

from core.data_sources.clob import CLOBDataSource


class PerformanceCalculator:
    """
    Calculates daily performance metrics from enriched trade data.

    Responsibilities:
    - Fetch market metrics from CLOB data
    - Calculate global bot metrics (portfolio-level)
    - Calculate controller-specific metrics (attribution)
    - Generate daily performance reports
    - Calculate break-even prices
    - Track position inventory

    IMPORTANT: Follows portfolio accounting principles:
    - Global bot metrics = real portfolio performance
    - Controller metrics = attribution/contribution (non-ownership)
    - Single source of truth for positions
    """

    def __init__(self, root_path: str = ""):
        """
        Initialize performance calculator.

        Args:
            root_path: Root path for the project
        """
        self.root_path = root_path
        self.clob = CLOBDataSource()

    async def get_market_metrics(
        self,
        connector_name: str,
        trading_pair: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, float]:
        """
        Get market metrics for a specific date range using daily candles.

        Args:
            connector_name: Exchange connector name (e.g., 'binance')
            trading_pair: Trading pair (e.g., 'SOL-USDC', 'USDT-BRL')
            start_date: Start date for metrics
            end_date: End date for metrics

        Returns:
            Dictionary with market metrics

        TODO:
        - Add support for multiple intervals (1h, 4h, 1d)
        - Cache market data to avoid repeated API calls
        - Add validation for data quality (gaps, outliers)
        - Support for alternative data sources
        """
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())

        try:
            # Get daily candles - Candles object has .data attribute
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
                return self._empty_market_metrics()

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

        except Exception as e:
            print(f"Warning: Could not fetch market data: {e}")
            return self._empty_market_metrics()

    def calculate_break_even_price(
        self,
        trades: pd.DataFrame,
        trade_type: str
    ) -> float:
        """
        Calculate break-even price for buy or sell trades.

        Uses volume-weighted average price (VWAP).

        Args:
            trades: DataFrame of trades filtered by type
            trade_type: 'BUY' or 'SELL'

        Returns:
            Break-even price

        TODO:
        - Add support for FIFO/LIFO break-even calculation
        - Include fees in break-even calculation
        - Add slippage estimation
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

        TODO:
        - Add position limits validation
        - Track multiple instruments separately
        - Add position reconciliation with exchange
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
    ) -> Dict[str, float]:
        """
        Calculate global bot metrics for a day (portfolio-level performance).

        Args:
            daily_trades: Trades for a specific day
            previous_day_end_position: Position at end of previous day

        Returns:
            Dictionary with global bot metrics

        TODO:
        - Add portfolio value tracking
        - Add return on capital metrics
        - Add turnover ratio
        - Track peak position size
        - Add intraday metrics (high/low position, max drawdown)
        """
        if daily_trades.empty:
            return self._empty_global_metrics(previous_day_end_position)

        # Volume metrics
        # amount = base asset volume (e.g., SOL, USDT)
        # quote_volume = quote asset volume (e.g., USDC, BRL) = price * amount
        volume_base = daily_trades['amount'].sum()
        volume_quote = daily_trades['quote_volume'].sum()

        # PnL metrics
        # CRITICAL: Use realized_pnl (incremental) NOT net_realized_pnl (cumulative)
        realized_pnl = daily_trades['realized_pnl'].sum() if 'realized_pnl' in daily_trades.columns else 0.0
        gross_pnl = daily_trades['gross_pnl'].sum() if 'gross_pnl' in daily_trades.columns else 0.0
        total_fees = daily_trades['trade_fee_in_quote'].sum() if 'trade_fee_in_quote' in daily_trades.columns else 0.0

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
    ) -> Dict[str, float]:
        """
        Calculate metrics for a specific controller (attribution, not ownership).

        Args:
            controller_trades: Trades for a specific controller
            controller_id: Controller identifier

        Returns:
            Dictionary with controller-specific metrics

        TODO:
        - Add controller efficiency metrics (PnL per trade, per volume)
        - Track controller state changes (active/inactive periods)
        - Add controller risk metrics (volatility, max position)
        - Compare controller performance vs others
        """
        if controller_trades.empty:
            return self._empty_controller_metrics(controller_id)

        # Volume metrics
        volume_base = controller_trades['amount'].sum()
        volume_quote = controller_trades['quote_volume'].sum()

        # PnL metrics - use incremental realized_pnl, not cumulative
        realized_pnl = controller_trades['realized_pnl'].sum() if 'realized_pnl' in controller_trades.columns else 0.0
        gross_pnl = controller_trades['gross_pnl'].sum() if 'gross_pnl' in controller_trades.columns else 0.0
        total_fees = controller_trades['trade_fee_in_quote'].sum() if 'trade_fee_in_quote' in controller_trades.columns else 0.0

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
        enriched_trades: pd.DataFrame,
        output_path: str = "daily_performance_report.csv"
    ) -> pd.DataFrame:
        """
        Generate daily performance report grouped by date and controller_id.

        Args:
            enriched_trades: Enriched trades with controller_id and PnL calculations
            output_path: Path to save the CSV report

        Returns:
            DataFrame with daily performance metrics

        TODO:
        - Add progress callbacks for long date ranges
        - Support for parallel processing of dates
        - Add data quality checks (missing days, gaps)
        - Include intraday metrics
        - Add controller comparison metrics
        """
        # Add date column
        enriched_trades['date'] = enriched_trades['timestamp'].dt.date

        # Get unique dates
        dates = sorted(enriched_trades['date'].unique())

        all_reports = []
        previous_position = 0.0

        for date in dates:
            print(f"Processing date: {date}")

            # Filter trades for this date
            daily_trades = enriched_trades[enriched_trades['date'] == date].copy()
            daily_trades = daily_trades.sort_values('timestamp')

            # Get market info
            if not daily_trades.empty:
                connector_name = daily_trades['market'].iloc[0]
                trading_pair = daily_trades['symbol'].iloc[0]

                # Get market metrics
                start_datetime = datetime.combine(date, datetime.min.time())
                end_datetime = datetime.combine(date, datetime.max.time())

                market_metrics = await self.get_market_metrics(
                    connector_name=connector_name,
                    trading_pair=trading_pair,
                    start_date=start_datetime,
                    end_date=end_datetime
                )

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

    # Helper methods for empty metrics
    def _empty_market_metrics(self) -> Dict[str, float]:
        """Return empty market metrics dict."""
        return {
            'market_volume_usdt': 0.0,
            'market_price_min': 0.0,
            'market_price_max': 0.0,
            'market_price_var_pct': 0.0,
            'market_trades_count': 0
        }

    def _empty_global_metrics(self, previous_position: float) -> Dict[str, float]:
        """Return empty global bot metrics dict."""
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
            'bot_initial_position': previous_position,
            'bot_final_position': previous_position,
        }

    def _empty_controller_metrics(self, controller_id: str) -> Dict[str, float]:
        """Return empty controller metrics dict."""
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
