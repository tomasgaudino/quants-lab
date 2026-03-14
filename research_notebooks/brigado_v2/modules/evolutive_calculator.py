"""
Evolutive Metrics Calculator Module

Calculates daily time series for all metrics from market, bot, and controller data
to show the evolution of performance over time.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import aiohttp
from datetime import datetime, timedelta


class EvolutiveMetricsCalculator:
    """
    Calculates daily time series for market, bot, and controller metrics.

    Produces a comprehensive daily evolution report showing:
    - Market metrics: volume, volatility, price movement
    - Bot metrics: trades, volume, P&L, positions
    - Controller metrics: per-controller performance evolution
    """

    def __init__(self, root_path: str, server_name: str = "brigado"):
        """
        Initialize evolutive metrics calculator.

        Args:
            root_path: Root path for the project
            server_name: Name of the server (e.g., 'brigado', 'old_brigado')
        """
        self.root_path = Path(root_path)
        self.server_name = server_name
        self.data_sources_path = self.root_path / f"research_notebooks/brigado_v2/data/{server_name}"

    async def calculate_evolutive_metrics(
        self,
        trades_with_ctrl: pd.DataFrame,
        controllers_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate daily evolutive metrics for all dimensions.

        Args:
            trades_with_ctrl: Trades DataFrame with controller_id column
            controllers_data: Controllers DataFrame

        Returns:
            DataFrame with daily metrics evolution
        """
        print("\n📈 Calculating evolutive metrics...")

        # Ensure timestamp is datetime
        trades_with_ctrl['timestamp'] = pd.to_datetime(trades_with_ctrl['timestamp'])
        trades_with_ctrl['date'] = trades_with_ctrl['timestamp'].dt.date

        # Get unique dates and symbols
        dates = sorted(trades_with_ctrl['date'].unique())
        symbols = trades_with_ctrl['symbol'].unique()

        print(f"  • Date range: {dates[0]} to {dates[-1]} ({len(dates)} days)")
        print(f"  • Trading pairs: {len(symbols)}")

        # Calculate daily metrics
        daily_metrics = []

        for date in dates:
            day_trades = trades_with_ctrl[trades_with_ctrl['date'] == date]

            for symbol in symbols:
                symbol_trades = day_trades[day_trades['symbol'] == symbol]

                if len(symbol_trades) == 0:
                    continue

                # Bot metrics
                bot_metrics = self._calculate_bot_metrics(symbol_trades)

                # Controller metrics
                controller_metrics = self._calculate_controller_metrics(symbol_trades)

                # Combine metrics
                daily_metrics.append({
                    'date': date,
                    'symbol': symbol,
                    **bot_metrics,
                    **controller_metrics
                })

        df = pd.DataFrame(daily_metrics)

        # Add market metrics
        df = await self._add_market_metrics(df)

        # Calculate cumulative metrics
        df = self._calculate_cumulative_metrics(df)

        print(f"  ✓ Generated {len(df)} daily metric records")

        return df

    def _calculate_bot_metrics(self, trades: pd.DataFrame) -> Dict:
        """Calculate bot-level metrics for a day/symbol."""
        if len(trades) == 0:
            return {
                'bot_trades_count': 0,
                'bot_volume_base': 0,
                'bot_volume_quote': 0,
                'bot_realized_pnl': 0,
                'bot_final_position': 0,
                'bot_avg_buy_price': 0,
                'bot_avg_sell_price': 0,
                'bot_spread_captured': 0
            }

        # Basic metrics
        total_trades = len(trades)
        volume_base = trades['amount'].sum()
        volume_quote = (trades['amount'] * trades['price']).sum()

        # P&L calculation
        trades_sorted = trades.sort_values('timestamp')
        position = 0
        avg_cost = 0
        realized_pnl = 0

        for _, trade in trades_sorted.iterrows():
            amount = trade['amount']
            price = trade['price']

            if trade['trade_type'].lower() == 'buy':
                # Update average cost
                total_cost = (position * avg_cost) + (amount * price)
                position += amount
                avg_cost = total_cost / position if position > 0 else 0
            else:  # sell
                # Realize P&L
                if position > 0:
                    realized_pnl += amount * (price - avg_cost)
                position -= amount

        # Spread metrics
        buys = trades[trades['trade_type'].str.lower() == 'buy']
        sells = trades[trades['trade_type'].str.lower() == 'sell']

        avg_buy_price = buys['price'].mean() if len(buys) > 0 else 0
        avg_sell_price = sells['price'].mean() if len(sells) > 0 else 0
        spread_captured = avg_sell_price - avg_buy_price if avg_buy_price > 0 else 0

        return {
            'bot_trades_count': total_trades,
            'bot_volume_base': volume_base,
            'bot_volume_quote': volume_quote,
            'bot_realized_pnl': realized_pnl,
            'bot_final_position': position,
            'bot_avg_buy_price': avg_buy_price,
            'bot_avg_sell_price': avg_sell_price,
            'bot_spread_captured': spread_captured
        }

    def _calculate_controller_metrics(self, trades: pd.DataFrame) -> Dict:
        """Calculate controller-level metrics for a day/symbol."""
        if len(trades) == 0 or 'controller_id' not in trades.columns:
            return {
                'controllers_active': 0,
                'controller_coverage_pct': 0,
                'orphan_trades_count': 0,
                'orphan_volume_quote': 0
            }

        # Controller coverage
        total_trades = len(trades)
        mapped_trades = trades['controller_id'].notna().sum()
        coverage_pct = (mapped_trades / total_trades * 100) if total_trades > 0 else 0

        # Active controllers
        active_controllers = trades['controller_id'].nunique()

        # Orphan metrics
        orphan_trades = trades[trades['controller_id'].isna()]
        orphan_count = len(orphan_trades)
        orphan_volume = (orphan_trades['amount'] * orphan_trades['price']).sum() if len(orphan_trades) > 0 else 0

        return {
            'controllers_active': active_controllers,
            'controller_coverage_pct': coverage_pct,
            'orphan_trades_count': orphan_count,
            'orphan_volume_quote': orphan_volume
        }

    async def _add_market_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market metrics to daily data."""
        print("\n  📊 Fetching market data...")

        # Try to load from existing market analysis
        try:
            market_data_path = self.data_sources_path / "market_analysis_data.parquet"
            if market_data_path.exists():
                market_df = pd.read_parquet(market_data_path)
                market_df['date'] = pd.to_datetime(market_df['date']).dt.date

                # Merge with daily metrics
                df = df.merge(
                    market_df[['date', 'symbol', 'base_volume', 'quote_volume', 'trades',
                               'open', 'high', 'low', 'close', 'volatility']],
                    on=['date', 'symbol'],
                    how='left',
                    suffixes=('', '_market')
                )

                # Rename market columns
                df = df.rename(columns={
                    'base_volume': 'market_volume_base',
                    'quote_volume': 'market_volume_quote',
                    'trades': 'market_trades_count',
                    'open': 'market_open',
                    'high': 'market_high',
                    'low': 'market_low',
                    'close': 'market_close',
                    'volatility': 'market_volatility'
                })

                # Calculate market share
                df['market_share_volume'] = (df['bot_volume_quote'] / df['market_volume_quote'] * 100).fillna(0)
                df['market_share_trades'] = (df['bot_trades_count'] / df['market_trades_count'] * 100).fillna(0)

                print(f"    ✓ Market data loaded from cache")
                return df

        except Exception as e:
            print(f"    ⚠ Could not load market data: {e}")

        # Fallback: add empty market columns
        df['market_volume_base'] = 0
        df['market_volume_quote'] = 0
        df['market_trades_count'] = 0
        df['market_open'] = 0
        df['market_high'] = 0
        df['market_low'] = 0
        df['market_close'] = 0
        df['market_volatility'] = 0
        df['market_share_volume'] = 0
        df['market_share_trades'] = 0

        return df

    def _calculate_cumulative_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cumulative metrics over time."""
        df = df.sort_values(['symbol', 'date'])

        # Calculate cumulative P&L per symbol
        df['cumulative_pnl'] = df.groupby('symbol')['bot_realized_pnl'].cumsum()

        # Calculate cumulative volume per symbol
        df['cumulative_volume_base'] = df.groupby('symbol')['bot_volume_base'].cumsum()
        df['cumulative_volume_quote'] = df.groupby('symbol')['bot_volume_quote'].cumsum()

        # Calculate cumulative trades
        df['cumulative_trades'] = df.groupby('symbol')['bot_trades_count'].cumsum()

        # Calculate rolling averages (7-day)
        for col in ['bot_volume_quote', 'bot_trades_count', 'bot_realized_pnl']:
            df[f'{col}_7d_avg'] = df.groupby('symbol')[col].transform(
                lambda x: x.rolling(window=7, min_periods=1).mean()
            )

        return df

    def save_evolutive_metrics(
        self,
        df: pd.DataFrame,
        output_path: Path = None
    ) -> Path:
        """
        Save evolutive metrics to parquet.

        Args:
            df: Evolutive metrics DataFrame
            output_path: Optional output path (defaults to data_sources)

        Returns:
            Path to saved file
        """
        if output_path is None:
            output_path = self.data_sources_path / "evolutive_metrics.parquet"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

        print(f"\n💾 Evolutive metrics saved to: {output_path}")
        return output_path
