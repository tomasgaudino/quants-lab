"""
Visualizer Module

Creates executive-level visualizations for trading performance analysis.

TODO - Future Improvements:
- Add interactive dashboards (Dash/Streamlit)
- Export to PDF/PNG for reports
- Add custom color schemes and branding
- Support for multiple timeframes in same view
- Add comparison charts (multiple controllers, strategies)
- Add statistical overlays (confidence intervals, trend lines)
"""

from typing import Optional, Dict
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class PerformanceVisualizer:
    """
    Creates executive-level visualizations for trading performance.

    Responsibilities:
    - Generate daily performance report summaries
    - Create executive dashboard (portfolio value, PnL, volume, etc.)
    - Create risk metrics dashboard
    - Create controller comparison charts
    - Format and display summary statistics
    """

    def __init__(self, rebate_pct: float = 0.015 / 100, quote_asset: str = "USDT"):
        """
        Initialize visualizer with configuration.

        Args:
            rebate_pct: Exchange rebate percentage (default: 0.015%)
            quote_asset: Quote asset symbol (e.g., 'USDC', 'USDT', 'BRL')
        """
        self.rebate_pct = rebate_pct
        self.quote_asset = quote_asset

    def print_last_7_days_summary(self, df: pd.DataFrame) -> None:
        """
        Print last 7 days performance summary table.

        Args:
            df: Daily performance report DataFrame

        TODO:
        - Add color coding for positive/negative values
        - Add sparklines for trends
        - Export to formatted HTML table
        """
        # Get last 7 days
        last_7_days = df.tail(7).copy()

        # Calculate additional metrics
        last_7_days['rebate'] = last_7_days['bot_volume_quote'] * self.rebate_pct
        last_7_days['pnl_with_rebate'] = last_7_days['bot_realized_pnl'] + last_7_days['rebate']
        last_7_days['return_pct'] = (
            last_7_days['pnl_with_rebate'] / last_7_days['bot_volume_base'] * 100
        ).replace([np.inf, -np.inf], 0).fillna(0)

        # Get base asset from data if available, otherwise default to BASE
        base_asset = 'BASE'
        if 'base_asset' in last_7_days.columns and len(last_7_days) > 0:
            base_asset = last_7_days['base_asset'].iloc[0] if pd.notna(last_7_days['base_asset'].iloc[0]) else 'BASE'

        # Create summary table
        summary_table = pd.DataFrame({
            'Date': last_7_days['date'].dt.strftime('%m/%d'),
            'Trades': last_7_days['bot_trades_count'].astype(int),
            f'Vol ({base_asset})': last_7_days['bot_volume_base'].round(0).astype(int),
            f'Vol ({self.quote_asset})': last_7_days['bot_volume_quote'].round(0).astype(int),
            'Bot P&L': last_7_days['bot_realized_pnl'].round(2),
            'Rebate': last_7_days['rebate'].round(2),
            'Net P&L': last_7_days['pnl_with_rebate'].round(2),
            'Return %': last_7_days['return_pct'].round(4),
            'Mkt Share %': last_7_days['bot_market_share_pct'].round(2),
        })

        # Transpose so days are columns
        summary_transposed = summary_table.set_index('Date').T

        print("=" * 100)
        print(" " * 35 + "LAST 7 DAYS PERFORMANCE")
        print("=" * 100)
        print()

        # Format display to avoid scientific notation
        pd.set_option('display.float_format', lambda x: f'{x:,.2f}' if abs(x) >= 1 else f'{x:.4f}')
        print(summary_transposed.to_string())
        pd.reset_option('display.float_format')

        print()
        print("=" * 100)
        print()

    def print_executive_header(self, df: pd.DataFrame, initial_portfolio_quote: float) -> None:
        """
        Print comprehensive executive header with key metrics.

        Args:
            df: Daily performance report DataFrame
            initial_portfolio_quote: Initial portfolio value in quote asset

        TODO:
        - Add period comparison (vs last week, last month)
        - Add benchmark comparisons
        - Include risk-adjusted metrics (Sharpe, Sortino)
        """
        # Extract configuration
        first_row = df.iloc[0]
        controller_id = first_row['controller_id'] if 'controller_id' in df.columns else 'N/A'

        # Get first trade price for initial portfolio composition
        first_price = df['bot_buy_break_even'].iloc[0] if df['bot_buy_break_even'].iloc[0] > 0 else 5.60
        initial_base_position = df['bot_initial_position'].iloc[0] if 'bot_initial_position' in df.columns else 0.0
        initial_base_value_quote = initial_base_position * first_price
        initial_quote_cash = initial_portfolio_quote - initial_base_value_quote

        # Period metrics
        num_days = len(df['date'].unique())
        start_date = df['date'].min()
        end_date = df['date'].max()

        # Calculate totals
        total_volume_base = df['bot_volume_base'].sum()
        total_volume_quote = df['bot_volume_quote'].sum()
        total_rebate = total_volume_quote * self.rebate_pct
        total_pnl_bot = df['bot_realized_pnl'].sum()
        total_pnl_financial = total_pnl_bot + total_rebate
        final_portfolio = initial_portfolio_quote + total_pnl_financial

        # Performance metrics
        portfolio_return_pct = (total_pnl_financial / initial_portfolio_quote * 100)
        avg_daily_volume_base = total_volume_base / num_days
        total_trades = df['bot_trades_count'].sum()

        # Win metrics
        daily_pnl = df.groupby('date')['bot_realized_pnl'].first()
        win_days = (daily_pnl > 0).sum()
        win_rate = (win_days / len(daily_pnl) * 100) if len(daily_pnl) > 0 else 0

        # Get base asset from data
        base_asset = 'BASE'
        if 'base_asset' in first_row.index and pd.notna(first_row['base_asset']):
            base_asset = first_row['base_asset']

        # Print header
        print("=" * 80)
        print(" " * 20 + "TRADING PERFORMANCE ANALYSIS")
        print("=" * 80)
        print()
        print("📋 CONFIGURATION")
        print("-" * 80)
        print(f"  Trading Pair:              {base_asset}-{self.quote_asset}")
        print(f"  Exchange:                  Binance")
        print(f"  Controller:                {controller_id}")
        print(f"  Rebate Rate:               {self.rebate_pct * 100:.3f}%")
        print()

        print("💼 INITIAL PORTFOLIO (Day 1)")
        print("-" * 80)
        print(f"  Total Allocation:          {initial_portfolio_quote:>15,.2f} {self.quote_asset}")
        print(f"  Initial Base Position:     {initial_base_position:>15,.2f} {base_asset}")
        print(f"  Est. {base_asset} Value (@ {first_price:.4f}): {initial_base_value_quote:>15,.2f} {self.quote_asset} "
              f"({initial_base_value_quote/initial_portfolio_quote*100:>5.1f}%)")
        print(f"  Est. {self.quote_asset} Cash:             {initial_quote_cash:>15,.2f} {self.quote_asset} "
              f"({initial_quote_cash/initial_portfolio_quote*100:>5.1f}%)")
        print()

        print("📊 PERIOD OVERVIEW")
        print("-" * 80)
        print(f"  Start Date:                {start_date}")
        print(f"  End Date:                  {end_date}")
        print(f"  Trading Days:              {num_days:>15,}")
        print(f"  Total Trades:              {total_trades:>15,}")
        print(f"  Avg Daily Volume:          {avg_daily_volume_base:>15,.2f} {base_asset}")
        print()

        print("💰 FINANCIAL SUMMARY")
        print("-" * 80)
        print(f"  Bot Trading P&L:           {total_pnl_bot:>15,.2f} {self.quote_asset}")
        print(f"  Exchange Rebates:          {total_rebate:>15,.2f} {self.quote_asset}")
        print(f"  Total Financial P&L:       {total_pnl_financial:>15,.2f} {self.quote_asset}")
        print(f"  Rebate Contribution:       "
              f"{(total_rebate/abs(total_pnl_financial)*100) if total_pnl_financial != 0 else 0:>15.1f}%")
        print()
        print(f"  Final Portfolio Value:     {final_portfolio:>15,.2f} {self.quote_asset}")
        print(f"  Portfolio Return:          {portfolio_return_pct:>15.4f}%")
        print()

        print("📈 KEY PERFORMANCE INDICATORS")
        print("-" * 80)
        print(f"  Total Volume (Base):       {total_volume_base:>15,.2f} {base_asset}")
        print(f"  Total Volume (Quote):      {total_volume_quote:>15,.2f} {self.quote_asset}")
        print(f"  ROI on Volume:             "
              f"{(total_pnl_financial/total_volume_base*100) if total_volume_base > 0 else 0:>15.4f}%")
        print(f"  Avg P&L per Day:           {total_pnl_financial/num_days:>15,.2f} {self.quote_asset}")
        print(f"  Avg P&L per Trade:         "
              f"{total_pnl_financial/total_trades if total_trades > 0 else 0:>15,.4f} {self.quote_asset}")
        print()

        print("🎯 WIN/LOSS ANALYSIS")
        print("-" * 80)
        print(f"  Winning Days:              {win_days:>15,} / {len(daily_pnl)}")
        print(f"  Win Rate:                  {win_rate:>15.1f}%")
        print(f"  Best Day P&L:              {daily_pnl.max():>15,.2f} {self.quote_asset}")
        print(f"  Worst Day P&L:             {daily_pnl.min():>15,.2f} {self.quote_asset}")
        print()

        print("⚠️  IMPORTANT NOTES")
        print("-" * 80)
        print("  • Bot P&L (blue) = Pure strategy performance without rebates")
        print("  • Financial P&L (green) = Real money including exchange rebates")
        print("  • Portfolio value assumes initial capital + cumulative financial P&L")
        print("  • All timestamps are in UTC")
        print("  • Position tracking is in base asset (USDT)")
        print()
        print("=" * 80)
        print()

    def create_executive_dashboard(
        self,
        df: pd.DataFrame,
        initial_portfolio_quote: float
    ) -> go.Figure:
        """
        Create comprehensive executive dashboard with 8 charts.

        Args:
            df: Daily performance report DataFrame
            initial_portfolio_quote: Initial portfolio value in quote asset

        Returns:
            Plotly figure with executive dashboard

        TODO:
        - Add drill-down capabilities (click to see intraday)
        - Add annotations for significant events
        - Support for custom date ranges
        - Add export to static image
        """
        # Prepare daily aggregated data
        daily_agg = self._prepare_daily_aggregated_data(df, initial_portfolio_quote)

        # Create subplots
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                f'Portfolio Value Over Time ({self.quote_asset})',
                'Daily Trading Volume (Base/Quote)',
                'Cumulative P&L: Bot vs Financial (with Rebate)',
                'Daily P&L (Realized vs Unrealized)',
                'Market Share %',
                'Daily Rebates (0.015%)',
                'Trading Activity',
                'Daily Returns'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": True}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ],
            vertical_spacing=0.09,
            horizontal_spacing=0.12
        )

        # Add all traces
        self._add_portfolio_value_trace(fig, daily_agg, initial_portfolio_quote)
        self._add_volume_traces(fig, daily_agg)
        self._add_cumulative_pnl_traces(fig, daily_agg)
        self._add_daily_pnl_traces(fig, daily_agg)
        self._add_market_share_trace(fig, daily_agg)
        self._add_rebate_trace(fig, daily_agg)
        self._add_trading_activity_trace(fig, daily_agg)
        self._add_daily_returns_trace(fig, daily_agg)

        # Update layout
        self._update_dashboard_layout(fig)

        return fig

    def _prepare_daily_aggregated_data(
        self,
        df: pd.DataFrame,
        initial_portfolio_quote: float
    ) -> pd.DataFrame:
        """Prepare daily aggregated data with rebates and portfolio tracking."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Aggregate by date
        daily_agg = df.groupby('date').agg({
            'bot_realized_pnl': 'first',
            'bot_unrealized_pnl': 'first',
            'bot_volume_base': 'first',
            'bot_volume_quote': 'first',
            'bot_trades_count': 'first',
            'bot_market_share_pct': 'first',
            'market_volume_usdt': 'first',
            'bot_initial_position': 'first',
            'bot_final_position': 'first',
        }).reset_index()

        # Calculate rebates and cumulative metrics
        daily_agg['daily_rebate'] = daily_agg['bot_volume_quote'] * self.rebate_pct
        daily_agg['bot_realized_pnl_with_rebate'] = daily_agg['bot_realized_pnl'] + daily_agg['daily_rebate']
        daily_agg['cumulative_realized_pnl'] = daily_agg['bot_realized_pnl'].cumsum()
        daily_agg['cumulative_pnl_with_rebate'] = daily_agg['bot_realized_pnl_with_rebate'].cumsum()
        daily_agg['portfolio_value'] = initial_portfolio_quote + daily_agg['cumulative_pnl_with_rebate']
        daily_agg['daily_return_pct'] = (
            daily_agg['bot_realized_pnl_with_rebate'] / daily_agg['bot_volume_base'] * 100
        ).replace([np.inf, -np.inf], 0).fillna(0)

        return daily_agg

    def _add_portfolio_value_trace(
        self,
        fig: go.Figure,
        daily_agg: pd.DataFrame,
        initial_portfolio_quote: float
    ) -> None:
        """Add portfolio value chart."""
        fig.add_trace(
            go.Scatter(
                x=daily_agg['date'],
                y=daily_agg['portfolio_value'],
                mode='lines+markers',
                name='Portfolio Value',
                line=dict(color='#06A77D', width=3),
                fill='tozeroy',
                fillcolor='rgba(6, 167, 125, 0.2)',
                hovertemplate=f'%{{y:,.2f}} {self.quote_asset}'
            ),
            row=1, col=1
        )
        fig.add_hline(
            y=initial_portfolio_quote,
            line=dict(color='gray', dash='dash', width=1),
            annotation_text=f'Initial: {initial_portfolio_quote:,.0f} {self.quote_asset}',
            row=1, col=1
        )

    def _add_volume_traces(self, fig: go.Figure, daily_agg: pd.DataFrame) -> None:
        """Add volume charts (base and quote)."""
        # Get base asset from data
        base_asset = 'BASE'
        if 'base_asset' in daily_agg.columns and len(daily_agg) > 0:
            base_asset = daily_agg['base_asset'].iloc[0] if pd.notna(daily_agg['base_asset'].iloc[0]) else 'BASE'

        fig.add_trace(
            go.Bar(
                x=daily_agg['date'],
                y=daily_agg['bot_volume_base'],
                name=f'Volume Base ({base_asset})',
                marker_color='#06A77D'
            ),
            row=1, col=2, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(
                x=daily_agg['date'],
                y=daily_agg['bot_volume_quote'],
                name=f'Volume Quote ({self.quote_asset})',
                mode='lines+markers',
                line=dict(color='#F77F00', width=2),
                marker=dict(size=6)
            ),
            row=1, col=2, secondary_y=True
        )

    def _add_cumulative_pnl_traces(self, fig: go.Figure, daily_agg: pd.DataFrame) -> None:
        """Add cumulative PnL comparison (bot vs financial)."""
        fig.add_trace(
            go.Scatter(
                x=daily_agg['date'],
                y=daily_agg['cumulative_realized_pnl'],
                mode='lines+markers',
                name='Bot P&L (no rebate)',
                line=dict(color='#2E86AB', width=2, dash='dot'),
                marker=dict(size=4)
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=daily_agg['date'],
                y=daily_agg['cumulative_pnl_with_rebate'],
                mode='lines+markers',
                name='Financial P&L (with rebate)',
                line=dict(color='#06A77D', width=3),
                fill='tozeroy',
                fillcolor='rgba(6, 167, 125, 0.2)'
            ),
            row=2, col=1
        )

    def _add_daily_pnl_traces(self, fig: go.Figure, daily_agg: pd.DataFrame) -> None:
        """Add daily PnL (realized vs unrealized)."""
        fig.add_trace(
            go.Bar(
                x=daily_agg['date'],
                y=daily_agg['bot_realized_pnl'],
                name='Realized P&L',
                marker_color='#2E86AB'
            ),
            row=2, col=2
        )
        fig.add_trace(
            go.Bar(
                x=daily_agg['date'],
                y=daily_agg['bot_unrealized_pnl'],
                name='Unrealized P&L',
                marker_color='#D62828',
                opacity=0.6
            ),
            row=2, col=2
        )

    def _add_market_share_trace(self, fig: go.Figure, daily_agg: pd.DataFrame) -> None:
        """Add market share chart."""
        fig.add_trace(
            go.Scatter(
                x=daily_agg['date'],
                y=daily_agg['bot_market_share_pct'],
                mode='lines+markers',
                name='Market Share',
                line=dict(color='#F77F00', width=2),
                marker=dict(size=8)
            ),
            row=3, col=1
        )

    def _add_rebate_trace(self, fig: go.Figure, daily_agg: pd.DataFrame) -> None:
        """Add daily rebates chart."""
        fig.add_trace(
            go.Bar(
                x=daily_agg['date'],
                y=daily_agg['daily_rebate'],
                name='Daily Rebate',
                marker_color='#06A77D',
                hovertemplate=f'%{{y:.2f}} {self.quote_asset}'
            ),
            row=3, col=2
        )

    def _add_trading_activity_trace(self, fig: go.Figure, daily_agg: pd.DataFrame) -> None:
        """Add trading activity (trade count) chart."""
        fig.add_trace(
            go.Bar(
                x=daily_agg['date'],
                y=daily_agg['bot_trades_count'],
                name='Trades Count',
                marker_color='#A23B72'
            ),
            row=4, col=1
        )

    def _add_daily_returns_trace(self, fig: go.Figure, daily_agg: pd.DataFrame) -> None:
        """Add daily returns chart with color coding."""
        colors = ['#06A77D' if x >= 0 else '#D62828' for x in daily_agg['daily_return_pct']]
        fig.add_trace(
            go.Bar(
                x=daily_agg['date'],
                y=daily_agg['daily_return_pct'],
                name='Daily Return % (w/ rebate)',
                marker_color=colors
            ),
            row=4, col=2
        )

    def _update_dashboard_layout(self, fig: go.Figure) -> None:
        """Update layout and axes labels."""
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=4, col=1)
        fig.update_xaxes(title_text="Date", row=4, col=2)
        fig.update_yaxes(title_text=self.quote_asset, row=1, col=1)
        # Note: Can't dynamically update this label here, handled in _add_volume_traces
        fig.update_yaxes(title_text="BASE", row=1, col=2, secondary_y=False)
        fig.update_yaxes(title_text=self.quote_asset, row=1, col=2, secondary_y=True)
        fig.update_yaxes(title_text=self.quote_asset, row=2, col=1)
        fig.update_yaxes(title_text=self.quote_asset, row=2, col=2)
        fig.update_yaxes(title_text="%", row=3, col=1)
        fig.update_yaxes(title_text=self.quote_asset, row=3, col=2)
        fig.update_yaxes(title_text="Count", row=4, col=1)
        fig.update_yaxes(title_text="%", row=4, col=2)

        # Update layout
        fig.update_layout(
            title={
                'text': 'Trading Performance Executive Dashboard',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': '#1F1F1F'}
            },
            height=1400,
            showlegend=True,
            template='plotly_white',
            font=dict(size=11),
            hovermode='x unified'
        )
