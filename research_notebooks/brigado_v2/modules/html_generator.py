"""
HTML Report Generator Module

Generates professional HTML reports for client delivery.
"""

from typing import Optional
import pandas as pd
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class HTMLReportGenerator:
    """
    Generates professional HTML reports for trading performance analysis.

    Creates standalone HTML files with embedded Plotly charts.
    """

    def __init__(self, rebate_pct: float = 0.015 / 100, quote_asset: str = "USDT"):
        """
        Initialize HTML report generator.

        Args:
            rebate_pct: Exchange rebate percentage
            quote_asset: Quote asset symbol (e.g., 'USDC', 'USDT', 'BRL')
        """
        self.rebate_pct = rebate_pct
        self.quote_asset = quote_asset

    def generate_report(
        self,
        daily_report: pd.DataFrame,
        initial_portfolio_quote: float,
        output_path: str,
        company_name: str = "Trading Performance Report",
        report_date: Optional[datetime] = None
    ) -> str:
        """
        Generate complete HTML report.

        Args:
            daily_report: Daily performance DataFrame
            initial_portfolio_quote: Initial portfolio value in quote asset
            output_path: Path to save HTML
            company_name: Company/report name for header
            report_date: Report generation date (defaults to now)

        Returns:
            Path to generated HTML
        """
        if report_date is None:
            report_date = datetime.now()

        # Prepare data
        df = daily_report.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Calculate metrics
        total_volume_base = df['bot_volume_base'].sum()
        total_volume_quote = df['bot_volume_quote'].sum()
        total_rebate = total_volume_quote * self.rebate_pct
        total_pnl_bot = df['bot_realized_pnl'].sum()
        total_pnl_financial = total_pnl_bot + total_rebate
        num_days = len(df['date'].unique())
        total_trades = df['bot_trades_count'].sum()

        daily_pnl = df.groupby('date')['bot_realized_pnl'].first()
        win_days = (daily_pnl > 0).sum()
        win_rate = (win_days / len(daily_pnl) * 100) if len(daily_pnl) > 0 else 0

        # Create performance chart
        daily_agg = df.groupby('date').agg({
            'bot_realized_pnl': 'first',
            'bot_volume_base': 'first',
            'bot_trades_count': 'first',
            'bot_volume_quote': 'first'
        }).reset_index()

        daily_agg['daily_rebate'] = daily_agg['bot_volume_quote'] * self.rebate_pct
        daily_agg['pnl_with_rebate'] = daily_agg['bot_realized_pnl'] + daily_agg['daily_rebate']
        daily_agg['cumulative_pnl'] = daily_agg['pnl_with_rebate'].cumsum()
        daily_agg['portfolio_value'] = initial_portfolio_quote + daily_agg['cumulative_pnl']

        fig = self._create_performance_chart(daily_agg, initial_portfolio_quote)
        chart_html = fig.to_html(include_plotlyjs='cdn', div_id='performance-chart')

        # Build HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{company_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #2E86AB 0%, #06A77D 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 32px;
            margin-bottom: 10px;
        }}

        .header p {{
            font-size: 16px;
            opacity: 0.9;
        }}

        .content {{
            padding: 40px;
        }}

        .section {{
            margin-bottom: 40px;
        }}

        .section h2 {{
            font-size: 24px;
            color: #1F1F1F;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #2E86AB;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .metric-card {{
            background: #f9f9f9;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #2E86AB;
        }}

        .metric-card .label {{
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }}

        .metric-card .value {{
            font-size: 24px;
            font-weight: bold;
            color: #1F1F1F;
        }}

        .metric-card.positive .value {{
            color: #06A77D;
        }}

        .metric-card.negative .value {{
            color: #D62828;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 14px;
        }}

        th {{
            background: #2E86AB;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}

        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #e0e0e0;
        }}

        tr:nth-child(even) {{
            background: #f9f9f9;
        }}

        tr:hover {{
            background: #f0f0f0;
        }}

        .positive-value {{
            color: #06A77D;
            font-weight: 600;
        }}

        .negative-value {{
            color: #D62828;
            font-weight: 600;
        }}

        .footer {{
            background: #f9f9f9;
            padding: 20px 40px;
            text-align: center;
            color: #666;
            font-size: 14px;
            border-top: 1px solid #e0e0e0;
        }}

        @media print {{
            body {{
                background: white;
                padding: 0;
            }}

            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{company_name}</h1>
            <p>Trading Performance Analysis</p>
            <p style="margin-top: 10px; font-size: 14px;">
                Period: {df['date'].min().strftime('%B %d, %Y')} - {df['date'].max().strftime('%B %d, %Y')} |
                Report Date: {report_date.strftime('%B %d, %Y')}
            </p>
        </div>

        <div class="content">
            <!-- Key Metrics -->
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="label">Initial Portfolio</div>
                        <div class="value">{initial_portfolio_quote:,.2f} {self.quote_asset}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Final Portfolio</div>
                        <div class="value">{initial_portfolio_quote + total_pnl_financial:,.2f} {self.quote_asset}</div>
                    </div>
                    <div class="metric-card {'positive' if total_pnl_financial > 0 else 'negative'}">
                        <div class="label">Total P&L</div>
                        <div class="value">{total_pnl_financial:+,.2f} {self.quote_asset}</div>
                    </div>
                    <div class="metric-card {'positive' if total_pnl_financial > 0 else 'negative'}">
                        <div class="label">Return</div>
                        <div class="value">{(total_pnl_financial / initial_portfolio_quote * 100):+.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Total Trades</div>
                        <div class="value">{total_trades:,}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Trading Days</div>
                        <div class="value">{num_days}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Win Rate</div>
                        <div class="value">{win_rate:.1f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Avg Daily P&L</div>
                        <div class="value">{total_pnl_financial / num_days:+,.2f} {self.quote_asset}</div>
                    </div>
                </div>
            </div>

            <!-- Performance Chart -->
            <div class="section">
                <h2>Performance Overview</h2>
                {chart_html}
            </div>

            <!-- Performance Highlights -->
            <div class="section">
                <h2>Performance Highlights</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th style="text-align: right;">Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Total Volume (Base)</td>
                            <td style="text-align: right;">{total_volume_base:,.2f}</td>
                        </tr>
                        <tr>
                            <td>Total Volume ({self.quote_asset})</td>
                            <td style="text-align: right;">{total_volume_quote:,.2f} {self.quote_asset}</td>
                        </tr>
                        <tr>
                            <td>Bot P&L (no rebate)</td>
                            <td style="text-align: right;" class="{'positive-value' if total_pnl_bot > 0 else 'negative-value'}">{total_pnl_bot:+,.2f} {self.quote_asset}</td>
                        </tr>
                        <tr>
                            <td>Exchange Rebates</td>
                            <td style="text-align: right;">{total_rebate:,.2f} {self.quote_asset}</td>
                        </tr>
                        <tr>
                            <td>Net Financial P&L</td>
                            <td style="text-align: right;" class="{'positive-value' if total_pnl_financial > 0 else 'negative-value'}">{total_pnl_financial:+,.2f} {self.quote_asset}</td>
                        </tr>
                        <tr>
                            <td>Average Daily Volume</td>
                            <td style="text-align: right;">{total_volume_base/num_days:,.2f}</td>
                        </tr>
                        <tr>
                            <td>Average P&L per Trade</td>
                            <td style="text-align: right;">{total_pnl_financial/total_trades:.4f} {self.quote_asset}</td>
                        </tr>
                        <tr>
                            <td>Best Day</td>
                            <td style="text-align: right;" class="positive-value">{daily_pnl.max():,.2f} {self.quote_asset}</td>
                        </tr>
                        <tr>
                            <td>Worst Day</td>
                            <td style="text-align: right;" class="negative-value">{daily_pnl.min():,.2f} {self.quote_asset}</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <!-- Daily Performance -->
            <div class="section">
                <h2>Daily Performance</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th style="text-align: right;">Trades</th>
                            <th style="text-align: right;">Volume (Base)</th>
                            <th style="text-align: right;">Volume ({self.quote_asset})</th>
                            <th style="text-align: right;">P&L</th>
                            <th style="text-align: right;">Position</th>
                        </tr>
                    </thead>
                    <tbody>
{self._create_daily_table_rows(df)}
                    </tbody>
                </table>
            </div>
        </div>

        <div class="footer">
            <p>Generated with brigado_v2 trading performance framework</p>
            <p style="margin-top: 5px; font-size: 12px; color: #999;">
                🤖 Generated with <a href="https://claude.com/claude-code" style="color: #2E86AB;">Claude Code</a>
            </p>
        </div>
    </div>
</body>
</html>
"""

        # Save HTML file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"HTML report generated: {output_path}")
        return output_path

    def _create_performance_chart(self, daily_agg: pd.DataFrame, initial_portfolio_quote: float) -> go.Figure:
        """Create interactive performance chart."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Portfolio Value Over Time', 'Daily P&L'),
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4]
        )

        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=daily_agg['date'],
                y=daily_agg['portfolio_value'],
                mode='lines+markers',
                name='Portfolio Value',
                line=dict(color='#06A77D', width=3),
                fill='tozeroy',
                fillcolor='rgba(6, 167, 125, 0.1)',
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>' +
                             f'Portfolio: %{{y:,.2f}} {self.quote_asset}<br>' +
                             '<extra></extra>'
            ),
            row=1, col=1
        )

        # Initial portfolio line
        fig.add_hline(
            y=initial_portfolio_quote,
            line=dict(color='gray', dash='dash', width=1),
            row=1, col=1,
            annotation_text=f"Initial: {initial_portfolio_quote:,.0f}",
            annotation_position="right"
        )

        # Daily PnL bars
        colors_pnl = ['#06A77D' if x >= 0 else '#D62828' for x in daily_agg['pnl_with_rebate']]
        fig.add_trace(
            go.Bar(
                x=daily_agg['date'],
                y=daily_agg['pnl_with_rebate'],
                name='Daily P&L',
                marker_color=colors_pnl,
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>' +
                             f'P&L: %{{y:,.2f}} {self.quote_asset}<br>' +
                             '<extra></extra>'
            ),
            row=2, col=1
        )

        fig.update_layout(
            height=700,
            showlegend=False,
            template='plotly_white',
            font=dict(size=12),
            hovermode='x unified',
            margin=dict(l=60, r=40, t=80, b=40)
        )

        fig.update_yaxes(title_text=f"Portfolio ({self.quote_asset})", row=1, col=1)
        fig.update_yaxes(title_text=f"Daily P&L ({self.quote_asset})", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)

        return fig

    def _create_daily_table_rows(self, df: pd.DataFrame) -> str:
        """Generate HTML table rows for daily performance."""
        rows = []
        for _, row in df.iterrows():
            rebate = row['bot_volume_quote'] * self.rebate_pct
            net_pnl = row['bot_realized_pnl'] + rebate
            pnl_class = 'positive-value' if net_pnl > 0 else 'negative-value'

            rows.append(f"""                        <tr>
                            <td>{row['date'].strftime('%Y-%m-%d')}</td>
                            <td style="text-align: right;">{int(row['bot_trades_count']):,}</td>
                            <td style="text-align: right;">{row['bot_volume_base']:,.2f}</td>
                            <td style="text-align: right;">{row['bot_volume_quote']:,.2f}</td>
                            <td style="text-align: right;" class="{pnl_class}">{net_pnl:+,.2f}</td>
                            <td style="text-align: right;">{row['bot_final_position']:,.2f}</td>
                        </tr>""")

        return '\n'.join(rows)


def generate_consolidation_report_html(
    output_path: Path,
    metadata: dict,
    databases: list,
    trades_with_ctrl: pd.DataFrame,
    controllers_data: pd.DataFrame,
    output_paths: dict
) -> Path:
    """
    Generate HTML consolidation report with detailed controller breakdown.

    Args:
        output_path: Path to save HTML file
        metadata: Consolidation metadata dict
        databases: List of database info dicts
        trades_with_ctrl: Trades DataFrame with controller_id column
        controllers_data: Controllers DataFrame
        output_paths: Dict of output file paths

    Returns:
        Path to generated HTML file
    """
    from datetime import datetime

    # Ensure quote_volume column exists
    if 'quote_volume' not in trades_with_ctrl.columns:
        trades_with_ctrl["quote_volume"] = trades_with_ctrl["amount"] * trades_with_ctrl["price"]

    # Build HTML content
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Consolidation Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #eaecef; background: #0b0e11; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: #1e2329; border-radius: 8px; border: 1px solid #2b3139; overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #2b3139 0%, #1e2329 100%); border-bottom: 3px solid #f0b90b; color: #f0b90b; padding: 40px; text-align: center; position: relative; }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; font-weight: 600; text-shadow: 0 0 20px rgba(240, 185, 11, 0.3); }}
        .header p {{ font-size: 1.1em; color: #848e9c; }}
        .nav-link {{ position: absolute; top: 20px; left: 20px; background: #2b3139; color: #f0b90b; padding: 10px 20px; border-radius: 4px; text-decoration: none; border: 1px solid #f0b90b; transition: all 0.3s; font-weight: 600; }}
        .nav-link:hover {{ background: #f0b90b; color: #0b0e11; }}
        .content {{ padding: 40px; }}
        .section {{ margin-bottom: 40px; }}
        .section h2 {{ font-size: 1.8em; margin-bottom: 20px; color: #f0b90b; border-bottom: 2px solid #f0b90b; padding-bottom: 10px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ background: #2b3139; padding: 25px; border-radius: 8px; border: 1px solid #3d4551; transition: border-color 0.3s; }}
        .metric-card:hover {{ border-color: #f0b90b; }}
        .metric-card h3 {{ font-size: 0.85em; color: #848e9c; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }}
        .metric-card .value {{ font-size: 2em; font-weight: bold; color: #f0b90b; }}
        .metric-card .subtitle {{ font-size: 0.9em; color: #b7bdc6; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; background: #2b3139; border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #3d4551; }}
        th {{ background: #1e2329; font-weight: 600; color: #f0b90b; text-transform: uppercase; font-size: 0.85em; }}
        tr.bot-row {{ background: #2b3139; font-weight: 500; cursor: pointer; transition: background 0.2s; }}
        tr.bot-row:hover {{ background: #3d4551; }}
        tr.bot-row td {{ border-bottom: 2px solid #f0b90b; }}
        tr.controller-row {{ background: #1e2329; font-size: 0.9em; color: #b7bdc6; }}
        tr.controller-row td {{ padding-left: 40px; border-bottom: 1px solid #3d4551; }}
        tr.controller-row:hover {{ background: #2b3139; }}
        tr.orphan-row {{ background: #2b2024; border: 1px solid #f6465d; }}
        tr.orphan-row:hover {{ background: #3d2d32; }}
        .expand-icon {{ display: inline-block; margin-right: 8px; transition: transform 0.3s; font-size: 0.8em; color: #f0b90b; }}
        .expand-icon.expanded {{ transform: rotate(90deg); }}
        .controller-row.hidden {{ display: none; }}
        .success-badge {{ background: #f0b90b; color: #0b0e11; padding: 8px 16px; border-radius: 4px; display: inline-block; font-weight: 600; }}
        .controller-badge {{ display: inline-block; background: #2b3139; color: #f0b90b; border: 1px solid #f0b90b; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; margin-left: 8px; font-weight: 600; }}
        .coverage-badge {{ display: inline-block; background: #0ecb81; color: #0b0e11; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; margin-left: 8px; font-weight: 600; }}
        .coverage-badge.warning {{ background: #f0b90b; }}
        .coverage-badge.error {{ background: #f6465d; }}
        .footer {{ background: #1e2329; border-top: 1px solid #2b3139; padding: 20px; text-align: center; color: #848e9c; font-size: 0.9em; }}
        .timestamp {{ color: #848e9c; font-size: 0.9em; margin-top: 10px; }}
    </style>
    <script>
        function toggleControllers(botIndex) {{
            const rows = document.querySelectorAll('.controller-row[data-bot="' + botIndex + '"]');
            const icon = document.querySelector('.expand-icon[data-bot="' + botIndex + '"]');

            rows.forEach(row => {{
                row.classList.toggle('hidden');
            }});

            if (icon) {{
                icon.classList.toggle('expanded');
            }}
        }}
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <a href="index.html" class="nav-link">← Back to Index</a>
            <h1>📊 Data Consolidation Report</h1>
            <p>Automated Database Consolidation Summary</p>
            <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        <div class="content">
            <div class="section">
                <h2>📋 Consolidation Overview</h2>
                <div style="text-align: center; margin: 20px 0;">
                    <span class="success-badge">✓ Consolidation Successful</span>
                </div>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h3>Total Trades</h3>
                        <div class="value">{metadata['record_counts']['trades']:,}</div>
                        <div class="subtitle">across all databases</div>
                    </div>
                    <div class="metric-card">
                        <h3>Total Orders</h3>
                        <div class="value">{metadata['record_counts']['orders']:,}</div>
                        <div class="subtitle">order records</div>
                    </div>
                    <div class="metric-card">
                        <h3>Total Executors</h3>
                        <div class="value">{metadata['record_counts']['executors']:,}</div>
                        <div class="subtitle">executor instances</div>
                    </div>
                    <div class="metric-card">
                        <h3>Controllers</h3>
                        <div class="value">{metadata['record_counts']['controllers']}</div>
                        <div class="subtitle">unique controllers</div>
                    </div>
                </div>
            </div>
            <div class="section">
                <h2>📈 Trading Statistics</h2>
    """

    # Calculate activity days
    start_date = trades_with_ctrl['timestamp'].min()
    end_date = trades_with_ctrl['timestamp'].max()
    total_days = (end_date - start_date).days + 1

    # Get unique trading days
    trades_with_ctrl['trade_date'] = pd.to_datetime(trades_with_ctrl['timestamp']).dt.date
    trading_days = trades_with_ctrl['trade_date'].nunique()
    no_trading_days = total_days - trading_days
    activity_sla = (trading_days / total_days * 100) if total_days > 0 else 0

    html_content += f"""
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h3>Date Range</h3>
                        <div class="value" style="font-size: 1.2em;">{trades_with_ctrl['timestamp'].min().strftime('%Y-%m-%d')}</div>
                        <div class="subtitle">to {trades_with_ctrl['timestamp'].max().strftime('%Y-%m-%d')}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Active Bots</h3>
                        <div class="value">{trades_with_ctrl['source_bot'].nunique()}</div>
                        <div class="subtitle">bot instances</div>
                    </div>
                    <div class="metric-card">
                        <h3>Total Volume</h3>
                        <div class="value">{trades_with_ctrl['quote_volume'].sum():,.0f}</div>
                        <div class="subtitle">quote currency</div>
                    </div>
                    <div class="metric-card">
                        <h3>Activity Days</h3>
                        <div class="value">{trading_days} <span style="color: #f6465d; font-size: 0.6em;">({no_trading_days})</span></div>
                        <div class="subtitle">SLA: {activity_sla:.1f}%</div>
                    </div>
                </div>

                <h3 style="margin-top: 30px; margin-bottom: 15px;">Per-Bot Performance <span style="font-size: 0.8em; color: #666; font-weight: normal;">(Click to expand controllers)</span></h3>
                <table>
                    <thead>
                        <tr>
                            <th>Bot / Controller</th>
                            <th>Trades</th>
                            <th>Total Amount</th>
                            <th>Total Volume</th>
                            <th>Trading Pairs</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    # Add bot rows with controller breakdowns
    # Get all unique bots from both trades and controllers (some bots may have executors but no trades)
    bots_from_trades = set(trades_with_ctrl['source_bot'].unique())
    bots_from_controllers = set(controllers_data['source_bot'].unique())
    all_bots = sorted(bots_from_trades | bots_from_controllers)

    bot_index = 0
    for bot in all_bots:
        bot_trades = trades_with_ctrl[trades_with_ctrl['source_bot'] == bot]
        # Filter out None values and convert to string
        unique_pairs = [str(p) for p in bot_trades['symbol'].unique() if p is not None]
        pairs = ', '.join(unique_pairs) if len(unique_pairs) > 0 else 'No trades'

        # Get controllers for this bot
        bot_controllers = controllers_data[controllers_data['source_bot'] == bot]['id'].tolist()

        # Calculate coverage for this bot
        bot_coverage = (bot_trades['controller_id'].notna().sum() / len(bot_trades)) * 100 if len(bot_trades) > 0 else 0

        num_controllers = len(bot_controllers)
        controller_badge = f'<span class="controller-badge">{num_controllers} controllers</span>' if num_controllers > 0 else ''

        # Color code coverage badge
        if bot_coverage >= 80:
            coverage_class = ""
        elif bot_coverage >= 50:
            coverage_class = " warning"
        else:
            coverage_class = " error"
        coverage_badge = f'<span class="coverage-badge{coverage_class}">{bot_coverage:.0f}% mapped</span>'

        html_content += f"""
                        <tr class="bot-row" onclick="toggleControllers({bot_index})">
                            <td>
                                <span class="expand-icon" data-bot="{bot_index}">▶</span>
                                <strong>{bot}</strong>
                                {controller_badge}
                                {coverage_badge}
                            </td>
                            <td><strong>{len(bot_trades):,}</strong></td>
                            <td><strong>{bot_trades['amount'].sum():,.2f}</strong></td>
                            <td><strong>{bot_trades['quote_volume'].sum():,.0f}</strong></td>
                            <td>{pairs}</td>
                        </tr>"""

        # Add controller breakdown rows
        if num_controllers > 0:
            for controller_id in bot_controllers:
                ctrl_trades = bot_trades[bot_trades['controller_id'] == controller_id]

                if len(ctrl_trades) > 0:
                    ctrl_pairs = ', '.join(ctrl_trades['symbol'].unique())
                    html_content += f"""
                        <tr class="controller-row hidden" data-bot="{bot_index}">
                            <td>↳ {controller_id}</td>
                            <td>{len(ctrl_trades):,}</td>
                            <td>{ctrl_trades['amount'].sum():,.2f}</td>
                            <td>{ctrl_trades['quote_volume'].sum():,.0f}</td>
                            <td>{ctrl_pairs}</td>
                        </tr>"""
                else:
                    html_content += f"""
                        <tr class="controller-row hidden" data-bot="{bot_index}">
                            <td>↳ {controller_id}</td>
                            <td colspan="4" style="color: #999; font-style: italic;">No trades mapped</td>
                        </tr>"""

        bot_index += 1

    # Add Orphan Trades section
    orphan_trades = trades_with_ctrl[trades_with_ctrl['controller_id'].isna()]
    if len(orphan_trades) > 0:
        # Group orphan trades by trading pair
        orphan_by_pair = orphan_trades.groupby('symbol').agg({
            'order_id': 'count',
            'amount': 'sum',
            'quote_volume': 'sum'
        }).reset_index()
        orphan_by_pair.columns = ['symbol', 'trades', 'amount', 'volume']

        html_content += f"""
                        <tr class="orphan-row" style="cursor: default;">
                            <td>
                                <strong>⚠️ Orphan Trades</strong>
                                <span class="controller-badge" style="background: #f6465d; color: #fff; border-color: #f6465d;">{len(orphan_trades):,} unmapped</span>
                            </td>
                            <td><strong>{len(orphan_trades):,}</strong></td>
                            <td><strong>{orphan_trades['amount'].sum():,.2f}</strong></td>
                            <td><strong>{orphan_trades['quote_volume'].sum():,.0f}</strong></td>
                            <td>Multiple pairs</td>
                        </tr>"""

        # Add breakdown by trading pair
        for _, pair_row in orphan_by_pair.iterrows():
            html_content += f"""
                        <tr class="controller-row" style="background: #2b3139;">
                            <td style="padding-left: 40px; color: #f0b90b;">↳ {pair_row['symbol']}</td>
                            <td>{int(pair_row['trades']):,}</td>
                            <td>{pair_row['amount']:,.2f}</td>
                            <td>{pair_row['volume']:,.0f}</td>
                            <td>{pair_row['symbol']}</td>
                        </tr>"""

    html_content += """
                    </tbody>
                </table>
            </div>
        </div>
        <div class="footer">
            <p>Generated by Brigado v2 Data Consolidator</p>
            <p style="margin-top: 5px; font-size: 0.85em;">Click bot rows to expand/collapse controller details</p>
        </div>
    </div>
</body>
</html>"""

    # Save HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return output_path


def generate_index_html(output_dir: Path, metadata: dict = None) -> Path:
    """
    Generate index.html navigation page for all reports.

    Args:
        output_dir: Directory containing the reports (data_sources)
        metadata: Optional consolidation metadata for stats

    Returns:
        Path to generated index.html
    """
    from datetime import datetime
    import os

    # Check which reports exist
    reports = {
        'consolidation': output_dir / 'consolidation_report.html',
        'market_analysis': output_dir / 'market_analysis_report.html',
        'evolutive': output_dir / 'evolutive_report.html',
        'portfolio_status': output_dir / 'portfolio_status_report.html',
        'pnl': output_dir / 'pnl_report.html',
    }

    available_reports = {name: path for name, path in reports.items() if path.exists()}

    # Get file sizes and modification times
    report_info = {}
    for name, path in available_reports.items():
        stat = path.stat()
        report_info[name] = {
            'path': path.name,
            'size_kb': stat.st_size / 1024,
            'modified': datetime.fromtimestamp(stat.st_mtime)
        }

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Brigado v2 - Performance Reports</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #eaecef;
            background: #0b0e11;
            min-height: 100vh;
            padding: 40px 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        .header {{
            text-align: center;
            color: #f0b90b;
            margin-bottom: 50px;
        }}

        .header h1 {{
            font-size: 3em;
            margin-bottom: 10px;
            font-weight: 700;
            text-shadow: 0 0 20px rgba(240, 185, 11, 0.3);
        }}

        .header p {{
            font-size: 1.2em;
            color: #848e9c;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}

        .stat-card {{
            background: #1e2329;
            padding: 25px;
            border-radius: 8px;
            border: 1px solid #2b3139;
            text-align: center;
            transition: border-color 0.3s;
        }}

        .stat-card:hover {{
            border-color: #f0b90b;
        }}

        .stat-card h3 {{
            font-size: 0.85em;
            color: #848e9c;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .stat-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #f0b90b;
        }}

        .reports-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 30px;
        }}

        .report-card {{
            background: #1e2329;
            border-radius: 8px;
            border: 1px solid #2b3139;
            overflow: hidden;
            transition: transform 0.3s, border-color 0.3s, box-shadow 0.3s;
            text-decoration: none;
            color: inherit;
            display: block;
        }}

        .report-card:hover {{
            transform: translateY(-5px);
            border-color: #f0b90b;
            box-shadow: 0 8px 25px rgba(240, 185, 11, 0.15);
        }}

        .report-header {{
            background: linear-gradient(135deg, #2b3139 0%, #1e2329 100%);
            border-bottom: 2px solid #f0b90b;
            color: #f0b90b;
            padding: 30px;
            text-align: center;
        }}

        .report-header .icon {{
            font-size: 3em;
            margin-bottom: 10px;
            filter: drop-shadow(0 0 10px rgba(240, 185, 11, 0.3));
        }}

        .report-header h2 {{
            font-size: 1.5em;
            margin-bottom: 5px;
            color: #eaecef;
        }}

        .report-header p {{
            color: #848e9c;
            font-size: 0.9em;
        }}

        .report-body {{
            padding: 25px;
        }}

        .report-meta {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid #2b3139;
        }}

        .meta-item {{
            display: flex;
            flex-direction: column;
        }}

        .meta-label {{
            font-size: 0.75em;
            color: #848e9c;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}

        .meta-value {{
            font-size: 0.9em;
            color: #eaecef;
            font-weight: 500;
        }}

        .report-description {{
            color: #b7bdc6;
            font-size: 0.95em;
            line-height: 1.6;
        }}

        .view-button {{
            display: block;
            background: #f0b90b;
            color: #0b0e11;
            text-align: center;
            padding: 12px;
            border-radius: 4px;
            margin-top: 20px;
            font-weight: 600;
            transition: background 0.3s, transform 0.2s;
        }}

        .view-button:hover {{
            background: #fcd535;
            transform: scale(1.02);
        }}

        .no-reports {{
            background: #1e2329;
            padding: 60px;
            border-radius: 8px;
            border: 1px solid #2b3139;
            text-align: center;
        }}

        .no-reports h2 {{
            color: #f0b90b;
            margin-bottom: 15px;
        }}

        .no-reports p {{
            color: #848e9c;
            font-size: 1.1em;
        }}

        .footer {{
            text-align: center;
            color: #848e9c;
            margin-top: 50px;
            font-size: 0.9em;
        }}

        .footer p {{
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Brigado v2</h1>
            <p>Performance Analysis Dashboard</p>
        </div>
"""

    # Add stats if metadata is available
    if metadata:
        html_content += f"""
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Trades</h3>
                <div class="value">{metadata['record_counts']['trades']:,}</div>
            </div>
            <div class="stat-card">
                <h3>Total Orders</h3>
                <div class="value">{metadata['record_counts']['orders']:,}</div>
            </div>
            <div class="stat-card">
                <h3>Executors</h3>
                <div class="value">{metadata['record_counts']['executors']:,}</div>
            </div>
            <div class="stat-card">
                <h3>Controllers</h3>
                <div class="value">{metadata['record_counts']['controllers']}</div>
            </div>
        </div>
"""

    if available_reports:
        html_content += """
        <div class="reports-grid">
"""

        # Consolidation Report
        if 'consolidation' in available_reports:
            info = report_info['consolidation']
            html_content += f"""
            <a href="{info['path']}" class="report-card">
                <div class="report-header">
                    <div class="icon">📋</div>
                    <h2>Consolidation Report</h2>
                    <p>Data consolidation summary</p>
                </div>
                <div class="report-body">
                    <div class="report-meta">
                        <div class="meta-item">
                            <span class="meta-label">Last Updated</span>
                            <span class="meta-value">{info['modified'].strftime('%Y-%m-%d %H:%M')}</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">Size</span>
                            <span class="meta-value">{info['size_kb']:.1f} KB</span>
                        </div>
                    </div>
                    <div class="report-description">
                        View consolidated data from all bot databases. Includes controller breakdown,
                        orphan trades, and coverage statistics per bot.
                    </div>
                    <div class="view-button">View Report →</div>
                </div>
            </a>
"""

        # Market Analysis Report
        if 'market_analysis' in available_reports:
            info = report_info['market_analysis']
            html_content += f"""
            <a href="{info['path']}" class="report-card">
                <div class="report-header">
                    <div class="icon">📈</div>
                    <h2>Market Analysis</h2>
                    <p>Market share & performance</p>
                </div>
                <div class="report-body">
                    <div class="report-meta">
                        <div class="meta-item">
                            <span class="meta-label">Last Updated</span>
                            <span class="meta-value">{info['modified'].strftime('%Y-%m-%d %H:%M')}</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">Size</span>
                            <span class="meta-value">{info['size_kb']:.1f} KB</span>
                        </div>
                    </div>
                    <div class="report-description">
                        Controller performance vs market activity by trading pair.
                        Includes OHLC data, market share calculations, and volume comparisons.
                    </div>
                    <div class="view-button">View Report →</div>
                </div>
            </a>
"""

        # Evolutive Report
        if 'evolutive' in available_reports:
            info = report_info['evolutive']
            html_content += f"""
            <a href="{info['path']}" class="report-card">
                <div class="report-header">
                    <div class="icon">📊</div>
                    <h2>Evolutive Report</h2>
                    <p>Daily metrics evolution</p>
                </div>
                <div class="report-body">
                    <div class="report-meta">
                        <div class="meta-item">
                            <span class="meta-label">Last Updated</span>
                            <span class="meta-value">{info['modified'].strftime('%Y-%m-%d %H:%M')}</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">Size</span>
                            <span class="meta-value">{info['size_kb']:.1f} KB</span>
                        </div>
                    </div>
                    <div class="report-description">
                        Daily evolution of market, bot, and controller metrics over time.
                        Interactive charts showing performance trends, volume evolution, and market share progression.
                    </div>
                    <div class="view-button">View Report →</div>
                </div>
            </a>
"""

        # Portfolio Status Report
        if 'portfolio_status' in available_reports:
            info = report_info['portfolio_status']
            html_content += f"""
            <a href="{info['path']}" class="report-card">
                <div class="report-header">
                    <div class="icon">💼</div>
                    <h2>Portfolio Status</h2>
                    <p>Current holdings & evolution</p>
                </div>
                <div class="report-body">
                    <div class="report-meta">
                        <div class="meta-item">
                            <span class="meta-label">Last Updated</span>
                            <span class="meta-value">{info['modified'].strftime('%Y-%m-%d %H:%M')}</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">Size</span>
                            <span class="meta-value">{info['size_kb']:.1f} KB</span>
                        </div>
                    </div>
                    <div class="report-description">
                        Portfolio value breakdown and token holdings evolution.
                        View current allocation, historical snapshots, and asset performance over time.
                    </div>
                    <div class="view-button">View Report →</div>
                </div>
            </a>
"""

        # PnL Report
        if 'pnl' in available_reports:
            info = report_info['pnl']
            html_content += f"""
            <a href="{info['path']}" class="report-card">
                <div class="report-header">
                    <div class="icon">💹</div>
                    <h2>PnL Analysis</h2>
                    <p>Profit & loss tracking</p>
                </div>
                <div class="report-body">
                    <div class="report-meta">
                        <div class="meta-item">
                            <span class="meta-label">Last Updated</span>
                            <span class="meta-value">{info['modified'].strftime('%Y-%m-%d %H:%M')}</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">Size</span>
                            <span class="meta-value">{info['size_kb']:.1f} KB</span>
                        </div>
                    </div>
                    <div class="report-description">
                        Bot-level profit and loss analysis with position tracking, break-even prices,
                        realized/unrealized PnL, and daily performance breakdown.
                    </div>
                    <div class="view-button">View Report →</div>
                </div>
            </a>
"""

        html_content += """
        </div>
"""
    else:
        html_content += """
        <div class="no-reports">
            <h2>No Reports Available</h2>
            <p>Run the consolidation and analysis notebooks to generate reports.</p>
        </div>
"""

    html_content += f"""
        <div class="footer">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
            <p>Brigado v2 Performance Analysis System</p>
        </div>
    </div>
</body>
</html>"""

    # Save index.html
    index_path = output_dir / 'index.html'
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return index_path
