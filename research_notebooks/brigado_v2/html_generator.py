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
    Generate HTML consolidation report.

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
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; background: #f5f5f5; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; text-align: center; }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; font-weight: 600; }}
        .header p {{ font-size: 1.1em; opacity: 0.9; }}
        .content {{ padding: 40px; }}
        .section {{ margin-bottom: 40px; }}
        .section h2 {{ font-size: 1.8em; margin-bottom: 20px; color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 25px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .metric-card h3 {{ font-size: 0.9em; color: #666; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }}
        .metric-card .value {{ font-size: 2em; font-weight: bold; color: #333; }}
        .metric-card .subtitle {{ font-size: 0.9em; color: #777; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; background: white; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
        th {{ background: #f8f9fa; font-weight: 600; color: #555; text-transform: uppercase; font-size: 0.85em; }}
        .success-badge {{ background: #10b981; color: white; padding: 8px 16px; border-radius: 20px; display: inline-block; font-weight: 600; }}
        .footer {{ background: #f8f9fa; padding: 20px; text-align: center; color: #666; font-size: 0.9em; }}
        .timestamp {{ color: #eee; font-size: 0.9em; margin-top: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
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
                </div>
                <table>
                    <thead>
                        <tr><th>Bot</th><th>Trades</th><th>Volume</th><th>Controllers</th></tr>
                    </thead>
                    <tbody>
"""

    # Add bot rows
    for bot in sorted(trades_with_ctrl['source_bot'].unique()):
        bot_trades = trades_with_ctrl[trades_with_ctrl['source_bot'] == bot]
        bot_controllers = controllers_data[controllers_data['source_bot'] == bot]['id'].nunique()
        html_content += f"""
                        <tr>
                            <td><strong>{bot}</strong></td>
                            <td>{len(bot_trades):,}</td>
                            <td>{bot_trades['quote_volume'].sum():,.0f}</td>
                            <td>{bot_controllers}</td>
                        </tr>"""

    html_content += """
                    </tbody>
                </table>
            </div>
        </div>
        <div class="footer">
            <p>Generated by Brigado v2 Data Consolidator</p>
        </div>
    </div>
</body>
</html>"""

    # Save HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return output_path
