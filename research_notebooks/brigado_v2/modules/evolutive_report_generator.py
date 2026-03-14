"""
Evolutive Report HTML Generator

Generates interactive HTML report showing daily evolution of metrics
from market, bot, and controller perspectives.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
from typing import List
from research_notebooks.brigado_v2.modules.pivot_table_generator import generate_pivot_table_html


def generate_evolutive_report_html(
    evolutive_metrics: pd.DataFrame,
    output_path: Path,
    metadata: dict = None,
    trades_with_ctrl: pd.DataFrame = None
) -> Path:
    """
    Generate evolutive HTML report with time series charts.

    Args:
        evolutive_metrics: DataFrame with daily metrics evolution
        output_path: Path to save HTML file
        metadata: Optional consolidation metadata
        trades_with_ctrl: Enriched trades DataFrame with controller mapping

    Returns:
        Path to generated HTML file
    """
    print("\n📊 Generating evolutive report HTML...")

    # Convert date to datetime for plotting
    df = evolutive_metrics.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Get unique symbols
    symbols = sorted(df['symbol'].unique())

    print(f"  • Processing {len(symbols)} trading pairs")
    print(f"  • Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Build HTML
    html_content = _generate_html_header()

    # Add summary section
    html_content += _generate_summary_section(df, metadata)

    # Add tab navigation
    html_content += """
            <div class="tabs">
"""
    for i, symbol in enumerate(symbols):
        active_class = "active" if i == 0 else ""
        html_content += f"""                <div class="tab {active_class}" onclick="switchTab('{symbol}')">{symbol}</div>
"""
    html_content += """            </div>
"""

    # Calculate total trades across all symbols for percentage calculation
    total_bot_trades = df['bot_trades_count'].sum()

    # Add tab content for each symbol
    for i, symbol in enumerate(symbols):
        symbol_data = df[df['symbol'] == symbol].sort_values('date')
        symbol_trades = trades_with_ctrl[trades_with_ctrl['symbol'] == symbol] if trades_with_ctrl is not None else None
        active_class = "active" if i == 0 else ""
        html_content += f"""
            <div class="tab-content {active_class}" id="tab-{symbol}">
"""
        html_content += _generate_symbol_section(symbol, symbol_data, symbol_trades, total_bot_trades)
        html_content += """
            </div>
"""

    # Add JavaScript for tab switching
    html_content += _generate_tab_switching_js(symbols)

    # Footer
    html_content += _generate_html_footer()

    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"  ✓ Evolutive report saved to: {output_path}")
    return output_path


def _generate_html_header() -> str:
    """Generate HTML header with styles."""
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evolutive Performance Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #eaecef;
            background: #0b0e11;
            padding: 20px;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: #1e2329;
            border-radius: 8px;
            border: 1px solid #2b3139;
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2b3139 0%, #1e2329 100%);
            border-bottom: 3px solid #f0b90b;
            color: #f0b90b;
            padding: 40px;
            text-align: center;
            position: relative;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 600;
            text-shadow: 0 0 20px rgba(240, 185, 11, 0.3);
        }}
        .header p {{
            font-size: 1.1em;
            color: #848e9c;
        }}
        .nav-link {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: #2b3139;
            color: #f0b90b;
            padding: 10px 20px;
            border-radius: 4px;
            text-decoration: none;
            border: 1px solid #f0b90b;
            transition: all 0.3s;
            font-weight: 600;
        }}
        .nav-link:hover {{
            background: #f0b90b;
            color: #0b0e11;
        }}
        .content {{
            padding: 40px;
        }}
        .section {{
            background: #2b3139;
            border-radius: 8px;
            padding: 30px;
            margin-bottom: 30px;
            border-left: 5px solid #f0b90b;
        }}
        .section h2 {{
            font-size: 2em;
            color: #f0b90b;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .section h3 {{
            font-size: 1.3em;
            color: #848e9c;
            margin: 25px 0 15px 0;
            border-bottom: 1px solid #3d4551;
            padding-bottom: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }}
        .metric-card {{
            background: #1e2329;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #3d4551;
            transition: border-color 0.3s;
        }}
        .metric-card:hover {{
            border-color: #f0b90b;
        }}
        .metric-card h4 {{
            font-size: 0.75em;
            color: #848e9c;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-card .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #f0b90b;
        }}
        .metric-card .subtitle {{
            font-size: 0.85em;
            color: #b7bdc6;
            margin-top: 5px;
        }}
        .metric-card.positive .value {{
            color: #0ecb81;
        }}
        .metric-card.negative .value {{
            color: #f6465d;
        }}
        .chart-container {{
            background: #1e2329;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            border: 1px solid #3d4551;
        }}
        .footer {{
            background: #1e2329;
            border-top: 1px solid #2b3139;
            padding: 20px;
            text-align: center;
            color: #848e9c;
            font-size: 0.9em;
        }}
        .timestamp {{
            color: #848e9c;
            font-size: 0.9em;
            margin-top: 10px;
        }}
        .nav-buttons {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }}
        .nav-button {{
            background: #2b3139;
            color: #f0b90b;
            border: 1px solid #f0b90b;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.3s;
            text-decoration: none;
            display: inline-block;
        }}
        .nav-button:hover {{
            background: #f0b90b;
            color: #0b0e11;
        }}
        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
            flex-wrap: wrap;
            border-bottom: 2px solid #2b3139;
            padding-bottom: 0;
        }}
        .tab {{
            background: #2b3139;
            color: #848e9c;
            border: 1px solid #3d4551;
            border-bottom: none;
            padding: 12px 24px;
            cursor: pointer;
            transition: all 0.3s;
            border-radius: 8px 8px 0 0;
            font-weight: 500;
            font-size: 1.1em;
            position: relative;
            bottom: -2px;
        }}
        .tab:hover {{
            background: #3d4551;
            color: #f0b90b;
        }}
        .tab.active {{
            background: #1e2329;
            color: #f0b90b;
            border-color: #f0b90b;
            border-bottom: 2px solid #1e2329;
            box-shadow: 0 0 20px rgba(240, 185, 11, 0.2);
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
            animation: fadeIn 0.3s;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <a href="index.html" class="nav-link">← Back to Index</a>
            <h1>📈 Evolutive Performance Report</h1>
            <p>Daily Evolution of Market, Bot, and Controller Metrics</p>
            <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        <div class="content">
"""


def _generate_summary_section(df: pd.DataFrame, metadata: dict = None) -> str:
    """Generate summary statistics section."""
    from datetime import datetime, timedelta

    # Calculate overall metrics
    total_days = df.groupby('date').ngroups
    total_trades = df['bot_trades_count'].sum()
    total_volume_quote = df['bot_volume_quote'].sum()

    # Calculate no-trading days
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    trading_days_set = set(pd.to_datetime(df['date']).dt.date)
    no_trading_days = len([d for d in date_range if d.date() not in trading_days_set])

    # Activity SLA %
    total_calendar_days = (pd.to_datetime(max_date) - pd.to_datetime(min_date)).days + 1
    activity_sla = (total_days / total_calendar_days * 100) if total_calendar_days > 0 else 0

    # Market share
    avg_market_share = df['market_share_volume'].mean() if 'market_share_volume' in df.columns else 0

    # Controller coverage
    avg_coverage = df['controller_coverage_pct'].mean() if 'controller_coverage_pct' in df.columns else 0

    html = f"""
            <div class="section">
                <h2>📊 Overall Summary</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h4>Activity Days</h4>
                        <div class="value">{total_days} <span style="color: #f6465d; font-size: 0.6em;">({no_trading_days})</span></div>
                        <div class="subtitle">SLA: {activity_sla:.1f}% active</div>
                    </div>
                    <div class="metric-card">
                        <h4>Total Trades</h4>
                        <div class="value">{total_trades:,}</div>
                        <div class="subtitle">{total_trades/total_days:.1f} per day</div>
                    </div>
                    <div class="metric-card">
                        <h4>Total Volume</h4>
                        <div class="value">{total_volume_quote:,.0f}</div>
                        <div class="subtitle">quote currency</div>
                    </div>
                    <div class="metric-card">
                        <h4>Avg Market Share</h4>
                        <div class="value">{avg_market_share:.2f}%</div>
                        <div class="subtitle">by volume</div>
                    </div>
                    <div class="metric-card">
                        <h4>Controller Coverage</h4>
                        <div class="value">{avg_coverage:.1f}%</div>
                        <div class="subtitle">trades mapped</div>
                    </div>
                </div>
            </div>
"""

    return html


def _generate_symbol_section(symbol: str, df: pd.DataFrame, trades: pd.DataFrame = None, total_bot_trades: int = None) -> str:
    """Generate section with charts for a specific symbol."""
    # Calculate symbol stats
    total_volume = df['bot_volume_quote'].sum()
    total_trades = df['bot_trades_count'].sum()
    total_market_trades = df['market_trades_count'].sum()
    avg_market_share = df['market_share_volume'].mean() if 'market_share_volume' in df.columns else 0

    # Calculate percentage of total market trades for this pair
    trade_percentage = (total_trades / total_market_trades * 100) if total_market_trades and total_market_trades > 0 else 0

    html = f"""
                <div class="section">
                    <h2>
                        <span>{symbol}</span>
                        <span style="font-size: 0.5em; color: #848e9c;">
                            {len(df)} days • {total_trades:,} trades
                        </span>
                    </h2>

                    <div class="metrics-grid" style="grid-template-columns: repeat(3, 1fr);">
                        <div class="metric-card">
                            <h4>Total Volume</h4>
                            <div class="value">{total_volume:,.0f}</div>
                        </div>
                        <div class="metric-card">
                            <h4>Avg Market Share</h4>
                            <div class="value">{avg_market_share:.2f}%</div>
                        </div>
                        <div class="metric-card">
                            <h4>Total Trades</h4>
                            <div class="value">{total_trades:,}</div>
                            <div class="subtitle">{trade_percentage:.1f}% of market</div>
                        </div>
                    </div>

                    <h3>📊 Volume & Market Share</h3>
                    <div class="chart-container" id="volume-{symbol}"></div>

                    <h3>🎯 Trading Activity</h3>
                    <div class="chart-container" id="activity-{symbol}"></div>

                    <h3>🎮 Controller Metrics</h3>
                    <div class="chart-container" id="controllers-{symbol}"></div>

                    {generate_pivot_table_html(symbol, df, trades)}

                    <script>
                        {_generate_volume_chart_js(symbol, df)}
                        {_generate_activity_chart_js(symbol, df)}
                        {_generate_controller_chart_js(symbol, df)}
                    </script>
                </div>
"""

    return html


def _generate_daily_breakdown_table(symbol: str, df: pd.DataFrame) -> str:
    """Generate daily breakdown table with Market/Bots/Controllers sections."""
    html = """
                <h3 style="margin-top: 30px;">📋 Daily Breakdown</h3>
                <table style="margin-top: 15px;">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Section</th>
                            <th>Trades</th>
                            <th>Volume (Base)</th>
                            <th>Volume (Quote)</th>
                            <th>Market Share %</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    # Sort by date
    df_sorted = df.sort_values('date', ascending=False)

    for _, row in df_sorted.iterrows():
        date_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d')

        # Market row
        market_trades = row.get('market_trades_count', 0)
        market_volume_base = row.get('market_volume_base', 0)
        market_volume_quote = row.get('market_volume_quote', 0)

        html += f"""
                        <tr style="background: #1e2329; border-top: 2px solid #f0b90b;">
                            <td rowspan="3" style="vertical-align: middle; font-weight: bold;">{date_str}</td>
                            <td><strong style="color: #848e9c;">📊 Market</strong></td>
                            <td>{market_trades:,}</td>
                            <td>{market_volume_base:,.2f}</td>
                            <td>{market_volume_quote:,.0f}</td>
                            <td>100.00%</td>
                        </tr>
"""

        # All Bots row
        bot_trades = row.get('bot_trades_count', 0)
        bot_volume_base = row.get('bot_volume_base', 0)
        bot_volume_quote = row.get('bot_volume_quote', 0)
        market_share = row.get('market_share_volume', 0)

        html += f"""
                        <tr style="background: #2b3139;">
                            <td><strong style="color: #f0b90b;">🤖 All Bots</strong></td>
                            <td>{bot_trades:,}</td>
                            <td>{bot_volume_base:,.2f}</td>
                            <td>{bot_volume_quote:,.0f}</td>
                            <td style="color: #f0b90b; font-weight: 600;">{market_share:.2f}%</td>
                        </tr>
"""

        # Controllers + Orphans row
        controllers_active = row.get('controllers_active', 0)
        coverage_pct = row.get('controller_coverage_pct', 0)
        orphan_trades = row.get('orphan_trades_count', 0)
        orphan_volume = row.get('orphan_volume_quote', 0)

        controller_text = f"{controllers_active} ctrl • {coverage_pct:.1f}% cov"
        orphan_text = f"{orphan_trades} orphan" if orphan_trades > 0 else ""
        section_label = f"🎮 Controllers ({controller_text})"
        if orphan_text:
            section_label += f" + {orphan_text}"

        orphan_color = "#f6465d" if orphan_trades > 0 else "#0ecb81"

        html += f"""
                        <tr style="background: #1e2329;">
                            <td><span style="color: {orphan_color};">{section_label}</span></td>
                            <td>{bot_trades - orphan_trades:,}</td>
                            <td>{bot_volume_base:,.2f}</td>
                            <td>{bot_volume_quote - orphan_volume:,.0f}</td>
                            <td style="color: #0ecb81; font-weight: 600;">{coverage_pct:.1f}%</td>
                        </tr>
"""

    html += """
                    </tbody>
                </table>
"""

    return html


def _generate_volume_chart_js(symbol: str, df: pd.DataFrame) -> str:
    """Generate JavaScript for volume and market share chart."""
    dates = df['date'].dt.strftime('%Y-%m-%d').tolist()
    bot_volume = df['bot_volume_quote'].fillna(0).tolist()
    market_volume = df['market_volume_quote'].fillna(0).tolist() if 'market_volume_quote' in df.columns else [0] * len(dates)
    market_share = df['market_share_volume'].fillna(0).tolist() if 'market_share_volume' in df.columns else [0] * len(dates)

    return f"""
        var volumeData = [
            {{
                x: {dates},
                y: {bot_volume},
                type: 'bar',
                name: 'Bot Volume',
                marker: {{ color: '#f0b90b' }},
                yaxis: 'y1'
            }},
            {{
                x: {dates},
                y: {market_volume},
                type: 'scatter',
                mode: 'lines',
                name: 'Market Volume',
                line: {{ color: '#848e9c', width: 2, dash: 'dot' }},
                yaxis: 'y1'
            }},
            {{
                x: {dates},
                y: {market_share},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Market Share %',
                line: {{ color: '#0ecb81', width: 2 }},
                yaxis: 'y2'
            }}
        ];

        var volumeLayout = {{
            paper_bgcolor: '#1e2329',
            plot_bgcolor: '#1e2329',
            font: {{ color: '#eaecef' }},
            height: 400,
            autosize: true,
            margin: {{ t: 30, b: 50, l: 60, r: 60 }},
            hovermode: 'x unified',
            showlegend: true,
            legend: {{ x: 0.01, y: 0.99, bgcolor: 'rgba(30, 35, 41, 0.8)' }},
            xaxis: {{
                gridcolor: '#3d4551',
                title: 'Date'
            }},
            yaxis: {{
                title: 'Volume (Quote)',
                gridcolor: '#3d4551',
                side: 'left'
            }},
            yaxis2: {{
                title: 'Market Share (%)',
                gridcolor: '#3d4551',
                side: 'right',
                overlaying: 'y'
            }}
        }};

        Plotly.newPlot('volume-{symbol}', volumeData, volumeLayout, {{responsive: true, displayModeBar: false}});
    """


def _generate_activity_chart_js(symbol: str, df: pd.DataFrame) -> str:
    """Generate JavaScript for trading activity chart."""
    dates = df['date'].dt.strftime('%Y-%m-%d').tolist()
    bot_trades = df['bot_trades_count'].fillna(0).tolist()
    market_trades = df['market_trades_count'].fillna(0).tolist() if 'market_trades_count' in df.columns else [0] * len(dates)
    bot_position = df['bot_final_position'].fillna(0).tolist()

    return f"""
        var activityData = [
            {{
                x: {dates},
                y: {bot_trades},
                type: 'bar',
                name: 'Bot Trades',
                marker: {{ color: '#f0b90b' }},
                yaxis: 'y1'
            }},
            {{
                x: {dates},
                y: {market_trades},
                type: 'scatter',
                mode: 'lines',
                name: 'Market Trades',
                line: {{ color: '#848e9c', width: 2, dash: 'dot' }},
                yaxis: 'y1'
            }},
            {{
                x: {dates},
                y: {bot_position},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Bot Position',
                line: {{ color: '#0ecb81', width: 2 }},
                yaxis: 'y2'
            }}
        ];

        var activityLayout = {{
            paper_bgcolor: '#1e2329',
            plot_bgcolor: '#1e2329',
            font: {{ color: '#eaecef' }},
            height: 400,
            autosize: true,
            margin: {{ t: 30, b: 50, l: 60, r: 60 }},
            hovermode: 'x unified',
            showlegend: true,
            legend: {{ x: 0.01, y: 0.99, bgcolor: 'rgba(30, 35, 41, 0.8)' }},
            xaxis: {{
                gridcolor: '#3d4551',
                title: 'Date'
            }},
            yaxis: {{
                title: 'Trade Count',
                gridcolor: '#3d4551',
                side: 'left'
            }},
            yaxis2: {{
                title: 'Position (Base)',
                gridcolor: '#3d4551',
                side: 'right',
                overlaying: 'y'
            }}
        }};

        Plotly.newPlot('activity-{symbol}', activityData, activityLayout, {{responsive: true, displayModeBar: false}});
    """


def _generate_controller_chart_js(symbol: str, df: pd.DataFrame) -> str:
    """Generate JavaScript for controller metrics chart."""
    dates = df['date'].dt.strftime('%Y-%m-%d').tolist()
    coverage = df['controller_coverage_pct'].fillna(0).tolist()
    active_controllers = df['controllers_active'].fillna(0).tolist()
    orphan_volume = df['orphan_volume_quote'].fillna(0).tolist()

    return f"""
        var controllerData = [
            {{
                x: {dates},
                y: {coverage},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Coverage %',
                line: {{ color: '#0ecb81', width: 3 }},
                fill: 'tozeroy',
                fillcolor: 'rgba(14, 203, 129, 0.1)',
                yaxis: 'y1'
            }},
            {{
                x: {dates},
                y: {active_controllers},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Active Controllers',
                line: {{ color: '#f0b90b', width: 2 }},
                yaxis: 'y2'
            }},
            {{
                x: {dates},
                y: {orphan_volume},
                type: 'bar',
                name: 'Orphan Volume',
                marker: {{ color: '#f6465d', opacity: 0.6 }},
                yaxis: 'y3'
            }}
        ];

        var controllerLayout = {{
            paper_bgcolor: '#1e2329',
            plot_bgcolor: '#1e2329',
            font: {{ color: '#eaecef' }},
            height: 400,
            autosize: true,
            margin: {{ t: 30, b: 50, l: 60, r: 60 }},
            hovermode: 'x unified',
            showlegend: true,
            legend: {{ x: 0.01, y: 0.99, bgcolor: 'rgba(30, 35, 41, 0.8)' }},
            xaxis: {{
                gridcolor: '#3d4551',
                title: 'Date'
            }},
            yaxis: {{
                title: 'Coverage %',
                gridcolor: '#3d4551',
                side: 'left',
                range: [0, 105]
            }},
            yaxis2: {{
                title: 'Active Controllers',
                gridcolor: '#3d4551',
                side: 'right',
                overlaying: 'y',
                showgrid: false
            }},
            yaxis3: {{
                title: 'Orphan Volume',
                anchor: 'free',
                overlaying: 'y',
                side: 'right',
                position: 0.95,
                showgrid: false
            }}
        }};

        Plotly.newPlot('controllers-{symbol}', controllerData, controllerLayout, {{responsive: true, displayModeBar: false}});
    """


def _generate_tab_switching_js(symbols: List[str]) -> str:
    """Generate JavaScript for tab switching functionality."""
    return """
            <script>
                function switchTab(symbol) {
                    // Hide all tab contents
                    var tabContents = document.getElementsByClassName('tab-content');
                    for (var i = 0; i < tabContents.length; i++) {
                        tabContents[i].classList.remove('active');
                    }

                    // Remove active class from all tabs
                    var tabs = document.getElementsByClassName('tab');
                    for (var i = 0; i < tabs.length; i++) {
                        tabs[i].classList.remove('active');
                    }

                    // Show the selected tab content
                    document.getElementById('tab-' + symbol).classList.add('active');

                    // Add active class to the clicked tab
                    event.target.classList.add('active');

                    // Relayout Plotly charts to fix dimensions
                    setTimeout(function() {
                        var charts = document.getElementById('tab-' + symbol).querySelectorAll('.plotly-graph-div');
                        charts.forEach(function(chart) {
                            Plotly.relayout(chart, {autosize: true});
                        });
                    }, 100);

                    // Scroll to top of content
                    window.scrollTo({
                        top: document.querySelector('.tabs').offsetTop - 20,
                        behavior: 'smooth'
                    });
                }
            </script>
"""


def _generate_html_footer() -> str:
    """Generate HTML footer."""
    return """
        </div>
        <div class="footer">
            <p>Generated by Brigado v2 Evolutive Report Generator</p>
            <p style="margin-top: 5px;">
                🤖 Generated with <a href="https://claude.com/claude-code" style="color: #f0b90b;">Claude Code</a>
            </p>
        </div>
    </div>
</body>
</html>
"""
