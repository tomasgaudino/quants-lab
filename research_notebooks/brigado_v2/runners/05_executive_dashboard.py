#!/usr/bin/env python3
"""
Executive Dashboard

Single-page executive view with all key metrics:
- Total Net Asset Value (NAV)
- Current Positions (end of day)
- Daily Portfolio Evolution
- Overall Bots Metrics (from daily pivot)
- Trading pair selector for detailed base/quote volumes
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from research_notebooks.brigado_v2.modules.file_manager import FileManager


def print_header(title: str):
    """Print formatted header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_metric(label: str, value: str, indent: int = 2):
    """Print formatted metric."""
    spaces = " " * indent
    print(f"{spaces}{label:<40} {value:>35}")


def generate_executive_dashboard(
    nav_evolution: pd.DataFrame,
    trades: pd.DataFrame,
    market_data: pd.DataFrame = None
):
    """
    Generate executive dashboard HTML with 2-column layout.

    Args:
        nav_evolution: DataFrame with daily NAV evolution
        trades: Consolidated trades DataFrame
        market_data: Market data DataFrame (optional)
    """

    # Get latest data
    latest_date = nav_evolution['date'].max()
    latest_nav = nav_evolution[nav_evolution['date'] == latest_date].iloc[0]

    current_nav_brl = latest_nav['nav_brl']
    current_nav_usdt = latest_nav['nav_usdt']

    # Get current positions
    btc_balance = latest_nav['btc_balance']
    usdt_balance = latest_nav['usdt_balance']
    brl_balance = latest_nav['brl_balance']

    btc_price = latest_nav['btc_price']
    usdt_price = latest_nav['usdt_price']

    btc_value = latest_nav['btc_value']
    usdt_value = latest_nav['usdt_value']
    brl_value = latest_nav['brl_value']

    # Calculate percentages
    btc_pct = (btc_value / current_nav_brl * 100) if current_nav_brl > 0 else 0
    usdt_pct = (usdt_value / current_nav_brl * 100) if current_nav_brl > 0 else 0
    brl_pct = (brl_value / current_nav_brl * 100) if current_nav_brl > 0 else 0

    # Calculate daily bot metrics
    trades_df = trades.copy()
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    trades_df['date'] = trades_df['timestamp'].dt.date

    # Get overall daily metrics by trading pair
    daily_metrics = []
    for symbol in trades_df['symbol'].unique():
        symbol_trades = trades_df[trades_df['symbol'] == symbol]

        for date in sorted(symbol_trades['date'].unique()):
            day_trades = symbol_trades[symbol_trades['date'] == date]

            # Calculate base and quote volumes
            base_volume = day_trades['amount'].sum()
            quote_volume = (day_trades['amount'] * day_trades['price']).sum()

            daily_metrics.append({
                'date': date,
                'symbol': symbol,
                'trades': len(day_trades),
                'base_volume': base_volume,
                'quote_volume': quote_volume
            })

    daily_metrics_df = pd.DataFrame(daily_metrics)

    # Aggregate overall metrics (all pairs combined, showing in quote)
    overall_daily = daily_metrics_df.groupby('date').agg({
        'trades': 'sum',
        'quote_volume': 'sum'
    }).reset_index()

    # Calculate total volume traded (sum of all daily volumes)
    total_volume_brl = overall_daily['quote_volume'].sum()

    # Generate current positions table rows
    position_rows = f"""
        <tr>
            <td>BRL</td>
            <td>{brl_balance:,.2f}</td>
            <td>R$ 1.00</td>
            <td>R$ {brl_value:,.2f}</td>
            <td>{brl_pct:.2f}%</td>
        </tr>
        <tr>
            <td>BTC</td>
            <td>{btc_balance:.8f}</td>
            <td>R$ {btc_price:,.2f}</td>
            <td>R$ {btc_value:,.2f}</td>
            <td>{btc_pct:.2f}%</td>
        </tr>
        <tr>
            <td>USDT</td>
            <td>{usdt_balance:,.2f}</td>
            <td>R$ {usdt_price:.4f}</td>
            <td>R$ {usdt_value:,.2f}</td>
            <td>{usdt_pct:.2f}%</td>
        </tr>
    """

    # Generate daily portfolio evolution rows (descending)
    import math
    portfolio_evolution_rows = []
    for _, row in nav_evolution.sort_values('date', ascending=False).iterrows():
        nav_change_brl = row.get('nav_change_brl', 0)
        nav_change_pct = row.get('nav_change_pct', 0)

        if math.isnan(nav_change_brl):
            change_brl_str = '-'
            change_class = ''
        else:
            change_brl_str = f"R$ {nav_change_brl:,.2f}"
            change_class = 'positive' if nav_change_brl >= 0 else 'negative'

        if math.isnan(nav_change_pct):
            change_pct_str = '-'
            pct_class = ''
        else:
            change_pct_str = f"{nav_change_pct:.2f}%"
            pct_class = 'positive' if nav_change_pct >= 0 else 'negative'

        portfolio_evolution_rows.append(f"""
            <tr>
                <td>{row['date']}</td>
                <td>R$ {row['nav_brl']:,.2f}</td>
                <td>$ {row['nav_usdt']:,.2f}</td>
                <td class="{change_class}">{change_brl_str}</td>
                <td class="{pct_class}">{change_pct_str}</td>
            </tr>
        """)

    # Generate overall bots metrics rows (descending) - Simple summary
    bots_metrics_rows = []
    for _, row in overall_daily.sort_values('date', ascending=False).iterrows():
        bots_metrics_rows.append(f"""
            <tr>
                <td>{row['date']}</td>
                <td>{row['trades']:,}</td>
                <td>R$ {row['quote_volume']:,.2f}</td>
            </tr>
        """)

    # Generate per-symbol metrics for selector with market data
    symbol_options = []
    symbol_metrics_html = {}

    for symbol in sorted(daily_metrics_df['symbol'].unique()):
        symbol_data = daily_metrics_df[daily_metrics_df['symbol'] == symbol].sort_values('date', ascending=False)
        symbol_options.append(f'<option value="{symbol}">{symbol}</option>')

        symbol_rows = []
        for _, row in symbol_data.iterrows():
            # Get market data for this symbol and date
            market_trades = '-'
            market_volume_brl = '-'
            participation_pct = '-'
            volume_share_pct = '-'

            if market_data is not None and len(market_data) > 0:
                # Convert date to same type for comparison
                market_data_copy = market_data.copy()
                if 'date' in market_data_copy.columns:
                    market_data_copy['date'] = pd.to_datetime(market_data_copy['date']).dt.date

                market_day = market_data_copy[
                    (market_data_copy['date'] == row['date']) &
                    (market_data_copy['symbol'] == symbol)
                ]

                if len(market_day) > 0:
                    market_total_trades = int(market_day['trades'].iloc[0])
                    market_total_volume = float(market_day['quote_volume'].iloc[0])

                    if market_total_trades > 0:
                        market_trades = f"{market_total_trades:,}"
                        participation_pct = f"{(row['trades'] / market_total_trades * 100):.2f}%"

                    if market_total_volume > 0:
                        market_volume_brl = f"R$ {market_total_volume:,.0f}"
                        volume_share_pct = f"{(row['quote_volume'] / market_total_volume * 100):.2f}%"

            symbol_rows.append(f"""
                <tr>
                    <td>{row['date']}</td>
                    <td>{row['trades']:,}</td>
                    <td>{market_trades}</td>
                    <td>{participation_pct}</td>
                    <td>R$ {row['quote_volume']:,.2f}</td>
                    <td>{market_volume_brl}</td>
                    <td>{volume_share_pct}</td>
                </tr>
            """)

        symbol_metrics_html[symbol] = ''.join(symbol_rows)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Executive Dashboard - Brigado v2</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.5;
            color: #eaecef;
            background: #0b0e11;
            padding: 20px;
            font-size: 14px;
        }}

        .container {{
            max-width: 1800px;
            margin: 0 auto;
        }}

        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 30px;
            background: linear-gradient(135deg, #1e2329 0%, #2b3139 100%);
            border-radius: 8px;
            border: 1px solid #2b3139;
        }}

        .header h1 {{
            font-size: 2.5em;
            color: #f0b90b;
            margin-bottom: 10px;
            text-shadow: 0 0 20px rgba(240, 185, 11, 0.3);
        }}

        .header .subtitle {{
            font-size: 1.1em;
            color: #848e9c;
        }}

        .nav-section {{
            text-align: center;
            margin-bottom: 30px;
        }}

        .nav-button {{
            display: inline-block;
            background: #2b3139;
            color: #f0b90b;
            padding: 12px 30px;
            border-radius: 4px;
            text-decoration: none;
            margin: 0 10px;
            transition: all 0.3s;
            border: 1px solid #2b3139;
        }}

        .nav-button:hover {{
            background: #f0b90b;
            color: #0b0e11;
            transform: translateY(-2px);
        }}

        .nav-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .metric-card {{
            background: #1e2329;
            padding: 25px;
            border-radius: 8px;
            border: 1px solid #2b3139;
            text-align: center;
        }}

        .metric-card h3 {{
            font-size: 0.85em;
            color: #848e9c;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .metric-card .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #f0b90b;
        }}

        .metric-card .subvalue {{
            font-size: 1em;
            color: #26a69a;
            margin-top: 5px;
        }}

        .two-col-layout {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}

        .section {{
            background: #1e2329;
            padding: 25px;
            border-radius: 8px;
            border: 1px solid #2b3139;
            margin-bottom: 20px;
        }}

        .section h2 {{
            color: #f0b90b;
            margin-bottom: 15px;
            font-size: 1.1em;
            border-bottom: 2px solid #2b3139;
            padding-bottom: 8px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}

        th, td {{
            padding: 8px 10px;
            text-align: left;
            border-bottom: 1px solid #2b3139;
            font-size: 0.95em;
        }}

        th {{
            background: #2b3139;
            color: #848e9c;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75em;
            letter-spacing: 0.5px;
        }}

        td {{
            color: #eaecef;
        }}

        tr:hover td {{
            background: #2b3139;
        }}

        .positive {{
            color: #0ecb81;
            font-weight: 600;
        }}

        .negative {{
            color: #f6465d;
            font-weight: 600;
        }}

        select {{
            background: #2b3139;
            color: #eaecef;
            border: 1px solid #2b3139;
            padding: 10px 15px;
            border-radius: 4px;
            font-size: 1em;
            cursor: pointer;
            margin-bottom: 15px;
            width: 100%;
        }}

        select:hover {{
            border-color: #f0b90b;
        }}

        .footer {{
            text-align: center;
            color: #848e9c;
            margin-top: 30px;
            padding: 20px;
            font-size: 0.9em;
        }}

        @media (max-width: 1200px) {{
            .two-col-layout {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Executive Dashboard</h1>
            <div class="subtitle">Real-time Portfolio & Trading Performance Overview</div>
        </div>

        <div class="nav-section">
            <a href="reports_index.html" class="nav-button">📋 All Reports</a>
            <a href="portfolio_evolution.html" class="nav-button">💰 Portfolio Details</a>
            <a href="market_analysis_report.html" class="nav-button">📈 Market Analysis</a>
        </div>

        <!-- NAV Metrics -->
        <div class="nav-metrics">
            <div class="metric-card">
                <h3>Total NAV (BRL)</h3>
                <div class="value">R$ {current_nav_brl:,.2f}</div>
            </div>
            <div class="metric-card">
                <h3>Total NAV (USD)</h3>
                <div class="value">$ {current_nav_usdt:,.2f}</div>
                <div class="subvalue">USDT/BRL: {usdt_price:.4f}</div>
            </div>
            <div class="metric-card">
                <h3>Total Volume Traded</h3>
                <div class="value">R$ {total_volume_brl:,.0f}</div>
                <div class="subvalue">All trading pairs</div>
            </div>
            <div class="metric-card">
                <h3>Latest Update</h3>
                <div class="value">{latest_date}</div>
            </div>
        </div>

        <!-- Two Column Layout -->
        <div class="two-col-layout">
            <!-- Left Column: Portfolio -->
            <div>
                <div class="section">
                    <h2>Current Positions</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Asset</th>
                                <th>Balance</th>
                                <th>Price</th>
                                <th>Value (BRL)</th>
                                <th>% Portfolio</th>
                            </tr>
                        </thead>
                        <tbody>
                            {position_rows}
                        </tbody>
                    </table>
                </div>

                <div class="section">
                    <h2>Daily Portfolio Evolution</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>NAV (BRL)</th>
                                <th>NAV (USDT)</th>
                                <th>Change</th>
                                <th>% Change</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(portfolio_evolution_rows)}
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Right Column: Trading -->
            <div>
                <div class="section">
                    <h2>Overall Bots Performance</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Bot Trades</th>
                                <th>Bot Volume (BRL)</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(bots_metrics_rows)}
                        </tbody>
                    </table>
                </div>

                <div class="section">
                    <h2>Trading Pair Details</h2>
                    <select id="pairSelector" onchange="showPairMetrics()">
                        <option value="">Select a trading pair...</option>
                        {''.join(symbol_options)}
                    </select>

                    <div id="pairMetrics" style="display: none;">
                        <table>
                            <thead>
                                <tr>
                                    <th>Date</th>
                                    <th>Bot Trades</th>
                                    <th>Market Trades</th>
                                    <th>% Participation</th>
                                    <th>Bot Volume (BRL)</th>
                                    <th>Market Volume (BRL)</th>
                                    <th>Market Share</th>
                                </tr>
                            </thead>
                            <tbody id="pairMetricsBody">
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
            <p>Brigado v2 Executive Dashboard</p>
        </div>
    </div>

    <script>
        // Store pair metrics data
        const pairMetricsData = {{
            {', '.join([f'"{symbol}": `{html}`' for symbol, html in symbol_metrics_html.items()])}
        }};

        function showPairMetrics() {{
            const selector = document.getElementById('pairSelector');
            const selectedPair = selector.value;
            const metricsDiv = document.getElementById('pairMetrics');
            const metricsBody = document.getElementById('pairMetricsBody');

            if (selectedPair && pairMetricsData[selectedPair]) {{
                metricsBody.innerHTML = pairMetricsData[selectedPair];
                metricsDiv.style.display = 'block';
            }} else {{
                metricsDiv.style.display = 'none';
            }}
        }}
    </script>
</body>
</html>
"""

    return html_content


def main():
    """Main execution."""
    start_time = datetime.now()

    print_header("EXECUTIVE DASHBOARD GENERATOR")

    try:
        # Initialize file manager
        file_manager = FileManager(server_name='brigado')
        data_dir = file_manager.base_path
        reports_dir = file_manager.reports_dir

        # Load portfolio evolution data
        print("\n📊 Loading portfolio data...")
        token_states_path = data_dir / "brigado" / "token_states_consolidated.parquet"
        trades_path = data_dir / "brigado" / "consolidated_trades.parquet"

        if not token_states_path.exists() or not trades_path.exists():
            print("❌ Required data files not found. Please run portfolio evolution first.")
            return 1

        # Calculate NAV evolution
        # Import the function from the module
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "portfolio_evolution",
            Path(__file__).parent / "04_portfolio_evolution.py"
        )
        portfolio_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(portfolio_module)

        token_states = pd.read_parquet(token_states_path)
        trades = pd.read_parquet(trades_path)

        nav_evolution = portfolio_module.calculate_nav_from_snapshots(token_states, trades)
        print_metric("NAV days tracked", f"{len(nav_evolution):,}")

        # Load market data if available
        market_data = None
        market_data_path = data_dir / "brigado" / "market_analysis_data.parquet"
        if market_data_path.exists():
            market_data = pd.read_parquet(market_data_path)
            print_metric("Market data loaded", f"{len(market_data):,} records")
        else:
            print("⚠️  Market data not found. Run market analysis first.")

        # Generate dashboard
        print("\n🎨 Generating executive dashboard...")
        html_content = generate_executive_dashboard(nav_evolution, trades, market_data)

        # Save report
        output_path = reports_dir / "executive_dashboard.html"
        output_path.write_text(html_content)
        print_metric("Report saved", str(output_path))

        # Update index
        from research_notebooks.brigado_v2.modules.html_generator import generate_index_html
        generate_index_html(reports_dir)

        # Summary
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\n{'='*80}")
        print(f"✅ Executive Dashboard completed in {duration:.2f}s")
        print(f"{'='*80}\n")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
