#!/usr/bin/env python3
"""
Portfolio Evolution Report

Generates a daily portfolio evolution dashboard with:
- Initial state from token_states snapshots (T0)
- Event sourcing through trades chronologically
- Reconciliation with actual portfolio snapshots
- Weighted Average Cost (WAC) calculations
- Realized and Unrealized PnL tracking
- NAV (Net Asset Value) evolution with delta tracking
- Portfolio composition charts
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from research_notebooks.brigado_v2.modules.portfolio_reconciler import PortfolioReconciler

# Column mapping for consolidated_trades.parquet
COLUMN_MAP = {
    'timestamp': 'timestamp',
    'symbol': 'symbol',
    'side': 'trade_type',  # BUY or SELL
    'price': 'price',
    'amount': 'amount',
    'fee': 'trade_fee_in_quote',
    'base_asset': 'base_asset',
    'quote_asset': 'quote_asset'
}


def print_header(title: str):
    """Print formatted header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_metric(label: str, value: str, indent: int = 2):
    """Print formatted metric."""
    spaces = " " * indent
    print(f"{spaces}{label:<40} {value:>35}")


def load_trades(data_dir: Path) -> pd.DataFrame:
    """Load consolidated trades."""
    trades_path = data_dir / "brigado" / "consolidated_trades.parquet"

    if not trades_path.exists():
        raise FileNotFoundError(f"Trades file not found: {trades_path}")

    df = pd.read_parquet(trades_path)

    # Rename columns according to mapping
    df = df.rename(columns={
        COLUMN_MAP['side']: 'side',
        COLUMN_MAP['fee']: 'fee'
    })

    # Sort by timestamp
    df = df.sort_values('timestamp')

    # Add date column
    df['date'] = df['timestamp'].dt.date

    return df


def calculate_wac_and_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Weighted Average Cost (WAC), Realized PnL, and inventory evolution.

    Logic:
    - BUY: Adds to inventory, updates WAC
    - SELL: Reduces inventory at current WAC, generates Realized PnL
    """

    # Group by symbol for separate tracking
    results = []

    for symbol in trades['symbol'].unique():
        symbol_trades = trades[trades['symbol'] == symbol].copy()

        # Initialize tracking variables
        inventory = 0.0
        wac = 0.0
        daily_records = []

        for idx, trade in symbol_trades.iterrows():
            price = trade['price']
            amount = trade['amount']
            fee = trade['fee']
            side = trade['side']
            date = trade['date']

            if side == 'BUY':
                # Add to inventory and update WAC
                total_cost = inventory * wac
                new_cost = price * amount + fee
                inventory += amount
                if inventory > 0:
                    wac = (total_cost + new_cost) / inventory

                realized_pnl = 0.0

            elif side == 'SELL':
                # Sell at current WAC generates realized PnL
                sell_proceeds = price * amount - fee
                cost_basis = wac * amount
                realized_pnl = sell_proceeds - cost_basis

                inventory -= amount
                if inventory < 1e-10:  # Close to zero
                    inventory = 0.0
                    wac = 0.0

            else:
                realized_pnl = 0.0

            daily_records.append({
                'date': date,
                'symbol': symbol,
                'base_asset': trade['base_asset'],
                'quote_asset': trade['quote_asset'],
                'timestamp': trade['timestamp'],
                'side': side,
                'price': price,
                'amount': amount,
                'fee': fee,
                'inventory': inventory,
                'wac': wac,
                'realized_pnl': realized_pnl
            })

        results.extend(daily_records)

    return pd.DataFrame(results)


def aggregate_daily_positions(wac_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate positions at end of each day."""

    # Get last record of each day per symbol
    daily_positions = (
        wac_df.groupby(['date', 'symbol'])
        .last()
        .reset_index()
    )

    # Calculate daily realized PnL
    daily_realized = (
        wac_df.groupby(['date', 'symbol'])['realized_pnl']
        .sum()
        .reset_index()
        .rename(columns={'realized_pnl': 'daily_realized_pnl'})
    )

    daily_positions = daily_positions.merge(
        daily_realized,
        on=['date', 'symbol'],
        how='left'
    )

    return daily_positions


def fetch_current_prices() -> dict:
    """
    Fetch current market prices for unrealized PnL calculation.
    For now, use last trade price from the data.
    """
    # TODO: In production, fetch from exchange API
    # For now, return placeholder
    return {
        'BTC-BRL': 0.0,
        'USDT-BRL': 0.0
    }


def calculate_unrealized_pnl(daily_positions: pd.DataFrame, current_prices: dict) -> pd.DataFrame:
    """Calculate unrealized PnL for each position."""

    def calc_unrealized(row):
        if row['inventory'] > 0 and row['symbol'] in current_prices:
            current_price = current_prices.get(row['symbol'], row['price'])
            return (current_price - row['wac']) * row['inventory']
        return 0.0

    daily_positions['unrealized_pnl'] = daily_positions.apply(calc_unrealized, axis=1)

    return daily_positions


def calculate_nav_from_snapshots(token_states: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate NAV evolution from token_states snapshots.
    Uses actual portfolio snapshots with market prices from trades.
    """

    # Prepare token states
    token_states = token_states.copy()
    token_states['timestamp'] = pd.to_datetime(token_states['timestamp']).dt.tz_localize(None)
    token_states['date'] = token_states['timestamp'].dt.date

    # Get latest snapshot per day
    daily_snapshots = (
        token_states.sort_values('timestamp')
        .groupby(['date', 'token'])
        .last()
        .reset_index()
    )

    # Get market prices per day (last trade of day for each pair)
    trades_df = trades.copy()
    trades_df['date'] = pd.to_datetime(trades_df['timestamp']).dt.date

    daily_prices = {}
    for symbol in ['BTC-BRL', 'USDT-BRL']:
        symbol_trades = trades_df[trades_df['symbol'] == symbol]
        if len(symbol_trades) > 0:
            prices = (
                symbol_trades.sort_values('timestamp')
                .groupby('date')['price']
                .last()
                .to_dict()
            )
            daily_prices[symbol] = prices

    # Calculate NAV for each day
    nav_records = []
    for date in sorted(daily_snapshots['date'].unique()):
        day_snapshot = daily_snapshots[daily_snapshots['date'] == date]

        # Get balances
        balances = {row['token']: float(row['units']) for _, row in day_snapshot.iterrows()}

        btc_balance = balances.get('BTC', 0)
        usdt_balance = balances.get('USDT', 0)
        brl_balance = balances.get('BRL', 0)
        bnb_balance = balances.get('BNB', 0)

        # Get prices for this day
        btc_price = daily_prices.get('BTC-BRL', {}).get(date, 0)
        usdt_price = daily_prices.get('USDT-BRL', {}).get(date, 1.0)

        # Calculate values in BRL
        btc_value = btc_balance * btc_price
        usdt_value = usdt_balance * usdt_price
        brl_value = brl_balance
        bnb_value = 0  # BNB is negligible

        total_nav_brl = btc_value + usdt_value + brl_value + bnb_value

        # Calculate NAV in USDT (for USD basis comparison)
        total_nav_usdt = total_nav_brl / usdt_price if usdt_price > 0 else 0

        nav_records.append({
            'date': date,
            'btc_balance': btc_balance,
            'usdt_balance': usdt_balance,
            'brl_balance': brl_balance,
            'bnb_balance': bnb_balance,
            'btc_price': btc_price,
            'usdt_price': usdt_price,
            'btc_value': btc_value,
            'usdt_value': usdt_value,
            'brl_value': brl_value,
            'nav_brl': total_nav_brl,
            'nav_usdt': total_nav_usdt
        })

    nav_df = pd.DataFrame(nav_records)

    # Calculate daily changes in BRL
    if len(nav_df) > 0:
        nav_df['nav_change_brl'] = nav_df['nav_brl'].diff()
        nav_df['nav_change_pct'] = nav_df['nav_brl'].pct_change() * 100

        # Calculate daily changes in USDT
        nav_df['nav_change_usdt'] = nav_df['nav_usdt'].diff()

    return nav_df


def calculate_nav_evolution(daily_positions: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Net Asset Value (NAV) evolution.
    NAV = BRL balance + BTC position value + USDT position value

    BRL balance = cumulative cash flow from all trades
    """

    # Get all unique dates
    all_dates = sorted(daily_positions['date'].unique())

    # Calculate cumulative BRL cash flow
    # When we BUY: BRL decreases (negative cash flow)
    # When we SELL: BRL increases (positive cash flow)
    trades_with_cash = trades.copy()
    trades_with_cash['brl_flow'] = trades_with_cash.apply(
        lambda x: (x['price'] * x['amount'] + x['fee']) * (-1 if x['side'] == 'BUY' else 1),
        axis=1
    )

    nav_records = []

    for date in all_dates:
        day_positions = daily_positions[daily_positions['date'] == date]

        # Get cumulative BRL flow up to this date
        trades_up_to_date = trades_with_cash[trades_with_cash['date'] <= date]
        brl_balance = trades_up_to_date['brl_flow'].sum()

        # Calculate total value in BRL (quote asset)
        btc_position = day_positions[day_positions['symbol'] == 'BTC-BRL']
        usdt_position = day_positions[day_positions['symbol'] == 'USDT-BRL']

        btc_value = 0.0
        usdt_value = 0.0
        btc_inventory = 0.0
        usdt_inventory = 0.0
        btc_price = 0.0
        usdt_price = 0.0

        if len(btc_position) > 0:
            btc_row = btc_position.iloc[-1]
            btc_inventory = btc_row['inventory']
            btc_price = btc_row['price']
            btc_value = btc_price * btc_inventory

        if len(usdt_position) > 0:
            usdt_row = usdt_position.iloc[-1]
            usdt_inventory = usdt_row['inventory']
            usdt_price = usdt_row['price']
            usdt_value = usdt_price * usdt_inventory

        # Sum realized PnL for the day only
        day_trades = trades_up_to_date[trades_up_to_date['date'] == date]
        daily_realized_pnl = day_positions['daily_realized_pnl'].sum()

        # Sum unrealized PnL
        daily_unrealized_pnl = day_positions['unrealized_pnl'].sum()

        # NAV = BRL balance + crypto position values
        nav = brl_balance + btc_value + usdt_value

        nav_records.append({
            'date': date,
            'brl_balance': brl_balance,
            'btc_inventory': btc_inventory,
            'btc_price': btc_price,
            'btc_value_brl': btc_value,
            'usdt_inventory': usdt_inventory,
            'usdt_price': usdt_price,
            'usdt_value_brl': usdt_value,
            'nav_brl': nav,
            'realized_pnl': daily_realized_pnl,
            'unrealized_pnl': daily_unrealized_pnl
        })

    return pd.DataFrame(nav_records)


def generate_html_report(
    daily_positions: pd.DataFrame,
    nav_evolution: pd.DataFrame,
    output_path: Path,
    current_prices: dict
):
    """Generate HTML dashboard with Brigado v2 styling."""

    # Get latest positions
    latest_date = daily_positions['date'].max()
    latest_positions = daily_positions[daily_positions['date'] == latest_date]

    # Get current NAV
    latest_nav = nav_evolution[nav_evolution['date'] == latest_date].iloc[0]
    current_nav = latest_nav['nav_brl']
    brl_balance = latest_nav['brl_balance']

    # Calculate position values and percentages
    brl_value = brl_balance
    brl_pct = (brl_value / current_nav * 100) if current_nav > 0 else 0

    # Prepare position table rows - start with BRL
    position_rows = [f"""
        <tr>
            <td>BRL</td>
            <td>{brl_balance:.2f}</td>
            <td>1.00</td>
            <td>{brl_value:.2f}</td>
            <td>{brl_pct:.2f}%</td>
        </tr>
    """]

    for _, pos in latest_positions.iterrows():
        pos_value = pos['inventory'] * pos['price']
        pos_pct = (pos_value / current_nav * 100) if current_nav > 0 else 0

        position_rows.append(f"""
            <tr>
                <td>{pos['base_asset']}</td>
                <td>{pos['inventory']:.8f}</td>
                <td>{pos['price']:.2f}</td>
                <td>{pos_value:.2f}</td>
                <td>{pos_pct:.2f}%</td>
            </tr>
        """)

    # Prepare daily evolution table rows (sorted descending by date)
    daily_evolution_rows = []
    for _, row in nav_evolution.sort_values('date', ascending=False).iterrows():
        nav_change_brl = row.get('nav_change_brl', 0)
        nav_change_pct = row.get('nav_change_pct', 0)

        # Format change columns - show "-" for NaN/first row
        import math
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

        daily_evolution_rows.append(f"""
            <tr>
                <td>{row['date']}</td>
                <td>{row['btc_balance']:.8f}</td>
                <td>{row['usdt_balance']:.2f}</td>
                <td>{row['brl_balance']:.2f}</td>
                <td>R$ {row['btc_price']:,.2f}</td>
                <td>R$ {row['usdt_price']:.4f}</td>
                <td>R$ {row['nav_brl']:,.2f}</td>
                <td>$ {row['nav_usdt']:,.2f}</td>
                <td class="{change_class}">{change_brl_str}</td>
                <td class="{pct_class}">{change_pct_str}</td>
            </tr>
        """)

    # Prepare chart data
    dates = [str(d) for d in nav_evolution['date']]
    nav_values = nav_evolution['nav_brl'].tolist()
    realized_pnl = nav_evolution.get('realized_pnl', [0] * len(nav_evolution)).tolist() if 'realized_pnl' in nav_evolution.columns else [0] * len(nav_evolution)
    unrealized_pnl = nav_evolution.get('unrealized_pnl', [0] * len(nav_evolution)).tolist() if 'unrealized_pnl' in nav_evolution.columns else [0] * len(nav_evolution)
    brl_values = nav_evolution['brl_value'].tolist()
    btc_values = nav_evolution['btc_value'].tolist()
    usdt_values = nav_evolution['usdt_value'].tolist()

    # Calculate percentages for composition
    brl_pct = [(abs(brl) / nav * 100) if nav > 0 else 0 for brl, nav in zip(brl_values, nav_values)]
    btc_pct = [(btc / nav * 100) if nav > 0 else 0 for btc, nav in zip(btc_values, nav_values)]
    usdt_pct = [(usdt / nav * 100) if nav > 0 else 0 for usdt, nav in zip(usdt_values, nav_values)]

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Portfolio Evolution - Brigado v2</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #eaecef;
            background: #0b0e11;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding: 30px;
            background: linear-gradient(135deg, #1e2329 0%, #0b0e11 100%);
            border-radius: 8px;
            border: 1px solid #2b3139;
        }}

        .header h1 {{
            font-size: 2.5em;
            color: #f0b90b;
            margin-bottom: 10px;
            text-shadow: 0 0 20px rgba(240, 185, 11, 0.3);
        }}

        .header .date {{
            font-size: 1.1em;
            color: #848e9c;
            margin-bottom: 20px;
        }}

        .nav-display {{
            font-size: 2em;
            color: #0ecb81;
            font-weight: bold;
            margin-top: 15px;
        }}

        .nav-label {{
            font-size: 0.9em;
            color: #848e9c;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .section {{
            background: #1e2329;
            border-radius: 8px;
            border: 1px solid #2b3139;
            padding: 30px;
            margin-bottom: 30px;
        }}

        .section h2 {{
            font-size: 1.5em;
            color: #f0b90b;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #2b3139;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}

        th {{
            background: #2b3139;
            color: #848e9c;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 1px;
            padding: 15px;
            text-align: left;
            border-bottom: 2px solid #f0b90b;
        }}

        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #2b3139;
            color: #eaecef;
        }}

        tr:hover {{
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

        .chart-container {{
            margin: 30px 0;
            padding: 20px;
            background: #0b0e11;
            border-radius: 8px;
            min-height: 500px;
        }}

        .navigation {{
            text-align: center;
            margin: 40px 0;
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
            border-color: #f0b90b;
            transform: translateY(-2px);
        }}

        .footer {{
            text-align: center;
            color: #848e9c;
            margin-top: 50px;
            padding: 20px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="navigation">
            <a href="index.html" class="nav-button">← Back to Dashboard</a>
        </div>

        <div class="header">
            <h1>📊 Portfolio Evolution</h1>
            <div class="date">Report Date: {latest_date}</div>
            <div class="nav-label">Total Net Asset Value</div>
            <div class="nav-display">R$ {current_nav:,.2f}</div>
        </div>

        <div class="section">
            <h2>Current Positions (End of Day)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Asset</th>
                        <th>Balance</th>
                        <th>Current Price</th>
                        <th>Notional Value (BRL)</th>
                        <th>% of Portfolio</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(position_rows)}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Daily Portfolio Evolution</h2>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>BTC Balance</th>
                        <th>USDT Balance</th>
                        <th>BRL Balance</th>
                        <th>BTC Price</th>
                        <th>USDT Price</th>
                        <th>NAV (BRL)</th>
                        <th>NAV (USDT)</th>
                        <th>Daily Change</th>
                        <th>% Change</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(daily_evolution_rows)}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Equity Curve</h2>
            <div class="chart-container" id="navChart"></div>
        </div>

        <div class="section">
            <h2>Portfolio Composition</h2>
            <div class="chart-container" id="compositionChart"></div>
        </div>

        <div class="section">
            <h2>Daily PnL</h2>
            <div class="chart-container" id="pnlChart"></div>
        </div>

        <div class="footer">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
            <p>Brigado v2 Portfolio Evolution System</p>
        </div>
    </div>

    <script>
        // NAV Evolution Chart
        var navTrace = {{
            x: {dates},
            y: {nav_values},
            type: 'scatter',
            mode: 'lines+markers',
            name: 'NAV (BRL)',
            line: {{
                color: '#0ecb81',
                width: 3
            }},
            marker: {{
                size: 6,
                color: '#0ecb81'
            }}
        }};

        var navLayout = {{
            paper_bgcolor: '#0b0e11',
            plot_bgcolor: '#1e2329',
            font: {{ color: '#eaecef' }},
            xaxis: {{
                gridcolor: '#2b3139',
                title: 'Date'
            }},
            yaxis: {{
                gridcolor: '#2b3139',
                title: 'NAV (BRL)'
            }},
            height: 450,
            margin: {{ t: 20, r: 20, b: 50, l: 70 }}
        }};

        Plotly.newPlot('navChart', [navTrace], navLayout, {{responsive: true}});

        // Portfolio Composition Chart
        var brlTrace = {{
            x: {dates},
            y: {brl_pct},
            type: 'scatter',
            mode: 'lines',
            name: 'BRL %',
            stackgroup: 'one',
            fillcolor: '#0ecb81',
            line: {{ width: 0 }}
        }};

        var btcTrace = {{
            x: {dates},
            y: {btc_pct},
            type: 'scatter',
            mode: 'lines',
            name: 'BTC %',
            stackgroup: 'one',
            fillcolor: '#f0b90b',
            line: {{ width: 0 }}
        }};

        var usdtTrace = {{
            x: {dates},
            y: {usdt_pct},
            type: 'scatter',
            mode: 'lines',
            name: 'USDT %',
            stackgroup: 'one',
            fillcolor: '#26a69a',
            line: {{ width: 0 }}
        }};

        var compositionLayout = {{
            paper_bgcolor: '#0b0e11',
            plot_bgcolor: '#1e2329',
            font: {{ color: '#eaecef' }},
            xaxis: {{
                gridcolor: '#2b3139',
                title: 'Date'
            }},
            yaxis: {{
                gridcolor: '#2b3139',
                title: 'Percentage (%)',
                range: [0, 100]
            }},
            height: 450,
            margin: {{ t: 20, r: 20, b: 50, l: 70 }}
        }};

        Plotly.newPlot('compositionChart', [brlTrace, btcTrace, usdtTrace], compositionLayout, {{responsive: true}});

        // Daily PnL Chart
        var realizedTrace = {{
            x: {dates},
            y: {realized_pnl},
            type: 'bar',
            name: 'Realized PnL',
            marker: {{
                color: {realized_pnl}.map(v => v >= 0 ? '#0ecb81' : '#f6465d')
            }}
        }};

        var unrealizedTrace = {{
            x: {dates},
            y: {unrealized_pnl},
            type: 'bar',
            name: 'Unrealized PnL',
            marker: {{
                color: {unrealized_pnl}.map(v => v >= 0 ? '#0ecb8180' : '#f6465d80')
            }}
        }};

        var pnlLayout = {{
            paper_bgcolor: '#0b0e11',
            plot_bgcolor: '#1e2329',
            font: {{ color: '#eaecef' }},
            barmode: 'group',
            xaxis: {{
                gridcolor: '#2b3139',
                title: 'Date'
            }},
            yaxis: {{
                gridcolor: '#2b3139',
                title: 'PnL (BRL)',
                zeroline: true,
                zerolinecolor: '#848e9c'
            }},
            height: 450,
            margin: {{ t: 20, r: 20, b: 50, l: 70 }}
        }};

        Plotly.newPlot('pnlChart', [realizedTrace, unrealizedTrace], pnlLayout, {{responsive: true}});
    </script>
</body>
</html>"""

    output_path.write_text(html_content)


def main():
    """Main execution."""
    print_header("PORTFOLIO EVOLUTION REPORT GENERATOR")

    start_time = datetime.now()

    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    reports_dir = script_dir.parent / "reports"
    reports_dir.mkdir(exist_ok=True)

    try:
        # Step 1: Load trades
        print("📥 Loading consolidated trades...")
        trades = load_trades(data_dir)
        print_metric("Total trades loaded", f"{len(trades):,}")
        print_metric("Date range", f"{trades['date'].min()} to {trades['date'].max()}")
        print_metric("Symbols", f"{', '.join(trades['symbol'].unique())}")

        # Step 1b: Load token_states
        print("\n📊 Loading token states...")
        token_states_path = data_dir / "brigado" / "token_states_consolidated.parquet"

        if not token_states_path.exists():
            print(f"⚠️  Token states not found: {token_states_path}")
            print("   Running without reconciliation...")
            token_states = None
        else:
            token_states = pd.read_parquet(token_states_path)
            print_metric("Snapshots loaded", f"{len(token_states):,}")
            print_metric("Snapshot range", f"{token_states['timestamp'].min()} to {token_states['timestamp'].max()}")

        # Step 2: Reconciliation (if token_states available)
        if token_states is not None:
            print("\n🔄 Performing portfolio reconciliation...")
            # Load trades again without column renaming for reconciler
            trades_for_reconciler = pd.read_parquet(data_dir / "brigado" / "consolidated_trades.parquet")
            reconciler = PortfolioReconciler(trades_for_reconciler, token_states)

            # Get initial state
            initial_state = reconciler.get_initial_state()
            print_metric("Initial state timestamp", str(initial_state['t0_timestamp']))

            # Calculate balance evolution
            balance_evolution = reconciler.calculate_balance_evolution()
            print_metric("Balance records", f"{len(balance_evolution):,}")

            # Reconcile with snapshots
            reconciliation_deltas = reconciler.reconcile_with_snapshots()
            print_metric("Reconciliation records", f"{len(reconciliation_deltas):,}")

            # Get summary
            summary = reconciler.get_summary()
            print_metric("Total delta (BRL)", f"R$ {summary['total_delta_brl']:,.2f}")
            print_metric("Avg daily delta", f"R$ {summary['avg_daily_delta_brl']:,.2f}")

            # Store reconciler for later use
            reconciler_obj = reconciler
        else:
            reconciler_obj = None

        # Step 3: Calculate WAC and PnL (legacy method)
        print("\n💰 Calculating WAC and Realized PnL...")
        wac_df = calculate_wac_and_pnl(trades)
        print_metric("Trades processed", f"{len(wac_df):,}")

        # Step 3b: Aggregate daily positions
        print("\n📊 Aggregating daily positions...")
        daily_positions = aggregate_daily_positions(wac_df)
        print_metric("Daily position records", f"{len(daily_positions):,}")

        # Step 3c: Override latest positions with actual snapshot data if available
        if reconciler_obj is not None and token_states is not None:
            print("\n🔄 Applying snapshot corrections to latest positions...")
            # Get latest snapshot
            token_states_df = token_states.copy()
            token_states_df['timestamp'] = pd.to_datetime(token_states_df['timestamp']).dt.tz_localize(None)
            latest_snapshot_time = token_states_df['timestamp'].max()
            latest_snapshot = token_states_df[token_states_df['timestamp'] == latest_snapshot_time]

            # Get latest date in daily_positions
            latest_date = daily_positions['date'].max()

            # Update balances from snapshot
            for _, snap_row in latest_snapshot.iterrows():
                asset = snap_row['token']
                actual_balance = float(snap_row['units'])

                # Find matching position in daily_positions
                mask = (daily_positions['date'] == latest_date) & (daily_positions['base_asset'] == asset)
                if mask.any():
                    # Update the inventory with actual snapshot balance
                    daily_positions.loc[mask, 'inventory'] = actual_balance
                    print_metric(f"  Updated {asset} balance", f"{actual_balance:.8f}")

            print_metric("Snapshot corrections applied", "Latest positions updated")

        # Step 4: Fetch current prices and calculate unrealized PnL
        print("\n💵 Calculating unrealized PnL...")
        current_prices = fetch_current_prices()

        # Use last traded price as current price
        for symbol in trades['symbol'].unique():
            symbol_trades = trades[trades['symbol'] == symbol]
            current_prices[symbol] = symbol_trades.iloc[-1]['price']

        daily_positions = calculate_unrealized_pnl(daily_positions, current_prices)
        print_metric("Current BTC-BRL price", f"R$ {current_prices.get('BTC-BRL', 0):,.2f}")
        print_metric("Current USDT-BRL price", f"R$ {current_prices.get('USDT-BRL', 0):,.4f}")

        # Step 5: Calculate NAV evolution from snapshots
        print("\n📈 Calculating NAV evolution from snapshots...")
        if token_states is not None:
            nav_evolution = calculate_nav_from_snapshots(token_states, trades)
            print_metric("Days tracked", f"{len(nav_evolution):,}")
            latest_nav = nav_evolution.iloc[-1]['nav_brl']
            print_metric("Latest NAV (from snapshots)", f"R$ {latest_nav:,.2f}")
        else:
            # Fallback to legacy calculation
            nav_evolution = calculate_nav_evolution(daily_positions, trades)
            print_metric("Days tracked", f"{len(nav_evolution):,}")
            latest_nav = nav_evolution.iloc[-1]['nav_brl']
            print_metric("Latest NAV (calculated)", f"R$ {latest_nav:,.2f}")

        # Step 6: Generate HTML report
        print("\n🎨 Generating HTML report...")
        output_path = reports_dir / "portfolio_evolution.html"
        generate_html_report(daily_positions, nav_evolution, output_path, current_prices)
        print_metric("Report saved", str(output_path))

        # Summary
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\n{'='*80}")
        print(f"✅ Portfolio Evolution Report completed in {duration:.2f}s")
        print(f"{'='*80}\n")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
