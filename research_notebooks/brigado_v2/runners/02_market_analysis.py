#!/usr/bin/env python3
"""
Market Analysis Runner

Analyzes controller performance vs market activity by trading pair.
Fetches market data from Binance and calculates market share.
"""

import sys
from pathlib import Path
from datetime import datetime
import asyncio
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from research_notebooks.brigado_v2.data_consolidator import DataConsolidator
from research_notebooks.brigado_v2.file_manager import FileManager
from research_notebooks.brigado_v2.html_generator import generate_index_html
from core.data_sources.clob import CLOBDataSource


def print_header(title: str):
    """Print a styled header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_step(step: str, status: str = "⏳"):
    """Print a step with status."""
    print(f"{status} {step}")


def print_success(message: str):
    """Print a success message."""
    print(f"✅ {message}")


def print_info(message: str, indent: int = 2):
    """Print an info message."""
    print(f"{' ' * indent}→ {message}")


def print_metric(label: str, value: str, indent: int = 4):
    """Print a metric."""
    print(f"{' ' * indent}{label}: {value}")


async def fetch_market_data_for_pair(clob, pair: str, start_date, end_date):
    """Fetch market data for a single trading pair - returns daily breakdown."""
    try:
        binance_pair = pair.replace('-', '')
        print_info(f"Fetching {pair} ({binance_pair})...")

        # Fetch 1-day candles for daily data
        candles = await clob.get_candles(
            connector_name='binance',
            trading_pair=binance_pair,
            interval='1d',  # Changed to 1d for daily candles
            start_time=int(start_date.timestamp()),
            end_time=int(end_date.timestamp())
        )

        if candles.data is not None and len(candles.data) > 0:
            df = candles.data

            # Ensure timestamp column
            if 'timestamp' not in df.columns and df.index.name == 'timestamp':
                df = df.reset_index()

            # Convert timestamp to date
            # Timestamps from Binance are in seconds (Unix timestamp)
            df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date

            # Calculate daily metrics
            daily_data = []
            for _, row in df.iterrows():
                daily_data.append({
                    'date': row['date'],
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'base_volume': float(row['volume']),
                    'quote_volume': float(row['quote_asset_volume']) if 'quote_asset_volume' in row else 0,
                    'n_trades': int(row['n_trades']) if 'n_trades' in row else 0,
                    'high_low_pct': ((float(row['high']) - float(row['low'])) / float(row['low']) * 100) if float(row['low']) > 0 else 0,
                })

            # Summary stats
            total_volume = sum(d['quote_volume'] for d in daily_data)
            total_trades = sum(d['n_trades'] for d in daily_data)

            print_metric("Days", f"{len(daily_data)}", indent=6)
            print_metric("Total Volume", f"{total_volume:,.0f}", indent=6)
            print_metric("Total Trades", f"{total_trades:,}", indent=6)

            return pair, daily_data  # Return list of daily dicts
        else:
            print_info("⚠️  No data available", indent=6)
            return pair, None

    except Exception as e:
        print_info(f"❌ Error: {str(e)[:50]}", indent=6)
        return pair, None


async def main_async():
    """Main async workflow."""
    start_time = datetime.now()

    print_header("📈 MARKET ANALYSIS RUNNER")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Initialize
    print_step("Initializing components...")
    consolidator = DataConsolidator()
    file_manager = FileManager()
    clob = CLOBDataSource()
    print_success("Components initialized")

    # Load consolidated data
    print_step("Loading consolidated data...")
    data = consolidator.load_consolidated_data(use_latest=True)
    print_success("Data loaded")

    print_info("Summary:")
    print_metric("Trades", f"{len(data['trades']):,}", indent=6)
    print_metric("Executors", f"{len(data['executors']):,}", indent=6)
    print_metric("Controllers", f"{len(data['controllers']):,}", indent=6)

    # Create trade-to-controller mapping
    print_step("Creating trade-to-controller mapping...")
    order_to_controller = {}

    if 'executors' in data and len(data['executors']) > 0:
        executors_df = data['executors']
        executors_filtered = executors_df[executors_df['net_pnl_quote'] != 0]

        for _, executor in executors_filtered.iterrows():
            controller_id = executor.get('controller_id')
            if controller_id and 'custom_info_parsed' in executor and isinstance(executor['custom_info_parsed'], dict):
                order_ids = executor['custom_info_parsed'].get('order_ids', [])
                if isinstance(order_ids, np.ndarray):
                    order_ids = order_ids.tolist()
                elif not isinstance(order_ids, (list, tuple)):
                    order_ids = list(order_ids) if order_ids else []
                for oid in order_ids:
                    if oid:
                        order_to_controller[str(oid)] = controller_id

    trades = data['trades'].copy()
    trades['controller_id'] = trades['order_id'].map(order_to_controller)
    trades['quote_volume'] = trades['amount'] * trades['price']

    coverage = (trades['controller_id'].notna().sum() / len(trades)) * 100
    print_success(f"Mapped {len(order_to_controller):,} orders ({coverage:.1f}% coverage)")

    # Get trading pairs and date range
    trading_pairs = trades['symbol'].unique()
    first_trade = trades['timestamp'].min()
    last_trade = trades['timestamp'].max()

    # Expand to full days (00:00:00 to 23:59:59) to get complete market data
    start_date = pd.Timestamp(first_trade.date())  # Beginning of first day
    end_date = pd.Timestamp(last_trade.date()) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # End of last day

    print_step(f"Fetching market data for {len(trading_pairs)} pairs...")
    print_info(f"Date range: {start_date.date()} to {end_date.date()} (full days)")
    print_info(f"Trading pairs: {', '.join(trading_pairs)}")

    # Fetch market data for all pairs
    market_data = {}
    for pair in trading_pairs:
        pair_str, data_dict = await fetch_market_data_for_pair(clob, pair, start_date, end_date)
        market_data[pair_str] = data_dict

    successful = len([m for m in market_data.values() if m is not None])
    print_success(f"Market data fetched: {successful}/{len(trading_pairs)} pairs")

    # Calculate market share
    print_step("Calculating controller market share...")
    market_share_data = {}

    for pair in trading_pairs:
        pair_trades = trades[trades['symbol'] == pair]

        # Calculate controller volumes
        controller_volumes = {}
        for controller_id in pair_trades['controller_id'].dropna().unique():
            ctrl_trades = pair_trades[pair_trades['controller_id'] == controller_id]
            bot_name = ctrl_trades['source_bot'].iloc[0] if len(ctrl_trades) > 0 else 'Unknown'

            # Calculate time since first trade
            first_trade_time = ctrl_trades['timestamp'].min()
            last_trade_time = ctrl_trades['timestamp'].max()
            time_elapsed = last_trade_time - first_trade_time

            controller_volumes[controller_id] = {
                'bot_name': bot_name,
                'trades': len(ctrl_trades),
                'base_volume': float(ctrl_trades['amount'].sum()),
                'quote_volume': float(ctrl_trades['quote_volume'].sum()),
                'first_trade_time': first_trade_time,
                'time_elapsed': time_elapsed
            }

        # Total bot volume for this pair
        total_bot_base_volume = float(pair_trades['amount'].sum())
        total_bot_quote_volume = float(pair_trades['quote_volume'].sum())

        # Market share calculations
        if market_data.get(pair) and market_data[pair]:
            # Aggregate daily volumes into period totals
            daily_data_list = market_data[pair]
            market_base_volume = sum(d['base_volume'] for d in daily_data_list)
            market_quote_volume = sum(d['quote_volume'] for d in daily_data_list)

            overall_base_share = (total_bot_base_volume / market_base_volume) * 100 if market_base_volume > 0 else 0
            overall_quote_share = (total_bot_quote_volume / market_quote_volume) * 100 if market_quote_volume > 0 else 0

            # Calculate per-controller market share
            for controller_id, vol_data in controller_volumes.items():
                vol_data['base_market_share'] = (vol_data['base_volume'] / market_base_volume) * 100 if market_base_volume > 0 else 0
                vol_data['quote_market_share'] = (vol_data['quote_volume'] / market_quote_volume) * 100 if market_quote_volume > 0 else 0
        else:
            overall_base_share = 0
            overall_quote_share = 0
            for vol_data in controller_volumes.values():
                vol_data['base_market_share'] = 0
                vol_data['quote_market_share'] = 0

        # Aggregate daily data for HTML report
        aggregated_market_data = None
        if market_data.get(pair) and market_data[pair]:
            daily_data_list = market_data[pair]
            aggregated_market_data = {
                'base_volume': sum(d['base_volume'] for d in daily_data_list),
                'quote_volume': sum(d['quote_volume'] for d in daily_data_list),
                'n_trades': sum(d['n_trades'] for d in daily_data_list),
                'open': daily_data_list[0]['open'],  # First day's open
                'high': max(d['high'] for d in daily_data_list),  # Period high
                'low': min(d['low'] for d in daily_data_list),  # Period low
                'close': daily_data_list[-1]['close'],  # Last day's close
                'high_low_pct': ((max(d['high'] for d in daily_data_list) - min(d['low'] for d in daily_data_list)) / min(d['low'] for d in daily_data_list) * 100)
            }

        market_share_data[pair] = {
            'market_data': aggregated_market_data,
            'total_bot_base_volume': total_bot_base_volume,
            'total_bot_quote_volume': total_bot_quote_volume,
            'total_bot_trades': len(pair_trades),
            'overall_base_market_share': overall_base_share,
            'overall_quote_market_share': overall_quote_share,
            'controllers': controller_volumes
        }

        print_info(f"{pair}:")
        print_metric("Bot volume", f"{total_bot_quote_volume:,.0f} quote", indent=6)
        print_metric("Market share", f"{overall_quote_share:.4f}%", indent=6)
        print_metric("Controllers", f"{len(controller_volumes)}", indent=6)

    print_success("Market share calculated")

    # Save market data to parquet for evolutive report
    print_step("Saving market data for evolutive report...")
    market_df_records = []
    for pair in trading_pairs:
        if pair in market_data and market_data[pair]:
            daily_data_list = market_data[pair]  # Now a list of daily dicts
            for daily_dict in daily_data_list:
                market_df_records.append({
                    'date': daily_dict['date'],
                    'symbol': pair,
                    'base_volume': daily_dict['base_volume'],
                    'quote_volume': daily_dict['quote_volume'],
                    'trades': daily_dict['n_trades'],
                    'open': daily_dict['open'],
                    'high': daily_dict['high'],
                    'low': daily_dict['low'],
                    'close': daily_dict['close'],
                    'volatility': daily_dict['high_low_pct']
                })

    if market_df_records:
        market_df = pd.DataFrame(market_df_records)
        market_data_path = file_manager.data_sources_dir / "market_analysis_data.parquet"
        market_df.to_parquet(market_data_path, index=False)
        print_success(f"Market data saved: {market_data_path.name}")
    else:
        print_info("⚠️  No market data to save", indent=2)

    # Generate HTML report
    print_step("Generating market analysis report...")

    html_content = generate_market_html(market_share_data, start_date, end_date)
    html_path = file_manager.data_sources_dir / "market_analysis_report.html"

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print_success(f"Report generated: {html_path.name}")

    # Update index
    print_step("Updating index page...")
    metadata = consolidator.get_consolidation_info()
    index_path = generate_index_html(file_manager.data_sources_dir, metadata=metadata)
    print_success(f"Index updated: {index_path.name}")

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print_header("✨ MARKET ANALYSIS COMPLETE")
    print(f"Duration: {duration:.2f}s")
    print(f"Output directory: {file_manager.data_sources_dir}")
    print(f"\n💡 Open {index_path} in your browser to view reports\n")

    return 0


def generate_market_html(market_share_data, start_date, end_date):
    """Generate HTML content for market analysis report."""
    from datetime import datetime

    def format_timedelta(td):
        """Format timedelta as XdXhXm"""
        total_seconds = int(td.total_seconds())
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0 or not parts:
            parts.append(f"{minutes}m")
        return " ".join(parts)

    symbols = sorted(market_share_data.keys())

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Market Performance Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #eaecef; background: #0b0e11; padding: 20px; }}
        .container {{ max-width: 1600px; margin: 0 auto; background: #1e2329; border-radius: 8px; border: 1px solid #2b3139; overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #2b3139 0%, #1e2329 100%); border-bottom: 3px solid #f0b90b; color: #f0b90b; padding: 40px; text-align: center; position: relative; }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; font-weight: 600; text-shadow: 0 0 20px rgba(240, 185, 11, 0.3); }}
        .header p {{ font-size: 1.1em; color: #848e9c; }}
        .nav-link {{ position: absolute; top: 20px; left: 20px; background: #2b3139; color: #f0b90b; padding: 10px 20px; border-radius: 4px; text-decoration: none; border: 1px solid #f0b90b; transition: all 0.3s; font-weight: 600; }}
        .nav-link:hover {{ background: #f0b90b; color: #0b0e11; }}
        .content {{ padding: 40px; }}
        .tabs {{ display: flex; gap: 10px; margin-bottom: 30px; border-bottom: 2px solid #3d4551; padding-bottom: 10px; }}
        .tab {{ background: #2b3139; color: #848e9c; padding: 12px 24px; border-radius: 4px 4px 0 0; cursor: pointer; border: 1px solid #3d4551; border-bottom: none; transition: all 0.3s; font-weight: 600; }}
        .tab:hover {{ background: #3d4551; color: #f0b90b; }}
        .tab.active {{ background: #f0b90b; color: #0b0e11; border-color: #f0b90b; }}
        .tab-content {{ display: none; animation: fadeIn 0.3s; }}
        .tab-content.active {{ display: block; }}
        @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
        .pair-section {{ background: #2b3139; border-radius: 8px; padding: 30px; margin-bottom: 30px; border-left: 5px solid #f0b90b; }}
        .pair-header {{ font-size: 2em; color: #f0b90b; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between; }}
        .market-share-badge {{ background: #f0b90b; color: #0b0e11; padding: 8px 16px; border-radius: 4px; font-size: 0.5em; font-weight: 600; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 25px; }}
        .metric-card {{ background: #1e2329; padding: 20px; border-radius: 8px; border: 1px solid #3d4551; }}
        .metric-card h4 {{ font-size: 0.75em; color: #848e9c; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }}
        .metric-card .value {{ font-size: 1.5em; font-weight: bold; color: #f0b90b; }}
        .metric-card.highlight {{ background: #3d4551; border: 1px solid #f0b90b; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: #2b3139; border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #3d4551; }}
        th {{ background: #1e2329; color: #f0b90b; font-weight: 600; text-transform: uppercase; font-size: 0.75em; }}
        tr:hover {{ background: #3d4551; }}
        .bot-name {{ font-size: 0.8em; color: #848e9c; font-style: italic; }}
        .time-elapsed {{ font-size: 0.7em; color: #0ecb81; margin-left: 8px; }}
        .share-high {{ color: #0ecb81; font-weight: 600; }}
        .share-medium {{ color: #f0b90b; font-weight: 600; }}
        .share-low {{ color: #848e9c; }}
        .footer {{ background: #1e2329; border-top: 1px solid #2b3139; padding: 20px; text-align: center; color: #848e9c; font-size: 0.9em; }}
    </style>
    <script>
        function switchTab(symbol) {{
            // Hide all tab contents
            var tabContents = document.getElementsByClassName('tab-content');
            for (var i = 0; i < tabContents.length; i++) {{
                tabContents[i].classList.remove('active');
            }}

            // Remove active class from all tabs
            var tabs = document.getElementsByClassName('tab');
            for (var i = 0; i < tabs.length; i++) {{
                tabs[i].classList.remove('active');
            }}

            // Show selected tab content
            document.getElementById('tab-' + symbol).classList.add('active');

            // Add active class to clicked tab
            event.target.classList.add('active');
        }}
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <a href="index.html" class="nav-link">← Back to Index</a>
            <h1>📊 Market Performance Analysis</h1>
            <p>Controller Performance vs Market Activity by Trading Pair</p>
            <p style="margin-top: 10px;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        <div class="content">
            <div class="tabs">
"""

    # Add tab buttons
    for idx, symbol in enumerate(symbols):
        active_class = ' active' if idx == 0 else ''
        html_content += f'                <div class="tab{active_class}" onclick="switchTab(\'{symbol}\')">{symbol}</div>\n'

    html_content += "            </div>\n\n"

    # Add tab contents
    for idx, pair in enumerate(symbols):
        data_point = market_share_data[pair]
        mkt_data = data_point['market_data']
        active_class = ' active' if idx == 0 else ''

        html_content += f"""
            <div id="tab-{pair}" class="tab-content{active_class}">
                <div class="pair-section">
                    <div class="pair-header">
                        <span>{pair}</span>
                        <span class="market-share-badge">
                            Market Share: {data_point['overall_base_market_share']:.4f}%
                        </span>
                    </div>
"""

        if mkt_data:
            html_content += f"""
                <h3 style="margin-bottom: 15px; color: #848e9c;">Market Data ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})</h3>
                <div class="metrics-grid" style="grid-template-columns: repeat(3, 1fr);">
                    <div class="metric-card highlight">
                        <h4>Base Volume</h4>
                        <div class="value">{mkt_data['base_volume']:,.2f}</div>
                    </div>
                    <div class="metric-card highlight">
                        <h4>Quote Volume</h4>
                        <div class="value">{mkt_data['quote_volume']:,.0f}</div>
                    </div>
                    <div class="metric-card highlight">
                        <h4>Trades</h4>
                        <div class="value">{mkt_data['n_trades']:,}</div>
                    </div>
                </div>
                <div class="metrics-grid" style="grid-template-columns: repeat(5, 1fr); margin-top: 15px;">
                    <div class="metric-card">
                        <h4>Open</h4>
                        <div class="value">{mkt_data['open']:,.2f}</div>
                    </div>
                    <div class="metric-card">
                        <h4>High</h4>
                        <div class="value">{mkt_data['high']:,.2f}</div>
                    </div>
                    <div class="metric-card">
                        <h4>Low</h4>
                        <div class="value">{mkt_data['low']:,.2f}</div>
                    </div>
                    <div class="metric-card">
                        <h4>Close</h4>
                        <div class="value">{mkt_data['close']:,.2f}</div>
                    </div>
                    <div class="metric-card">
                        <h4>High-Low %</h4>
                        <div class="value">{mkt_data['high_low_pct']:.2f}%</div>
                    </div>
                </div>
"""

        if data_point['controllers']:
            html_content += """
                <h3 style="margin-top: 30px; margin-bottom: 15px; color: #848e9c;">Controller Performance</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Controller ID</th>
                            <th>Bot Name</th>
                            <th>Trades</th>
                            <th>Base Volume</th>
                            <th>Quote Volume</th>
                            <th>Market Share</th>
                        </tr>
                    </thead>
                    <tbody>
"""

            sorted_controllers = sorted(data_point['controllers'].items(), key=lambda x: x[1]['quote_volume'], reverse=True)

            for controller_id, ctrl_data in sorted_controllers:
                share_class = "share-high" if ctrl_data['quote_market_share'] >= 0.1 else "share-medium" if ctrl_data['quote_market_share'] >= 0.01 else "share-low"
                time_str = format_timedelta(ctrl_data['time_elapsed'])

                html_content += f"""
                        <tr>
                            <td><strong>{controller_id}</strong></td>
                            <td><span class="bot-name">{ctrl_data['bot_name']}</span><span class="time-elapsed">({time_str})</span></td>
                            <td>{ctrl_data['trades']:,}</td>
                            <td>{ctrl_data['base_volume']:,.4f}</td>
                            <td>{ctrl_data['quote_volume']:,.0f}</td>
                            <td class="{share_class}">{ctrl_data['base_market_share']:.4f}%</td>
                        </tr>
"""

            html_content += """
                    </tbody>
                </table>
"""

        html_content += """
                </div>
            </div>
"""

    html_content += """
        </div>
        <div class="footer">
            <p>Generated by Brigado v2 Market Analysis</p>
            <p style="margin-top: 5px; font-size: 0.85em;">Market data from Binance</p>
        </div>
    </div>
</body>
</html>
"""

    return html_content


def main():
    """Entry point."""
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
