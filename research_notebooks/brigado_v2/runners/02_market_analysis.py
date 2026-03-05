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
    """Fetch market data for a single trading pair."""
    try:
        binance_pair = pair.replace('-', '')
        print_info(f"Fetching {pair} ({binance_pair})...")

        candles = await clob.get_candles(
            connector_name='binance',
            trading_pair=binance_pair,
            interval='1m',
            start_time=int(start_date.timestamp()),
            end_time=int(end_date.timestamp())
        )

        if candles.data is not None and len(candles.data) > 0:
            df = candles.data

            data = {
                'open': float(df['open'].iloc[0]),
                'high': float(df['high'].max()),
                'low': float(df['low'].min()),
                'close': float(df['close'].iloc[-1]),
                'base_volume': float(df['volume'].sum()),
                'quote_volume': float(df['quote_asset_volume'].sum()) if 'quote_asset_volume' in df.columns else 0,
                'n_trades': int(df['n_trades'].sum()) if 'n_trades' in df.columns else 0,
                'high_low_pct': ((float(df['high'].max()) - float(df['low'].min())) / float(df['low'].min()) * 100),
            }

            print_metric("Base Volume", f"{data['base_volume']:,.2f}", indent=6)
            print_metric("Quote Volume", f"{data['quote_volume']:,.0f}", indent=6)
            print_metric("Trades", f"{data['n_trades']:,}", indent=6)
            print_metric("OHLC", f"{data['open']:.2f} → {data['close']:.2f} (Δ{data['high_low_pct']:.2f}%)", indent=6)

            return pair, data
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
    start_date = trades['timestamp'].min()
    end_date = trades['timestamp'].max()

    print_step(f"Fetching market data for {len(trading_pairs)} pairs...")
    print_info(f"Date range: {start_date.date()} to {end_date.date()}")
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

            controller_volumes[controller_id] = {
                'bot_name': bot_name,
                'trades': len(ctrl_trades),
                'base_volume': float(ctrl_trades['amount'].sum()),
                'quote_volume': float(ctrl_trades['quote_volume'].sum())
            }

        # Total bot volume for this pair
        total_bot_base_volume = float(pair_trades['amount'].sum())
        total_bot_quote_volume = float(pair_trades['quote_volume'].sum())

        # Market share calculations
        if market_data.get(pair) and market_data[pair]:
            market_base_volume = market_data[pair]['base_volume']
            market_quote_volume = market_data[pair]['quote_volume']

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

        market_share_data[pair] = {
            'market_data': market_data.get(pair),
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
        .header {{ background: linear-gradient(135deg, #2b3139 0%, #1e2329 100%); border-bottom: 3px solid #f0b90b; color: #f0b90b; padding: 40px; text-align: center; }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; font-weight: 600; text-shadow: 0 0 20px rgba(240, 185, 11, 0.3); }}
        .header p {{ font-size: 1.1em; color: #848e9c; }}
        .content {{ padding: 40px; }}
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
        .share-high {{ color: #0ecb81; font-weight: 600; }}
        .share-medium {{ color: #f0b90b; font-weight: 600; }}
        .share-low {{ color: #848e9c; }}
        .footer {{ background: #1e2329; border-top: 1px solid #2b3139; padding: 20px; text-align: center; color: #848e9c; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Market Performance Analysis</h1>
            <p>Controller Performance vs Market Activity by Trading Pair</p>
            <p style="margin-top: 10px;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        <div class="content">
"""

    # Add sections for each pair
    for pair in sorted(market_share_data.keys()):
        data_point = market_share_data[pair]
        mkt_data = data_point['market_data']

        html_content += f"""
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

                html_content += f"""
                        <tr>
                            <td><strong>{controller_id}</strong></td>
                            <td><span class="bot-name">{ctrl_data['bot_name']}</span></td>
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
