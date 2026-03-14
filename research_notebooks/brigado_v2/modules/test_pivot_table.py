#!/usr/bin/env python3
"""
Test script for pivot table market share calculation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pivot_table_generator import generate_pivot_table_html
from pathlib import Path

# Create test data
print("=" * 80)
print("PIVOT TABLE MARKET SHARE TEST")
print("=" * 80)

# Create dates
base_date = datetime(2026, 3, 1).date()
dates = [base_date + timedelta(days=i) for i in range(5)]

# Create test trades DataFrame
trades_data = []
for i, date in enumerate(dates):
    # Market generates ~10,000 trades, ~1M volume
    # Our bots generate ~300 trades, ~30k volume (3% market share)

    # Bot trades for controller 1
    for j in range(100):
        trades_data.append({
            'timestamp': pd.Timestamp(date),
            'date': date,
            'symbol': 'BTC-BRL',
            'amount': 0.1,
            'price': 370000 + (i * 1000),
            'controller_id': 'ctrl-1',
            'trade_type': 'buy' if j % 2 == 0 else 'sell',
            'order_id': f'order-{date}-ctrl1-{j}'
        })

    # Bot trades for controller 2
    for j in range(100):
        trades_data.append({
            'timestamp': pd.Timestamp(date),
            'date': date,
            'symbol': 'BTC-BRL',
            'amount': 0.08,
            'price': 370000 + (i * 1000),
            'controller_id': 'ctrl-2',
            'trade_type': 'buy' if j % 2 == 0 else 'sell',
            'order_id': f'order-{date}-ctrl2-{j}'
        })

    # Orphan trades (no controller)
    for j in range(50):
        trades_data.append({
            'timestamp': pd.Timestamp(date),
            'date': date,
            'symbol': 'BTC-BRL',
            'amount': 0.05,
            'price': 370000 + (i * 1000),
            'controller_id': None,
            'trade_type': 'buy' if j % 2 == 0 else 'sell',
            'order_id': f'order-{date}-orphan-{j}'
        })

trades_df = pd.DataFrame(trades_data)

# Create test metrics DataFrame (with market data)
metrics_data = []
for i, date in enumerate(dates):
    price = 370000 + (i * 1000)

    # Calculate actual bot volumes from trades
    day_trades = trades_df[trades_df['date'] == date]
    bot_volume_quote = (day_trades['amount'] * day_trades['price']).sum()
    bot_trades_count = len(day_trades)

    # Market data (much larger than bot)
    market_volume_quote = 1_000_000_000  # 1 billion BRL
    market_trades = 50_000

    metrics_data.append({
        'date': date,
        'symbol': 'BTC-BRL',
        # Market data
        'market_open': price - 2000,
        'market_high': price + 3000,
        'market_low': price - 3000,
        'market_close': price + 1000,
        'market_volume_base': market_volume_quote / price,
        'market_volume_quote': market_volume_quote,
        'market_trades_count': market_trades,
        # Bot data (calculated from trades)
        'bot_volume_quote': bot_volume_quote,
        'bot_trades_count': bot_trades_count,
    })

metrics_df = pd.DataFrame(metrics_data)

# Print test data summary
print("\n📊 Test Data Summary:")
print(f"  Dates: {len(dates)} days")
print(f"  Total trades: {len(trades_df)}")
print(f"  Controllers: {trades_df['controller_id'].dropna().nunique()}")
print()

# Print per-day summary
print("Per-Day Breakdown:")
for date in dates:
    day_trades = trades_df[trades_df['date'] == date]
    day_metrics = metrics_df[metrics_df['date'] == date].iloc[0]

    bot_volume = (day_trades['amount'] * day_trades['price']).sum()
    market_volume = day_metrics['market_volume_quote']
    expected_share = (bot_volume / market_volume * 100) if market_volume > 0 else 0

    print(f"\n  {date}:")
    print(f"    Market volume: {market_volume:,.0f}")
    print(f"    Bot volume:    {bot_volume:,.0f}")
    print(f"    Expected share: {expected_share:.4f}%")

    # Per controller
    for ctrl in ['ctrl-1', 'ctrl-2']:
        ctrl_trades = day_trades[day_trades['controller_id'] == ctrl]
        ctrl_volume = (ctrl_trades['amount'] * ctrl_trades['price']).sum()
        ctrl_share = (ctrl_volume / market_volume * 100) if market_volume > 0 else 0
        print(f"    {ctrl} volume: {ctrl_volume:,.0f} ({ctrl_share:.4f}%)")

# Generate pivot table HTML
print("\n" + "=" * 80)
print("GENERATING PIVOT TABLE...")
print("=" * 80)

output_path = Path(__file__).parent / "test_pivot_output.html"
html = generate_pivot_table_html('BTC-BRL', metrics_df, trades_df)

# Save to file
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Pivot Table Test</title>
</head>
<body>
{html}
</body>
</html>
""")

print(f"\n✅ Test HTML saved to: {output_path}")
print("\nOpen this file in a browser to verify:")
print("  1. Market share values are showing (not empty)")
print("  2. Values match expected percentages above")
print("  3. Heatmap colors are applied")
print("  4. All sections (Market, Bots, Controllers) are present")
print()

# Extract and verify market share from generated HTML
if "Market Share %" in html and "rgba(" in html:
    print("✅ Heatmap colors detected in HTML")
else:
    print("❌ WARNING: Heatmap colors may not be applied")

if "-" not in html or "0.00" not in html:
    print("✅ Market share values appear to be calculated")
else:
    print("⚠️  Check HTML - may have empty market share cells")

print("\n" + "=" * 80)
