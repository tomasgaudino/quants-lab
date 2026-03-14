"""
Pivot Table Generator for Evolutive Report

Generates horizontal pivot tables with days as columns and sections/metrics as rows.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def _get_heatmap_color(value: float, min_val: float, max_val: float, metric_type: str = 'default') -> Tuple[str, str]:
    """
    Get background and text color for heatmap cell.

    Args:
        value: Current cell value
        min_val: Minimum value in the row
        max_val: Maximum value in the row
        metric_type: Type of metric ('price_diff', 'market_share', 'default')

    Returns:
        Tuple of (background_color, text_color)
    """
    if value == 0 or max_val == min_val:
        return '#1e2329', '#848e9c'  # Dark bg, gray text for zero/no variation

    # Price difference: red to green
    if metric_type == 'price_diff':
        if value > 0:
            # Positive: light green to dark green
            intensity = min(abs(value) / (max_val if max_val > 0 else 1), 1.0)
            alpha = 0.2 + (intensity * 0.5)  # 0.2 to 0.7
            return f'rgba(14, 203, 129, {alpha})', '#eaecef'
        else:
            # Negative: light red to dark red
            intensity = min(abs(value) / (abs(min_val) if min_val < 0 else 1), 1.0)
            alpha = 0.2 + (intensity * 0.5)
            return f'rgba(246, 70, 93, {alpha})', '#eaecef'

    # Market share: yellow gradient
    elif metric_type == 'market_share':
        # Normalize to 0-1 range
        normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0
        alpha = 0.2 + (normalized * 0.6)  # 0.2 to 0.8
        return f'rgba(240, 185, 11, {alpha})', '#0b0e11'

    # Default: blue gradient
    else:
        normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0
        alpha = 0.15 + (normalized * 0.5)  # 0.15 to 0.65
        return f'rgba(46, 134, 171, {alpha})', '#eaecef'


def generate_pivot_table_html(symbol: str, df: pd.DataFrame, trades: pd.DataFrame = None) -> str:
    """
    Generate pivot table with days as columns.

    Structure:
    - MARKET section: OHLC, total trades, volumes, price diff
    - OVERALL BOTS section: trades, volumes, market share per day
    - Per-CONTROLLER sections: trades, volumes, market share for each controller

    Args:
        symbol: Trading pair symbol
        df: Daily metrics DataFrame
        trades: Enriched trades DataFrame with controller_id

    Returns:
        HTML table string
    """
    if trades is None or len(trades) == 0:
        return "<p>No trade data available for pivot table</p>"

    # Ensure timestamp is datetime
    trades = trades.copy()
    trades['timestamp'] = pd.to_datetime(trades['timestamp'])
    trades['date'] = trades['timestamp'].dt.date

    # Get sorted unique dates
    dates = sorted(trades['date'].unique())

    # Build pivot data structure
    pivot_data = _build_pivot_data(df, trades, dates)

    # Generate HTML
    html = """
                <h3 style="margin-top: 30px;">📊 Daily Pivot Table</h3>
                <div style="overflow-x: auto; margin-top: 15px;">
                <table style="min-width: 100%;">
                    <thead>
                        <tr>
                            <th style="position: sticky; left: 0; background: #1e2329; z-index: 10; min-width: 200px;">Section</th>
                            <th style="position: sticky; left: 200px; background: #1e2329; z-index: 10; min-width: 150px;">Metric</th>
"""

    # Date columns
    for date in dates:
        date_str = pd.to_datetime(date).strftime('%m/%d')
        html += f"""                            <th style="min-width: 100px; text-align: right;">{date_str}</th>
"""

    html += """                        </tr>
                    </thead>
                    <tbody>
"""

    # MARKET section
    html += _generate_market_section(pivot_data['market'], dates)

    # OVERALL BOTS section
    html += _generate_overall_bots_section(pivot_data['bots'], dates)

    # Per-CONTROLLER sections
    for controller_id in sorted(pivot_data['controllers'].keys()):
        html += _generate_controller_section(controller_id, pivot_data['controllers'][controller_id], dates)

    html += """                    </tbody>
                </table>
                </div>
"""

    return html


def _build_pivot_data(df: pd.DataFrame, trades: pd.DataFrame, dates: List) -> Dict:
    """Build pivot data structure from metrics and trades."""
    pivot_data = {
        'market': {},
        'bots': {},
        'controllers': {}
    }

    # Convert df dates to date objects for matching
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.date

    for date in dates:
        # Market data - try to get from df first, fallback to calculating from trades
        day_metrics = df[df['date'] == date]
        market_volume_quote = 0

        if len(day_metrics) > 0:
            row = day_metrics.iloc[0]
            market_volume_quote = float(row.get('market_volume_quote', 0))

            import numpy as np

            def safe_float(val, default=0):
                try:
                    if pd.isna(val):
                        return default
                    return float(val)
                except:
                    return default

            def safe_int(val, default=0):
                try:
                    if pd.isna(val):
                        return default
                    return int(val)
                except:
                    return default

            pivot_data['market'][date] = {
                'open': safe_float(row.get('market_open', 0)),
                'high': safe_float(row.get('market_high', 0)),
                'low': safe_float(row.get('market_low', 0)),
                'close': safe_float(row.get('market_close', 0)),
                'trades': safe_int(row.get('market_trades_count', 0)),
                'volume_base': safe_float(row.get('market_volume_base', 0)),
                'volume_quote': market_volume_quote,
            }
            pivot_data['market'][date]['price_diff'] = pivot_data['market'][date]['close'] - pivot_data['market'][date]['open']
        else:
            # No market data available
            pivot_data['market'][date] = {
                'open': 0, 'high': 0, 'low': 0, 'close': 0,
                'trades': 0, 'volume_base': 0, 'volume_quote': 0, 'price_diff': 0
            }

        # Overall bots data
        day_trades = trades[trades['date'] == date]
        if len(day_trades) > 0:
            bot_volume_quote = float((day_trades['amount'] * day_trades['price']).sum())
            pivot_data['bots'][date] = {
                'trades': len(day_trades),
                'volume_base': float(day_trades['amount'].sum()),
                'volume_quote': bot_volume_quote,
            }
            # Calculate market share - use the market_volume_quote we extracted above
            if market_volume_quote > 0:
                pivot_data['bots'][date]['market_share'] = (bot_volume_quote / market_volume_quote) * 100
            else:
                pivot_data['bots'][date]['market_share'] = 0

        # Per-controller data
        controllers = day_trades['controller_id'].dropna().unique()
        for controller_id in controllers:
            if controller_id not in pivot_data['controllers']:
                pivot_data['controllers'][controller_id] = {}

            ctrl_trades = day_trades[day_trades['controller_id'] == controller_id]
            ctrl_volume_quote = float((ctrl_trades['amount'] * ctrl_trades['price']).sum())

            pivot_data['controllers'][controller_id][date] = {
                'trades': len(ctrl_trades),
                'volume_base': float(ctrl_trades['amount'].sum()),
                'volume_quote': ctrl_volume_quote,
            }
            # Market share - use the market_volume_quote we extracted above
            if market_volume_quote > 0:
                pivot_data['controllers'][controller_id][date]['market_share'] = (ctrl_volume_quote / market_volume_quote) * 100
            else:
                pivot_data['controllers'][controller_id][date]['market_share'] = 0

    return pivot_data


def _generate_market_section(market_data: Dict, dates: List) -> str:
    """Generate MARKET section rows."""
    metrics = [
        ('Open', 'open', ',.2f', 'default'),
        ('High', 'high', ',.2f', 'default'),
        ('Low', 'low', ',.2f', 'default'),
        ('Close', 'close', ',.2f', 'default'),
        ('Open-Close Δ', 'price_diff', '+,.2f', 'price_diff'),
        ('Total Trades', 'trades', ',', 'default'),
        ('Base Volume', 'volume_base', ',.2f', 'default'),
        ('Quote Volume', 'volume_quote', ',.0f', 'default'),
    ]

    html = ""
    for i, (label, key, fmt, color_type) in enumerate(metrics):
        row_class = "background: #1e2329; border-top: 2px solid #f0b90b;" if i == 0 else "background: #1e2329;"
        section_cell = f'<td rowspan="{len(metrics)}" style="position: sticky; left: 0; background: #1e2329; vertical-align: middle; font-weight: bold; color: #848e9c; border-right: 1px solid #3d4551;">📊 MARKET</td>' if i == 0 else ''

        html += f"""                        <tr style="{row_class}">
                            {section_cell}
                            <td style="position: sticky; left: 200px; background: #1e2329; border-right: 1px solid #3d4551;">{label}</td>
"""

        # Collect values for heatmap calculation
        values = [market_data.get(date, {}).get(key, 0) for date in dates]
        min_val = min(values) if values else 0
        max_val = max(values) if values else 0

        for date in dates:
            value = market_data.get(date, {}).get(key, 0)
            formatted = f"{{:{fmt}}}".format(value) if value != 0 else "-"

            # Get heatmap colors
            bg_color, text_color = _get_heatmap_color(value, min_val, max_val, color_type)

            html += f"""                            <td style="text-align: right; background: {bg_color}; color: {text_color};">{formatted}</td>
"""

        html += """                        </tr>
"""

    return html


def _generate_overall_bots_section(bots_data: Dict, dates: List) -> str:
    """Generate OVERALL BOTS section rows."""
    metrics = [
        ('Trades', 'trades', ',', 'default'),
        ('Base Volume', 'volume_base', ',.2f', 'default'),
        ('Quote Volume', 'volume_quote', ',.0f', 'default'),
        ('Market Share %', 'market_share', ',.2f', 'market_share'),
    ]

    html = ""
    for i, (label, key, fmt, color_type) in enumerate(metrics):
        row_class = "background: #2b3139; border-top: 2px solid #f0b90b;" if i == 0 else "background: #2b3139;"
        section_cell = f'<td rowspan="{len(metrics)}" style="position: sticky; left: 0; background: #2b3139; vertical-align: middle; font-weight: bold; color: #f0b90b; border-right: 1px solid #3d4551;">🤖 OVERALL BOTS</td>' if i == 0 else ''

        html += f"""                        <tr style="{row_class}">
                            {section_cell}
                            <td style="position: sticky; left: 200px; background: #2b3139; border-right: 1px solid #3d4551;">{label}</td>
"""

        # Collect values for heatmap calculation
        values = [bots_data.get(date, {}).get(key, 0) for date in dates]
        min_val = min(values) if values else 0
        max_val = max(values) if values else 0

        for date in dates:
            value = bots_data.get(date, {}).get(key, 0)
            formatted = f"{{:{fmt}}}".format(value) if value != 0 else "-"

            # Get heatmap colors
            bg_color, text_color = _get_heatmap_color(value, min_val, max_val, color_type)

            html += f"""                            <td style="text-align: right; background: {bg_color}; color: {text_color};">{formatted}</td>
"""

        html += """                        </tr>
"""

    return html


def _generate_controller_section(controller_id: str, ctrl_data: Dict, dates: List) -> str:
    """Generate per-CONTROLLER section rows."""
    metrics = [
        ('Trades', 'trades', ',', 'default'),
        ('Base Volume', 'volume_base', ',.2f', 'default'),
        ('Quote Volume', 'volume_quote', ',.0f', 'default'),
        ('Market Share %', 'market_share', ',.2f', 'market_share'),
    ]

    html = ""
    for i, (label, key, fmt, color_type) in enumerate(metrics):
        row_class = "background: #1e2329; border-top: 1px solid #3d4551;" if i == 0 else "background: #1e2329;"
        section_cell = f'<td rowspan="{len(metrics)}" style="position: sticky; left: 0; background: #1e2329; vertical-align: middle; font-size: 0.85em; color: #0ecb81; border-right: 1px solid #3d4551;">🎮 {controller_id}</td>' if i == 0 else ''

        html += f"""                        <tr style="{row_class}">
                            {section_cell}
                            <td style="position: sticky; left: 200px; background: #1e2329; border-right: 1px solid #3d4551; font-size: 0.9em;">{label}</td>
"""

        # Collect values for heatmap calculation
        values = [ctrl_data.get(date, {}).get(key, 0) for date in dates]
        min_val = min(values) if values else 0
        max_val = max(values) if values else 0

        for date in dates:
            value = ctrl_data.get(date, {}).get(key, 0)
            formatted = f"{{:{fmt}}}".format(value) if value > 0 else "-"

            # Get heatmap colors
            bg_color, text_color = _get_heatmap_color(value, min_val, max_val, color_type)

            html += f"""                            <td style="text-align: right; background: {bg_color}; color: {text_color}; font-size: 0.9em;">{formatted}</td>
"""

        html += """                        </tr>
"""

    return html
