"""
Market Features Module

Calculates market diagnostic metrics for price, order book, and trades.
Supports resampling to create time-series DataFrames suitable for analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from scipy import stats


class FeatureCalculationError(Exception):
    """Raised when feature calculation fails."""
    pass


def compute_all_features(
    candles: pd.DataFrame,
    orderbook: pd.DataFrame,
    trades: pd.DataFrame,
    config: Dict[str, Any],
    resample: bool = True
) -> pd.DataFrame:
    """
    Calculate all market features for the diagnostic.

    Args:
        candles: DataFrame with OHLCV (timestamp, open, high, low, close, volume)
        orderbook: DataFrame with snapshots (timestamp, best_bid, best_ask, etc.)
        trades: DataFrame with trades (timestamp, price, amount, side)
        config: Configuration loaded from YAML
        resample: If True, resample data into time windows and return DataFrame.
                  If False, compute features for entire dataset and return single-row DataFrame.

    Returns:
        DataFrame with columns for each feature, indexed by time window (if resampled)
        or single row with all features (if not resampled).
    """
    try:
        if resample:
            return _compute_features_resampled(candles, orderbook, trades, config)
        else:
            return _compute_features_single(candles, orderbook, trades, config)

    except Exception as e:
        raise FeatureCalculationError(f"Error computing features: {str(e)}") from e


def _compute_features_resampled(
    candles: pd.DataFrame,
    orderbook: pd.DataFrame,
    trades: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Compute features with resampling into time windows.

    Returns a DataFrame where each row represents a time window with all computed features.
    """
    # Get resampling configuration
    interval = config.get('resampling', {}).get('interval', '1H')
    label = config.get('resampling', {}).get('label', 'end')
    include_partial = config.get('resampling', {}).get('include_partial', False)

    # Ensure timestamp columns are datetime
    candles = _prepare_dataframe(candles, 'timestamp')
    orderbook = _prepare_dataframe(orderbook, 'timestamp')
    trades = _prepare_dataframe(trades, 'timestamp')

    # Find common time range across all datasets
    min_time = max(
        candles['timestamp'].min(),
        orderbook['timestamp'].min(),
        trades['timestamp'].min()
    )
    max_time = min(
        candles['timestamp'].max(),
        orderbook['timestamp'].max(),
        trades['timestamp'].max()
    )

    # Create time bins - ensure we cover the full range
    # Use closed='left' to include the start time and add one extra period at the end
    time_bins = pd.date_range(start=min_time, end=max_time, freq=interval, inclusive='both')

    # If we don't have enough bins, add one more period at the end
    if len(time_bins) < 2:
        # Try to infer the time delta from interval
        try:
            delta = pd.Timedelta(interval)
            time_bins = pd.DatetimeIndex([min_time, min_time + delta, max_time])
        except:
            raise FeatureCalculationError(
                f"Insufficient data for resampling with interval {interval}. "
                f"Time range: {min_time} to {max_time}"
            )

    # Initialize results list
    results = []
    errors = []

    # Iterate over time windows
    for i in range(len(time_bins) - 1):
        window_start = time_bins[i]
        window_end = time_bins[i + 1]

        # Filter data for this window
        candles_window = candles[
            (candles['timestamp'] >= window_start) &
            (candles['timestamp'] < window_end)
        ]
        orderbook_window = orderbook[
            (orderbook['timestamp'] >= window_start) &
            (orderbook['timestamp'] < window_end)
        ]
        trades_window = trades[
            (trades['timestamp'] >= window_start) &
            (trades['timestamp'] < window_end)
        ]

        # Skip if any dataset is empty in this window
        if len(candles_window) == 0 or len(orderbook_window) == 0 or len(trades_window) == 0:
            errors.append(f"Window {window_start} to {window_end}: Empty data (candles={len(candles_window)}, orderbook={len(orderbook_window)}, trades={len(trades_window)})")
            continue

        # Compute features for this window
        try:
            price_features = _compute_price_features(candles_window, config)
            order_book_features = _compute_order_book_features(orderbook_window, config)
            trades_features = _compute_trades_features(trades_window, config)

            # Combine all features
            row = {
                'timestamp': window_end if label == 'end' else window_start,
                **{f'price_{k}': v for k, v in price_features.items()},
                **{f'order_book_{k}': v for k, v in order_book_features.items()},
                **{f'trades_{k}': v for k, v in trades_features.items()}
            }

            results.append(row)

        except Exception as e:
            # Track errors for debugging
            errors.append(f"Window {window_start} to {window_end}: {str(e)}")
            continue

    if len(results) == 0:
        error_msg = f"No valid time windows produced features.\n"
        error_msg += f"Time range: {min_time} to {max_time}\n"
        error_msg += f"Interval: {interval}\n"
        error_msg += f"Number of bins: {len(time_bins)}\n"
        error_msg += f"Errors encountered:\n" + "\n".join(errors[:5])  # Show first 5 errors
        raise FeatureCalculationError(error_msg)

    # Convert to DataFrame
    df = pd.DataFrame(results)
    df.set_index('timestamp', inplace=True)

    return df


def _compute_features_single(
    candles: pd.DataFrame,
    orderbook: pd.DataFrame,
    trades: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Compute features for the entire dataset as a single observation.

    Returns a single-row DataFrame with all features.
    """
    price_features = _compute_price_features(candles, config)
    order_book_features = _compute_order_book_features(orderbook, config)
    trades_features = _compute_trades_features(trades, config)

    # Combine all features
    row = {
        **{f'price_{k}': v for k, v in price_features.items()},
        **{f'order_book_{k}': v for k, v in order_book_features.items()},
        **{f'trades_{k}': v for k, v in trades_features.items()}
    }

    df = pd.DataFrame([row])
    return df


def _prepare_dataframe(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """Ensure DataFrame has datetime timestamp column."""
    df = df.copy()
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    else:
        raise FeatureCalculationError(f"DataFrame missing '{timestamp_col}' column")
    return df


# ==============================================================================
# PRICE FEATURES
# ==============================================================================

def _compute_price_features(candles: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute all price-related features from candles.

    Uses close prices as primary source. For last_mid_price, uses close as proxy
    (in production, could combine with orderbook best_bid/best_ask if needed).
    """
    if 'close' not in candles.columns:
        raise FeatureCalculationError("Candles DataFrame must have 'close' column")

    close = candles['close'].values
    high = candles['high'].values if 'high' in candles.columns else close
    low = candles['low'].values if 'low' in candles.columns else close

    # Windows from config (default values if not present)
    short_window = config.get('price', {}).get('short_window', 15)
    long_window = config.get('price', {}).get('long_window', 60)

    # last_mid_price: using last close as proxy
    last_mid_price = float(close[-1]) if len(close) > 0 else 0.0

    # price_avg: mean close price in window
    price_avg = float(np.mean(close))

    # price_std: std of close prices
    price_std = float(np.std(close))

    # price_diff_pct_t_1: % change from previous close
    price_diff_pct_t_1 = float(((close[-1] / close[-2]) - 1) * 100) if len(close) >= 2 else 0.0

    # price_velocity: ΔP / ΔT (using short window)
    # Assume equally spaced candles
    n_steps = min(short_window, len(close) - 1)
    if n_steps > 0:
        price_velocity = float((close[-1] - close[-n_steps - 1]) / n_steps)
    else:
        price_velocity = 0.0

    # high, low, delta_high_low
    max_high = float(np.max(high))
    min_low = float(np.min(low))
    delta_high_low = float(max_high - min_low)

    # return_volatility: std of log returns
    if len(close) > 1:
        log_returns = np.log(close[1:] / close[:-1])
        return_volatility = float(np.std(log_returns))
    else:
        return_volatility = 0.0

    # microtrend_slope: linear regression slope over recent window
    window_size = min(short_window, len(close))
    if window_size >= 2:
        x = np.arange(window_size)
        y = close[-window_size:]
        slope, _, _, _, _ = stats.linregress(x, y)
        microtrend_slope = float(slope)
    else:
        microtrend_slope = 0.0

    # short_vs_long_velocity_ratio
    if len(close) > long_window:
        short_vel = (close[-1] - close[-short_window - 1]) / short_window
        long_vel = (close[-1] - close[-long_window - 1]) / long_window
        short_vs_long_velocity_ratio = float(short_vel / long_vel) if long_vel != 0 else 0.0
    else:
        short_vs_long_velocity_ratio = 0.0

    return {
        'last_mid_price': last_mid_price,
        'price_avg': price_avg,
        'price_std': price_std,
        'price_diff_pct_t_1': price_diff_pct_t_1,
        'price_velocity': price_velocity,
        'high': max_high,
        'low': min_low,
        'delta_high_low': delta_high_low,
        'return_volatility': return_volatility,
        'microtrend_slope': microtrend_slope,
        'short_vs_long_velocity_ratio': short_vs_long_velocity_ratio
    }


# ==============================================================================
# ORDER BOOK FEATURES
# ==============================================================================

def _compute_order_book_features(orderbook: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute all order book features from orderbook snapshots.

    Note: depth_5, depth_10, depth_20 use simplified proxy if full depth not available.
    Assumes best_bid, best_ask, and optionally best_bid_size, best_ask_size columns.
    """
    if 'best_bid' not in orderbook.columns or 'best_ask' not in orderbook.columns:
        raise FeatureCalculationError("Orderbook must have 'best_bid' and 'best_ask' columns")

    best_bid = orderbook['best_bid'].values
    best_ask = orderbook['best_ask'].values

    # mid_price: mean of mids across snapshots
    mids = (best_bid + best_ask) / 2
    mid_price = float(np.mean(mids))
    last_mid = mids[-1] if len(mids) > 0 else mid_price

    # best_bid and best_ask: last values
    last_best_bid = float(best_bid[-1]) if len(best_bid) > 0 else 0.0
    last_best_ask = float(best_ask[-1]) if len(best_ask) > 0 else 0.0

    # spread_abs and spread_rel
    spreads_abs = best_ask - best_bid
    spread_abs = float(np.mean(spreads_abs))
    spread_rel = float(spread_abs / mid_price) if mid_price > 0 else 0.0

    # Liquidity metrics (using best_bid_size, best_ask_size if available, else use proxy)
    if 'best_bid_size' in orderbook.columns and 'best_ask_size' in orderbook.columns:
        bid_sizes = orderbook['best_bid_size'].values
        ask_sizes = orderbook['best_ask_size'].values
    else:
        # Proxy: assume unit size if not available
        bid_sizes = np.ones(len(orderbook))
        ask_sizes = np.ones(len(orderbook))

    total_bid_liquidity = float(np.sum(bid_sizes))
    total_ask_liquidity = float(np.sum(ask_sizes))

    # liquidity_imbalance
    total_liq = total_bid_liquidity + total_ask_liquidity
    if total_liq > 0:
        liquidity_imbalance = float((total_bid_liquidity - total_ask_liquidity) / total_liq)
    else:
        liquidity_imbalance = 0.0

    # depth_5, depth_10, depth_20: simplified proxy as average available liquidity at best level
    # In production, would sum across multiple levels up to N ticks from mid
    # Here we approximate by scaling best level liquidity
    avg_bid_size = float(np.mean(bid_sizes))
    avg_ask_size = float(np.mean(ask_sizes))

    # Proxy: assume linear depth, multiply by number of levels
    depth_5 = float((avg_bid_size + avg_ask_size) * 5)
    depth_10 = float((avg_bid_size + avg_ask_size) * 10)
    depth_20 = float((avg_bid_size + avg_ask_size) * 20)

    # market_pressure_index: combines imbalance and spread
    # Formula: (liquidity_imbalance) / (1 + spread_rel)
    # Higher positive = buy pressure, negative = sell pressure
    market_pressure_index = float(liquidity_imbalance / (1 + spread_rel)) if spread_rel >= 0 else 0.0

    # queue_dynamics: proportion of snapshots where best bid/ask changed
    if len(orderbook) > 1:
        bid_changes = np.sum(np.diff(best_bid) != 0)
        ask_changes = np.sum(np.diff(best_ask) != 0)
        total_changes = bid_changes + ask_changes
        queue_dynamics = float(total_changes / (2 * (len(orderbook) - 1)))
    else:
        queue_dynamics = 0.0

    # order_book_convexity: proxy measuring liquidity concentration near mid
    # Higher value = more liquidity near mid (assumes sizes decrease with distance)
    # Simplified: use std of bid/ask sizes as proxy (lower std = more concentrated)
    convexity_proxy = 1.0 / (1.0 + float(np.std(bid_sizes) + np.std(ask_sizes)))
    order_book_convexity = convexity_proxy

    # slippage_cost: estimated cost in % to execute standard order size
    # Get standard order size from config or use default
    standard_order_size = config.get('order_book', {}).get('standard_order_size', 1000.0)

    # Simplified: assume slippage proportional to (order_size / avg_depth) * spread
    avg_depth = (avg_bid_size + avg_ask_size) / 2
    if avg_depth > 0:
        slippage_cost = float((standard_order_size / avg_depth) * spread_rel * 100)  # in %
    else:
        slippage_cost = 0.0

    # order_book_entropy: Shannon entropy of liquidity distribution
    # Simplified: use bid/ask size distribution
    # Normalize sizes to probabilities
    all_sizes = np.concatenate([bid_sizes, ask_sizes])
    total_size = np.sum(all_sizes)
    if total_size > 0:
        probs = all_sizes / total_size
        probs = probs[probs > 0]  # Remove zeros
        entropy = -np.sum(probs * np.log2(probs))
        order_book_entropy = float(entropy)
    else:
        order_book_entropy = 0.0

    return {
        'mid_price': mid_price,
        'best_bid': last_best_bid,
        'best_ask': last_best_ask,
        'spread_abs': spread_abs,
        'spread_rel': spread_rel,
        'depth_5': depth_5,
        'depth_10': depth_10,
        'depth_20': depth_20,
        'total_bid_liquidity': total_bid_liquidity,
        'total_ask_liquidity': total_ask_liquidity,
        'liquidity_imbalance': liquidity_imbalance,
        'market_pressure_index': market_pressure_index,
        'queue_dynamics': queue_dynamics,
        'order_book_convexity': order_book_convexity,
        'slippage_cost': slippage_cost,
        'order_book_entropy': order_book_entropy
    }


# ==============================================================================
# TRADES FEATURES
# ==============================================================================

def _compute_trades_features(trades: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute all trade flow features from trades data.

    Uses 'side' column if available, otherwise applies tick rule to infer direction.
    """
    if 'size' not in trades.columns or 'price' not in trades.columns:
        raise FeatureCalculationError("Trades must have 'price' and 'size' columns")

    amounts = trades['size'].values
    prices = trades['price'].values

    # total_volume
    total_volume = float(np.sum(amounts))

    # Separate buy and sell volumes
    if 'side' in trades.columns:
        buy_mask = trades['side'].str.lower() == 'buy'
        sell_mask = trades['side'].str.lower() == 'sell'
    else:
        # Use tick rule: uptick = buy, downtick = sell
        price_diff = np.diff(prices, prepend=prices[0])
        buy_mask = price_diff > 0
        sell_mask = price_diff < 0

    buy_volume = float(np.sum(amounts[buy_mask]))
    sell_volume = float(np.sum(amounts[sell_mask]))

    # taker_aggressiveness_ratio
    if sell_volume > 0:
        taker_aggressiveness_ratio = float(buy_volume / sell_volume)
    else:
        taker_aggressiveness_ratio = float('inf') if buy_volume > 0 else 0.0

    # vwap: volume-weighted average price
    total_value = np.sum(prices * amounts)
    vwap = float(total_value / total_volume) if total_volume > 0 else 0.0

    # volume_imbalance
    total_vol = buy_volume + sell_volume
    if total_vol > 0:
        volume_imbalance = float((buy_volume - sell_volume) / total_vol)
    else:
        volume_imbalance = 0.0

    # microstructural_volume_signature: ratio of 90th percentile to median trade size
    if len(amounts) > 0:
        p90 = np.percentile(amounts, 90)
        median = np.median(amounts)
        microstructural_volume_signature = float(p90 / median) if median > 0 else 0.0
    else:
        microstructural_volume_signature = 0.0

    # volume_burst_score: z-score of total volume vs median
    # Simplified: use rolling window from config
    window = config.get('trades', {}).get('volume_window', 50)
    if len(trades) >= window:
        # Rolling volume
        rolling_amounts = pd.Series(amounts).rolling(window=window, min_periods=1).sum()
        vol_mean = rolling_amounts.mean()
        vol_std = rolling_amounts.std()
        current_vol = rolling_amounts.iloc[-1]
        if vol_std > 0:
            volume_burst_score = float((current_vol - vol_mean) / vol_std)
        else:
            volume_burst_score = 0.0
    else:
        volume_burst_score = 0.0

    # average_trade_size
    average_trade_size = float(np.mean(amounts)) if len(amounts) > 0 else 0.0

    # volume_time_distribution: simple descriptor of temporal concentration
    # Return dict with concentration index (higher = more concentrated in time)
    # Using Gini coefficient as proxy
    if 'timestamp' in trades.columns and len(trades) > 1:
        trades_copy = trades.copy()
        trades_copy['timestamp'] = pd.to_datetime(trades_copy['timestamp'])

        # Aggregate volume by time bins
        time_range = trades_copy['timestamp'].max() - trades_copy['timestamp'].min()
        n_bins = min(20, len(trades))

        if time_range.total_seconds() > 0:
            trades_copy['time_bin'] = pd.cut(trades_copy['timestamp'], bins=n_bins, labels=False)
            vol_by_bin = trades_copy.groupby('time_bin')['size'].sum().values

            # Gini coefficient
            sorted_vol = np.sort(vol_by_bin)
            n = len(sorted_vol)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_vol)) / (n * np.sum(sorted_vol)) - (n + 1) / n
            volume_time_distribution = float(gini)
        else:
            volume_time_distribution = 0.0
    else:
        volume_time_distribution = 0.0

    # trade_flow_imbalance: rolling imbalance
    # Compute rolling buy/sell volumes and their imbalance
    window_flow = config.get('trades', {}).get('flow_window', 20)
    if len(trades) >= window_flow:
        buy_series = pd.Series(amounts * buy_mask, dtype=float)
        sell_series = pd.Series(amounts * sell_mask, dtype=float)

        rolling_buy = buy_series.rolling(window=window_flow, min_periods=1).sum()
        rolling_sell = sell_series.rolling(window=window_flow, min_periods=1).sum()

        rolling_total = rolling_buy + rolling_sell
        rolling_imbalance = (rolling_buy - rolling_sell) / rolling_total
        rolling_imbalance = rolling_imbalance.fillna(0)

        trade_flow_imbalance = float(rolling_imbalance.iloc[-1])
    else:
        trade_flow_imbalance = volume_imbalance  # Fallback to overall imbalance

    # inter_arrival_time: average time between trades
    if 'timestamp' in trades.columns and len(trades) > 1:
        trades_copy = trades.copy()
        trades_copy['timestamp'] = pd.to_datetime(trades_copy['timestamp'])

        time_diffs = trades_copy['timestamp'].diff().dt.total_seconds()
        inter_arrival_time = float(time_diffs.mean())
    else:
        inter_arrival_time = 0.0

    return {
        'total_volume': total_volume,
        'buy_volume': buy_volume,
        'sell_volume': sell_volume,
        'taker_aggressiveness_ratio': taker_aggressiveness_ratio,
        'vwap': vwap,
        'volume_imbalance': volume_imbalance,
        'microstructural_volume_signature': microstructural_volume_signature,
        'volume_burst_score': volume_burst_score,
        'average_trade_size': average_trade_size,
        'volume_time_distribution': volume_time_distribution,
        'trade_flow_imbalance': trade_flow_imbalance,
        'inter_arrival_time': inter_arrival_time
    }
