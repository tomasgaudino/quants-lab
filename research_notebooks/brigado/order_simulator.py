import asyncio
import sys
import os
from datetime import datetime
import logging
import pandas as pd
import numpy as np

from typing import Dict, Any, List, Tuple
import plotly.express as px

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from sklearn.mixture import GaussianMixture

from core.data_sources import CLOBDataSource
from core.data_structures.candles import Candles

logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("pandas").setLevel(logging.CRITICAL)


async def main(config: Dict[str, Any], root_path: str):
    candles = await get_candles(config, root_path)
    frames = generate_frames(candles, config)
    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.7, 0.3],
        shared_xaxes=False,
        shared_yaxes='rows',
        column_widths=[0.7, 0.3],
        horizontal_spacing=0.06,
        vertical_spacing=0.08
    )

    if frames:
        fig.add_traces(frames[0].data)
    fig.frames = frames
    fig.update_layout(shapes=[])
    fig.update_layout(
        height=800,
        yaxis=dict(title="Price", range=[candles.data.close.min(), candles.data.close.max()]),
        yaxis2=dict(title="VWAP", range=[candles.data.close.min(), candles.data.close.max()]),
        updatemenus=[{
            "type": "buttons",
            "buttons": [{
                "label": "Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 150, "redraw": True}, "fromcurrent": False}]
            }]
        }]
    )
    fig.write_html(
        "order_simulation.html",
        include_plotlyjs="cdn",
        full_html=True
    )


async def get_candles(config: Dict[str, Any], root_path: str):
    connector_name = config["candles_config"]["connector_name"]
    trading_pair = config["candles_config"]["trading_pair"]
    interval = config["candles_config"]["interval"]
    start_time = config["candles_config"]["start_time"]
    end_time = config["candles_config"]["end_time"]
    fetch = config["candles_config"]["fetch"]

    clob = CLOBDataSource()
    if fetch:
        candles = await clob.get_candles(connector_name, trading_pair, interval, start_time, end_time)
        clob.dump_candles_cache(root_path)
    else:
        clob.load_candles_cache(root_path)
        candles = clob.get_candles_from_cache(connector_name, trading_pair, interval)
    return candles


def generate_frames(candles: Candles, config: Dict[str, Any]):
    frames = []
    vwap_timeline = []
    window_size = config["window_size"]
    window_step = config["window_step"]
    candles_df: pd.DataFrame = candles.data.copy()
    for i in range(0, len(candles_df) - window_size, window_step):
        window = candles_df.iloc[i:i+window_size]
        processed_data = simulate_processed_data(window, config)
        vwap_timeline.append((window.index[-1], processed_data["VWAP"]))
        start_price, end_price = get_price_limits(processed_data)
        prices = processed_data["PRICE_LEVELS"]
        sizes = processed_data["VOLUME_LEVELS"] / sum(processed_data["VOLUME_LEVELS"])
        # amounts = [size * config["total_amount_quote"] for size in sizes]
        if start_price is not None and end_price is not None:
            filtered_prices = [
                (price, size)
                for price, size in zip(prices, sizes)
                if start_price <= price <= end_price
            ]
        else:
            filtered_prices = [(price, size) for price, size in zip(prices, sizes)]

        frame = frame_price_with_distribution_and_amounts(window, filtered_prices, processed_data, vwap_timeline,
                                                          config["only_min_max"])
        frames.append(frame)
    return frames


def get_volume_and_price_levels(candles_df: pd.DataFrame, bins: int = 10):
    price_bins = np.linspace(candles_df['low'].min(), candles_df['high'].max(), bins)
    volume_by_price = pd.cut(candles_df['close'], bins=price_bins)
    volume_agg = candles_df.groupby(volume_by_price, observed=False)['volume'].sum()

    price_levels = [price_bin.mid for price_bin in volume_agg.index]
    volumes = volume_agg.values
    return price_levels, volumes


def simulate_processed_data(window: pd.DataFrame, config: Dict[str, Any]):
    bins = config["n_bins"]
    price_levels, volumes = get_volume_and_price_levels(window, bins)

    # Volume distribution metrics
    vwap = np.average(price_levels, weights=volumes)
    std = np.sqrt(np.average((price_levels - vwap) ** 2, weights=volumes))
    mode_price = price_levels[np.argmax(volumes)]

    # Skew and Kurtosis
    dist_skewness = skew(volumes)
    dist_kurtosis = kurtosis(volumes)

    # Classify volume distribution by skew and kurtosis
    if dist_skewness > 0.5:
        skew_label = "Right-skewed"  # (tail toward high prices)
    elif dist_skewness < -0.5:
        skew_label = "Left-skewed"  # (tail toward low prices)
    else:
        skew_label = "Symmetric"

    if dist_kurtosis > 0.5:
        kurtosis_label = "Leptokurtic"  # (peaked, fat tails)
    elif dist_kurtosis < -0.5:
        kurtosis_label = "Platykurtic"  # (flat, light tails)
    else:
        kurtosis_label = "Mesokurtic"

    # Determine Peaks
    peaks, _ = find_peaks(volumes, prominence=np.max(volumes) * config["peak_prominence"])
    num_peaks = len(peaks)
    peaks_prices = [price_levels[i] for i in peaks]

    first_vwap_std = config["first_vwap_std"]
    second_vwap_std = config["second_vwap_std"]
    third_vwap_std = config["third_vwap_std"]

    processed_data = {
        "WINDOW": window,
        "SKEW_LABEL": skew_label,
        "KURTOSIS_LABEL": kurtosis_label,
        "VWAP": vwap,
        "STD": std,
        "MODE": mode_price,
        "N_PEAKS": num_peaks,
        "PRICE_LEVELS": price_levels,
        "VOLUME_LEVELS": volumes,
        "PEAKS_PRICES": peaks_prices,
        "FIRST_LOWER_VWAP_LEVEL": vwap - first_vwap_std * std,
        "FIRST_UPPER_VWAP_LEVEL": vwap + first_vwap_std * std,
        "SECOND_LOWER_VWAP_LEVEL": vwap - second_vwap_std * std,
        "SECOND_UPPER_VWAP_LEVEL": vwap + second_vwap_std * std,
        "THIRD_LOWER_VWAP_LEVEL": vwap - third_vwap_std * std,
        "THIRD_UPPER_VWAP_LEVEL": vwap + third_vwap_std * std,
    }
    return processed_data


def get_price_limits(processed_data: Dict[str, Any]):
    if processed_data["N_PEAKS"] == 1:
        if processed_data["KURTOSIS_LABEL"] == "Leptokurtic":
            start_price = processed_data["FIRST_LOWER_VWAP_LEVEL"]
            end_price = processed_data["FIRST_UPPER_VWAP_LEVEL"]
        elif processed_data["KURTOSIS_LABEL"] == "Platykurtic":
            start_price = processed_data["THIRD_LOWER_VWAP_LEVEL"]
            end_price = processed_data["THIRD_UPPER_VWAP_LEVEL"]
        else:
            start_price = processed_data["SECOND_LOWER_VWAP_LEVEL"]
            end_price = processed_data["SECOND_UPPER_VWAP_LEVEL"]
    elif processed_data["N_PEAKS"] > 1:
        start_price = min(processed_data["PEAKS_PRICES"])
        end_price = max(processed_data["PEAKS_PRICES"])
    else:
        start_price = end_price = None
    return start_price, end_price


def frame_price_with_distribution_and_amounts(window: pd.DataFrame,
                                              prices: List[Tuple[float, float]],
                                              processed_data: Dict[str, Any],
                                              vwap_timeline: List[Tuple[pd.Timestamp, float]],
                                              only_min_max: bool = True) -> go.Frame:
    # Column 1: Line chart of close prices
    trace_close = go.Scatter(
        x=window.index,
        y=window["close"],
        mode="lines",
        name=f"Close Price {window.index[0]}",
        line=dict(color="black", width=0.5)
    )

    # Column 2: Volume distribution bar chart
    trace_volume_dist = go.Bar(
        x=processed_data["VOLUME_LEVELS"],
        y=processed_data["PRICE_LEVELS"],
        orientation="h",
        name=f"Volume Dist {window.index[0]}",
        marker=dict(color="rgba(0, 150, 255, 0.6)"),
        xaxis="x2",
    )
    # Shapes for horizontal dashed lines across both charts
    prices_values = [price for price, _ in prices]
    if only_min_max:
        min_price = min(prices_values)
        max_price = max(prices_values)
        prices_values = [min_price, max_price]
    shapes = [
        dict(
            type="line",
            xref="paper",
            yref="y",  # shared y-axis
            x0=0, x1=1,  # span full width
            y0=price, y1=price,
            line=dict(color="gray", dash="dash", width=1),
        )
        for price in prices_values
    ] + [
        dict(
            type="line",
            xref="paper",
            yref="y",
            x0=0, x1=1,
            y0=vwap_timeline[-1][1], y1=vwap_timeline[-1][1],
            line=dict(color="green", width=2)
        )
    ]
    trace_vwap_line = go.Scatter(
        x=[t for t, _ in vwap_timeline],
        y=[v for _, v in vwap_timeline],
        mode="lines+markers",
        name="VWAP (accum)",
        line=dict(color="orange"),
        xaxis="x3",
        yaxis="y3"
    )

    return go.Frame(
        data=[trace_close, trace_volume_dist, trace_vwap_line],
        name=str(window.index[0]),
        layout=go.Layout(shapes=shapes)
    )


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    sys.path.append(root)
    conf = {
        "peak_prominence": 0.1,
        "first_vwap_std": 1.,
        "second_vwap_std": 1.5,
        "third_vwap_std": 2.,
        "window_size": 60 * 24 * 14,  # 14 days
        "window_step": 60 * 4,  # 4 hours
        "candles_config": {
            "connector_name": "binance",
            "trading_pair": "USDT-BRL",
            "interval": "1m",
            "start_time": datetime(2025, 3, 1).timestamp(),
            "end_time": datetime(2025, 5, 16).timestamp(),
            "fetch": False,
        },
        "tick_size": 0.001,
        "total_amount_quote": 1000.0,
        "n_bins": 30,
        "only_min_max": True,
    }
    asyncio.run(main(conf, root))
