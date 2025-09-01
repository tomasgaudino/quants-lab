import asyncio
from datetime import datetime

import streamlit as st
import sys
import os
from dotenv import load_dotenv
import logging
import plotly.graph_objects as go


async def calculate_volume_and_target_by_market(rolling_volume,
                                                trading_pair: str,
                                                first_target: float,
                                                last_target: float):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        name="30d rolling mean",
        x=rolling_volume.index,
        y=rolling_volume.values)
    )
    fig.add_trace(go.Scatter(
        name="0.5% target",
        x=rolling_volume.index,
        y=rolling_volume.values * first_target)
    )
    fig.add_trace(go.Scatter(
        name="1% target",
        x=rolling_volume.index,
        y=rolling_volume.values * last_target)
    )

    fig.update_layout(
        yaxis=dict(type='log'),
        title=f'[{trading_pair}] 30d Rolling Volume with Targets',
        xaxis_title='Date',
        yaxis_title='Volume (log scale)'
    )
    return fig


async def main():
    st.set_page_config(page_title="Brigado!", page_icon="🇧🇷", layout="wide")
    st.title("Brigado!")
    candles_dict = {}
    connector_name = "binance"
    trading_pairs = ["BTC-BRL", "USDT-BRL", "SOL-BRL"]
    interval = "1d"
    start_timestamp = int(datetime(2025, 1, 1).timestamp())
    end_timestamp = int(datetime.now().timestamp())
    first_target = 0.005
    last_target = 0.01

    for trading_pair in trading_pairs:
        clob = CLOBDataSource()
        candles = await clob.get_candles(connector_name, trading_pair, interval, start_timestamp, end_timestamp)
        candles_dict[trading_pair] = candles
        col1, col2 = st.columns([4, 1])
        with col1:
            volume_tab, candles_tab = st.tabs(["Volume targets", "Candles"])
            with volume_tab:
                rolling_volume = candles.data["volume"].rolling(30).mean().dropna()
                fig = await calculate_volume_and_target_by_market(rolling_volume, trading_pair, first_target, last_target)
                st.plotly_chart(fig, use_container_width=True)
            with candles_tab:
                fig = go.Figure(go.Candlestick(x=candles.data.index,
                                               open=candles.data["open"],
                                               high=candles.data["high"],
                                               low=candles.data["low"],
                                               close=candles.data["close"]))
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.metric(f"Last {100 * first_target:.2f}% target", f"{rolling_volume.iloc[-1] * first_target:.2f}")
            st.metric(f"Last {100 * last_target:.2f}% target", f"{rolling_volume.iloc[-1] * last_target:.2f}")


if __name__ == "__main__":
    load_dotenv()
    root_path = os.path.abspath(os.path.join(os.getcwd()))
    sys.path.append(root_path)

    from core.data_sources import CLOBDataSource

    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    logging.getLogger("hummingbot").setLevel(logging.ERROR)

    asyncio.run(main())
