import asyncio
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import sys
import os
from dotenv import load_dotenv
import logging
import plotly.graph_objects as go
import plotly.express as px


# Controller config variables
BASE_KEEP = ["total_volume_usdt", "pnl_usdt", "config_file_path"]
CONTROLLER_SPECIFIC_COLS = {
    "pmm": {
        "float": [
            "total_amount_quote", "portfolio_allocation", "target_base_pct", "min_base_pct",
            "max_base_pct", "executor_refresh_time", "cooldown_time", "max_skew",
        ],
        "list": ["buy_spreads"],
        "bool": ["tick_mode"],
        "string": ["controller_name", "connector_name", "trading_pair"],
    },
    "pmm_mister": {
        "float": [
            "buy_cooldown_time", "sell_cooldown_time", "buy_position_effectivization_time",
            "sell_position_effectivization_time", "min_buy_price_distance_pct",
            "min_sell_price_distance_pct", "breakeven_buffer_pct", "dynamic_cooldown_multiplier",
            "max_active_executors_by_level",
        ],
        "list": ["buy_spreads"],
        "bool": ["tick_mode"],
        "string": ["controller_name", "connector_name", "trading_pair"]
    }
}


# Helpers
def normalize_config(df: pd.DataFrame, col: str = "config") -> pd.DataFrame:
    """Aplana el dict de 'config' en columnas nuevas."""
    cfg = pd.json_normalize(df[col].fillna({}))
    cfg.index = df.index  # alinea por índice
    return pd.concat([df.drop(columns=[col]), cfg], axis=1)


def flatter(list_of_lists: list) -> list:
    flat = []
    for item in list_of_lists:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


# Figures
def volume_target_fig(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Daily 1% Target",
            x=df["date"],
            y=df["target_0.01"]
        )
    )

    fig.add_trace(
        go.Bar(
            name="Bot Daily Volume",
            x=df["date"],
            y=df["total_usdt_volume"],
            marker_color="lime"
        )
    )

    fig.add_trace(
        go.Scatter(
            name="Total Market Share",
            x=df["date"],
            y=df["market_participation"],
            mode="lines+markers",
            yaxis="y2",  # 👈 put this trace on the right axis
            line=dict(color="white", width=2)
        )
    )

    fig.update_layout(
        yaxis=dict(
            title="Volume (USDT)"
        ),
        yaxis2=dict(
            title="Market Share (%)",
            overlaying="y",  # share same x
            side="right",
            tickformat=".2%"  # format as percentage
        ),
        barmode="group",
        height=800
    )
    return fig


async def main():
    st.set_page_config(page_title="Brigado!", page_icon="🇧🇷", layout="wide")
    st.title("Brigado!")

    db_names = [path for path in os.listdir(os.path.join(root_path, "data", "live_bot_databases", "brigado_server")) if
                path != ".gitignore"]

    # Get dbs from local storage
    dbs = []
    for db_name in db_names:
        if db_name.endswith(".sqlite"):
            db = HummingbotDatabase(db_name=db_name, server_name="brigado_server", root_path=root_path)
            dbs.append(db)

    stats = []

    # Extract trades and controllers from healthy dbs
    for db in dbs:
        if db.status["trade_fill"] == "Correct":
            # Trades
            trades_df = db.get_trade_fills()
            trades_df["date"] = pd.to_datetime(trades_df["timestamp"]).dt.strftime("%Y-%m-%d")
            first_row = trades_df.iloc[0]

            # Controllers
            controller_df = db.get_controller_data()
            if len(controller_df) == 0:
                config = None
            else:
                if len(controller_df) > 1:
                    print(f"{db.db_name}: Found {len(controller_df)} controllers, only keeping first config found. Please develop multicontroller :P")
                config = controller_df["config"][0]

            stats_dict = {
                "config_file_path": first_row["config_file_path"],
                "exchange": first_row["market"],
                "trading_pair": first_row["symbol"],
                "daily_quote_volume": trades_df.groupby("date")["amount"].sum().to_dict(),
                "total_volume_usdt": trades_df["amount"].sum(),
                "pnl_usdt": trades_df["net_realized_pnl"].iloc[-1],
                "config": config,
            }
            stats.append(stats_dict)

    st.warning(f"We found problems in the following databases: {[db.db_name for db in dbs if db.status["trade_fill"] != "Correct"]}")

    # Expand stats info
    stats_df = pd.DataFrame(stats)

    df_expanded = (
        stats_df
        .set_index(["config_file_path", "exchange", "trading_pair"])
        ["daily_quote_volume"]
        .apply(pd.Series)
        .stack()
        .reset_index()
        .rename(columns={"level_3": "date", 0: "total_usdt_volume"})
    )

    # Generate exchange data JSON
    exchanges = list(df_expanded["exchange"].unique())
    trading_pairs = list(df_expanded["trading_pair"].unique())
    interval = "1d"
    start_date = df_expanded["date"].min()
    days = (datetime.now() - pd.to_datetime(start_date)).days

    exchange_data = {exchange: [] for exchange in exchanges}
    for exchange in exchanges:
        clob = CLOBDataSource()
        candles = await clob.get_candles_batch_last_days(connector_name=exchange,
                                                         trading_pairs=trading_pairs,
                                                         interval=interval,
                                                         days=days)
        for trading_pair in trading_pairs:
            candles_df = [candle.data for candle in candles if candle.trading_pair == trading_pair][0].copy()
            candles_df["date"] = pd.to_datetime(candles_df["timestamp"], unit="s").dt.strftime("%Y-%m-%d")
            exchange_data[exchange].append(
                {
                    "candles_df": candles_df,
                    "activity": df_expanded[df_expanded["exchange"] == exchange],
                    "trading_pair": trading_pair,
                }
            )

    # Volume target figure
    exchange = "binance"
    trading_pair = "USDT-BRL"

    data = [data for data in exchange_data[exchange] if data["trading_pair"] == trading_pair][0]
    bot_daily_volume = data["activity"].groupby("date")["total_usdt_volume"].sum().reset_index()

    daily_volume_df = data["candles_df"].groupby("date")["volume"].sum().reset_index()

    candles_fig = go.Figure()
    candles_fig.add_trace(
        go.Candlestick(x=data["candles_df"]["date"],
                       open=data["candles_df"]["open"],
                       high=data["candles_df"]["high"],
                       low=data["candles_df"]["low"],
                       close=data["candles_df"]["close"]
                       )
    )

    overall_volume_df = daily_volume_df.merge(bot_daily_volume, on="date", how="left")
    overall_volume_df["target_0.01"] = overall_volume_df["volume"] * 0.01
    overall_volume_df["market_participation"] = overall_volume_df["total_usdt_volume"] / overall_volume_df["volume"]
    total_bot_volume = overall_volume_df["total_usdt_volume"].sum()

    # Layout
    st.metric("Overall Bot Volume (USDT):", f"$ {total_bot_volume:.2f})")
    st.plotly_chart(volume_target_fig(overall_volume_df), use_container_width=True)

    # Parallel Coordinates Plot
    plain_stats_df = normalize_config(stats_df)
    controller_names = list(plain_stats_df["controller_name"].unique())
    controller_name = st.selectbox("Controller Names", controller_names)

    plain_stats_df_filtered = plain_stats_df[plain_stats_df["controller_name"] == controller_name]
    controller_columns = flatter(
        BASE_KEEP + [columns for dtype, columns in CONTROLLER_SPECIFIC_COLS[controller_name].items()])
    controller_df = plain_stats_df_filtered[controller_columns].copy()
    controller_df = controller_df.loc[:, ~controller_df.columns.duplicated(keep=False)]

    for col in CONTROLLER_SPECIFIC_COLS[controller_name]["list"]:
        controller_df.loc[:, col + "_max"] = controller_df[col].apply(lambda x: max(x))
        controller_df.loc[:, col + "_n_levels"] = controller_df[col].apply(lambda x: len(x))
        controller_df.loc[:, col + "_avg_spread"] = controller_df[col].apply(lambda x: float(np.nanmean(np.diff(x))))
        controller_df.drop(columns=[col], inplace=True)

    for col in CONTROLLER_SPECIFIC_COLS[controller_name]["float"]:
        controller_df[col] = controller_df[col].astype(float)

    controller_df["pnl_volume_ratio"] = controller_df["pnl_usdt"] / controller_df["total_volume_usdt"]

    parallel_fig = px.parallel_coordinates(controller_df, color="pnl_volume_ratio", dimensions=controller_df.columns)
    st.plotly_chart(parallel_fig, use_container_width=True)


if __name__ == "__main__":
    load_dotenv()
    root_path = os.path.abspath(os.path.join(os.getcwd()))
    sys.path.append(root_path)

    from core.data_sources import CLOBDataSource
    from core.data_sources.hummingbot_database import HummingbotDatabase

    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    logging.getLogger("hummingbot").setLevel(logging.ERROR)

    asyncio.run(main())
