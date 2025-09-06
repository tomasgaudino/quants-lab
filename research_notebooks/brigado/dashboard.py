import asyncio
import io
from datetime import datetime
from typing import List, Dict, Any

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


# Custom styled metric cards
def styled_metric_card(title: str, value: str, delta: str = None, color: str = "blue"):
    """Create a styled metric card with conditional colors and rounded corners."""
    
    # Color schemes for different metric types
    color_schemes = {
        "green": {
            "bg_color": "rgba(34, 197, 94, 0.1)",
            "border_color": "rgba(34, 197, 94, 0.3)",
            "text_color": "#059669",
            "value_color": "#047857"
        },
        "red": {
            "bg_color": "rgba(239, 68, 68, 0.1)",
            "border_color": "rgba(239, 68, 68, 0.3)",
            "text_color": "#DC2626",
            "value_color": "#B91C1C"
        },
        "blue": {
            "bg_color": "rgba(59, 130, 246, 0.1)",
            "border_color": "rgba(59, 130, 246, 0.3)",
            "text_color": "#2563EB",
            "value_color": "#1D4ED8"
        },
        "purple": {
            "bg_color": "rgba(147, 51, 234, 0.1)",
            "border_color": "rgba(147, 51, 234, 0.3)",
            "text_color": "#7C3AED",
            "value_color": "#6D28D9"
        },
        "orange": {
            "bg_color": "rgba(249, 115, 22, 0.1)",
            "border_color": "rgba(249, 115, 22, 0.3)",
            "text_color": "#EA580C",
            "value_color": "#C2410C"
        },
        "teal": {
            "bg_color": "rgba(20, 184, 166, 0.1)",
            "border_color": "rgba(20, 184, 166, 0.3)",
            "text_color": "#0D9488",
            "value_color": "#0F766E"
        }
    }
    
    scheme = color_schemes.get(color, color_schemes["blue"])
    
    delta_html = ""
    if delta:
        delta_color = "#059669" if not delta.startswith("-") else "#DC2626"
        delta_html = f"<p style='margin: 0; font-size: 0.875rem; color: {delta_color}; font-weight: 500;'>{delta}</p>"
    
    card_html = f"""
    <div style='
        background: {scheme["bg_color"]};
        border: 2px solid {scheme["border_color"]};
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    '>
        <h4 style='
            margin: 0 0 0.5rem 0;
            font-size: 0.875rem;
            color: {scheme["text_color"]};
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        '>{title}</h4>
        <p style='
            margin: 0;
            font-size: 1.875rem;
            font-weight: 700;
            color: {scheme["value_color"]};
            line-height: 1;
        '>{value}</p>
        {delta_html}
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)


def get_metric_color(value: float, metric_type: str) -> str:
    """Determine color based on metric value and type."""
    if metric_type == "pnl":
        return "green" if value > 0 else "red" if value < 0 else "blue"
    elif metric_type == "volume":
        if value > 100000:
            return "green"
        elif value > 50000:
            return "orange"
        else:
            return "red"
    elif metric_type == "ratio":
        if value > 0.8:
            return "green"
        elif value > 0.5:
            return "orange"
        else:
            return "red"
    elif metric_type == "percentage":
        if value > 0.05:  # > 5%
            return "green"
        elif value > 0.02:  # > 2%
            return "orange"
        else:
            return "red"
    else:
        return "blue"


# Figures
def volume_target_fig(df: pd.DataFrame):
    fig = go.Figure()
    
    # Add target volume bar
    fig.add_trace(
        go.Bar(
            name="Daily 1% Target",
            x=df["date"],
            y=df["target_0.01"],
            marker_color="rgba(99, 110, 250, 0.6)",
            hovertemplate="<b>Target Volume</b><br>Date: %{x}<br>Volume: $%{y:,.2f}<extra></extra>"
        )
    )

    # Add bot volume bar
    fig.add_trace(
        go.Bar(
            name="Bot Daily Volume",
            x=df["date"],
            y=df["total_usdt_volume"],
            marker_color="rgba(0, 212, 170, 0.8)",
            hovertemplate="<b>Bot Volume</b><br>Date: %{x}<br>Volume: $%{y:,.2f}<extra></extra>"
        )
    )

    # Add market share line
    fig.add_trace(
        go.Scatter(
            name="Market Share",
            x=df["date"],
            y=df["market_participation"],
            mode="lines+markers",
            yaxis="y2",
            line=dict(color="#FF6B6B", width=3),
            marker=dict(size=6, color="#FF6B6B"),
            hovertemplate="<b>Market Share</b><br>Date: %{x}<br>Share: %{y:.2%}<extra></extra>"
        )
    )

    fig.update_layout(
        title={
            'text': "Bot Volume Performance vs Market Target",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128, 128, 128, 0.2)"
        ),
        yaxis=dict(
            title="Volume (USDT)",
            tickformat="$,.0f",
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128, 128, 128, 0.2)"
        ),
        yaxis2=dict(
            title="Market Share (%)",
            overlaying="y",
            side="right",
            tickformat=".2%",
            showgrid=False
        ),
        barmode="group",
        height=500,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig


@st.cache_data
def generate_stats_df(db_names: List[str], root_path: str) -> pd.DataFrame:
    """Generate stats dataframe with caching support."""
    stats = []

    # Get dbs from local storage
    dbs = []
    for db_name in db_names:
        if db_name.endswith(".sqlite"):
            db = HummingbotDatabase(db_name=db_name, server_name="brigado_server", root_path=root_path)
            dbs.append(db)

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

    problematic_dbs = [db.db_name for db in dbs if db.status["trade_fill"] != "Correct"]
    if problematic_dbs:
        st.warning(f"We found problems in the following databases: {problematic_dbs}")

    # Expand stats info
    return pd.DataFrame(stats)


# 1) Cacheá el recurso (cliente) — objetos no serializables van en cache_resource
@st.cache_resource
def get_clob_client():
    return CLOBDataSource()

def _make_key(exchanges: List[str], trading_pairs: List[str], start_date: str) -> tuple:
    # Clave hashable para el caché en session_state
    return (tuple(sorted(exchanges)), tuple(sorted(trading_pairs)), str(start_date))

def _df_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()

def _parquet_bytes_to_df(b: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(b))

async def get_exchange_data(exchanges: List[str], trading_pairs: List[str], start_date: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Nota: NO decorada con st.cache_data por ser async.
    El caché se maneja manualmente en st.session_state.
    """
    key = _make_key(exchanges, trading_pairs, start_date)

    # 2) Si ya está en caché, devolvémoslo (reconstruyendo los DataFrames)
    if "exchange_cache" in st.session_state and key in st.session_state["exchange_cache"]:
        cached = st.session_state["exchange_cache"][key]
        # Reconstruir DFs desde bytes
        result: Dict[str, List[Dict[str, Any]]] = {}
        for exchange, items in cached.items():
            result[exchange] = []
            for it in items:
                result[exchange].append({
                    "trading_pair": it["trading_pair"],
                    "candles_df": _parquet_bytes_to_df(it["candles_df_parquet"]),
                })
        return result

    # 3) Si no está en caché, lo generamos
    days = (datetime.now() - pd.to_datetime(start_date)).days
    interval = "1d"
    clob = get_clob_client()

    exchange_data: Dict[str, List[Dict[str, Any]]] = {exchange: [] for exchange in exchanges}

    # Podés paralelizar por exchange con gather si tu cliente lo permite
    for exchange in exchanges:
        candles = await clob.get_candles_batch_last_days(
            connector_name=exchange,
            trading_pairs=trading_pairs,
            interval=interval,
            days=days
        )
        for trading_pair in trading_pairs:
            # Seleccionar el df correspondiente al par
            candles_df = [candle.data for candle in candles if candle.trading_pair == trading_pair][0].copy()
            candles_df["date"] = pd.to_datetime(candles_df["timestamp"], unit="s").dt.strftime("%Y-%m-%d")
            exchange_data[exchange].append(
                {
                    "candles_df": candles_df,
                    "trading_pair": trading_pair,
                }
            )

    # 4) Guardamos en caché (como bytes Parquet) para que sea serializable
    serializable_cache: Dict[str, List[Dict[str, Any]]] = {}
    for exchange, items in exchange_data.items():
        serializable_cache[exchange] = []
        for it in items:
            serializable_cache[exchange].append({
                "trading_pair": it["trading_pair"],
                "candles_df_parquet": _df_to_parquet_bytes(it["candles_df"]),
            })

    if "exchange_cache" not in st.session_state:
        st.session_state["exchange_cache"] = {}
    st.session_state["exchange_cache"][key] = serializable_cache

    return exchange_data

async def main(root_path: str):
    st.set_page_config(
        page_title="Brigado Trading Dashboard",
        page_icon="🇧🇷",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Main title with emoji and subtitle
    st.markdown(
        """
        # 🇧🇷 Brigado Trading Dashboard
        ### Real-time Bot Performance Analytics
        """
    )

    # Sidebar with key metrics and controls
    with st.sidebar:
        st.header("📊 Dashboard Controls")
        
        # Cache management
        with st.expander("🔄 Data Management", expanded=False):
            st.info("Stats data cached in memory for better performance")
            if st.button("Refresh Data", help="Clear cache and reload data", use_container_width=True):
                generate_stats_df.clear()
                st.rerun()
        
        db_names = [path for path in os.listdir(os.path.join(root_path, "data", "live_bot_databases", "brigado_server")) if
                    path != ".gitignore"]

        # Generate stats dataframe with caching
        stats_df = generate_stats_df(db_names, root_path)
        
        # Key metrics in sidebar with styled cards
        st.markdown("---")
        st.subheader("📈 Key Metrics")
        
        total_pnl = stats_df['pnl_usdt'].sum()
        total_volume = stats_df['total_volume_usdt'].sum()
        num_bots = len(stats_df)
        avg_pnl_per_bot = total_pnl / num_bots if num_bots > 0 else 0
        profitable_bots = len(stats_df[stats_df['pnl_usdt'] > 0])
        profitability_ratio = profitable_bots / num_bots if num_bots > 0 else 0
        
        # Total PnL with conditional coloring
        pnl_color = get_metric_color(total_pnl, "pnl")
        pnl_delta = f"+${total_pnl:,.2f}" if total_pnl > 0 else f"${total_pnl:,.2f}" if total_pnl < 0 else "$0.00"
        styled_metric_card("Total PnL", f"${abs(total_pnl):,.2f}", pnl_delta, pnl_color)
        
        # Total Volume
        volume_color = get_metric_color(total_volume, "volume")
        styled_metric_card("Total Volume", f"${total_volume:,.0f}", f"{num_bots} active bots", volume_color)
        
        # Profitability Ratio
        ratio_color = get_metric_color(profitability_ratio, "ratio")
        styled_metric_card("Profitable Bots", f"{profitable_bots}/{num_bots}", f"{profitability_ratio:.1%} success rate", ratio_color)
        
        # Average PnL per Bot
        avg_pnl_color = get_metric_color(avg_pnl_per_bot, "pnl")
        avg_delta = f"+${avg_pnl_per_bot:,.2f}" if avg_pnl_per_bot > 0 else f"${avg_pnl_per_bot:,.2f}" if avg_pnl_per_bot < 0 else "$0.00"
        styled_metric_card("Avg PnL/Bot", f"${abs(avg_pnl_per_bot):,.2f}", avg_delta, avg_pnl_color)

    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Performance", "⚙️ Configuration Analysis", "📋 Data Tables"])
    
    # Process data
    df_expanded = (
        stats_df
        .set_index(["config_file_path", "exchange", "trading_pair"])
        ["daily_quote_volume"]
        .apply(pd.Series)
        .stack()
        .reset_index()
        .rename(columns={"level_3": "date", 0: "total_usdt_volume"})
    )

    # Generate exchange data with caching
    exchanges = list(df_expanded["exchange"].unique())
    trading_pairs = list(df_expanded["trading_pair"].unique())
    start_date = df_expanded["date"].min()

    exchange_data = await get_exchange_data(exchanges, trading_pairs, start_date)
    
    # Add activity data to exchange_data (not cached since it depends on stats_df)
    for exchange in exchanges:
        for data in exchange_data[exchange]:
            data["activity"] = df_expanded[df_expanded["exchange"] == exchange]
    
    # Exchange and pair selection in sidebar
    with st.sidebar:
        st.markdown("---")
        st.subheader("🎯 Selection")
        exchange = st.selectbox("Exchange", exchanges, index=0)
        available_pairs = [data["trading_pair"] for data in exchange_data[exchange]]
        trading_pair = st.selectbox("Trading Pair", available_pairs, index=0)
    
    with tab1:
        st.header("Trading Overview")
        
        data = [data for data in exchange_data[exchange] if data["trading_pair"] == trading_pair][0]
        bot_daily_volume = data["activity"].groupby("date")["total_usdt_volume"].sum().reset_index()
        daily_volume_df = data["candles_df"].groupby("date")["volume"].sum().reset_index()
        
        overall_volume_df = daily_volume_df.merge(bot_daily_volume, on="date", how="left")
        overall_volume_df["target_0.01"] = overall_volume_df["volume"] * 0.01
        overall_volume_df["market_participation"] = overall_volume_df["total_usdt_volume"] / overall_volume_df["volume"]
        total_bot_volume = overall_volume_df["total_usdt_volume"].sum()
        
        # Key metrics row with styled cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            volume_color = get_metric_color(total_bot_volume, "volume")
            styled_metric_card("Bot Volume", f"${total_bot_volume:,.0f}", f"{exchange.upper()}", volume_color)
        
        with col2:
            avg_participation = overall_volume_df["market_participation"].mean()
            participation_color = get_metric_color(avg_participation, "percentage")
            styled_metric_card("Avg Market Share", f"{avg_participation:.2%}", "Daily average", participation_color)
        
        with col3:
            max_participation = overall_volume_df["market_participation"].max()
            max_color = get_metric_color(max_participation, "percentage")
            styled_metric_card("Peak Market Share", f"{max_participation:.2%}", "Best day", max_color)
        
        with col4:
            trading_days = len(overall_volume_df)
            styled_metric_card("Trading Days", f"{trading_days}", f"{trading_pair}", "teal")
        
        # Volume target chart
        st.plotly_chart(volume_target_fig(overall_volume_df), use_container_width=True)
        
        # Price chart
        st.subheader(f"Price Chart - {trading_pair}")
        candles_fig = go.Figure()
        candles_fig.add_trace(
            go.Candlestick(
                x=data["candles_df"]["date"],
                open=data["candles_df"]["open"],
                high=data["candles_df"]["high"],
                low=data["candles_df"]["low"],
                close=data["candles_df"]["close"],
                name="Price"
            )
        )
        candles_fig.update_layout(
            title=f"{exchange.upper()} {trading_pair} Price Chart",
            yaxis_title="Price",
            xaxis_title="Date",
            height=400
        )
        st.plotly_chart(candles_fig, use_container_width=True)
    
    with tab2:
        st.header("Performance Analysis")
        
        # Performance metrics with styled cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            profitable_bots = len(stats_df[stats_df['pnl_usdt'] > 0])
            profit_ratio = profitable_bots / num_bots
            ratio_color = get_metric_color(profit_ratio, "ratio")
            styled_metric_card("Profitable Bots", f"{profitable_bots}/{num_bots}", f"{profit_ratio:.1%} success rate", ratio_color)
            
        with col2:
            best_performer = stats_df.loc[stats_df['pnl_usdt'].idxmax()]
            best_pnl = best_performer['pnl_usdt']
            best_color = get_metric_color(best_pnl, "pnl")
            styled_metric_card("Best Performer", f"${best_pnl:,.2f}", f"{best_performer['exchange']} - {best_performer['trading_pair']}", best_color)
            
        with col3:
            worst_performer = stats_df.loc[stats_df['pnl_usdt'].idxmin()]
            worst_pnl = worst_performer['pnl_usdt']
            worst_color = get_metric_color(worst_pnl, "pnl")
            styled_metric_card("Worst Performer", f"${abs(worst_pnl):,.2f}", f"{worst_performer['exchange']} - {worst_performer['trading_pair']}", worst_color)
        
        # PnL distribution
        pnl_fig = px.histogram(
            stats_df, 
            x='pnl_usdt', 
            title='PnL Distribution',
            nbins=20,
            color_discrete_sequence=['#00D4AA']
        )
        pnl_fig.update_layout(height=400)
        st.plotly_chart(pnl_fig, use_container_width=True)
        
        # Volume vs PnL scatter
        scatter_fig = px.scatter(
            stats_df,
            x='total_volume_usdt',
            y='pnl_usdt',
            color='exchange',
            size='total_volume_usdt',
            title='Volume vs PnL Analysis',
            hover_data=['trading_pair']
        )
        scatter_fig.update_layout(height=400)
        st.plotly_chart(scatter_fig, use_container_width=True)
    
    with tab3:
        st.header("Configuration Analysis")
        
        # Parallel Coordinates Plot
        plain_stats_df = normalize_config(stats_df)
        controller_names = list(plain_stats_df["controller_name"].unique())
        controller_name = st.selectbox("Select Controller Type", controller_names)

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

        parallel_fig = px.parallel_coordinates(
            controller_df, 
            color="pnl_volume_ratio", 
            dimensions=controller_df.columns,
            title=f"Configuration Analysis - {controller_name}"
        )
        parallel_fig.update_layout(height=600)
        st.plotly_chart(parallel_fig, use_container_width=True)
    
    with tab4:
        st.header("Data Tables")
        
        # Bot performance table
        st.subheader("Bot Performance Summary")
        display_df = stats_df.copy()
        display_df['pnl_usdt'] = display_df['pnl_usdt'].apply(lambda x: f"${x:,.2f}")
        display_df['total_volume_usdt'] = display_df['total_volume_usdt'].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(
            display_df[['config_file_path', 'exchange', 'trading_pair', 'total_volume_usdt', 'pnl_usdt']],
            use_container_width=True
        )
        
        # Daily volume breakdown
        st.subheader("Daily Volume Breakdown")
        st.dataframe(overall_volume_df, use_container_width=True)


if __name__ == "__main__":
    load_dotenv()
    root_path = os.path.abspath(os.path.join(os.getcwd()))
    sys.path.append(root_path)

    from core.data_sources import CLOBDataSource
    from core.data_sources.hummingbot_database import HummingbotDatabase

    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    logging.getLogger("hummingbot").setLevel(logging.ERROR)

    asyncio.run(main(root_path))
