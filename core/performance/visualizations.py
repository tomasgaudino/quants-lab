import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

from core.data_structures.candles import Candles


class Visualizer:

    @staticmethod
    async def candles_with_pnl(candles: Candles, df: pd.DataFrame, side: int = 1):
        # Create a subplot with 2 rows
        side = "Long" if side == 1 else "Short"
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=(f"{side} OHLC Chart with Break-Even Levels", "PnL and Fees Over Time"))

        # ---------------------- FIG 1: Candlestick Chart & Break Even ----------------------
        # OHLC Candlestick Chart
        fig.add_trace(go.Candlestick(name="OHLC",
                                     x=candles.data.index,
                                     open=candles.data["open"],
                                     high=candles.data["high"],
                                     low=candles.data["low"],
                                     close=candles.data["close"]),
                      row=1, col=1)

        # Break Even Open
        fig.add_trace(go.Scatter(name="Break Even Open",
                                 x=df["datetime"],
                                 y=df["break_even_open"],
                                 marker_color="olive",
                                 line_shape="hv"),
                      row=1, col=1)

        # Break Even Close
        fig.add_trace(go.Scatter(name="Break Even Close",
                                 x=df["datetime"],
                                 y=df["break_even_close"],
                                 marker_color="red",
                                 line_shape="hv"),
                      row=1, col=1)

        # Markers for trade positions (buy/sell signals)
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(df["timestamp"], unit="s"),
            y=df["price"],
            mode="markers",
            marker=dict(
                symbol=df["position_multiplier"].apply(lambda x: "triangle-up" if x > 0 else "triangle-down"),
                size=8,
                color="white"
            ),
            showlegend=False),
            row=1, col=1
        )

        # ---------------------- FIG 2: PnL and Fees ----------------------
        # Realized PnL
        fig.add_trace(go.Scatter(x=df.datetime,
                                 y=df.realized_pnl,
                                 name="Realized PnL"),
                      row=2, col=1)

        # Unrealized PnL
        fig.add_trace(go.Scatter(x=df.datetime,
                                 y=df.unrealized_pnl,
                                 name="Unrealized PnL"),
                      row=2, col=1)

        # Global PnL (filled area)
        fig.add_trace(go.Scatter(x=df.datetime,
                                 y=df.global_pnl,
                                 line_shape="hv",
                                 fill="tozeroy",
                                 name="Global PnL"),
                      row=2, col=1)

        # Cumulative Fee Paid
        fig.add_trace(go.Scatter(x=df.datetime,
                                 y=df.cumulative_fee_paid_quote.cumsum(),
                                 line_shape="hv",
                                 name="Cumulative Fee"),
                      row=2, col=1)

        # ---------------------- Layout Adjustments ----------------------
        fig.update_layout(
            height=1000,
            xaxis_rangeslider_visible=False,  # Remove range slider from first plot
            showlegend=True,
            title_text="Trading Performance Overview",
            xaxis2=dict(title="Time")  # Label x-axis only for second row
        )
        return fig

    @staticmethod
    def instance_gantt(trades_df: pd.DataFrame, executors_df: pd.DataFrame, side: int = 1):
        agg_trades_df = (
            trades_df
            .groupby(["trading_pair", "db_name"], as_index=False)
            .agg(min_timestamp=("timestamp", "min"), max_timestamp=("timestamp", "max"))
        )
        agg_executors_df = (
            executors_df
            .groupby("db_name", as_index=False)
            .agg(net_pnl_quote=("net_pnl_quote", "sum"), total_volume=("filled_amount_quote", "sum"))
        )
        global_performance = agg_trades_df.merge(agg_executors_df, on="db_name")
        global_performance["start_datetime"] = pd.to_datetime(global_performance["min_timestamp"], unit="s")
        global_performance["end_datetime"] = pd.to_datetime(global_performance["max_timestamp"], unit="s")

        # Crear Gantt chart dataframe format
        global_performance.sort_values(by="start_datetime", inplace=True)
        gantt_data = [
            dict(Task=row["db_name"], Start=row["start_datetime"], Finish=row["end_datetime"])
            for _, row in global_performance.iterrows()
        ]

        # Extraer valores únicos para definir colores
        unique_tasks = list(set(row["db_name"] for _, row in global_performance.iterrows()))

        # Generar una lista de colores lo suficientemente grande
        colors = px.colors.qualitative.Set3  # Usa una paleta con suficiente variedad
        if len(colors) < len(unique_tasks):
            colors = px.colors.qualitative.Alphabet  # Usa más colores si es necesario

        # Crear un diccionario que asigne colores a cada tarea
        color_dict = {task: colors[i % len(colors)] for i, task in enumerate(unique_tasks)}

        # Crear Gantt chart con colores dinámicos
        gantt_fig = ff.create_gantt(
            gantt_data,
            index_col="Task",
            colors=color_dict,  # Agregar colores dinámicos
            show_colorbar=False,
            group_tasks=True,
            showgrid_x=True,
            showgrid_y=True
        )

        # Create subplots with 3 rows
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.5, 0.25, 0.25],  # 50%-25%-25%
            shared_xaxes=True,
            subplot_titles=[
                "Bot History Gantt",
                "Cumulative PNL Over Time",
                "Cumulative Volume Over Time"
            ],
            vertical_spacing=0.1
        )

        # Add Gantt chart traces
        for trace in gantt_fig.data:
            fig.add_trace(trace, row=1, col=1)

        global_performance.sort_values(by="end_datetime", inplace=True)
        # Add cumulative PNL scatter plot
        fig.add_trace(
            go.Scatter(
                x=global_performance["end_datetime"],
                y=global_performance["net_pnl_quote"].cumsum(),
                mode="lines+markers",
                name="Cumulative PNL",
                line=dict(color="purple")
            ),
            row=2, col=1
        )

        # Add cumulative Volume scatter plot
        fig.add_trace(
            go.Scatter(
                x=global_performance["end_datetime"],
                y=global_performance["total_volume"].cumsum(),
                mode="lines+markers",
                name="Cumulative Volume",
                line=dict(color="orange")
            ),
            row=3, col=1
        )

        # Update layout with annotations
        fig.update_layout(
            height=900,
            title_text="Bot Performance Overview",
            xaxis3_title="End DateTime",
            showlegend=False,
        )
        return fig

    @staticmethod
    def volume_treemap(trades_df: pd.DataFrame):
        # Filter data
        df = trades_df.copy()

        # Round the volume column to integers
        df["quote_amount"] = df["quote_amount"].round(0).astype(int)

        # Create the treemap
        fig = px.treemap(
            df,
            path=["trading_pair"],  # No hierarchy, just trading pairs
            values="quote_amount",
            color="quote_amount",
            color_continuous_scale="viridis"
        )

        # Customize text to show volume without decimals
        fig.update_traces(
            texttemplate="<b>%{label}</b><br>Vol: %{value:,}"
        )

        # Optimize layout to reduce empty space
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),  # Remove extra margins
            autosize=True,
            height=600,  # Adjust height
            width=800  # Adjust width
        )
        return fig

