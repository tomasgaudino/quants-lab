import asyncio
import json
from datetime import datetime
from typing import List
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
import pandas as pd
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo
from hummingbot_api_client import HummingbotAPIClient
import os
import sys


class StrategyBlock:
    def __init__(self, start_ts: float, end_ts: float, duration_s: float, rank_label: str):
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.duration_s = duration_s
        self.rank_label = rank_label
        self.df_executors = pd.DataFrame()
        self.df_trades = pd.DataFrame()
        self.df_candles = pd.DataFrame()
        self.metrics = {}

    def load_order_book_data(self, orderbook_snapshots: List[dict]):
        """Filter order book snapshots within the block's time range."""
        filtered_snapshots = [
            snapshot for snapshot in orderbook_snapshots
            if self.start_ts <= snapshot['ts'] <= self.end_ts
        ]
        self.orderbook_snapshots = filtered_snapshots

    def load_trades_data(self, df_trades: pd.DataFrame):
        """Filter trades within the block's time range."""
        if not df_trades.empty:
            self.df_trades = df_trades[
                (df_trades['ts'] >= self.start_ts) &
                (df_trades['ts'] <= self.end_ts)
            ].copy()

    def load_executors_data(self, df_executors: pd.DataFrame):
        """Filter executors that overlap with the block's time range."""
        if not df_executors.empty:
            self.df_executors = df_executors[
                (df_executors['timestamp'] <= self.end_ts) &
                ((df_executors['close_timestamp'].isna()) | (df_executors['close_timestamp'] >= self.start_ts))
            ].copy()

    def load_candles_data(self, df_candles: pd.DataFrame):
        """Filter candles within the block's time range."""
        if not df_candles.empty:
            # Assuming df_candles has a timestamp column (adjust column name if needed)
            timestamp_col = 'timestamp' if 'timestamp' in df_candles.columns else df_candles.columns[0]
            self.df_candles = df_candles[
                (df_candles[timestamp_col] >= self.start_ts) &
                (df_candles[timestamp_col] <= self.end_ts)
            ].copy()


class StrategyRuns:
    def __init__(self, host: str, port: int = 8000, root_path: str = ""):
        self.client: HummingbotAPIClient = HummingbotAPIClient(base_url=f"http://{host}:{port}")
        self.root_path = root_path
        self.dbs: List[str] = []
        self.executors_info: List[ExecutorInfo] = []
        self.df_candles: pd.DataFrame = pd.DataFrame()
        self.orderbook_snapshots: List[dict] = []
        self.df_trades: pd.DataFrame = pd.DataFrame()
        self.blocks: List[StrategyBlock] = []

    async def init(self):
        await self.client.init()
        self.dbs = await self.client.archived_bots.list_databases()
        print(self.dbs)

    async def fetch_data(self, db_name: str):
        executors_info = []
        start_timestamps = []
        end_timestamps = []
        executors = await self.client.archived_bots.get_database_executors(db_name)
        for executor in executors["executors"]:
            start_timestamps.append(executor['timestamp'])
            end_timestamps.append(executor['close_timestamp'])
            executor["config"] = json.loads(executor["config"])
            executor["custom_info"] = json.loads(executor["custom_info"])
            executors_info.append(ExecutorInfo(**executor))
        self.executors_info = executors_info

        candles = await self.client.market_data.get_historical_candles(connector_name="binance",
                                                                       trading_pair="USDT-BRL",
                                                                       interval="1m",
                                                                       start_time=min(start_timestamps),
                                                                       end_time=max(end_timestamps))
        self.df_candles = pd.DataFrame(candles)

        exchange = "binance"
        trading_pair = "USDT-BRL"
        current_date = datetime.now().strftime("%Y-%m-%d")

        try:
            file_path = f"{exchange}_{trading_pair}_order_book_snapshots_{current_date}.txt"
            full_path = os.path.join(self.root_path, 'data', 'market_data', file_path)
            print(f"Loading data from: {file_path}")
            # Load order book snapshots
            snapshots = []
            with open(full_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        snapshots.append(json.loads(line))

            print(f"Loaded {len(snapshots)} snapshots")
            print(f"Time range: {snapshots[0]['ts']:.2f} to {snapshots[-1]['ts']:.2f}")
            print(f"Duration: {snapshots[-1]['ts'] - snapshots[0]['ts']:.2f} seconds")
        except Exception as e:
            print(f"Error loading order book snapshots: {e}")
            snapshots = []

        self.orderbook_snapshots = snapshots

        try:
            file_path = f"{exchange}_{trading_pair}_trades_{current_date}.txt"
            full_path = os.path.join(self.root_path, 'data', 'market_data', file_path)
            print(f"Loading data from: {file_path}")
            # Load trades records
            trades = []
            with open(full_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        trades.append(json.loads(line))

            print(f"Loaded {len(trades)} trades")
            print(f"Time range: {trades[0]['ts']:.2f} to {trades[-1]['ts']:.2f}")
            print(f"Duration: {trades[-1]['ts'] - trades[0]['ts']:.2f} seconds")
        except Exception as e:
            print(f"Error loading order book snapshots: {e}")
            trades = []

        self.df_trades = pd.DataFrame(trades)

    def get_executors_df(self):
        executors_data = []
        for executor in self.executors_info:
            executors_data.append({
                'id': executor.id,
                'timestamp': executor.timestamp,
                'type': executor.type,
                'status': executor.status.name,
                'trading_pair': executor.config.trading_pair,
                'connector_name': executor.config.connector_name,
                'side': executor.config.side.name,
                'entry_price': float(executor.config.entry_price),
                'amount': float(executor.config.amount),
                'leverage': executor.config.leverage,
                'take_profit': float(
                    executor.config.triple_barrier_config.take_profit) if executor.config.triple_barrier_config.take_profit else None,
                'net_pnl_pct': float(executor.net_pnl_pct),
                'net_pnl_quote': float(executor.net_pnl_quote),
                'cum_fees_quote': float(executor.cum_fees_quote),
                'filled_amount_quote': float(executor.filled_amount_quote),
                'is_active': executor.is_active,
                'is_trading': executor.is_trading,
                'close_timestamp': executor.close_timestamp,
                'close_type': executor.close_type.name if executor.close_type else None,
                'level_id': executor.custom_info.get('level_id'),
                'close_price': executor.custom_info.get('close_price'),
                'custom_info': executor.custom_info,
            })

        df_executors = pd.DataFrame(executors_data)
        df_executors["amount_usdt"] = np.where(df_executors["filled_amount_quote"] == 0, df_executors["amount"],
                                               df_executors["filled_amount_quote"] / df_executors["entry_price"])
        df_executors["amount_brl"] = df_executors["amount_usdt"] * df_executors["entry_price"]
        return df_executors

    @staticmethod
    def group_executors_by_close_type_and_side(df_executors: pd.DataFrame):
        grouped_executors = df_executors.groupby(["close_type", "side"]).agg(
            {"amount_usdt": "sum", "amount_brl": "sum"}).reset_index()
        grouped_executors["breakeven_price"] = grouped_executors["amount_brl"] / grouped_executors['amount_usdt']
        return grouped_executors

    @staticmethod
    def analyze_position_hold_pnl(grouped_executors: pd.DataFrame):

        # Filter only position hold operations
        df_filtered = grouped_executors[grouped_executors.close_type == "POSITION_HOLD"]

        # Calculate percentage difference between breakevens (SELL - BUY)
        breakeven_buy = df_filtered.loc[df_filtered.side == "BUY", "breakeven_price"].mean()
        breakeven_sell = df_filtered.loc[df_filtered.side == "SELL", "breakeven_price"].mean()

        pct_diff = ((breakeven_sell - breakeven_buy) / breakeven_buy) * 100
        direction = "profit" if pct_diff > 0 else "loss"
        color_line = "green" if pct_diff > 0 else "red"

        # Create scatter plot
        fig = px.scatter(
            df_filtered,
            x="side",
            y="breakeven_price",
            size="amount_usdt",
            color="side",
            hover_data=["amount_brl"],
            title="Breakeven price by side (only POSITION_HOLD)",
        )

        # Add line between BUY and SELL breakevens
        fig.add_trace(
            go.Scatter(
                x=["BUY", "SELL"],
                y=[breakeven_buy, breakeven_sell],
                mode="lines+markers",
                line=dict(color=color_line, width=3, dash="dot"),
                name=f"Δ {pct_diff:.2f}% ({direction})",
            )
        )

        # Add annotation in the middle of the line
        fig.add_annotation(
            x=0.5,  # middle between BUY and SELL
            y=(breakeven_buy + breakeven_sell) / 2,
            text=f"Δ {pct_diff:+.2f}% ({direction})",
            showarrow=False,
            font=dict(size=14, color=color_line, family="Arial Black"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=color_line,
            borderwidth=2,
            borderpad=4,
        )

        # Final touches
        fig.update_layout(
            yaxis_title="Breakeven Price",
            xaxis_title="Side",
            showlegend=True,
            template="plotly_dark",
            title_x=0.5,
        )

        return fig

    def build_blocks(self, df_executors: pd.DataFrame):
        blocks = self._get_top_inactivity_stretches(df_executors)
        for block in blocks:
            block.load_order_book_data(self.orderbook_snapshots)
            block.load_trades_data(self.df_trades)
            block.load_executors_data(df_executors)
            block.load_candles_data(self.df_candles)

            # TODO
            # block.metrics = self.calculate_metrics(block)
        self.blocks = blocks

    @staticmethod
    def _get_top_inactivity_stretches(df_executors: pd.DataFrame, n: int = 3):
        """
        Identify the top-n inactivity (EARLY_STOP) stretches in executor data.
        """
        exe = df_executors.sort_values("timestamp").copy()
        exe["is_early"] = exe["close_type"].eq("EARLY_STOP")
        exe["block"] = (exe["is_early"] != exe["is_early"].shift()).cumsum()

        blocks = (
            exe[exe["is_early"]]
            .groupby("block", as_index=False)
            .agg(
                start_ts=("timestamp", "min"),
                end_ts=("close_timestamp", "max"),
            )
        )

        blocks["duration_s"] = blocks["end_ts"] - blocks["start_ts"]
        top_blocks = blocks.nlargest(n, "duration_s").reset_index(drop=True)

        # Handle fewer than n inactivity periods gracefully
        rank_labels = ["Main inactivity", "Second inactivity", "Third inactivity",
                       "Fourth inactivity", "Fifth inactivity"]
        top_blocks["rank_label"] = rank_labels[: len(top_blocks)]
        blocks = []
        for _, row in top_blocks.iterrows():
            blocks.append(StrategyBlock(start_ts=row["start_ts"],
                                        end_ts=row["end_ts"],
                                        duration_s=row["duration_s"],
                                        rank_label=row["rank_label"]))
        return blocks

    @staticmethod
    def _side_color(side, close_type):
        if close_type == "EARLY_STOP":
            return "gray"
        return "olive" if str(side).upper() == "BUY" else "red"

    @staticmethod
    def _line_dash(close_type):
        return "dash" if close_type in ("POSITION_HOLD", "EARLY_STOP") else "solid"

    @staticmethod
    def _align_bot_to_candles(cand_ts_s: np.ndarray,
                              bot_ts_s: np.ndarray,
                              bot_amount: np.ndarray) -> pd.DataFrame:
        """
        Aligns cumulative bot notional to candle timestamps with step/forward-fill.
        Returns a DataFrame with columns: ts_s, ts_dt, cum_bot
        """
        grid = pd.DataFrame({"ts_s": cand_ts_s})
        grid["ts_dt"] = pd.to_datetime(grid["ts_s"], unit="s")

        if bot_ts_s.size == 0:
            grid["cum_bot"] = 0.0
            return grid

        bot = pd.DataFrame({"ts_s": bot_ts_s, "amt": bot_amount}).sort_values("ts_s")
        bot["cum_bot"] = bot["amt"].cumsum()
        bot["ts_dt"] = pd.to_datetime(bot["ts_s"], unit="s")

        aligned = pd.merge_asof(
            grid[["ts_dt"]].sort_values("ts_dt"),
            bot[["ts_dt", "cum_bot"]].sort_values("ts_dt"),
            on="ts_dt", direction="backward"
        )
        aligned["cum_bot"] = aligned["cum_bot"].fillna(0.0)
        return grid.join(aligned["cum_bot"])

    @staticmethod
    def _merge_intervals(intervals):
        """Merge possibly overlapping [start, end] intervals (epoch seconds)."""
        out = []
        for s, e in sorted(intervals, key=lambda x: x[0]):
            if not out or s > out[-1][1]:
                out.append([s, e])
            else:
                out[-1][1] = max(out[-1][1], e)
        return out

    @staticmethod
    def _complement_intervals(range_start, range_end, blocked_intervals):
        """Return complement of blocked_intervals within [range_start, range_end]."""
        active = []
        cur = range_start
        for s, e in blocked_intervals:
            if cur < s:
                active.append((cur, s))
            cur = max(cur, e)
        if cur < range_end:
            active.append((cur, range_end))
        return active

    def plot_execution_and_volume_analysis(self, df_executors: pd.DataFrame):
        """
        Create a dual-panel chart showing:
        - Row 1: Executions overlaid on price
        - Row 2: Standardized cumulative volumes (log scale) comparing bot vs market
        """
        from plotly.subplots import make_subplots

        # Styling constants
        EPS = 1e-9
        MARKET_COLOR = "#FFFFFF"
        BOT_COLOR = "#FFD400"
        FILL_ACTIVE = "rgba(255, 212, 0, 0.12)"
        FILL_INACT = "rgba(255, 255, 255, 0.08)"

        # Create figure
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
            specs=[[{}], [{}]], row_heights=[0.7, 0.3]
        )

        # Row 1: Executions + Price
        for _, ex in df_executors.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[pd.to_datetime(ex["timestamp"], unit="s"),
                       pd.to_datetime(ex["close_timestamp"], unit="s")],
                    y=[ex["entry_price"], ex["close_price"]],
                    mode="lines",
                    line=dict(width=1, dash=self._line_dash(ex["close_type"]),
                              color=self._side_color(ex["side"], ex["close_type"])),
                    name=str(ex["id"]),
                    showlegend=False
                ),
                row=1, col=1
            )

        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(self.df_candles["timestamp"], unit="s"),
                y=self.df_candles["close"],
                mode="lines",
                name="Price"
            ),
            row=1, col=1
        )

        # Shade and label inactivity windows
        for block in self.blocks:
            x0 = pd.to_datetime(block.start_ts, unit="s")
            x1 = pd.to_datetime(block.end_ts, unit="s")
            dur_hours = block.duration_s / 3600.0

            fig.add_vrect(x0=x0, x1=x1, fillcolor="gray", opacity=0.12, layer="below", line_width=0)
            fig.add_annotation(
                x=x0 + (x1 - x0) / 2, y=1.02, yref="paper", showarrow=False,
                text=f"{block.rank_label}  •  {dur_hours:.1f} h",
                font=dict(size=12)
            )

        # Build active windows (complement of blocks within candle range)
        min_ts = int(self.df_candles["timestamp"].min())
        max_ts = int(self.df_candles["timestamp"].max())

        inact = []
        for block in sorted(self.blocks, key=lambda b: b.start_ts):
            s = int(max(min_ts, block.start_ts))
            e = int(min(max_ts, block.end_ts))
            if s < e:
                inact.append([s, e])

        inact = self._merge_intervals(inact)
        active_windows = self._complement_intervals(min_ts, max_ts, inact)

        # Row 2: Standardized cumulative volumes
        legend_once = {"mkt": True, "bot": True}
        y2_max = 1.0

        # ACTIVE windows
        for j, (a0, a1) in enumerate(active_windows, start=1):
            # Market segment
            m_mask = (self.df_candles["timestamp"] >= a0) & (self.df_candles["timestamp"] <= a1)
            m_seg = self.df_candles.loc[m_mask, ["timestamp", "volume"]].sort_values("timestamp").copy()
            if m_seg.empty:
                continue
            m_seg["cum_mkt"] = m_seg["volume"].cumsum()

            # Bot segment
            b_mask = (df_executors["timestamp"] >= a0) & (df_executors["timestamp"] <= a1)
            b_seg = df_executors.loc[b_mask, ["timestamp", "amount_usdt"]].copy()

            aligned = self._align_bot_to_candles(
                cand_ts_s=m_seg["timestamp"].values,
                bot_ts_s=b_seg["timestamp"].values if not b_seg.empty else np.array([], dtype=int),
                bot_amount=b_seg["amount_usdt"].values if not b_seg.empty else np.array([], dtype=float)
            )

            x = pd.to_datetime(m_seg["timestamp"], unit="s")
            y_mkt_true = m_seg["cum_mkt"].values
            y_bot_true = aligned["cum_bot"].values
            y_mkt = y_mkt_true + EPS
            y_bot = y_bot_true + EPS

            y2_max = max(y2_max, float(y_mkt_true.max()), float(y_bot_true.max()))

            # Market line
            fig.add_trace(
                go.Scatter(
                    x=x, y=y_mkt, mode="lines",
                    line=dict(width=2, color=MARKET_COLOR),
                    name="Market • Cum vol",
                    legendgroup="mkt",
                    showlegend=legend_once["mkt"],
                    customdata=y_mkt_true,
                    hovertemplate="<b>Market cum</b><br>Date: %{x}<br>Mkt: %{customdata:,.0f}<extra></extra>"
                ),
                row=2, col=1
            )
            legend_once["mkt"] = False

            # Bot line
            fig.add_trace(
                go.Scatter(
                    x=x, y=y_bot, mode="lines",
                    line=dict(width=2, color=BOT_COLOR, dash="dash"),
                    name="Bot • Cum vol (USDT)",
                    legendgroup="bot",
                    showlegend=legend_once["bot"],
                    fill="tonexty", fillcolor=FILL_ACTIVE,
                    customdata=np.stack([y_bot_true, y_mkt_true], axis=1),
                    hovertemplate="<b>Bot cum</b><br>Date: %{x}"
                                  "<br>Bot: %{customdata[0]:,.0f} USDT"
                                  "<br>Mkt: %{customdata[1]:,.0f}<extra></extra>"
                ),
                row=2, col=1
            )
            legend_once["bot"] = False

            # End marker
            fig.add_trace(
                go.Scatter(
                    x=[x.iloc[-1]], y=[y_bot[-1]],
                    mode="markers",
                    marker=dict(size=7, color=BOT_COLOR, symbol="circle"),
                    hoverinfo="skip", showlegend=False
                ),
                row=2, col=1
            )

            # Annotation
            bot_total = float(y_bot_true[-1])
            mkt_total = float(y_mkt_true[-1])
            pct = 0.0 if mkt_total <= 0 else 100.0 * bot_total / mkt_total

            fig.add_annotation(
                x=x.iloc[-1], xref="x",
                y=1.065, yref="paper",
                text=f"{bot_total:,.0f} USDT • {pct:.1f}% de mercado",
                showarrow=False,
                font=dict(size=11, color=BOT_COLOR),
                bgcolor="rgba(0,0,0,0.45)",
                bordercolor=BOT_COLOR,
                borderwidth=0,
                align="center",
            )

        # INACTIVITY windows
        for block in sorted(self.blocks, key=lambda b: b.start_ts):
            m_mask = (self.df_candles["timestamp"] >= block.start_ts) & (self.df_candles["timestamp"] <= block.end_ts)
            m_seg = self.df_candles.loc[m_mask, ["timestamp", "volume"]].sort_values("timestamp").copy()
            if m_seg.empty:
                continue
            m_seg["cum_mkt"] = m_seg["volume"].cumsum()
            x = pd.to_datetime(m_seg["timestamp"], unit="s")
            y_base = np.full(len(x), 1.0) * EPS
            y_mkt_true = m_seg["cum_mkt"].values
            y_mkt = y_mkt_true + EPS
            y2_max = max(y2_max, float(y_mkt_true.max()))

            # Hidden baseline then market with fill-to-previous
            fig.add_trace(
                go.Scatter(x=x, y=y_base, mode="lines",
                           line=dict(width=0), hoverinfo="skip", showlegend=False),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=x, y=y_mkt, mode="lines",
                    line=dict(width=2, color=MARKET_COLOR),
                    fill="tonexty", fillcolor=FILL_INACT,
                    name="Market • Cum vol (inactive)",
                    legendgroup="mkt", showlegend=False
                ),
                row=2, col=1
            )

        # Axes and layout
        upper_log = np.log10(max(y2_max, 1.0)) + 0.25
        fig.update_yaxes(type="log", range=[1, upper_log], row=2, col=1)
        fig.update_yaxes(title_text="Cum. Volume (log)", row=2, col=1)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=1, secondary_y=True)
        fig.update_layout(margin=dict(t=110), height=900)
        fig.update_yaxes(title_text="Price", row=1, col=1)

        return fig


async def main(root_path: str):
    strategy_runs = StrategyRuns(host=os.getenv("HUMMINGBOT_API_HOST", "localhost"), root_path=root_path)
    await strategy_runs.init()
    await strategy_runs.fetch_data("bots/archived/pmm-mister-brigado-binance-13-3-20251010-1440/data/pmm-mister-brigado-binance-13-3-20251010-1440-20251010-144042.sqlite")
    df_executors = strategy_runs.get_executors_df()
    strategy_runs.build_blocks(df_executors)
    print("done!")

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    root_path = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    sys.path.append(root_path)

    asyncio.run(main(root_path))
