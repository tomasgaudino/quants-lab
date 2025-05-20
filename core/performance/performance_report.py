import logging
import os
import warnings

from typing import Dict, Any, List

import numpy as np
import pandas as pd

from core.data_sources import CLOBDataSource
from core.data_sources.hummingbot_database import HummingbotDatabase
from core.data_structures.candles import Candles
from core.performance.models import TradingSession
from core.performance.visualizations import Visualizer
from core.services.mongodb_client import MongoClient
import core.performance.utils as utils

logging.getLogger("asyncio").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")


class PerformanceReport:
    def __init__(self,
                 mongo_uri: str,
                 database: str,
                 root_path: str = "",
                 owner: str = "master",
                 controller_names: List[str] = None):
        self.mongo_client = MongoClient(uri=mongo_uri, database=database)
        self.root_path = root_path
        self.visualizer = Visualizer()
        self.dbs_index: Dict[str, Any] = {}
        self.controller_names = controller_names
        self.owner = owner
        self.controllers_df = pd.DataFrame()
        self.executors_df = pd.DataFrame()
        self.trades_df = pd.DataFrame()
        self.trading_sessions = []
        self.trading_pairs = []
        self.market_data = []

    async def initialize(self):
        await self.mongo_client.connect()

    def summary(self):
        if len(self.executors_df) > 0:
            total_instances = len(self.executors_df["database_id"].unique())
            return f"Total instances: {total_instances}"
        else:
            return "No data loaded"

    def get_filtered_options(self, current_selection):
        df = self.trades_df.copy()

        for col, selected_values in current_selection.items():
            if selected_values and "All" not in selected_values:
                df = df[df[col].isin(selected_values)]

        filtered_options = {
            "controller_id": sorted(df["controller_id"].unique()),
            "database_id": sorted(df["database_id"].unique()),
            "trading_pair": sorted(df["trading_pair"].unique()),
            "connector_name": sorted(df["connector_name"].unique()),
            "controller_name": sorted(df["controller_name"].unique()),
        }

        for key in filtered_options:
            filtered_options[key] = list(filtered_options[key])

        return filtered_options

    def load_data(self, db_list: List[str], server_name: str = ""):
        for database_id in db_list:
            try:
                database = HummingbotDatabase(db_name=database_id, root_path=self.root_path, server_name=server_name)
                self.executors_df = pd.concat([self.executors_df, self.get_executors(database=database, database_id=database_id)])
                self.controllers_df = pd.concat([self.controllers_df, self.get_controllers(database)])
                self.trades_df = pd.concat([self.trades_df, self.get_trades_df()])
                self.dbs_index[database_id] = {
                    "start_time": self.trades_df.timestamp.min(),
                    "end_time": self.trades_df.timestamp.max(),
                }
            except Exception as e:
                print(e.with_traceback(None))
                continue
        self.trading_pairs = list(self.trades_df["trading_pair"].unique())
        self.market_data: Dict[str, Candles] = None

    @staticmethod
    def get_executors(database: HummingbotDatabase, database_id: str) -> pd.DataFrame:
        executors_df = database.get_executors_data()
        executors_df["database_id"] = database_id
        executors_df.rename(columns={"id": "executor_id"}, inplace=True)
        return executors_df

    def get_controllers(self, database: HummingbotDatabase) -> pd.DataFrame:
        controllers_df = database.get_controller_data()
        controllers_df.drop(columns=["controller_id"], inplace=True)
        controllers_df.rename(columns={"id": "controller_id", "type": "controller_type"}, inplace=True)
        controllers_df["database_id"] = database.db_name
        controllers_df["controller_name"] = controllers_df["config"].apply(lambda x: x["controller_name"])
        if self.controller_names:
            controllers_df = controllers_df[controllers_df["controller_name"].isin(self.controller_names)]
        return controllers_df

    def get_trades_df(self) -> pd.DataFrame:
        all_trades = []
        for _, executor in self.executors_df.iterrows():
            # TODO: Check with @cardosofede if this will be stable over time
            custom_info = executor["custom_info"]
            controller_name = self.controllers_df[
                self.controllers_df["controller_id"] == executor["config"]["controller_id"]]["controller_name"].iloc[0]
            controller_type = self.controllers_df[
                self.controllers_df["controller_id"] == executor["config"]["controller_id"]]["controller_type"].iloc[0]

            for order_filled in custom_info["filled_orders"]:
                for _, fill in order_filled["order_fills"].items():
                    trade_type = order_filled["trade_type"]  # BUY or SELL
                    position_action = order_filled["position"]  # OPEN or CLOSE
                    position_multiplier = 1 if (trade_type == "BUY" and position_action == "OPEN") or (
                                trade_type == "SELL" and position_action == "CLOSE") else -1
                    fill_dict = {
                        "database_id": executor["database_id"],
                        "controller_id": executor["controller_id"],
                        "client_order_id": fill["client_order_id"],
                        "exchange_order_id": fill["exchange_order_id"],
                        "connector_name": executor["config"]["connector_name"],  # TODO: add connector_name to fills
                        "controller_type": controller_type,
                        "controller_name": controller_name,
                        "executor_id": executor["executor_id"],
                        "side": executor["config"]["side"],
                        "trading_pair": order_filled["trading_pair"],
                        "order_type": order_filled["order_type"],
                        "trade_type": trade_type,
                        "cumulative_fee_paid_quote": sum(
                            [float(flat_fee["amount"]) for flat_fee in fill["fee"]["flat_fees"]]),
                        "position_action": order_filled["position"],
                        "timestamp": utils.ensure_timestamp_in_seconds(fill["fill_timestamp"]),
                        "price": float(fill["fill_price"]),
                        "base_amount": float(fill["fill_base_amount"]),
                        "quote_amount": float(fill["fill_quote_amount"]),
                        "position_multiplier": position_multiplier
                    }
                    all_trades.append(fill_dict)
        all_trades_df = pd.DataFrame(all_trades)
        return all_trades_df

    async def build_trading_sessions(self) -> List[TradingSession]:
        raise NotImplementedError

    @staticmethod
    def calculate_performance_fields(trades_df: pd.DataFrame, side: int = 1):
        performance_df = trades_df.copy()
        performance_df.sort_values("timestamp", inplace=True)
        performance_df = performance_df[performance_df["side"] == side]
        performance_df["datetime"] = pd.to_datetime(performance_df["timestamp"], unit="s")

        # Initialize columns
        performance_df["base_amount_open"] = np.where(performance_df["position_action"] == "OPEN",
                                                      performance_df["base_amount"], 0)
        performance_df["base_amount_close"] = np.where(performance_df["position_action"] == "CLOSE",
                                                       performance_df["base_amount"], 0)
        performance_df["cum_base_open"] = performance_df["base_amount_open"].cumsum()
        performance_df["cum_base_close"] = performance_df["base_amount_close"].cumsum()

        performance_df["cum_quote_open"] = (performance_df["base_amount_open"] * performance_df["price"]).cumsum()
        performance_df["cum_quote_close"] = (performance_df["base_amount_close"] * performance_df["price"]).cumsum()

        # Break-even calculation
        performance_df["break_even_open"] = performance_df["cum_quote_open"] / performance_df["cum_base_open"]
        performance_df["break_even_close"] = performance_df["cum_quote_close"] / performance_df["cum_base_close"]

        # PnL calculations
        if side == 1:  # Long
            performance_df["realized_pnl"] = (performance_df["break_even_close"] - performance_df["break_even_open"]) * \
                                             performance_df["cum_base_close"]
            performance_df["unrealized_pnl"] = (performance_df["price"] - performance_df["break_even_open"]) * (
                        performance_df["cum_base_open"] - performance_df["cum_base_close"])
        else:  # Short (side=2)
            performance_df["realized_pnl"] = (performance_df["break_even_open"] - performance_df["break_even_close"]) * \
                                             performance_df["cum_base_close"]
            performance_df["unrealized_pnl"] = (performance_df["break_even_open"] - performance_df["price"]) * (
                        performance_df["cum_base_open"] - performance_df["cum_base_close"])

        # Global PnL
        performance_df["global_pnl"] = (performance_df["realized_pnl"] + performance_df["unrealized_pnl"] -
                                        performance_df["cumulative_fee_paid_quote"].cumsum())
        cols_to_show = ["datetime", "database_id", "executor_id", "connector_name", "base_amount", "quote_amount",
                        "base_amount_open", "base_amount_close", "cum_base_open", "cum_base_close", "cum_quote_open",
                        "cum_quote_close", "break_even_open", "break_even_close", "realized_pnl", "unrealized_pnl",
                        "global_pnl"]
        return performance_df[cols_to_show]

    @staticmethod
    def summarize_performance_metrics(df: pd.DataFrame, side: int = 1, total_amount_quote: float = 1000.0):
        global_pnl = df["global_pnl"].iloc[-1]
        max_draw_down = df["global_pnl"].min() / total_amount_quote
        max_run_up = df["global_pnl"].max() / total_amount_quote
        total_trades = len(df)
        total_quote_volume = df["quote_amount"].sum()
        total_duration_minutes = (df["timestamp"].max() - df["timestamp"].min()) / 60
        metrics = {
            "trading_pair": df["trading_pair"].iloc[0],
            "global_pnl": global_pnl,
            "max_draw_down": max_draw_down,
            "max_run_up": max_run_up,
            "total_trades": total_trades,
            "total_quote_volume": total_quote_volume,
            "total_duration_minutes": total_duration_minutes,
            "side": "long" if side == 1 else "short",
        }
        return metrics
