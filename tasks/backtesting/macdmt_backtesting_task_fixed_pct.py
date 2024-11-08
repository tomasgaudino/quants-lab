import asyncio
import datetime
import logging
import os
from datetime import timedelta
from typing import Any, Dict

import pandas as pd
from dotenv import load_dotenv

from core.backtesting.optimizer import StrategyOptimizer,BacktestingConfig, BaseStrategyConfigGenerator
from core.services.timescale_client import TimescaleClient
from core.task_base import BaseTask
from hummingbot.strategy_v2.backtesting import DirectionalTradingBacktesting
from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop
#from research_notebooks.macd_mt.madbb_dynamic_config_generation import MACDMTConfigGenerator
from research_notebooks.macd_mt.madbb_fixed_pct_config_generation import MACDMTConfigGenerator

from core.data_sources import CLOBDataSource

from decimal import Decimal
import optuna

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


class MACDMTBacktestingTaskFixedPCT(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name, frequency, config)
        self.resolution = self.config["resolution"]
        self.screener_config = self.config
        self.root_path = self.config.get('root_path', "")
        self.backtesting_days = self.config["backtesting_days"]
        self.connector_name = self.config["connector_name"]

    async def execute(self):
        ts_client = TimescaleClient(
            host=self.config["timescale_config"]["host"],
            port=self.config["timescale_config"]["port"],
            user=self.config["timescale_config"]["user"],
            password=self.config["timescale_config"]["password"],
            database=self.config["timescale_config"]["database"]
        )
        await ts_client.connect()


        logger.info("Generating top markets report")
        CONNECTOR_NAME = "binance_perpetual"
        # Trading Rules Filter
        QUOTE_ASSET = "USDT"
        MIN_NOTIONAL_SIZE = 5  # In USDT
        clob = CLOBDataSource()

        trading_rules = await clob.get_trading_rules(CONNECTOR_NAME)
        trading_pairs = trading_rules.filter_by_quote_asset(QUOTE_ASSET) \
            .filter_by_min_notional_size(Decimal(MIN_NOTIONAL_SIZE)) \
            .get_all_trading_pairs()
        trading_pairs_available = await ts_client.get_available_pairs()
        trading_pairs_available = [pair[1] for pair in trading_pairs_available if pair[0] == CONNECTOR_NAME]

        trading_pairs = [trading_pair for trading_pair in trading_pairs_available if trading_pair in trading_pairs]

        optimizer = StrategyOptimizer(engine="postgres",
                                      root_path=self.root_path,
                                      resolution=self.resolution,
                                      db_client=ts_client,
                                      db_host=self.config["optuna_config"]["host"],
                                      db_port=self.config["optuna_config"]["port"],
                                      db_user=self.config["optuna_config"]["user"],
                                      db_pass=self.config["optuna_config"]["password"],
                                      database_name=self.config["optuna_config"]["database"],
                                      )

        logger.info("Optimizing strategy for top markets: {}".format(len(trading_pairs)))
        start_date = datetime.datetime.now() - datetime.timedelta(days=self.backtesting_days)
        end_date = datetime.datetime.now()
        for trading_pair in trading_pairs:
            config_generator = MACDMTConfigGenerator(start_date=start_date, end_date=end_date, backtester=DirectionalTradingBacktesting())
            config_generator.trading_pair = trading_pair
            config_generator.candles_trading_pair = trading_pair
            candles = await optimizer._db_client.get_candles(self.connector_name, trading_pair,
                                                             self.resolution, start_date.timestamp(), end_date.timestamp())
            start_time = candles.data["timestamp"].min()
            end_time = candles.data["timestamp"].max()
            config_generator.backtester.backtesting_data_provider.candles_feeds[
                f"{self.connector_name}_{trading_pair}_{self.resolution}"] = candles.data
            config_generator.start = start_time
            config_generator.end = end_time
            logger.info(f"Fetching candles for {self.connector_name} {trading_pair} {start_date} {end_date}")
            candles = await optimizer._db_client.get_candles(self.connector_name, trading_pair,
                                                             self.resolution, start_date.timestamp(), end_date.timestamp())
            start_time = candles.data["timestamp"].min()
            end_time = candles.data["timestamp"].max()
            config_generator.backtester.backtesting_data_provider.candles_feeds[
                f"{self.connector_name}_{trading_pair}_{self.resolution}"] = candles.data
            config_generator.start = start_time
            config_generator.end = end_time
            today_str = datetime.datetime.now().strftime("%Y-%m-%d")
            await optimizer.optimize(study_name=f"macdmt_fixed_pct_task_{today_str}",
                                     config_generator=config_generator, n_trials=100)


async def main():

    timescale_config = {
        "host": "63.250.52.93",
        "port": 5432,
        "user": "admin",
        "password": "admin",
        "database": "timescaledb",
    }
    optuna_config = {
        "host": "63.250.52.93",
        "port": 5433,
        "user": "admin",
        "password": "admin",
        "database": "optimization_database"
    }

    config = {
        "root_path": os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
        "resolution": "1s",
        "backtesting_days": 7, 
        "connector_name": "binance_perpetual",
        "timescale_config": timescale_config,
        "optuna_config": optuna_config

    }
    task = MACDMTBacktestingTaskFixedPCT("Backtesting", timedelta(hours=12), config)
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main())
