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
from research_notebooks.macd_mt.madbb_dynamic_config_generation import MACDMTConfigGenerator
#from research_notebooks.macd_mt.madbb_fixed_pct_generation import MACDMTConfigGenerator
from core.data_sources import CLOBDataSource


from decimal import Decimal
import optuna

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


class MACDMTBacktestingTask(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name, frequency, config)
        self.resolution = self.config["resolution"]
        self.screener_config = self.config
        self.root_path = self.config.get('root_path', "")
        self.backtesting_days = self.config["backtesting_days"]
        self.connector_name = self.config["connector_name"]
        self.trading_pairs = self.config["selected_pairs"]
        self.trials = self.config["trials_per_pair"]

    async def execute(self):
        kwargs = {
            "root_path": self.config["root_path"],
            "db_host": self.config["optuna_config"]["host"],
            "db_port": self.config["optuna_config"]["port"],
            "db_user": self.config["optuna_config"]["user"],
            "db_pass": self.config["optuna_config"]["password"],
            "database_name": self.config["optuna_config"]["database"],
        }
        storage_name = StrategyOptimizer.get_storage_name(
            engine=self.config.get("engine", "sqlite"),
            **kwargs)
        
        selected_pairs = self.trading_pairs
        logger.info("Optimizing strategy for top markets: {}".format(len(selected_pairs)))        
        
        for trading_pair in selected_pairs:
            optimizer = StrategyOptimizer(
                storage_name=storage_name,
                resolution=self.resolution,
                root_path=self.config["root_path"],
                custom_backtester=DirectionalTradingBacktesting()
            )
            optimizer.load_candles_cache_by_connector_pair(connector_name=self.connector_name, trading_pair=trading_pair)
            candles_1s = optimizer._backtesting_engine._bt_engine.backtesting_data_provider.candles_feeds[
                (f"{self.connector_name}_{trading_pair}_{self.resolution}")]
            start_date = candles_1s.index.min()
            end_date = candles_1s.index.max()
            logger.info(f"Optimizing strategy for {self.connector_name} {trading_pair} {start_date} {end_date}")
            config_generator = MACDMTConfigGenerator(start_date=start_date, end_date=end_date)
            config_generator.trading_pair = trading_pair
            config_generator.candles_trading_pair = trading_pair
            today_str = datetime.datetime.now().strftime("%Y-%m-%d")
            await optimizer.optimize(study_name=f"macdmt_task_dynamic{today_str}",
                                     config_generator=config_generator, n_trials=self.trials)


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
        "engine": "sqlite", 
        "connector_name": "binance_perpetual",
        "timescale_config": timescale_config,
        "optuna_config": optuna_config, 
        "selected_pairs": ['1000BONK-USDT', '1000PEPE-USDT']

    }
    task = MACDMTBacktestingTask("Backtesting", timedelta(hours=12), config)
    await task.execute()



if __name__ == "__main__":
    asyncio.run(main())
