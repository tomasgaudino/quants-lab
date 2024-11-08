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
from controllers.directional_trading.macd_mt_dca import MacdMTDCAControllerConfig

from decimal import Decimal
import optuna

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


class MACDMTConfigGenerator(BaseStrategyConfigGenerator):
    """
    Strategy configuration generator for MACD MT optimization.
    """
    async def generate_config(self, trial) -> BacktestingConfig:
            
        # paramemeters when dynamic = False
        stop_loss = trial.suggest_float("stop_loss", 0.01, 0.1, step=0.01)
        trailing_stop_activation_price = trial.suggest_float("trailing_stop_activation_price", 0.004, 0.02, step=0.001)
        trailing_delta_ratio = trial.suggest_float("trailing_delta_ratio", 0.05, 0.1, step=0.01)
        trailing_stop_trailing_delta = trailing_stop_activation_price * trailing_delta_ratio
        dca_spread_1 = trial.suggest_float("dca_spread_1", 0.01, 0.04, step=0.01)       
        bb_interval = "1m"
        bb_length = 200
        bb_std = 2

        dynamic_order_spread = False
        dynamic_target =  False



        min_stop_loss=Decimal("0.007")
        max_stop_loss=Decimal("0.1")
        min_trailing_stop=Decimal("0.0025")
        max_trailing_stop=Decimal("0.03")
        
        #bb_interval = trial.suggest_categorical("bb_interval", ["1m", "5m", "15m", "1h"])
        #bb_length = trial.suggest_int("bb_length", 50, 200, step=50)
        #bb_std = trial.suggest_float("bb_std", 1.5, 3, step=0.5)
        #stop_loss = trial.suggest_float("stop_loss", 0.1, 1, step=0.1)
        #dca_spread_1 = trial.suggest_float("dca_spread_1", 0.25, 1, step=0.25)       
        #trailing_stop_activation_price = trial.suggest_float("trailing_stop_activation_price", 0.25, 1, step=0.25)
        #trailing_delta_ratio = trial.suggest_float("trailing_delta_ratio", 0.05, 0.1, step=0.01)
        #trailing_stop_trailing_delta = trailing_stop_activation_price * trailing_delta_ratio

        trailing_stop = TrailingStop(
            activation_price=Decimal(trailing_stop_activation_price),
            trailing_delta=Decimal(trailing_stop_trailing_delta)
        )

        dca_amount_1 = trial.suggest_float("dca_amount_1", 1.0, 3.0, step=1.0)  
        max_executors_per_side = trial.suggest_int("max_executors_per_side", 1, 3, step=1)
        macd_interval_1 = trial.suggest_categorical("macd_interval_1", ["1m", "5m", "15m", "1h"])
        macd_signal_type_1 = trial.suggest_categorical("macd_signal_type_1", ["mean_reversion_1", "mean_reversion_2", "trend_following"])
        macd_fast_1 = trial.suggest_int("macd_fast_1", 9, 59, step=10)
        macd_slow_1 = trial.suggest_int("macd_slow_1", 21, 201, step=10)
        macd_signal_1 = trial.suggest_int("macd_signal_1", 10, 100, step=10)
        macd_number_of_candles_1 = trial.suggest_int("macd_number_of_candles_1", 1, 6, step=2)
        macd_interval_2 = trial.suggest_categorical("macd_interval_2", ["1m", "5m", "15m", "1h"])
        macd_signal_type_2 = trial.suggest_categorical("macd_signal_type_2", ["mean_reversion_1", "mean_reversion_2", "trend_following"])
        macd_fast_2 = trial.suggest_int("macd_fast_2", 9, 59, step=10)
        macd_slow_2 = trial.suggest_int("macd_slow_2", 21, 201, step=10)
        macd_signal_2 = trial.suggest_int("macd_signal_2", 10, 100, step=10)
        macd_number_of_candles_2 = trial.suggest_int("macd_number_of_candles_2", 1, 6, step=2)
        cooldown_time =  trial.suggest_int("cooldown_time", 60*60, 60*60*5, step=60*60)
        take_profit = 100
        time_limit = 60 * 60 * 24 * 2
        total_amount_quote = 100
        executor_refresh_time = 60*2
        
        if (macd_fast_1 >= macd_slow_1 or macd_fast_2 >= macd_slow_2):
            raise optuna.TrialPruned()
        # Create the strategy configuration
        config = MacdMTDCAControllerConfig(
            connector_name="binance_perpetual",
            trading_pair=self.trading_pair,
            candles_trading_pair=self.candles_trading_pair,
            macd_interval_1 = macd_interval_1,
            macd_signal_type_1 = macd_signal_type_1,
            bb_interval = bb_interval,
            bb_length = bb_length,
            bb_std = bb_std,
            dynamic_order_spread = dynamic_order_spread,
            dynamic_target = dynamic_target,
            min_stop_loss = min_stop_loss,
            max_stop_loss = max_stop_loss,
            min_trailing_stop = min_trailing_stop,
            max_trailing_stop = max_trailing_stop,
            macd_fast_1 = macd_fast_1,
            macd_slow_1 = macd_slow_1,
            macd_signal_1 = macd_signal_1,
            macd_number_of_candles_1 = macd_number_of_candles_1,
            macd_interval_2 = macd_interval_2,
            macd_signal_type_2 = macd_signal_type_2,
            macd_fast_2 = macd_fast_2,
            macd_slow_2 = macd_slow_2,
            macd_signal_2 = macd_signal_2,
            macd_number_of_candles_2 = macd_number_of_candles_2,
            dca_amounts = [Decimal("1"), dca_amount_1],
            dca_spreads = [Decimal("-0.00000001"), dca_spread_1],
            total_amount_quote=Decimal(total_amount_quote),
            take_profit=Decimal(take_profit),
            stop_loss=Decimal(stop_loss),
            trailing_stop=trailing_stop,
            time_limit=time_limit,
            max_executors_per_side=max_executors_per_side,
            executor_refresh_time = executor_refresh_time,
            cooldown_time=cooldown_time,
        )

        # Return the configuration encapsulated in BacktestingConfig
        return BacktestingConfig(config=config, start=self.start, end=self.end)

class MACDMTBacktestingTask(BaseTask):
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
        trading_pairs_available = await ts_client.get_available_pairs()
        trading_pairs = [pair[1] for pair in trading_pairs_available if pair[0] == self.connector_name]
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
            await optimizer.optimize(study_name=f"macdmt_task_{today_str}",
                                     config_generator=config_generator, n_trials=50)


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
    task = MACDMTBacktestingTask("Backtesting", timedelta(hours=12), config)
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main())
