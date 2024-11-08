import logging
from core.backtesting.optimizer import BacktestingConfig, BaseStrategyConfigGenerator
from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop
from controllers.directional_trading.macd_mt_dca import MacdMTDCAControllerConfig

from decimal import Decimal
import optuna

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MACDMTConfigGenerator(BaseStrategyConfigGenerator):
    """
    Strategy configuration generator for MACD MT optimization.
    """
    async def generate_config(self, trial) -> BacktestingConfig:
        dynamic_order_spread = True
        dynamic_target =  True
        min_stop_loss=Decimal("0.007")
        max_stop_loss=Decimal("0.1")
        min_trailing_stop=Decimal("0.0025")
        max_trailing_stop=Decimal("0.03")
        bb_interval = trial.suggest_categorical("bb_interval", ["1m", "5m", "15m", "1h"])
        bb_length = trial.suggest_int("bb_length", 50, 200, step=50)
        bb_std = trial.suggest_float("bb_std", 1.5, 3, step=0.5)
        stop_loss = trial.suggest_float("stop_loss", 0.1, 1, step=0.1)
        dca_spread_1 = trial.suggest_float("dca_spread_1", 0.25, 1, step=0.25)       
        trailing_stop_activation_price = trial.suggest_float("trailing_stop_activation_price", 0.25, 1, step=0.25)
        trailing_delta_ratio = trial.suggest_float("trailing_delta_ratio", 0.05, 0.1, step=0.01)
        trailing_stop_trailing_delta = trailing_stop_activation_price * trailing_delta_ratio
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
        
        # Apply logical constraints, pruning if violated
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

