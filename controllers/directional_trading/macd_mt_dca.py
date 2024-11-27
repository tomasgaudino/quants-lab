from decimal import Decimal
from typing import List, Optional
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import pandas_ta as ta  # noqa: F401
from pydantic import Field, validator
from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.core.data_type.common import TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)
from hummingbot.strategy_v2.executors.dca_executor.data_types import DCAExecutorConfig, DCAMode
from hummingbot.strategy_v2.models.executor_actions import ExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop


class MacdMTDCAControllerConfig(DirectionalTradingControllerConfigBase):
    controller_name = "macd_mt_dca"


    bb_interval: str = Field(
        default="3m",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ",
            prompt_on_new=False))
    bb_length: int = Field(
        default=100,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the Bollinger Bands length: ",
            prompt_on_new=False))
    bb_std: float = Field(
        default=2.0,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the Bollinger Bands standard deviation: ",
            prompt_on_new=False))
    
    dynamic_order_spread: bool = Field(
        default=None,
        client_data=ClientFieldData(
            prompt=lambda mi: "Do you want to make the spread dynamic? (Yes/No) ",
            prompt_on_new=False))
    dynamic_target: bool = Field(
        default=None,
        client_data=ClientFieldData(
            prompt=lambda mi: "Do you want to make the target dynamic? (Yes/No) ",
            prompt_on_new=False))
    min_stop_loss: Decimal = Field(
        default=Decimal("0.01"),
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the minimum stop loss (as a decimal, e.g., 0.01 for 1%): ",
            prompt_on_new=False))
    max_stop_loss: Decimal = Field(
        default=Decimal("0.1"),
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the maximum stop loss (as a decimal, e.g., 0.1 for 10%): ",
            prompt_on_new=False))
    min_trailing_stop: Decimal = Field(
        default=Decimal("0.005"),
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the minimum trailing stop (as a decimal, e.g., 0.01 for 1%): ",
            prompt_on_new=False))
    max_trailing_stop: Decimal = Field(
        default=Decimal("0.2"),
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the maximum trailing stop (as a decimal, e.g., 0.1 for 10%): ",
            prompt_on_new=False))
    #min_distance_between_orders: Decimal = Field(
    #    default=Decimal("0.01"),
    #    client_data=ClientFieldData(
    #        prompt=lambda mi: "Enter the minimum distance between orders (as a decimal, e.g., 0.01 for 1%): ",
    #        prompt_on_new=False))

    #DCA CONFIG
    dca_spreads: List[Decimal] = Field(
        default="-0.00001,0.02,0.04,0.08",
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter a comma-separated list of spreads for each DCA level: "))
    dca_amounts: List[Decimal] = Field(
        default="0.1,0.2,0.4,0.8",
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter a comma-separated list of amounts for each DCA level: "))
    time_limit: int = Field(
        default=60 * 60 * 24 * 7, gt=0,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the time limit for each DCA level: ",
            prompt_on_new=False))
    stop_loss: Decimal = Field(
        default=Decimal("0.03"), gt=0,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the stop loss (as a decimal, e.g., 0.03 for 3%): ",
            prompt_on_new=True))
    executor_refresh_time: Optional[float] = Field(
        default=None,
        client_data=ClientFieldData(
            is_updatable=True,
            prompt_on_new=False))
    executor_activation_bounds: Optional[List[Decimal]] = Field(
        default=None,
        client_data=ClientFieldData(
            is_updatable=True,
            prompt=lambda mi: "Enter the activation bounds for the orders "
                              "(e.g., 0.01 activates the next order when the price is closer than 1%): ",
            prompt_on_new=False))
    #candles_data
    candles_config: List[CandlesConfig] = []
    candles_connector: str = Field(
        default=None,
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the connector for the candles data, leave empty to use the same exchange as the connector: ", )
    )
    candles_trading_pair: str = Field(
        default=None,
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the trading pair for the candles data, leave empty to use the same trading pair as the connector: ", )
    )

    #MACDS CONFIG
    macd_interval_1: str = Field(
        default="3m",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ",
            prompt_on_new=False))
    macd_fast_1: int = Field(
        default=21,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD fast period: ",
            prompt_on_new=True))
    macd_slow_1: int = Field(
        default=42,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD slow period: ",
            prompt_on_new=True))
    macd_signal_1: int = Field(
        default=9,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD signal period: ",
            prompt_on_new=True))
    macd_signal_type_1: str = Field(
        default="mean_reversion_1",
        client_data=ClientFieldData(
            prompt=lambda mi: "mean_reversion_1/trend_following/mean_reversion_2",
            prompt_on_new=False))
    macd_number_of_candles_1: int = Field(
        default=4,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD increasing-decr periods: ",
            prompt_on_new=True))
    macd_interval_2: str = Field(
        default="3m",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ",
            prompt_on_new=False))
    macd_fast_2: int = Field(
        default=21,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD fast period: ",
            prompt_on_new=True))
    macd_slow_2: int = Field(
        default=42,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD slow period: ",
            prompt_on_new=True))
    macd_signal_2: int = Field(
        default=9,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD signal period: ",
            prompt_on_new=True))
    macd_signal_type_2: str = Field(
        default="mean_reversion",
        client_data=ClientFieldData(
            prompt=lambda mi: "mean_reversion/trend_following",
            prompt_on_new=False))
    macd_number_of_candles_2: int = Field(
        default=4,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD increasing-decr periods: ",
            prompt_on_new=True))
    
    @validator("executor_activation_bounds", pre=True, always=True)
    def parse_activation_bounds(cls, v):
        if isinstance(v, list):
            return [Decimal(val) for val in v]
        elif isinstance(v, str):
            if v == "":
                return None
            return [Decimal(val) for val in v.split(",")]
        return v

    @validator('dca_spreads', pre=True, always=True)
    def parse_spreads(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            if v == "":
                return []
            return [float(x.strip()) for x in v.split(',')]
        return v

    @validator('dca_amounts', pre=True, always=True)
    def parse_and_validate_amounts(cls, v, values, field):
        if v is None or v == "":
            return [1 for _ in values[values['dca_spreads']]]
        if isinstance(v, str):
            return [float(x.strip()) for x in v.split(',')]
        elif isinstance(v, list) and len(v) != len(values['dca_spreads']):
            raise ValueError(
                f"The number of {field.name} must match the number of {values['dca_spreads']}.")
        return v

    @validator("candles_connector", pre=True, always=True)
    def set_candles_connector(cls, v, values):
        if v is None or v == "":
            return values.get("connector_name")
        return v

    @validator("candles_trading_pair", pre=True, always=True)
    def set_candles_trading_pair(cls, v, values):
        if v is None or v == "":
            return values.get("trading_pair")
        return v
    
    def get_spreads_and_amounts_in_quote(self,
                                         trade_type: TradeType,
                                         total_amount_quote: Decimal):
        # Equally distribute if amounts_pct is not set
        amounts_pct = self.dca_amounts
        if amounts_pct is None:
            # Equally distribute if amounts_pct is not set
            spreads = self.dca_spreads
            normalized_amounts_pct = [Decimal('1.0') / len(spreads) for _ in spreads]
        else:
            if trade_type == TradeType.BUY:
                normalized_amounts_pct = [Decimal(amt_pct) / sum(amounts_pct) for amt_pct in amounts_pct]
            else:  # TradeType.SELL
                normalized_amounts_pct = [Decimal(amt_pct) / sum(amounts_pct) for amt_pct in amounts_pct]

        return self.dca_spreads, [amt_pct * total_amount_quote for amt_pct in normalized_amounts_pct]

class MacdMTDCAController(DirectionalTradingControllerBase):
    def __init__(self, config: MacdMTDCAControllerConfig, *args, **kwargs):
        self.config = config
        self.dca_amounts_pct = [Decimal(amount) / sum(self.config.dca_amounts) for amount in self.config.dca_amounts]
        self.spreads = self.config.dca_spreads
        max_records_list = self.get_candle_max_records()
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.macd_interval_1,
                max_records=max_records_list[0]
            ), CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.macd_interval_2,
                max_records=max_records_list[1]
            ), CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.bb_interval,
                max_records=max_records_list[2]
            ),
            ]
        super().__init__(config, *args, **kwargs)

    def calculate_derivative(self, df, window_size, column_key, s=0):
        signals = []
        for i in range(len(df)):
            if i < window_size:
                # Not enough data points to fit the polynomial
                signals.append(np.nan)
            else:
                # Fit a polynomial to the last 'window_size' points
                y = df[column_key].iloc[i - window_size:i]
                x = np.arange(window_size)
                if y.isna().any():
                    signals.append(np.nan)
                else:
                    k = min(5, (window_size-1))
                    spline = UnivariateSpline(x, y, k=k, s=s)
                    # Derivative of the polynomial at the current point
                    derivative = spline.derivative()(window_size)
                    signals.append(derivative.min())
        # Normalize the signal values between 0 and 1
        
        return df
    
    async def update_processed_data(self):

        def check_increasing(series, number_of_candles):
            # Check if the series has 3 consecutive increasing values
            return (series.iloc[-number_of_candles] <= series.iloc[-1])

        def check_decreasing(series, number_of_candles):
            # Check if the series has 3 consecutive increasing values
            return (series.iloc[-number_of_candles] >= series.iloc[-1])

        
        df_bb = self.market_data_provider.get_candles_df(connector_name=self.config.candles_connector,
                                                      trading_pair=self.config.candles_trading_pair,
                                                      interval=self.config.bb_interval,
                                                      max_records=self.config.candles_config[2].max_records)
        df_bb.ta.bbands(length=self.config.bb_length, std=self.config.bb_std, append=True)
        df_bb['time'] = pd.to_datetime(df_bb['timestamp'], unit='s')

        
        # Add indicators
        # macd1
        df_macd_1 = self.market_data_provider.get_candles_df(connector_name=self.config.candles_connector,
                                                      trading_pair=self.config.candles_trading_pair,
                                                      interval=self.config.macd_interval_1,
                                                      max_records=self.config.candles_config[0].max_records)
        df_macd_1.ta.macd(fast=self.config.macd_fast_1, slow=self.config.macd_slow_1, signal=self.config.macd_signal_1, append=True)
        macd_1_col = f'MACD_{self.config.macd_fast_1}_{self.config.macd_slow_1}_{self.config.macd_signal_1}'
        macd_1_h_col = f'MACDh_{self.config.macd_fast_1}_{self.config.macd_slow_1}_{self.config.macd_signal_1}'
        # ADD TIME COLUMN TO MERGE
        df_macd_1['time'] = pd.to_datetime(df_macd_1['timestamp'], unit='s')
        # SET MACD SIGNAL
        df_macd_1["signal_macd_1"] = 0
        number_of_candles_1 = self.config.macd_number_of_candles_1

        if self.config.macd_signal_type_1 == 'mean_reversion_1':
            long_condition_1 = (df_macd_1[macd_1_h_col] > 0) & (df_macd_1[macd_1_col] < 0)
            short_condition_1 = (df_macd_1[macd_1_h_col] < 0) & (df_macd_1[macd_1_col] > 0)
            df_macd_1.loc[long_condition_1, "signal_macd_1"] = 1
            df_macd_1.loc[short_condition_1, "signal_macd_1"] = -1
        else:
            df_macd_1["diff_macd_1"] = 0
            if number_of_candles_1 == 1:
                number_of_candles_1 += 1
            increasing_condition_1 = df_macd_1[macd_1_h_col].rolling(window=number_of_candles_1).apply(
                lambda x: check_increasing(x, number_of_candles_1), raw=False).fillna(0).astype(int)
            decreasing_condition_1 = df_macd_1[macd_1_h_col].rolling(window=number_of_candles_1).apply(
                lambda x: check_decreasing(x, number_of_candles_1), raw=False).fillna(0).astype(int)
            df_macd_1.loc[increasing_condition_1.astype(bool), 'diff_macd_1'] = 1
            df_macd_1.loc[decreasing_condition_1.astype(bool), 'diff_macd_1'] = -1
            if self.config.macd_signal_type_1 == 'trend_following':
                df_macd_1['signal_macd_1'] = df_macd_1['diff_macd_1']
            else:
                long_condition_1 = (df_macd_1['diff_macd_1'] > 0) & (df_macd_1[macd_1_col] < 0)
                short_condition_1 = (df_macd_1['diff_macd_1'] < 0) & (df_macd_1[macd_1_col] > 0)
                df_macd_1.loc[long_condition_1, "signal_macd_1"] = 1
                df_macd_1.loc[short_condition_1, "signal_macd_1"] = -1

        #same to macd2
        df_macd_2 = self.market_data_provider.get_candles_df(connector_name=self.config.candles_connector,
                                                             trading_pair=self.config.candles_trading_pair,
                                                             interval=self.config.macd_interval_2,
                                                             max_records=self.config.candles_config[1].max_records)
        # Add indicators
        df_macd_2.ta.macd(fast=self.config.macd_fast_2, slow=self.config.macd_slow_2, signal=self.config.macd_signal_2,
                          append=True)
        macd_2_col = f'MACD_{self.config.macd_fast_2}_{self.config.macd_slow_2}_{self.config.macd_signal_2}'
        macd_2_h_col = f'MACDh_{self.config.macd_fast_2}_{self.config.macd_slow_2}_{self.config.macd_signal_2}'
        # ADD TIME COLUMN TO MERGE
        df_macd_2['time'] = pd.to_datetime(df_macd_2['timestamp'], unit='s')
        # SET MACD SIGNAL
        df_macd_2["signal_macd_2"] = 0
        number_of_candles_2 = self.config.macd_number_of_candles_2

        if self.config.macd_signal_type_2 == 'mean_reversion_1':
            long_condition_2 = (df_macd_2[macd_2_h_col] > 0) & (df_macd_2[macd_2_col] < 0)
            short_condition_2 = (df_macd_2[macd_2_h_col] < 0) & (df_macd_2[macd_2_col] > 0)
            df_macd_2.loc[long_condition_2, "signal_macd_2"] = 1
            df_macd_2.loc[short_condition_2, "signal_macd_2"] = -1
        else:
            #df_macd_2 = self.calculate_derivative(df_macd_2, 10, macd_2_h_col, 1)
            if number_of_candles_2 == 1:
                number_of_candles_2 += 1
            df_macd_2["diff_macd_2"] = 0
            increasing_condition_2 = df_macd_2[macd_2_h_col].rolling(window=number_of_candles_2).apply(
                lambda x: check_increasing(x, number_of_candles_2), raw=False).fillna(0).astype(int)
            decreasing_condition_2 = df_macd_2[macd_2_h_col].rolling(window=number_of_candles_2).apply(
                lambda x: check_decreasing(x, number_of_candles_2), raw=False).fillna(0).astype(int)
            df_macd_2.loc[increasing_condition_2.astype(bool), 'diff_macd_2'] = 1
            df_macd_2.loc[decreasing_condition_2.astype(bool), 'diff_macd_2'] = -1
            if self.config.macd_signal_type_2 == 'trend_following':
                df_macd_2['signal_macd_2'] = df_macd_2['diff_macd_2']
            else:
                long_condition_2 = (df_macd_2['diff_macd_2'] > 0) & (df_macd_2[macd_2_col] < 0)
                short_condition_2 = (df_macd_2['diff_macd_2'] < 0) & (df_macd_2[macd_2_col] > 0)
                df_macd_2.loc[long_condition_2, "signal_macd_2"] = 1
                df_macd_2.loc[short_condition_2, "signal_macd_2"] = -1
        # Generate signal
        # Merge DataFrames on timestamp
        df_macd_1['time'] = pd.to_datetime(df_macd_1['timestamp'], unit='s')
        df_macd_2['time'] = pd.to_datetime(df_macd_2['timestamp'], unit='s')
        df_bb['time'] = pd.to_datetime(df_bb['timestamp'], unit='s')

        df_merged = pd.merge_asof(df_macd_1[['time', 'signal_macd_1', 'timestamp']],df_macd_2[['time', 'signal_macd_2']],
                                  on='time',
                                  direction='backward',)

        bb_with_col = f"BBB_{self.config.bb_length}_{self.config.bb_std}"
        df_merged = pd.merge_asof(df_merged, df_bb[['time', bb_with_col]], on='time', direction='backward',)


        # Compute final signal
        df_merged["signal"] = df_merged.apply(
            lambda row: row['signal_macd_1'] if row['signal_macd_1'] == row['signal_macd_2'] else 0, axis=1)

        # Update processed data
        self.processed_data["signal"] = df_merged["signal"].iloc[-1]
        self.processed_data["features"] = df_merged

    def get_candle_max_records(self):
        #returns list with 2 records: position 0 candle records to macd - position 1 candle records  to bb
        result = []
        interval_durations = {
            '1s': 1,
            '1m': 60,
            '3m': 3 * 60,
            '5m': 5 * 60,
            '15m': 15 * 60,
            '30m': 30 * 60,
            '1h': 60 * 60,
            '4h': 4 * 60 * 60,
            '1d': 24 * 60 * 60,
        }
        total_macd_1_seconds = interval_durations[self.config.macd_interval_1] * (self.config.macd_slow_1 + self.config.macd_signal_1)
        total_macd_2_seconds = interval_durations[self.config.macd_interval_2] * (self.config.macd_slow_2 + self.config.macd_signal_2)
        total_bb_seconds = interval_durations[self.config.bb_interval] * (self.config.bb_length)
        max_seconds = max(total_macd_1_seconds, total_macd_2_seconds, total_bb_seconds)
        max_records_macd_1 = max_seconds // interval_durations[self.config.macd_interval_1]
        max_records_macd_2 = max_seconds // interval_durations[self.config.macd_interval_2]
        max_records_bb = max_seconds // interval_durations[self.config.bb_interval]
        result.append(max_records_macd_1)
        result.append(max_records_macd_2)        
        result.append(max_records_bb)        
        return result

    def get_spread_multiplier(self) -> Decimal:
        if self.config.dynamic_order_spread:
            df = self.processed_data["features"]
            bb_width = df[f"BBB_{self.config.bb_length}_{self.config.bb_std}"].iloc[-1]
            return Decimal(bb_width / 200)
        else:
            return Decimal("1.0")
        


    def order_level_refresh_condition(self, executor):
        return self.market_data_provider.time() - executor.timestamp > self.config.executor_refresh_time * 1000

    def executors_to_refresh(self) -> List[ExecutorAction]:
        executors_to_refresh = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda x: not x.is_trading and x.is_active and (self.order_level_refresh_condition(x)))
        return [StopExecutorAction(
            controller_id=self.config.id,
            executor_id=executor.id) for executor in executors_to_refresh]

    def get_executor_config(self, trade_type: TradeType, price: Decimal, amount: Decimal ):
        
        spread, amounts_quote = self.config.get_spreads_and_amounts_in_quote(trade_type, amount * price)
        spread_multiplier = self.get_spread_multiplier()

        if trade_type == TradeType.BUY:
            prices = [price * (1 - spread * spread_multiplier) for spread in spread]
        else:
            prices = [price * (1 + spread * spread_multiplier) for spread in spread]
        stop_loss = max(self.config.min_stop_loss,
                        min(self.config.max_stop_loss, self.config.stop_loss * spread_multiplier))
        take_profit_activation_price = max(self.config.min_trailing_stop,
                                           min(self.config.max_trailing_stop,
                                               self.config.trailing_stop.activation_price * spread_multiplier))
        trailing_stop = TrailingStop(activation_price=take_profit_activation_price,
                                     trailing_delta=self.config.trailing_stop.trailing_delta * take_profit_activation_price)

        return DCAExecutorConfig(
            timestamp=self.market_data_provider.time(),
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            mode=DCAMode.MAKER,
            side=trade_type,
            prices=prices,
            amounts_quote=amounts_quote,
            time_limit=self.config.time_limit,
            stop_loss=stop_loss,
            take_profit=self.config.take_profit,
            trailing_stop=trailing_stop,
            activation_bounds=self.config.executor_activation_bounds,
            leverage=self.config.leverage,
        )
