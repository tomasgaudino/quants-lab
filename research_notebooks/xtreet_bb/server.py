import os
import sys
import asyncio
import pandas as pd
import time
import pickle
from decimal import Decimal
import warnings
warnings.filterwarnings("ignore")


root_path = os.path.abspath(os.path.join(os.getcwd(), '../..'))
sys.path.append(root_path)


from core.features.candles.volatility import VolatilityConfig
from research_notebooks.xtreet_bb.utils import generate_config, dump_dict_to_yaml
from core.features.candles.volume import VolumeConfig
from research_notebooks.xtreet_bb.utils import generate_screener_report
from core.backtesting import BacktestingEngine
from research_notebooks.xtreet_bb.utils import read_yaml_to_dict
from core.data_sources import CLOBDataSource
from controllers.directional_trading.xtreet_bb import XtreetBBControllerConfig
from research_notebooks.xtreet_bb.xtreet_bt import XtreetBacktesting

# General parameters
FETCH_CANDLES = True
DUMP_CONFIGS = True
BACKTEST = True

# Screener parameters
CONNECTOR_NAME = "okx_perpetual"
INTERVALS = ["1m"]
DAYS = 7
BATCH_CANDLES_REQUEST = 1
SLEEP_REQUEST = 2.0
VOLUME_THRESHOLD = 0.3  # From percentile VOLUME_THRESHOLD to 1
VOLATILITY_THRESHOLD = 0.3  # From percentile VOLATILITY_THRESHOLD to 1

# Trading Rules Filter
QUOTE_ASSET = "USDT"
MIN_NOTIONAL_SIZE = 10  # In USDT
MAX_PRICE_STEP = 0.001  # Min price step in % (tick size)

VOLATILITY_WINDOW = 60  # In bars
VOLUME_FAST_WINDOW = 20  # No se usa
VOLUME_SLOW_WINDOW = 100  # No se usa

# Config generation
TOTAL_AMOUNT = 1000  # General total amount for all markets
ACTIVATION_BOUNDS = 0.002  # Input activation bounds
MAX_EXECUTORS_PER_SIDE = 1  # Maximum number of executors per side
COOLDOWN_TIME = 0
LEVERAGE = 20 # Should be for each trading pair
TIME_LIMIT = 60 * 60 * 24
BOLLINGER_LENGTHS = [50, 100, 150, 200]
BOLLINGER_STDS = [1.0, 1.4, 1.8, 2.2, 2.6]
SL_STD_MULTIPLIER = 1
TS_DELTA_MULTIPLIER = 0.2
MAX_DCA_AMOUNT_RATIO = 4 # Amount 2 / Amount 1


# Config filtering
MIN_DISTANCE_BETWEEN_ORDERS = 0.01
MAX_TS_SL_RATIO = 0.5

# Backtesting variables
EXPERIMENT_NAME = pd.to_datetime(time.time()).strftime('%Y-%m-%d %H:%M:%S')
TRADE_COST = 0.0007
BACKTESTING_RESOLUTION = "1m"


async def main():
    clob = CLOBDataSource()
    trading_rules = await clob.get_trading_rules(CONNECTOR_NAME)
    trading_pairs = trading_rules.filter_by_quote_asset(QUOTE_ASSET) \
        .filter_by_min_notional_size(Decimal(MIN_NOTIONAL_SIZE)) \
        .get_all_trading_pairs()

    if FETCH_CANDLES:
        number_of_calls = (len(trading_pairs) // BATCH_CANDLES_REQUEST) + 1

        all_candles = {}

        for i in range(number_of_calls):
            try:
                print(f"Batch {i + 1}/{number_of_calls}")
                start = i * BATCH_CANDLES_REQUEST
                end = (i + 1) * BATCH_CANDLES_REQUEST
                print(f"Start: {start}, End: {end}")
                end = min(end, len(trading_pairs))
                trading_pairs_batch = trading_pairs[start:end]

                tasks = [clob.get_candles_last_days(
                    connector_name=CONNECTOR_NAME,
                    trading_pair=trading_pair,
                    interval=interval,
                    days=DAYS,

                ) for trading_pair in trading_pairs_batch for interval in INTERVALS]

                candles = await asyncio.gather(*tasks)
                candles = {trading_pair: candle for trading_pair, candle in zip(trading_pairs, candles)}
                all_candles.update(candles)
                if i != number_of_calls - 1:
                    print(f"Sleeping for {SLEEP_REQUEST} seconds")
                    await asyncio.sleep(SLEEP_REQUEST)
            except Exception as e:
                print(f"Error in batch {i + 1}: {e}")
                continue
        clob.dump_candles_cache(os.path.join(root_path, "data"))
    else:
        clob.load_candles_cache(os.path.join(root_path, "data"))
    candles = [value for key, value in clob.candles_cache.items() if key[2] in INTERVALS and key[0] == CONNECTOR_NAME]


    screner_report = generate_screener_report(
        candles=candles,
        trading_rules=trading_rules,
        volatility_config=VolatilityConfig(window=VOLATILITY_WINDOW),
        volume_config=VolumeConfig(short_window=VOLUME_FAST_WINDOW, long_window=VOLUME_FAST_WINDOW))
    screner_report["url"] = screner_report["trading_pair"].apply(lambda x: f"https://www.okx.com/trade-swap/{x}-swap")
    screner_report.sort_values("mean_natr", ascending=False, inplace=True)
    screner_report.to_csv(f"screener_report_{pd.to_datetime(time.time()).strftime('%Y-%m-%d %H:%M:%S')}.csv", index=False)

    # Calculate the 20th percentile (0.2 quantile) for both columns
    natr_percentile = screner_report['mean_natr'].quantile(VOLATILITY_THRESHOLD)
    volume_percentile = screner_report['average_volume_per_hour'].quantile(VOLUME_THRESHOLD)

    # Filter the DataFrame to get observations where mean_natr is greater than its 20th percentile
    # and average_volume_per_hour is greater than its 20th percentile
    screener_top_markets = screner_report[
        (screner_report['mean_natr'] > natr_percentile) &
        (screner_report['average_volume_per_hour'] > volume_percentile) &
        (screner_report["price_step_pct"] < MAX_PRICE_STEP)
    ].sort_values(by="average_volume_per_hour")

    strategy_configs = generate_config(
        connector_name=CONNECTOR_NAME,
        intervals=INTERVALS,
        screener_top_markets=screener_top_markets,
        candles=candles,
        total_amount=TOTAL_AMOUNT,
        max_executors_per_side=MAX_EXECUTORS_PER_SIDE,
        cooldown_time=COOLDOWN_TIME,
        leverage=LEVERAGE,
        time_limit=TIME_LIMIT,
        bb_lengths=BOLLINGER_LENGTHS,
        bb_stds=BOLLINGER_STDS,
        sl_std_multiplier=SL_STD_MULTIPLIER,
        min_distance_between_orders=MIN_DISTANCE_BETWEEN_ORDERS,
        max_ts_sl_ratio=MAX_TS_SL_RATIO,
        ts_delta_multiplier=TS_DELTA_MULTIPLIER,
        max_dca_amount_ratio=MAX_DCA_AMOUNT_RATIO
    )

    print(f"Total Trading Pairs: {len(set([config['trading_pair'] for config in strategy_configs]))}")
    print(f"Total Configs: {len(strategy_configs)}")

    if DUMP_CONFIGS:
        for config in strategy_configs:
            dump_dict_to_yaml("configs/", config)

    if BACKTEST:
        controllers_conf_dir_path = os.path.join(root_path, "research_notebooks", "xtreet_bb", "configs")
        config_files = os.listdir(controllers_conf_dir_path)
        bt_results = []

        for i, config_file in enumerate(config_files):
            try:
                print(f"Experiment {i}/{len(config_files)}: {config_file}")
                controller_config = read_yaml_to_dict(os.path.join(controllers_conf_dir_path, config_file))
                config = XtreetBBControllerConfig(**controller_config)
                connector = config.connector_name
                trading_pair = config.trading_pair
                backtesting_engine = BacktestingEngine(root_path=root_path, load_cached_data=True)

                start_time = \
                backtesting_engine._dt_bt.backtesting_data_provider.candles_feeds[f"{connector}_{trading_pair}_1m"][
                    "timestamp"].min() + 10
                end_time = backtesting_engine._dt_bt.backtesting_data_provider.candles_feeds[f"{connector}_{trading_pair}_1m"][
                               "timestamp"].max() - 10

                custom_backtester = XtreetBacktesting()
                custom_backtester.backtesting_data_provider.start_time = start_time
                custom_backtester.backtesting_data_provider.end_time = end_time
                custom_backtester.backtesting_data_provider.candles_feeds = backtesting_engine._dt_bt.backtesting_data_provider.candles_feeds

                backtesting_result = await backtesting_engine.run_backtesting(
                    config=config,
                    trade_cost=TRADE_COST,
                    start=int(start_time),
                    end=int(end_time),
                    backtesting_resolution=BACKTESTING_RESOLUTION,
                    backtester=custom_backtester,
                )
                print(backtesting_result.get_results_summary())
                bt_results.append(backtesting_result)
            except Exception as e:
                print(f"Error with {config_file}: {e}")
                # raise
        with open(f"bt_results_{EXPERIMENT_NAME}.pkl", "wb") as f:
            pickle.dump(bt_results, f)


if __name__ == "__main__":
    asyncio.run(main())
