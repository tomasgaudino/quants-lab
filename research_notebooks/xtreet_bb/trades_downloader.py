import os
import sys

import pandas as pd
import asyncio
import logging
import time
from decimal import Decimal
from aiohttp import ClientResponseError

root_path = os.path.abspath(os.path.join(os.getcwd(), '../..'))
sys.path.append(root_path)
logging.basicConfig(filename='trades_downloader.log')

from core.data_sources import CLOBDataSource
from core.data_sources.trades_feed.connectors.binance_perpetual import BinancePerpetualTradesFeed


async def main():
    logging.info(f"Starting trades downloader at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    clob = CLOBDataSource()
    trades_feed = BinancePerpetualTradesFeed()
    trading_rules = await clob.get_trading_rules("binance_perpetual")
    trading_pairs = trading_rules.filter_by_quote_asset("USDT") \
        .filter_by_min_notional_size(Decimal(str(10))) \
        .get_all_trading_pairs()
    DAYS_TO_DOWNLOAD = 1
    i = 0
    for trading_pair in trading_pairs:
        i += 1
        logging.info(f"Fetching trades for {trading_pair} [{i} from {len(trading_pairs)}]")
        try:
            input_path = os.path.join(root_path, "data/candles", f"binance_perpetual|{trading_pair}|1s.csv")
            base = pd.read_csv(input_path)
            first_trade_id = base['first_trade_id'].max()
            base = base[base['first_trade_id'] < first_trade_id]
            trades = await trades_feed._get_historical_trades(trading_pair, end_time=time.time(), start_time=None, from_id=first_trade_id)
            pandas_interval = clob.convert_interval_to_pandas_freq("1s")
            candles_df = trades.resample(pandas_interval).agg(
                {"price": "ohlc", "volume": "sum", 'id': 'first'}).ffill()
            candles_df.columns = candles_df.columns.droplevel(0)
            candles_df.rename(columns={'id': 'first_trade_id'}, inplace=True)
            candles_df["timestamp"] = pd.to_numeric(candles_df.index) // 1e9
            candles_df = pd.concat([base, candles_df])
            output_path = os.path.join(root_path, "data/candles", f"binance_perpetual|{trading_pair}|1s.csv")
            candles_df.to_csv(output_path, index=False)

        except FileNotFoundError:
            logging.info(f"Archivo no encontrado: data/candles/candles/binance_perpetual|{trading_pair}|1s.csv")
            try:
                trades = await clob.get_candles_last_days(connector_name="binance_perpetual", trading_pair=trading_pair,
                                                          interval="1s", days=DAYS_TO_DOWNLOAD, from_trades=True)
                output_path = os.path.join(root_path, "data/candles", f"binance_perpetual|{trading_pair}|1s.csv")
                trades.data.to_csv(output_path, index=False)
            except ClientResponseError as ce:
                logging.exception(f"ClientResponseError for trading pair {trading_pair}:\n {ce}")
                print(ce)
                continue
            continue

        except Exception as e:
            print(e)
            logging.exception(f"An error occurred during the data load for trading pair {trading_pair}:\n {e}")
            continue


if __name__ == "__main__":
    asyncio.run(main())