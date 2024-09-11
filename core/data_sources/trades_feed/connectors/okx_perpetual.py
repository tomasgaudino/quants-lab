import asyncio
import logging
from time import time
from typing import Dict

import aiohttp
import pandas as pd

from core.data_sources.trades_feed.trades_feed_base import TradesFeedBase


class OkxPerpetualTradesFeed(TradesFeedBase):
    _base_url = "https://www.okx.com"
    _endpoints = {
        "historical_agg_trades": "/api/v5/market/history-trades"
    }
    _logger = None

    REQUEST_WEIGHT_LIMIT = 20
    REQUEST_WEIGHT = 1
    ONE_MINUTE = 2  # seconds

    def __init__(self):
        super().__init__()
        self._request_timestamps = []
        self._session = aiohttp.ClientSession()

    @classmethod
    def logger(cls):
        if cls._logger is None:
            cls._logger = logging.getLogger(__name__)
        return cls._logger

    def get_exchange_trading_pair(self, trading_pair: str) -> str:
        base, quote = trading_pair.split("-")
        return f"{base}-{quote}-SWAP"

    async def _get_historical_trades(self, trading_pair: str, start_time: int, end_time: int):
        all_trades_collected = False
        end_ts = int(end_time * 1000)
        start_ts = int(start_time * 1000)
        from_id = None
        all_trades = []
        ex_trading_pair = self.get_exchange_trading_pair(trading_pair)

        while not all_trades_collected:
            await self._enforce_rate_limit()  # Enforce rate limit before making a request

            params = {
                "instId": ex_trading_pair,
                "limit": 100,
                "type": 1 if from_id is not None else 2,
                "after": from_id if from_id is not None else end_ts,
            }

            trades_data = await self._get_historical_trades_request(params)
            trades = trades_data["data"]
            if trades:
                first_timestamp = int(trades[-1]["ts"])
                all_trades.extend(trades)
                all_trades_collected = first_timestamp <= start_ts
                from_id = trades[-1]["tradeId"]
            else:
                all_trades_collected = True

        df = pd.DataFrame(all_trades)
        df.rename(columns={"ts": "timestamp", "px": "price", "sz": "volume", "side": "sell_taker", "tradeId": "id"}, inplace=True)
        df.drop(columns=["instId"], inplace=True)
        df["timestamp"] = df["timestamp"].astype(int) / 1000
        df.index = pd.to_datetime(df["timestamp"], unit="s")
        df["price"] = df["price"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df

    async def _get_historical_trades_request(self, params: Dict):
        try:
            url = f"{self._base_url}{self._endpoints['historical_agg_trades']}"
            async with self._session.get(url, params=params) as response:
                response.raise_for_status()
                self._record_request()  # Record the timestamp of this request
                return await response.json()
        except aiohttp.ClientResponseError as e:
            self.logger().error(f"Error fetching historical trades for {params}: {e}")
            if e.status == 429:
                await asyncio.sleep(1)  # Sleep to respect rate limits
            raise e
        except Exception as e:
            self.logger().error(f"Error fetching historical trades for {params}: {e}")
            raise e

    async def _enforce_rate_limit(self):
        current_time = time()
        self._request_timestamps = [t for t in self._request_timestamps if t > current_time - self.ONE_MINUTE]

        # Calculate the current weight usage
        current_weight_usage = len(self._request_timestamps) * self.REQUEST_WEIGHT

        if current_weight_usage >= self.REQUEST_WEIGHT_LIMIT:
            # Calculate how long to sleep to stay within the rate limit
            sleep_time = self.ONE_MINUTE - (current_time - self._request_timestamps[0])
            self.logger().info(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
            await asyncio.sleep(sleep_time)

    def _record_request(self):
        """Records the timestamp of a request."""
        self._request_timestamps.append(time())
