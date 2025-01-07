from datetime import timedelta
from dotenv import load_dotenv
import logging
import time
import aiohttp
import asyncio
import pandas as pd
from typing import List, Dict, Any

from core.data_structures.trading_rules import TradingRules
from core.data_sources import CLOBDataSource
from core.services.mongodb_client import MongoDBClient
from core.task_base import BaseTask

logging.getLogger("asyncio").setLevel(logging.CRITICAL)
load_dotenv()


class FundingRatesTask(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name=name, frequency=frequency, config=config)
        self.mongo_client = MongoDBClient()

    async def get_trading_pairs(self) -> List[str]:
        clob = CLOBDataSource()
        trading_rules: TradingRules = await clob.get_trading_rules(
            connector_name=self.config.get("connector_name", "binance_perpetual")
        )
        return trading_rules.filter_by_quote_asset(
            self.config.get("quote_asset", "USDT")
        ).get_all_trading_pairs()

    async def get_binance_perpetual_funding_rates_history(
        self,
        trading_pairs: list,
        start_time: int = None,
        end_time: int = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Fetch funding rate history for multiple trading pairs from Binance Futures.

        Args:
            trading_pairs (list): List of trading pairs (e.g., ["BTCUSDT", "ETHUSDT"])
            start_time (int): Start time in milliseconds
            end_time (int): End time in milliseconds
            limit (int): Number of records to fetch (default 1000, max 1000)

        Returns:
            list: List of funding rate records
        """
        base_url = "https://fapi.binance.com/fapi/v1/fundingRate"

        async def fetch_pair_history(session: aiohttp.ClientSession, trading_pair: str) -> list:
            params = {
                "symbol": trading_pair.replace("-", ""),
                "limit": limit
            }
            if start_time is not None:
                params["startTime"] = start_time
            if end_time is not None:
                params["endTime"] = end_time

            try:
                async with session.get(base_url, params=params) as response:
                    if response.status == 200:
                        funding_rates = await response.json()
                        for funding_rate in funding_rates:
                            funding_rate["symbol"] = trading_pair
                        return funding_rates
                    else:
                        logging.error(f"Error fetching {trading_pair}: {response.status}")
                        return []
            except Exception as e:
                logging.error(f"Exception fetching {trading_pair}: {str(e)}")
                return []

        async with aiohttp.ClientSession() as session:
            tasks = [fetch_pair_history(session, pair) for pair in trading_pairs]
            results = await asyncio.gather(*tasks)

        # Flatten results and sort by fundingTime
        all_rates = []
        for rates in results:
            all_rates.extend(rates)

        df = pd.DataFrame(all_rates)
        df.rename(columns={
            "symbol": "trading_pair",
            "fundingTime": "funding_time",
            "fundingRate": "funding_rate",
            "markPrice": "mark_price"
        }, inplace=True)
        df["funding_rate"] = df["funding_rate"].astype(float)
        df["connector_name"] = "binance_perpetual"
        df["timestamp"] = time.time()
        return df.to_dict(orient="records")

    async def initialize(self):
        """Initialize connections and resources."""
        await self.mongo_client.connect()

    async def execute(self):
        """Main task execution logic."""
        try:
            trading_pairs = await self.get_trading_pairs()

            funding_rates_history = await self.get_binance_perpetual_funding_rates_history(
                trading_pairs, 
                limit=1
            )

            await self.mongo_client.add_funding_rates_data(funding_rates_history)
            logging.info(f"Successfully added {len(funding_rates_history)} funding rate records")

        except Exception as e:
            logging.error(f"Error in FundingRatesTask: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup resources."""
        await self.mongo_client.disconnect()


async def main():
    task_config = {
        "connector_name": "binance_perpetual",
        "min_volume": 1_000_000,
        "quote_asset": "USDT",
    }
    task = FundingRatesTask(name="funding_rate_task",
                            frequency=timedelta(hours=1),
                            config=task_config)
    asyncio.run(task.execute())


if __name__ == "__main__":
    asyncio.run(main())
