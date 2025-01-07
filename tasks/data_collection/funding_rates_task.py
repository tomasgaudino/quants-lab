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
        self.mongo_client = MongoDBClient(**config.get("db_config", {}))
        self.clob = CLOBDataSource()

    async def initialize(self):
        """Initialize connections and resources."""
        await self.mongo_client.connect()

    async def execute(self):
        """Main task execution logic."""
        try:
            await self.initialize()
            for connector_name in self.config.get("connector_names", ["binance_perpetual"]):
                connector = self.clob.get_connector(connector_name)
                trading_rules: TradingRules = await self.clob.get_trading_rules(
                    connector_name=self.config.get("connector_name", "binance_perpetual")
                )
                trading_pairs = trading_rules.filter_by_quote_asset(
                    self.config.get("quote_asset", "USDT")
                ).get_all_trading_pairs()

                tasks = []
                for trading_pair in trading_pairs:
                    tasks.append(connector._orderbook_ds.get_funding_info(trading_pair))

                funding_rates_response = await asyncio.gather(*tasks)
                funding_rates = []
                timestamp = time.time()
                for funding_rate in funding_rates_response:
                    funding_rates.append({
                        "index_price": float(funding_rate.index_price),
                        "mark_price": float(funding_rate.mark_price),
                        "next_funding_utc_timestamp": funding_rate.next_funding_utc_timestamp,
                        "rate": float(funding_rate.rate),
                        "trading_pair": funding_rate.trading_pair,
                        "connector_name": connector_name,
                        "timestamp": timestamp
                    })

                await self.mongo_client.add_funding_rates_data(funding_rates)
                logging.info(f"Successfully added {len(funding_rates)} funding rate records")

        except Exception as e:
            logging.error(f"Error in FundingRatesTask: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup resources."""
        await self.mongo_client.disconnect()


async def main():
    mongodb_config = {
        "username": os.getenv('MONGO_INITDB_ROOT_USERNAME', "admin"),
        "password": os.getenv('MONGO_INITDB_ROOT_PASSWORD', "admin"),
        "host": os.getenv('MONGO_HOST', 'localhost'),
        "port": os.getenv('MONGO_PORT', 27017),
        "database": "mongodb"
    }
    task_config = {
        "connector_names": ["binance_perpetual"],
        "quote_asset": "USDT",
        "db_config": mongodb_config
    }
    task = FundingRatesTask(name="funding_rate_task",
                            frequency=timedelta(hours=1),
                            config=task_config)
    asyncio.run(task.execute())


if __name__ == "__main__":
    asyncio.run(main())
