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


class CointegrationTask(BaseTask):
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
            cointegration_results: List[Dict[str, Any]] = {}
            await self.mongo_client.add_funding_rates_data(cointegration_results)
            logging.info(f"Successfully added {len(cointegration_results)} cointegration records")

        except Exception as e:
            logging.error(f"Error in Cointegration Task: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup resources."""
        await self.mongo_client.disconnect()


async def main():
    task_config = {
        "connector_names": ["binance_perpetual"],
        "quote_asset": "USDT",
    }
    task = CointegrationTask(name="cointegration_task",
                             frequency=timedelta(hours=1),
                             config=task_config)
    asyncio.run(task.execute())


if __name__ == "__main__":
    asyncio.run(main())
