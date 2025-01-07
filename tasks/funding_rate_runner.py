import asyncio
import logging
import os
import sys
from datetime import timedelta
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    from core.task_base import TaskOrchestrator
    from tasks.data_collection.funding_rates_task import FundingRatesTask

    orchestrator = TaskOrchestrator()

    mongodb_config = {
        "username": os.getenv('MONGO_INITDB_ROOT_USERNAME', "admin"),
        "password": os.getenv('MONGO_INITDB_ROOT_PASSWORD', "admin"),
        "host": os.getenv('MONGO_HOST', 'localhost'),
        "port": os.getenv('MONGO_PORT', 27017),
        "database": "mongodb"
    }

    config = {
        "db_config": mongodb_config,
        "connector_names": ["binance_perpetual"],
        "quote_asset": "USDT",
    }

    pools_task = FundingRatesTask(
        name="Funding Rate Collector",
        frequency=timedelta(minutes=10),
        config=config
    )

    orchestrator.add_task(pools_task)
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
