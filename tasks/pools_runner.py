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
    from tasks.data_collection.pools_task import PoolsTask

    orchestrator = TaskOrchestrator()

    config = {
        'MIN_FDV': 70_000,
        'MAX_FDV': 5_000_000,
        'MIN_POOL_AGE_DAYS': 2,
        'MIN_VOLUME_24H': 150_000,
        'MIN_LIQUIDITY': 50_000,
        'MIN_TRANSACTIONS_24H': 300,
        'NETWORK': "solana",
        'QUOTE_ASSET': "SOL"
    }

    pools_task = PoolsTask(
        name="Pools Data Collector",
        frequency=timedelta(minutes=1),  # Run every 15 minutes
        config=config
    )
    
    orchestrator.add_task(pools_task)
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
