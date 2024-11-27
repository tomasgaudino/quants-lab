import asyncio
import logging
import os
import sys
from datetime import timedelta

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    from core.task_base import TaskOrchestrator
    from tasks.backtesting.macdmt_backtesting_task import MACDMTBacktestingTask
    from tasks.backtesting.macdmt_backtesting_task_fixed_pct import MACDMTBacktestingTaskFixedPCT

    orchestrator = TaskOrchestrator()

    timescale_config = {
        "host": os.getenv("TIMESCALE_HOST", "63.250.52.93"),
        "port": os.getenv("TIMESCALE_PORT", 5432),
        "user": os.getenv("TIMESCALE_USER", "admin"),
        "password": os.getenv("TIMESCALE_PASSWORD", "admin"),
        "database": os.getenv("TIMESCALE_DB", "timescaledb")
    }
    optuna_config = {
        "host": os.getenv("OPTUNA_HOST", "63.250.52.93"),
        "port": os.getenv("OPTUNA_PORT", 5433),
        "user": os.getenv("OPTUNA_USER", "admin"),
        "password": os.getenv("OPTUNA_PASSWORD", "admin"),
        "database": os.getenv("OPTUNA_DB", "optimization_database")
    }

    config = {
        "root_path": os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
        "resolution": "1s",
        "backtesting_days": 7, 
        "connector_name": "binance_perpetual",
        "timescale_config": timescale_config,
        "optuna_config": optuna_config,
        "selected_pairs": ['1000BONK-USDT', '1000PEPE-USDT', "HIGH-USDT"],
        "trials_per_pair": 5
    }



    #backtesting_task = MACDMTBacktestingTaskFixedPCT("Backtesting", timedelta(hours=12), config)
    #orchestrator.add_task(backtesting_task)

    
    backtesting_task = MACDMTBacktestingTask("Backtesting", timedelta(hours=12), config)
    orchestrator.add_task(backtesting_task)

    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
