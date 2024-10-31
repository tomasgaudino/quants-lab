import asyncio
import logging
import os
import sys
from datetime import timedelta

from tasks.data_reporting.data_reporting_task import ReportGeneratorTask

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    from core.task_base import TaskOrchestrator
    from core.utils import load_dict_from_yaml
    from tasks.backtesting.xtreet_backtesting_task import BacktestingTask
    from tasks.data_collection.trades_downloader_task import TradesDownloaderTask
    orchestrator = TaskOrchestrator()

    backtesting_config = {
        'config': load_dict_from_yaml(file_name="binance_config.yml", folder=f"{project_root}/tasks/config"),
        'resolution': "1s",
        'root_path': project_root
    }
    backtesting_task = BacktestingTask("Backtesting", timedelta(hours=12), backtesting_config)

    trades_downloader_config = {
        'connector_name': 'binance_perpetual',
        'quote_asset': 'USDT',
        'min_notional_size': 10.0,
        'days_data_retention': 10
    }
    trades_downloader_task = TradesDownloaderTask("Trades Downloader Binance", timedelta(hours=5),
                                                  trades_downloader_config)

    report_task = ReportGeneratorTask(
        name="Report Generator",
        config={
            "host": "63.250.52.93",
            "port": 5432,
            "user": "admin",
            "password": "admin",
            "database": "timescaledb",
        },
        frequency=timedelta(hours=12))

    orchestrator.add_task(backtesting_task)
    orchestrator.add_task(trades_downloader_task)
    orchestrator.add_task(report_task)

    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
