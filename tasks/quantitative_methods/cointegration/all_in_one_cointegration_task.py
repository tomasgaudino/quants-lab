import asyncio
import logging
import os
from dotenv import load_dotenv
from datetime import timedelta

from core.services.mongodb_client import MongoDBClient
from core.task_base import BaseTask

from tasks.data_collection.funding_rates_task import FundingRatesTask
from tasks.quantitative_methods.cointegration.cointegration_task import CointegrationTask
from tasks.quantitative_methods.cointegration.stat_arb_config_generator_task import StatArbConfigGeneratorTask


class AllInOneTask(BaseTask):
    def __init__(self, name: str, frequency: str, config: dict):
        super().__init__(name, frequency, config)

    async def execute(self):
        try:
            coint_task = CointegrationTask("cointegration_task",
                                           frequency=timedelta(hours=1),
                                           config=self.config["cointegration"])
            funding_task = FundingRatesTask("funding_rate_task",
                                            frequency=timedelta(hours=1),
                                            config=self.config["funding_rate"])
            stat_arb_config_gen_task = StatArbConfigGeneratorTask("stat_arb_config_generator_task",
                                                                  frequency=timedelta(hours=1),
                                                                  config=self.config["stat_arb_config_generator"])
            await coint_task.execute()
            await funding_task.execute()
            await stat_arb_config_gen_task.execute()
        except Exception as e:
            logging.error(e)
