from datetime import timedelta

import pandas as pd
from dotenv import load_dotenv
import logging
import asyncio
import os
from typing import Dict, Any, List

from core.data_sources import CLOBDataSource
from core.services.mongodb_client import MongoDBClient
from core.task_base import BaseTask

logging.getLogger("asyncio").setLevel(logging.CRITICAL)
load_dotenv()


class CombinedAnalysisTask(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name=name, frequency=frequency, config=config)
        self.mongo_client = MongoDBClient(**config.get("db_config", {}))
        self.clob = CLOBDataSource()
        self.collection_name = "combined_analysis"

    async def initialize(self):
        """Initialize connections and resources."""
        await self.mongo_client.connect()

    async def execute(self):
        """Main task execution logic."""
        try:
            await self.initialize()

            # Get latest cointegration and funding rate data from their respective collections
            cointegration_data = await self.mongo_client.db["cointegration"].find().sort("timestamp", -1).limit(1).to_list(length=None)
            funding_rates_data = await self.mongo_client.db["funding_rates_processed"].find().sort("timestamp", -1).limit(1).to_list(length=None)

            # Convert to DataFrames
            cointegration_df = pd.DataFrame(cointegration_data)
            funding_rates_df = pd.DataFrame(funding_rates_data)

            # Find matching pairs between cointegration and funding rates
            combined_results = []
            
            for _, coint_row in cointegration_df.iterrows():
                base_pair = coint_row['base']
                quote_pair = coint_row['quote']
                
                # Look for matching funding rate pairs
                funding_matches = funding_rates_df[
                    ((funding_rates_df['pair1'] == base_pair) & (funding_rates_df['pair2'] == quote_pair)) |
                    ((funding_rates_df['pair1'] == quote_pair) & (funding_rates_df['pair2'] == base_pair))
                ]

                if not funding_matches.empty:
                    for _, funding_row in funding_matches.iterrows():
                        combined_results.append({
                            'base_pair': base_pair,
                            'quote_pair': quote_pair,
                            'cointegration_value': float(coint_row['coint_value']),
                            'base_grid': coint_row['grid_base'],
                            'quote_grid': coint_row['grid_quote'],
                            'funding_rate_difference': float(funding_row['rate_difference']),
                            'timestamp': funding_row['timestamp']
                        })

            # Store combined results directly in MongoDB using the collection
            if combined_results:
                await self.mongo_client.db[self.collection_name].insert_many(combined_results)
                logging.info(f"Successfully added {len(combined_results)} combined analysis records")
            else:
                logging.info("No matching pairs found between cointegration and funding rates")

        except Exception as e:
            logging.error(f"Error in CombinedAnalysisTask: {str(e)}")
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
        "db_config": mongodb_config
    }
    task = CombinedAnalysisTask(name="combined_analysis_task",
                               frequency=timedelta(hours=1),
                               config=task_config)
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main())

