import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, Any
from dotenv import load_dotenv

from core.services.mongodb_client import MongoDBClient
from geckoterminal_py import GeckoTerminalAsyncClient
from core.task_base import BaseTask

logging.basicConfig(level=logging.INFO)
load_dotenv()


class PoolsTask(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name=name, frequency=frequency, config=config)
        self.name = "pools_task"
        self.gt = GeckoTerminalAsyncClient()
        
        # Initialize MongoDB client with config
        mongodb_config = config.get('mongodb_config', {})
        self.mongo_client = MongoDBClient(
            username=mongodb_config.get('username'),
            password=mongodb_config.get('password'),
            host=mongodb_config.get('host'),
            port=mongodb_config.get('port'),
            database=mongodb_config.get('database'),
            debug_mode=False
        )

    async def pre_execute(self) -> None:
        """Pre-execution setup"""
        await self.mongo_client.connect()

    async def post_execute(self) -> None:
        """Post-execution cleanup"""
        await self.mongo_client.disconnect()

    def clean_pools(self, pools: pd.DataFrame) -> pd.DataFrame:
        """Clean and enrich pools dataframe with calculated metrics"""
        try:
            pools["fdv_usd"] = pd.to_numeric(pools["fdv_usd"])
            pools["volume_usd_h24"] = pd.to_numeric(pools["volume_usd_h24"])
            pools["reserve_in_usd"] = pd.to_numeric(pools["reserve_in_usd"])
            pools["pool_created_at"] = pd.to_datetime(pools["pool_created_at"]).dt.tz_localize(None)
            pools["base"] = pools["name"].apply(lambda x: x.split("/")[0].strip())
            pools["quote"] = pools["name"].apply(lambda x: x.split("/")[1].strip())
            
            # Calculate ratios only for non-zero denominators
            pools["volume_liquidity_ratio"] = pools.apply(
                lambda x: x["volume_usd_h24"] / x["reserve_in_usd"] if x["reserve_in_usd"] != 0 else 0, 
                axis=1
            )
            pools["fdv_liquidity_ratio"] = pools.apply(
                lambda x: x["fdv_usd"] / x["reserve_in_usd"] if x["reserve_in_usd"] != 0 else 0, 
                axis=1
            )
            pools["fdv_volume_ratio"] = pools.apply(
                lambda x: x["fdv_usd"] / x["volume_usd_h24"] if x["volume_usd_h24"] != 0 else 0, 
                axis=1
            )
            
            pools["transactions_h24_buys"] = pd.to_numeric(pools["transactions_h24_buys"])
            pools["transactions_h24_sells"] = pd.to_numeric(pools["transactions_h24_sells"])
            pools["price_change_percentage_h1"] = pd.to_numeric(pools["price_change_percentage_h1"])
            pools["price_change_percentage_h24"] = pd.to_numeric(pools["price_change_percentage_h24"])
            
            # Filter quote asset if specified
            if self.config.get('QUOTE_ASSET'):
                pools = pools[pools['quote'] == self.config['QUOTE_ASSET']]
                
            return pools
        except Exception as e:
            logging.error(f"Error cleaning pools data: {str(e)}")
            return pd.DataFrame()

    def filter_pools(self, pools: pd.DataFrame) -> pd.DataFrame:
        """Filter pools based on configured criteria"""
        try:
            min_date = datetime.now() - pd.Timedelta(days=self.config['MIN_POOL_AGE_DAYS'])
            
            filtered_pools = pools[
                (pools["pool_created_at"] > min_date) &
                (pools["fdv_usd"] >= self.config['MIN_FDV']) & 
                (pools["fdv_usd"] <= self.config['MAX_FDV']) &
                (pools["volume_usd_h24"] >= self.config['MIN_VOLUME_24H']) &
                (pools["reserve_in_usd"] >= self.config['MIN_LIQUIDITY']) &
                (pools["transactions_h24_buys"] >= self.config['MIN_TRANSACTIONS_24H']) & 
                (pools["transactions_h24_sells"] >= self.config['MIN_TRANSACTIONS_24H'])
            ]
            
            return filtered_pools
        except Exception as e:
            logging.error(f"Error filtering pools: {str(e)}")
            return pd.DataFrame()

    async def execute(self) -> Dict[str, Any]:
        """Main execution logic"""
        try:
            await self.pre_execute()

            # Fetch data
            trending_pools_df = await self.gt.get_top_pools_by_network(self.config['NETWORK'])
            new_pools_df = await self.gt.get_new_pools_by_network(self.config['NETWORK'])
            
            # Clean and filter data
            cleaned_trending = self.clean_pools(trending_pools_df.copy())
            cleaned_new = self.clean_pools(new_pools_df.copy())
            
            filtered_trending = self.filter_pools(cleaned_trending.copy())
            filtered_new = self.filter_pools(cleaned_new.copy())
            
            # Store data
            await self.mongo_client.add_pools_data(
                trending_pools_df=trending_pools_df,
                new_pools_df=new_pools_df,
                filtered_trending_pools_df=filtered_trending,
                filtered_new_pools_df=filtered_new
            )
            
            return {
                'trending_pools': trending_pools_df,
                'new_pools': new_pools_df,
                'filtered_trending_pools': filtered_trending,
                'filtered_new_pools': filtered_new
            }
            
        except Exception as e:
            logging.error(f"Error executing pools task: {str(e)}")
            return {
                'trending_pools': pd.DataFrame(),
                'new_pools': pd.DataFrame(),
                'filtered_trending_pools': pd.DataFrame(),
                'filtered_new_pools': pd.DataFrame()
            }
        finally:
            await self.post_execute()


async def main(config):
    pools_task = PoolsTask(
        name="pools_task",
        frequency=timedelta(minutes=1),
        config=config
    )
    await pools_task.execute()

if __name__ == "__main__":
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
    asyncio.run(main(config))