import asyncio
import pandas as pd
from datetime import datetime, timedelta

from geckoterminal_py import GeckoTerminalAsyncClient
from core.services.mongodb_client import MongoDBClient


def clean_pools(pools, config):
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
        if config.get('QUOTE_ASSET'):
            pools = pools[pools['quote'] == config['QUOTE_ASSET']]
            
        return pools
    except Exception as e:
        print(f"Error cleaning pools data: {str(e)}")
        return pd.DataFrame()


def filter_pools(pools, config):
    """Filter pools based on configured criteria"""
    try:
        min_date = datetime.now() - timedelta(days=config['MIN_POOL_AGE_DAYS'])
        
        filtered_pools = pools[
            (pools["pool_created_at"] > min_date) &
            (pools["fdv_usd"] >= config['MIN_FDV']) & 
            (pools["fdv_usd"] <= config['MAX_FDV']) &
            (pools["volume_usd_h24"] >= config['MIN_VOLUME_24H']) &
            (pools["reserve_in_usd"] >= config['MIN_LIQUIDITY']) &
            (pools["transactions_h24_buys"] >= config['MIN_TRANSACTIONS_24H']) & 
            (pools["transactions_h24_sells"] >= config['MIN_TRANSACTIONS_24H'])
        ]
        
        return filtered_pools
    except Exception as e:
        print(f"Error filtering pools: {str(e)}")
        return pd.DataFrame()


async def main():
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

    gt = GeckoTerminalAsyncClient()
    mongo_client = MongoDBClient()
    
    try:
        # Connect to MongoDB
        await mongo_client.connect()
        
        # Fetch data
        trending_pools_df = await gt.get_top_pools_by_network(config['NETWORK'])
        new_pools_df = await gt.get_new_pools_by_network(config['NETWORK'])
        
        # Process trending pools
        if not trending_pools_df.empty:
            print("\nTrending Pools DataFrame Info:")
            print(trending_pools_df.info())
            print("\nSample of trending pools data:")
            print(trending_pools_df.head())
            
            trending_pools_df = clean_pools(trending_pools_df, config)
            trending_pools_df = filter_pools(trending_pools_df, config)
            if not trending_pools_df.empty:
                print("\nCleaned Trending Pools DataFrame Info:")
                print(trending_pools_df.info())
                await mongo_client.add_trending_pools(trending_pools_df)
        
        # Process new pools
        if not new_pools_df.empty:
            print("\nNew Pools DataFrame Info:")
            print(new_pools_df.info())
            print("\nSample of new pools data:")
            print(new_pools_df.head())
            
            new_pools_df = clean_pools(new_pools_df, config)
            new_pools_df = filter_pools(new_pools_df, config)
            if not new_pools_df.empty:
                print("\nCleaned New Pools DataFrame Info:")
                print(new_pools_df.info())
                await mongo_client.add_new_pools(new_pools_df)
                
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        
    finally:
        await mongo_client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
