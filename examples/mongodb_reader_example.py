import asyncio
from core.services.mongodb_client import MongoDBClient
import pandas as pd


async def print_dataframe_info(name: str, df: pd.DataFrame):
    """Helper function to print DataFrame information"""
    print(f"\n{name} DataFrame Info:")
    if df.empty:
        print(f"No data found for {name}")
    else:
        print(f"Shape: {df.shape}")
        print("\nColumns:", df.columns.tolist())
        print("\nSample data:")
        print(df.head())
        print("\nBasic statistics:")
        if 'fdv_usd' in df.columns:
            print(df[['fdv_usd', 'volume_usd_h24', 'reserve_in_usd']].describe())


async def main():
    mongo_client = MongoDBClient(debug_mode=False)  # Set to False to keep historical data
    
    try:
        await mongo_client.connect()
        
        # Get latest snapshot
        print("\n=== Latest Data Snapshot ===")
        latest_data = await mongo_client.get_latest_pools_data()
        
        if latest_data['timestamp']:
            print(f"\nTimestamp: {latest_data['timestamp']}")
            
            for key in ['trending_pools', 'filtered_trending_pools', 'new_pools', 'filtered_new_pools']:
                await print_dataframe_info(key, latest_data[key])
        
        # Get historical data
        print("\n=== Historical Data (Last 24 Hours) ===")
        historical_data = await mongo_client.get_pools_data(hours_ago=24)
        
        print(f"\nNumber of timestamps: {len(historical_data['timestamps'])}")
        if historical_data['timestamps']:
            print("Time range:", min(historical_data['timestamps']), "to", max(historical_data['timestamps']))
            
            for key in ['trending_pools', 'filtered_trending_pools', 'new_pools', 'filtered_new_pools']:
                df = historical_data[key]
                print(f"\n{key}:")
                print(f"Total records: {len(df)}")
                if not df.empty:
                    print("Records per timestamp:")
                    print(df.groupby('timestamp').size())
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        
    finally:
        await mongo_client.disconnect()


if __name__ == "__main__":
    asyncio.run(main()) 