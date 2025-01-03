import asyncio
from core.services.mongodb_client import MongoDBClient

async def main():
    mongo_client = MongoDBClient()
    
    try:
        await mongo_client.connect()
        
        # Get all trending pools
        all_trending = await mongo_client.get_trending_pools()
        print("\nAll trending pools shape:", all_trending.shape)
        
        # Get trending pools from last 6 hours
        recent_trending = await mongo_client.get_trending_pools(hours_ago=6)
        print("\nRecent trending pools shape:", recent_trending.shape)
        
        # Get all latest data
        latest_data = await mongo_client.get_latest_data()
        for collection_name, df in latest_data.items():
            print(f"\n{collection_name} shape:", df.shape)
            if not df.empty:
                print(f"\nSample of {collection_name}:")
                print(df.head())
        
    finally:
        await mongo_client.disconnect()

if __name__ == "__main__":
    asyncio.run(main()) 