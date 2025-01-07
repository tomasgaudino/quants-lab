from typing import List, Optional, Dict, Any
import pandas as pd
from dotenv import load_dotenv
import logging
from motor.motor_asyncio import AsyncIOMotorClient
import os
from datetime import datetime, timedelta


class MongoDBClient:
    def __init__(
        self, 
        debug_mode: bool = False,
        username: Optional[str] = None,
        password: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None,
        database: str = "mongodb"
    ):
        self.client = None
        self.db = None
        self.debug_mode = debug_mode
        load_dotenv()
        
        # Connection parameters with env fallbacks
        self.username = username or os.getenv('MONGO_INITDB_ROOT_USERNAME', "admin")
        self.password = password or os.getenv('MONGO_INITDB_ROOT_PASSWORD', "admin")
        self.host = host or os.getenv('MONGO_HOST', 'localhost')
        self.port = port or os.getenv('MONGO_PORT', '27017')
        self.database = database
        
    async def connect(self):
        """Connect to MongoDB using provided or environment variables."""
        connection_string = f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/?authSource=admin"
        try:
            self.client = AsyncIOMotorClient(
                connection_string,
                serverSelectionTimeoutMS=5000
            )
            self.db = self.client[self.database]
            await self.db.command('ping')
            logging.info(f"Successfully connected to MongoDB at {self.host}:{self.port}")
            
            # Create index on timestamp if it doesn't exist
            await self.db.pools.create_index('timestamp', unique=True)
            
            # If in debug mode, reset collections
            if self.debug_mode:
                await self.reset_collections()
            
        except Exception as e:
            print(f"Failed to connect to MongoDB: {str(e)}")
            raise

    async def reset_collections(self):
        """Reset all collections in the database."""
        try:
            result = await self.db.pools.delete_many({})
            print(f"Reset collections: Deleted {result.deleted_count} documents from pools collection")
        except Exception as e:
            print(f"Error resetting collections: {str(e)}")
            raise

    async def disconnect(self):
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            print("Disconnected from MongoDB")

    async def add_pools_data(self, trending_pools_df: pd.DataFrame, new_pools_df: pd.DataFrame,
                            filtered_trending_pools_df: pd.DataFrame, filtered_new_pools_df: pd.DataFrame):
        """
        Add all pools data to MongoDB in a single document with timestamp as primary key.
        """
        collection = self.db.pools
        timestamp = datetime.utcnow()
        
        document = {
            'timestamp': timestamp,
            'trending_pools': trending_pools_df.to_dict('records') if not trending_pools_df.empty else [],
            'filtered_trending_pools': filtered_trending_pools_df.to_dict('records') if not filtered_trending_pools_df.empty else [],
            'new_pools': new_pools_df.to_dict('records') if not new_pools_df.empty else [],
            'filtered_new_pools': filtered_new_pools_df.to_dict('records') if not filtered_new_pools_df.empty else []
        }
        
        try:
            await collection.insert_one(document)
            print(f"Successfully inserted pools data for timestamp {timestamp}")
            
            # Verify counts
            print(f"Trending pools: {len(document['trending_pools'])}")
            print(f"Filtered trending pools: {len(document['filtered_trending_pools'])}")
            print(f"New pools: {len(document['new_pools'])}")
            print(f"Filtered new pools: {len(document['filtered_new_pools'])}")
            
        except Exception as e:
            print(f"Error inserting pools data: {str(e)}")
            raise

    async def get_pools_data(self, hours_ago: int = None) -> dict:
        """
        Get pools data from MongoDB.
        Args:
            hours_ago: If provided, only return data from the last N hours
        Returns:
            Dictionary containing DataFrames for each pool type
        """
        collection = self.db.pools
        query = {}
        
        if hours_ago is not None:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_ago)
            query = {'timestamp': {'$gte': cutoff_time}}
            
        try:
            cursor = collection.find(query).sort('timestamp', -1)  # Sort by timestamp descending
            documents = await cursor.to_list(length=None)
            
            if not documents:
                print("No pools data found")
                return {
                    'trending_pools': pd.DataFrame(),
                    'filtered_trending_pools': pd.DataFrame(),
                    'new_pools': pd.DataFrame(),
                    'filtered_new_pools': pd.DataFrame(),
                    'timestamps': []
                }
            
            # Separate the data into different DataFrames
            result = {
                'trending_pools': pd.DataFrame(),
                'filtered_trending_pools': pd.DataFrame(),
                'new_pools': pd.DataFrame(),
                'filtered_new_pools': pd.DataFrame(),
                'timestamps': [doc['timestamp'] for doc in documents]
            }
            
            # Combine all documents into single DataFrames
            for key in ['trending_pools', 'filtered_trending_pools', 'new_pools', 'filtered_new_pools']:
                all_records = []
                for doc in documents:
                    records = doc[key]
                    for record in records:
                        record['timestamp'] = doc['timestamp']
                    all_records.extend(records)
                
                if all_records:
                    result[key] = pd.DataFrame(all_records)
            
            print(f"Retrieved data for {len(documents)} timestamps")
            for key, df in result.items():
                if isinstance(df, pd.DataFrame):
                    print(f"{key}: {len(df)} records")
                    
            return result
            
        except Exception as e:
            print(f"Error retrieving pools data: {str(e)}")
            raise

    async def get_latest_pools_data(self) -> dict:
        """
        Get the most recent pools data entry.
        Returns:
            Dictionary containing DataFrames for each pool type from the latest entry
        """
        collection = self.db.pools
        
        try:
            document = await collection.find_one(sort=[('timestamp', -1)])
            
            if not document:
                print("No pools data found")
                return {
                    'trending_pools': pd.DataFrame(),
                    'filtered_trending_pools': pd.DataFrame(),
                    'new_pools': pd.DataFrame(),
                    'filtered_new_pools': pd.DataFrame(),
                    'timestamp': None
                }
            
            result = {
                'timestamp': document['timestamp'],
                'trending_pools': pd.DataFrame(document['trending_pools']),
                'filtered_trending_pools': pd.DataFrame(document['filtered_trending_pools']),
                'new_pools': pd.DataFrame(document['new_pools']),
                'filtered_new_pools': pd.DataFrame(document['filtered_new_pools'])
            }
            
            print(f"Retrieved latest data from {document['timestamp']}")
            for key, df in result.items():
                if isinstance(df, pd.DataFrame):
                    print(f"{key}: {len(df)} records")
                    
            return result
            
        except Exception as e:
            print(f"Error retrieving latest pools data: {str(e)}")
            raise

    async def add_funding_rates_data(self, funding_rates: List[Dict[str, Any]]) -> None:
        """
        Add funding rates data to MongoDB.
        
        Args:
            funding_rates (List[Dict]): List of funding rate records with structure:
            {
                "index_price": float,
                "mark_price": float,
                "next_funding_utc_timestamp": int,
                "rate": float,
                "trading_pair": str,
                "connector_name": str,
                "timestamp": float
            }
        """
        try:
            if not funding_rates:
                logging.warning("No funding rates data to insert")
                return

            collection = self.db.funding_rates
            
            # Create indexes if they don't exist
            await collection.create_index([
                ("trading_pair", 1),
                ("connector_name", 1),
                ("next_funding_utc_timestamp", 1)
            ])
            
            # Insert the funding rates data
            result = await collection.insert_many(funding_rates)
            logging.info(f"Successfully inserted {len(result.inserted_ids)} funding rate records")
            
        except Exception as e:
            logging.error(f"Error adding funding rates data: {str(e)}")
            raise
