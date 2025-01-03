from typing import List
import pandas as pd
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import os
from datetime import datetime, timedelta


class MongoDBClient:
    def __init__(self):
        self.client = None
        self.db = None
        load_dotenv()
        
    async def connect(self):
        """Connect to MongoDB using environment variables."""
        username = os.getenv('MONGO_INITDB_ROOT_USERNAME')
        password = os.getenv('MONGO_INITDB_ROOT_PASSWORD')
        port = os.getenv('MONGO_PORT', '27017')
        
        connection_string = f"mongodb://{username}:{password}@localhost:{port}/?authSource=admin"
        
        try:
            self.client = AsyncIOMotorClient(
                connection_string,
                serverSelectionTimeoutMS=5000
            )
            self.db = self.client.memedex_db
            await self.db.command('ping')
            print("Successfully connected to MongoDB")
        except Exception as e:
            print(f"Failed to connect to MongoDB: {str(e)}")
            raise

    async def disconnect(self):
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            print("Disconnected from MongoDB")

    async def add_trending_pools(self, df: pd.DataFrame):
        """
        Add trending pools data to MongoDB.
        Expected DataFrame columns: TBD - adjust according to your data structure
        """
        if df.empty:
            return
        
        collection = self.db.trending_pools
        
        # Convert DataFrame to list of dictionaries and add timestamp
        records = df.to_dict('records')
        timestamp = datetime.utcnow()
        for record in records:
            record['timestamp'] = timestamp
        
        try:
            result = await collection.insert_many(records)
            inserted_count = len(result.inserted_ids)
            print(f"Successfully inserted {inserted_count} trending pools")
            
            # Verify the insertion
            count = await collection.count_documents({'timestamp': timestamp})
            print(f"Verified {count} documents with timestamp {timestamp}")
            return inserted_count
            
        except Exception as e:
            print(f"Error inserting trending pools: {str(e)}")
            raise

    async def add_new_pools(self, df: pd.DataFrame):
        """
        Add new pools data to MongoDB.
        Expected DataFrame columns: TBD - adjust according to your data structure
        """
        if df.empty:
            return
        
        collection = self.db.new_pools
        
        records = df.to_dict('records')
        timestamp = datetime.utcnow()
        for record in records:
            record['timestamp'] = timestamp
        
        try:
            result = await collection.insert_many(records)
            inserted_count = len(result.inserted_ids)
            print(f"Successfully inserted {inserted_count} new pools")
            
            # Verify the insertion
            count = await collection.count_documents({'timestamp': timestamp})
            print(f"Verified {count} documents with timestamp {timestamp}")
            return inserted_count
            
        except Exception as e:
            print(f"Error inserting new pools: {str(e)}")
            raise

    async def add_stakers(self, df: pd.DataFrame):
        """
        Add stakers data to MongoDB.
        Expected DataFrame columns: TBD - adjust according to your data structure
        """
        if df.empty:
            return
        
        collection = self.db.stakers
        
        records = df.to_dict('records')
        timestamp = datetime.utcnow()
        for record in records:
            record['timestamp'] = timestamp
        
        try:
            await collection.insert_many(records)
            print(f"Successfully inserted {len(records)} stakers")
        except Exception as e:
            print(f"Error inserting stakers: {str(e)}")
            raise

    async def add_top_holders(self, df: pd.DataFrame):
        """
        Add top holders data to MongoDB.
        Expected DataFrame columns: TBD - adjust according to your data structure
        """
        if df.empty:
            return
        
        collection = self.db.top_holders
        
        records = df.to_dict('records')
        timestamp = datetime.utcnow()
        for record in records:
            record['timestamp'] = timestamp
        
        try:
            await collection.insert_many(records)
            print(f"Successfully inserted {len(records)} top holders")
        except Exception as e:
            print(f"Error inserting top holders: {str(e)}")
            raise 

    async def get_trending_pools(self, hours_ago: int = None) -> pd.DataFrame:
        """
        Get trending pools data from MongoDB.
        Args:
            hours_ago: If provided, only return data from the last N hours
        Returns:
            DataFrame with trending pools data
        """
        collection = self.db.trending_pools
        query = {}
        
        if hours_ago is not None:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_ago)
            query = {'timestamp': {'$gte': cutoff_time}}
            
        try:
            cursor = collection.find(query)
            documents = await cursor.to_list(length=None)
            if not documents:
                print("No trending pools data found")
                return pd.DataFrame()
                
            df = pd.DataFrame(documents)
            # Remove MongoDB's _id column
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
                
            print(f"Retrieved {len(df)} trending pools records")
            return df
            
        except Exception as e:
            print(f"Error retrieving trending pools: {str(e)}")
            raise

    async def get_new_pools(self, hours_ago: int = None) -> pd.DataFrame:
        """
        Get new pools data from MongoDB.
        Args:
            hours_ago: If provided, only return data from the last N hours
        Returns:
            DataFrame with new pools data
        """
        collection = self.db.new_pools
        query = {}
        
        if hours_ago is not None:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_ago)
            query = {'timestamp': {'$gte': cutoff_time}}
            
        try:
            cursor = collection.find(query)
            documents = await cursor.to_list(length=None)
            if not documents:
                print("No new pools data found")
                return pd.DataFrame()
                
            df = pd.DataFrame(documents)
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
                
            print(f"Retrieved {len(df)} new pools records")
            return df
            
        except Exception as e:
            print(f"Error retrieving new pools: {str(e)}")
            raise

    async def get_stakers(self, hours_ago: int = None) -> pd.DataFrame:
        """
        Get stakers data from MongoDB.
        Args:
            hours_ago: If provided, only return data from the last N hours
        Returns:
            DataFrame with stakers data
        """
        collection = self.db.stakers
        query = {}
        
        if hours_ago is not None:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_ago)
            query = {'timestamp': {'$gte': cutoff_time}}
            
        try:
            cursor = collection.find(query)
            documents = await cursor.to_list(length=None)
            if not documents:
                print("No stakers data found")
                return pd.DataFrame()
                
            df = pd.DataFrame(documents)
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
                
            print(f"Retrieved {len(df)} stakers records")
            return df
            
        except Exception as e:
            print(f"Error retrieving stakers: {str(e)}")
            raise

    async def get_top_holders(self, hours_ago: int = None) -> pd.DataFrame:
        """
        Get top holders data from MongoDB.
        Args:
            hours_ago: If provided, only return data from the last N hours
        Returns:
            DataFrame with top holders data
        """
        collection = self.db.top_holders
        query = {}
        
        if hours_ago is not None:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_ago)
            query = {'timestamp': {'$gte': cutoff_time}}
            
        try:
            cursor = collection.find(query)
            documents = await cursor.to_list(length=None)
            if not documents:
                print("No top holders data found")
                return pd.DataFrame()
                
            df = pd.DataFrame(documents)
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
                
            print(f"Retrieved {len(df)} top holders records")
            return df
            
        except Exception as e:
            print(f"Error retrieving top holders: {str(e)}")
            raise

    async def get_latest_data(self) -> dict:
        """
        Get the latest data from all collections.
        Returns:
            Dictionary containing DataFrames for each collection
        """
        try:
            latest_data = {
                'trending_pools': await self.get_trending_pools(hours_ago=24),
                'new_pools': await self.get_new_pools(hours_ago=24),
                'stakers': await self.get_stakers(hours_ago=24),
                'top_holders': await self.get_top_holders(hours_ago=24)
            }
            return latest_data
            
        except Exception as e:
            print(f"Error retrieving latest data: {str(e)}")
            raise 