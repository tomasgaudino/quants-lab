"""
Data Loader Module

Handles database connections and raw data loading for Hummingbot performance analysis.

TODO - Future Improvements:
- Add connection pooling for multiple databases
- Implement caching layer for frequently accessed data
- Add data validation on load
- Support for remote database connections
"""

from typing import Optional
import pandas as pd
from core.data_sources.hummingbot_database import HummingbotDatabase


class DataLoader:
    """
    Loads raw data from Hummingbot SQLite database.

    Responsibilities:
    - Database connection management
    - Loading raw tables (TradeFill, Order, Executor, Controller)
    - Basic data type conversions
    """

    def __init__(
        self,
        db_name: str,
        root_path: str,
        server_name: str = "brigado_server"
    ):
        """
        Initialize data loader with database connection.

        Args:
            db_name: SQLite database filename
            root_path: Root path for the project
            server_name: Server name for database location
        """
        self.db_name = db_name
        self.root_path = root_path
        self.server_name = server_name
        self.db = HummingbotDatabase(
            db_name=db_name,
            root_path=root_path,
            server_name=server_name
        )
        self._quote_asset = None

    def load_trade_fills(
        self,
        config_file_path: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load raw trade fills from database without cumulative calculations.

        Args:
            config_file_path: Optional filter by config file
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            DataFrame with basic trade fills (no cumulative metrics)

        TODO:
        - Add data quality checks (missing timestamps, invalid prices)
        - Add filters for specific trading pairs or markets
        """
        float_cols = ["amount", "price", "trade_fee_in_quote"]
        query = "SELECT * FROM TradeFill"
        trade_fills = pd.read_sql_query(query, self.db.connection)

        # Detect quote asset from data
        if self._quote_asset is None and 'quote_asset' in trade_fills.columns:
            self._quote_asset = trade_fills['quote_asset'].mode()[0] if len(trade_fills) > 0 else 'USDT'

        # Basic conversions only
        trade_fills[float_cols] = trade_fills[float_cols] / 1e6
        trade_fills["net_amount"] = trade_fills['amount'] * trade_fills['trade_type'].apply(
            lambda x: 1 if x == 'BUY' else -1
        )
        trade_fills["net_amount_quote"] = trade_fills['net_amount'] * trade_fills['price']
        trade_fills["timestamp"] = pd.to_datetime(trade_fills["timestamp"], unit="ms")
        trade_fills["quote_volume"] = trade_fills["price"] * trade_fills["amount"]

        return trade_fills

    def get_quote_asset(self) -> str:
        """
        Get the detected quote asset from the database.

        Returns:
            Quote asset symbol (e.g., 'USDC', 'USDT', 'BRL')
        """
        if self._quote_asset is None:
            # Load a sample to detect quote asset
            query = "SELECT DISTINCT quote_asset FROM TradeFill LIMIT 1"
            result = pd.read_sql_query(query, self.db.connection)
            self._quote_asset = result['quote_asset'].iloc[0] if len(result) > 0 else 'USDT'
        return self._quote_asset

    def load_orders(self) -> pd.DataFrame:
        """
        Load orders from database.

        Returns:
            DataFrame with orders

        TODO:
        - Add order status filtering
        - Add order type analytics
        """
        return self.db.get_orders()

    def load_executors(self) -> pd.DataFrame:
        """
        Load executors with JSON fields parsed.

        Returns:
            DataFrame with executors including config and custom_info

        TODO:
        - Validate JSON structure
        - Add executor performance metrics
        """
        return self.db.get_executors_data()

    def load_controllers(self) -> pd.DataFrame:
        """
        Load controllers with JSON config parsed.

        Returns:
            DataFrame with controllers including config

        TODO:
        - Add controller strategy type detection
        - Extract and validate all strategy parameters
        """
        return self.db.get_controller_data()

    def load_all(self) -> dict:
        """
        Load all tables in a single call.

        Returns:
            Dictionary with keys: trade_fills, orders, executors, controllers

        TODO:
        - Add parallel loading for performance
        - Add progress callbacks for large datasets
        """
        return {
            'trade_fills': self.load_trade_fills(),
            'orders': self.load_orders(),
            'executors': self.load_executors(),
            'controllers': self.load_controllers()
        }
