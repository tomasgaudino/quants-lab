"""
Data Consolidator Module

Consolidates data from multiple live databases into a single unified data source.

This module scans all databases in live_databases/ and creates a consolidated
parquet file with all trades, orders, executors, and controller configs combined.

TODO - Future Improvements:
- Add incremental updates (only process new data)
- Add data quality validation
- Support for filtering by date range
- Add parallel processing for multiple databases
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# CloseType mapping (from hummingbot.strategy_v2.models.executors)
CLOSE_TYPE_MAP = {
    'TIME_LIMIT': 1,
    'STOP_LOSS': 2,
    'TAKE_PROFIT': 3,
    'EXPIRED': 4,
    'EARLY_STOP': 5,
    'TRAILING_STOP': 6,
    'INSUFFICIENT_BALANCE': 7,
    'FAILED': 8,
    'COMPLETED': 9,
    'POSITION_HOLD': 10,
    'SYSTEM_CLEANUP': 11,  # Custom value for system cleanup
    # Also handle int values
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11,
    # Handle string representations of numbers
    '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11,
}

# RunnableStatus mapping (from hummingbot.strategy_v2.models.base)
RUNNABLE_STATUS_MAP = {
    'NOT_STARTED': 1,
    'RUNNING': 2,
    'SHUTTING_DOWN': 3,
    'TERMINATED': 4,
    # Also handle int values
    1: 1, 2: 2, 3: 3, 4: 4,
    # Handle string representations of numbers
    '1': 1, '2': 2, '3': 3, '4': 4,
}


def standardize_close_type(value):
    """Convert close_type to integer, handling both enum names and int values."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        # Try to get from mapping
        if value in CLOSE_TYPE_MAP:
            return CLOSE_TYPE_MAP[value]
        # Try to convert string number
        try:
            return int(value)
        except ValueError:
            logger.warning(f"Unknown close_type value: {value}, setting to None")
            return None
    return None


def standardize_status(value):
    """Convert status to integer, handling both enum names and int values."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        # Try to get from mapping
        if value in RUNNABLE_STATUS_MAP:
            return RUNNABLE_STATUS_MAP[value]
        # Try to convert string number
        try:
            return int(value)
        except ValueError:
            logger.warning(f"Unknown status value: {value}, setting to None")
            return None
    return None


class DataConsolidator:
    """
    Consolidates data from multiple Hummingbot databases into unified data sources.

    Responsibilities:
    - Auto-discover databases in live_databases/
    - Extract and combine trades, orders, executors, controllers
    - Save consolidated data in efficient parquet format
    - Maintain metadata about source databases
    """

    def __init__(self, base_path: Optional[str] = None, server_name: str = "brigado"):
        """
        Initialize data consolidator.

        Args:
            base_path: Base path for brigado_v2 directory
            server_name: Name of the server (e.g., 'brigado', 'old_brigado')
        """
        if base_path is None:
            current_file = Path(__file__)
            base_path = current_file.parent.parent  # Go up to brigado_v2

        self.base_path = Path(base_path)
        self.server_name = server_name

        # Use FileManager for consistent path management
        from research_notebooks.brigado_v2.modules.file_manager import FileManager
        self.file_manager = FileManager(server_name=server_name)

        self.live_databases_dir = self.file_manager.live_databases_dir
        self.data_sources_dir = self.file_manager.data_sources_dir

        # Hummingbot API directory (optional, for direct API exports)
        self.hummingbot_api_dir = self.file_manager.server_dir / "hummingbot_api"

    def discover_databases(self) -> List[Dict[str, any]]:
        """
        Discover all SQLite databases in live_databases directory.

        Returns:
            List of dicts with database info (bot_name, db_path, config_path, size)
        """
        # Use FileManager's discover method for consistency
        databases = self.file_manager.discover_live_databases()
        logger.info(f"Discovered {len(databases)} database(s) for server '{self.server_name}'")
        return databases

    def load_from_database(self, db_path: Path, bot_name: str) -> Dict[str, pd.DataFrame]:
        """
        Load all tables from a single database.

        Args:
            db_path: Path to SQLite database
            bot_name: Bot instance name

        Returns:
            Dict with DataFrames for trades, orders, executors, controllers
        """
        logger.info(f"Loading data from {bot_name}...")

        conn = sqlite3.connect(db_path)

        try:
            # Load TradeFill
            trades = pd.read_sql_query("SELECT * FROM TradeFill", conn)
            if len(trades) > 0:
                # Convert scaled integers to floats
                float_cols = ["amount", "price", "trade_fee_in_quote"]
                for col in float_cols:
                    if col in trades.columns:
                        trades[col] = trades[col] / 1e6

                # Add timestamp conversion
                if 'timestamp' in trades.columns:
                    trades["timestamp"] = pd.to_datetime(trades["timestamp"], unit="ms")

                # Add source metadata
                trades['source_bot'] = bot_name
                trades['source_db'] = str(db_path)

            # Load Order
            try:
                orders = pd.read_sql_query("SELECT * FROM 'Order'", conn)
                if len(orders) > 0:
                    # Convert scaled integers
                    if 'amount' in orders.columns:
                        orders['amount'] = orders['amount'] / 1e6
                    if 'price' in orders.columns:
                        orders['price'] = orders['price'] / 1e6
                    if 'creation_timestamp' in orders.columns:
                        orders['creation_timestamp'] = pd.to_datetime(orders['creation_timestamp'], unit="ms")
                    if 'last_update_timestamp' in orders.columns:
                        orders['last_update_timestamp'] = pd.to_datetime(orders['last_update_timestamp'], unit="ms")

                    orders['source_bot'] = bot_name
                    orders['source_db'] = str(db_path)
            except Exception as e:
                logger.warning(f"Could not load orders from {bot_name}: {e}")
                orders = pd.DataFrame()

            # Load Executors (exclude close_type = 7)
            try:
                executors = pd.read_sql_query("SELECT * FROM Executors WHERE close_type != 7", conn)
                if len(executors) > 0:
                    # Parse JSON fields
                    if 'config' in executors.columns:
                        executors['config_parsed'] = executors['config'].apply(
                            lambda x: json.loads(x) if x else {}
                        )
                    if 'custom_info' in executors.columns:
                        executors['custom_info_parsed'] = executors['custom_info'].apply(
                            lambda x: json.loads(x) if x else {}
                        )

                    # Convert timestamps
                    if 'timestamp' in executors.columns:
                        executors['timestamp'] = pd.to_datetime(executors['timestamp'], unit="ms")

                    executors['source_bot'] = bot_name
                    executors['source_db'] = str(db_path)
            except Exception as e:
                logger.warning(f"Could not load executors from {bot_name}: {e}")
                executors = pd.DataFrame()

            # Load Controllers
            try:
                controllers = pd.read_sql_query("SELECT * FROM Controllers", conn)
                if len(controllers) > 0:
                    # Parse JSON config
                    if 'config' in controllers.columns:
                        controllers['config_parsed'] = controllers['config'].apply(
                            lambda x: json.loads(x) if x else {}
                        )

                    # Convert timestamps
                    if 'timestamp' in controllers.columns:
                        controllers['timestamp'] = pd.to_datetime(controllers['timestamp'], unit="ms")

                    controllers['source_bot'] = bot_name
                    controllers['source_db'] = str(db_path)
            except Exception as e:
                logger.warning(f"Could not load controllers from {bot_name}: {e}")
                controllers = pd.DataFrame()

            logger.info(f"  Loaded {len(trades)} trades, {len(orders)} orders, "
                       f"{len(executors)} executors, {len(controllers)} controllers")

            return {
                'trades': trades,
                'orders': orders,
                'executors': executors,
                'controllers': controllers
            }

        finally:
            conn.close()

    def load_from_hummingbot_api(self) -> Dict[str, pd.DataFrame]:
        """
        Load data from hummingbot-api PostgreSQL exports.

        Returns:
            Dict with DataFrames for trades, executors and controllers
        """
        logger.info("Loading data from hummingbot-api...")

        data = {
            'trades': pd.DataFrame(),
            'orders': pd.DataFrame(),
            'executors': pd.DataFrame(),
            'controllers': pd.DataFrame()
        }

        # Check if hummingbot_api directory exists
        if not self.hummingbot_api_dir.exists():
            logger.warning(f"Hummingbot API directory not found: {self.hummingbot_api_dir}")
            return data

        # Load trades
        trades_path = self.hummingbot_api_dir / "trades.parquet"
        if trades_path.exists():
            data['trades'] = pd.read_parquet(trades_path)
            logger.info(f"  Loaded {len(data['trades'])} trades from hummingbot-api")
        else:
            logger.warning(f"  No trades file found: {trades_path}")

        # Load executors
        executors_path = self.hummingbot_api_dir / "executors.parquet"
        if executors_path.exists():
            data['executors'] = pd.read_parquet(executors_path)

            # Re-parse JSON fields
            if 'config' in data['executors'].columns:
                data['executors']['config_parsed'] = data['executors']['config'].apply(
                    lambda x: json.loads(x) if x and isinstance(x, str) else (x if isinstance(x, dict) else {})
                )
            if 'custom_info' in data['executors'].columns:
                data['executors']['custom_info_parsed'] = data['executors']['custom_info'].apply(
                    lambda x: json.loads(x) if x and isinstance(x, str) else (x if isinstance(x, dict) else {})
                )

            logger.info(f"  Loaded {len(data['executors'])} executors from hummingbot-api")
        else:
            logger.warning(f"  No executors file found: {executors_path}")

        # Load controllers
        controllers_path = self.hummingbot_api_dir / "controllers.parquet"
        if controllers_path.exists():
            data['controllers'] = pd.read_parquet(controllers_path)

            # Re-parse JSON config
            if 'config' in data['controllers'].columns:
                data['controllers']['config_parsed'] = data['controllers']['config'].apply(
                    lambda x: json.loads(x) if x and isinstance(x, str) else (x if isinstance(x, dict) else {})
                )

            logger.info(f"  Loaded {len(data['controllers'])} controllers from hummingbot-api")
        else:
            logger.warning(f"  No controllers file found: {controllers_path}")

        return data

    def consolidate_all(self, force_refresh: bool = False) -> Dict[str, Path]:
        """
        Consolidate data from all discovered databases.

        Args:
            force_refresh: If True, rebuild even if consolidated files exist

        Returns:
            Dict with paths to consolidated parquet files
        """
        logger.info("="*80)
        logger.info("STARTING DATA CONSOLIDATION")
        logger.info("="*80)

        # Discover all databases
        databases = self.discover_databases()

        if not databases:
            logger.warning("No SQLite databases found!")

        # Initialize combined DataFrames
        all_trades = []
        all_orders = []
        all_executors = []
        all_controllers = []

        # Load data from each database
        for db_info in databases:
            data = self.load_from_database(db_info['db_path'], db_info['bot_name'])

            if not data['trades'].empty:
                all_trades.append(data['trades'])
            if not data['orders'].empty:
                all_orders.append(data['orders'])
            if not data['executors'].empty:
                all_executors.append(data['executors'])
            if not data['controllers'].empty:
                all_controllers.append(data['controllers'])

        # Load data from hummingbot-api (PostgreSQL)
        logger.info("\nLoading data from hummingbot-api...")
        hbapi_data = self.load_from_hummingbot_api()

        if not hbapi_data['trades'].empty:
            all_trades.append(hbapi_data['trades'])
            logger.info(f"  Added {len(hbapi_data['trades'])} trades from hummingbot-api")

        if not hbapi_data['executors'].empty:
            all_executors.append(hbapi_data['executors'])
            logger.info(f"  Added {len(hbapi_data['executors'])} executors from hummingbot-api")

        if not hbapi_data['controllers'].empty:
            all_controllers.append(hbapi_data['controllers'])
            logger.info(f"  Added {len(hbapi_data['controllers'])} controllers from hummingbot-api")

        # Combine all DataFrames
        logger.info("\nCombining data from all sources...")

        output_paths = {}

        if all_trades:
            trades_df = pd.concat(all_trades, ignore_index=True)

            # Standardize data types before saving
            # Convert order_id to string (mixed types from different sources)
            if 'order_id' in trades_df.columns:
                trades_df['order_id'] = trades_df['order_id'].astype(str)

            trades_path = self.data_sources_dir / "consolidated_trades.parquet"
            trades_df.to_parquet(trades_path, index=False)
            output_paths['trades'] = trades_path
            logger.info(f"  ✓ Saved {len(trades_df):,} trades to {trades_path.name}")

        if all_orders:
            orders_df = pd.concat(all_orders, ignore_index=True)
            orders_path = self.data_sources_dir / "consolidated_orders.parquet"
            orders_df.to_parquet(orders_path, index=False)
            output_paths['orders'] = orders_path
            logger.info(f"  ✓ Saved {len(orders_df):,} orders to {orders_path.name}")

        if all_executors:
            executors_df = pd.concat(all_executors, ignore_index=True)

            # Standardize data types before saving
            # Convert id column to string (mixed types from different sources)
            if 'id' in executors_df.columns:
                executors_df['id'] = executors_df['id'].astype(str)

            # Standardize close_type to integer (handles both enum names and int values)
            if 'close_type' in executors_df.columns:
                executors_df['close_type'] = executors_df['close_type'].apply(standardize_close_type)

            # Standardize status to integer (handles both enum names and int values)
            if 'status' in executors_df.columns:
                executors_df['status'] = executors_df['status'].apply(standardize_status)

            executors_path = self.data_sources_dir / "consolidated_executors.parquet"
            # Drop parsed columns before saving (parquet can't serialize complex objects)
            executors_to_save = executors_df.drop(columns=['config_parsed', 'custom_info_parsed'], errors='ignore')
            executors_to_save.to_parquet(executors_path, index=False)
            output_paths['executors'] = executors_path
            logger.info(f"  ✓ Saved {len(executors_df):,} executors to {executors_path.name}")

        if all_controllers:
            controllers_df = pd.concat(all_controllers, ignore_index=True)

            # Standardize data types before saving
            # Convert id column to string (mixed types from different sources)
            if 'id' in controllers_df.columns:
                controllers_df['id'] = controllers_df['id'].astype(str)

            controllers_path = self.data_sources_dir / "consolidated_controllers.parquet"
            # Drop parsed column before saving (parquet can't serialize complex objects)
            controllers_to_save = controllers_df.drop(columns=['config_parsed'], errors='ignore')
            controllers_to_save.to_parquet(controllers_path, index=False)
            output_paths['controllers'] = controllers_path
            logger.info(f"  ✓ Saved {len(controllers_df):,} controllers to {controllers_path.name}")

        # Save metadata
        metadata = {
            'consolidation_datetime': datetime.now().isoformat(),
            'num_databases': len(databases),
            'databases': [
                {
                    'bot_name': db['bot_name'],
                    'db_path': str(db['db_path']),
                    'size_mb': db['size_mb'],
                    'modified': db['modified'].isoformat()
                }
                for db in databases
            ],
            'output_files': {k: str(v) for k, v in output_paths.items()},
            'record_counts': {
                'trades': len(trades_df) if all_trades else 0,
                'orders': len(orders_df) if all_orders else 0,
                'executors': len(executors_df) if all_executors else 0,
                'controllers': len(controllers_df) if all_controllers else 0
            }
        }

        metadata_path = self.data_sources_dir / "consolidation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        output_paths['metadata'] = metadata_path
        logger.info(f"  ✓ Saved metadata to {metadata_path.name}")

        logger.info("\n" + "="*80)
        logger.info("CONSOLIDATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Total databases processed: {len(databases)}")
        logger.info(f"Total trades: {metadata['record_counts']['trades']:,}")
        logger.info(f"Total orders: {metadata['record_counts']['orders']:,}")
        logger.info(f"Total executors: {metadata['record_counts']['executors']:,}")
        logger.info(f"Total controllers: {metadata['record_counts']['controllers']:,}")
        logger.info(f"Output directory: {self.data_sources_dir}")
        logger.info("="*80)

        return output_paths

    def load_consolidated_data(self, use_latest: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load consolidated data from parquet files.

        Args:
            use_latest: Deprecated parameter, kept for compatibility.

        Returns:
            Dict with DataFrames for trades, orders, executors, controllers
        """
        data = {}

        # Simple file paths without versioning
        trades_path = self.data_sources_dir / "consolidated_trades.parquet"
        orders_path = self.data_sources_dir / "consolidated_orders.parquet"
        executors_path = self.data_sources_dir / "consolidated_executors.parquet"
        controllers_path = self.data_sources_dir / "consolidated_controllers.parquet"

        # Load each file if it exists
        if trades_path and trades_path.exists():
            data['trades'] = pd.read_parquet(trades_path)
            logger.info(f"Loaded {len(data['trades']):,} trades from {trades_path.name}")

        if orders_path and orders_path.exists():
            data['orders'] = pd.read_parquet(orders_path)
            logger.info(f"Loaded {len(data['orders']):,} orders from {orders_path.name}")

        if executors_path and executors_path.exists():
            data['executors'] = pd.read_parquet(executors_path)

            # Re-parse JSON fields
            if 'config' in data['executors'].columns:
                data['executors']['config_parsed'] = data['executors']['config'].apply(
                    lambda x: json.loads(x) if x and isinstance(x, str) else (x if isinstance(x, dict) else {})
                )
            if 'custom_info' in data['executors'].columns:
                data['executors']['custom_info_parsed'] = data['executors']['custom_info'].apply(
                    lambda x: json.loads(x) if x and isinstance(x, str) else (x if isinstance(x, dict) else {})
                )

            logger.info(f"Loaded {len(data['executors']):,} executors from {executors_path.name}")

        if controllers_path and controllers_path.exists():
            data['controllers'] = pd.read_parquet(controllers_path)

            # Re-parse JSON config
            if 'config' in data['controllers'].columns:
                data['controllers']['config_parsed'] = data['controllers']['config'].apply(
                    lambda x: json.loads(x) if x and isinstance(x, str) else (x if isinstance(x, dict) else {})
                )

            logger.info(f"Loaded {len(data['controllers']):,} controllers from {controllers_path.name}")

        return data

    def get_consolidation_info(self) -> Optional[Dict]:
        """
        Get info about the most recent consolidation.

        Returns:
            Dict with consolidation metadata or None if no consolidation exists
        """
        metadata_path = self.data_sources_dir / "consolidation_metadata.json"

        if not metadata_path.exists():
            return None

        with open(metadata_path, 'r') as f:
            return json.load(f)
