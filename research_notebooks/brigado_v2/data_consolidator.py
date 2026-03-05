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


class DataConsolidator:
    """
    Consolidates data from multiple Hummingbot databases into unified data sources.

    Responsibilities:
    - Auto-discover databases in live_databases/
    - Extract and combine trades, orders, executors, controllers
    - Save consolidated data in efficient parquet format
    - Maintain metadata about source databases
    """

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize data consolidator.

        Args:
            base_path: Base path for brigado_v2 directory
        """
        if base_path is None:
            current_file = Path(__file__)
            base_path = current_file.parent

        self.base_path = Path(base_path)
        self.live_databases_dir = self.base_path / "data" / "live_databases"
        self.data_sources_dir = self.base_path / "data" / "data_sources"

        # Ensure data_sources directory exists
        self.data_sources_dir.mkdir(parents=True, exist_ok=True)

    def discover_databases(self) -> List[Dict[str, any]]:
        """
        Discover all SQLite databases in live_databases directory.

        Returns:
            List of dicts with database info (bot_name, db_path, config_path, size)
        """
        databases = []

        if not self.live_databases_dir.exists():
            logger.warning(f"Live databases directory not found: {self.live_databases_dir}")
            return databases

        # Iterate through bot instance directories
        for bot_dir in self.live_databases_dir.iterdir():
            if not bot_dir.is_dir():
                continue  # Skip files (like fetch logs)

            # Look for .sqlite file in data/ subdirectory
            data_dir = bot_dir / "data"
            if not data_dir.exists():
                logger.warning(f"No data directory found for bot: {bot_dir.name}")
                continue

            sqlite_files = list(data_dir.glob("*.sqlite"))
            if not sqlite_files:
                logger.warning(f"No SQLite database found in: {data_dir}")
                continue

            # Get the first .sqlite file (should only be one)
            db_path = sqlite_files[0]

            # Look for config files
            config_dir = bot_dir / "conf" / "controllers"
            config_files = list(config_dir.glob("*.yml")) if config_dir.exists() else []

            databases.append({
                'bot_name': bot_dir.name,
                'db_path': db_path,
                'config_dir': config_dir if config_dir.exists() else None,
                'config_files': config_files,
                'size_mb': db_path.stat().st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(db_path.stat().st_mtime)
            })

        logger.info(f"Discovered {len(databases)} database(s)")
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

            # Load Executors
            try:
                executors = pd.read_sql_query("SELECT * FROM Executors", conn)
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
            logger.error("No databases found to consolidate!")
            return {}

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

        # Combine all DataFrames
        logger.info("\nCombining data from all sources...")

        output_paths = {}

        if all_trades:
            trades_df = pd.concat(all_trades, ignore_index=True)
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
            executors_path = self.data_sources_dir / "consolidated_executors.parquet"
            executors_df.to_parquet(executors_path, index=False)
            output_paths['executors'] = executors_path
            logger.info(f"  ✓ Saved {len(executors_df):,} executors to {executors_path.name}")

        if all_controllers:
            controllers_df = pd.concat(all_controllers, ignore_index=True)
            controllers_path = self.data_sources_dir / "consolidated_controllers.parquet"
            controllers_df.to_parquet(controllers_path, index=False)
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
            logger.info(f"Loaded {len(data['executors']):,} executors from {executors_path.name}")

        if controllers_path and controllers_path.exists():
            data['controllers'] = pd.read_parquet(controllers_path)
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
