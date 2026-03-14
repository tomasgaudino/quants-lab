#!/usr/bin/env python3
"""
PostgreSQL Data Consolidator

Consolidates PostgreSQL data:
1. Creates token_states_consolidated.parquet (joined with account_states)
2. Maps postgres trades to consolidated_trades structure
3. Merges postgres trades with existing SQLite trades
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from research_notebooks.brigado_v2.modules.file_manager import FileManager


def print_header(title: str):
    """Print formatted header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_step(message: str):
    """Print step message."""
    print(f"⏳ {message}")


def print_success(message: str):
    """Print success message."""
    print(f"✅ {message}")


def print_metric(label: str, value: str, indent: int = 2):
    """Print formatted metric."""
    spaces = " " * indent
    print(f"{spaces}{label:<40} {value:>35}")


def consolidate_token_states(file_manager: FileManager) -> pd.DataFrame:
    """
    Consolidate token_states with account_states.

    Returns:
        DataFrame with joined token and account state data
    """
    print_step("Consolidating token_states...")

    postgres_dir = file_manager.server_dir / 'postgres'

    # Load postgres data
    token_states = pd.read_parquet(postgres_dir / 'token_states.parquet')
    account_states = pd.read_parquet(postgres_dir / 'account_states.parquet')

    print_metric("Token states loaded", f"{len(token_states):,} rows")
    print_metric("Account states loaded", f"{len(account_states):,} rows")

    # Join token_states with account_states
    consolidated = token_states.merge(
        account_states,
        left_on='account_state_id',
        right_on='id',
        suffixes=('', '_account')
    )

    # Drop redundant columns
    consolidated = consolidated.drop(columns=['id_account', 'account_state_id'])

    # Rename for clarity
    consolidated = consolidated.rename(columns={
        'id': 'token_state_id'
    })

    # Reorder columns
    column_order = [
        'token_state_id',
        'timestamp',
        'account_name',
        'connector_name',
        'token',
        'units',
        'available_units',
        'price',
        'value'
    ]
    consolidated = consolidated[column_order]

    # Sort by timestamp
    consolidated = consolidated.sort_values('timestamp')

    print_metric("Consolidated rows", f"{len(consolidated):,}")
    print_metric("Date range", f"{consolidated['timestamp'].min()} to {consolidated['timestamp'].max()}")
    print_metric("Unique tokens", f"{consolidated['token'].nunique()}")
    print_metric("Unique snapshots", f"{consolidated['timestamp'].nunique()}")

    # Save consolidated file
    output_path = file_manager.server_dir / 'token_states_consolidated.parquet'
    consolidated.to_parquet(output_path, index=False)
    print_success(f"Saved to: {output_path}")

    return consolidated


def map_postgres_trades_to_structure(file_manager: FileManager, existing_trades: pd.DataFrame) -> pd.DataFrame:
    """
    Map postgres trades to consolidated_trades structure.

    Args:
        file_manager: FileManager instance
        existing_trades: Existing consolidated_trades DataFrame for reference

    Returns:
        DataFrame with postgres trades mapped to consolidated structure
    """
    print_step("Mapping postgres trades to structure...")

    postgres_dir = file_manager.server_dir / 'postgres'

    # Load postgres trades
    pg_trades = pd.read_parquet(postgres_dir / 'trades.parquet')
    print_metric("Postgres trades loaded", f"{len(pg_trades):,} rows")

    # Parse trading_pair into base_asset and quote_asset
    # Format: "BTC-BRL" -> base="BTC", quote="BRL"
    pg_trades[['base_asset', 'quote_asset']] = pg_trades['trading_pair'].str.split('-', expand=True)

    # Create mapped DataFrame matching consolidated_trades structure
    mapped_trades = pd.DataFrame({
        # Required fields (from postgres) - remove timezone to match SQLite data
        'timestamp': pd.to_datetime(pg_trades['timestamp']).dt.tz_localize(None),
        'symbol': pg_trades['trading_pair'],
        'base_asset': pg_trades['base_asset'],
        'quote_asset': pg_trades['quote_asset'],
        'trade_type': pg_trades['trade_type'],
        'amount': pg_trades['amount'].astype(float),
        'price': pg_trades['price'].astype(float),
        'trade_fee_in_quote': pg_trades['fee_paid'].astype(float),
        'exchange_trade_id': pg_trades['trade_id'],

        # Fields we can derive or set to defaults
        'source_bot': 'hummingbot-api-postgres',
        'source_db': 'hummingbot_api',
        'market': 'binance',  # From postgres connector_name
        'strategy': 'unknown',  # Not available in postgres
        'order_id': pg_trades['order_id'].astype(str),
        'leverage': 1,
        'position': 'NIL',

        # Fields that need to be blank/null (not available in postgres)
        'config_file_path': '',
        'order_type': '',
        'trade_fee': ''
    })

    # Reorder to match existing structure
    column_order = existing_trades.columns.tolist()
    mapped_trades = mapped_trades[column_order]

    print_metric("Mapped trades", f"{len(mapped_trades):,} rows")
    print_metric("Date range", f"{mapped_trades['timestamp'].min()} to {mapped_trades['timestamp'].max()}")
    print_metric("Symbols", f"{', '.join(mapped_trades['symbol'].unique())}")

    return mapped_trades


def merge_trades(existing_trades: pd.DataFrame, postgres_trades: pd.DataFrame) -> pd.DataFrame:
    """
    Merge postgres trades with existing SQLite trades.

    Args:
        existing_trades: Existing consolidated_trades DataFrame
        postgres_trades: Mapped postgres trades DataFrame

    Returns:
        Merged DataFrame with all trades
    """
    print_step("Merging postgres trades with existing trades...")

    print_metric("Existing trades", f"{len(existing_trades):,} rows")
    print_metric("Postgres trades", f"{len(postgres_trades):,} rows")

    # Check for duplicates based on exchange_trade_id
    existing_trade_ids = set(existing_trades['exchange_trade_id'].dropna())
    postgres_trade_ids = set(postgres_trades['exchange_trade_id'].dropna())

    duplicate_ids = existing_trade_ids & postgres_trade_ids
    if duplicate_ids:
        print_metric("Duplicate trade IDs found", f"{len(duplicate_ids):,}")
        # Remove duplicates from postgres trades
        postgres_trades = postgres_trades[~postgres_trades['exchange_trade_id'].isin(duplicate_ids)]
        print_metric("After deduplication", f"{len(postgres_trades):,} postgres trades")

    # Concatenate
    merged_trades = pd.concat([existing_trades, postgres_trades], ignore_index=True)

    # Sort by timestamp
    merged_trades = merged_trades.sort_values('timestamp').reset_index(drop=True)

    print_metric("Total merged trades", f"{len(merged_trades):,} rows")
    print_metric("Date range", f"{merged_trades['timestamp'].min()} to {merged_trades['timestamp'].max()}")

    return merged_trades


def main():
    """Main execution."""
    print_header("POSTGRESQL DATA CONSOLIDATOR")

    start_time = datetime.now()

    # Setup paths
    script_dir = Path(__file__).parent
    file_manager = FileManager(server_name='brigado')
    postgres_dir = file_manager.server_dir / 'postgres'

    # Check if postgres data exists
    if not postgres_dir.exists():
        print(f"❌ PostgreSQL data directory not found: {postgres_dir}")
        print("   Please run 00_fetch_live_databases.py first")
        return 1

    required_files = ['token_states.parquet', 'account_states.parquet', 'trades.parquet']
    missing_files = [f for f in required_files if not (postgres_dir / f).exists()]

    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        print("   Please ensure PostgreSQL fetch completed successfully")
        return 1

    try:
        # Step 1: Consolidate token_states
        token_states_consolidated = consolidate_token_states(file_manager)

        # Step 2: Load existing consolidated_trades
        print_step("Loading existing consolidated_trades...")
        existing_trades_path = file_manager.server_dir / 'consolidated_trades.parquet'

        if not existing_trades_path.exists():
            print(f"❌ Consolidated trades not found: {existing_trades_path}")
            print("   Please run 01_consolidate_data.py first")
            return 1

        existing_trades = pd.read_parquet(existing_trades_path)
        print_metric("Existing trades loaded", f"{len(existing_trades):,} rows")

        # Step 3: Map postgres trades
        postgres_trades = map_postgres_trades_to_structure(file_manager, existing_trades)

        # Step 4: Merge trades
        merged_trades = merge_trades(existing_trades, postgres_trades)

        # Step 5: Save merged trades
        print_step("Saving merged trades...")
        output_path = file_manager.server_dir / 'consolidated_trades.parquet'
        merged_trades.to_parquet(output_path, index=False)
        print_success(f"Saved to: {output_path}")

        # Summary
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\n{'='*80}")
        print(f"✅ PostgreSQL consolidation completed in {duration:.2f}s")
        print(f"{'='*80}\n")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
