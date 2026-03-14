#!/usr/bin/env python3
"""
Data Consolidation Runner

Consolidates data from all live databases into unified parquet files
and generates an HTML report.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from research_notebooks.brigado_v2.modules.data_consolidator import DataConsolidator
from research_notebooks.brigado_v2.modules.file_manager import FileManager
from research_notebooks.brigado_v2.modules.html_generator import generate_consolidation_report_html, generate_index_html

# Get server name from environment or use default
SERVER_NAME = os.getenv('BRIGADO_SERVER', 'brigado')


def print_header(title: str):
    """Print a styled header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_step(step: str, status: str = "⏳"):
    """Print a step with status."""
    print(f"{status} {step}")


def print_success(message: str):
    """Print a success message."""
    print(f"✅ {message}")


def print_info(message: str, indent: int = 2):
    """Print an info message."""
    print(f"{' ' * indent}→ {message}")


def print_metric(label: str, value: str, indent: int = 4):
    """Print a metric."""
    print(f"{' ' * indent}{label}: {value}")


def main():
    """Main consolidation workflow."""
    start_time = datetime.now()

    print_header("🔄 DATA CONSOLIDATION RUNNER")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Initialize
    print_step(f"Initializing components for server '{SERVER_NAME}'...")
    consolidator = DataConsolidator(server_name=SERVER_NAME)
    file_manager = FileManager(server_name=SERVER_NAME)
    print_success(f"Components initialized for server '{SERVER_NAME}'")

    # Discover databases
    print_step("Discovering databases...")
    databases = consolidator.discover_databases()

    if not databases:
        print("❌ No databases found!")
        print_info("Please run fetch_live_databases first", indent=4)
        return 1

    print_success(f"Found {len(databases)} database(s)")
    for db in databases:
        print_info(f"{db['bot_name']} - {db['size_mb']:.2f} MB")

    # Consolidate data
    print_step("Consolidating data from all databases...")
    output_paths = consolidator.consolidate_all(force_refresh=True)
    print_success("Data consolidated successfully")

    # Show output files
    print_info("Generated files:")
    for data_type, path in output_paths.items():
        if data_type != 'metadata':
            size_mb = path.stat().st_size / (1024 * 1024)
            print_metric(data_type, f"{path.name} ({size_mb:.2f} MB)", indent=6)

    # Load consolidated data (only what we need for HTML report)
    print_step("Loading consolidated data...")
    data = consolidator.load_consolidated_data(use_latest=True)
    metadata = consolidator.get_consolidation_info()
    print_success("Data loaded")

    print_info("Summary:")
    print_metric("Trades", f"{len(data['trades']):,}", indent=6)
    print_metric("Orders", f"{len(data['orders']):,}", indent=6)
    print_metric("Executors", f"{len(data['executors']):,}", indent=6)
    print_metric("Controllers", f"{len(data['controllers']):,}", indent=6)

    # Store executor count before we start processing
    total_executors = len(data['executors'])

    # Create trade-to-controller mapping (optimized for large datasets)
    print_step("Creating trade-to-controller mapping...")
    order_to_controller = {}

    if 'executors' in data and len(data['executors']) > 0:
        executors_df = data['executors']

        # Filter first to reduce memory
        executors_filtered = executors_df[
            (executors_df['net_pnl_quote'] != 0) &
            (executors_df['controller_id'].notna())
        ][['controller_id', 'custom_info_parsed']].copy()

        print_info(f"Processing {len(executors_filtered):,} executors with controllers...")

        # Process in chunks to avoid memory issues
        chunk_size = 100000
        for i in range(0, len(executors_filtered), chunk_size):
            chunk = executors_filtered.iloc[i:i+chunk_size]

            for _, executor in chunk.iterrows():
                controller_id = executor['controller_id']
                custom_info = executor.get('custom_info_parsed')

                if isinstance(custom_info, dict):
                    order_ids = custom_info.get('order_ids', [])
                    if isinstance(order_ids, np.ndarray):
                        order_ids = order_ids.tolist()
                    elif not isinstance(order_ids, (list, tuple)):
                        order_ids = list(order_ids) if order_ids else []

                    for oid in order_ids:
                        if oid:
                            order_to_controller[str(oid)] = controller_id

            if (i + chunk_size) < len(executors_filtered):
                print_info(f"  Processed {i + chunk_size:,} / {len(executors_filtered):,} executors...", indent=4)

        # Clean up to free memory
        del executors_filtered
        del executors_df
        del data['executors']  # Free up executor memory - we don't need it anymore
        import gc
        gc.collect()

    trades_with_ctrl = data['trades'].copy()
    trades_with_ctrl['controller_id'] = trades_with_ctrl['order_id'].map(order_to_controller)
    coverage = (trades_with_ctrl['controller_id'].notna().sum() / len(trades_with_ctrl)) * 100

    print_success(f"Mapped {len(order_to_controller):,} orders to controllers")
    print_info(f"Coverage: {coverage:.1f}% of trades mapped")

    # Show per-bot coverage
    print_info("Coverage by bot:")
    for bot in trades_with_ctrl['source_bot'].unique():
        bot_trades = trades_with_ctrl[trades_with_ctrl['source_bot'] == bot]
        bot_mapped = bot_trades['controller_id'].notna().sum()
        bot_total = len(bot_trades)
        bot_pct = (bot_mapped / bot_total * 100) if bot_total > 0 else 0
        print_metric(bot, f"{bot_mapped}/{bot_total} ({bot_pct:.1f}%)", indent=6)

    # Generate HTML report
    print_step("Generating HTML consolidation report...")
    html_path = file_manager.reports_dir / "consolidation_report.html"
    generate_consolidation_report_html(
        output_path=html_path,
        metadata=metadata,
        databases=databases,
        trades_with_ctrl=trades_with_ctrl,
        controllers_data=data['controllers'],
        output_paths=output_paths
    )
    print_success(f"Report generated: {html_path.name}")

    # Generate index
    print_step("Updating index page...")
    index_path = generate_index_html(file_manager.reports_dir, metadata=metadata)
    print_success(f"Index updated: {index_path.name}")

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print_header("✨ CONSOLIDATION COMPLETE")
    print(f"Duration: {duration:.2f}s")
    print(f"Data directory: {file_manager.data_sources_dir}")
    print(f"Reports directory: {file_manager.reports_dir}")
    print(f"\n💡 Open {index_path} in your browser to view reports\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
