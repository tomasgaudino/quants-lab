#!/usr/bin/env python3
"""
Data Consolidation Runner

Consolidates data from all live databases into unified parquet files
and generates an HTML report.
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from research_notebooks.brigado_v2.data_consolidator import DataConsolidator
from research_notebooks.brigado_v2.file_manager import FileManager
from research_notebooks.brigado_v2.html_generator import generate_consolidation_report_html, generate_index_html


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
    print_step("Initializing components...")
    consolidator = DataConsolidator()
    file_manager = FileManager()
    print_success("Components initialized")

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

    # Load consolidated data
    print_step("Loading consolidated data...")
    data = consolidator.load_consolidated_data(use_latest=True)
    metadata = consolidator.get_consolidation_info()
    print_success("Data loaded")

    print_info("Summary:")
    print_metric("Trades", f"{len(data['trades']):,}", indent=6)
    print_metric("Orders", f"{len(data['orders']):,}", indent=6)
    print_metric("Executors", f"{len(data['executors']):,}", indent=6)
    print_metric("Controllers", f"{len(data['controllers']):,}", indent=6)

    # Create trade-to-controller mapping
    print_step("Creating trade-to-controller mapping...")
    order_to_controller = {}

    if 'executors' in data and len(data['executors']) > 0:
        executors_df = data['executors']
        executors_filtered = executors_df[executors_df['net_pnl_quote'] != 0]

        for _, executor in executors_filtered.iterrows():
            controller_id = executor.get('controller_id')
            if controller_id and 'custom_info_parsed' in executor and isinstance(executor['custom_info_parsed'], dict):
                order_ids = executor['custom_info_parsed'].get('order_ids', [])
                if isinstance(order_ids, np.ndarray):
                    order_ids = order_ids.tolist()
                elif not isinstance(order_ids, (list, tuple)):
                    order_ids = list(order_ids) if order_ids else []
                for oid in order_ids:
                    if oid:
                        order_to_controller[str(oid)] = controller_id

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
    html_path = file_manager.data_sources_dir / "consolidation_report.html"
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
    index_path = generate_index_html(file_manager.data_sources_dir, metadata=metadata)
    print_success(f"Index updated: {index_path.name}")

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print_header("✨ CONSOLIDATION COMPLETE")
    print(f"Duration: {duration:.2f}s")
    print(f"Output directory: {file_manager.data_sources_dir}")
    print(f"\n💡 Open {index_path} in your browser to view reports\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
