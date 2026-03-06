#!/usr/bin/env python3
"""
Evolutive Report Runner

Generates daily evolution report showing metrics progression over time
for market, bot, and controller perspectives.
"""

import sys
from pathlib import Path
from datetime import datetime
import asyncio
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from research_notebooks.brigado_v2.data_consolidator import DataConsolidator
from research_notebooks.brigado_v2.file_manager import FileManager
from research_notebooks.brigado_v2.evolutive_calculator import EvolutiveMetricsCalculator
from research_notebooks.brigado_v2.evolutive_report_generator import generate_evolutive_report_html
from research_notebooks.brigado_v2.html_generator import generate_index_html


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


async def main_async():
    """Main async workflow."""
    start_time = datetime.now()

    print_header("📈 EVOLUTIVE REPORT RUNNER")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Initialize
    print_step("Initializing components...")
    consolidator = DataConsolidator()
    file_manager = FileManager()
    root_path = str(Path(__file__).parent.parent.parent.parent)
    calculator = EvolutiveMetricsCalculator(root_path=root_path)
    print_success("Components initialized")

    # Load consolidated data
    print_step("Loading consolidated data...")
    data = consolidator.load_consolidated_data(use_latest=True)
    print_success("Data loaded")

    print_info("Summary:")
    print_metric("Trades", f"{len(data['trades']):,}", indent=6)
    print_metric("Executors", f"{len(data['executors']):,}", indent=6)
    print_metric("Controllers", f"{len(data['controllers']):,}", indent=6)

    # Create trade-to-controller mapping
    print_step("Creating trade-to-controller mapping...")
    import numpy as np

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

    trades = data['trades'].copy()
    trades['controller_id'] = trades['order_id'].map(order_to_controller)

    coverage = (trades['controller_id'].notna().sum() / len(trades)) * 100
    print_success(f"Mapped {len(order_to_controller):,} orders ({coverage:.1f}% coverage)")

    # Calculate evolutive metrics
    print_step("Calculating daily evolutive metrics...")
    evolutive_metrics = await calculator.calculate_evolutive_metrics(
        trades_with_ctrl=trades,
        controllers_data=data['controllers']
    )
    print_success(f"Generated {len(evolutive_metrics)} daily metric records")

    # Save metrics to parquet
    print_step("Saving evolutive metrics...")
    metrics_path = calculator.save_evolutive_metrics(evolutive_metrics)
    print_success(f"Metrics saved: {metrics_path.name}")

    # Generate HTML report
    print_step("Generating evolutive HTML report...")
    html_path = file_manager.data_sources_dir / "evolutive_report.html"

    generate_evolutive_report_html(
        evolutive_metrics=evolutive_metrics,
        output_path=html_path,
        metadata=consolidator.get_consolidation_info(),
        trades_with_ctrl=trades  # Pass enriched trades for pivot table
    )
    print_success(f"Report generated: {html_path.name}")

    # Update index
    print_step("Updating index page...")
    metadata = consolidator.get_consolidation_info()
    index_path = generate_index_html(file_manager.data_sources_dir, metadata=metadata)
    print_success(f"Index updated: {index_path.name}")

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print_header("✨ EVOLUTIVE REPORT COMPLETE")
    print(f"Duration: {duration:.2f}s")
    print(f"Output directory: {file_manager.data_sources_dir}")
    print(f"\n💡 Open {index_path} in your browser to view reports\n")

    return 0


def main():
    """Entry point."""
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
