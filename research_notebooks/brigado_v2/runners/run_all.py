#!/usr/bin/env python3
"""
Run All Reports

Master runner that executes all report generation scripts in sequence.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime


def print_banner():
    """Print ASCII banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                                                                           ║
    ║                        🚀 BRIGADO V2 REPORT SUITE                         ║
    ║                                                                           ║
    ║                     Automated Performance Analysis                        ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_separator():
    """Print a separator line."""
    print(f"\n{'─' * 80}\n")


def run_script(script_path: Path, name: str) -> tuple:
    """Run a script and return its exit code and duration."""
    print(f"\n┌{'─' * 78}┐")
    print(f"│ Running: {name:<68} │")
    print(f"└{'─' * 78}┘\n")

    script_start = datetime.now()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent.parent.parent.parent,
            check=False
        )
        duration = (datetime.now() - script_start).total_seconds()
        return result.returncode, duration
    except Exception as e:
        print(f"❌ Error running {name}: {e}")
        duration = (datetime.now() - script_start).total_seconds()
        return 1, duration


def main():
    """Main workflow."""
    start_time = datetime.now()

    print_banner()
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator()

    # Get script directory
    script_dir = Path(__file__).parent

    # Define runners in order
    runners = [
        (script_dir / "00_fetch_live_databases.py", "Fetch Live Databases"),
        (script_dir / "00b_consolidate_postgres_data.py", "PostgreSQL Data Consolidation"),
        (script_dir / "01_consolidate_data.py", "Data Consolidation"),
        (script_dir / "02_market_analysis.py", "Market Analysis"),
        (script_dir / "03_evolutive_report.py", "Evolutive Report"),
        (script_dir / "04_portfolio_evolution.py", "Portfolio Evolution"),
        (script_dir / "05_executive_dashboard.py", "Executive Dashboard"),
    ]

    # Track results
    results = []

    # Run each script
    for script_path, name in runners:
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            results.append((name, 1, 0.0))
            continue

        exit_code, duration = run_script(script_path, name)
        results.append((name, exit_code, duration))

        if exit_code != 0:
            print(f"\n⚠️  {name} completed with errors (exit code: {exit_code}) - {duration:.1f}s")
        else:
            print(f"\n✅ {name} completed successfully - {duration:.1f}s")

        print_separator()

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n╔═══════════════════════════════════════════════════════════════════════════╗")
    print(f"║                              EXECUTION SUMMARY                            ║")
    print(f"╚═══════════════════════════════════════════════════════════════════════════╝\n")

    for name, exit_code, script_duration in results:
        status = "✅ SUCCESS" if exit_code == 0 else "❌ FAILED "
        print(f"  {status}  {name:<45} ({script_duration:>6.1f}s)")

    print(f"\n{'─' * 80}")
    print(f"Total Duration: {duration:.2f}s ({duration/60:.1f} minutes)")
    print(f"Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Overall status
    all_success = all(exit_code == 0 for _, exit_code, _ in results)

    if all_success:
        print(f"\n{'='*80}")
        print("✨ ALL REPORTS GENERATED SUCCESSFULLY!")
        print(f"{'='*80}\n")
        print("💡 Open data/data_sources/index.html in your browser to view reports")
        print()
        return 0
    else:
        print(f"\n{'='*80}")
        print("⚠️  SOME REPORTS FAILED - CHECK LOGS ABOVE")
        print(f"{'='*80}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
