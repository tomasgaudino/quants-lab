#!/usr/bin/env python3
"""
Live Database Fetcher

Fetches SQLite databases and YAML configuration files from the remote Hummingbot server.
"""

import sys
import subprocess
import os
from datetime import datetime
from pathlib import Path
import logging
import json
from typing import List, Dict, Optional
import sqlite3
import pandas as pd


# ============================================================================
#                            CONFIGURATION
# ============================================================================

# SSH Configuration
SSH_HOST = "brigado"
REMOTE_BASE_PATH = "deploy/hummingbot-api/bots/instances"

# Local Configuration
LOCAL_BASE_PATH = Path(__file__).parent.parent / "data" / "live_databases"
LOCAL_BASE_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================================
#                         LOGGING FUNCTIONS
# ============================================================================

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


def print_warning(message: str):
    """Print a warning message."""
    print(f"⚠️  {message}")


def print_error(message: str):
    """Print an error message."""
    print(f"❌ {message}")


def print_info(message: str, indent: int = 2):
    """Print an info message."""
    print(f"{' ' * indent}→ {message}")


def print_metric(label: str, value: str, indent: int = 4):
    """Print a metric."""
    print(f"{' ' * indent}{label}: {value}")


# ============================================================================
#                         SSH UTILITY FUNCTIONS
# ============================================================================

def run_ssh_command(command: str) -> tuple:
    """
    Execute command on remote server via SSH.

    Returns:
        tuple: (return_code, stdout, stderr)
    """
    full_command = f"ssh {SSH_HOST} '{command}'"

    try:
        result = subprocess.run(
            full_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def list_remote_directories(remote_path: str) -> List[str]:
    """
    List all directories in remote path.

    Returns:
        List of directory names
    """
    command = f"ls -1 {remote_path}"
    returncode, stdout, stderr = run_ssh_command(command)

    if returncode != 0:
        print_error(f"Failed to list directories: {stderr}")
        return []

    directories = [d.strip() for d in stdout.split('\n') if d.strip()]
    return directories


def scp_download(remote_path: str, local_path: Path, recursive: bool = False) -> bool:
    """
    Download file or directory from remote server using scp.

    Returns:
        bool: True if successful, False otherwise
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)

    recursive_flag = "-r" if recursive else ""
    command = f"scp {recursive_flag} {SSH_HOST}:{remote_path} {local_path}"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes for large files
        )

        return result.returncode == 0
    except Exception:
        return False


def check_remote_path_exists(remote_path: str) -> bool:
    """Check if remote path exists."""
    command = f"test -e {remote_path} && echo 'exists' || echo 'not_found'"
    returncode, stdout, stderr = run_ssh_command(command)
    return 'exists' in stdout.lower()


# ============================================================================
#                      DATABASE FETCH FUNCTIONS
# ============================================================================

def fetch_bot_instance_data(bot_name: str) -> Dict:
    """
    Fetch SQLite database and YAML configs for a specific bot instance.

    Returns:
        Dict with fetch results
    """
    print_info(f"Processing: {bot_name}")

    result = {
        'bot_name': bot_name,
        'timestamp': datetime.now().isoformat(),
        'databases': [],
        'configs': [],
        'errors': [],
        'replaced_files': []
    }

    # Create local directory for this bot
    bot_local_path = LOCAL_BASE_PATH / bot_name
    bot_local_path.mkdir(parents=True, exist_ok=True)

    # Fetch SQLite databases from data/ directory
    remote_data_path = f"{REMOTE_BASE_PATH}/{bot_name}/data"

    if check_remote_path_exists(remote_data_path):
        # List .sqlite files
        command = f"ls -1 {remote_data_path}/*.sqlite 2>/dev/null || echo 'no_sqlite_files'"
        returncode, stdout, stderr = run_ssh_command(command)

        if 'no_sqlite_files' not in stdout:
            sqlite_files = [f.strip() for f in stdout.split('\n') if f.strip() and f.endswith('.sqlite')]

            for sqlite_file in sqlite_files:
                filename = os.path.basename(sqlite_file)
                local_db_path = bot_local_path / "data" / filename
                local_db_path.parent.mkdir(parents=True, exist_ok=True)

                # Check if file exists (will be replaced)
                file_existed = local_db_path.exists()
                if file_existed:
                    result['replaced_files'].append({
                        'filename': filename,
                        'type': 'database'
                    })

                success = scp_download(sqlite_file, local_db_path)

                if success:
                    file_size = local_db_path.stat().st_size
                    result['databases'].append({
                        'filename': filename,
                        'local_path': str(local_db_path),
                        'size_bytes': file_size,
                        'size_mb': round(file_size / (1024 * 1024), 2),
                        'replaced': file_existed
                    })
                    print_metric(filename, f"{round(file_size / (1024 * 1024), 2)} MB", indent=6)
                else:
                    result['errors'].append(f"Failed to download {filename}")

    # Fetch YAML configs from conf/controllers/ directory
    remote_config_path = f"{REMOTE_BASE_PATH}/{bot_name}/conf/controllers"

    if check_remote_path_exists(remote_config_path):
        # List .yml files
        command = f"ls -1 {remote_config_path}/*.yml 2>/dev/null || echo 'no_yml_files'"
        returncode, stdout, stderr = run_ssh_command(command)

        if 'no_yml_files' not in stdout:
            yml_files = [f.strip() for f in stdout.split('\n') if f.strip() and f.endswith('.yml')]

            for yml_file in yml_files:
                filename = os.path.basename(yml_file)
                local_config_path = bot_local_path / "conf" / "controllers" / filename
                local_config_path.parent.mkdir(parents=True, exist_ok=True)

                file_existed = local_config_path.exists()
                if file_existed:
                    result['replaced_files'].append({
                        'filename': filename,
                        'type': 'config'
                    })

                success = scp_download(yml_file, local_config_path)

                if success:
                    file_size = local_config_path.stat().st_size
                    result['configs'].append({
                        'filename': filename,
                        'local_path': str(local_config_path),
                        'size_bytes': file_size,
                        'replaced': file_existed
                    })

    return result


def check_database_corruption(db_path: Path) -> tuple:
    """
    Check if database is corrupted and attempt recovery if needed.

    Returns:
        tuple: (is_corrupt, recovery_success, details)
    """
    try:
        check_cmd = f'sqlite3 "{db_path}" "PRAGMA integrity_check;"'
        result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True, timeout=30)

        is_corrupt = False
        if result.returncode != 0 or 'ok' not in result.stdout.lower():
            is_corrupt = True

        if not is_corrupt:
            return False, None, "Database integrity OK"

        # Attempt recovery
        print_warning(f"Corruption detected in {db_path.name}, attempting recovery...")

        backup_path = db_path.parent / f"{db_path.stem}_corrupted_backup.sqlite"
        recovered_path = db_path.parent / f"{db_path.stem}_recovered.sqlite"

        # Run .recover command
        dump_cmd = f'sqlite3 "{db_path}" ".recover"'
        restore_cmd = f'sqlite3 "{recovered_path}"'

        try:
            dump_proc = subprocess.Popen(dump_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            restore_proc = subprocess.Popen(restore_cmd, shell=True, stdin=dump_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            dump_proc.stdout.close()

            stdout, stderr = restore_proc.communicate(timeout=120)

            if restore_proc.returncode == 0 and recovered_path.exists():
                # Verify recovered database
                verify_cmd = f'sqlite3 "{recovered_path}" "SELECT COUNT(*) FROM Executors WHERE net_pnl_quote != 0;"'
                verify_result = subprocess.run(verify_cmd, shell=True, capture_output=True, text=True, timeout=30)

                if verify_result.returncode == 0:
                    executor_count = verify_result.stdout.strip()

                    # Replace corrupted with recovered, then delete backup
                    db_path.rename(backup_path)
                    recovered_path.rename(db_path)
                    backup_path.unlink()  # Delete corrupted backup

                    return True, True, f"Recovered successfully ({executor_count} executors verified)"
                else:
                    return True, False, f"Recovery verification failed: {verify_result.stderr}"
            else:
                return True, False, f"Recovery failed: {stderr.decode()}"

        except subprocess.TimeoutExpired:
            return True, False, "Recovery timeout"
        except Exception as e:
            return True, False, f"Recovery exception: {e}"

    except Exception as e:
        return None, None, f"Check failed: {e}"


def verify_trade_mapping(db_path: Path) -> Dict:
    """
    Verify trade-to-controller mapping coverage.

    Returns:
        Dict with mapping statistics
    """
    try:
        conn = sqlite3.connect(db_path)

        # Load trades
        trades = pd.read_sql_query("SELECT order_id FROM TradeFill", conn)
        total_trades = len(trades)

        # Load executors and parse custom_info
        executors = pd.read_sql_query(
            "SELECT controller_id, custom_info, net_pnl_quote FROM Executors WHERE net_pnl_quote != 0",
            conn
        )

        # Parse custom_info and create mapping
        order_to_controller = {}
        executors_with_order_ids = 0

        for _, executor in executors.iterrows():
            controller_id = executor['controller_id']
            custom_info_str = executor['custom_info']

            if custom_info_str:
                try:
                    custom_info = json.loads(custom_info_str)
                    order_ids = custom_info.get('order_ids', [])

                    if order_ids:
                        executors_with_order_ids += 1
                        if isinstance(order_ids, list):
                            for oid in order_ids:
                                if oid:
                                    order_to_controller[str(oid)] = controller_id
                except:
                    pass

        # Test mapping
        trades['controller_id'] = trades['order_id'].map(order_to_controller)
        mapped_trades = trades['controller_id'].notna().sum()
        coverage_pct = (mapped_trades / total_trades * 100) if total_trades > 0 else 0

        # Get controller breakdown
        controller_counts = trades[trades['controller_id'].notna()]['controller_id'].value_counts().to_dict()

        conn.close()

        return {
            'total_trades': int(total_trades),
            'total_executors': int(len(executors)),
            'executors_with_order_ids': int(executors_with_order_ids),
            'total_order_mappings': len(order_to_controller),
            'mapped_trades': int(mapped_trades),
            'unmapped_trades': int(total_trades - mapped_trades),
            'coverage_percent': round(coverage_pct, 2),
            'controller_breakdown': controller_counts
        }

    except Exception as e:
        return {'error': str(e)}


def main():
    """Main workflow."""
    start_time = datetime.now()
    fetch_timestamp = start_time.strftime("%Y-%m-%d_%H%M%S")

    print_header("📡 LIVE DATABASE FETCHER")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Remote Server: {SSH_HOST}")
    print(f"Local Path: {LOCAL_BASE_PATH}\n")

    # Get list of all bot instances
    print_step("Discovering bot instances on server...")
    bot_instances = list_remote_directories(REMOTE_BASE_PATH)

    if not bot_instances:
        print_error("No bot instances found on server!")
        return 1

    print_success(f"Found {len(bot_instances)} bot instance(s)")
    for bot in bot_instances:
        print_info(bot)

    # Fetch data for each bot instance
    print_step("Fetching databases and configs...")
    results = []

    for bot_name in bot_instances:
        result = fetch_bot_instance_data(bot_name)
        results.append(result)

    print_success(f"Fetch completed for {len(results)} bot(s)")

    # Database corruption check
    print_step("Checking database integrity...")
    corruption_results = []

    for result in results:
        for db in result['databases']:
            db_path = Path(db['local_path'])
            is_corrupt, recovery_success, details = check_database_corruption(db_path)

            corruption_results.append({
                'database': db['filename'],
                'bot_name': result['bot_name'],
                'corrupt': is_corrupt,
                'recovered': recovery_success,
                'details': details
            })

            if is_corrupt is False:
                print_info(f"{db['filename']}: OK", indent=4)
            elif recovery_success:
                print_success(f"{db['filename']}: {details}")
            else:
                print_warning(f"{db['filename']}: {details}")

    # Trade mapping verification
    print_step("Verifying trade-to-controller mapping...")
    mapping_results = []

    for result in results:
        for db in result['databases']:
            db_path = Path(db['local_path'])
            mapping_stats = verify_trade_mapping(db_path)

            if 'error' not in mapping_stats:
                mapping_results.append({
                    'database': db['filename'],
                    'bot_name': result['bot_name'],
                    **mapping_stats
                })

                coverage = mapping_stats['coverage_percent']
                print_info(f"{db['filename']}: {coverage:.1f}% coverage ({mapping_stats['mapped_trades']}/{mapping_stats['total_trades']} trades)", indent=4)
            else:
                print_warning(f"{db['filename']}: {mapping_stats['error']}")

    # Calculate totals
    total_databases = sum(len(r['databases']) for r in results)
    total_configs = sum(len(r['configs']) for r in results)
    total_replaced = sum(len(r['replaced_files']) for r in results)
    total_corrupted = sum(1 for cr in corruption_results if cr['corrupt'])
    total_recovered = sum(1 for cr in corruption_results if cr['recovered'])

    total_trades_all = sum(r.get('total_trades', 0) for r in mapping_results)
    total_mapped_all = sum(r.get('mapped_trades', 0) for r in mapping_results)
    overall_coverage = (total_mapped_all / total_trades_all * 100) if total_trades_all > 0 else 0

    # Save logs
    log_file = LOCAL_BASE_PATH / f"fetch_log_{fetch_timestamp}.json"

    summary = {
        'status': 'completed',
        'timestamp': datetime.now().isoformat(),
        'fetch_id': fetch_timestamp,
        'ssh_host': SSH_HOST,
        'remote_base_path': REMOTE_BASE_PATH,
        'local_base_path': str(LOCAL_BASE_PATH),
        'total_instances': len(bot_instances),
        'total_databases': total_databases,
        'total_configs': total_configs,
        'total_replaced_files': total_replaced,
        'corruption_summary': {
            'databases_checked': len(corruption_results),
            'databases_corrupted': total_corrupted,
            'databases_recovered': total_recovered,
        },
        'mapping_summary': {
            'total_trades': total_trades_all,
            'total_mapped': total_mapped_all,
            'overall_coverage': round(overall_coverage, 2)
        },
        'instances': results,
        'corruption_details': corruption_results,
        'mapping_details': mapping_results
    }

    with open(log_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print_header("✨ FETCH COMPLETE")
    print(f"Duration: {duration:.2f}s")
    print(f"Total Instances: {len(bot_instances)}")
    print(f"Total Databases: {total_databases}")
    print(f"Total Configs: {total_configs}")
    print(f"Files Replaced: {total_replaced}")
    print(f"Databases Corrupted: {total_corrupted}")
    print(f"Databases Recovered: {total_recovered}")
    print(f"Overall Mapping Coverage: {overall_coverage:.1f}%")
    print(f"\nLog saved to: {log_file}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
