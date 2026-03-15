#!/usr/bin/env python3
"""
Live Database Fetcher

Fetches SQLite databases and YAML configuration files from the remote Hummingbot server.
Also fetches PostgreSQL tables (token_states, account_states) and saves as parquet.
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
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from research_notebooks.brigado_v2.modules.file_manager import FileManager

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

# ============================================================================
#                            CONFIGURATION
# ============================================================================

# Get server name from environment or use default
SERVER_NAME = os.getenv('BRIGADO_SERVER', 'brigado')

# SSH Configuration - hostname to connect to
SSH_HOST = os.getenv('SSH_HOST', 'brigado')  # SSH hostname from ~/.ssh/config or IP
REMOTE_PATHS_STR = os.getenv('REMOTE_PATHS', 'hummingbot-api/bots/instances,hummingbot-api/bots/archived')
REMOTE_PATHS = [p.strip() for p in REMOTE_PATHS_STR.split(',')]

# PostgreSQL Configuration
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'brigado')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'hummingbot_api')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'hbot')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'hummingbot-api')
POSTGRES_DOCKER_CONTAINER = os.getenv('POSTGRES_DOCKER_CONTAINER', 'hummingbot-postgres')
POSTGRES_TABLES_STR = os.getenv('POSTGRES_TABLES', 'token_states,account_states')
POSTGRES_TABLES = [t.strip() for t in POSTGRES_TABLES_STR.split(',')]

# Local Configuration - use FileManager for consistent paths
file_manager = FileManager(server_name=SERVER_NAME)
LOCAL_BASE_PATH = file_manager.live_databases_dir
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
    # High-latency connection options
    ssh_opts = "-o ServerAliveInterval=60 -o ConnectTimeout=30 -o Compression=yes"
    full_command = f"ssh {ssh_opts} {SSH_HOST} '{command}'"

    try:
        result = subprocess.run(
            full_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60  # Increased timeout for high-latency
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out (exceeded 60s)"
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
    Download file or directory from remote server using rsync.

    rsync is superior to scp for high-latency connections because:
    - Delta sync: Only transfers changed portions
    - Resume capability: Can continue after interruptions
    - Built-in compression: Reduces transfer time
    - Progress monitoring: Shows transfer status
    - Bandwidth efficiency: Better protocol for long distances

    Returns:
        bool: True if successful, False otherwise
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # rsync options:
    # -a: archive mode (preserves permissions, timestamps, etc.)
    # -v: verbose (for debugging)
    # -z: compress during transfer
    # --timeout=300: timeout for I/O operations (5 minutes)
    # --contimeout=30: timeout for connection establishment
    # --partial: keep partially transferred files (enables resume)
    # --inplace: update files in-place (faster for large files)
    #
    # Note: SSH options (ServerAliveInterval, Compression, ControlMaster, etc.)
    # are configured in ~/.ssh/config for the brigado host

    if recursive:
        # For directories
        rsync_opts = "-avz --timeout=300 --contimeout=30 --partial --inplace"
        # Ensure trailing slash on source for correct behavior
        remote_source = f"{SSH_HOST}:{remote_path}/"
        target = local_path
    else:
        # For single files
        rsync_opts = "-avz --timeout=300 --contimeout=30 --partial --inplace"
        remote_source = f"{SSH_HOST}:{remote_path}"
        target = local_path

    command = f"rsync {rsync_opts} {remote_source} {target}"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes total timeout
        )

        if result.returncode != 0:
            # Log the actual error for debugging
            stderr = result.stderr.strip()
            print_error(f"rsync failed for {remote_path}: {stderr}")

        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print_error(f"rsync timeout for {remote_path} (exceeded 10 minutes)")
        return False
    except Exception as e:
        print_error(f"rsync exception for {remote_path}: {str(e)}")
        return False


def check_remote_path_exists(remote_path: str) -> bool:
    """Check if remote path exists."""
    command = f"test -e {remote_path} && echo 'exists' || echo 'not_found'"
    returncode, stdout, stderr = run_ssh_command(command)
    return 'exists' in stdout.lower()


# ============================================================================
#                      DATABASE FETCH FUNCTIONS
# ============================================================================

def fetch_bot_instance_data(bot_name: str, remote_base_path: str) -> Dict:
    """
    Fetch SQLite database and YAML configs for a specific bot instance.

    Args:
        bot_name: Name of the bot instance
        remote_base_path: Base path on remote server (instances or archived)

    Returns:
        Dict with fetch results

    Note:
        - Active instances: Always fetch (live databases)
        - Archived instances: Only fetch if not already present locally
    """
    # Determine source type from path
    source_type = "archived" if "archived" in remote_base_path else "active"
    is_archived = source_type == "archived"
    print_info(f"Processing: {bot_name} ({source_type})")

    result = {
        'bot_name': bot_name,
        'source_type': source_type,
        'remote_base_path': remote_base_path,
        'timestamp': datetime.now().isoformat(),
        'databases': [],
        'configs': [],
        'errors': [],
        'replaced_files': [],
        'skipped_files': []
    }

    # Create local directory for this bot
    bot_local_path = LOCAL_BASE_PATH / bot_name
    bot_local_path.mkdir(parents=True, exist_ok=True)

    # Fetch SQLite databases from data/ directory
    remote_data_path = f"{remote_base_path}/{bot_name}/data"

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

                # Check if file exists
                file_existed = local_db_path.exists()

                # For archived databases, skip if already exists locally
                if is_archived and file_existed:
                    file_size = local_db_path.stat().st_size
                    result['skipped_files'].append({
                        'filename': filename,
                        'type': 'database',
                        'reason': 'already_exists'
                    })
                    result['databases'].append({
                        'filename': filename,
                        'local_path': str(local_db_path),
                        'size_bytes': file_size,
                        'size_mb': round(file_size / (1024 * 1024), 2),
                        'skipped': True
                    })
                    print_metric(filename, f"{round(file_size / (1024 * 1024), 2)} MB (skipped - already exists)", indent=6)
                    continue

                # For active instances or new archived files, download
                if file_existed:
                    result['replaced_files'].append({
                        'filename': filename,
                        'type': 'database'
                    })

                success = scp_download(sqlite_file, local_db_path)

                if success:
                    file_size = local_db_path.stat().st_size
                    action = "replaced" if file_existed else "downloaded"

                    # For active instances, check and recover immediately after download
                    if not is_archived:
                        print_metric(filename, f"{round(file_size / (1024 * 1024), 2)} MB ({action}) - checking integrity...", indent=6)
                        is_corrupt, recovery_success, details = check_database_corruption(local_db_path)

                        if is_corrupt:
                            if recovery_success:
                                print_success(f"      ✓ Recovered: {details}")
                                action += ", recovered"
                            else:
                                print_warning(f"      ⚠ Recovery failed: {details}")
                                action += ", CORRUPT"
                        else:
                            print_info(f"      ✓ Integrity OK", indent=6)
                    else:
                        print_metric(filename, f"{round(file_size / (1024 * 1024), 2)} MB ({action})", indent=6)

                    result['databases'].append({
                        'filename': filename,
                        'local_path': str(local_db_path),
                        'size_bytes': file_size,
                        'size_mb': round(file_size / (1024 * 1024), 2),
                        'replaced': file_existed,
                        'skipped': False
                    })
                else:
                    result['errors'].append(f"Failed to download {filename}")

    # Fetch YAML configs from conf/controllers/ directory
    remote_config_path = f"{remote_base_path}/{bot_name}/conf/controllers"

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

                # For archived configs, skip if already exists locally
                if is_archived and file_existed:
                    file_size = local_config_path.stat().st_size
                    result['skipped_files'].append({
                        'filename': filename,
                        'type': 'config',
                        'reason': 'already_exists'
                    })
                    result['configs'].append({
                        'filename': filename,
                        'local_path': str(local_config_path),
                        'size_bytes': file_size,
                        'skipped': True
                    })
                    continue

                # For active instances or new archived files, download
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
                        'replaced': file_existed,
                        'skipped': False
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


def fetch_postgres_tables() -> Dict:
    """
    Fetch specified tables from PostgreSQL and save as parquet files.

    Returns:
        Dict with fetch results
    """
    print_step("Fetching PostgreSQL tables...")

    results = {
        'success': False,
        'tables_fetched': [],
        'errors': []
    }

    # Target directory for postgres data
    postgres_data_dir = file_manager.server_dir / 'postgres'
    postgres_data_dir.mkdir(exist_ok=True)

    try:
        for table_name in POSTGRES_TABLES:
            print_info(f"Fetching table: {table_name}")

            # Build psql command to export as CSV
            # We use docker exec to run psql inside the container
            psql_cmd = (
                f"docker exec {POSTGRES_DOCKER_CONTAINER} "
                f"psql -U {POSTGRES_USER} -d {POSTGRES_DB} "
                f"-c \"\\copy (SELECT * FROM {table_name}) TO STDOUT WITH CSV HEADER\""
            )

            ssh_cmd = f"ssh {POSTGRES_HOST} '{psql_cmd}'"

            try:
                result = subprocess.run(
                    ssh_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if result.returncode != 0:
                    error_msg = f"Failed to fetch {table_name}: {result.stderr}"
                    print_error(f"  {error_msg}")
                    results['errors'].append(error_msg)
                    continue

                # Parse CSV data with pandas
                from io import StringIO
                df = pd.read_csv(StringIO(result.stdout))

                # Save as parquet
                output_file = postgres_data_dir / f"{table_name}.parquet"
                df.to_parquet(output_file, index=False)

                file_size = output_file.stat().st_size / (1024 * 1024)  # MB
                print_success(f"  {table_name}: {len(df):,} rows, {file_size:.2f} MB")

                results['tables_fetched'].append({
                    'table': table_name,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'file': str(output_file),
                    'size_mb': round(file_size, 2)
                })

            except subprocess.TimeoutExpired:
                error_msg = f"Timeout fetching {table_name}"
                print_error(f"  {error_msg}")
                results['errors'].append(error_msg)
            except Exception as e:
                error_msg = f"Error processing {table_name}: {str(e)}"
                print_error(f"  {error_msg}")
                results['errors'].append(error_msg)

        results['success'] = len(results['tables_fetched']) > 0

        if results['success']:
            print_success(f"PostgreSQL fetch complete: {len(results['tables_fetched'])} table(s)")
        else:
            print_warning("No PostgreSQL tables were fetched successfully")

    except Exception as e:
        error_msg = f"PostgreSQL fetch failed: {str(e)}"
        print_error(error_msg)
        results['errors'].append(error_msg)

    return results


def main():
    """Main workflow."""
    start_time = datetime.now()
    fetch_timestamp = start_time.strftime("%Y-%m-%d_%H%M%S")

    print_header("📡 LIVE DATABASE FETCHER")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Server Name: {SERVER_NAME}")
    print(f"SSH Host: {SSH_HOST}")
    print(f"Remote Paths: {', '.join(REMOTE_PATHS)}")
    print(f"Local Path: {LOCAL_BASE_PATH}\n")

    # Discover bot instances from all remote paths
    print_step("Discovering bot instances on server...")
    all_bot_instances = []

    for remote_path in REMOTE_PATHS:
        source_type = "archived" if "archived" in remote_path else "active"
        print_info(f"Scanning {source_type}: {remote_path}")

        bot_instances = list_remote_directories(remote_path)

        if bot_instances:
            print_success(f"  Found {len(bot_instances)} bot(s) in {source_type}")
            for bot in bot_instances:
                all_bot_instances.append({
                    'bot_name': bot,
                    'remote_path': remote_path,
                    'source_type': source_type
                })
        else:
            print_warning(f"  No bots found in {source_type}")

    if not all_bot_instances:
        print_error("No bot instances found on server!")
        return 1

    print_success(f"Total: {len(all_bot_instances)} bot instance(s) across all paths")

    # Fetch data for each bot instance
    print_step("Fetching databases and configs...")
    results = []

    for bot_info in all_bot_instances:
        result = fetch_bot_instance_data(bot_info['bot_name'], bot_info['remote_path'])
        results.append(result)

    print_success(f"Fetch completed for {len(results)} bot(s)")

    # Fetch PostgreSQL tables
    postgres_results = fetch_postgres_tables()

    # Database corruption check for archived/skipped databases only
    # Active databases were already checked immediately after download
    print_step("Checking integrity of archived databases...")
    corruption_results = []

    for result in results:
        is_archived = result['source_type'] == 'archived'

        for db in result['databases']:
            db_path = Path(db['local_path'])

            # Skip if this was already checked during download (active instances)
            if not is_archived and not db.get('skipped', False):
                continue

            is_corrupt, recovery_success, details = check_database_corruption(db_path)

            corruption_results.append({
                'database': db['filename'],
                'bot_name': result['bot_name'],
                'source_type': result['source_type'],
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
    total_skipped = sum(len(r['skipped_files']) for r in results)
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
        'remote_paths': REMOTE_PATHS,
        'local_base_path': str(LOCAL_BASE_PATH),
        'total_instances': len(all_bot_instances),
        'total_databases': total_databases,
        'total_configs': total_configs,
        'total_replaced_files': total_replaced,
        'total_skipped_files': total_skipped,
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
        'postgres_summary': {
            'success': postgres_results['success'],
            'tables_fetched': len(postgres_results['tables_fetched']),
            'errors': len(postgres_results['errors'])
        },
        'instances': results,
        'corruption_details': corruption_results,
        'mapping_details': mapping_results,
        'postgres_details': postgres_results
    }

    with open(log_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print_header("✨ FETCH COMPLETE")
    print(f"Duration: {duration:.2f}s")
    print(f"Total Instances: {len(all_bot_instances)}")
    print(f"Total Databases: {total_databases}")
    print(f"Total Configs: {total_configs}")
    print(f"Files Replaced: {total_replaced}")
    print(f"Files Skipped (archived): {total_skipped}")
    print(f"Databases Corrupted: {total_corrupted}")
    print(f"Databases Recovered: {total_recovered}")
    print(f"Overall Mapping Coverage: {overall_coverage:.1f}%")
    print(f"PostgreSQL Tables Fetched: {len(postgres_results['tables_fetched'])}")
    if postgres_results['errors']:
        print(f"PostgreSQL Errors: {len(postgres_results['errors'])}")
    print(f"\nLog saved to: {log_file}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
