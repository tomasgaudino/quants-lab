"""
File Manager Module

Provides utilities for managing generated files with consistent naming conventions.

TODO - Future Improvements:
- Add automatic file archiving
- Implement file versioning
- Add compression for old files
- Support for cloud storage (S3, GCS)
"""

import os
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import pandas as pd


class FileManager:
    """
    Manages file naming, discovery, and organization for analysis outputs.

    File Naming Convention:
        {file_type}_{controller_id}_{start_date}_{end_date}.{extension}

    Examples:
        daily_report_pmm-mister-brigado-binance-18-1_20251223_20260128.csv
        enriched_trades_all_20251223_20260128.csv
        full_export_pmm-mister-brigado-binance-18-1_20251223_20260128.xlsx
    """

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize file manager.

        Args:
            base_path: Base path for data directory (defaults to brigado_v2/data)
        """
        if base_path is None:
            # Default to brigado_v2/data
            current_file = Path(__file__)
            base_path = current_file.parent / "data"

        self.base_path = Path(base_path)
        self.reports_dir = self.base_path / "reports"
        self.enriched_dir = self.base_path / "enriched"
        self.exports_dir = self.base_path / "exports"
        self.html_dir = self.base_path / "exports"  # HTML reports go in exports
        self.live_databases_dir = self.base_path / "live_databases"
        self.data_sources_dir = self.base_path / "data_sources"

        # Ensure directories exist
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create directory structure if it doesn't exist."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.enriched_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.live_databases_dir.mkdir(parents=True, exist_ok=True)
        self.data_sources_dir.mkdir(parents=True, exist_ok=True)

    def build_filename(
        self,
        file_type: str,
        controller_id: str = "all",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        extension: str = "csv",
        use_latest: bool = False
    ) -> str:
        """
        Build filename following the naming convention.

        Args:
            file_type: Type of file (daily_report, enriched_trades, full_export)
            controller_id: Controller ID or 'all' for multi-controller
            start_date: Start date in YYYYMMDD format (optional)
            end_date: End date in YYYYMMDD format (optional)
            extension: File extension (csv, xlsx, json)
            use_latest: Use 'latest' instead of dates

        Returns:
            Filename string

        TODO:
        - Add validation for date formats
        - Support for custom suffixes
        """
        if use_latest:
            return f"{file_type}_latest.{extension}"

        # Build filename parts
        parts = [file_type]

        # Sanitize controller_id for filename
        if controller_id:
            safe_controller_id = controller_id.replace("/", "-").replace("\\", "-")
            parts.append(safe_controller_id)

        # Add dates if provided
        if start_date and end_date:
            parts.append(start_date)
            parts.append(end_date)
        elif start_date:
            parts.append(start_date)

        filename = "_".join(parts) + f".{extension}"
        return filename

    def get_daily_report_path(
        self,
        controller_id: str = "all",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_latest: bool = False
    ) -> Path:
        """
        Get path for daily performance report.

        Args:
            controller_id: Controller ID
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            use_latest: Use latest file naming

        Returns:
            Full path to daily report CSV
        """
        filename = self.build_filename(
            file_type="daily_report",
            controller_id=controller_id,
            start_date=start_date,
            end_date=end_date,
            extension="csv",
            use_latest=use_latest
        )
        return self.reports_dir / filename

    def get_enriched_trades_path(
        self,
        controller_id: str = "all",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_latest: bool = False
    ) -> Path:
        """
        Get path for enriched trades CSV.

        Args:
            controller_id: Controller ID
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            use_latest: Use latest file naming

        Returns:
            Full path to enriched trades CSV
        """
        filename = self.build_filename(
            file_type="enriched_trades",
            controller_id=controller_id,
            start_date=start_date,
            end_date=end_date,
            extension="csv",
            use_latest=use_latest
        )
        return self.enriched_dir / filename

    def get_html_report_path(
        self,
        controller_id: str = "all",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_latest: bool = False
    ) -> Path:
        """
        Get path for HTML report.

        Args:
            controller_id: Controller ID
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            use_latest: Use latest file naming

        Returns:
            Full path to HTML report
        """
        filename = self.build_filename(
            file_type="report",
            controller_id=controller_id,
            start_date=start_date,
            end_date=end_date,
            extension="html",
            use_latest=use_latest
        )
        return self.html_dir / filename

    def get_full_export_path(
        self,
        controller_id: str = "all",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_latest: bool = False
    ) -> Path:
        """
        Get path for full Excel export.

        Args:
            controller_id: Controller ID
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            use_latest: Use latest file naming

        Returns:
            Full path to Excel export
        """
        filename = self.build_filename(
            file_type="full_export",
            controller_id=controller_id,
            start_date=start_date,
            end_date=end_date,
            extension="xlsx",
            use_latest=use_latest
        )
        return self.exports_dir / filename

    def get_latest_daily_report(self) -> Optional[Path]:
        """
        Get the most recently modified daily report file.

        Returns:
            Path to latest daily report or None if no reports exist

        TODO:
        - Add filtering by controller_id
        - Return metadata (dates, size, etc)
        """
        return self._get_latest_file(self.reports_dir, "daily_report_*.csv")

    def get_latest_enriched_trades(self) -> Optional[Path]:
        """
        Get the most recently modified enriched trades file.

        Returns:
            Path to latest enriched trades or None if no files exist
        """
        return self._get_latest_file(self.enriched_dir, "enriched_trades_*.csv")

    def get_latest_export(self) -> Optional[Path]:
        """
        Get the most recently modified Excel export.

        Returns:
            Path to latest export or None if no exports exist
        """
        return self._get_latest_file(self.exports_dir, "full_export_*.xlsx")

    def _get_latest_file(self, directory: Path, pattern: str) -> Optional[Path]:
        """Get the most recently modified file matching pattern."""
        files = list(directory.glob(pattern))
        if not files:
            return None
        # Sort by modification time, newest first
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return files[0]

    def list_daily_reports(self) -> List[dict]:
        """
        List all daily reports with metadata.

        Returns:
            List of dicts with file info (path, controller_id, dates, size, modified)

        TODO:
        - Add filtering options
        - Add sorting options
        - Parse controller_id and dates from filename
        """
        reports = []
        for file_path in self.reports_dir.glob("daily_report_*.csv"):
            reports.append({
                'path': file_path,
                'filename': file_path.name,
                'size_mb': file_path.stat().st_size / 1024 / 1024,
                'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
            })
        # Sort by modification time, newest first
        reports.sort(key=lambda x: x['modified'], reverse=True)
        return reports

    def list_enriched_trades(self) -> List[dict]:
        """
        List all enriched trade files with metadata.

        Returns:
            List of dicts with file info
        """
        files = []
        for file_path in self.enriched_dir.glob("enriched_trades_*.csv"):
            files.append({
                'path': file_path,
                'filename': file_path.name,
                'size_mb': file_path.stat().st_size / 1024 / 1024,
                'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
            })
        files.sort(key=lambda x: x['modified'], reverse=True)
        return files

    def list_exports(self) -> List[dict]:
        """
        List all Excel exports with metadata.

        Returns:
            List of dicts with file info
        """
        files = []
        for file_path in self.exports_dir.glob("full_export_*.xlsx"):
            files.append({
                'path': file_path,
                'filename': file_path.name,
                'size_mb': file_path.stat().st_size / 1024 / 1024,
                'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
            })
        files.sort(key=lambda x: x['modified'], reverse=True)
        return files

    def extract_dates_from_dataframe(self, df: pd.DataFrame, date_column: str = 'date') -> tuple:
        """
        Extract start and end dates from DataFrame.

        Args:
            df: DataFrame with date column
            date_column: Name of date column

        Returns:
            Tuple of (start_date, end_date) in YYYYMMDD format

        TODO:
        - Add validation for date column
        - Support for different date formats
        """
        df[date_column] = pd.to_datetime(df[date_column])
        start_date = df[date_column].min().strftime('%Y%m%d')
        end_date = df[date_column].max().strftime('%Y%m%d')
        return start_date, end_date

    def cleanup_old_files(self, days: int = 30, dry_run: bool = True) -> List[Path]:
        """
        Remove files older than specified days.

        Args:
            days: Delete files older than this many days
            dry_run: If True, only list files without deleting

        Returns:
            List of files that were (or would be) deleted

        TODO:
        - Add size-based cleanup
        - Keep most recent N files regardless of age
        - Archive instead of delete
        """
        from datetime import timedelta

        cutoff_time = datetime.now() - timedelta(days=days)
        cutoff_timestamp = cutoff_time.timestamp()

        files_to_delete = []

        # Check all data directories
        for directory in [self.reports_dir, self.enriched_dir, self.exports_dir]:
            for file_path in directory.iterdir():
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_timestamp:
                    files_to_delete.append(file_path)

        if not dry_run:
            for file_path in files_to_delete:
                file_path.unlink()
                print(f"Deleted: {file_path}")
        else:
            print(f"Would delete {len(files_to_delete)} files (dry run)")
            for file_path in files_to_delete:
                print(f"  - {file_path}")

        return files_to_delete

    def discover_live_databases(self) -> List[dict]:
        """
        Discover all live databases in the live_databases directory.

        Returns:
            List of dicts with database info (bot_name, db_path, config_files, size, modified)

        TODO:
        - Add filtering by bot name pattern
        - Add sorting options
        """
        databases = []

        if not self.live_databases_dir.exists():
            return databases

        for bot_dir in self.live_databases_dir.iterdir():
            if not bot_dir.is_dir():
                continue  # Skip files like fetch logs

            # Look for SQLite database
            data_dir = bot_dir / "data"
            if not data_dir.exists():
                continue

            sqlite_files = list(data_dir.glob("*.sqlite"))
            if not sqlite_files:
                continue

            db_path = sqlite_files[0]

            # Look for config files
            config_dir = bot_dir / "conf" / "controllers"
            config_files = list(config_dir.glob("*.yml")) if config_dir.exists() else []

            databases.append({
                'bot_name': bot_dir.name,
                'db_path': db_path,
                'config_dir': config_dir if config_dir.exists() else None,
                'config_files': [f.name for f in config_files],
                'num_configs': len(config_files),
                'size_mb': db_path.stat().st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(db_path.stat().st_mtime)
            })

        # Sort by modification time, newest first
        databases.sort(key=lambda x: x['modified'], reverse=True)
        return databases

    def get_latest_consolidated_data(self) -> Optional[Path]:
        """
        Get path to latest consolidated data parquet file.

        Returns:
            Path to latest consolidated trades parquet or None
        """
        latest_path = self.data_sources_dir / "consolidated_trades_latest.parquet"
        if latest_path.exists():
            return latest_path
        return None

    def list_consolidated_data(self) -> List[dict]:
        """
        List all consolidated data files with metadata.

        Returns:
            List of dicts with file info
        """
        files = []
        for file_path in self.data_sources_dir.glob("consolidated_trades_*.parquet"):
            if "latest" not in file_path.name:  # Skip the latest symlink-like file
                files.append({
                    'path': file_path,
                    'filename': file_path.name,
                    'type': 'trades',
                    'size_mb': file_path.stat().st_size / 1024 / 1024,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
                })

        for file_path in self.data_sources_dir.glob("consolidated_orders_*.parquet"):
            if "latest" not in file_path.name:
                files.append({
                    'path': file_path,
                    'filename': file_path.name,
                    'type': 'orders',
                    'size_mb': file_path.stat().st_size / 1024 / 1024,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
                })

        files.sort(key=lambda x: x['modified'], reverse=True)
        return files

    def print_summary(self) -> None:
        """Print summary of all files in data directory."""
        print("=" * 80)
        print(" " * 25 + "DATA DIRECTORY SUMMARY")
        print("=" * 80)
        print()

        # Daily reports
        reports = self.list_daily_reports()
        print(f"📊 Daily Reports ({len(reports)} files)")
        print(f"   Location: {self.reports_dir}")
        if reports:
            latest = reports[0]
            print(f"   Latest: {latest['filename']}")
            print(f"   Modified: {latest['modified']}")
            print(f"   Size: {latest['size_mb']:.2f} MB")
        print()

        # Enriched trades
        enriched = self.list_enriched_trades()
        print(f"📈 Enriched Trades ({len(enriched)} files)")
        print(f"   Location: {self.enriched_dir}")
        if enriched:
            latest = enriched[0]
            print(f"   Latest: {latest['filename']}")
            print(f"   Modified: {latest['modified']}")
            print(f"   Size: {latest['size_mb']:.2f} MB")
        print()

        # Excel exports
        exports = self.list_exports()
        print(f"📁 Excel Exports ({len(exports)} files)")
        print(f"   Location: {self.exports_dir}")
        if exports:
            latest = exports[0]
            print(f"   Latest: {latest['filename']}")
            print(f"   Modified: {latest['modified']}")
            print(f"   Size: {latest['size_mb']:.2f} MB")
        print()

        # Live databases
        live_dbs = self.discover_live_databases()
        print(f"🗄️  Live Databases ({len(live_dbs)} instances)")
        print(f"   Location: {self.live_databases_dir}")
        if live_dbs:
            for db in live_dbs[:3]:  # Show first 3
                print(f"   - {db['bot_name']}: {db['size_mb']:.2f} MB, {db['num_configs']} configs")
        print()

        # Consolidated data sources
        consolidated = self.list_consolidated_data()
        print(f"📦 Consolidated Data ({len(consolidated)} files)")
        print(f"   Location: {self.data_sources_dir}")
        if consolidated:
            latest = consolidated[0]
            print(f"   Latest: {latest['filename']}")
            print(f"   Modified: {latest['modified']}")
            print(f"   Size: {latest['size_mb']:.2f} MB")
        print()

        # Total size
        total_size = sum(f['size_mb'] for f in reports + enriched + exports)
        live_db_size = sum(db['size_mb'] for db in live_dbs)
        consolidated_size = sum(f['size_mb'] for f in consolidated)
        print(f"💾 Total Size: {total_size + live_db_size + consolidated_size:.2f} MB")
        print(f"   Reports/Exports: {total_size:.2f} MB")
        print(f"   Live Databases: {live_db_size:.2f} MB")
        print(f"   Consolidated Data: {consolidated_size:.2f} MB")
        print("=" * 80)
