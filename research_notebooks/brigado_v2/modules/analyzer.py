"""
Trading Performance Analyzer

Main orchestrator class that coordinates all components for Hummingbot performance analysis.

TODO - Future Improvements:
- Add caching for expensive operations (enrichment, market data)
- Implement async operations for parallel processing
- Add configuration file support (YAML/JSON)
- Add command-line interface
- Support for live streaming updates
- Add export to multiple formats (Excel, PDF, HTML)
- Add email/slack notifications for reports
"""

import asyncio
from typing import Optional
import pandas as pd

from research_notebooks.brigado_v2.modules.data_loader import DataLoader
from research_notebooks.brigado_v2.modules.trade_enricher import TradeEnricher
from research_notebooks.brigado_v2.modules.performance_calculator import PerformanceCalculator
from research_notebooks.brigado_v2.modules.visualizer import PerformanceVisualizer
from research_notebooks.brigado_v2.modules.file_manager import FileManager
from research_notebooks.brigado_v2.modules.html_generator import HTMLReportGenerator


class TradingPerformanceAnalyzer:
    """
    Main orchestrator for Hummingbot trading performance analysis.

    This class coordinates all the components:
    1. DataLoader: Loads raw data from SQLite
    2. TradeEnricher: Enriches trades with controller attribution
    3. PerformanceCalculator: Calculates daily metrics
    4. PerformanceVisualizer: Creates visualizations

    Usage:
        analyzer = TradingPerformanceAnalyzer(
            db_name="your-database.sqlite",
            root_path="/path/to/project"
        )

        # Generate complete analysis
        await analyzer.run_full_analysis()

        # Or run step by step
        await analyzer.load_data()
        await analyzer.enrich_trades()
        await analyzer.calculate_performance()
        analyzer.create_visualizations()
    """

    def __init__(
        self,
        db_name: str,
        root_path: str,
        server_name: str = "brigado_server",
        rebate_pct: float = 0.015 / 100,
        initial_portfolio_quote: Optional[float] = None
    ):
        """
        Initialize the trading performance analyzer.

        Args:
            db_name: SQLite database filename
            root_path: Root path for the project
            server_name: Server name for database location
            rebate_pct: Exchange rebate percentage (default: 0.015%)
            initial_portfolio_quote: Initial portfolio in quote asset (auto-detected if None)

        TODO:
        - Add validation for parameters
        - Support for multiple databases
        - Add configuration from file
        """
        self.db_name = db_name
        self.root_path = root_path
        self.server_name = server_name
        self.rebate_pct = rebate_pct
        self.initial_portfolio_quote = initial_portfolio_quote

        # Initialize components
        self.data_loader = DataLoader(
            db_name=db_name,
            root_path=root_path,
            server_name=server_name
        )
        self.trade_enricher = TradeEnricher()
        self.performance_calculator = PerformanceCalculator(root_path=root_path)
        self.visualizer = PerformanceVisualizer(rebate_pct=rebate_pct)
        self.file_manager = FileManager()
        self.html_generator = HTMLReportGenerator(rebate_pct=rebate_pct)

        # Data storage
        self.raw_data: Optional[dict] = None
        self.enriched_trades: Optional[pd.DataFrame] = None
        self.daily_report: Optional[pd.DataFrame] = None
        self.controller_id: Optional[str] = None  # Extracted from data
        self.quote_asset: str = "USDT"  # Will be detected from database

    async def run_full_analysis(
        self,
        use_standard_naming: bool = True,
        save_enriched_trades: bool = True
    ) -> pd.DataFrame:
        """
        Run complete analysis pipeline: load -> enrich -> calculate -> visualize.

        Args:
            use_standard_naming: Use standard file naming convention (recommended)
            save_enriched_trades: Whether to save enriched trades to CSV

        Returns:
            Daily performance report DataFrame

        TODO:
        - Add progress callbacks
        - Add error recovery (checkpoint/resume)
        - Add validation at each step
        - Support for incremental updates (only process new data)
        """
        print("=" * 80)
        print(" " * 25 + "TRADING PERFORMANCE ANALYSIS")
        print("=" * 80)
        print()

        # Step 1: Load data
        print("Step 1/4: Loading data from database...")
        await self.load_data()
        print(f"  ✓ Loaded {len(self.raw_data['trade_fills'])} trade fills")
        print(f"  ✓ Loaded {len(self.raw_data['executors'])} executors")
        print(f"  ✓ Loaded {len(self.raw_data['controllers'])} controllers")
        print()

        # Step 2: Enrich trades
        print("Step 2/4: Enriching trades with controller attribution...")
        self.enrich_trades()
        controller_coverage = (
            self.enriched_trades['controller_id'].notna().sum() /
            len(self.enriched_trades) * 100
        )
        print(f"  ✓ Enriched {len(self.enriched_trades)} trades")
        print(f"  ✓ Controller coverage: {controller_coverage:.2f}%")

        # Extract controller_id for file naming
        self.controller_id = self.enriched_trades['controller_id'].mode()[0] if 'controller_id' in self.enriched_trades.columns else "all"

        # Determine file paths
        if use_standard_naming:
            start_date, end_date = self.file_manager.extract_dates_from_dataframe(self.enriched_trades, 'timestamp')
            enriched_path = self.file_manager.get_enriched_trades_path(
                controller_id=self.controller_id,
                start_date=start_date,
                end_date=end_date
            )
        else:
            enriched_path = self.file_manager.get_enriched_trades_path(use_latest=True)

        if save_enriched_trades:
            self.enriched_trades.to_csv(enriched_path, index=False)
            print(f"  ✓ Saved enriched trades to: {enriched_path}")
        print()

        # Step 3: Calculate performance
        print("Step 3/4: Calculating daily performance metrics...")

        # Determine report path
        if use_standard_naming:
            report_path = self.file_manager.get_daily_report_path(
                controller_id=self.controller_id,
                start_date=start_date,
                end_date=end_date
            )
        else:
            report_path = self.file_manager.get_daily_report_path(use_latest=True)

        await self.calculate_performance(output_path=str(report_path))
        print(f"  ✓ Generated report for {len(self.daily_report['date'].unique())} days")
        print(f"  ✓ Tracking {self.daily_report['controller_id'].nunique()} controllers")
        print(f"  ✓ Saved report to: {report_path}")
        print()

        # Step 4: Create visualizations
        print("Step 4/4: Creating visualizations...")
        self.create_visualizations()
        print("  ✓ Executive dashboard created")
        print("  ✓ Summary tables printed")
        print()

        print("=" * 80)
        print("Analysis complete!")
        print("=" * 80)
        print()

        return self.daily_report

    async def load_data(self) -> dict:
        """
        Load all data from database.

        Returns:
            Dictionary with trade_fills, orders, executors, controllers

        TODO:
        - Add data validation
        - Add data quality checks
        - Cache loaded data
        """
        self.raw_data = self.data_loader.load_all()
        # Detect quote asset from loaded data
        self.quote_asset = self.data_loader.get_quote_asset()
        # Update components with detected quote asset
        self.visualizer.quote_asset = self.quote_asset
        self.html_generator.quote_asset = self.quote_asset
        return self.raw_data

    def enrich_trades(self) -> pd.DataFrame:
        """
        Enrich trades with controller attribution and PnL calculations.

        Returns:
            Enriched trades DataFrame

        TODO:
        - Add validation for mapping quality
        - Add warnings for orphan trades
        - Add position reconciliation checks
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self.enriched_trades = self.trade_enricher.enrich_trades(
            trade_fills=self.raw_data['trade_fills'],
            orders=self.raw_data['orders'],
            executors=self.raw_data['executors'],
            controllers=self.raw_data['controllers']
        )

        return self.enriched_trades

    async def calculate_performance(
        self,
        output_path: str = "daily_performance_report.csv"
    ) -> pd.DataFrame:
        """
        Calculate daily performance metrics.

        Args:
            output_path: Path to save CSV report

        Returns:
            Daily performance report DataFrame

        TODO:
        - Add parallel processing for dates
        - Add caching for market data
        - Support for custom date ranges
        """
        if self.enriched_trades is None:
            raise ValueError("Trades not enriched. Call enrich_trades() first.")

        self.daily_report = await self.performance_calculator.generate_daily_report(
            enriched_trades=self.enriched_trades,
            output_path=output_path
        )

        # Auto-detect initial portfolio if not provided
        if self.initial_portfolio_quote is None:
            if 'total_amount_quote' in self.daily_report.columns:
                first_val = self.daily_report['total_amount_quote'].iloc[0]
                if pd.notna(first_val):
                    self.initial_portfolio_quote = float(first_val)
                    print(f"  ✓ Auto-detected initial portfolio: {self.initial_portfolio_quote:,.2f} {self.quote_asset}")

            if self.initial_portfolio_quote is None:
                self.initial_portfolio_quote = 150000  # Fallback
                print(f"  ⚠ Using default initial portfolio: {self.initial_portfolio_quote:,.2f} {self.quote_asset}")

        return self.daily_report

    def create_visualizations(self, show_plots: bool = True) -> None:
        """
        Create all visualizations.

        Args:
            show_plots: Whether to display plots (default: True)

        TODO:
        - Add option to save plots to files
        - Add custom plot selection
        - Support for different output formats
        """
        if self.daily_report is None:
            raise ValueError("Performance not calculated. Call calculate_performance() first.")

        # Prepare data
        df = self.daily_report.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Print last 7 days summary
        self.visualizer.print_last_7_days_summary(df)

        # Print executive header
        self.visualizer.print_executive_header(df, self.initial_portfolio_quote)

        # Create executive dashboard
        if show_plots:
            fig = self.visualizer.create_executive_dashboard(df, self.initial_portfolio_quote)
            fig.show()

    def get_summary_metrics(self) -> dict:
        """
        Get summary metrics for the entire period.

        Returns:
            Dictionary with summary metrics

        TODO:
        - Add more advanced metrics (Sharpe, Sortino, Calmar)
        - Add comparison with benchmarks
        - Add risk metrics (VaR, CVaR)
        """
        if self.daily_report is None:
            raise ValueError("Performance not calculated. Call calculate_performance() first.")

        df = self.daily_report
        total_volume_base = df['bot_volume_base'].sum()
        total_volume_quote = df['bot_volume_quote'].sum()
        total_rebate = total_volume_quote * self.rebate_pct
        total_pnl_bot = df['bot_realized_pnl'].sum()
        total_pnl_financial = total_pnl_bot + total_rebate
        num_days = len(df['date'].unique())
        total_trades = df['bot_trades_count'].sum()

        # Win metrics
        daily_pnl = df.groupby('date')['bot_realized_pnl'].first()
        win_days = (daily_pnl > 0).sum()

        return {
            'total_volume_base': total_volume_base,
            'total_volume_quote': total_volume_quote,
            'total_rebate': total_rebate,
            'total_pnl_bot': total_pnl_bot,
            'total_pnl_financial': total_pnl_financial,
            'portfolio_return_pct': (total_pnl_financial / self.initial_portfolio_quote * 100),
            'num_days': num_days,
            'total_trades': total_trades,
            'avg_daily_volume': total_volume_base / num_days,
            'avg_pnl_per_day': total_pnl_financial / num_days,
            'avg_pnl_per_trade': total_pnl_financial / total_trades if total_trades > 0 else 0,
            'win_rate': (win_days / len(daily_pnl) * 100) if len(daily_pnl) > 0 else 0,
            'best_day_pnl': daily_pnl.max(),
            'worst_day_pnl': daily_pnl.min(),
        }

    def export_to_excel(self, use_standard_naming: bool = True) -> str:
        """
        Export all data to Excel with multiple sheets.

        Args:
            use_standard_naming: Use standard file naming convention

        Returns:
            Path to exported Excel file

        TODO:
        - Add formatted Excel with colors and charts
        - Add summary sheet
        - Add pivot tables
        """
        if self.daily_report is None or self.enriched_trades is None:
            raise ValueError("Analysis not complete. Run run_full_analysis() first.")

        # Determine output path
        if use_standard_naming and self.controller_id:
            start_date, end_date = self.file_manager.extract_dates_from_dataframe(
                self.daily_report, 'date'
            )
            output_path = self.file_manager.get_full_export_path(
                controller_id=self.controller_id,
                start_date=start_date,
                end_date=end_date
            )
        else:
            output_path = self.file_manager.get_full_export_path(use_latest=True)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            self.daily_report.to_excel(writer, sheet_name='Daily Report', index=False)
            self.enriched_trades.to_excel(writer, sheet_name='Enriched Trades', index=False)

            # Add summary sheet
            summary_df = pd.DataFrame([self.get_summary_metrics()])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

        print(f"Exported to: {output_path}")
        return str(output_path)

    def export_to_html(
        self,
        use_standard_naming: bool = True,
        company_name: str = "Trading Performance Report"
    ) -> str:
        """
        Export analysis to professional HTML report for clients.

        Args:
            use_standard_naming: Use standard file naming convention
            company_name: Company/report name for HTML header

        Returns:
            Path to exported HTML file
        """
        if self.daily_report is None:
            raise ValueError("Analysis not complete. Run run_full_analysis() first.")

        # Determine output path
        if use_standard_naming and self.controller_id:
            start_date, end_date = self.file_manager.extract_dates_from_dataframe(
                self.daily_report, 'date'
            )
            output_path = self.file_manager.get_html_report_path(
                controller_id=self.controller_id,
                start_date=start_date,
                end_date=end_date
            )
        else:
            output_path = self.file_manager.get_html_report_path(use_latest=True)

        # Generate HTML
        self.html_generator.generate_report(
            daily_report=self.daily_report,
            initial_portfolio_quote=self.initial_portfolio_quote,
            output_path=str(output_path),
            company_name=company_name
        )

        print(f"HTML report generated: {output_path}")
        return str(output_path)


# Convenience function for quick analysis
async def analyze_trading_performance(
    db_name: str,
    root_path: str,
    server_name: str = "brigado_server",
    rebate_pct: float = 0.015 / 100,
    initial_portfolio_quote: Optional[float] = None,
    use_standard_naming: bool = True
) -> TradingPerformanceAnalyzer:
    """
    Convenience function to run complete analysis with minimal setup.

    Args:
        db_name: SQLite database filename
        root_path: Root path for the project
        server_name: Server name for database location
        rebate_pct: Exchange rebate percentage
        initial_portfolio_quote: Initial portfolio in quote asset
        use_standard_naming: Use standard file naming with dates and controller_id

    Returns:
        Analyzer instance with all data loaded

    Example:
        analyzer = await analyze_trading_performance(
            db_name="pmm-mister-20-1-20251223-1025-20251223-102533.sqlite",
            root_path="/Users/tomasgaudino/PycharmProjects/quants-lab"
        )

        # Files will be auto-named like:
        # data/reports/daily_report_pmm-mister-brigado-binance-18-1_20251223_20260128.csv
        # data/enriched/enriched_trades_pmm-mister-brigado-binance-18-1_20251223_20260128.csv
    """
    analyzer = TradingPerformanceAnalyzer(
        db_name=db_name,
        root_path=root_path,
        server_name=server_name,
        rebate_pct=rebate_pct,
        initial_portfolio_quote=initial_portfolio_quote
    )

    await analyzer.run_full_analysis(use_standard_naming=use_standard_naming)

    return analyzer
