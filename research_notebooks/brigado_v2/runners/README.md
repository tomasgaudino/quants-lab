

# Brigado v2 Report Runners

Python scripts for generating performance analysis reports. These replace the Jupyter notebooks with cleaner, executable scripts that have beautiful logging.

## 📁 Structure

```
runners/
├── README.md                  # This file
├── run_all.py                 # Master runner - executes all reports
├── 00_fetch_live_databases.py # Fetch databases from remote server
├── 01_consolidate_data.py     # Data consolidation + HTML report
└── 02_market_analysis.py      # Market analysis + HTML report
```

## 🚀 Quick Start

### Run All Reports

```bash
cd research_notebooks/brigado_v2/runners
python run_all.py
```

This will:
1. ✅ Fetch databases from remote server
2. ✅ Check database integrity & recover if needed
3. ✅ Verify trade-to-controller mapping
4. ✅ Consolidate data from all databases
5. ✅ Generate consolidation report
6. ✅ Fetch market data from Binance
7. ✅ Generate market analysis report
8. ✅ Update index.html

### Run Individual Reports

```bash
# Fetch databases from remote server
python 00_fetch_live_databases.py

# Data consolidation only
python 01_consolidate_data.py

# Market analysis only
python 02_market_analysis.py
```

## 📊 Output

All reports are generated in:
```
data/data_sources/
├── index.html                      # Main navigation page
├── consolidation_report.html       # Consolidation report
├── market_analysis_report.html     # Market analysis report
├── consolidated_trades.parquet     # Data files
├── consolidated_orders.parquet
├── consolidated_executors.parquet
└── consolidated_controllers.parquet
```

## 🎨 Features

### Beautiful Logging
- ✅ Styled headers and sections
- ⏳ Progress indicators
- ✅ Success/error messages
- 📊 Metrics and summaries
- ⏱️  Execution time tracking

### Clean Output
```
================================================================================
  🔄 DATA CONSOLIDATION RUNNER
================================================================================

Started: 2026-03-04 21:45:30

⏳ Initializing components...
✅ Components initialized

⏳ Discovering databases...
✅ Found 2 database(s)
  → pmm-mister-all-20260303-011107 - 43.87 MB
  → pmm-btcbrl-20260303-012028 - 42.45 MB

⏳ Consolidating data from all databases...
✅ Data consolidated successfully
  → Generated files:
      trades: consolidated_trades.parquet (0.34 MB)
      orders: consolidated_orders.parquet (0.56 MB)
      executors: consolidated_executors.parquet (33.60 MB)
      controllers: consolidated_controllers.parquet (0.02 MB)
```

## 🎯 Advantages Over Notebooks

1. **Cleaner Execution**: No cell-by-cell execution, just run the script
2. **Better Logging**: Structured, easy-to-read output
3. **Version Control**: Easier to diff and track changes
4. **Automation**: Can be scheduled with cron/systemd
5. **CI/CD Ready**: Easy to integrate into pipelines
6. **No Jupyter Required**: Run anywhere with Python

## 📝 Requirements

- Python 3.8+
- All dependencies from `requirements.txt`
- Consolidated databases in `data/live_databases/`

## 🔄 Workflow

1. **Fetch databases**: `python 00_fetch_live_databases.py`
2. **Run consolidation**: `python 01_consolidate_data.py`
3. **Run market analysis**: `python 02_market_analysis.py`
4. **Or run all**: `python run_all.py`
5. **View reports**: Open `data/data_sources/index.html`

## 💡 Tips

- Run `run_all.py` for complete report generation
- Individual scripts can be run independently
- All scripts show execution time and summary
- Index page is auto-updated after each run
- Check exit codes for success (0) or failure (non-zero)

## 🎨 Theme

All reports use Binance-style dark theme:
- Dark background (#0b0e11)
- Yellow accents (#f0b90b)
- Clean, professional design
- Responsive layouts

## 🐛 Troubleshooting

**No databases found:**
```bash
# Run database fetch first
jupyter notebook fetch_live_databases.ipynb
```

**Import errors:**
```bash
# Ensure you're in the project root or scripts handle paths correctly
cd /path/to/quants-lab
python research_notebooks/brigado_v2/runners/run_all.py
```

**Market data fetch fails:**
- Check network connection
- Verify Binance API is accessible
- Check trading pair symbols match Binance format

## 📞 Support

Issues? Check the logs in the script output. Each step shows clear success/failure status.
