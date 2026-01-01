# Bot Performance Analysis Tool

## Overview

This comprehensive analysis tool provides detailed performance metrics and historical trade analysis for all trading bots in your portfolio. It generates professional reports with statistical analysis, risk metrics, and actionable insights.

## Features

### 📊 Comprehensive Metrics
- **Win Rate Analysis**: Overall, TP1, TP2, TP3 hit rates
- **P&L Statistics**: Total, average, best/worst trades
- **Risk Metrics**: Profit factor, Sharpe ratio, maximum drawdown
- **Trade Duration**: Average and median holding times
- **Symbol Performance**: Per-symbol breakdown with detailed stats

### 📈 Advanced Analytics
- **Time-Based Analysis**: Daily, weekly, monthly P&L tracking
- **Portfolio Summary**: Aggregate statistics across all bots
- **Bot Rankings**: Performance comparison and leaderboards
- **Symbol Analysis**: Best/worst performing symbols per bot

### 📄 Export Formats
- **HTML Report**: Beautiful, interactive web report
- **JSON Data**: Complete structured data for further analysis
- **CSV Summary**: Spreadsheet-compatible summary table
- **Console Output**: Formatted terminal output with detailed breakdown

## Installation

No additional dependencies required! The script uses Python standard library only.

```bash
# Make the script executable
chmod +x analyze_bot_performance.py
```

## Usage

### Basic Usage

```bash
# Full analysis of all bots (console output)
python3 analyze_bot_performance.py

# Summary only (no individual bot details)
python3 analyze_bot_performance.py --summary

# Detailed analysis with trade lists
python3 analyze_bot_performance.py --detailed
```

### Analyze Specific Bot

```bash
# Analyze a specific bot
python3 analyze_bot_performance.py --bot orb

# Detailed analysis for one bot
python3 analyze_bot_performance.py --bot harmonic --detailed

# Partial name matching works too
python3 analyze_bot_performance.py --bot fib
```

### Export Reports

```bash
# Generate HTML report
python3 analyze_bot_performance.py --export html

# Generate JSON data file
python3 analyze_bot_performance.py --export json

# Generate CSV summary
python3 analyze_bot_performance.py --export csv

# Custom output filename
python3 analyze_bot_performance.py --export html --output my_report.html
```

### Combined Options

```bash
# Detailed analysis with HTML export
python3 analyze_bot_performance.py --detailed --export html

# Specific bot with JSON export
python3 analyze_bot_performance.py --bot orb --export json --output orb_analysis.json
```

## Available Bots

The tool analyzes the following trading bots:

1. **DIY Bot** - Multi-indicator confluence strategy (30+ indicators)
2. **Fibonacci Swing Bot** - Fibonacci retracement levels with swing confirmation
3. **Funding Rate Bot** - Funding rate arbitrage strategy
4. **Harmonic Bot** - Harmonic pattern detection (Gartley, Butterfly, Bat, Crab)
5. **Harmonic Pattern Bot** - Advanced harmonic pattern recognition
6. **Liquidation Bot** - Liquidation cluster analysis
7. **MOST Bot** - Moving Stop Loss (MOST) indicator with EMA
8. **MTF Bot** - Multi-timeframe analysis strategy
9. **ORB Bot** - Opening Range Breakout strategy
10. **PSAR Bot** - Parabolic SAR with trailing stop
11. **STRAT Bot** - Rob Smith's STRAT strategy
12. **Volume Bot** - Volume profile and volume node analysis
13. **Volume Profile Bot** - Volume profile POC and value area

## Output Metrics Explained

### Performance Metrics

- **Total Trades**: All signals generated (open + closed)
- **Open Trades**: Currently active signals
- **Closed Trades**: Completed trades with results
- **Win Rate**: Percentage of TP hits vs total closed trades
- **TP1/TP2/TP3 Hit Rate**: Individual take profit level hit rates

### P&L Metrics

- **Total P&L**: Cumulative profit/loss percentage
- **Average P&L**: Mean P&L per trade
- **Average Win**: Mean profit on winning trades
- **Average Loss**: Mean loss on losing trades
- **Best/Worst Trade**: Highest profit and largest loss

### Risk Metrics

- **Profit Factor**: Gross profit / Gross loss (>1 is profitable)
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Max Drawdown**: Largest peak-to-trough decline
- **Max DD Duration**: Number of trades in drawdown period

### Trade Duration

- **Average Duration**: Mean time from entry to exit
- **Median Duration**: Middle value of all trade durations

### Symbol Analysis

- **Symbols Traded**: Number of unique symbols
- **Most Traded**: Symbol with most trades
- **Best Symbol**: Symbol with highest total P&L
- **Worst Symbol**: Symbol with lowest total P&L

## Report Examples

### Console Output Example

```
================================================================================
  ORB BOT
  Opening Range Breakout strategy
================================================================================

  OVERVIEW
  ----------------------------------------------------------------------------
  Total Trades:                121
  Open Trades:                  25
  Closed Trades:                96

  RESULTS BREAKDOWN
  ----------------------------------------------------------------------------
  Wins:                         65 (  67.7%)
  Losses:                       31
  TP1 Hits:                     62 (  64.6%)
  TP2 Hits:                      3 (   3.1%)
  SL Hits:                      31

  P&L PERFORMANCE
  ----------------------------------------------------------------------------
  Total P&L:                 +1.77%
  Average P&L:               +0.02%
  Average Win:               +1.75%
  Average Loss:              -2.99%
  Best Trade:               +10.24%
  Worst Trade:              -33.08%

  RISK METRICS
  ----------------------------------------------------------------------------
  Profit Factor:              1.02
  Sharpe Ratio:               0.00
  Max Drawdown:              40.21%

  SYMBOL ANALYSIS
  ----------------------------------------------------------------------------
  Symbols Traded:               14
  Most Traded:                 ON/USDT
  Best Symbol:                RLS/USDT ( +13.62%)
  Worst Symbol:             POWER/USDT ( -19.43%)
```

### HTML Report Features

The HTML report includes:
- Professional styling with color-coded metrics
- Responsive design for mobile viewing
- Interactive tables with hover effects
- Portfolio summary dashboard
- Individual bot sections with detailed breakdowns
- Symbol performance tables

### JSON Data Structure

```json
{
  "generated_at": "2026-01-01T04:30:43+00:00",
  "bots": [
    {
      "bot_name": "ORB Bot",
      "description": "Opening Range Breakout strategy",
      "total_trades": 121,
      "closed_trades": 96,
      "win_rate": 67.7,
      "total_pnl": 1.77,
      "profit_factor": 1.02,
      "symbol_stats": [...],
      "recent_trades": [...],
      ...
    }
  ]
}
```

### CSV Format

The CSV export includes one row per bot with key metrics:
- Bot name and description
- Trade counts (total, open, closed)
- Win/loss statistics
- P&L metrics
- Risk metrics
- Symbol information

## Performance Summary (Current Data)

Based on the latest analysis:

### 🏆 Top Performers

1. **PSAR Bot**: 100% win rate, +3.06% total P&L (3 trades)
2. **ORB Bot**: 67.7% win rate, +1.77% total P&L (96 trades)
3. **Liquidation Bot**: 45.4% win rate, +0.59% total P&L (97 trades)

### 📊 Portfolio Statistics

- **Active Bots**: 9 out of 11
- **Total Trades**: 1,563 (187 open, 1,376 closed)
- **Overall Win Rate**: 29.6%
- **Total Portfolio P&L**: -535.87%

### 🎯 Best Metrics

- **Best Win Rate**: PSAR Bot (100.0%)
- **Best Profit Factor**: ORB Bot (1.02)
- **Best Sharpe Ratio**: PSAR Bot (1.49)

### ⚠️ Areas for Improvement

Several bots are showing negative P&L:
- Fibonacci Swing Bot: -277.64%
- Volume Bot: -84.76%
- STRAT Bot: -78.15%
- MOST Bot: -74.39%

**Recommendations**:
1. Review and optimize underperforming strategies
2. Consider adjusting TP/SL levels for better risk/reward
3. Analyze symbol selection for problematic bots
4. Focus on successful patterns from ORB and PSAR bots

## Data Sources

The analysis tool reads from:
- `{bot_name}/logs/{bot_name}_stats.json` - Historical trade data
- Stats files contain:
  - `open`: Currently open signals
  - `history`: Closed trades with results
  - Trade metadata (entry, exit, P&L, timestamps, etc.)

## Troubleshooting

### No Data for Bot

If you see "No closed trades found for {Bot Name}":
- The bot hasn't closed any trades yet
- Stats file doesn't exist or is empty
- Bot is newly deployed

### Missing Stats File

If you see "Stats file not found":
- Check that the bot's logs directory exists
- Ensure the bot has been running and generating stats
- Verify the file path in `BOT_CONFIGS` dictionary

### Incorrect Metrics

If metrics seem wrong:
- Verify the stats file format matches expected structure
- Check for data corruption in JSON files
- Review bot's SignalStats implementation

## Advanced Usage

### Filtering by Date

To analyze specific time periods, you can modify the JSON data before analysis or use the exported JSON for custom filtering:

```python
import json
from datetime import datetime

# Load exported data
with open('bot_performance_data.json') as f:
    data = json.load(f)

# Filter trades by date
for bot in data['bots']:
    bot['recent_trades'] = [
        t for t in bot['recent_trades']
        if datetime.fromisoformat(t['closed_at']) > datetime(2025, 12, 1)
    ]
```

### Custom Analysis

The JSON export can be imported into:
- Python for custom analysis
- Excel/Google Sheets for pivot tables
- BI tools like Tableau or Power BI
- Custom dashboards and monitoring systems

## Automation

### Scheduled Reports

Add to crontab for daily reports:

```bash
# Daily report at 9 AM
0 9 * * * cd /path/to/bots && python3 analyze_bot_performance.py --export html --output daily_report_$(date +\%Y\%m\%d).html

# Weekly summary every Monday
0 9 * * 1 cd /path/to/bots && python3 analyze_bot_performance.py --summary > weekly_summary_$(date +\%Y\%m\%d).txt
```

### Email Reports

Combine with email tools:

```bash
#!/bin/bash
cd /path/to/bots
python3 analyze_bot_performance.py --export html --output report.html
echo "See attached performance report" | mail -s "Daily Bot Performance" -a report.html your@email.com
```

## Contributing

To add a new bot to the analysis:

1. Add bot configuration to `BOT_CONFIGS` dictionary:
```python
"new_bot": {
    "name": "New Bot",
    "stats_file": BASE_DIR / "new_bot" / "logs" / "new_bot_stats.json",
    "description": "Bot description",
}
```

2. Ensure the bot uses the `SignalStats` class for tracking trades

3. Stats file should follow the standard format:
```json
{
  "open": { ... },
  "history": [
    {
      "id": "signal_id",
      "symbol": "BTC/USDT",
      "direction": "LONG",
      "entry": 50000.0,
      "exit": 51000.0,
      "result": "TP1",
      "pnl_pct": 2.0,
      "created_at": "2025-01-01T00:00:00+00:00",
      "closed_at": "2025-01-01T01:00:00+00:00"
    }
  ]
}
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the bot's individual stats file
3. Verify the bot is using SignalStats correctly
4. Check bot logs for errors

## Version History

- **v1.0** (2026-01-01): Initial release
  - Full bot performance analysis
  - HTML, JSON, CSV export formats
  - Risk metrics and symbol analysis
  - Portfolio summary and rankings

## License

This tool is part of the Azure Bots trading system.
