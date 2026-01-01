# Backtest Script for Fixed Bots

## Overview

The `backtest_fixed_bots.py` script allows you to test how the trading bots would perform with all the critical fixes applied. It simulates trading with:

- ✅ Maximum stop loss limits (2.5%)
- ✅ Emergency stop loss (5%)
- ✅ Standardized R:R requirements (1.2:1 TP1, 2.0:1 TP2)
- ✅ Fixed TP1/TP2 ordering
- ✅ ATR-adjusted stops
- ✅ Trailing stops
- ✅ Drawdown protection (25%)
- ✅ Breakeven stops (1.5% trigger)
- ✅ Maximum trade duration (4 hours)

## Quick Start

### Basic Usage

```bash
# Test all bots with default settings (30 days, 5m timeframe)
python3 backtest_fixed_bots.py

# Test specific bot
python3 backtest_fixed_bots.py --bot fib

# Test with custom timeframe and lookback
python3 backtest_fixed_bots.py --days 60 --timeframe 15m

# Test specific symbols
python3 backtest_fixed_bots.py --symbols POWER/USDT BLUAI/USDT IRYS/USDT

# Verbose output
python3 backtest_fixed_bots.py --verbose
```

### Command Line Options

```
--days N              Lookback period in days (default: 30)
--timeframe STR       Timeframe (default: 5m, options: 1m, 5m, 15m, 1h, 4h)
--bot STR             Test specific bot (fib, volume, strat, most, orb, psar)
--symbols SYM ...      Test specific symbols
--verbose             Show detailed output
--output-dir DIR      Output directory for results
--no-compare          Skip before/after comparison
```

## Example Output

```
======================================================================
COMPREHENSIVE BACKTEST - FIXED BOTS
======================================================================
Date: 2026-01-01 12:00:00 UTC
Config: 30 days, 5m timeframe
Symbols: 10
======================================================================

======================================================================
BACKTESTING: FIB BOT
======================================================================
Symbols: 10
Timeframe: 5m
Lookback: 30 days

FIXES APPLIED:
  ✓ Max stop loss: 2.5%
  ✓ Emergency stop: 5.0%
  ✓ Min R:R: 1.2:1 (TP1), 2.0:1 (TP2)
  ✓ Max trade duration: 4.0h
  ✓ Breakeven trigger: 1.5%
  ✓ Drawdown protection: 25.0%
======================================================================

  Fetching POWER/USDT data... ✓ 4320 candles
  Processing POWER/USDT...
    ...

======================================================================
RESULTS: FIB BOT
======================================================================
Total Trades: 45
Win Rate: 42.22%
Total P&L: -12.50%
Avg P&L: -0.28%
Profit Factor: 1.15
Max Drawdown: 8.50%
Sharpe Ratio: 0.85

Exits:
  TP1: 12 | TP2: 7 | TP3: 0
  SL: 18 | Emergency: 2 | Expired: 6
======================================================================
```

## How It Works

### 1. Data Fetching
- Fetches historical OHLCV data from Binance Futures
- Supports multiple timeframes (1m, 5m, 15m, 1h, 4h)
- Handles rate limiting and errors gracefully

### 2. Signal Generation
- Currently uses simplified trend-following signals (EMA crossover)
- Can be extended to use actual bot signal generation logic
- All signals are validated with the fixed risk management rules

### 3. Trade Simulation
- Simulates trade execution with all fixes:
  - Maximum stop loss enforcement (2.5%)
  - Emergency stop triggers (5%)
  - Breakeven stop movement (1.5% profit)
  - Maximum trade duration (4 hours)
  - Correct TP1/TP2 ordering
  - Trailing stops (where applicable)

### 4. Risk Management
- All trades validated before entry:
  - Stop loss ≤ 2.5%
  - R:R ratio ≥ 1.2:1 (TP1), 2.0:1 (TP2)
  - Correct TP ordering
- Portfolio-level protection:
  - Drawdown monitoring
  - Emergency stops
  - Breakeven protection

### 5. Results Calculation
- Comprehensive metrics:
  - Win rate
  - Total P&L
  - Profit factor
  - Sharpe ratio
  - Max drawdown
  - TP/SL hit rates

## Extending the Script

### Adding Real Bot Signal Generation

To use actual bot signal generation, modify the `generate_simple_signals` method:

```python
def generate_simple_signals(self, ohlcv: List[List], bot_name: str) -> List[Trade]:
    """Generate signals using actual bot logic."""
    signals = []
    
    # Import bot class
    if bot_name == "fib":
        from fib_swing_bot.fib_swing_bot import FibSwingBot, FibSignalEvaluator
        evaluator = FibSignalEvaluator()
        # Generate signals using evaluator
        # ...
    
    elif bot_name == "volume":
        from volume_bot.volume_vn_bot import VolumeAnalyzer
        analyzer = VolumeAnalyzer()
        # Generate signals using analyzer
        # ...
    
    return signals
```

### Adding More Bots

1. Add bot name to `config.test_bots`
2. Implement signal generation for that bot
3. Test with `--bot <bot_name>`

## Output Files

Results are exported to `backtest_results/` directory:

- `backtest_results_YYYYMMDD_HHMMSS.json` - Complete results in JSON format
- Can be extended to export CSV and HTML reports

## Limitations

### Current Limitations

1. **Simplified Signals**: Currently uses basic EMA crossover signals. To get accurate results, integrate actual bot signal generation logic.

2. **No Slippage Modeling**: Assumes perfect execution. Real trading has slippage and spread costs.

3. **No Partial Fills**: Assumes full position fills at exact prices.

4. **No Market Impact**: Doesn't account for order book depth or market impact.

### Future Improvements

- [ ] Integrate actual bot signal generation
- [ ] Add slippage and spread modeling
- [ ] Support partial fills
- [ ] Add market impact simulation
- [ ] Generate HTML reports
- [ ] Before/after comparison charts
- [ ] Monte Carlo simulation
- [ ] Walk-forward optimization

## Interpreting Results

### Good Results
- ✅ Win rate > 40%
- ✅ Profit factor > 1.2
- ✅ Max drawdown < 20%
- ✅ Sharpe ratio > 1.0
- ✅ No emergency stops triggered
- ✅ TP1 hits > SL hits

### Warning Signs
- ⚠️ Win rate < 30%
- ⚠️ Profit factor < 1.0
- ⚠️ Max drawdown > 30%
- ⚠️ Many emergency stops
- ⚠️ SL hits > TP hits

### Red Flags
- 🚨 Win rate < 25%
- 🚨 Profit factor < 0.8
- 🚨 Max drawdown > 50%
- 🚨 Emergency stops > 10% of trades
- 🚨 Total P&L < -50%

## Comparison with Live Trading

### Differences

1. **Signal Quality**: Backtest uses simplified signals. Live bots may have better/worse signal quality.

2. **Execution**: Backtest assumes perfect execution. Live trading has:
   - Slippage
   - Spread costs
   - Order rejection
   - Network delays

3. **Market Conditions**: Backtest uses historical data. Live trading faces:
   - Current market conditions
   - News events
   - Liquidity changes
   - Volatility shifts

### Use Backtest Results As

- ✅ **Guideline**: Expected performance range
- ✅ **Validation**: Confirms fixes work correctly
- ✅ **Comparison**: Before vs after fixes
- ❌ **Not Guarantee**: Actual live performance

## Troubleshooting

### Common Issues

**Issue**: "Error fetching data"
- **Solution**: Check internet connection and Binance API access

**Issue**: "Insufficient data"
- **Solution**: Increase `--days` or check symbol availability

**Issue**: "No signals generated"
- **Solution**: Check symbol has sufficient data and volatility

**Issue**: "Import errors"
- **Solution**: Ensure all dependencies installed: `pip install -r requirements.txt`

## Next Steps

1. **Run Backtest**: Test with default settings
2. **Review Results**: Check if fixes improve performance
3. **Extend Script**: Add actual bot signal generation
4. **Compare**: Compare backtest vs live results
5. **Optimize**: Fine-tune parameters based on results

## Support

For issues or questions:
1. Check this README
2. Review `FIXES_APPLIED.md` for fix details
3. Check bot logs for errors
4. Verify configuration files

---

**Version**: 1.0  
**Last Updated**: January 1, 2026  
**Status**: ✅ Ready for Testing
