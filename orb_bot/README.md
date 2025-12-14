# Opening Range Breakout (ORB) Bot

## 📋 Overview
The **ORB Bot** implements a sophisticated multi-stage Opening Range Breakout strategy based on the LuxyBig Dynamic ORB indicator from TradingView.

## 🎯 Strategy Summary

### What is ORB?
Opening Range Breakout (ORB) is a trading strategy that identifies breakouts from the price range established during the opening minutes of a trading session.

### Key Features
✅ **Multi-Stage Detection**: Tracks 4 ORB levels simultaneously
- **ORB5**: First 5 minutes
- **ORB15**: First 15 minutes  
- **ORB30**: First 30 minutes
- **ORB60**: First 60 minutes

✅ **Breakout Detection**: Identifies when price breaks above/below ORB range
✅ **Retest Tracking**: Detects when price returns to test ORB levels after breakout
✅ **Cycle Counting**: Tracks multiple breakout-retest cycles (up to 6 cycles)
✅ **Volume Filtering**: Optional volume confirmation for breakouts
✅ **Position Sizing**: Integrated TP/SL calculation with R/R ratios

## 📁 Files

```
orb_bot/
├── orb_bot.py              # Main bot logic
├── orb_watchlist.json      # Symbols to monitor
├── orb_state.json          # Session state tracking
└── logs/
    ├── orb_bot.log         # Main bot logs
    └── orb_stats.json      # Performance statistics
```

## ⚙️ Configuration

### Watchlist Format (`orb_watchlist.json`)
```json
{
  "symbol": "BTC/USDT",
  "period": "1m",
  "cooldown_minutes": 30,
  "orb_stages": [5, 15, 30, 60]
}
```

### Bot Configuration Options
- `breakout_buffer_pct`: Extra distance required for breakout (default: 0.2%)
- `retest_buffer_pct`: Tolerance for retest detection (default: 0.2%)
- `min_retest_distance_pct`: Min distance before retest valid (default: 0.5%)
- `min_bars_outside`: Bars required outside ORB for committed breakout (default: 2)
- `max_cycles`: Maximum breakout-retest cycles to track (default: 6)
- `enable_volume_filter`: Require volume confirmation (default: false)
- `volume_multiplier`: Volume threshold multiplier (default: 1.5x)

## 🚀 Usage

### Start the Bot
```bash
# Single scan
./start_orb_bot.sh

# Or directly
cd orb_bot
python3 orb_bot.py --loop

# Test mode (no notifications)
python3 orb_bot.py --loop --test
```

### Stop the Bot
```bash
pkill -f "orb_bot.py"
```

### Monitor Logs
```bash
tail -f orb_bot/logs/orb_bot.log
```

## 📊 Signal Types

### 1. **Breakout Up** 🔼
- Price breaks above ORB high with buffer
- Must stay outside range for minimum bars
- Optional volume confirmation

Example:
```
🔼 ORB15 BREAKOUT UP
Symbol: BTC/USDT
Price: $45,250.00
ORB Range: $44,800.00 - $45,000.00
Range: 0.45%
Cycle: #1

📊 Trade Setup:
Entry: $45,250.00
TP: $45,650.00
SL: $44,900.00
R/R: 2.5
```

### 2. **Breakout Down** 🔽
- Price breaks below ORB low with buffer
- Must stay outside range for minimum bars
- Optional volume confirmation

### 3. **Retest Up** 🔄
- Price breaks up, then returns to ORB high
- Signals potential re-entry opportunity
- Tracks cycle number

### 4. **Retest Down** 🔄
- Price breaks down, then returns to ORB low
- Signals potential re-entry opportunity
- Tracks cycle number

## 🎨 Strategy Logic

### Session Detection
```
Market Hours: 9:30 AM - 4:00 PM EST
- Bot tracks opening range from 9:30 AM
- Collects 1-minute candles
- Builds ORB levels progressively
```

### Breakout Confirmation
```
1. Price closes above/below ORB level + buffer
2. Stays outside for minimum bars (default: 2)
3. Volume check (if enabled): Current volume > MA × multiplier
4. Signal triggered
```

### Retest Detection
```
1. Breakout occurs and price moves away
2. Price travels minimum distance (default: 0.5%)
3. Price returns to ORB level ± buffer
4. Retest signal triggered
5. Cycle counter increments
```

## 📈 Trading Workflow

### Morning Setup (9:30 AM)
1. Bot initializes new session
2. Starts collecting 1-minute candles
3. Builds ORB5 after 5 minutes
4. Builds ORB15 after 15 minutes
5. Continues through ORB30 and ORB60

### Breakout Detection (10:00 AM+)
1. Price breaks above ORB15 high
2. Confirms with 2 bars outside
3. Checks volume (if enabled)
4. Sends breakout signal with TP/SL
5. Tracks position

### Retest Management
1. Price pulls back to ORB15 high
2. Detects retest opportunity
3. Sends retest signal
4. Increments cycle counter
5. Watches for new breakout

## 🔧 Customization

### Aggressive Settings
```python
config = {
    'breakout_buffer_pct': 0.1,       # Tighter breakout
    'min_bars_outside': 1,            # Faster signals
    'max_cycles': 10,                 # More signals
    'enable_volume_filter': False,    # No filter
}
```

### Conservative Settings
```python
config = {
    'breakout_buffer_pct': 0.5,       # Wider breakout
    'min_bars_outside': 3,            # More confirmation
    'max_cycles': 3,                  # Fewer signals
    'enable_volume_filter': True,     # Volume required
    'volume_multiplier': 2.0,         # Strong volume
}
```

## 📚 Best Practices

### Symbol Selection
✅ **Good**: Liquid assets with tight spreads
- BTC/USDT, ETH/USDT, SPY, QQQ
✅ **Good**: High volume stocks (>1M daily)
❌ **Avoid**: Low volume altcoins
❌ **Avoid**: Wide spread instruments

### Timeframe Recommendations
- **Day Trading**: ORB5, ORB15
- **Swing Trading**: ORB30, ORB60
- **Best Chart**: 5-minute for monitoring

### Risk Management
- ✅ Use provided TP/SL levels
- ✅ Risk 0.5-1% per trade
- ✅ Monitor R/R ratio (aim for >2.0)
- ✅ Respect cycle limits
- ❌ Don't chase after 3+ cycles

## 📊 Performance Metrics

The bot tracks:
- Total signals generated
- Breakouts vs retests
- Cycle distribution
- Symbol performance
- Time-based patterns

View stats:
```bash
cat orb_bot/logs/orb_stats.json
```

## 🐛 Troubleshooting

### No Signals
- Check if market is open (9:30-4:00 EST)
- Verify watchlist symbols are valid
- Check if ORB stages are complete
- Review breakout buffer settings

### Too Many Signals
- Increase `breakout_buffer_pct`
- Increase `min_bars_outside`
- Enable volume filter
- Reduce `max_cycles`

### Session Not Starting
- Check system time/timezone
- Verify market hours detection
- Review logs for errors

## 🔗 Integration

### With Other Bots
The ORB bot works alongside:
- **Consensus Bot**: Aggregates ORB signals
- **Volume Bot**: Confirms volume spikes
- **Liquidation Bot**: Aligns with liquidation levels

### With External Tools
- Telegram notifications via `notifier.py`
- Stats tracking via `signal_stats.py`
- TP/SL calculation via `tp_sl_calculator.py`

## 📝 Notes

⚠️ **Market Hours**: Bot designed for stock market hours (9:30-4:00 EST)
⚠️ **Crypto Adaptation**: Can run 24/7 but ORB concept strongest at session open
⚠️ **Backtesting**: Review historical ORB patterns before live trading
⚠️ **Paper Trade First**: Test with small positions before going live

## 🎓 Learn More

**ORB Strategy Resources**:
- Mark Fisher's "Logical Trader"
- Opening Range Breakout studies
- Institutional trading patterns

**Related Concepts**:
- Gap and Go strategy
- Market structure breakouts
- Support/resistance levels
- Volume profile analysis

---

**Created**: December 13, 2025  
**Version**: 1.0  
**Based On**: LuxyBig Dynamic ORB v5 (TradingView)  
**Author**: Automated Bot Framework
