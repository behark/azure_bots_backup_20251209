# Trading Bots Feature Comparison

## Signal Tracking Features

### ✅ Liquidation Bot - **HAS TRACKING**

**Features:**
- ✅ Monitors open signals continuously
- ✅ Tracks TP1, TP2, and SL levels
- ✅ Sends Telegram alerts when targets hit
- ✅ Integrated with `SignalStats` for performance tracking
- ✅ Automatically closes and removes completed signals
- ✅ Handles ticker fetch errors gracefully

**How it works:**
```
_monitor_open_signals() runs every cycle:
1. Fetches current price for each open signal
2. Checks if TP1, TP2, or SL hit
3. Sends alert: "🎯 BTC/USDT BULLISH TP1 hit!"
4. Records stats and removes signal
```

**Example Output:**
```
🎯 BTC/USDT BULLISH TP1 hit!
Entry 45000.00 | Last 45500.00
TP1 45500.00 | TP2 46000.00 | SL 44500.00
```

---

### ✅ Funding Bot - **HAS TRACKING**

**Features:**
- ✅ Monitors open signals continuously
- ✅ Tracks TP1, TP2, and SL levels
- ✅ Sends Telegram alerts when targets hit
- ✅ Integrated with `SignalStats` for performance tracking
- ✅ Automatically closes and removes completed signals
- ✅ Handles ticker fetch errors gracefully

**How it works:**
```
_monitor_open_signals() runs every cycle:
1. Fetches current price for each open signal
2. Checks if TP1, TP2, or SL hit
3. Sends alert with entry/exit details
4. Records stats and removes signal
```

**Identical to Liquidation Bot tracking system**

---

### ✅ Volume Bot - **HAS TRACKING**

**Features:**
- ✅ Monitors open signals continuously via `SignalTracker`
- ✅ Tracks TP1 (primary), TP2 (secondary), and SL
- ✅ Sends Telegram alerts when targets hit
- ✅ Automatically closes and removes completed signals
- ✅ Handles ticker fetch errors gracefully

**How it works:**
```
tracker.check_open_signals() runs every cycle:
1. Fetches current price for each open signal
2. Checks if TP1, TP2, or SL hit
3. Sends alert: "🎯 ETH/USDT 15m LONG TP1 hit!"
4. Removes signal from tracker
```

**Example Output:**
```
🎯 ICP/USDT 15m LONG SL hit!
Entry 0.095000 | Last 0.093500
SL 0.093500 | TP1 0.096000 | TP2 0.097000
```

---

## Feature Comparison Table

| Feature | Liquidation Bot | Funding Bot | Volume Bot |
|---------|----------------|-------------|------------|
| **Signal Tracking** | ✅ Yes | ✅ Yes | ✅ Yes |
| **TP1 Monitoring** | ✅ Yes | ✅ Yes | ✅ Yes |
| **TP2 Monitoring** | ✅ Yes | ✅ Yes | ✅ Yes |
| **SL Monitoring** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Auto Close Signals** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Telegram Alerts** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Stats Tracking** | ✅ Yes (SignalStats) | ✅ Yes (SignalStats) | ❌ No |
| **Performance Reports** | ✅ Yes | ✅ Yes | ❌ No |
| **Error Handling** | ✅ Yes | ✅ Yes | ✅ Yes |

---

## Key Differences

### Liquidation & Funding Bots
- **Have `SignalStats` integration** - tracks win/loss ratios, profit/loss
- **Generate performance reports** - summary messages with stats
- Directions: BULLISH/BEARISH
- TPs: TP1, TP2
- More sophisticated stats tracking

### Volume Bot
- **Simpler tracking** - just monitors and alerts
- **No performance stats** - doesn't track win/loss ratios
- Directions: LONG/SHORT
- TPs: Primary (TP1), Secondary (TP2)
- More straightforward implementation

---

## All Bots Monitor:

1. **Entry Price** - where signal was triggered
2. **Current Price** - real-time market price
3. **Take Profit 1** - first target
4. **Take Profit 2** - second target
5. **Stop Loss** - risk limit

## What Happens When Target Hits:

### Common Flow (All Bots):
```
1. Bot detects TP1/TP2/SL hit
2. Sends Telegram notification
3. Logs the result
4. Removes signal from tracking
5. Continues monitoring other signals
```

### Liquidation/Funding Bots (Extra):
```
6. Records to SignalStats
7. Generates performance summary
8. Includes win rate, avg profit, etc.
```

---

## Example Tracking in Action

### You'll See These Alerts:

**When TP1 Hits:**
```
🎯 BTC/USDT BULLISH TP1 hit!
Entry 45000.00 | Last 45500.00
TP1 45500.00 | TP2 46000.00 | SL 44500.00
```

**When TP2 Hits:**
```
🎯 ETH/USDT 15m LONG TP2 hit!
Entry 0.095000 | Last 0.097000
SL 0.093500 | TP1 0.096000 | TP2 0.097000
```

**When SL Hits:**
```
🎯 ADA/USDT BEARISH SL hit!
Entry 0.350000 | Last 0.355000
TP1 0.345000 | TP2 0.340000 | SL 0.355000
```

---

## Tracking Frequency

- **Liquidation Bot**: Checks every 5 minutes (300s)
- **Funding Bot**: Checks every 5 minutes (300s)
- **Volume Bot**: Checks every 1 minute (60s)

All bots check **ALL open signals** every cycle!

---

## State Files (Where Signals Stored)

- Liquidation: `liquidation_bot/liquidation_state.json`
- Funding: `funding_bot/funding_state.json`
- Volume: `volume_bot/volume_vn_state.json`

You can view open signals anytime:
```bash
cat liquidation_bot/liquidation_state.json | jq '.open_signals'
cat funding_bot/funding_state.json | jq '.open_signals'
cat volume_bot/volume_vn_state.json | jq '.open_signals'
```

---

## Summary

### ✅ ALL THREE BOTS HAVE SIGNAL TRACKING!

**What They Do:**
1. Generate entry signals with TP/SL levels
2. Store signals in state files
3. Monitor price continuously
4. Alert when TP1, TP2, or SL hit
5. Automatically close completed signals

**Main Difference:**
- **Liquidation & Funding** = Full tracking + performance analytics
- **Volume** = Full tracking, no performance analytics

All bots actively monitor your positions and alert you when targets are hit! 🎯
