# Architecture Documentation
**Cryptocurrency Trading Bot System - Azure Bots Backup**
**Last Updated:** 2025-12-14

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Bot Types & Responsibilities](#bot-types--responsibilities)
4. [Data Flow](#data-flow)
5. [State Management](#state-management)
6. [Configuration Management](#configuration-management)
7. [Signal Processing](#signal-processing)
8. [Risk Management](#risk-management)
9. [Monitoring & Health Checks](#monitoring--health-checks)
10. [Deployment Architecture](#deployment-architecture)
11. [Security](#security)
12. [Development Guidelines](#development-guidelines)

---

## System Overview

This is a **distributed cryptocurrency trading bot ecosystem** consisting of 15 specialized bots that analyze the MEXC exchange using different technical analysis strategies. The system generates trading signals based on pattern recognition, technical indicators, market data, and consensus algorithms.

### Key Characteristics
- **Modular Design:** Each bot is independent and specialized
- **Consensus-Based:** Signals are aggregated for higher confidence
- **Risk-Managed:** ATR-based stop-loss and take-profit calculations
- **Production-Ready:** Systemd integration, health monitoring, graceful shutdown
- **Event-Driven:** React to market conditions in real-time
- **Fault-Tolerant:** Rate limiting, backoff mechanisms, state persistence

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MEXC Exchange API                            │
│                    (Market Data, Funding, Liquidations)              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    Rate Limit Handler                                │
│              (Backoff, Retry Logic, API Protection)                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ↓                    ↓                    ↓
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Data Collection │  │ Pattern Recognition│ │ Indicator/Trend  │
│      Bots        │  │       Bots         │ │      Bots        │
├──────────────────┤  ├──────────────────┤ ├──────────────────┤
│ • funding_bot    │  │ • harmonic_bot   │ │ • mtf_bot        │
│ • liquidation_bot│  │ • candlestick_bot│ │ • psar_bot       │
│ • volume_bot     │  │ • fib_reversal   │ │ • diy_bot        │
│                  │  │ • strat_bot      │ │ • most_bot       │
│                  │  │ • fib_swing_bot  │ │                  │
└────────┬─────────┘  └────────┬─────────┘ └────────┬─────────┘
         │                     │                     │
         └──────────┬──────────┴──────────┬──────────┘
                    ↓                     ↓
          ┌──────────────────┐  ┌──────────────────┐
          │  Special Purpose │  │ Volume Profile   │
          │      Bots        │  │      Bot         │
          ├──────────────────┤  └────────┬─────────┘
          │ • orb_bot        │           │
          │ • consensus_bot  │←──────────┘
          └────────┬─────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      Shared Components                               │
├─────────────────────────────────────────────────────────────────────┤
│ • TP/SL Calculator     • Signal Stats         • Trade Config        │
│ • Health Monitor       • Notifier (Telegram)  • File Lock           │
│ • Rate Limit Handler   • Volume Profile       • State Manager       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ↓                ↓                ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Telegram    │  │ State Files  │  │  Log Files   │
│ Notifications│  │   (JSON)     │  │   Analysis   │
└──────────────┘  └──────────────┘  └──────────────┘
```

---

## Bot Types & Responsibilities

### 1. Data Collection Bots (3 bots)

#### funding_bot
- **Purpose:** Monitors funding rates and open interest extremes
- **Signals:** Extreme funding rates indicating overbought/oversold conditions
- **Key Metrics:** Funding rate %, OI change, EMA crossovers
- **Interval:** 300s (5 minutes)
- **Timeframe:** 5m candles

#### liquidation_bot
- **Purpose:** Tracks liquidation clusters and cascade events
- **Signals:** Large liquidation events, bid/ask imbalances
- **Key Metrics:** Liquidation volume, taker buy/sell ratio, order flow
- **Interval:** 300s
- **Timeframe:** 15m candles

#### volume_bot
- **Purpose:** Analyzes volume profiles and unusual activity
- **Signals:** Volume spikes, unusual patterns, accumulation/distribution
- **Key Metrics:** Volume deviation, relative volume, volume trends
- **Interval:** 300s
- **Timeframe:** 15m candles

### 2. Pattern Recognition Bots (5 bots)

#### harmonic_bot
- **Purpose:** Identifies harmonic patterns (Bat, Butterfly, Gartley, Crab, Shark)
- **Signals:** Completed harmonic patterns with Fibonacci levels
- **Key Metrics:** Pattern completion %, Fibonacci ratios
- **Interval:** 300s
- **Timeframe:** 5m candles

#### candlestick_bot
- **Purpose:** Detects candlestick patterns (Hammer, Engulfing, Stars, etc.)
- **Signals:** Bullish/bearish reversal patterns
- **Key Metrics:** Pattern strength, volume confirmation
- **Interval:** 300s
- **Timeframe:** 5m candles

#### fib_reversal_bot
- **Purpose:** Fibonacci reversal patterns with speed modes
- **Signals:** Retracements to key Fib levels (0.382, 0.5, 0.618)
- **Key Metrics:** Fibonacci levels, momentum, trend alignment
- **Interval:** 300s
- **Timeframe:** 15m candles

#### strat_bot
- **Purpose:** Rob Smith's 1-2-3 price action system
- **Signals:** 2-1-2 reversal, 2-2 continuation patterns
- **Key Metrics:** Bar types (inside, outside, directional), SMA50 filter
- **Interval:** 300s
- **Timeframe:** 15m candles

#### fib_swing_bot
- **Purpose:** Fibonacci swing trading on trending moves
- **Signals:** Swing highs/lows with Fib extensions
- **Key Metrics:** Swing points, Fib extensions, trend strength
- **Interval:** 300s
- **Timeframe:** 15m candles

### 3. Indicator/Trend Bots (5 bots)

#### mtf_bot
- **Purpose:** Multi-timeframe confluence analysis
- **Signals:** Aligned signals across multiple timeframes
- **Key Metrics:** RSI, MACD, trend alignment across 5m/15m/1h
- **Interval:** 300s
- **Timeframes:** 5m, 15m, 1h

#### psar_bot
- **Purpose:** Parabolic SAR trend following
- **Signals:** SAR flips indicating trend changes
- **Key Metrics:** PSAR position, acceleration factor, trend strength
- **Interval:** 300s
- **Timeframe:** 5m candles

#### diy_bot
- **Purpose:** 30+ indicators confluence system
- **Signals:** Multiple indicator agreement (RSI, MACD, ADX, Stoch, etc.)
- **Key Metrics:** Indicator confluence score, divergences
- **Interval:** 300s
- **Timeframe:** 5m candles

#### most_bot
- **Purpose:** Moving Stop Loss (MOST) + EMA trend following
- **Signals:** EMA crossovers with trailing stop
- **Key Metrics:** EMA position, MOST level, trailing stop distance
- **Interval:** 300s
- **Timeframe:** 5m candles

### 4. Special Purpose Bots (3 bots)

#### orb_bot
- **Purpose:** Opening Range Breakout strategy
- **Signals:** Breakouts from defined opening ranges
- **Key Metrics:** ORB levels (5min, 15min, 30min), breakout volume
- **Interval:** 60s (1 minute)
- **Timeframe:** 1m candles

#### volume_profile_bot
- **Purpose:** Volume Profile analysis (POC, VAH, VAL)
- **Signals:** Price reactions at volume nodes
- **Key Metrics:** Point of Control, Value Area High/Low, HVN levels
- **Interval:** 300s
- **Timeframe:** 5m candles

#### consensus_bot
- **Purpose:** Aggregates signals from all other bots
- **Signals:** 2-8+ bot agreement levels for high-confidence trades
- **Key Metrics:** Bot consensus count, signal agreement
- **Interval:** 30s
- **No Watchlist:** Monitors other bots' signals

---

## Data Flow

### 1. Market Data Collection
```
MEXC API → Rate Limit Handler → Bot Cycle
   ↓
fetch_ohlcv(symbol, timeframe)
fetch_ticker(symbol)
fetch_funding_rate(symbol)        [funding_bot]
fetch_liquidations(symbol)        [liquidation_bot]
```

### 2. Signal Generation
```
Market Data → Technical Analysis → Signal Detection
   ↓
Bot-Specific Logic
   ├─ Pattern Matching         [harmonic, candlestick, strat]
   ├─ Indicator Calculation    [mtf, psar, diy, most]
   ├─ Volume Analysis          [volume, volume_profile]
   └─ Market Microstructure    [funding, liquidation, orb]
   ↓
Signal Creation (if conditions met)
```

### 3. Risk Calculation
```
Signal → TP/SL Calculator → Trade Levels
   ↓
ATR Calculation (14-period)
   ↓
Stop Loss = Entry ± (ATR × sl_multiplier)
TP1 = Entry ± (ATR × tp1_multiplier)
TP2 = Entry ± (ATR × tp2_multiplier)
TP3 = Entry ± (ATR × tp3_multiplier)
   ↓
Risk/Reward Validation (min 1.3:1)
```

### 4. Signal Notification
```
Trade Levels → Notifier → Telegram
   ↓
Format Message (Emoji, HTML)
   ↓
Send via Telegram Bot API
   ↓
Log to signals_log.json
```

### 5. Signal Tracking
```
Open Signals → Monitor Cycle → Price Checks
   ↓
Check if TP1/TP2/TP3 hit
Check if Stop Loss hit
   ↓
Update Signal Status
   ↓
Save to State File
Log to Stats
```

---

## State Management

### State Files (Per Bot)
**Location:** `{bot_dir}/{bot_name}_state.json`
**Permissions:** 0o600 (owner read/write only)

#### Structure
```json
{
  "last_alert": {
    "SYMBOL": "2025-12-14T10:30:00Z"
  },
  "open_signals": {
    "signal_id_1": {
      "symbol": "BTC/USDT",
      "direction": "LONG",
      "entry": 45000.0,
      "stop_loss": 44500.0,
      "take_profit_1": 45500.0,
      "take_profit_2": 46000.0,
      "timestamp": "2025-12-14T10:30:00Z",
      "status": "OPEN"
    }
  }
}
```

### File Locking
- **Mechanism:** fcntl.flock (POSIX file locks)
- **Implementation:** `file_lock.py` SafeStateManager
- **Protection:** Prevents race conditions in multi-process scenarios
- **Atomic Writes:** Write to temp file → rename for atomicity

### State Lifecycle
1. **Load on startup:** Read from state file or initialize empty
2. **Update on signal:** Add to open_signals, update last_alert
3. **Monitor cycle:** Check TP/SL hits, update status
4. **Save on change:** Atomic write with file locking
5. **Persist on shutdown:** Save final state in signal handler

---

## Configuration Management

### Global Configuration
**File:** `global_config.json`
**Purpose:** Centralized configuration for all 15 bots

#### Structure
```json
{
  "global_risk": {
    "sl_atr_multiplier": 1.5,
    "tp1_atr_multiplier": 2.0,
    "tp2_atr_multiplier": 3.0,
    "tp3_atr_multiplier": 4.5,
    "min_risk_reward": 1.3,
    "use_trailing_stop": false
  },
  "bots": {
    "bot_name": {
      "enabled": true,
      "max_open_signals": 55,
      "interval_seconds": 300,
      "default_cooldown_minutes": 5,
      "risk": { /* bot-specific overrides */ },
      "symbols": [
        { "symbol": "BTC", "period": "5m", "cooldown_minutes": 5 }
      ]
    }
  }
}
```

### Bot-Specific Watchlists
**Location:** `{bot_dir}/{bot_name}_watchlist.json`
**Purpose:** Per-bot symbol configurations (legacy, moving to global_config)

### Environment Variables (.env)
```bash
# Telegram Configuration
TELEGRAM_BOT_TOKEN=              # Main bot token
TELEGRAM_CHAT_ID=                # Target chat ID
TELEGRAM_BOT_TOKEN_{BOT}=        # Bot-specific tokens (optional)

# MEXC API (for orb_bot)
MEXC_API_KEY=
MEXC_API_SECRET=

# Paths (optional)
TELEMETRY_ROOT=~/bots_telemetry
PROJECT_ROOT=~/azure_bots_backup_20251209
```

---

## Signal Processing

### Signal Structure
```python
@dataclass
class TradingSignal:
    symbol: str                  # e.g., "BTC/USDT"
    direction: str               # "LONG" or "SHORT"
    entry: float                 # Entry price
    stop_loss: float             # Stop loss level
    take_profit_1: float         # First TP target
    take_profit_2: float         # Second TP target
    take_profit_3: Optional[float]  # Third TP (optional)
    timestamp: str               # ISO 8601 format
    signal_id: str               # Unique identifier
    bot_name: str                # Source bot
    confidence: float            # 0.0 - 1.0
    metadata: Dict[str, Any]     # Bot-specific data
```

### Cooldown Management
- **Purpose:** Prevent signal spam for same symbol
- **Mechanism:** Track `last_alert` timestamp per symbol
- **Check:** `if now - last_alert < cooldown_minutes: skip`
- **Per-Bot:** Each bot maintains its own cooldown
- **Per-Symbol:** Configurable cooldown per symbol in watchlist

### Signal Lifecycle
```
1. Detect Pattern/Condition
2. Check Cooldown → Skip if too soon
3. Calculate Entry/SL/TP using ATR
4. Validate Risk/Reward ratio
5. Create Signal object
6. Send Telegram notification
7. Log to signals_log.json
8. Add to open_signals state
9. Monitor for TP/SL hits
10. Update status when closed
11. Log to signal_stats
```

---

## Risk Management

### ATR-Based Calculation
```python
# Calculate 14-period ATR
atr = calculate_atr(highs, lows, closes, period=14)

# Stop Loss
if direction == "LONG":
    stop_loss = entry - (atr * sl_atr_multiplier)
else:  # SHORT
    stop_loss = entry + (atr * sl_atr_multiplier)

# Take Profit Levels
risk = abs(entry - stop_loss)
reward1 = risk * tp1_atr_multiplier
reward2 = risk * tp2_atr_multiplier

# Risk/Reward Validation
if reward1 / risk < min_risk_reward:
    reject_signal()
```

### Position Sizing (Future Enhancement)
- **Max Risk per Trade:** 2% of capital
- **Max Open Signals:** Configurable per bot (default: 55)
- **Correlation Check:** Avoid multiple signals on correlated pairs

### Trailing Stop (MOST & PSAR bots)
- **Activation:** When price moves in favor by activation multiplier
- **Distance:** Maintained at distance × ATR behind price
- **Update:** Moves only in profit direction, never reverses

---

## Monitoring & Health Checks

### Health Monitor Component
**File:** `health_monitor.py`

#### Features
- **Startup Message:** Sent when bot starts
- **Shutdown Message:** Sent on graceful termination
- **Heartbeat:** Hourly "still alive" messages
- **Error Tracking:** Records and reports errors
- **Cycle Tracking:** Monitors successful cycles

### Graceful Shutdown
**Implementation:** Signal handlers (SIGINT, SIGTERM)

```python
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    shutdown_requested = True
    logger.info("Shutting down gracefully...")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

while not shutdown_requested:
    run_cycle()
    if shutdown_requested:
        break
    time.sleep(interval)
```

### Logging
- **Location:** `{bot_dir}/logs/{bot_name}.log`
- **Format:** `%(asctime)s | %(levelname)s | %(message)s`
- **Rotation:** Manual (future: implement RotatingFileHandler)
- **Levels:** INFO for normal operations, ERROR for exceptions

### Performance Tracking
**Files:** `signal_stats.py`, `analyze_bot_performance.py`

#### Metrics
- **Win Rate:** TP hits / Total signals
- **P&L:** Cumulative % gain/loss
- **Signals per Symbol:** Distribution analysis
- **TP1/TP2/TP3 Hit Rate:** Individual target success
- **Stop Loss Hit Rate:** Risk realization frequency

---

## Deployment Architecture

### Systemd Integration
**Location:** `{bot_name}.service` files

```ini
[Unit]
Description=Funding Bot - Cryptocurrency Trading
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/home/trader/azure_bots_backup_20251209
ExecStart=/usr/bin/python3 funding_bot/funding_bot.py --loop
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

### Deployment Workflow
```
1. Deploy to server
2. Install dependencies: pip install -r requirements.txt
3. Configure .env file
4. Set file permissions: chmod 600 .env
5. Create systemd services (optional)
6. Enable and start services
7. Monitor logs and health checks
```

### Monitoring Strategy
- **Process Monitoring:** systemd keeps bots running
- **Health Monitoring:** Telegram heartbeats every hour
- **Log Monitoring:** Centralized logging (future: ELK stack)
- **Performance Monitoring:** Signal stats analysis scripts

---

## Security

### Secrets Management
- **API Keys:** Stored in `.env` file (gitignored)
- **File Permissions:** 0o600 for .env, state files, logs
- **Environment Variables:** Loaded via python-dotenv
- **No Hardcoding:** Never commit secrets to git

### File Permissions
- **State Files:** 0o600 (owner read/write only)
- **Lock Files:** 0o600
- **Log Directories:** 0o700 (owner access only)
- **Configuration Files:** 0o644 (readable, writable by owner)

### API Security
- **Rate Limiting:** RateLimitHandler prevents hitting exchange limits
- **Backoff Mechanism:** Exponential backoff on errors
- **Circuit Breaker:** Stops calling failing endpoints
- **SSL/TLS:** HTTPS only for all API calls

### Input Validation
- **JSON Schema:** Validate configuration files (future enhancement)
- **Symbol Validation:** Check format before API calls
- **Numeric Validation:** Ensure prices/volumes are positive
- **Type Checking:** Strict type validation with TypedDict

---

## Development Guidelines

### Code Structure
```
bot_dir/
├── {bot_name}.py           # Main bot logic
├── {bot_name}_watchlist.json  # Symbol configuration
├── {bot_name}_state.json   # Runtime state (gitignored)
├── logs/                   # Log files (gitignored)
│   ├── {bot_name}.log
│   ├── {bot_name}_signals.json
│   └── {bot_name}_stats.json
└── README.md               # Bot-specific documentation
```

### Adding a New Bot

1. **Create Bot Directory**
   ```bash
   mkdir new_bot
   cd new_bot
   ```

2. **Create Main Bot File**
   ```python
   #!/usr/bin/env python3
   import signal
   from pathlib import Path

   # Standard imports
   BASE_DIR = Path(__file__).resolve().parent
   LOG_DIR = BASE_DIR / "logs"
   STATE_FILE = BASE_DIR / "new_bot_state.json"

   # Graceful shutdown
   shutdown_requested = False
   def signal_handler(signum, frame):
       global shutdown_requested
       shutdown_requested = True

   signal.signal(signal.SIGINT, signal_handler)
   signal.signal(signal.SIGTERM, signal_handler)

   # Bot class with run() method
   class NewBot:
       def run(self, loop=False):
           while not shutdown_requested:
               self._run_cycle()
               if not loop or shutdown_requested:
                   break
               time.sleep(self.interval)
   ```

3. **Create Watchlist**
   ```json
   [
     { "symbol": "BTC", "period": "5m", "cooldown_minutes": 5 }
   ]
   ```

4. **Add to global_config.json**
   ```json
   {
     "bots": {
       "new_bot": {
         "enabled": true,
         "max_open_signals": 55,
         "interval_seconds": 300,
         "symbols": [ /* ... */ ]
       }
     }
   }
   ```

5. **Create Start Script**
   ```bash
   #!/bin/bash
   cd "$(dirname "$0")"
   python3 new_bot/new_bot.py --loop
   ```

### Testing Checklist
- [ ] Syntax validation: `python3 -m py_compile bot.py`
- [ ] Manual test: `python3 bot.py` (single cycle)
- [ ] Loop test: `python3 bot.py --loop` (Ctrl+C to exit)
- [ ] State persistence: Check state file after restart
- [ ] Telegram notifications: Verify signal messages
- [ ] Graceful shutdown: Verify SIGTERM handling
- [ ] Log files: Check for errors
- [ ] Performance: Monitor CPU/memory usage

### Code Style
- **PEP 8:** Follow Python style guide
- **Type Hints:** Use for all function signatures
- **Docstrings:** Google-style for all public methods
- **Logging:** Use logger, not print statements
- **Error Handling:** Catch specific exceptions, not bare except

### Git Workflow
```bash
# Create feature branch
git checkout -b feature/new-bot

# Make changes
git add new_bot/
git commit -m "Add new_bot with XYZ strategy"

# Push and create PR
git push origin feature/new-bot
```

---

## Performance Considerations

### Optimization Strategies
- **Parallel API Calls:** Use asyncio for concurrent requests (future)
- **Caching:** Cache expensive calculations (ATR, indicators)
- **Batch Processing:** Process multiple symbols efficiently
- **Lazy Loading:** Load data only when needed
- **Connection Pooling:** Reuse HTTP connections

### Resource Limits
- **API Rate Limits:** MEXC allows ~20 requests/sec
- **Memory:** Each bot ~50-100MB
- **CPU:** Minimal when idle, spikes during calculations
- **Storage:** Log files grow ~10MB/day per bot

### Scalability
- **Horizontal:** Run bots on different machines
- **Vertical:** Optimize individual bot performance
- **Load Balancing:** Distribute symbols across bot instances
- **Database:** Future enhancement for signal storage

---

## Future Enhancements

### Planned Features
1. **HTTP Health Check Endpoints** (In Progress)
   - `/health` - Liveness check
   - `/ready` - Readiness check
   - `/metrics` - Prometheus metrics

2. **Comprehensive Testing**
   - Unit tests for all components
   - Integration tests for bot logic
   - Backtest framework

3. **Enhanced Monitoring**
   - Grafana dashboards
   - Prometheus metrics export
   - Centralized logging (ELK)

4. **Advanced Risk Management**
   - Position sizing calculator
   - Correlation analysis
   - Portfolio risk tracking

5. **Machine Learning**
   - Pattern recognition optimization
   - Signal confidence scoring
   - Adaptive parameters

---

## Troubleshooting

### Common Issues

#### Bot Not Starting
```bash
# Check syntax
python3 -m py_compile bot_name/bot_name.py

# Check .env file
cat .env

# Check logs
tail -f bot_name/logs/bot_name.log
```

#### No Signals Generated
- Check watchlist has symbols
- Verify cooldown hasn't prevented signals
- Check Telegram credentials
- Review bot logic for entry conditions

#### API Errors
- Verify MEXC API is accessible
- Check rate limit backoff
- Review exchange connectivity
- Check symbol format (BTC vs BTC/USDT)

#### State File Corruption
```bash
# Backup current state
cp bot_name/bot_state.json bot_name/bot_state.json.bak

# Reset state
echo '{"last_alert": {}, "open_signals": {}}' > bot_name/bot_state.json
```

---

## Contributing

### Guidelines
1. Follow existing code structure
2. Add comprehensive docstrings
3. Include unit tests
4. Update documentation
5. Test thoroughly before PR
6. Follow git workflow

### Code Review Checklist
- [ ] Code follows PEP 8
- [ ] All functions have type hints
- [ ] Docstrings are comprehensive
- [ ] Tests are passing
- [ ] No hardcoded secrets
- [ ] Graceful shutdown implemented
- [ ] Error handling is robust
- [ ] Logging is appropriate

---

## References

### Internal Documentation
- `COMPREHENSIVE_AUDIT_AND_FIX_PLAN.md` - Complete audit results
- `PHASE1_COMPLETION_SUMMARY.md` - Critical fixes completed
- `PHASE2_PROGRESS_REPORT.md` - Current progress
- `SETUP_INSTRUCTIONS.md` - Deployment guide
- `11_BOTS_COMPLETE_GUIDE.md` - Original system overview

### External Resources
- [MEXC API Documentation](https://mxcdevelop.github.io/apidocs/spot_v3_en/)
- [CCXT Library](https://docs.ccxt.com/)
- [Telegram Bot API](https://core.telegram.org/bots/api)
- [Technical Analysis Guide](https://www.investopedia.com/technical-analysis-4689657)

---

**Document Version:** 1.0
**Author:** Automated Documentation System
**Last Review:** 2025-12-14
**Next Review:** Quarterly or after major changes
