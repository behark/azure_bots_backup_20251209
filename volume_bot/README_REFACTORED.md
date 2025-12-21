# Volume VN Bot - Refactored Version

## 🎯 What's New in This Version

This is the **refactored and production-ready** version of the Volume VN Bot with all critical issues from the code review resolved.

### ✅ Fixed Critical Issues

1. **Look-Ahead Bias Fixed** - Now uses only closed candles for analysis
2. **Proper Duplicate Detection** - Checks exchange and timeframe
3. **Thread-Safe State Persistence** - File locking prevents corruption
4. **Enabled Stale Signal Cleanup** - Automatic archiving of old signals
5. **Comprehensive Error Handling** - Specific exceptions for network errors
6. **Request Timeouts Added** - Prevents hanging on network issues
7. **Symbol Normalization** - Proper reversal detection across exchanges
8. **API Credential Validation** - Startup validation of exchange credentials

### 🔐 Security Improvements

- Telegram credential validation
- API key format checking
- No hardcoded secrets
- Environment variable validation
- Configuration externalized

### ⚙️ Configuration Management

All settings are now configurable via:
- Environment variables (.env file)
- JSON configuration file (config.json)
- Command-line arguments

## 📋 Requirements

- Python 3.8+
- Exchange API credentials (Binance/MEXC/Bybit)
- Telegram bot token and chat ID

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd volume_bot
pip install -r ../requirements.txt
```

### 2. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit with your credentials
nano .env
```

**Minimum required variables:**
```bash
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
BINANCEUSDM_API_KEY=your_api_key_here
BINANCEUSDM_SECRET=your_secret_here
```

### 3. Create Watchlist

Create `volume_watchlist.json`:

```json
[
  {
    "symbol": "BTC/USDT",
    "timeframe": "5m",
    "exchange": "binanceusdm",
    "market_type": "swap"
  },
  {
    "symbol": "ETH/USDT",
    "timeframe": "5m",
    "exchange": "binanceusdm",
    "market_type": "swap"
  }
]
```

### 4. Validate Setup

```bash
# Test configuration (runs one cycle)
python volume_vn_bot.py --once

# Check logs
tail -f logs/volume_vn_bot.log
```

### 5. Run Bot

```bash
# Run continuously
python volume_vn_bot.py

# Run with custom cooldown
python volume_vn_bot.py --cooldown 10

# Only check open signals (no new signals)
python volume_vn_bot.py --track

# Use custom config file
python volume_vn_bot.py --config custom_config.json
```

## 📁 File Structure

```
volume_bot/
├── volume_vn_bot.py          # Main bot script (refactored)
├── config.py                 # NEW: Configuration module
├── volume_profile.py         # Volume profile calculations
├── .env                      # Your credentials (DO NOT COMMIT)
├── .env.example              # NEW: Template for .env
├── config.json.example       # NEW: Template for config
├── volume_watchlist.json     # Your trading pairs
├── volume_vn_state.json      # Bot state (auto-created)
├── volume_vn_signals.json    # Signal log (auto-created)
└── logs/
    ├── volume_vn_bot.log     # Bot logs
    └── volume_stats.json     # Performance stats
```

## ⚙️ Configuration Options

### Environment Variables

All configuration can be set via environment variables. See `.env.example` for complete list.

**Key settings:**

| Variable | Default | Description |
|----------|---------|-------------|
| `VOLUME_BOT_MAX_OPEN_SIGNALS` | 50 | Maximum concurrent signals |
| `VOLUME_BOT_COOLDOWN_MINUTES` | 5 | Cooldown between alerts |
| `VOLUME_BOT_STOP_LOSS_PCT` | 1.5 | Stop loss percentage |
| `VOLUME_BOT_MAX_SIGNAL_AGE_HOURS` | 24 | Auto-close age |
| `VOLUME_BOT_SPIKE_THRESHOLD` | 1.5 | Volume spike multiplier |

### JSON Configuration

Create `config.json` for advanced settings (see `config.json.example`):

```json
{
  "volume_bot": {
    "risk_management": {
      "max_open_signals": 30,
      "tp1_multiplier": 2.5,
      "tp2_multiplier": 4.0
    }
  }
}
```

Use with: `python volume_vn_bot.py --config config.json`

## 🔍 Monitoring

### Check Bot Health

```bash
# View logs in real-time
tail -f logs/volume_vn_bot.log

# Check for errors
grep ERROR logs/volume_vn_bot.log

# View performance stats
cat logs/volume_stats.json | python -m json.tool
```

### Telegram Commands

The bot sends notifications for:
- ✅ New signals (LONG/SHORT)
- 🎯 TP1/TP2/SL hits
- ⚠️ Signal reversals
- 🚀 Startup/shutdown messages

## 🐛 Troubleshooting

### "Missing required environment variables"

**Solution:** Check your `.env` file has:
```bash
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

### "Invalid API credentials for binanceusdm"

**Solution:** 
1. Verify keys on exchange website
2. Check IP whitelist settings
3. Ensure API has trading permissions

### "Network error fetching ticker"

**Solution:**
1. Check internet connection
2. Verify exchange API status
3. Check if rate limited (bot will auto-backoff)

### No Signals Generated

**Solution:**
1. Check `volume_watchlist.json` exists and is valid
2. Verify symbols format (e.g., `BTC/USDT:USDT` for futures)
3. Check logs for analysis errors
4. Ensure market conditions meet signal criteria

### State File Corruption

**Solution:** The refactored bot has file locking, but if corrupted:
```bash
# Backup first
cp volume_vn_state.json volume_vn_state.json.backup

# Delete to reset
rm volume_vn_state.json

# Bot will create fresh state file
```

## 📊 Performance Stats

View win rate and statistics:

```python
import json

with open('logs/volume_stats.json', 'r') as f:
    stats = json.load(f)
    
print(f"Total Signals: {stats['total_signals']}")
print(f"Win Rate: {stats['win_rate']:.1f}%")
print(f"TP Hits: {stats['tp_hits']}")
print(f"SL Hits: {stats['sl_hits']}")
```

## 🔒 Security Best Practices

1. **API Keys:**
   - Use read-only keys when possible
   - Enable IP whitelist on exchange
   - Never share or commit API keys
   - Rotate keys regularly

2. **Environment:**
   - Keep `.env` out of version control
   - Use strong passwords
   - Limit API permissions to trading only

3. **Monitoring:**
   - Review logs daily
   - Check for unusual activity
   - Monitor open positions
   - Set up alerts for errors

## 🚦 Production Deployment

### Systemd Service (Linux)

Create `/etc/systemd/system/volume_vn_bot.service`:

```ini
[Unit]
Description=Volume VN Trading Bot
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/azure_bots_backup_20251209/volume_bot
ExecStart=/usr/bin/python3 volume_vn_bot.py
Restart=on-failure
RestartSec=10
StandardOutput=append:/path/to/volume_bot/logs/service.log
StandardError=append:/path/to/volume_bot/logs/service_error.log

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable volume_vn_bot
sudo systemctl start volume_vn_bot
sudo systemctl status volume_vn_bot
```

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY volume_bot/ ./volume_bot/
COPY notifier.py health_monitor.py signal_stats.py ./

WORKDIR /app/volume_bot

CMD ["python", "volume_vn_bot.py"]
```

Run:
```bash
docker build -t volume-vn-bot .
docker run -d --name volume-bot --env-file .env volume-vn-bot
```

## 📈 Next Steps

After successful deployment:

1. **Monitor for 1 week** in production
2. **Backtest on historical data** (use backtesting framework from report)
3. **Tune parameters** based on performance
4. **Add trailing stops** for profit protection
5. **Implement position sizing** based on account balance
6. **Set up database logging** for better analytics

## 🆘 Support

- **Documentation:** See [VOLUME_BOT_CODE_REVIEW.md](../VOLUME_BOT_CODE_REVIEW.md)
- **Issues:** Check logs first, then review troubleshooting section
- **Configuration:** See `.env.example` and `config.json.example`

## 📝 Changelog

### v2.0.0 (Refactored) - 2025-12-17

**Critical Fixes:**
- ✅ Fixed look-ahead bias in volume calculations
- ✅ Added file locking for state persistence
- ✅ Enabled stale signal cleanup with archiving
- ✅ Fixed duplicate signal detection logic
- ✅ Added proper error handling and timeouts
- ✅ Fixed symbol reversal detection
- ✅ Added API credential validation

**New Features:**
- ✅ Configuration module (config.py)
- ✅ Environment validation on startup
- ✅ Configurable risk management
- ✅ Better logging and monitoring
- ✅ Command-line configuration override

**Security:**
- ✅ Telegram credential validation
- ✅ No hardcoded secrets
- ✅ API key format checking

## ⚖️ License

See parent project for license information.

## ⚠️ Disclaimer

This software is for educational purposes only. Trading cryptocurrencies carries risk. Always test thoroughly before using real funds. The developers are not responsible for any financial losses.

---

**Status:** ✅ **PRODUCTION READY**

All critical issues resolved. Recommended for live trading after proper testing.
