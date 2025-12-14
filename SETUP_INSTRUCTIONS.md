# Trading Bots Setup & Management

## Critical Features Implemented ✅

1. **Process Management** - Systemd services for auto-restart
2. **Security** - .gitignore and secured .env file
3. **Health Monitoring** - Hourly heartbeat alerts to Telegram
4. **API Rate Limiting** - Protection against rate limits

---

## Quick Start (Current Session)

### Option 1: Run with startup scripts (current method)
```bash
cd /home/behar/Desktop/azure_bots_backup_20251209

# Kill any existing bots
pkill -f liquidation_bot.py
pkill -f funding_bot.py
pkill -f volume_vn_bot.py

# Start all bots in background
nohup bash start_liquidation_bot.sh > /dev/null 2>&1 &
nohup bash start_funding_bot.sh > /dev/null 2>&1 &
nohup bash start_volume_vn_bot.sh > /dev/null 2>&1 &

# Check if running
ps aux | grep -E "(liquidation_bot|funding_bot|volume_vn_bot)" | grep -v grep
```

---

## Production Setup (Recommended - Systemd Services)

### 1. Install Services
```bash
cd /home/behar/Desktop/azure_bots_backup_20251209

# Copy service files to systemd
sudo cp liquidation_bot.service /etc/systemd/system/
sudo cp funding_bot.service /etc/systemd/system/
sudo cp volume_bot.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload
```

### 2. Enable Services (Auto-start on boot)
```bash
sudo systemctl enable liquidation_bot.service
sudo systemctl enable funding_bot.service
sudo systemctl enable volume_bot.service
```

### 3. Start Services
```bash
sudo systemctl start liquidation_bot.service
sudo systemctl start funding_bot.service
sudo systemctl start volume_bot.service
```

### 4. Check Status
```bash
sudo systemctl status liquidation_bot.service
sudo systemctl status funding_bot.service
sudo systemctl status volume_bot.service
```

### 5. View Logs
```bash
# Real-time logs
sudo journalctl -u liquidation_bot.service -f
sudo journalctl -u funding_bot.service -f
sudo journalctl -u volume_bot.service -f

# Or check bot-specific logs
tail -f liquidation_bot/logs/liquidation_bot.log
tail -f funding_bot/logs/funding_bot.log
tail -f volume_bot/logs/volume_vn_bot.log
```

### 6. Stop/Restart Services
```bash
# Stop
sudo systemctl stop liquidation_bot.service
sudo systemctl stop funding_bot.service
sudo systemctl stop volume_bot.service

# Restart
sudo systemctl restart liquidation_bot.service
sudo systemctl restart funding_bot.service
sudo systemctl restart volume_bot.service
```

---

## New Features Explained

### 1. Health Monitoring 💚
- **Hourly heartbeat** messages to Telegram showing bot status
- **Error tracking** - tracks and reports errors in heartbeat
- **Startup/shutdown** notifications
- **Uptime tracking** - see how long bots have been running

### 2. API Rate Limiting ⚡
- **Automatic delays** between API calls
- **Exponential backoff** on errors (prevents hammering failing APIs)
- **Circuit breaker** - temporarily stops calling failing endpoints
- **Per-endpoint tracking** - tracks issues per symbol

### 3. Better Error Handling 🛡️
- **Auto-recovery** - bots retry on errors instead of crashing
- **Error reporting** - errors sent to health monitor
- **Graceful shutdown** - cleanup and notification on stop

### 4. Security 🔒
- `.env` file secured with 600 permissions (owner-only)
- `.gitignore` prevents committing secrets to git
- Logs rotation-ready

---

## Monitoring Your Bots

### Telegram Notifications You'll Receive:

1. **🚀 Startup Messages** - When bot starts
2. **💚 Hourly Heartbeats** - Status, uptime, cycle count, errors
3. **🛑 Shutdown Messages** - When bot stops (clean exit)
4. **📊 Trading Signals** - Your normal trading alerts
5. **⚠️ Errors** - Reported in hourly heartbeat

### What to Watch For:

- If you don't receive heartbeat for >2 hours → Bot might be down
- High error count in heartbeat → Check logs for API issues
- No startup message after restart → Check systemd status

---

## Configuration Files

### Bot Configuration
- `liquidation_bot/liquidation_watchlist.json` - Symbols to monitor
- `funding_bot/funding_watchlist.json` - Symbols to monitor
- `volume_bot/volume_watchlist.json` - Symbols to monitor

### Environment Variables (.env)
```
TELEGRAM_BOT_TOKEN_LIQUIDATION=your_token
TELEGRAM_BOT_TOKEN_FUNDING=your_token
TELEGRAM_BOT_TOKEN_VOLUME=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Rate Limiting Settings
Edit the bot files to adjust:
```python
# In __init__ method of each bot
self.rate_limiter = RateLimiter(
    calls_per_minute=60,  # Max 60 calls/minute
    backoff_file=LOG_DIR / "rate_limiter.json"
)
```

---

## Troubleshooting

### Bot not starting?
```bash
# Check Python and venv
/home/behar/Desktop/azure_bots_backup_20251209/venv/bin/python --version

# Check dependencies
/home/behar/Desktop/azure_bots_backup_20251209/venv/bin/pip list

# Check logs
tail -100 liquidation_bot/logs/liquidation_bot.log
```

### Not receiving Telegram messages?
```bash
# Check .env file
cat /home/behar/Desktop/azure_bots_backup_20251209/.env

# Test Telegram credentials
# (run in Python)
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv('/home/behar/Desktop/azure_bots_backup_20251209/.env')
print('Token:', os.getenv('TELEGRAM_BOT_TOKEN_LIQUIDATION'))
print('Chat ID:', os.getenv('TELEGRAM_CHAT_ID'))
"
```

### High API error rate?
- Check rate_limiter.json files - might show backoff delays
- Consider reducing `calls_per_minute` in rate limiter
- Check exchange API status

---

## Maintenance

### Daily
- Check Telegram for heartbeat messages
- Verify signals are being sent

### Weekly
- Review logs for persistent errors
- Check disk space: `df -h`
- Clean old logs: `find */logs -name "*.log" -mtime +30 -delete`

### Monthly
- Review bot performance and signal quality
- Update dependencies: `venv/bin/pip install --upgrade ccxt python-dotenv numpy`
- Backup state files: `tar -czf backup.tar.gz *_state.json`

---

## Emergency Commands

### Kill all bots immediately
```bash
pkill -9 -f "liquidation_bot.py|funding_bot.py|volume_vn_bot.py"
```

### Check if bots are running
```bash
ps aux | grep -E "liquidation_bot|funding_bot|volume_vn_bot" | grep -v grep
```

### View all bot logs at once
```bash
tail -f liquidation_bot/logs/*.log funding_bot/logs/*.log volume_bot/logs/*.log
```

---

## Need Help?

- Check logs in `*/logs/` directories
- Review Telegram heartbeat messages for clues
- Check systemd status if using services
- Verify .env credentials are correct
