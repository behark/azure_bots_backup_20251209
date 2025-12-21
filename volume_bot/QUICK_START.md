# Volume Bot - Quick Start Guide
**For Complete Beginners**

## 🚀 Start the Bot (3 Easy Steps)

### **Step 1: Make sure your .env file is ready**
```bash
# Check it exists
ls -la .env

# Should contain:
# TELEGRAM_BOT_TOKEN=your_bot_token
# TELEGRAM_CHAT_ID=your_chat_id
# BINANCEUSDM_API_KEY=your_api_key
# BINANCEUSDM_SECRET=your_secret
```

### **Step 2: Run the bot with optimized settings**
```bash
python3 volume_vn_bot.py --config config.json
```

That's it! The bot is now running with optimal settings.

### **Step 3: Monitor the logs**
```bash
# In another terminal, watch what's happening:
tail -f logs/volume_vn_bot.log
```

---

## 📱 What You'll See in Telegram

### **1. New Signal Alert:**
```
🟢 LONG BTC/USDT ✅ GOOD

🆔 BTC-USDT-L-X7K2

💰 Entry: 43250.00
🛑 Stop Loss: 42400.00
🎯 Take Profit 1: 44500.00
🎯 Take Profit 2: 45750.00

🕒 Timeframe: 15m | 🏦 Exchange: BINANCEUSDM
📊 POC/VAH/VAL: 43100.00 / 43800.00 / 42600.00

📈 BTC/USDT history: TP 15 / SL 3 (Win rate: 83.3%)

📝 Factors: Price > EMA20, RSI favorable (65.4), Bullish Momentum (RSI Trend), Volume spike
```

### **2. Signal Result (TP/SL Hit):**
```
🟢 LONG BTC/USDT - TP1 HIT!

💰 Entry: 43250.00
🎯 Exit: 44500.00
📊 PnL: +2.89%

🛑 SL: 42400.00
🎯 TP1: 44500.00
🎯 TP2: 45750.00

🕒 Timeframe: 15m | 🏦 Exchange: BINANCEUSDM

📈 BTC/USDT Performance History:
   TP1: 12 | TP2: 5 | SL: 3
   Win Rate: 85.0% (17/20)
   Avg PnL: +3.45%

🆔 BTC/USDT-15m-binanceusdm-2025-12-19T16:30:15
```

**Note:** You won't get duplicate result notifications for the same symbol within 15 minutes!

---

## ⚙️ Common Commands

```bash
# Normal run (recommended)
python3 volume_vn_bot.py --config config.json

# Test mode (one cycle only)
python3 volume_vn_bot.py --config config.json --once

# Debug mode (see everything)
python3 volume_vn_bot.py --config config.json --log-level DEBUG

# Extra detailed logs
python3 volume_vn_bot.py --config config.json --detailed-logging

# Check open signals only (no new analysis)
python3 volume_vn_bot.py --track

# Run in background
nohup python3 volume_vn_bot.py --config config.json > /dev/null 2>&1 &

# Stop background process
pkill -f volume_vn_bot.py
```

---

## 📊 Understanding the Settings (Simple Explanation)

### **In `config.json`:**

**Quality Control (How picky the bot is):**
- `min_confidence_score: 5.0` - Only trade when bot sees 5+ bullish/bearish signs
- `min_factors_long: 4` - Need 4+ reasons to go LONG
- `volume_spike_threshold: 1.8` - Volume must be 1.8x normal (big moves)

**Risk Protection:**
- `max_open_signals: 15` - Maximum 15 trades at once
- `default_stop_loss_pct: 2.0` - Lose max 2% per trade
- `tp1_multiplier: 2.5` - Take profit 1 at 2.5x the risk (example: risk $100, profit $250)
- `tp2_multiplier: 4.0` - Take profit 2 at 4x the risk (example: risk $100, profit $400)

**Alert Control:**
- `cooldown_minutes: 15` - Wait 15min before repeating same symbol alert
- `result_notification_cooldown_minutes: 15` - Wait 15min between TP/SL notifications

---

## 🎯 Recommended Settings for Different Goals

### **More Signals (but lower win rate):**
Change these in `config.json`:
```json
{
  "signal_management": {
    "min_confidence_score": 4.0  // Lower from 5.0
  },
  "filtering": {
    "min_factors_long": 3,        // Lower from 4
    "min_factors_short": 3
  }
}
```

### **Fewer Signals (but higher win rate):**
```json
{
  "signal_management": {
    "min_confidence_score": 6.0  // Higher from 5.0
  },
  "filtering": {
    "min_factors_long": 5,        // Higher from 4
    "min_factors_short": 5
  }
}
```

### **More Conservative (safer):**
```json
{
  "risk_management": {
    "max_open_signals": 10,       // Lower from 15
    "default_stop_loss_pct": 2.5, // Wider from 2.0
    "min_risk_reward": 2.5        // Higher from 2.0
  }
}
```

---

## 🔍 Monitoring Your Bot

### **Check if bot is running:**
```bash
ps aux | grep volume_vn_bot
```

### **View recent activity:**
```bash
# Last 50 lines
tail -n 50 logs/volume_vn_bot.log

# Watch live
tail -f logs/volume_vn_bot.log
```

### **Check for errors:**
```bash
tail -n 20 logs/volume_vn_errors.log
```

### **Count today's signals:**
```bash
grep "Signal.*closed" logs/volume_vn_bot.log | grep "$(date +%Y-%m-%d)" | wc -l
```

### **See performance for specific symbol:**
```bash
grep "BTC/USDT" logs/volume_vn_bot.log | grep "closed"
```

---

## ❓ What If...

### **"Not getting any signals"**
1. Check if bot is analyzing:
   ```bash
   tail -f logs/volume_vn_bot.log | grep "Analyzing"
   ```
2. Lower the confidence threshold in `config.json`:
   ```json
   "min_confidence_score": 4.0  // Was 5.0
   ```

### **"Getting too many signals"**
1. Increase confidence threshold in `config.json`:
   ```json
   "min_confidence_score": 6.0  // Was 5.0
   ```
2. Increase cooldown:
   ```json
   "cooldown_minutes": 30  // Was 15
   ```

### **"Bot stopped working"**
1. Check the error log:
   ```bash
   tail -n 50 logs/volume_vn_errors.log
   ```
2. Restart with:
   ```bash
   python3 volume_vn_bot.py --config config.json
   ```

### **"Getting duplicate notifications"**
- **Signal alerts:** Should not repeat for same symbol within 15min
- **Result alerts:** Should not repeat for same symbol within 15min
- If you still see duplicates, increase cooldown in `config.json`

### **"Hit rate limit (429 error)"**
1. Bot will automatically back off for 2 minutes
2. If it keeps happening, reduce API calls in `config.json`:
   ```json
   "calls_per_minute": 30  // Was 40
   ```

---

## 📈 Expected Results

With **optimized config.json settings:**

- **Signals per day:** 5-15 signals
- **Win rate target:** 60-70%
- **Average profit per winner:** 2.5-4x the risk
- **Risk per trade:** 2% max

**Example:**
- Risk $100 per trade
- Win 65% of trades
- Average winner: 3x = $300
- Average loser: 1x = $100
- Over 100 trades: 65 wins ($19,500) - 35 losses ($3,500) = $16,000 profit

---

## ✅ Daily Checklist

1. **Morning (Once):**
   - Check bot is still running: `ps aux | grep volume_vn_bot`
   - Quick glance at errors: `tail -n 20 logs/volume_vn_errors.log`

2. **Evening (Once):**
   - Check today's performance: `grep "closed" logs/volume_vn_bot.log | grep "$(date +%Y-%m-%d)"`
   - Check Telegram for signal results

3. **Weekly:**
   - Calculate win rate from logs or Telegram messages
   - Adjust settings if win rate < 55% or > 75%

---

## 📚 Additional Resources

- **Full optimization guide:** `OPTIMIZATION_GUIDE.md` (detailed explanations)
- **All fixes report:** `FIXES_COMPLETE_REPORT.md` (what was fixed)
- **Verify bot health:** Run `python3 verify_fixes.py`

---

## 🆘 Quick Fixes

**Bot not starting:**
```bash
# Check Python version (need 3.8+)
python3 --version

# Check dependencies
pip3 install -r requirements.txt  # If requirements.txt exists

# Check syntax
python3 -m py_compile volume_vn_bot.py
```

**No Telegram messages:**
```bash
# Test .env file
cat .env | grep TELEGRAM

# Check bot is running
ps aux | grep volume_vn_bot

# Check logs
grep "Telegram" logs/volume_vn_bot.log
```

**Too many logs (disk space):**
```bash
# Logs auto-rotate at 10MB
# To clean old logs:
rm logs/volume_vn_bot.log.*
rm logs/volume_vn_errors.log.*
```

---

## 🎓 Pro Tips

1. **Start small:** Test with small position sizes for first 2 weeks
2. **Monitor win rate:** Should be 60-70% with these settings
3. **Trust the system:** Don't override every signal manually
4. **Keep logs:** They help identify what works and what doesn't
5. **Be patient:** Crypto markets have slow days - normal to get 0-2 signals sometimes
6. **Performance history:** The bot now shows you how each symbol performs historically!

---

**Need more help?**
- Read `OPTIMIZATION_GUIDE.md` for detailed explanations
- Check logs: `tail -f logs/volume_vn_bot.log`
- Run verification: `python3 verify_fixes.py`

---

**Status:** 🟢 Ready to Trade
**Version:** 2.1 (Enhanced Edition)
