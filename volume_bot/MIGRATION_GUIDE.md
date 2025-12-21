# Migration Guide: Upgrading to Refactored Volume VN Bot

## Overview

This guide helps you migrate from the original `volume_vn_bot.py` to the refactored version with all critical fixes applied.

## What Changed?

### Files Created
- ✅ `config.py` - New configuration module
- ✅ `.env.example` - Environment variable template
- ✅ `config.json.example` - JSON configuration template
- ✅ `README_REFACTORED.md` - Updated documentation

### Files Modified
- ✅ `volume_vn_bot.py` - Complete refactor with all fixes

### Files Preserved (Compatible)
- ✅ `volume_vn_state.json` - Your existing state file works
- ✅ `volume_vn_signals.json` - Signal log preserved
- ✅ `volume_watchlist.json` - No changes needed
- ✅ `logs/volume_stats.json` - Statistics carry over

## Migration Steps

### Step 1: Backup Your Data

**CRITICAL: Do this first!**

```bash
cd volume_bot

# Backup state files
cp volume_vn_state.json volume_vn_state.json.backup
cp volume_vn_signals.json volume_vn_signals.json.backup
cp logs/volume_stats.json logs/volume_stats.json.backup

# Backup watchlist
cp volume_watchlist.json volume_watchlist.json.backup

# Backup old script
cp volume_vn_bot.py volume_vn_bot.py.old
```

### Step 2: Stop Running Bot

```bash
# If running in terminal, press Ctrl+C

# If running as service
sudo systemctl stop volume_vn_bot

# Verify stopped
ps aux | grep volume_vn_bot
```

### Step 3: Update Files

The refactored files are already in place. No additional downloads needed.

### Step 4: Create Environment File

```bash
# Copy template
cp .env.example .env

# Edit with your credentials
nano .env
```

**Migrate your credentials:**

If you previously had credentials in environment or hardcoded:

```bash
# Example migration from old setup
# Old: export TELEGRAM_BOT_TOKEN="123..."
# New: Add to .env file:
TELEGRAM_BOT_TOKEN=123...
TELEGRAM_CHAT_ID=-100...
BINANCEUSDM_API_KEY=your_key
BINANCEUSDM_SECRET=your_secret
```

### Step 5: Test Configuration

```bash
# Validate environment
python volume_vn_bot.py --once

# Check for errors
tail -20 logs/volume_vn_bot.log
```

Expected output:
```
✅ All required environment variables present
✅ BINANCEUSDM credentials validated successfully
Configuration loaded successfully
Starting Volume VN cycle for X pairs
```

### Step 6: Verify State Compatibility

```bash
# Check state file is valid
python -c "import json; print(json.load(open('volume_vn_state.json'))['open_signals'].__len__(), 'open signals')"

# If errors, state file may be corrupted
# Use backup: cp volume_vn_state.json.backup volume_vn_state.json
```

### Step 7: Resume Operation

```bash
# Run in terminal (testing)
python volume_vn_bot.py

# Or restart service
sudo systemctl start volume_vn_bot
sudo systemctl status volume_vn_bot
```

## Configuration Changes

### Old Hardcoded Values → New Config

| Old Location | New Location | Default |
|--------------|--------------|---------|
| Line 390: `0.985` | `VOLUME_BOT_STOP_LOSS_PCT` | 1.5% |
| Line 434: `1.015` | `VOLUME_BOT_STOP_LOSS_PCT` | 1.5% |
| Line 595: `MAX_OPEN_SIGNALS = 50` | `VOLUME_BOT_MAX_OPEN_SIGNALS` | 50 |
| Line 207: `1.5` | `VOLUME_BOT_SPIKE_THRESHOLD` | 1.5 |
| Line 356: `1.2` | `VOLUME_BOT_PRESSURE_THRESHOLD` | 1.2 |
| Line 580: `max_age_hours=24` | `VOLUME_BOT_MAX_SIGNAL_AGE_HOURS` | 24 |
| Line 638: `60` seconds | `VOLUME_BOT_CYCLE_INTERVAL` | 60 |

### Customizing Settings

**Option 1: Environment Variables (Recommended)**

Edit `.env`:
```bash
VOLUME_BOT_MAX_OPEN_SIGNALS=30
VOLUME_BOT_STOP_LOSS_PCT=2.0
VOLUME_BOT_TP1_MULTIPLIER=2.5
```

**Option 2: JSON Configuration**

Copy and edit template:
```bash
cp config.json.example config.json
nano config.json
```

Run with:
```bash
python volume_vn_bot.py --config config.json
```

**Option 3: Command-Line Arguments**

```bash
python volume_vn_bot.py --cooldown 10
```

## Breaking Changes

### 1. VolumeAnalyzer Initialization

**Old:**
```python
analyzer = VolumeAnalyzer()
```

**New:**
```python
config = load_config()
analyzer = VolumeAnalyzer(config=config)
```

**Migration:** If you have custom scripts importing VolumeAnalyzer, update to pass config.

### 2. SignalTracker Initialization

**Old:**
```python
tracker = SignalTracker(analyzer, stats=stats)
```

**New:**
```python
tracker = SignalTracker(analyzer, stats=stats, config=config)
```

### 3. has_open_signal Method

**Old:**
```python
tracker.has_open_signal(symbol)
```

**New:**
```python
tracker.has_open_signal(symbol, exchange=exchange, timeframe=timeframe)
```

**Impact:** More accurate duplicate detection, but now checks exchange by default.

### 4. cleanup_stale_signals Method

**Old:**
```python
# Was disabled, returned 0
```

**New:**
```python
# Now active, removes signals older than max_age_hours
removed_count = tracker.cleanup_stale_signals(max_age_hours=24)
```

**Impact:** Old signals will now be auto-closed and archived to stats.

## New Features You Can Use

### 1. Environment Validation

```bash
# Bot now validates on startup
python volume_vn_bot.py

# Skip validation (not recommended)
python volume_vn_bot.py --skip-validation
```

### 2. Custom Config Files

```bash
# Production config
python volume_vn_bot.py --config config.prod.json

# Testing config
python volume_vn_bot.py --config config.test.json
```

### 3. Improved Error Messages

- Specific exception types (NetworkError vs ExchangeError)
- Better logging with context
- Automatic retry with exponential backoff

### 4. File Locking

State file now uses file locking to prevent corruption from multiple instances.

### 5. Credential Validation

Bot validates exchange credentials on startup:
```
✅ BINANCEUSDM credentials validated successfully
```

## Rollback Instructions

If you need to revert to the old version:

```bash
# Stop new bot
pkill -f volume_vn_bot.py

# Restore old version
cp volume_vn_bot.py.old volume_vn_bot.py

# Restore state files if needed
cp volume_vn_state.json.backup volume_vn_state.json

# Restart
python volume_vn_bot.py
```

## Testing Checklist

Before deploying to production, verify:

- [ ] Environment validation passes
- [ ] Telegram notifications work
- [ ] Exchange credentials validated
- [ ] Watchlist loaded successfully
- [ ] Existing open signals preserved
- [ ] Stats history maintained
- [ ] Signals generated correctly
- [ ] TP/SL logic works as expected
- [ ] Stale signals cleaned up after 24h
- [ ] Duplicate detection works
- [ ] Logs are clean (no errors)

## Performance Impact

### Expected Improvements
- ✅ **Fewer false signals** (look-ahead bias fixed)
- ✅ **No state corruption** (file locking)
- ✅ **Better duplicate handling** (fewer missed opportunities)
- ✅ **Cleaner state file** (automatic cleanup)

### Potential Changes
- ⚠️ **More signals possible** - Better duplicate detection may allow more valid signals
- ⚠️ **Different TP/SL levels** - Fixed calculations may change entry/exit points
- ⚠️ **Signals expire** - Stale cleanup now active (good for risk management)

## Troubleshooting Migration

### Issue: "No module named 'config'"

**Solution:**
```bash
# Ensure config.py exists in volume_bot directory
ls -la config.py

# If missing, copy from refactored version
```

### Issue: "State file corrupted"

**Solution:**
```bash
# Use backup
cp volume_vn_state.json.backup volume_vn_state.json

# Or start fresh
rm volume_vn_state.json
python volume_vn_bot.py --once
```

### Issue: "Different signal behavior"

**Explanation:** This is expected due to:
1. Look-ahead bias fix (uses only closed candles)
2. Better duplicate detection
3. Fixed volume spike calculation

**Action:** Monitor for 24-48 hours to verify new behavior is correct.

### Issue: "Missing environment variables"

**Solution:**
```bash
# Check .env file exists
ls -la .env

# Validate format
cat .env | grep TELEGRAM

# Ensure no spaces around =
# Good: TELEGRAM_BOT_TOKEN=123
# Bad:  TELEGRAM_BOT_TOKEN = 123
```

## Support

If you encounter issues during migration:

1. **Check logs:** `tail -50 logs/volume_vn_bot.log`
2. **Review checklist** in this guide
3. **Compare with `.env.example`**
4. **Test with `--once` flag** before continuous running
5. **Use backups** if state corruption occurs

## Post-Migration Monitoring

For the first week after migration:

- **Daily:** Check logs for errors
- **Daily:** Verify signal counts are reasonable
- **Daily:** Check TP/SL hit rates in stats
- **Weekly:** Compare performance with pre-migration
- **Weekly:** Review and tune configuration

## Next Steps

After successful migration:

1. ✅ Review `README_REFACTORED.md` for full feature list
2. ✅ Set up monitoring/alerting
3. ✅ Consider implementing features from [VOLUME_BOT_CODE_REVIEW.md](../VOLUME_BOT_CODE_REVIEW.md)
4. ✅ Optimize configuration based on performance
5. ✅ Set up automated backups

---

**Migration Status:** Ready for production deployment

**Estimated Migration Time:** 15-30 minutes

**Downtime Required:** 5-10 minutes

**Data Loss Risk:** None (if backups are made)
