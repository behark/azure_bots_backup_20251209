# Comprehensive Project Audit & Fix Plan
**Date:** 2025-12-14
**Project:** Azure Bots Cryptocurrency Trading System
**Total Issues Found:** 62 issues across 4 categories

---

## Executive Summary

A deep scan of the cryptocurrency trading bot project has identified **62 issues** across 4 categories:
- **Code Quality:** 17 issues (1 critical, 5 high, 5 medium, 7 low)
- **Security:** 17 issues (1 critical, 6 high, 6 medium, 4 low)
- **Configuration:** 13 issues (2 critical, 5 high, 4 medium, 3 low)
- **Maintenance:** 50+ issues (4 critical, 11 high, 20 medium, 15+ low)

**Total by Severity:**
- **CRITICAL:** 8 issues requiring immediate attention
- **HIGH:** 27 issues with significant impact
- **MEDIUM:** 35 issues affecting reliability
- **LOW:** 29 issues for long-term improvement

---

## PHASE 1: CRITICAL FIXES (Immediate - Day 1)

### 1.1 Fix Runtime-Breaking Bug
**Issue:** strat_bot.py:642 - Undefined variable 'sl'
**Impact:** Bot will crash with NameError
**File:** `/home/user/azure_bots_backup_20251209/strat_bot/strat_bot.py`
**Fix:**
```python
# Line 642: Change from
stop_loss=sl,  # ERROR: 'sl' is not defined

# To:
stop_loss=stop_loss,
```

### 1.2 Create requirements.txt
**Issue:** Missing dependency specification
**Impact:** Cannot install or reproduce environment
**File:** `/home/user/azure_bots_backup_20251209/requirements.txt`
**Fix:**
```txt
ccxt>=4.0.0
numpy>=1.24.0
pandas>=2.0.0
python-dotenv>=1.0.0
requests>=2.31.0
```

### 1.3 Create .env Template
**Issue:** Missing environment configuration
**Impact:** Bots cannot start - missing API keys and tokens
**File:** `/home/user/azure_bots_backup_20251209/.env.example`
**Fix:**
```bash
# Main Telegram Configuration
TELEGRAM_BOT_TOKEN=your_main_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Bot-specific Telegram Tokens (optional - fallback to main)
TELEGRAM_BOT_TOKEN_FUNDING=
TELEGRAM_BOT_TOKEN_LIQUIDATION=
TELEGRAM_BOT_TOKEN_VOLUME=
TELEGRAM_BOT_TOKEN_HARMONIC=
TELEGRAM_BOT_TOKEN_CANDLESTICK=
TELEGRAM_BOT_TOKEN_STRAT=
TELEGRAM_BOT_TOKEN_MTF=
TELEGRAM_BOT_TOKEN_PSAR=
TELEGRAM_BOT_TOKEN_DIY=
TELEGRAM_BOT_TOKEN_MOST=
TELEGRAM_BOT_TOKEN_FIB_REVERSAL=
TELEGRAM_BOT_TOKEN_FIB=
TELEGRAM_BOT_TOKEN_ORB=
TELEGRAM_BOT_TOKEN_CONSENSUS=
TELEGRAM_BOT_TOKEN_VOLUME_PROFILE=

# MEXC API Configuration (required for orb_bot)
MEXC_API_KEY=
MEXC_API_SECRET=
```

### 1.4 Fix Insecure Random Number Generation
**Issue:** notifier.py:19 - Using random instead of secrets
**Impact:** Signal IDs can be predicted/brute-forced
**File:** `/home/user/azure_bots_backup_20251209/notifier.py`
**Fix:**
```python
# Line 1: Add import
import secrets

# Line 19: Change from
random_part = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))

# To:
random_part = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(4))
```

### 1.5 Add API Key Validation
**Issue:** Multiple bots - Empty API keys passed to exchange
**Impact:** Unexpected behavior, potential security exposure
**Files:** All bot files loading API credentials
**Fix:** Add validation after loading credentials:
```python
api_key = os.getenv("MEXC_API_KEY", "")
api_secret = os.getenv("MEXC_API_SECRET", "")

if not api_key or not api_secret:
    raise ValueError("MEXC_API_KEY and MEXC_API_SECRET must be set in environment")
```

### 1.6 Fix Bare Except Clauses
**Issue:** analyze_toxic_breakdown.py:44 - Catches system exits
**Impact:** Difficult debugging, masks critical errors
**Files:** Multiple files
**Fix:**
```python
# Change all instances from:
except:
    print(f"Error: {e}")

# To:
except Exception as e:
    print(f"Error: {e}")
```

### 1.7 Replace Broad Exception Handling (160 instances)
**Issue:** All bot files - Catching generic Exception
**Impact:** Masks specific errors, makes debugging difficult
**Fix:** Replace with specific exceptions:
```python
# Instead of:
except Exception as e:
    logger.error(f"Failed: {e}")

# Use:
except (ccxt.NetworkError, ccxt.ExchangeError) as e:
    logger.error(f"Exchange API error: {e}")
except ValueError as e:
    logger.error(f"Invalid data: {e}")
```

### 1.8 Remove Duplicate Import
**Issue:** most_bot.py:19-20 - Duplicate typing import
**Impact:** Code clutter
**File:** `/home/user/azure_bots_backup_20251209/most_bot/most_bot.py`
**Fix:** Remove one of the duplicate import lines

---

## PHASE 2: HIGH PRIORITY FIXES (Days 2-3)

### 2.1 Configuration Management

#### 2.1.1 Add Missing Bots to global_config.json
**Files:** `global_config.json`
**Fix:** Add orb_bot, consensus_bot, volume_profile_bot configurations

#### 2.1.2 Resolve Watchlist Inconsistencies
**Issue:** Symbols differ between global_config.json and individual watchlist files
**Decision Required:** Choose single source of truth:
- Option A: Use global_config.json only
- Option B: Use individual watchlist files only
- Option C: Merge and sync both

#### 2.1.3 Standardize Watchlist Schema
**Issue:** Different schema between bots
**Fix:** Choose one schema and convert all watchlist files

### 2.2 Security Improvements

#### 2.2.1 Implement JSON Schema Validation
**Files:** All files loading JSON configurations
**Fix:**
```python
import jsonschema

watchlist_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["symbol", "period", "cooldown_minutes"],
        "properties": {
            "symbol": {"type": "string"},
            "period": {"type": "string"},
            "cooldown_minutes": {"type": "number"}
        }
    }
}

with open(path, 'r') as f:
    data = json.load(f)
    jsonschema.validate(data, watchlist_schema)
```

#### 2.2.2 Fix File Lock Permissions
**File:** `file_lock.py`
**Fix:** Set explicit permissions on lock files (0o600)

#### 2.2.3 Remove Hardcoded User Paths
**Files:** sync_bot_telemetry.py, SETUP_INSTRUCTIONS.md, START_HERE.txt
**Fix:** Replace `/home/behar/` with environment variables or relative paths

### 2.3 Code Quality

#### 2.3.1 Remove Duplicate Backoff Logic
**Files:** psar_bot.py:593-600, strat_bot.py
**Fix:** Consolidate duplicate backoff checks into single check

#### 2.3.2 Add Division by Zero Protection
**Files:** analyze_toxic_breakdown.py:78, analyze_toxic_backups.py:85, diy_bot.py:299
**Fix:** Add zero checks before division:
```python
win_rate = (wins / total_count * 100) if total_count > 0 else 0.0
```

### 2.4 Operational Improvements

#### 2.4.1 Add Graceful Shutdown Handlers to All Bots
**Files:** All bot files except volume_profile_bot
**Fix:**
```python
import signal
import sys

def signal_handler(sig, frame):
    logger.info(f"Received signal {sig}, shutting down gracefully...")
    bot.shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
```

#### 2.4.2 Implement HTTP Health Check Endpoints
**Files:** All bot files
**Fix:** Add Flask/FastAPI lightweight server:
```python
from flask import Flask, jsonify
import threading

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "bot": "funding_bot"}), 200

@app.route('/ready')
def ready():
    return jsonify({"ready": bot.is_ready()}), 200

# Run in separate thread
threading.Thread(target=lambda: app.run(port=8080), daemon=True).start()
```

#### 2.4.3 Ensure All API Calls Use RateLimitHandler
**Files:** All bot files
**Fix:** Audit all exchange API calls and wrap with RateLimitHandler

### 2.5 Documentation

#### 2.5.1 Create ARCHITECTURE.md
**File:** `/home/user/azure_bots_backup_20251209/ARCHITECTURE.md`
**Content:**
- System overview diagram
- Bot interaction flow
- Data flow diagrams
- State management strategy
- Deployment architecture

---

## PHASE 3: MEDIUM PRIORITY FIXES (Days 4-5)

### 3.1 Logging & Monitoring

#### 3.1.1 Standardize Logging Configuration
**Fix:** Create shared logging module:
```python
# shared_logging.py
import logging

def setup_logger(name: str, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler with rotation
    from logging.handlers import RotatingFileHandler
    fh = RotatingFileHandler(
        f'logs/{name}.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
```

#### 3.1.2 Add Request/Response Logging
**File:** rate_limit_handler.py
**Fix:** Add DEBUG level logging for all API calls

#### 3.1.3 Implement Structured Logging
**Fix:** Use python-json-logger for JSON-formatted logs

#### 3.1.4 Add Correlation IDs
**Fix:** Generate UUID for each signal and include in all related logs

### 3.2 Error Handling

#### 3.2.1 Standardize Error Messages
**Fix:** Create error message templates:
```python
def format_error(component: str, action: str, reason: str) -> str:
    return f"{component}: {action} failed - {reason}"
```

#### 3.2.2 Add Error Context
**Fix:** Include symbol, timeframe, state in error logs:
```python
logger.error(
    f"Failed to collect data for {symbol}",
    extra={"symbol": symbol, "timeframe": timeframe, "bot": "funding"}
)
```

### 3.3 State Management

#### 3.3.1 Implement Atomic File Writes
**Files:** All state file operations
**Fix:**
```python
import tempfile, shutil

def atomic_write(file_path, data):
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        json.dump(data, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
    shutil.move(tmp.name, file_path)
```

#### 3.3.2 Add File Locking for stats.json
**File:** signal_stats.py
**Fix:** Use file_lock utility for all stats operations

### 3.4 Configuration

#### 3.4.1 Update Bots to Use global_config.json
**Files:** All bot files
**Fix:** Use TradeConfigManager instead of direct watchlist loading

#### 3.4.2 Create Comprehensive .env.example
**Fix:** Document all environment variables with examples

#### 3.4.3 Standardize Cooldown Configuration
**Fix:** Choose consistent cooldown values across bots

### 3.5 Resource Management

#### 3.5.1 Convert to Context Managers
**Files:** 46 instances of open() without 'with'
**Fix:** Convert all to context managers

#### 3.5.2 Close Exchange Connections
**Fix:** Add to shutdown handlers:
```python
if hasattr(self.exchange, 'close'):
    await self.exchange.close()
```

### 3.6 Security

#### 3.6.1 Implement Log Rotation
**Fix:** Use RotatingFileHandler with size limits

#### 3.6.2 Sanitize Sensitive Data in Logs
**Fix:** Filter API keys, tokens from error messages

#### 3.6.3 Set File Permissions Explicitly
**Fix:** Set 0o600 on all state files, logs, lock files

#### 3.6.4 Add Response Validation
**Fix:** Validate all API responses before processing

### 3.7 Documentation

#### 3.7.1 Add Docstrings to All Functions
**Target:** 40% coverage → 100%
**Fix:** Add comprehensive docstrings:
```python
def calculate_volume_profile(highs, lows, closes, volumes, num_rows=24):
    """Calculate volume profile with POC, VAH, VAL.

    Args:
        highs: List of high prices for each candle
        lows: List of low prices for each candle
        closes: List of close prices for each candle
        volumes: List of volumes for each candle
        num_rows: Number of price levels to divide range into

    Returns:
        Dict containing poc, vah, val, hvn_levels, and volume_profile
    """
```

#### 3.7.2 Document Complex Algorithms
**Files:** volume_profile.py, tp_sl_calculator.py
**Fix:** Add detailed comments explaining methodology

#### 3.7.3 Create Bot-Specific READMEs
**Fix:** Create README.md for each bot (currently only 3/13 have them)

### 3.8 Type Safety

#### 3.8.1 Add Type Hints to All Functions
**Target:** 45% coverage → 100%
**Fix:**
```python
from typing import List, Dict, Optional

def load_watchlist() -> List[WatchItem]:
    """Load watchlist from JSON file."""
    ...

def can_alert(self, symbol: str, cooldown_minutes: int) -> bool:
    """Check if alert can be sent."""
    ...
```

#### 3.8.2 Create mypy Configuration
**File:** mypy.ini
**Fix:**
```ini
[mypy]
python_version = 3.9
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
```

---

## PHASE 4: LOW PRIORITY FIXES (Days 6-7)

### 4.1 Code Cleanup

#### 4.1.1 Remove Commented Code (117 instances)
**Fix:** Delete all commented-out code blocks

#### 4.1.2 Remove Unused Imports
**Fix:** Run autoflake or pylint to detect and remove

#### 4.1.3 Remove Unused Functions
**Fix:** Use code coverage analysis to identify

### 4.2 Code Style

#### 4.2.1 Standardize Naming Conventions
**Fix:** Run pylint/flake8, fix PEP 8 violations

#### 4.2.2 Replace Magic Numbers with Constants
**Fix:** Define module-level constants

#### 4.2.3 Fix Inconsistent Error Formatting
**Fix:** Use standard format throughout

### 4.3 Configuration

#### 4.3.1 Create Systemd Service Files for All Bots
**Fix:** Create .service files for 12 missing bots

#### 4.3.2 Pin Dependency Versions
**Fix:** Update requirements.txt with exact versions

#### 4.3.3 Standardize State File Locations
**Fix:** Use consistent naming and location

### 4.4 Security (Minor)

#### 4.4.1 Increase Signal ID Entropy
**Fix:** Increase from 4 to 8-12 characters

#### 4.4.2 Expand .gitignore Patterns
**Fix:** Add additional secret file patterns

#### 4.4.3 Set Directory Permissions
**Fix:** Add mode=0o700 to mkdir calls

#### 4.4.4 Add Config File Integrity Checks
**Fix:** Implement HMAC-based verification

### 4.5 Documentation

#### 4.5.1 Create Comprehensive README.md
**Fix:** Add sections for:
- Quick start
- Architecture
- Bot catalog
- Configuration guide
- Development guide
- Troubleshooting

#### 4.5.2 Create CHANGELOG.md
**Fix:** Track version history

#### 4.5.3 Remove Hardcoded Paths from Documentation
**Fix:** Use $PROJECT_ROOT placeholders

### 4.6 Monitoring

#### 4.6.1 Add Metrics Endpoint
**Fix:** Implement Prometheus metrics

#### 4.6.2 Create Monitoring Dashboard
**Fix:** Set up Grafana dashboard

---

## PHASE 5: TESTING & CI/CD (Days 8-10)

### 5.1 Create Test Suite (CRITICAL)
**Issue:** No tests exist - 0% coverage
**Impact:** Cannot refactor safely, high regression risk

#### 5.1.1 Create Test Structure
```
tests/
├── unit/
│   ├── test_tp_sl_calculator.py
│   ├── test_signal_stats.py
│   ├── test_rate_limit_handler.py
│   ├── test_volume_profile.py
│   ├── test_notifier.py
│   └── test_file_lock.py
├── integration/
│   ├── test_funding_bot.py
│   ├── test_liquidation_bot.py
│   ├── test_volume_bot.py
│   ├── test_harmonic_bot.py
│   ├── test_candlestick_bot.py
│   ├── test_psar_bot.py
│   ├── test_most_bot.py
│   ├── test_mtf_bot.py
│   ├── test_diy_bot.py
│   ├── test_strat_bot.py
│   ├── test_fib_reversal_bot.py
│   ├── test_fib_swing_bot.py
│   ├── test_orb_bot.py
│   ├── test_consensus_bot.py
│   └── test_volume_profile_bot.py
├── fixtures/
│   ├── mock_ohlcv.json
│   ├── mock_ticker.json
│   ├── mock_funding_rate.json
│   └── mock_liquidations.json
└── conftest.py
```

#### 5.1.2 Priority Test Coverage
1. **TP/SL Calculation** (tp_sl_calculator.py)
   - Test ATR calculations
   - Test TP1/TP2/TP3 levels
   - Test edge cases (zero ATR, negative prices)

2. **Volume Profile** (volume_profile.py)
   - Test POC calculation
   - Test VAH/VAL calculation
   - Test with various price ranges

3. **Signal Detection Logic**
   - Test each bot's signal detection
   - Test pattern recognition
   - Test indicator calculations

4. **State Management** (file_lock.py)
   - Test atomic writes
   - Test file locking
   - Test concurrent access

5. **Rate Limiting** (rate_limit_handler.py)
   - Test backoff logic
   - Test retry mechanism
   - Test rate limit detection

#### 5.1.3 Mock Exchange API
```python
# tests/conftest.py
import pytest
from unittest.mock import Mock

@pytest.fixture
def mock_exchange():
    exchange = Mock()
    exchange.fetch_ohlcv.return_value = [
        [1234567890000, 100, 105, 98, 102, 1000],  # timestamp, o, h, l, c, v
        # ... more data
    ]
    exchange.fetch_ticker.return_value = {
        'bid': 100, 'ask': 101, 'last': 100.5
    }
    return exchange
```

#### 5.1.4 Run Tests
```bash
pytest tests/ --cov=. --cov-report=html --cov-report=term
```

### 5.2 Set Up CI/CD Pipeline

#### 5.2.1 Create GitHub Actions Workflow
**File:** `.github/workflows/tests.yml`
```yaml
name: Tests and Linting

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pylint flake8 mypy bandit

    - name: Run tests
      run: pytest --cov=. --cov-report=xml --cov-report=term

    - name: Lint with pylint
      run: pylint **/*.py || true

    - name: Check with flake8
      run: flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

    - name: Type check with mypy
      run: mypy . || true

    - name: Security scan with bandit
      run: bandit -r . -ll

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

#### 5.2.2 Add Pre-commit Hooks
**File:** `.pre-commit-config.yaml`
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
      - id: black

  - repo: https://github.com/PyCQA/flake8
    rev: 6.1.0
    hooks:
      - id: flake8

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ['-ll']
```

### 5.3 Load/Stress Testing

#### 5.3.1 Create Performance Tests
**File:** `tests/performance/test_load.py`
```python
import time
from locust import HttpUser, task, between

class BotHealthCheck(HttpUser):
    wait_time = between(1, 3)

    @task
    def health_check(self):
        self.client.get("/health")
```

---

## PHASE 6: ADVANCED IMPROVEMENTS (Days 11-14)

### 6.1 Centralized Configuration
- Migrate all bots to use global_config.json
- Remove individual watchlist files or sync them
- Create configuration management UI

### 6.2 Enhanced Monitoring
- Implement Prometheus metrics export
- Set up Grafana dashboards
- Add alerting rules
- Implement distributed tracing

### 6.3 Rate Limit Coordination
- Implement shared rate limiter using Redis
- Coordinate API calls across all bots
- Add global backoff mechanism

### 6.4 Enhanced Security
- Encrypt state files at rest
- Implement certificate pinning for exchange APIs
- Add audit logging for all critical operations
- Implement secrets management (HashiCorp Vault)

### 6.5 Advanced Logging
- Implement structured JSON logging
- Set up centralized log aggregation (ELK stack)
- Add log analysis and anomaly detection

---

## Implementation Order

### Week 1: Critical & High Priority
**Days 1-2:** Phase 1 (Critical Fixes)
- Fix runtime bugs
- Create requirements.txt and .env.example
- Fix security vulnerabilities
- Replace broad exception handling

**Days 3-4:** Phase 2 (High Priority)
- Configuration management
- Security improvements
- Add graceful shutdown
- HTTP health checks
- Documentation

### Week 2: Medium Priority & Testing
**Days 5-6:** Phase 3 (Medium Priority)
- Logging standardization
- Error handling improvements
- State management fixes
- Type hints
- Resource management

**Days 7-8:** Phase 4 (Low Priority)
- Code cleanup
- Style fixes
- Minor security improvements

**Days 9-10:** Phase 5 (Testing & CI/CD)
- Create comprehensive test suite
- Set up CI/CD pipeline
- Achieve >80% code coverage

### Week 3: Advanced (Optional)
**Days 11-14:** Phase 6 (Advanced Improvements)
- Centralized configuration
- Enhanced monitoring
- Advanced security
- Performance optimization

---

## Success Metrics

### Code Quality
- ✅ 0 critical bugs
- ✅ 0 duplicate code blocks
- ✅ 100% PEP 8 compliance
- ✅ 100% type hint coverage

### Security
- ✅ 0 critical/high vulnerabilities
- ✅ All secrets in environment variables
- ✅ All sensitive files encrypted/protected
- ✅ Security scan passing in CI

### Testing
- ✅ >80% code coverage
- ✅ All critical paths tested
- ✅ Integration tests for all bots
- ✅ CI/CD pipeline green

### Documentation
- ✅ 100% public API documented
- ✅ Architecture documentation complete
- ✅ All bots have README files
- ✅ Setup guide tested and working

### Operations
- ✅ All bots have health check endpoints
- ✅ Graceful shutdown implemented
- ✅ Monitoring dashboard operational
- ✅ Automated deployment working

---

## Risk Assessment

### High Risk Items
1. **Changing exception handling** - May expose hidden bugs
2. **Migrating configuration** - Could break existing bots
3. **Adding tests** - May reveal existing issues
4. **Rate limit changes** - Could affect trading performance

### Mitigation Strategies
1. Test in staging environment first
2. Deploy changes incrementally
3. Keep rollback plan ready
4. Monitor closely after each change
5. Maintain backwards compatibility where possible

---

## Rollout Strategy

### Development Environment
1. Create feature branch for each phase
2. Test thoroughly in dev
3. Run full test suite
4. Security scan

### Staging Environment
1. Deploy to staging
2. Run integration tests
3. Monitor for 24 hours
4. Performance testing

### Production Environment
1. Deploy during low-traffic hours
2. Deploy one bot at a time
3. Monitor metrics closely
4. Keep previous version ready for rollback
5. Full rollback if any critical issues

---

## Post-Implementation Maintenance

### Weekly Tasks
- Review error logs
- Check health check status
- Update dependencies
- Review performance metrics

### Monthly Tasks
- Security audit
- Dependency vulnerability scan
- Performance optimization review
- Documentation updates

### Quarterly Tasks
- Full system audit
- Load testing
- Disaster recovery drill
- Technology stack review

---

## Appendix: Quick Reference

### Critical Files to Fix First
1. `/home/user/azure_bots_backup_20251209/strat_bot/strat_bot.py` (Line 642)
2. `/home/user/azure_bots_backup_20251209/notifier.py` (Line 19)
3. `/home/user/azure_bots_backup_20251209/most_bot/most_bot.py` (Lines 19-20)
4. `/home/user/azure_bots_backup_20251209/analyze_toxic_breakdown.py` (Line 44)

### Commands to Run After Fixes
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ --cov=. --cov-report=html

# Lint code
pylint **/*.py

# Type check
mypy .

# Security scan
bandit -r . -ll

# Format code
black .

# Start all bots
./start_all_bots.sh

# Check status
./check_bots_status.sh
```

### Monitoring URLs (After Implementation)
- Funding Bot Health: http://localhost:8001/health
- Liquidation Bot Health: http://localhost:8002/health
- Volume Bot Health: http://localhost:8003/health
- [... all bots ...]
- Grafana Dashboard: http://localhost:3000
- Prometheus Metrics: http://localhost:9090

---

**Document Version:** 1.0
**Last Updated:** 2025-12-14
**Next Review:** After Phase 1 completion
