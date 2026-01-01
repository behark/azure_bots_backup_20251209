#!/usr/bin/env python3
"""
A comprehensive trading bot that analyzes 30+ technical indicators across multiple markets
to identify high-confluence trading signals. Features include:

CORE FEATURES:
- Multi-indicator confluence analysis (RSI, MACD, ADX, Bollinger Bands, EMA, Volume, etc.)
- Dynamic market-condition weighting (trending vs. ranging markets)
- ATR-based stop loss and take profit calculations
- Batch ticker fetching for efficient TP/SL monitoring
- Signal persistence and trade tracking with statistics

PERFORMANCE OPTIMIZATIONS:
- O(n) rolling standard deviation calculation using cumulative sums (HIGH PRIORITY FIX)
- API response caching with TTL (OHLCV 60s, Ticker 5s) to reduce API calls
- Batch ticker fetches for monitoring multiple signals in 1 API call instead of N
- Rate limiting and backoff handling for exchange API limits

RISK MANAGEMENT:
- Maximum open signals limit (configurable)
- Per-symbol signal limits to prevent overexposure
- Minimum confidence threshold filtering
- Risk/reward ratio validation
- Price tolerance for TP/SL level matching

SIGNAL TRACKING:
- Persistent signal state with thread-safe operations and atomic file writes
- Open signal archival to statistics on startup (prevents loss after restart)
- Stale signal cleanup with automatic archival to stats
- Result notification cooldown to prevent spam

MONITORING & ALERTS:
- Telegram notifications for new signals and trade results
- Health monitoring with heartbeat messages (no-signal detection)
- Signal statistics tracking (win rate, P&L, TP/SL counts per symbol)
- Detailed performance history in result messages

INDICATORS ANALYZED:
1. RSI (Relative Strength Index) - momentum/overbought-oversold
2. MACD - trend following
3. EMA Crossover (9/21, 50/200) - trend direction
4. ADX/DMI - trend strength
5. Bollinger Bands - volatility and support/resistance
6. Volume Analysis - volume surge confirmation
7. Momentum (5-period) - price change rate
8. Stochastic RSI - timing/oscillator
9. TrendPosition - price vs major EMAs (50/200)
10+ Additional indicators via analyze_indicators()

CONFIG STRUCTURE:
- exchange_settings: Active exchange and duplicate checking
- execution: Cycle interval, logging, notifications
- risk: Max signals, risk/reward ratios, per-symbol limits
- analysis: Indicator parameters and market-condition weights
- tp_sl: ATR-based TP/SL calculation settings
- rate_limit: API throttling and backoff
- signal: Cooldowns and age limits
- telegram: Notification settings
- cache: TTL for OHLCV and ticker data

CRITICAL FIXES:
1. Bounds checking in indicator calculations (RSI, Momentum)
2. Vectorized rolling std for O(n) instead of O(n²) complexity
3. Look-ahead bias prevention by excluding incomplete candles
4. Batch ticker fetching to reduce API calls
5. Thread-safe state persistence with atomic writes
6. Safe division with explicit zero checks
7. Signal syncing to stats on startup for persistence
8. Price tolerance for TP/SL matching to account for slippage

USAGE:
    python diy_bot.py              # Loop indefinitely
    python diy_bot.py --once       # Single analysis cycle
    python diy_bot.py --track      # Track-only: monitor open signals
    python diy_bot.py --debug      # Debug logging enabled
    python diy_bot.py --validate   # Validate environment

ENVIRONMENT VARIABLES:
    TELEGRAM_BOT_TOKEN_DIY    : Telegram bot token (required)
    TELEGRAM_CHAT_ID          : Telegram chat ID (required)
    MEXC_API_KEY / MEXC_SECRET         : MEXC exchange credentials
    BINANCEUSDM_API_KEY / BINANCEUSDM_SECRET : Binance USDM credentials
    BYBIT_API_KEY / BYBIT_SECRET       : Bybit exchange credentials
"""

from __future__ import annotations

import argparse
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    # Windows doesn't have fcntl - use a no-op fallback
    HAS_FCNTL = False
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import FrameType
from typing import Any, Dict, List, Optional, Tuple, TypedDict, cast

import ccxt  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

# Graceful shutdown handling
shutdown_requested = False


def signal_handler(signum: int, frame: Optional[FrameType]) -> None:
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    sig_name = signal.Signals(signum).name
    logger.info(f"Received {sig_name}, initiating graceful shutdown...")
    shutdown_requested = True

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
STATE_FILE = BASE_DIR / "diy_state.json"
WATCHLIST_FILE = BASE_DIR / "diy_watchlist.json"
CONFIG_FILE = BASE_DIR / "diy_config.json"
STATS_FILE = LOG_DIR / "diy_stats.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# Setup enhanced logging with rotation
def setup_logging(log_level: str = "INFO", enable_detailed: bool = False) -> logging.Logger:
    """Setup enhanced logging with rotation."""
    detailed_format = "%(asctime)s | %(levelname)-8s | [%(name)s:%(funcName)s:%(lineno)d] | %(message)s"
    simple_format = "%(asctime)s | %(levelname)-8s | %(message)s"

    formatter = logging.Formatter(detailed_format if enable_detailed else simple_format)

    # Get logger and clear existing handlers to prevent duplicates
    logger = logging.getLogger("diy_bot")
    logger.handlers.clear()  # Fix: Clear existing handlers before adding new ones

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Rotating file handlers
    file_handler = RotatingFileHandler(
        LOG_DIR / "diy_bot.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)

    error_handler = RotatingFileHandler(
        LOG_DIR / "diy_errors.log",
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    # Configure logger
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)

    return logger

# Initialize logger (will be reconfigured after loading config)
logger = setup_logging()

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
    load_dotenv(BASE_DIR.parent / ".env")
except ImportError:
    pass

if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))

# Required imports (fail fast if missing)
from message_templates import format_signal_message, format_result_message
from notifier import TelegramNotifier
from signal_stats import SignalStats
from tp_sl_calculator import TPSLCalculator
from trade_config import get_config_manager

# NEW: Import unified signal system
from core.bot_signal_mixin import BotSignalMixin, create_price_fetcher

# Optional imports (safe fallback)
from safe_import import safe_import
HealthMonitor = safe_import('health_monitor', 'HealthMonitor')
RateLimiter = None  # Disabled for testing - rate_limit_handler has different API


class WatchItem(TypedDict, total=False):
    symbol: str
    timeframe: str
    exchange: str
    market_type: str


def validate_environment() -> bool:
    """Validate all required environment variables are set (from Volume Bot)."""
    errors = []

    # Check for Telegram configuration
    token = os.getenv("TELEGRAM_BOT_TOKEN_DIY") or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token:
        errors.append("Missing TELEGRAM_BOT_TOKEN_DIY or TELEGRAM_BOT_TOKEN")
    elif ":" not in token:
        errors.append("Invalid Telegram bot token format")

    if not chat_id:
        errors.append("Missing TELEGRAM_CHAT_ID")
    elif not (chat_id.startswith("-") or chat_id.lstrip("-").isdigit()):
        errors.append("Invalid TELEGRAM_CHAT_ID format")

    # Check for at least one exchange credential
    exchanges_found = []
    for exchange in ["MEXC", "BINANCEUSDM", "BYBIT"]:
        api_key = os.getenv(f"{exchange}_API_KEY")
        # Check both naming conventions: _SECRET and _API_SECRET
        secret = os.getenv(f"{exchange}_SECRET") or os.getenv(f"{exchange}_API_SECRET")
        if api_key and secret:
            exchanges_found.append(exchange)
            # Validate API key format
            if len(api_key) < 10:
                errors.append(f"Invalid {exchange}_API_KEY format (too short)")

    if not exchanges_found:
        errors.append("No exchange credentials configured (need at least one: MEXC_API_KEY/SECRET, BINANCEUSDM_API_KEY/SECRET, or BYBIT_API_KEY/SECRET)")

    if errors:
        logger.error("Environment validation failed:")
        for error in errors:
            logger.error("  - %s", error)
        return False

    logger.info("Environment validation passed. Configured exchanges: %s", ", ".join(exchanges_found))
    return True


def get_default_config() -> Dict[str, Any]:
    """Return complete default configuration with all required fields."""
    return {
        "exchange_settings": {
            "active_exchange": "binanceusdm",
            "check_exchange_for_duplicates": True,
            "check_timeframe_for_duplicates": True,
        },
        "execution": {
            "cycle_interval_seconds": 60,
            "symbol_delay_seconds": 1,
            "health_check_interval_seconds": 3600,
            "enable_startup_notification": True,
            "enable_shutdown_notification": True,
            "log_level": "INFO",
            "enable_detailed_logging": False,
        },
        "risk": {
            "max_open_signals": 25,
            "min_risk_reward_ratio": 1.95,
            "max_risk_per_trade_pct": 1.5,
            "enable_stop_loss": True,
            "max_same_symbol_signals": 2,
        },
        "analysis": {
            "candle_limit": 200,
            "request_timeout_seconds": 30,
            "min_confidence_threshold": 40.0,
            "indicator_weights": {
                "trending_market": {
                    "RSI": 1.0, "MACD": 2.0, "EMA": 2.0, "ADX": 1.5,
                    "BollingerBands": 0.5, "Volume": 1.0, "Momentum": 1.0,
                    "StochRSI": 1.0, "TrendPosition": 2.0,
                },
                "ranging_market": {
                    "RSI": 2.0, "MACD": 1.0, "EMA": 0.5, "ADX": 0.5,
                    "BollingerBands": 2.0, "Volume": 1.0, "Momentum": 2.0,
                    "StochRSI": 1.5, "TrendPosition": 0.5,
                },
            },
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "ema_fast": 9,
            "ema_slow": 21,
            "ema_trend_fast": 50,
            "ema_trend_slow": 200,
            "adx_period": 14,
            "adx_threshold": 20,
            "adx_trending": 25,
            "bb_period": 20,
            "bb_std_dev": 2.0,
            "volume_sma_period": 20,
            "volume_surge_multiplier": 1.2,
            "stoch_rsi_period": 14,
            "stoch_rsi_smooth": 3,
        },
        "tp_sl": {
            "use_tpsl_calculator": True,
            "calculation_method": "ATR",
            "atr_period": 14,
            "atr_sl_multiplier": 1.0,
            "atr_tp1_multiplier": 2.0,
            "atr_tp2_multiplier": 3.0,
            "min_risk_reward_tp1": 1.95,
            "min_risk_reward_tp2": 2.9,
            "price_tolerance": 0.005,
        },
        "rate_limit": {
            "calls_per_minute": 60,
            "max_retries": 5,
            "base_delay_seconds": 0.5,
            "backoff_multiplier": 2.0,
            "rate_limit_backoff_base": 60,
            "rate_limit_backoff_max": 300,
        },
        "signal": {
            "cooldown_minutes": 10,
            "max_signal_age_hours": 24,
            "result_notification_cooldown_minutes": 15,
            "no_signal_alert_hours": 6,
        },
        "telegram": {
            "bot_token_env_var": "TELEGRAM_BOT_TOKEN_DIY",
            "chat_id_env_var": "TELEGRAM_CHAT_ID",
            "enabled": True,
        },
        "cache": {
            "ohlcv_ttl_seconds": 60,
            "ticker_ttl_seconds": 5,
        },
    }


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def validate_config(config: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
    """
    Validate configuration and return merged config with warnings.

    HIGH PRIORITY FIX: Comprehensive config validation with defaults.

    Returns:
        Tuple of (validated_config, list_of_warnings)
    """
    warnings = []
    defaults = get_default_config()

    # Merge with defaults
    validated = deep_merge(defaults, config)

    # Validate specific ranges and types
    validations = [
        # (path, min_val, max_val, description)
        (("execution", "cycle_interval_seconds"), 10, 3600, "Cycle interval"),
        (("analysis", "candle_limit"), 50, 1000, "Candle limit"),
        (("analysis", "min_confidence_threshold"), 0, 100, "Confidence threshold"),
        (("risk", "max_open_signals"), 1, 100, "Max open signals"),
        (("risk", "min_risk_reward_ratio"), 0.5, 10.0, "Min risk/reward"),
        (("tp_sl", "atr_sl_multiplier"), 0.1, 5.0, "ATR SL multiplier"),
        (("tp_sl", "atr_tp1_multiplier"), 0.5, 10.0, "ATR TP1 multiplier"),
        (("tp_sl", "atr_tp2_multiplier"), 1.0, 20.0, "ATR TP2 multiplier"),
        (("rate_limit", "calls_per_minute"), 1, 300, "Calls per minute"),
        (("signal", "cooldown_minutes"), 1, 1440, "Signal cooldown"),
    ]

    for path, min_val, max_val, desc in validations:
        # Navigate to the value
        current = validated
        for key in path[:-1]:
            current = current.get(key, {})
        value = current.get(path[-1])

        if value is not None:
            try:
                num_value = float(value)
                if num_value < min_val or num_value > max_val:
                    warnings.append(f"{desc} ({'.'.join(path)}) = {value} is outside range [{min_val}, {max_val}]")
                    # Clamp to range
                    current[path[-1]] = max(min_val, min(max_val, num_value))
            except (ValueError, TypeError):
                warnings.append(f"{desc} ({'.'.join(path)}) = {value} is not a valid number")

    # Validate log level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    log_level = validated.get("execution", {}).get("log_level", "INFO")
    if log_level.upper() not in valid_levels:
        warnings.append(f"Invalid log_level '{log_level}', using INFO")
        validated["execution"]["log_level"] = "INFO"

    # Validate TP multipliers order (TP2 should be > TP1)
    tp_sl = validated.get("tp_sl", {})
    if tp_sl.get("atr_tp2_multiplier", 3.0) <= tp_sl.get("atr_tp1_multiplier", 2.0):
        warnings.append("atr_tp2_multiplier should be greater than atr_tp1_multiplier")

    return validated, warnings


def load_config(config_path: Path = CONFIG_FILE) -> Dict[str, Any]:
    """Load and validate configuration from JSON file."""
    if not config_path.exists():
        logger.warning("Config file missing: %s. Using defaults.", config_path)
        return get_default_config()

    try:
        raw_config = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        logger.error("Invalid config JSON: %s. Using defaults.", exc)
        return get_default_config()

    # HIGH PRIORITY FIX: Validate and merge with defaults
    validated, warnings = validate_config(raw_config)

    if warnings:
        logger.warning("Config validation warnings:")
        for warning in warnings:
            logger.warning("  - %s", warning)

    return validated


def load_watchlist() -> List[WatchItem]:
    """Load watchlist from JSON file."""
    if not WATCHLIST_FILE.exists():
        logger.error("Watchlist file missing: %s", WATCHLIST_FILE)
        return []

    try:
        data = json.loads(WATCHLIST_FILE.read_text())
    except json.JSONDecodeError as exc:
        logger.error("Invalid watchlist JSON: %s", exc)
        return []

    normalized: List[WatchItem] = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue

        # Handle both old and new format
        symbol_val = item.get("symbol")
        if not isinstance(symbol_val, str):
            continue

        # Ensure symbol has /USDT format
        if "/" not in symbol_val:
            symbol_val = f"{symbol_val}/USDT"

        # Get timeframe (support old 'period' field)
        timeframe_val = item.get("timeframe") or item.get("period", "5m")
        exchange_val = item.get("exchange", "binanceusdm")
        market_type_val = item.get("market_type", "swap")

        normalized.append({
            "symbol": symbol_val.upper(),
            "timeframe": timeframe_val if isinstance(timeframe_val, str) else "5m",
            "exchange": exchange_val if isinstance(exchange_val, str) else "mexc",
            "market_type": market_type_val if isinstance(market_type_val, str) else "swap",
        })

    return normalized


def human_ts() -> str:
    """Return human-readable UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class MultiIndicatorAnalyzer:
    """Analyzes 30+ indicators for confluence."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("analysis", {})
        self.indicator_weights = self.config.get("indicator_weights", {})

    # Core calculation methods
    @staticmethod
    def calculate_ema(prices: npt.NDArray[np.floating[Any]], period: int) -> npt.NDArray[np.floating[Any]]:
        """Calculate EMA with proper initialization to avoid false crossover signals."""
        n = len(prices)
        ema = np.full(n, np.nan)  # Use NaN for invalid/uninitialized values
        if n < period:
            return ema
        multiplier = 2 / (period + 1)
        # Initialize with SMA of first 'period' values
        ema[period-1] = np.mean(prices[:period])
        for i in range(period, n):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
        # Fill initial values with expanding mean to avoid NaN comparisons
        for i in range(period - 1):
            ema[i] = np.mean(prices[:i+1])
        return ema

    @staticmethod
    def _calculate_rolling_std(values: npt.NDArray[np.floating[Any]], period: int) -> npt.NDArray[np.floating[Any]]:
        """
        Calculate rolling standard deviation efficiently using cumulative sums.

        HIGH PRIORITY FIX: O(n) complexity instead of O(n²).

        Uses the formula: Var(X) = E[X²] - E[X]²
        """
        n = len(values)
        if n < period:
            return np.full(n, np.std(values) if n > 0 else 0.0)

        # Pad for cumsum calculation
        padded = np.concatenate([[0], values])
        padded_sq = np.concatenate([[0], values ** 2])

        # Cumulative sums
        cumsum = np.cumsum(padded)
        cumsum_sq = np.cumsum(padded_sq)

        # Rolling sum and sum of squares
        rolling_sum = cumsum[period:] - cumsum[:-period]
        rolling_sum_sq = cumsum_sq[period:] - cumsum_sq[:-period]

        # Variance = E[X²] - E[X]²
        rolling_mean = rolling_sum / period
        rolling_mean_sq = rolling_sum_sq / period
        variance = rolling_mean_sq - rolling_mean ** 2

        # Handle floating point errors that might make variance slightly negative
        variance = np.maximum(variance, 0)
        rolling_std = np.sqrt(variance)

        # Pad the beginning with the first valid std value
        result = np.zeros(n)
        result[period-1:] = rolling_std
        # Fill initial values with expanding window std
        for i in range(period - 1):
            result[i] = np.std(values[:i+1]) if i > 0 else 0.0

        return result

    def calculate_rsi(self, closes: npt.NDArray[np.floating[Any]], period: Optional[int] = None) -> npt.NDArray[np.floating[Any]]:
        """Calculate RSI with proper bounds checking."""
        if period is None:
            period = self.config.get("rsi_period", 14)

        # CRITICAL FIX: Bounds check - need at least period+1 data points
        if len(closes) <= period:
            logger.debug("RSI: Insufficient data (%d <= %d), returning neutral", len(closes), period)
            return np.full(len(closes), 50.0)  # Return neutral RSI

        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.zeros(len(closes))
        avg_loss = np.zeros(len(closes))

        # Safe indexing after bounds check
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])

        for i in range(period + 1, len(closes)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period

        with np.errstate(divide='ignore', invalid='ignore'):
            rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
            rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_stoch_rsi(self, closes: npt.NDArray[np.floating[Any]]) -> Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
        """Calculate Stochastic RSI (K, D)."""
        period = self.config.get("stoch_rsi_period", 14)
        smooth = self.config.get("stoch_rsi_smooth", 3)

        rsi = self.calculate_rsi(closes, period)
        stoch = np.zeros(len(rsi))
        for i in range(period, len(rsi)):
            window = rsi[i - period + 1 : i + 1]
            min_rsi = np.min(window)
            max_rsi = np.max(window)
            denom = max(max_rsi - min_rsi, 1e-9)
            stoch[i] = (rsi[i] - min_rsi) / denom * 100

        def smooth_values(values: npt.NDArray[np.floating[Any]], length: int = 3) -> npt.NDArray[np.floating[Any]]:
            if len(values) < length:
                return values
            kernel = np.ones(length) / length
            smoothed = np.convolve(values, kernel, mode="same")
            return smoothed

        k_line = smooth_values(stoch, smooth)
        d_line = smooth_values(k_line, smooth)
        return k_line, d_line

    def calculate_macd(self, closes: npt.NDArray[np.floating[Any]]) -> Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
        """Calculate MACD."""
        fast_period = self.config.get("macd_fast", 12)
        slow_period = self.config.get("macd_slow", 26)
        signal_period = self.config.get("macd_signal", 9)

        ema_fast = self.calculate_ema(closes, fast_period)
        ema_slow = self.calculate_ema(closes, slow_period)
        macd = ema_fast - ema_slow
        signal = self.calculate_ema(macd, signal_period)
        return macd, signal

    def calculate_adx(self, highs: npt.NDArray[np.floating[Any]], lows: npt.NDArray[np.floating[Any]], closes: npt.NDArray[np.floating[Any]]) -> Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
        """Calculate ADX and DI."""
        period = self.config.get("adx_period", 14)

        plus_dm = np.zeros(len(closes))
        minus_dm = np.zeros(len(closes))

        for i in range(1, len(closes)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move

        # Calculate ATR for normalization
        tr = np.zeros(len(closes))
        tr[0] = highs[0] - lows[0]
        for i in range(1, len(closes)):
            tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))

        atr = self.calculate_ema(tr, period)
        plus_di = 100 * self.calculate_ema(plus_dm, period) / np.where(atr != 0, atr, 1)
        minus_di = 100 * self.calculate_ema(minus_dm, period) / np.where(atr != 0, atr, 1)

        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, plus_di + minus_di, 1)
        adx = self.calculate_ema(dx, period)

        return adx, plus_di, minus_di

    def analyze_indicators(
        self,
        highs: npt.NDArray[np.floating[Any]],
        lows: npt.NDArray[np.floating[Any]],
        closes: npt.NDArray[np.floating[Any]],
        volumes: npt.NDArray[np.floating[Any]],
    ) -> Dict[str, Dict[str, object]]:
        """Analyze all indicators and return signals."""
        results = {}

        # 1. RSI
        rsi = self.calculate_rsi(closes)
        rsi_val = rsi[-1]
        if rsi_val < 30:
            rsi_signal = 'LONG'
            rsi_strength = 100.0
        elif rsi_val > 70:
            rsi_signal = 'SHORT'
            rsi_strength = 100.0
        elif 50 < rsi_val < 70:
            rsi_signal = 'LONG'
            rsi_strength = float((rsi_val - 50) * 3)
        elif 30 < rsi_val < 50:
            rsi_signal = 'SHORT'
            rsi_strength = float((50 - rsi_val) * 3)
        else:
            rsi_signal = 'NEUTRAL'
            rsi_strength = 20.0
        results['RSI'] = {
            'signal': rsi_signal,
            'strength': max(20.0, min(rsi_strength, 100.0)),
            'value': rsi_val
        }

        # 2. MACD
        macd, signal = self.calculate_macd(closes)
        results['MACD'] = {
            'signal': 'LONG' if macd[-1] > signal[-1] else 'SHORT',
            'strength': min(abs(macd[-1] - signal[-1]) * 1000, 100),
            'value': macd[-1]
        }

        # 3. EMA Crossover
        ema_fast = self.config.get("ema_fast", 9)
        ema_slow = self.config.get("ema_slow", 21)
        ema9 = self.calculate_ema(closes, ema_fast)
        ema21 = self.calculate_ema(closes, ema_slow)
        # Protect against division by zero
        ema_strength = abs((ema9[-1] - ema21[-1]) / ema21[-1] * 100) * 10 if ema21[-1] != 0 else 50
        results['EMA'] = {
            'signal': 'LONG' if ema9[-1] > ema21[-1] else 'SHORT',
            'strength': min(ema_strength, 100),
            'value': ema9[-1]
        }

        # 4. ADX/DMI
        adx_threshold = self.config.get("adx_threshold", 20)
        adx, plus_di, minus_di = self.calculate_adx(highs, lows, closes)
        results['ADX'] = {
            'signal': 'LONG' if plus_di[-1] > minus_di[-1] and adx[-1] > adx_threshold else 'SHORT' if minus_di[-1] > plus_di[-1] and adx[-1] > adx_threshold else 'NEUTRAL',
            'strength': min(adx[-1], 100) if adx[-1] > adx_threshold else 20,
            'value': adx[-1]
        }

        # 5. Bollinger Bands
        # FIXED: Context-aware logic (trending vs ranging markets)
        bb_period = self.config.get("bb_period", 20)
        bb_std = self.config.get("bb_std_dev", 2.0)
        sma20 = np.convolve(closes, np.ones(bb_period)/bb_period, mode='same')

        # Efficient rolling std using cumulative sums
        std = self._calculate_rolling_std(closes, bb_period)

        bb_upper = sma20 + bb_std * std
        bb_lower = sma20 - bb_std * std
        bb_range = bb_upper[-1] - bb_lower[-1]
        bb_pos = ((closes[-1] - bb_lower[-1]) / bb_range * 100) if bb_range > 1e-10 else 50.0

        # FIXED: Check if market is trending (bands expanding) or ranging
        # Get historical BB width to detect expansion/contraction
        if len(std) > 20:
            historical_avg_width = np.mean(bb_upper[:-1] - bb_lower[:-1])
            current_width = bb_upper[-1] - bb_lower[-1]
            bandwidth_ratio = current_width / max(historical_avg_width, 1e-10)
        else:
            bandwidth_ratio = 1.0

        # Context-dependent logic
        if bandwidth_ratio > 1.5:  # Bands expanded = trending market
            # In trends: bands are price boundaries, not reversals
            if closes[-1] > sma20[-1]:
                bb_signal = 'LONG'
                bb_strength = min(abs(bb_pos - 50) * 1.2, 100)  # Boost strength in trends
            else:
                bb_signal = 'SHORT'
                bb_strength = min(abs(bb_pos - 50) * 1.2, 100)
        else:  # Normal/tight bands = ranging market
            # In ranges: bands are mean reversion levels
            bb_signal = 'LONG' if bb_pos < 30 else 'SHORT' if bb_pos > 70 else 'NEUTRAL'
            bb_strength = abs(bb_pos - 50)

        results['BollingerBands'] = {
            'signal': bb_signal,
            'strength': max(20.0, min(bb_strength, 100.0)),
            'value': bb_pos,
            'bandwidth_ratio': bandwidth_ratio
        }

        # 6. Volume Trend
        vol_sma_period = self.config.get("volume_sma_period", 20)
        vol_multiplier = self.config.get("volume_surge_multiplier", 1.2)
        vol_sma = np.convolve(volumes, np.ones(vol_sma_period)/vol_sma_period, mode='same')
        results['Volume'] = {
            'signal': 'LONG' if volumes[-1] > vol_sma[-1] * vol_multiplier and closes[-1] > closes[-2] else 'SHORT' if volumes[-1] > vol_sma[-1] * vol_multiplier and closes[-1] < closes[-2] else 'NEUTRAL',
            'strength': 60 if volumes[-1] > vol_sma[-1] * vol_multiplier else 30,
            'value': volumes[-1]
        }

        # 7. Momentum (5-period)
        # FIXED: Adaptive momentum with available data + volatility normalization
        momentum_period = min(5, len(closes) - 1)  # Use available data if < 5 candles

        if len(closes) >= 2 and closes[-momentum_period] != 0:
            momentum_pct = ((closes[-1] - closes[-momentum_period]) / closes[-momentum_period]) * 100

            # FIXED: Normalize by volatility (ATR) for context
            atr_period = self.config.get("atr_period", 14)
            atr = max(self.calculate_atr(highs, lows, closes, atr_period), 1e-10)
            atr_pct = (atr / closes[-1]) * 100

            # Normalize: momentum relative to volatility
            momentum = momentum_pct / max(atr_pct, 0.1)  # Avoid division by zero

            # Calculate signal strength
            momentum_strength = min(abs(momentum) * 15, 100)  # Scale appropriately
        else:
            momentum = 0.0
            momentum_strength = 20.0
            logger.debug("Momentum: Insufficient data (used %d candles), returning neutral", momentum_period)

        results['Momentum'] = {
            'signal': 'LONG' if momentum > 0.1 else 'SHORT' if momentum < -0.1 else 'NEUTRAL',
            'strength': momentum_strength,
            'value': momentum,
            'period_used': momentum_period
        }

        # 8. Stochastic RSI (timing)
        stoch_k, stoch_d = self.calculate_stoch_rsi(closes)
        k_val = stoch_k[-1]
        d_val = stoch_d[-1]
        if k_val < 20 and k_val > d_val:
            stoch_signal = 'LONG'
            # Lower k_val = more oversold = stronger signal (0-100 range)
            stoch_strength = min((20 - k_val) * 5, 100)
        elif k_val > 80 and k_val < d_val:
            stoch_signal = 'SHORT'
            # Higher k_val = more overbought = stronger signal (0-100 range)
            stoch_strength = min((k_val - 80) * 5, 100)
        else:
            stoch_signal = 'NEUTRAL'
            stoch_strength = abs(k_val - d_val)
        results['StochRSI'] = {
            'signal': stoch_signal,
            'strength': max(20.0, float(stoch_strength)),
            'value': {'k': float(k_val), 'd': float(d_val)}
        }

        # 9. Price vs EMAs (50, 200)
        ema_trend_fast = self.config.get("ema_trend_fast", 50)
        ema_trend_slow = self.config.get("ema_trend_slow", 200)
        ema50 = self.calculate_ema(closes, ema_trend_fast)
        ema200 = self.calculate_ema(closes, ema_trend_slow)
        above_50 = closes[-1] > ema50[-1]
        above_200 = closes[-1] > ema200[-1]
        results['TrendPosition'] = {
            'signal': 'LONG' if above_50 and above_200 else 'SHORT' if not above_50 and not above_200 else 'NEUTRAL',
            'strength': 80 if (above_50 and above_200) or (not above_50 and not above_200) else 40,
            'value': 1 if above_50 else 0
        }

        return results

    def detect_market_regime(self, highs: npt.NDArray[np.floating[Any]], lows: npt.NDArray[np.floating[Any]], closes: npt.NDArray[np.floating[Any]]) -> Dict[str, Any]:
        """
        ISSUE #2 FIX: Detect market regime (trending vs ranging/choppy).

        Analyzes:
        - ADX: Trend strength (> 25 = trending, < 20 = ranging)
        - Bollinger Bands bandwidth: High = volatile trending, Low = ranging/choppy
        - Momentum persistence: How long direction is consistent

        Returns dict with:
        - regime: 'TRENDING', 'RANGING', or 'CHOPPY'
        - adx_strength: ADX value (0-100)
        - volatility_score: Bollinger bandwidth ratio (0-1)
        - momentum_strength: Directional consistency (0-1)
        - confidence: Overall regime confidence (0-1)
        """
        # Get ADX for trend strength
        adx, plus_di, minus_di = self.calculate_adx(highs, lows, closes)
        adx_value = float(adx[-1]) if len(adx) > 0 else 0.0
        adx_trending_threshold = self.config.get("adx_trending", 25)
        adx_ranging_threshold = self.config.get("adx_threshold", 20)

        # Get volatility from Bollinger Bands
        bb_period = self.config.get("bb_period", 20)

        # Simple SMA calculation using numpy convolve (same as Volume calculation)
        sma = np.convolve(closes, np.ones(bb_period)/bb_period, mode='same')
        std_dev = self._calculate_rolling_std(closes, bb_period)

        # Bollinger Bands bandwidth ratio (normalized between 0-1)
        # High bandwidth = volatile (more trending potential)
        # Low bandwidth = compressed (ranging/choppy)
        current_sma = sma[-1] if len(sma) > 0 else closes[-1]
        current_std = std_dev[-1] if len(std_dev) > 0 else 0.0

        # Bandwidth = (Upper - Lower) / Middle * 100 = (4 * std) / SMA
        bandwidth = (4 * current_std) / (current_sma + 1e-10)  # Normalized
        max_bandwidth = 0.20  # 20% is high volatility
        volatility_score = min(1.0, bandwidth / max_bandwidth)

        # Get momentum consistency (how many consecutive closes in same direction)
        momentum_period = min(5, len(closes) - 1)
        if momentum_period > 0:
            recent_closes = closes[-momentum_period:]
            price_changes = np.diff(recent_closes)
            same_direction = np.sum(price_changes > 0) if len(price_changes) > 0 else 0
            momentum_strength = float(same_direction) / max(len(price_changes), 1)
        else:
            momentum_strength = 0.5

        # Determine regime
        if adx_value > adx_trending_threshold:
            # Strong trend
            regime = 'TRENDING'
            confidence = min(1.0, (adx_value / 50.0)) * volatility_score * momentum_strength
        elif adx_value < adx_ranging_threshold:
            # Ranging market
            if volatility_score < 0.3:
                regime = 'CHOPPY'  # Low volatility + low ADX = choppy/sideways
                confidence = 1.0 - volatility_score  # Higher confidence in choppy when bandwidth is low
            else:
                regime = 'RANGING'  # Moderate volatility but no trend
                confidence = 0.6 * volatility_score
        else:
            # Transition zone
            if volatility_score > 0.6:
                regime = 'TRENDING'
                confidence = 0.5
            else:
                regime = 'RANGING'
                confidence = 0.5

        logger.debug("Market Regime: %s (ADX=%.1f, Volatility=%.2f, Momentum=%.2f, Conf=%.2f%%)",
                    regime, adx_value, volatility_score, momentum_strength, confidence * 100)

        return {
            'regime': regime,
            'adx_strength': adx_value,
            'volatility_score': volatility_score,
            'momentum_strength': momentum_strength,
            'confidence': confidence,
        }

    def calculate_confluence(self, results: Dict[str, Dict[str, Any]]) -> Tuple[str, float, float, float]:
        """Calculate overall confluence score."""
        # CRITICAL FIX: Early return if no results to process
        if not results:
            logger.debug("Confluence: No indicator results to process")
            return "NEUTRAL", 0.0, 0.0, 0.0

        long_score = 0.0
        short_score = 0.0
        neutral_count = 0
        indicators_processed = 0

        adx_value = float(results.get('ADX', {}).get('value', 0) or 0)
        adx_trending = self.config.get("adx_trending", 25)
        trending = adx_value > adx_trending

        # Get weights from config
        trending_weights = self.indicator_weights.get("trending_market", {})
        ranging_weights = self.indicator_weights.get("ranging_market", {})

        # FIXED: Track veto indicators (0-weight) separately
        veto_indicators = {}  # Store 0-weight indicators for veto logic
        proposed_direction = None  # Track which direction is emerging

        for name, data in results.items():
            signal = data.get('signal', 'NEUTRAL')
            # Clamp strength to 0-100 to prevent inflated scores
            strength = max(0, min(100, float(data.get('strength', 50))))

            # Apply market-condition-based weights
            if trending:
                weight = float(trending_weights.get(name, 1.0))
            else:
                weight = float(ranging_weights.get(name, 1.0))

            # FIXED: Store 0-weight indicators for veto mechanism instead of just skipping
            if weight == 0.0:
                if signal != 'NEUTRAL':
                    veto_indicators[name] = signal
                continue

            indicators_processed += 1
            adjusted_strength = strength * weight

            if signal == 'LONG':
                long_score += adjusted_strength
                if proposed_direction is None:
                    proposed_direction = 'LONG'
            elif signal == 'SHORT':
                short_score += adjusted_strength
                if proposed_direction is None:
                    proposed_direction = 'SHORT'
            else:
                neutral_count += 1

        # FIXED: Apply veto mechanism - if veto indicators strongly oppose the signal, reduce confidence
        if veto_indicators:
            veto_opposing_count = 0
            for veto_name, veto_signal in veto_indicators.items():
                if proposed_direction and veto_signal != proposed_direction:
                    veto_opposing_count += 1

            # If 50%+ of veto indicators oppose, reduce overall score by 20%
            if veto_opposing_count > 0 and len(veto_indicators) > 0:
                veto_ratio = veto_opposing_count / len(veto_indicators)
                if veto_ratio >= 0.5:
                    logger.debug("Confluence: Veto indicators %d/%d oppose, reducing score by 20%%",
                               veto_opposing_count, len(veto_indicators))
                    long_score *= 0.8
                    short_score *= 0.8

        # CRITICAL FIX: Early return if no indicators were processed (all weights = 0)
        if indicators_processed == 0:
            logger.debug("Confluence: No indicators processed (all weights = 0)")
            return "NEUTRAL", 0.0, 0.0, 0.0

        # Calculate max possible score based on actual weights used
        total_weight = 0.0
        for name in results.keys():
            if trending:
                w = float(trending_weights.get(name, 1.0))
            else:
                w = float(ranging_weights.get(name, 1.0))
            if w > 0:
                total_weight += w

        # CRITICAL FIX: Prevent division by zero with explicit check
        if total_weight <= 0:
            max_score = float(len(results) * 100)
        else:
            max_score = total_weight * 100.0

        # Additional safety check
        if max_score <= 0:
            logger.warning("Confluence: max_score is 0, returning NEUTRAL")
            return "NEUTRAL", 0.0, long_score, short_score

        long_pct = (long_score / max_score) * 100
        short_pct = (short_score / max_score) * 100

        # Clamp to a clean 0-100 range for easier interpretation
        long_pct = max(0.0, min(100.0, float(long_pct)))
        short_pct = max(0.0, min(100.0, float(short_pct)))

        # WIN RATE FIX: Count how many indicators agree on each direction
        long_count = sum(1 for d in results.values() if d.get('signal') == 'LONG')
        short_count = sum(1 for d in results.values() if d.get('signal') == 'SHORT')
        total_indicators = len(results)

        # WIN RATE FIX: Require minimum indicator agreement (at least 50% of indicators)
        min_agreement_pct = 0.5
        min_agreement_count = max(4, int(total_indicators * min_agreement_pct))

        # WIN RATE FIX: Require clear margin between long and short (at least 15% difference)
        min_margin = 15.0
        score_margin = abs(long_pct - short_pct)

        # WIN RATE FIX: Higher direction threshold (45% instead of 25%)
        direction_threshold = 45.0

        # Determine direction with stricter requirements
        if (long_pct > short_pct and
            long_pct > direction_threshold and
            score_margin >= min_margin and
            long_count >= min_agreement_count):
            direction = "BULLISH"
            confidence = long_pct
        elif (short_pct > long_pct and
              short_pct > direction_threshold and
              score_margin >= min_margin and
              short_count >= min_agreement_count):
            direction = "BEARISH"
            confidence = short_pct
        else:
            direction = "NEUTRAL"
            confidence = 0
            if long_pct > 30 or short_pct > 30:
                logger.debug("Confluence: Rejected - margin=%.1f%% (need %.1f%%), long_count=%d, short_count=%d (need %d)",
                           score_margin, min_margin, long_count, short_count, min_agreement_count)

        return direction, confidence, long_score, short_score

    @staticmethod
    def calculate_atr(highs: npt.NDArray[np.floating[Any]], lows: npt.NDArray[np.floating[Any]], closes: npt.NDArray[np.floating[Any]], period: int = 14) -> float:
        """Calculate ATR."""
        tr = np.zeros(len(closes))
        tr[0] = highs[0] - lows[0]
        for i in range(1, len(closes)):
            tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))

        if len(tr) >= period:
            return float(np.mean(tr[-period:]))
        return float(np.mean(tr)) if len(tr) > 0 else 0.0


class MexcClient:
    """MEXC exchange client wrapper with caching."""

    def __init__(self, rate_limiter: Optional[Any] = None,
                 ohlcv_cache_ttl: int = 60, ticker_cache_ttl: int = 5) -> None:
        """
        Initialize MEXC client with caching.

        Args:
            rate_limiter: Optional rate limiter instance
            ohlcv_cache_ttl: OHLCV cache TTL in seconds (default: 60s)
            ticker_cache_ttl: Ticker cache TTL in seconds (default: 5s)
        """
        self.exchange: Any = ccxt.binanceusdm({
            "enableRateLimit": True,
            "options": {"defaultType": "swap"}
        })
        self.rate_limiter = rate_limiter

        # HIGH PRIORITY FIX: Add caching to reduce API calls
        self._ohlcv_cache: Dict[str, Tuple[Any, float]] = {}  # key -> (data, timestamp)
        self._ticker_cache: Dict[str, Tuple[Any, float]] = {}  # key -> (data, timestamp)
        self._ohlcv_cache_ttl = ohlcv_cache_ttl
        self._ticker_cache_ttl = ticker_cache_ttl
        self._cache_hits = 0
        self._cache_misses = 0

        self._load_markets_with_retry()

    def _load_markets_with_retry(self, max_retries: int = 3) -> None:
        """Load markets with retry logic for network resilience."""
        for attempt in range(max_retries):
            try:
                self.exchange.load_markets()
                logger.info("MEXC markets loaded successfully")
                return
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
                logger.warning(f"Failed to load MEXC markets (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))  # Exponential backoff
                else:
                    logger.error("Failed to load MEXC markets after all retries")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error loading MEXC markets: {e}")
                raise

    @staticmethod
    def _swap_symbol(symbol: str) -> str:
        """Convert symbol to swap format."""
        # If already in correct format, return as-is
        if ":USDT" in symbol:
            return symbol
        # If has /USDT, convert to swap format
        if "/USDT" in symbol:
            return f"{symbol}:USDT"
        # Otherwise, add both
        return f"{symbol}/USDT:USDT"

    def _is_cache_valid(self, cache_entry: Optional[Tuple[Any, float]], ttl: int) -> bool:
        """Check if cache entry is still valid."""
        if cache_entry is None:
            return False
        _, timestamp = cache_entry
        return (time.time() - timestamp) < ttl

    def _get_ohlcv_cache_key(self, symbol: str, timeframe: str, limit: int) -> str:
        """Generate cache key for OHLCV data."""
        return f"{symbol}:{timeframe}:{limit}"

    def fetch_ohlcv(self, symbol: str, timeframe: str = "5m", limit: int = 300) -> List[Any]:
        """Fetch OHLCV data with caching."""
        cache_key = self._get_ohlcv_cache_key(symbol, timeframe, limit)

        # HIGH PRIORITY FIX: Check cache first
        cache_entry = self._ohlcv_cache.get(cache_key)
        if self._is_cache_valid(cache_entry, self._ohlcv_cache_ttl) and cache_entry is not None:
            self._cache_hits += 1
            logger.debug("OHLCV cache hit for %s (hits: %d)", symbol, self._cache_hits)
            return cast(List[Any], cache_entry[0])  # Return cached data

        self._cache_misses += 1

        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()

        try:
            result = self.exchange.fetch_ohlcv(
                self._swap_symbol(symbol),
                timeframe=timeframe,
                limit=limit
            )
            # Store in cache
            self._ohlcv_cache[cache_key] = (result, time.time())

            if self.rate_limiter:
                self.rate_limiter.record_success(f"mexc_{symbol}")
            return cast(List[Any], result)
        except Exception:
            if self.rate_limiter:
                self.rate_limiter.record_error(f"mexc_{symbol}")
            raise

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch ticker data with caching."""
        # HIGH PRIORITY FIX: Check cache first
        cache_entry = self._ticker_cache.get(symbol)
        if self._is_cache_valid(cache_entry, self._ticker_cache_ttl) and cache_entry is not None:
            self._cache_hits += 1
            return cast(Dict[str, Any], cache_entry[0])  # Return cached data

        self._cache_misses += 1

        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()

        try:
            result = self.exchange.fetch_ticker(self._swap_symbol(symbol))
            # Store in cache
            self._ticker_cache[symbol] = (result, time.time())

            if self.rate_limiter:
                self.rate_limiter.record_success(f"mexc_{symbol}")
            return cast(Dict[str, Any], result)
        except Exception:
            if self.rate_limiter:
                self.rate_limiter.record_error(f"mexc_{symbol}")
            raise

    def fetch_tickers(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch multiple tickers in a single API call (batch).

        HIGH PRIORITY FIX: Reduces API calls when monitoring multiple open signals.

        Args:
            symbols: List of symbols to fetch

        Returns:
            Dictionary mapping symbol -> ticker data
        """
        if not symbols:
            return {}

        # Check which symbols need fetching (not in cache or expired)
        symbols_to_fetch = []
        result = {}

        for symbol in symbols:
            cache_entry = self._ticker_cache.get(symbol)
            if self._is_cache_valid(cache_entry, self._ticker_cache_ttl) and cache_entry is not None:
                result[symbol] = cache_entry[0]
                self._cache_hits += 1
            else:
                symbols_to_fetch.append(symbol)
                self._cache_misses += 1

        if not symbols_to_fetch:
            logger.debug("All %d tickers served from cache", len(symbols))
            return result

        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()

        try:
            # Fetch all tickers at once
            swap_symbols = [self._swap_symbol(s) for s in symbols_to_fetch]
            all_tickers = self.exchange.fetch_tickers(swap_symbols)

            # Map back to original symbols and cache
            now = time.time()
            for symbol in symbols_to_fetch:
                swap_symbol = self._swap_symbol(symbol)
                if swap_symbol in all_tickers:
                    ticker_data = all_tickers[swap_symbol]
                    self._ticker_cache[symbol] = (ticker_data, now)
                    result[symbol] = ticker_data

            if self.rate_limiter:
                self.rate_limiter.record_success("mexc_batch_tickers")

            logger.debug("Batch fetched %d tickers (%d from cache)",
                        len(symbols_to_fetch), len(symbols) - len(symbols_to_fetch))
            return result

        except Exception as e:
            logger.warning("Batch ticker fetch failed, falling back to individual: %s", e)
            # Fallback to individual fetches
            for symbol in symbols_to_fetch:
                try:
                    result[symbol] = self.fetch_ticker(symbol)
                except Exception as ex:
                    logger.warning("Failed to fetch ticker for %s: %s", symbol, ex)
            return result

    def get_cache_stats(self) -> Dict[str, int | float]:
        """Get cache statistics."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": (self._cache_hits / max(self._cache_hits + self._cache_misses, 1)) * 100,
            "ohlcv_entries": len(self._ohlcv_cache),
            "ticker_entries": len(self._ticker_cache),
        }

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._ohlcv_cache.clear()
        self._ticker_cache.clear()
        logger.info("Cache cleared")


@dataclass
class DIYSignal:
    """Represents a DIY multi-indicator signal."""
    symbol: str
    direction: str
    timestamp: str
    confidence: float
    long_score: float
    short_score: float
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    current_price: Optional[float] = None


class BotState:
    """Enhanced bot state persistence with thread-safety and result tracking."""

    def __init__(self, path: Path):
        self.path = path
        self.state_lock = threading.Lock()  # Thread-safe operations
        self.data: Dict[str, Any] = self._load()

    @staticmethod
    def _parse_ts(value: str) -> datetime:
        """Parse ISO format timestamp, ensuring UTC timezone."""
        try:
            dt = datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt

    def _empty_state(self) -> Dict[str, Any]:
        return {
            "last_alert": {},
            "open_signals": {},
            "closed_signals": {},
            "last_result_notifications": {}  # Track result notification cooldowns
        }

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return self._empty_state()
        try:
            data = json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return self._empty_state()
        if not isinstance(data, dict):
            return self._empty_state()
        if not isinstance(data.get("last_alert"), dict):
            data["last_alert"] = {}
        if not isinstance(data.get("open_signals"), dict):
            data["open_signals"] = {}
        if not isinstance(data.get("last_result_notifications"), dict):
            data["last_result_notifications"] = {}
        return data

    def save(self) -> None:
        """Thread-safe state persistence with file locking and atomic writes."""
        with self.state_lock:
            temp_file = self.path.with_suffix('.tmp')
            try:
                # Write to temp file with exclusive lock (Unix only)
                with open(temp_file, 'w') as f:
                    if HAS_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    json.dump(self.data, f, indent=2)
                    if HAS_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

                # Atomic rename
                temp_file.replace(self.path)
            except Exception as e:
                logger.error("Failed to save state: %s", e)
                if temp_file.exists():
                    temp_file.unlink()

    def can_alert(self, symbol: str, cooldown_minutes: int) -> bool:
        last_map = self.data.setdefault("last_alert", {})
        if not isinstance(last_map, dict):
            last_map = {}
            self.data["last_alert"] = last_map
        last_ts = last_map.get(symbol)
        if not isinstance(last_ts, str):
            return True
        delta = datetime.now(timezone.utc) - self._parse_ts(last_ts)
        return delta >= timedelta(minutes=cooldown_minutes)

    def mark_alert(self, symbol: str) -> None:
        last_map = self.data.setdefault("last_alert", {})
        if not isinstance(last_map, dict):
            last_map = {}
            self.data["last_alert"] = last_map
        last_map[symbol] = datetime.now(timezone.utc).isoformat()
        self.save()

    def should_notify_result(self, symbol: str, cooldown_minutes: int) -> bool:
        """Check if we should send result notification (cooldown prevention)."""
        last_notifs = self.data.setdefault("last_result_notifications", {})
        if not isinstance(last_notifs, dict):
            last_notifs = {}
            self.data["last_result_notifications"] = last_notifs

        last_ts = last_notifs.get(symbol)
        if not isinstance(last_ts, str):
            return True

        try:
            last_dt = self._parse_ts(last_ts)
            now = datetime.now(timezone.utc)
            return now - last_dt >= timedelta(minutes=cooldown_minutes)
        except Exception:
            return True

    def mark_result_notified(self, symbol: str) -> None:
        """Mark that we sent a result notification for this symbol."""
        last_notifs = self.data.setdefault("last_result_notifications", {})
        if not isinstance(last_notifs, dict):
            last_notifs = {}
            self.data["last_result_notifications"] = last_notifs
        last_notifs[symbol] = datetime.now(timezone.utc).isoformat()
        self.save()

    def add_signal(self, signal_id: str, payload: Dict[str, Any]) -> None:
        with self.state_lock:
            signals = self.data.setdefault("open_signals", {})
            if not isinstance(signals, dict):
                signals = {}
                self.data["open_signals"] = signals
            signals[signal_id] = payload
        self.save()

    def remove_signal(self, signal_id: str) -> None:
        removed = False
        with self.state_lock:
            signals = self.data.setdefault("open_signals", {})
            if not isinstance(signals, dict):
                self.data["open_signals"] = {}
                return
            if signal_id in signals:
                signals.pop(signal_id)
                removed = True
        if removed:
            self.save()

    def archive_signal(self, signal_id: str, payload: Dict[str, Any], exit_price: float, result: str, pnl_pct: float) -> None:
        """Archive signal to closed_signals before removal."""
        with self.state_lock:
            closed = self.data.setdefault("closed_signals", {})
            if not isinstance(closed, dict):
                closed = {}
                self.data["closed_signals"] = closed
            closed_signal = payload.copy()
            closed_signal["closed_at"] = datetime.now(timezone.utc).isoformat()
            closed_signal["exit_price"] = exit_price
            closed_signal["result"] = result
            closed_signal["pnl_percent"] = pnl_pct
            closed[signal_id] = closed_signal
        self.save()

    def iter_signals(self) -> Dict[str, Dict[str, Any]]:
        """Return a copy of signals to prevent external modification."""
        with self.state_lock:
            signals = self.data.setdefault("open_signals", {})
            if not isinstance(signals, dict):
                signals = {}
                self.data["open_signals"] = signals
            # Return a shallow copy to prevent external modification
            return cast(Dict[str, Dict[str, Any]], dict(signals))

    def cleanup_stale_signals(self, max_age_hours: int = 24, stats: Optional[Any] = None) -> int:
        """Remove signals older than max_age_hours and archive to stats."""
        signals = self.iter_signals()
        now = datetime.now(timezone.utc)
        stale_ids = []

        for signal_id, payload in signals.items():
            if not isinstance(payload, dict):
                stale_ids.append(signal_id)
                continue

            created_at = payload.get("created_at")
            if not isinstance(created_at, str):
                stale_ids.append(signal_id)
                continue

            created_dt = self._parse_ts(created_at)
            age = now - created_dt
            if age >= timedelta(hours=max_age_hours):
                stale_ids.append(signal_id)

                # Archive to stats as EXPIRED before removing
                if stats:
                    stats.record_close(
                        signal_id,
                        exit_price=payload.get("entry", 0),
                        result="EXPIRED"
                    )

        for signal_id in stale_ids:
            self.remove_signal(signal_id)

        # Also cleanup old closed signals to prevent unbounded growth
        closed_pruned = 0
        with self.state_lock:
            closed = self.data.get("closed_signals", {})
            max_closed_signals = 100  # Keep only the most recent 100 closed signals
            if isinstance(closed, dict) and len(closed) > max_closed_signals:
                sorted_closed = sorted(
                    closed.items(),
                    key=lambda x: x[1].get("closed_at", "") if isinstance(x[1], dict) else "",
                    reverse=True
                )
                self.data["closed_signals"] = dict(sorted_closed[:max_closed_signals])
                closed_pruned = len(closed) - max_closed_signals
                logger.info("Pruned %d old closed signals (keeping %d)", closed_pruned, max_closed_signals)

        if closed_pruned > 0:
            self.save()

        return len(stale_ids)


class DIYBot(BotSignalMixin):
    """Main DIY Multi-Indicator Bot with unified signal management."""

    def __init__(self, config_path: str = str(CONFIG_FILE)):
        # Load configuration
        self.config = load_config(Path(config_path))

        # Reconfigure logging based on config
        log_level = self.config.get("execution", {}).get("log_level", "INFO")
        enable_detailed = self.config.get("execution", {}).get("enable_detailed_logging", False)
        global logger
        logger = setup_logging(log_level, enable_detailed)

        # Load and validate watchlist
        self.watchlist: List[WatchItem] = self._load_and_validate_watchlist()

        # Initialize components
        self.notifier = self._init_notifier()
        self.stats = SignalStats("DIY Bot", STATS_FILE)

        # Initialize Health Monitor with RateLimiter
        health_check_interval = self.config.get("execution", {}).get("health_check_interval_seconds", 3600)
        self.health_monitor = (
            HealthMonitor("DIY Bot", self.notifier, heartbeat_interval=health_check_interval)
            if HealthMonitor and self.notifier else None
        )
        
        # NEW: Initialize unified signal adapter
        self._init_signal_adapter(
            bot_name="diy_bot",
            notifier=self.notifier,
            exchange="MEXC",
            default_timeframe="15m",
            notification_mode="signal_only",
        )

        # Get RateLimiter from HealthMonitor
        calls_per_minute = self.config.get("rate_limit", {}).get("calls_per_minute", 60)
        self.rate_limiter = (
            RateLimiter(calls_per_minute=calls_per_minute, backoff_file=LOG_DIR / "rate_limiter.json")
            if RateLimiter else None
        )

        # Exchange backoff tracking (from Volume Bot)
        self.exchange_backoff: Dict[str, float] = {}
        self.exchange_delay: Dict[str, float] = {}

        # Initialize exchange client with rate limiter
        self.client = MexcClient(rate_limiter=self.rate_limiter)

        # Initialize analyzer with config
        self.analyzer = MultiIndicatorAnalyzer(self.config)

        # Initialize state
        self.state = BotState(STATE_FILE)

        # ISSUE #3 FIX: Add thread-safe signal monitoring lock to prevent race conditions
        # Prevents concurrent cycles from missing TP/SL hits by serializing signal updates
        self.signal_monitor_lock = threading.Lock()
        self.signal_analysis_lock = threading.Lock()  # Separate lock for signal analysis

        # Sync open signals to stats on startup (prevents lost tracking after restart)
        if self.stats:
            self._sync_signals_to_stats()

        logger.info("DIY Bot initialized with %d symbols in watchlist", len(self.watchlist))

    def _load_and_validate_watchlist(self) -> List[WatchItem]:
        """Load and validate watchlist entries (from Volume Bot)."""
        raw_watchlist = load_watchlist()
        valid_entries = []
        invalid_count = 0

        valid_exchanges = ["binanceusdm", "mexc", "bybit"]
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"]

        for item in raw_watchlist:
            symbol = item.get("symbol")
            exchange = item.get("exchange", "binanceusdm")
            timeframe = item.get("timeframe", "5m")

            if not symbol:
                logger.warning("Invalid watchlist entry: missing symbol")
                invalid_count += 1
                continue

            if exchange not in valid_exchanges:
                logger.warning(f"Invalid exchange '{exchange}' for {symbol}, using mexc")
                item["exchange"] = "binanceusdm"

            if timeframe not in valid_timeframes:
                logger.warning(f"Invalid timeframe '{timeframe}' for {symbol}, using 5m")
                item["timeframe"] = "5m"

            valid_entries.append(item)

        if invalid_count > 0:
            logger.warning(f"Skipped {invalid_count} invalid watchlist entries")

        return valid_entries

    def _backoff_active(self, exchange: str) -> bool:
        """Check if exchange is in backoff period (from Volume Bot)."""
        until = self.exchange_backoff.get(exchange)
        return bool(until and time.time() < until)

    def _register_backoff(self, exchange: str) -> None:
        """Register rate limit backoff for an exchange (from Volume Bot)."""
        base_delay = self.config.get("rate_limit", {}).get("rate_limit_backoff_base", 60)
        max_delay = self.config.get("rate_limit", {}).get("rate_limit_backoff_max", 300)
        multiplier = self.config.get("rate_limit", {}).get("backoff_multiplier", 2.0)

        prev = self.exchange_delay.get(exchange, base_delay)
        new_delay = min(prev * multiplier, max_delay) if exchange in self.exchange_backoff else base_delay

        until = time.time() + new_delay
        self.exchange_backoff[exchange] = until
        self.exchange_delay[exchange] = new_delay
        logger.warning(f"{exchange} rate limit; backing off for {new_delay:.0f}s")

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """Check if exception is a rate limit error (from Volume Bot)."""
        msg = str(exc)
        return "Requests are too frequent" in msg or "429" in msg or "403 Forbidden" in msg

    def _count_symbol_signals(self, symbol: str) -> int:
        """Count open signals for a specific symbol (from Volume Bot)."""
        signals = self.state.iter_signals()
        normalized = symbol.upper().split("/")[0].replace("USDT", "")
        count = 0

        for payload in signals.values():
            if not isinstance(payload, dict):
                continue
            sig_symbol = payload.get("symbol", "")
            sig_normalized = sig_symbol.upper().split("/")[0].replace("USDT", "")
            if sig_normalized == normalized:
                count += 1

        return count

    def _init_notifier(self) -> Optional[Any]:
        if TelegramNotifier is None:
            logger.warning("Telegram notifier unavailable")
            return None

        telegram_config = self.config.get("telegram", {})
        token_env = telegram_config.get("bot_token_env_var", "TELEGRAM_BOT_TOKEN_DIY")
        chat_id_env = telegram_config.get("chat_id_env_var", "TELEGRAM_CHAT_ID")

        token = os.getenv(token_env) or os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv(chat_id_env)

        if not token or not chat_id:
            logger.warning("Telegram credentials missing (token: %s, chat_id: %s)", token_env, chat_id_env)
            return None

        return TelegramNotifier(
            bot_token=token,
            chat_id=chat_id,
            signals_log_file=str(LOG_DIR / "diy_signals.json")
        )

    def _sync_signals_to_stats(self) -> None:
        """Sync open signals from state to stats on startup (prevents lost tracking after restart)."""
        if not self.stats:
            return

        signals = self.state.iter_signals()
        stats_data = getattr(self.stats, 'data', {})
        stats_open: Dict[str, Any] = stats_data.get("open", {}) if isinstance(stats_data, dict) else {}
        synced = 0

        for signal_id, payload in signals.items():
            if signal_id not in stats_open:
                self.stats.record_open(
                    signal_id=signal_id,
                    symbol=payload.get("symbol", ""),
                    direction=payload.get("direction", "BULLISH"),
                    entry=payload.get("entry", 0),
                    created_at=payload.get("created_at", ""),
                    extra={
                        "timeframe": payload.get("timeframe", ""),
                        "exchange": payload.get("exchange", ""),
                    }
                )
                synced += 1

        if synced > 0:
            logger.info("✅ Synced %d open signals to stats tracker", synced)

    def _format_result_message(
        self,
        signal_id: str,
        symbol: str,
        direction: str,
        entry: float,
        exit: float,
        result: str,
        sl: float,
        tp1: float,
        tp2: float
    ) -> str:
        """Format enhanced result message with performance history."""
        # Get performance stats (ALWAYS included)
        tp1_count = 0
        tp2_count = 0
        sl_count = 0
        avg_pnl = 0.0
        if self.stats:
            counts = self.stats.symbol_tp_sl_counts(symbol)
            tp1_count = counts.get("TP1", 0)
            tp2_count = counts.get("TP2", 0)
            sl_count = counts.get("SL", 0)
            avg_pnl = self.stats.get_avg_pnl(symbol) if hasattr(self.stats, 'get_avg_pnl') else 0.0
        total = tp1_count + tp2_count + sl_count
        perf_stats = {
            "tp1": tp1_count,
            "tp2": tp2_count,
            "sl": sl_count,
            "wins": tp1_count + tp2_count,
            "total": total,
            "avg_pnl": avg_pnl,
        }

        return format_result_message(
            symbol=symbol,
            direction=direction,
            result=result,
            entry=entry,
            exit_price=exit,
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
            signal_id=signal_id,
            performance_stats=perf_stats,
        )

    def run(self, loop: bool = False, track_only: bool = False) -> None:
        global shutdown_requested
        if not self.watchlist:
            logger.error("Empty watchlist; exiting")
            return

        logger.info("Starting DIY Multi-Indicator Bot for %d symbols", len(self.watchlist))

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Send startup notification
        if self.health_monitor:
            enable_startup = self.config.get("execution", {}).get("enable_startup_notification", True)
            if enable_startup:
                self.health_monitor.send_startup_message()

        # Track-only mode: just check open signals (from Volume Bot)
        if track_only:
            logger.info("Running in track-only mode - checking open signals...")
            self._monitor_open_signals()
            logger.info("Track-only check complete")
            return

        interval = self.config.get("execution", {}).get("cycle_interval_seconds", 60)

        # No-signal heartbeat tracking (from Volume Bot)
        last_signal_time = datetime.now(timezone.utc)
        no_signal_hours = self.config.get("signal", {}).get("no_signal_alert_hours", 6)
        last_heartbeat_time = datetime.now(timezone.utc)

        try:
            while not shutdown_requested:
                try:
                    signals_this_cycle = self._run_cycle()

                    # Update last signal time if we got signals
                    if signals_this_cycle > 0:
                        last_signal_time = datetime.now(timezone.utc)

                    # Cleanup stale signals every cycle (with archiving to stats)
                    max_age = self.config.get("signal", {}).get("max_signal_age_hours", 24)
                    stale_count = self.state.cleanup_stale_signals(max_age_hours=max_age, stats=self.stats)
                    if stale_count > 0:
                        logger.info("Cleaned up %d stale signals (archived to stats)", stale_count)

                    self._monitor_open_signals()

                    if self.health_monitor:
                        self.health_monitor.record_cycle()

                    # No-signal heartbeat notification (from Volume Bot)
                    hours_since_signal = (datetime.now(timezone.utc) - last_signal_time).total_seconds() / 3600
                    hours_since_heartbeat = (datetime.now(timezone.utc) - last_heartbeat_time).total_seconds() / 3600

                    if hours_since_signal >= no_signal_hours and hours_since_heartbeat >= no_signal_hours:
                        if self.notifier:
                            open_signals = len(self.state.iter_signals())
                            msg = (
                                f"💓 <b>DIY Bot Heartbeat</b>\n\n"
                                f"⏰ No new signals for {hours_since_signal:.1f} hours\n"
                                f"📊 Open signals: {open_signals}\n"
                                f"👁️ Watching: {len(self.watchlist)} symbols\n"
                                f"✅ Bot is running normally"
                            )
                            self.notifier.send_message(msg, parse_mode="HTML")
                            last_heartbeat_time = datetime.now(timezone.utc)
                            logger.info(f"Heartbeat sent - no signals for {hours_since_signal:.1f} hours")

                    if not loop or shutdown_requested:
                        break

                    logger.info("Cycle complete; sleeping %ds", interval)
                    # Sleep in 1-second chunks to respond quickly to shutdown signals
                    for _ in range(interval):
                        if shutdown_requested:
                            logger.info("Shutdown requested during sleep")
                            break
                        time.sleep(1)

                except Exception as exc:
                    logger.error("Error in cycle: %s", exc, exc_info=True)
                    if self.health_monitor:
                        self.health_monitor.record_error(f"Cycle error: {exc}")
                    if not loop:
                        raise
                    time.sleep(10)
        finally:
            # Send shutdown notification
            if self.health_monitor:
                enable_shutdown = self.config.get("execution", {}).get("enable_shutdown_notification", True)
                if enable_shutdown:
                    self.health_monitor.send_shutdown_message()

    def _run_cycle(self) -> int:
        """Run one analysis cycle. Returns number of signals generated."""
        global shutdown_requested
        symbol_delay = self.config.get("execution", {}).get("symbol_delay_seconds", 1)
        max_same_symbol = self.config.get("risk", {}).get("max_same_symbol_signals", 2)
        signals_generated = 0

        for item in self.watchlist:
            # Check shutdown during watchlist scan
            if shutdown_requested:
                logger.info("Shutdown requested during watchlist scan")
                break

            symbol_val = item.get("symbol") if isinstance(item, dict) else None
            if not isinstance(symbol_val, str):
                continue

            symbol = symbol_val
            timeframe = item.get("timeframe", "5m") if isinstance(item, dict) else "5m"
            exchange = item.get("exchange", "binanceusdm") if isinstance(item, dict) else "mexc"
            cooldown = self.config.get("signal", {}).get("cooldown_minutes", 5)

            # Check exchange backoff (from Volume Bot)
            if self._backoff_active(exchange):
                logger.debug(f"Backoff active for {exchange}; skipping {symbol}")
                continue

            try:
                signal = self._analyze_symbol(symbol, timeframe)
            except ccxt.RateLimitExceeded as e:
                logger.warning("Rate limit exceeded for %s: %s", symbol, e)
                self._register_backoff(exchange)
                if self.health_monitor:
                    self.health_monitor.record_error(f"Rate limit: {symbol}")
                continue
            except ccxt.NetworkError as e:
                logger.warning("Network error for %s: %s", symbol, e)
                if self.health_monitor:
                    self.health_monitor.record_error(f"Network error: {symbol}")
                continue
            except Exception as exc:
                logger.error("Failed to analyze %s: %s", symbol, exc, exc_info=True)
                if self._is_rate_limit_error(exc):
                    self._register_backoff(exchange)
                if self.health_monitor:
                    self.health_monitor.record_error(f"Analysis error: {symbol}")
                continue

            if signal is None:
                logger.debug("%s: No confluence signal", symbol)
                continue

            # Check confidence threshold
            min_confidence = self.config.get("analysis", {}).get("min_confidence_threshold", 60.0)
            if signal.confidence < min_confidence:
                logger.debug("%s: Confidence too low (%.1f%% < %.1f%%)",
                           symbol, signal.confidence, min_confidence)
                continue

            if not self.state.can_alert(symbol, cooldown):
                logger.debug("Cooldown active for %s", symbol)
                continue

            # Max open signals limit
            max_open = self.config.get("risk", {}).get("max_open_signals", 30)
            current_open = len(self.state.iter_signals())
            if current_open >= max_open:
                logger.info(
                    "Max open signals limit reached (%d/%d). Skipping %s",
                    current_open, max_open, symbol
                )
                continue

            # Max same symbol signals check (from Volume Bot)
            symbol_count = self._count_symbol_signals(symbol)
            if symbol_count >= max_same_symbol:
                logger.debug(f"{symbol}: Max same symbol signals reached ({symbol_count}/{max_same_symbol})")
                continue

            # ISSUE #3 FIX: Use signal analysis lock when creating signals
            # Ensures atomicity - prevents race between confidence check and signal creation
            with self.signal_analysis_lock:
                # Re-check max open signals inside the lock (value might have changed)
                current_open = len(self.state.iter_signals())
                if current_open >= max_open:
                    logger.debug("%s: Max open signals reached during lock check", symbol)
                    continue

                # Send alert
                message = self._format_message(signal)
                self._dispatch(message)
                self.state.mark_alert(symbol)
                signals_generated += 1

                # Track signal
                signal_id = f"{symbol}-DIY-{signal.timestamp}"
                trade_data = {
                    "id": signal_id,
                    "symbol": symbol,
                    "direction": signal.direction,
                    "entry": signal.entry,
                    "stop_loss": signal.stop_loss,
                    "take_profit_1": signal.take_profit_1,
                    "take_profit_2": signal.take_profit_2,
                    "created_at": signal.timestamp,
                    "timeframe": timeframe,
                    "exchange": exchange.upper(),
                    "confidence": signal.confidence,
                }
                self.state.add_signal(signal_id, trade_data)

                if self.stats:
                    self.stats.record_open(
                        signal_id=signal_id,
                        symbol=symbol,
                        direction=signal.direction,
                        entry=signal.entry,
                        created_at=signal.timestamp,
                        extra={
                            "timeframe": timeframe,
                        "exchange": exchange.upper(),
                        "strategy": "DIY Multi-Indicator",
                        "confidence": signal.confidence,
                    },
                )

            time.sleep(symbol_delay)

        return signals_generated

    def _analyze_symbol(self, symbol: str, timeframe: str) -> Optional[DIYSignal]:
        """Analyze symbol with 30+ indicators."""
        # Fetch OHLCV data
        candle_limit = self.config.get("analysis", {}).get("candle_limit", 200)
        # CRITICAL FIX: Request candle_limit + 1 to account for removing incomplete candle
        ohlcv = self.client.fetch_ohlcv(symbol, timeframe, limit=candle_limit + 1)

        # Look-ahead bias prevention (from Volume Bot):
        # Exclude current (incomplete) candle to avoid using future data
        if len(ohlcv) > 1:
            ohlcv = ohlcv[:-1]  # Remove last incomplete candle

        # CRITICAL FIX: Check AFTER removing incomplete candle
        if len(ohlcv) < candle_limit:
            logger.debug("%s: Insufficient data (%d candles, need %d)", symbol, len(ohlcv), candle_limit)
            return None

        try:
            highs = np.array([x[2] for x in ohlcv], dtype=float)
            lows = np.array([x[3] for x in ohlcv], dtype=float)
            closes = np.array([x[4] for x in ohlcv], dtype=float)
            volumes = np.array([x[5] for x in ohlcv], dtype=float)
        except Exception as e:
            logger.error("%s: Failed to parse OHLCV data: %s", symbol, e)
            return None

        # Validate OHLCV values are not NaN/inf and are positive
        if np.any(np.isnan(highs)) or np.any(np.isnan(lows)) or \
           np.any(np.isnan(closes)) or np.any(np.isnan(volumes)):
            logger.warning("%s: OHLCV contains NaN values, skipping", symbol)
            return None

        if np.any(np.isinf(highs)) or np.any(np.isinf(lows)) or \
           np.any(np.isinf(closes)) or np.any(np.isinf(volumes)):
            logger.warning("%s: OHLCV contains infinite values, skipping", symbol)
            return None

        if np.any(closes <= 0) or np.any(highs <= 0) or np.any(lows <= 0):
            logger.warning("%s: OHLCV contains zero or negative prices, skipping", symbol)
            return None

        # Validate OHLCV logic: high >= low
        if np.any(highs < lows):
            logger.warning("%s: OHLCV has high < low, skipping", symbol)
            return None

        # Analyze indicators
        results = self.analyzer.analyze_indicators(highs, lows, closes, volumes)

        # ISSUE #2 FIX: Detect market regime and adjust confidence accordingly
        regime_info = self.analyzer.detect_market_regime(highs, lows, closes)
        regime = regime_info.get('regime', 'TRENDING')

        # Calculate confluence
        direction, confidence, long_score, short_score = self.analyzer.calculate_confluence(results)

        # WIN RATE FIX: Log choppy markets but don't skip (testing mode)
        if regime == 'CHOPPY':
            logger.debug("%s: Market is CHOPPY (ADX=%.1f, volatility=%.2f) - proceeding anyway",
                       symbol, regime_info.get('adx_strength', 0), regime_info.get('volatility_score', 0))

        # Apply market regime filter to confidence
        # In RANGING markets, reduce confidence by 20% (lower quality signals)
        # In TRENDING markets, keep full confidence (ideal conditions)
        if regime == 'RANGING':
            confidence_penalty = 0.20
            logger.debug("%s: Market is RANGING, applying %.0f%% confidence penalty",
                        symbol, confidence_penalty * 100)
        else:  # TRENDING
            confidence_penalty = 0.0
            logger.debug("%s: Market is TRENDING, no confidence penalty",
                        symbol)

        # Apply the penalty
        adjusted_confidence = confidence * (1.0 - confidence_penalty)
        logger.debug("%s: Confidence adjusted from %.1f%% to %.1f%% due to %s market",
                    symbol, confidence, adjusted_confidence, regime)
        confidence = adjusted_confidence

        # Debug: Log direction determination
        logger.debug("%s: Direction=%s, Confidence=%.1f, Long=%.1f, Short=%.1f, Regime=%s",
                     symbol, direction, confidence, long_score, short_score, regime)

        if direction == "NEUTRAL":
            return None

        # Calculate ATR
        atr_period = self.config.get("tp_sl", {}).get("atr_period", 14)
        atr = float(self.analyzer.calculate_atr(highs, lows, closes, atr_period))

        # Get current price
        ticker = self.client.fetch_ticker(symbol)
        current_price = ticker.get("last") if isinstance(ticker, dict) else None
        if current_price is None and isinstance(ticker, dict):
            current_price = ticker.get("close")
        if not isinstance(current_price, (int, float)):
            logger.warning("%s: Could not get current price", symbol)
            return None
        entry = float(current_price)

        # Calculate TP/SL using centralized TPSLCalculator
        # Get standard risk config from centralized trade_config
        config_mgr = get_config_manager()
        risk_config = config_mgr.get_effective_risk("diy_bot", symbol)

        # Bot-specific overrides (from local config)
        tp_sl_config = self.config.get("tp_sl", {})
        sl_mult = tp_sl_config.get("atr_sl_multiplier", risk_config.sl_atr_multiplier)
        tp1_mult = tp_sl_config.get("atr_tp1_multiplier", risk_config.tp1_atr_multiplier)
        tp2_mult = tp_sl_config.get("atr_tp2_multiplier", risk_config.tp2_atr_multiplier)
        min_sl_pct = tp_sl_config.get("min_sl_percent", 0.05)

        calculator = TPSLCalculator(min_risk_reward=risk_config.min_risk_reward, min_sl_percent=min_sl_pct)
        levels = calculator.calculate(
            entry=entry,
            direction=direction,
            atr=atr,
            sl_multiplier=sl_mult,
            tp1_multiplier=tp1_mult,
            tp2_multiplier=tp2_mult,
            market_regime=regime_info['regime'],
            adx_strength=regime_info['adx_strength'],
        )

        if not levels.is_valid:
            logger.info("%s: Signal rejected - %s", symbol, levels.rejection_reason)
            return None

        sl = float(levels.stop_loss)
        tp1 = float(levels.take_profit_1)
        tp2 = float(levels.take_profit_2)

        logger.debug("%s %s signal | Entry: %.6f | SL: %.6f | TP1: %.6f | TP2: %.6f | Conf: %.1f%%",
                   symbol, direction, entry, sl, tp1, tp2, confidence)

        return DIYSignal(
            symbol=symbol,
            direction=direction,
            timestamp=human_ts(),
            confidence=float(confidence),
            long_score=float(long_score),
            short_score=float(short_score),
            entry=entry,
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp2,
            current_price=float(current_price),
        )

    def _format_message(self, signal: DIYSignal) -> str:
        """Format Telegram message for signal using centralized template."""
        # Get performance stats (ALWAYS included)
        tp1_count = 0
        tp2_count = 0
        sl_count = 0
        avg_pnl = 0.0
        if self.stats is not None:
            symbol_key = signal.symbol
            counts = self.stats.symbol_tp_sl_counts(symbol_key)
            tp1_count = counts.get("TP1", 0)
            tp2_count = counts.get("TP2", 0)
            sl_count = counts.get("SL", 0)
            avg_pnl = self.stats.get_avg_pnl(symbol_key) if hasattr(self.stats, 'get_avg_pnl') else 0.0
        total = tp1_count + tp2_count + sl_count
        perf_stats = {
            "tp1": tp1_count,
            "tp2": tp2_count,
            "sl": sl_count,
            "wins": tp1_count + tp2_count,
            "total": total,
            "avg_pnl": avg_pnl,
        }

        # Build extra info with confluence scores
        extra_info = f"Long: {signal.long_score:.0f} | Short: {signal.short_score:.0f}"

        current_price = signal.current_price if isinstance(signal.current_price, (int, float)) else None

        return format_signal_message(
            bot_name="DIY",
            symbol=signal.symbol,
            direction=signal.direction,
            entry=signal.entry,
            stop_loss=signal.stop_loss,
            tp1=signal.take_profit_1,
            tp2=signal.take_profit_2,
            confidence=signal.confidence,
            exchange="MEXC",
            timeframe="5m",
            current_price=current_price,
            performance_stats=perf_stats,
            extra_info=extra_info,
        )

    def _monitor_open_signals(self) -> List[Dict[str, Any]]:
        """Monitor open signals for TP/SL hits using batch ticker fetches.

        ISSUE #3 FIX: Thread-safe signal monitoring with locking to prevent race conditions.
        Uses signal_monitor_lock to ensure only one cycle monitors/updates signals at a time.

        Returns:
            List of closed signals with their results (TP1/TP2/SL).
        """
        # Acquire lock to prevent concurrent cycles from missing TP/SL hits
        with self.signal_monitor_lock:
            hit_signals: List[Dict[str, Any]] = []
            signals = self.state.iter_signals()
            if not signals:
                return hit_signals

            # HIGH PRIORITY FIX: Collect all valid symbols first for batch fetching
            valid_signals: Dict[str, Dict[str, Any]] = {}
            invalid_signal_ids: List[str] = []

            for signal_id, payload in list(signals.items()):
                if not isinstance(payload, dict):
                    invalid_signal_ids.append(signal_id)
                    continue

                symbol = payload.get("symbol")
                if not isinstance(symbol, str):
                    invalid_signal_ids.append(signal_id)
                    continue

                direction = payload.get("direction")
                if not isinstance(direction, str):
                    invalid_signal_ids.append(signal_id)
                    continue

                entry_raw = payload.get("entry")
                tp1_raw = payload.get("take_profit_1")
                tp2_raw = payload.get("take_profit_2")
                sl_raw = payload.get("stop_loss")

                if not all(isinstance(v, (int, float, str)) for v in (entry_raw, tp1_raw, tp2_raw, sl_raw)):
                    invalid_signal_ids.append(signal_id)
                    continue

                try:
                    valid_signals[signal_id] = {
                        "symbol": symbol,
                        "direction": direction,
                        "entry": float(entry_raw),  # type: ignore[arg-type]
                        "tp1": float(tp1_raw),      # type: ignore[arg-type]
                        "tp2": float(tp2_raw),      # type: ignore[arg-type]
                        "sl": float(sl_raw),        # type: ignore[arg-type]
                        "original_payload": payload,  # Preserve original for archiving
                    }
                except (ValueError, TypeError):
                    invalid_signal_ids.append(signal_id)
                    continue

            # Clean up invalid signals
            for signal_id in invalid_signal_ids:
                self.state.remove_signal(signal_id)
                if self.stats:
                    self.stats.discard(signal_id)

            if not valid_signals:
                return hit_signals

            # HIGH PRIORITY FIX: Batch fetch all tickers at once (1 API call instead of N)
            symbols_to_fetch = list(set(sig["symbol"] for sig in valid_signals.values()))
            try:
                tickers = self.client.fetch_tickers(symbols_to_fetch)
                logger.debug("Batch fetched %d tickers for %d open signals",
                            len(tickers), len(valid_signals))
            except Exception as exc:
                logger.warning("Batch ticker fetch failed: %s", exc)
                return hit_signals

            # Process each signal with the fetched prices
            price_tolerance = self.config.get("tp_sl", {}).get("price_tolerance", 0.005)

            for signal_id, sig_data in valid_signals.items():
                symbol = sig_data["symbol"]
                ticker = tickers.get(symbol)

                if not ticker:
                    logger.debug("No ticker data for %s", symbol)
                    continue

                price = ticker.get("last") if isinstance(ticker, dict) else None
                if price is None and isinstance(ticker, dict):
                    price = ticker.get("close")

                if not isinstance(price, (int, float)):
                    continue

                direction = sig_data["direction"]
                entry = sig_data["entry"]
                tp1 = sig_data["tp1"]
                tp2 = sig_data["tp2"]
                sl = sig_data["sl"]

                # Check for TP/SL hits with price tolerance
                # Priority: TP2 > TP1 > SL (handled by if-elif ordering)
                # For BULLISH: price goes UP to hit TPs, DOWN to hit SL
                # For BEARISH: price goes DOWN to hit TPs, UP to hit SL
                if direction == "BULLISH":
                    tp2_threshold = tp2 * (1 - price_tolerance)
                    tp1_threshold = tp1 * (1 - price_tolerance)
                    sl_threshold = sl * (1 + price_tolerance)
                    hit_tp2 = price >= tp2_threshold
                    hit_tp1 = price >= tp1_threshold
                    hit_sl = price <= sl_threshold
                else:
                    tp2_threshold = tp2 * (1 + price_tolerance)
                    tp1_threshold = tp1 * (1 + price_tolerance)
                    sl_threshold = sl * (1 - price_tolerance)
                    hit_tp2 = price <= tp2_threshold
                    hit_tp1 = price <= tp1_threshold
                    hit_sl = price >= sl_threshold

                # Determine result - TP2 takes priority over TP1, both take priority over SL
                result = None
                if hit_tp2:
                    result = "TP2"
                elif hit_tp1:
                    result = "TP1"
                elif hit_sl:
                    result = "SL"

                if result:
                    # Calculate PnL
                    if direction == "BULLISH":
                        pnl_pct = ((price - entry) / entry) * 100
                    else:
                        pnl_pct = ((entry - price) / entry) * 100

                    # Add to hit signals list
                    hit_signals.append({
                        "signal_id": signal_id,
                        "symbol": symbol,
                        "direction": direction,
                        "result": result,
                        "entry": entry,
                        "exit": price,
                        "pnl_pct": pnl_pct,
                        "tp1": tp1,
                        "tp2": tp2,
                        "sl": sl,
                    })

                    # Record close in stats
                    if self.stats:
                        try:
                            stats_record = self.stats.record_close(
                                signal_id,
                                exit_price=price,
                                result=result,
                            )
                            if stats_record:
                                logger.info("Trade closed: %s | %s | Entry: %.6f | Exit: %.6f | Result: %s | P&L: %.2f%%",
                                           signal_id, symbol, entry, price, result, stats_record.pnl_pct)
                            else:
                                self.stats.discard(signal_id)
                        except Exception as stats_err:
                            logger.error("Failed to record stats for %s: %s", signal_id, stats_err)

                    # Check if result notifications are enabled
                    enable_result_notifs = self.config.get("telegram", {}).get("enable_result_notifications", True)
                    cooldown = self.config.get("signal", {}).get("result_notification_cooldown_minutes", 15)
                    if not enable_result_notifs:
                        logger.debug("Result notifications disabled, skipping for %s", signal_id)
                        should_notify = False
                    else:
                        # Check if we should notify (cooldown prevention)
                        should_notify = self.state.should_notify_result(symbol, cooldown)

                    if should_notify:
                        # Send enhanced result notification
                        message = self._format_result_message(
                            signal_id=signal_id,
                            symbol=symbol,
                            direction=direction,
                            entry=entry,
                            exit=price,
                            result=result,
                            sl=sl,
                            tp1=tp1,
                            tp2=tp2
                        )
                        self._dispatch(message)
                        self.state.mark_result_notified(symbol)
                        logger.info("📤 Result notification sent for %s", signal_id)
                    else:
                        logger.info("⏭️  Skipping duplicate result for %s (cooldown: %dm)", symbol, cooldown)

                    # Archive to closed_signals before removing
                    # Use original_payload to preserve all fields (created_at, timeframe, etc.)
                    original_payload = sig_data.get("original_payload", sig_data)
                    self.state.archive_signal(signal_id, original_payload, price, result, pnl_pct)
                    self.state.remove_signal(signal_id)

        return hit_signals

    def _dispatch(self, message: str) -> None:
        """Send message via Telegram."""
        if self.notifier:
            try:
                if self.notifier.send_message(message):
                    logger.info("Alert sent to Telegram")
                else:
                    logger.error("Failed to send Telegram message")
                    logger.info("%s", message)
            except Exception as notify_err:
                logger.error("Telegram notification error: %s", notify_err)
                logger.info("Message was: %s", message)
        else:
            logger.info("Alert:\n%s", message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DIY Multi-Indicator Bot")
    parser.add_argument("--once", action="store_true", help="Run only one cycle")
    parser.add_argument("--config", type=str, default=str(CONFIG_FILE), help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--track", action="store_true", help="Track-only mode: check open signals without generating new ones")
    parser.add_argument("--validate", action="store_true", help="Validate environment and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Override log level if --debug is passed
    if args.debug:
        global logger
        logger = setup_logging("DEBUG", enable_detailed=True)

    # Validate environment only mode
    if args.validate:
        if validate_environment():
            logger.info("Environment validation passed!")
            sys.exit(0)
        else:
            logger.error("Environment validation failed!")
            sys.exit(1)

    # Validate environment before starting
    if not validate_environment():
        logger.warning("Environment validation failed, but continuing...")

    bot = DIYBot(config_path=args.config)
    bot.run(loop=not args.once, track_only=args.track)


if __name__ == "__main__":
    main()
