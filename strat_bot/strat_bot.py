#!/usr/bin/env python3
"""
STRAT Bot - Rob Smith's 1-2-3 Price Action System
Detects inside bars (1), directional bars (2u/2d), outside bars (3),
and actionable STRAT combos for high-probability trades
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal as signal_module
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import FrameType
from typing import Any, Dict, List, Optional, Tuple, TypedDict, cast

import ccxt  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
STATE_FILE = BASE_DIR / "strat_state.json"
WATCHLIST_FILE = BASE_DIR / "strat_watchlist.json"
STATS_FILE = LOG_DIR / "strat_stats.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "strat_bot.log"),
    ],
)

logger = logging.getLogger("strat_bot")

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
    load_dotenv(BASE_DIR.parent / ".env")
except ImportError:
    pass

sys.path.append(str(BASE_DIR.parent))

# Required imports (fail fast if missing)
from message_templates import format_signal_message
from notifier import TelegramNotifier
from signal_stats import SignalStats
from tp_sl_calculator import TPSLCalculator
from trade_config import get_config_manager

# Optional imports (safe fallback)
from safe_import import safe_import
HealthMonitor = safe_import('health_monitor', 'HealthMonitor')
RateLimiter = None  # Disabled for testing
RateLimitHandler = safe_import('rate_limit_handler', 'RateLimitHandler')
class WatchItem(TypedDict, total=False):
    symbol: str
    period: str
    cooldown_minutes: int


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
    skipped_count = 0
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            logger.warning("Watchlist item #%d is not a dictionary, skipping", idx)
            skipped_count += 1
            continue

        symbol_val = item.get("symbol")
        if not isinstance(symbol_val, str) or not symbol_val:
            logger.warning("Watchlist item #%d has invalid/missing symbol, skipping: %s", idx, item)
            skipped_count += 1
            continue

        period_val = item.get("period", "5m")
        period = period_val if isinstance(period_val, str) else "5m"
        cooldown_val = item.get("cooldown_minutes", 5)
        try:
            cooldown = int(cooldown_val)
        except Exception:
            cooldown = 5

        normalized.append({
            "symbol": symbol_val.upper(),
            "period": period,
            "cooldown_minutes": cooldown,
        })

    if skipped_count > 0:
        logger.warning("Skipped %d invalid watchlist items", skipped_count)

    return normalized


def human_ts() -> str:
    """Return human-readable UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class STRATAnalyzer:
    """Analyzes STRAT 1-2-3 bar patterns and combos."""

    @staticmethod
    def get_bar_type(high: float, low: float, prev_high: float, prev_low: float) -> str:
        """
        Determine bar type:
        1 = Inside bar (high <= prev_high AND low >= prev_low)
        2u = 2 Up (high > prev_high, no lower low)
        2d = 2 Down (low < prev_low, no higher high)
        3 = Outside bar (high > prev_high AND low < prev_low)
        """
        is_higher_high = high > prev_high
        is_lower_low = low < prev_low

        if is_higher_high and is_lower_low:
            return '3'
        elif is_higher_high and not is_lower_low:
            return '2u'
        elif is_lower_low and not is_higher_high:
            return '2d'
        else:
            return '1'

    def classify_candles(self, highs: npt.NDArray[np.floating[Any]], lows: npt.NDArray[np.floating[Any]]) -> List[str]:
        """Classify all candles as 1, 2u, 2d, or 3."""
        bar_types = ['1']  # First bar is neutral

        for i in range(1, len(highs)):
            bar_type = self.get_bar_type(highs[i], lows[i], highs[i-1], lows[i-1])
            bar_types.append(bar_type)

        return bar_types

    def detect_actionable_patterns(
        self,
        opens: npt.NDArray[np.floating[Any]],
        highs: npt.NDArray[np.floating[Any]],
        lows: npt.NDArray[np.floating[Any]],
        closes: npt.NDArray[np.floating[Any]],
    ) -> Optional[Tuple[str, str]]:
        """Detect actionable Hammer/Shooter patterns using last closed bar."""
        if len(closes) < 3:
            return None
        current = len(closes) - 2
        candle_range = highs[current] - lows[current]
        if candle_range == 0:
            return None

        wick_threshold = candle_range * 0.75
        shooter_top = highs[current] - wick_threshold
        hammer_bottom = lows[current] + wick_threshold

        # Shooter: open and close in bottom 25% (bearish)
        if opens[current] < shooter_top and closes[current] < shooter_top:
            return ('SHOOTER', 'BEARISH')

        # Hammer: open and close in top 25% (bullish)
        if opens[current] > hammer_bottom and closes[current] > hammer_bottom:
            return ('HAMMER', 'BULLISH')

        return None

    def detect_strat_combos(self, bar_types: List[str]) -> List[Tuple[str, str]]:
        """
        Detect STRAT combos from bar sequence.
        Returns list of (combo_name, direction) tuples.
        """
        if len(bar_types) < 4:
            return []
        closed = bar_types[:-1]
        if len(closed) < 3:
            return []
        combos: List[Tuple[str, str]] = []
        b0, b1, b2 = closed[-1], closed[-2], closed[-3]

        # 222 Continuations (strong momentum)
        if b0 == '2d' and b1 == '2d' and b2 == '2d':
            combos.append(('222 Bearish Continuation', 'BEARISH'))
        if b0 == '2u' and b1 == '2u' and b2 == '2u':
            combos.append(('222 Bullish Continuation', 'BULLISH'))

        # 212 Measured Move (consolidation then continuation)
        if b0 == '2d' and b1 == '1' and b2 == '2d':
            combos.append(('212 Bearish Measured Move', 'BEARISH'))
        if b0 == '2u' and b1 == '1' and b2 == '2u':
            combos.append(('212 Bullish Measured Move', 'BULLISH'))

        # 212 Reversals
        if b0 == '2d' and b1 == '1' and b2 == '2u':
            combos.append(('212 Bearish Reversal', 'BEARISH'))
        if b0 == '2u' and b1 == '1' and b2 == '2d':
            combos.append(('212 Bullish Reversal', 'BULLISH'))

        # 22 Reversals (simple direction change)
        if b0 == '2d' and b1 == '2u':
            combos.append(('22 Bearish Reversal', 'BEARISH'))
        if b0 == '2u' and b1 == '2d':
            combos.append(('22 Bullish Reversal', 'BULLISH'))

        # 322 Reversals (outside bar followed by reversal)
        if b0 == '2d' and b1 == '2u' and b2 == '3':
            combos.append(('322 Bearish Reversal', 'BEARISH'))
        if b0 == '2u' and b1 == '2d' and b2 == '3':
            combos.append(('322 Bullish Reversal', 'BULLISH'))

        # 32 Reversals (outside bar then directional)
        if b0 == '2d' and b1 == '3':
            combos.append(('32 Bearish Reversal', 'BEARISH'))
        if b0 == '2u' and b1 == '3':
            combos.append(('32 Bullish Reversal', 'BULLISH'))

        return combos

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
    """Binance USDM exchange client wrapper."""

    def __init__(self) -> None:
        self.exchange: Any = ccxt.binanceusdm({
            "enableRateLimit": True,
            "options": {"defaultType": "swap"}
        })
        self.exchange.load_markets()
        self.rate_limiter: Any = RateLimitHandler(base_delay=0.5, max_retries=5) if RateLimitHandler else None

    @staticmethod
    def _swap_symbol(symbol: str) -> str:
        # Handle both FHE and FHE/USDT formats
        sym = symbol.upper().replace("/USDT", "")
        return f"{sym}/USDT:USDT"

    def fetch_ohlcv(self, symbol: str, timeframe: str = "5m", limit: int = 100) -> List[Any]:
        """Fetch OHLCV data."""
        if self.rate_limiter:
            return cast(List[Any], self.rate_limiter.execute(
                self.exchange.fetch_ohlcv,
                self._swap_symbol(symbol),
                timeframe=timeframe,
                limit=limit
            ))
        return cast(List[Any], self.exchange.fetch_ohlcv(
            self._swap_symbol(symbol),
            timeframe=timeframe,
            limit=limit
        ))

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch ticker data."""
        if self.rate_limiter:
            result = self.rate_limiter.execute(self.exchange.fetch_ticker, self._swap_symbol(symbol))
            return dict(result) if result else {}
        result = self.exchange.fetch_ticker(self._swap_symbol(symbol))
        return dict(result) if result else {}


@dataclass
class STRATSignal:
    """Represents a STRAT pattern signal."""
    symbol: str
    pattern_name: str
    direction: str
    timestamp: str
    bar_sequence: str
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    current_price: Optional[float] = None


OpenSignals = Dict[str, Dict[str, Any]]


class BotState:
    """Manages bot state persistence."""

    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, Any] = self._load()

    @staticmethod
    def _parse_ts(value: str) -> datetime:
        """Parse ISO format timestamp, ensuring UTC timezone."""
        try:
            dt = datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return datetime.now(timezone.utc)

        # Ensure UTC timezone
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            # Convert to UTC if not already
            dt = dt.astimezone(timezone.utc)
        return dt

    def _empty_state(self) -> Dict[str, Any]:
        return {"last_alert": {}, "open_signals": {}, "closed_signals": {}}

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return self._empty_state()
        try:
            data = json.loads(self.path.read_text())
            if not isinstance(data, dict):
                return self._empty_state()
            last_alert = data.get("last_alert")
            open_signals = data.get("open_signals")
            closed_signals = data.get("closed_signals")
            return {
                "last_alert": last_alert if isinstance(last_alert, dict) else {},
                "open_signals": open_signals if isinstance(open_signals, dict) else {},
                "closed_signals": closed_signals if isinstance(closed_signals, dict) else {},
            }
        except json.JSONDecodeError:
            return self._empty_state()

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2))

    def can_alert(self, symbol: str, cooldown_minutes: int) -> bool:
        """Check if enough time has passed since last alert for this symbol."""
        last_map = self.data.setdefault("last_alert", {})
        if not isinstance(last_map, dict):
            self.data["last_alert"] = {}
            return True
        last_ts = last_map.get(symbol)
        if not isinstance(last_ts, str):
            return True

        # Parse the last alert time and get current time once
        try:
            last_time = self._parse_ts(last_ts)
            current_time = datetime.now(timezone.utc)
            delta = current_time - last_time
            return delta >= timedelta(minutes=cooldown_minutes)
        except (ValueError, TypeError):
            return True

    def mark_alert(self, symbol: str) -> None:
        """Record the time of the last alert for this symbol."""
        last_map = self.data.setdefault("last_alert", {})
        if not isinstance(last_map, dict):
            last_map = {}
            self.data["last_alert"] = last_map
        # Store ISO format with explicit UTC timezone
        last_map[symbol] = datetime.now(timezone.utc).isoformat()
        self.save()

    def cleanup_stale_signals(self, max_age_hours: int = 24) -> int:
        """Remove signals older than max_age_hours and move to closed_signals."""
        signals = self.iter_signals()
        closed = self.data.setdefault("closed_signals", {})
        if not isinstance(closed, dict):
            closed = {}
            self.data["closed_signals"] = closed

        current_time = datetime.now(timezone.utc)
        removed_count = 0
        signal_ids_to_remove = []

        for signal_id, payload in list(signals.items()):
            if not isinstance(payload, dict):
                signal_ids_to_remove.append(signal_id)
                continue

            created_at_str = payload.get("created_at")
            if not isinstance(created_at_str, str):
                signal_ids_to_remove.append(signal_id)
                continue

            try:
                created_time = self._parse_ts(created_at_str)
                age = current_time - created_time

                if age >= timedelta(hours=max_age_hours):
                    # Move to closed signals with timeout status
                    closed[signal_id] = {**payload, "closed_reason": "TIMEOUT", "closed_at": current_time.isoformat()}
                    signal_ids_to_remove.append(signal_id)
                    removed_count += 1
                    logger.info("Stale signal removed: %s (age: %.1f hours)", signal_id, age.total_seconds() / 3600)
            except (ValueError, TypeError):
                signal_ids_to_remove.append(signal_id)

        for signal_id in signal_ids_to_remove:
            if signal_id in signals:
                signals.pop(signal_id)

        if removed_count > 0:
            self.save()

        return removed_count

    def add_signal(self, signal_id: str, payload: Dict[str, Any]) -> None:
        self.data.setdefault("open_signals", {})[signal_id] = payload
        self.save()

    def remove_signal(self, signal_id: str) -> None:
        signals = self.iter_signals()
        if signal_id in signals:
            signals.pop(signal_id)
            self.save()

    def iter_signals(self) -> OpenSignals:
        signals = self.data.setdefault("open_signals", {})
        if not isinstance(signals, dict):
            signals = {}
            self.data["open_signals"] = signals
        return cast(OpenSignals, signals)


class STRATBot:
    """Main STRAT Bot."""

    # Configuration constants
    MAX_OPEN_SIGNALS = 50  # Maximum concurrent signals
    MIN_RISK_PCT = 0.003  # 0.3% minimum risk threshold (increased from 0.1%)
    MAX_PRICE_DEVIATION_PCT = 0.02  # 2% max price deviation for staleness check
    MAX_VOLATILITY_PCT = 0.08  # 8% max ATR/price ratio (volatility filter)
    USE_ATR_TARGETS = True  # Use ATR for dynamic TP/SL sizing
    ATR_TP1_MULTIPLIER = 2.5  # ATR multiplier for TP1
    ATR_TP2_MULTIPLIER = 4.0  # ATR multiplier for TP2
    MIN_RR_RATIO = 1.8  # Minimum risk/reward ratio

    def __init__(self, interval: int = 60, default_cooldown: int = 5):
        self.interval = interval
        self.default_cooldown = default_cooldown
        self.watchlist: List[WatchItem] = load_watchlist()
        self.client = MexcClient()
        self.analyzer = STRATAnalyzer()
        self.state = BotState(STATE_FILE)
        self.notifier = self._init_notifier()
        self.stats = SignalStats("STRAT Bot", STATS_FILE)
        self.health_monitor = (
            HealthMonitor("STRAT Bot", self.notifier, heartbeat_interval=3600)
            if HealthMonitor and self.notifier else None
        )
        # Removed duplicate RateLimiter - using RateLimitHandler in MexcClient instead
        self.exchange_backoff: Dict[str, float] = {}
        self.exchange_delay: Dict[str, float] = {}
        # Cache for OHLCV data to reduce API calls
        self.ohlcv_cache: Dict[str, Tuple[List[Any], float]] = {}  # {"SYMBOL-5m": (data, timestamp)}
        self.cache_ttl = 60  # Cache TTL in seconds
        self.max_cache_size = 100  # Maximum cache entries
        # Graceful shutdown
        self.shutdown_requested = False
        signal_module.signal(signal_module.SIGTERM, self._signal_handler)
        signal_module.signal(signal_module.SIGINT, self._signal_handler)

    def _init_notifier(self) -> Optional[Any]:
        if TelegramNotifier is None:
            logger.warning("Telegram notifier unavailable")
            return None
        token = os.getenv("TELEGRAM_BOT_TOKEN_STRAT") or os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            logger.warning("Telegram credentials missing")
            return None
        return TelegramNotifier(
            bot_token=token,
            chat_id=chat_id,
            signals_log_file=str(LOG_DIR / "strat_signals.json")
        )

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """Check if exception is a rate limit error."""
        # Check exception type first
        if hasattr(ccxt, 'RateLimitExceeded') and isinstance(exc, ccxt.RateLimitExceeded):
            return True
        if hasattr(ccxt, 'DDoSProtection') and isinstance(exc, ccxt.DDoSProtection):
            return True

        # Fallback to message checking with more precise patterns
        msg = str(exc).lower()
        rate_limit_indicators = [
            "requests are too frequent",
            "rate limit",
            "too many requests",
            "ddos protection",
        ]

        # Check for 429 status code specifically (not just the number)
        if "429" in msg and ("status" in msg or "code" in msg or "error" in msg):
            return True

        # Check for rate limit messages
        return any(indicator in msg for indicator in rate_limit_indicators)

    def _backoff_active(self, exchange: str) -> bool:
        until = self.exchange_backoff.get(exchange)
        return bool(until and time.time() < until)

    def _register_backoff(self, exchange: str, base_delay: float = 60, max_delay: float = 300) -> None:
        prev = self.exchange_delay.get(exchange, base_delay)
        new_delay = min(prev * 2, max_delay) if exchange in self.exchange_backoff else base_delay
        until = time.time() + new_delay
        self.exchange_backoff[exchange] = until
        self.exchange_delay[exchange] = new_delay
        logger.warning("%s rate limit; backing off for %.0fs", exchange, new_delay)

    def _signal_handler(self, signum: int, frame: Optional[FrameType]) -> None:
        """Handle shutdown signals gracefully."""
        logger.info("Received signal %d, initiating graceful shutdown...", signum)
        self.shutdown_requested = True

    def _cleanup_cache(self) -> None:
        """Remove old cache entries to prevent memory leak."""
        if len(self.ohlcv_cache) <= self.max_cache_size:
            return

        now = time.time()
        # Remove entries older than TTL
        expired_keys = [
            key for key, (_, timestamp) in self.ohlcv_cache.items()
            if now - timestamp > self.cache_ttl
        ]

        for key in expired_keys:
            del self.ohlcv_cache[key]

        # If still too large, remove oldest entries
        if len(self.ohlcv_cache) > self.max_cache_size:
            sorted_items = sorted(self.ohlcv_cache.items(), key=lambda x: x[1][1])
            keys_to_remove = [k for k, _ in sorted_items[:len(self.ohlcv_cache) - self.max_cache_size]]
            for key in keys_to_remove:
                del self.ohlcv_cache[key]
            logger.debug("Cache cleanup: removed %d entries", len(keys_to_remove))

    def _fetch_ohlcv_cached(self, symbol: str, timeframe: str, limit: int = 50) -> Optional[List[Any]]:
        """Fetch OHLCV with caching to reduce API calls."""
        cache_key = f"{symbol}-{timeframe}"
        now = time.time()

        # Periodic cache cleanup
        self._cleanup_cache()

        # Check cache
        if cache_key in self.ohlcv_cache:
            cached_data, cached_time = self.ohlcv_cache[cache_key]
            if now - cached_time < self.cache_ttl:
                return cached_data

        # Fetch fresh data
        try:
            ohlcv = self.client.fetch_ohlcv(symbol, timeframe, limit)
            self.ohlcv_cache[cache_key] = (ohlcv, now)
            return ohlcv
        except Exception as exc:
            logger.warning("Failed to fetch OHLCV for %s: %s", symbol, exc)
            # Return cached data if available, even if stale
            if cache_key in self.ohlcv_cache:
                return self.ohlcv_cache[cache_key][0]
            return None

    def run(self, loop: bool = False) -> None:
        if not self.watchlist:
            logger.error("Empty watchlist; exiting")
            return

        logger.info("Starting STRAT Bot for %d symbols", len(self.watchlist))

        if self.health_monitor:
            self.health_monitor.send_startup_message()

        try:
            while True:
                if self.shutdown_requested:
                    logger.info("Shutdown requested, exiting gracefully...")
                    break

                try:
                    self._run_cycle()

                    # Cleanup stale signals every cycle
                    stale_count = self.state.cleanup_stale_signals(max_age_hours=24)
                    if stale_count > 0:
                        logger.info("Cleaned up %d stale signals", stale_count)

                    self._monitor_open_signals()

                    if self.health_monitor:
                        self.health_monitor.record_cycle()

                    if not loop:
                        break
                    logger.info("Cycle complete; sleeping %ds", self.interval)
                    # Sleep in 1-second chunks to respond quickly to shutdown signals
                    for _ in range(self.interval):
                        if self.shutdown_requested:
                            break
                        time.sleep(1)
                except Exception as exc:
                    logger.error("Error in cycle: %s", exc)
                    if self.health_monitor:
                        self.health_monitor.record_error(str(exc))
                    if not loop:
                        raise
                    time.sleep(10)
        finally:
            if self.health_monitor:
                self.health_monitor.send_shutdown_message()

    def _run_cycle(self) -> None:
        for item in self.watchlist:
            symbol_val = item.get("symbol") if isinstance(item, dict) else None
            if not isinstance(symbol_val, str):
                continue
            symbol = symbol_val
            period_val = item.get("period") if isinstance(item, dict) else None
            period = period_val if isinstance(period_val, str) else "5m"
            cooldown_raw = item.get("cooldown_minutes") if isinstance(item, dict) else None
            try:
                cooldown = int(cooldown_raw) if cooldown_raw is not None else self.default_cooldown
            except Exception:
                cooldown = self.default_cooldown

            # Check if backoff is active for exchange
            if self._backoff_active("binanceusdm"):
                logger.debug("Backoff active for binanceusdm; skipping %s", symbol)
                continue

            try:
                signal = self._analyze_symbol(symbol, period)
            except Exception as exc:
                logger.error("Failed to analyze %s: %s", symbol, exc)
                # Handle rate limit errors (consolidated check)
                if self._is_rate_limit_error(exc):
                    self._register_backoff("binanceusdm")
                if self.health_monitor:
                    self.health_monitor.record_error(f"Analysis error for {symbol}: {exc}")
                continue

            if signal is None:
                logger.debug("%s: No STRAT pattern", symbol)
                continue

            if not self.state.can_alert(symbol, cooldown):
                logger.debug("Cooldown active for %s", symbol)
                continue

            # Max open signals limit
            current_open = len(self.state.iter_signals())
            if current_open >= self.MAX_OPEN_SIGNALS:
                logger.info(
                    "Max open signals limit reached (%d/%d). Skipping %s",
                    current_open, self.MAX_OPEN_SIGNALS, symbol
                )
                continue

            # Send alert
            message = self._format_message(signal)
            self._dispatch(message)
            self.state.mark_alert(symbol)

            # Track signal
            signal_id = f"{symbol}-STRAT-{signal.timestamp}"
            trade_data = {
                "id": signal_id,
                "symbol": symbol,
                "pattern": signal.pattern_name,
                "direction": signal.direction,
                "entry": signal.entry,
                "stop_loss": signal.stop_loss,
                "take_profit_1": signal.take_profit_1,
                "take_profit_2": signal.take_profit_2,
                "created_at": signal.timestamp,
                "timeframe": period,
                "exchange": "binanceusdm",
            }
            self.state.add_signal(signal_id, trade_data)

            if self.stats:
                # Clean symbol format - avoid double /USDT
                clean_symbol = symbol.replace("/USDT", "") + "/USDT"
                self.stats.record_open(
                    signal_id=signal_id,
                    symbol=clean_symbol,
                    direction=signal.direction,
                    entry=signal.entry,
                    created_at=signal.timestamp,
                    extra={
                        "timeframe": period,
                        "exchange": "binanceusdm",
                        "strategy": "STRAT",
                        "pattern": signal.pattern_name,
                    },
                )

            time.sleep(0.5)

    def _analyze_symbol(self, symbol: str, timeframe: str) -> Optional[STRATSignal]:
        """Analyze symbol for STRAT patterns."""
        # Fetch OHLCV data with caching
        ohlcv = self._fetch_ohlcv_cached(symbol, timeframe, limit=50)

        if ohlcv is None:
            return None

        # Need at least 20 bars for reliable pattern detection and SMA calculation
        if len(ohlcv) < 20:
            logger.debug("Insufficient OHLCV data for %s: %d bars (need 20+)", symbol, len(ohlcv))
            return None

        opens = np.array([x[1] for x in ohlcv])
        highs = np.array([x[2] for x in ohlcv])
        lows = np.array([x[3] for x in ohlcv])
        closes = np.array([x[4] for x in ohlcv])
        if len(highs) < 3:
            return None
        # Get current price first for early filtering
        try:
            ticker = self.client.fetch_ticker(symbol)
            current_price_val = ticker.get("last") or ticker.get("close")
            if current_price_val is None:
                return None
            current_price = float(current_price_val)
        except (TypeError, ValueError, Exception) as exc:
            logger.debug("Failed to get current price for %s: %s", symbol, exc)
            return None

        # Calculate SMA for early filtering (before expensive pattern detection)
        closes_for_sma = closes[:-1] if len(closes) > 1 else closes
        if len(closes_for_sma) >= 50:
            sma50 = float(np.mean(closes_for_sma[-50:]))
        elif len(closes_for_sma) >= 20:
            sma50 = float(np.mean(closes_for_sma[-20:]))
            logger.debug("Using SMA20 fallback for %s (insufficient data for SMA50)", symbol)
        else:
            sma50 = None
            logger.debug("Skipping SMA filter for %s (insufficient data)", symbol)

        # Classify bars
        bar_types = self.analyzer.classify_candles(highs, lows)

        # Detect STRAT combos
        combos = self.analyzer.detect_strat_combos(bar_types)

        # Detect Hammer/Shooter
        actionable = self.analyzer.detect_actionable_patterns(opens, highs, lows, closes)

        # Prioritize patterns by strength: 222 > 212 Measured > 322 > 212 Reversal > 32 > 22 > Hammer/Shooter
        pattern_priority = {
            '222 Bullish Continuation': 1, '222 Bearish Continuation': 1,
            '212 Bullish Measured Move': 2, '212 Bearish Measured Move': 2,
            '322 Bullish Reversal': 3, '322 Bearish Reversal': 3,
            '212 Bullish Reversal': 4, '212 Bearish Reversal': 4,
            '32 Bullish Reversal': 5, '32 Bearish Reversal': 5,
            '22 Bullish Reversal': 6, '22 Bearish Reversal': 6,
        }

        if combos:
            # Sort combos by priority (lower number = higher priority)
            sorted_combos = sorted(combos, key=lambda x: pattern_priority.get(x[0], 999))
            pattern_name, direction = sorted_combos[0]
        elif actionable:
            pattern_name, direction = actionable
        else:
            return None

        # Apply SMA filter early (before calculating targets)
        if sma50 is not None:
            if direction == "BULLISH" and current_price <= sma50:
                logger.debug("%s: BULLISH signal rejected - price %.6f below SMA %.6f", symbol, current_price, sma50)
                return None
            if direction == "BEARISH" and current_price >= sma50:
                logger.debug("%s: BEARISH signal rejected - price %.6f above SMA %.6f", symbol, current_price, sma50)
                return None

        setup_high = float(highs[-2])
        setup_low = float(lows[-2])
        setup_close = float(closes[-2])

        # Use current price as entry (more realistic) instead of trigger price
        entry = current_price
        stop_loss = setup_low if direction == "BULLISH" else setup_high

        # Calculate ATR for volatility filter and target sizing
        atr = float(self.analyzer.calculate_atr(highs, lows, closes))

        if atr == 0:
            logger.debug("%s: ATR is 0, skipping", symbol)
            return None

        # Volatility filter - reject if ATR is too high relative to price
        volatility_ratio = atr / entry if entry > 0 else 0
        if volatility_ratio > self.MAX_VOLATILITY_PCT:
            logger.debug(
                "%s: Volatility too high: %.2f%% (max %.2f%%)",
                symbol, volatility_ratio * 100, self.MAX_VOLATILITY_PCT * 100
            )
            return None

        # Signal staleness check - reject if price moved too far from setup close
        price_deviation = abs(current_price - setup_close) / setup_close if setup_close > 0 else 0
        if price_deviation > self.MAX_PRICE_DEVIATION_PCT:
            logger.debug(
                "%s: Signal stale - price moved %.2f%% from setup (max %.2f%%)",
                symbol, price_deviation * 100, self.MAX_PRICE_DEVIATION_PCT * 100
            )
            return None

        # Validate stop loss position
        risk = abs(entry - stop_loss)

        # Validate minimum risk threshold (increased to 0.3%)
        min_risk = entry * self.MIN_RISK_PCT
        if risk < min_risk:
            logger.debug("%s: Risk too low: %.6f < %.6f (%.2f%%)", symbol, risk, min_risk, self.MIN_RISK_PCT * 100)
            return None

        if direction == "BULLISH" and stop_loss >= entry:
            logger.debug("%s: Invalid BULLISH setup - SL %.6f >= entry %.6f", symbol, stop_loss, entry)
            return None
        if direction == "BEARISH" and stop_loss <= entry:
            logger.debug("%s: Invalid BEARISH setup - SL %.6f <= entry %.6f", symbol, stop_loss, entry)
            return None

        # Calculate targets using centralized TPSLCalculator
        # STRAT uses structure-based SL (setup bar high/low), so we pass custom_sl
        config_mgr = get_config_manager()
        risk_config = config_mgr.get_effective_risk("strat_bot", symbol)

        calculator = TPSLCalculator(min_risk_reward=risk_config.min_risk_reward)
        dir_normalized = "LONG" if direction == "BULLISH" else "SHORT"
        levels = calculator.calculate(
            entry=entry,
            direction=dir_normalized,
            atr=atr,
            tp1_multiplier=self.ATR_TP1_MULTIPLIER,
            tp2_multiplier=self.ATR_TP2_MULTIPLIER,
            custom_sl=stop_loss,  # Structure-based SL from setup bar
        )

        if not levels.is_valid:
            logger.debug("%s: TPSLCalculator rejected - %s", symbol, levels.rejection_reason)
            return None

        tp1 = levels.take_profit_1
        tp2 = levels.take_profit_2
        rr_ratio = levels.risk_reward_1

        closed_seq = bar_types[:-1]
        # Show last 6 bars for better context, or all if fewer
        display_count = min(6, len(closed_seq))
        bar_sequence = '-'.join(closed_seq[-display_count:]) if closed_seq else '1'

        # Add detailed debug logging for successful signal
        logger.debug("%s %s STRAT signal | Entry: %.6f | SL: %.6f | TP1: %.6f | TP2: %.6f | R:R: 1:%.2f | Bars: %s",
                   symbol, direction, entry, stop_loss, tp1, tp2, rr_ratio, bar_sequence)

        return STRATSignal(
            symbol=symbol,
            pattern_name=pattern_name,
            direction=direction,
            timestamp=human_ts(),
            bar_sequence=bar_sequence,
            entry=entry,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            current_price=current_price,
        )

    def _format_message(self, signal: STRATSignal) -> str:
        """Format Telegram message for signal using centralized template."""
        # Get performance stats
        perf_stats = None
        # Clean symbol format - avoid double /USDT
        clean_symbol = signal.symbol.replace("/USDT", "") + "/USDT"
        if self.stats is not None:
            symbol_key = clean_symbol
            counts = self.stats.symbol_tp_sl_counts(symbol_key)
            tp1_count = counts.get("TP1", 0)
            tp2_count = counts.get("TP2", 0)
            sl_count = counts.get("SL", 0)
            total = tp1_count + tp2_count + sl_count

            if total > 0:
                perf_stats = {
                    "tp1": tp1_count,
                    "tp2": tp2_count,
                    "sl": sl_count,
                    "wins": tp1_count + tp2_count,
                    "total": total,
                }

        # Build extra info with STRAT-specific data
        extra_info = f"Bar Sequence: {signal.bar_sequence}"

        return format_signal_message(
            bot_name="STRAT",
            symbol=clean_symbol,
            direction=signal.direction,
            entry=signal.entry,
            stop_loss=signal.stop_loss,
            tp1=signal.take_profit_1,
            tp2=signal.take_profit_2,
            pattern_name=signal.pattern_name,
            exchange="binanceusdm",
            timeframe="15m",
            current_price=signal.current_price,
            performance_stats=perf_stats,
            extra_info=extra_info,
        )

    def _monitor_open_signals(self) -> None:
        """Monitor open signals for TP/SL hits."""
        signals = self.state.iter_signals()
        if not signals:
            return

        for signal_id, payload_obj in list(signals.items()):
            if not isinstance(payload_obj, dict):
                self.state.remove_signal(signal_id)
                continue

            symbol_val = payload_obj.get("symbol")
            if not isinstance(symbol_val, str) or not symbol_val:
                self.state.remove_signal(signal_id)
                continue
            symbol = symbol_val

            try:
                ticker = self.client.fetch_ticker(symbol)
                price_raw = ticker.get("last") if isinstance(ticker, dict) else None
                if price_raw is None and isinstance(ticker, dict):
                    price_raw = ticker.get("close")
            except Exception as exc:
                logger.warning("Failed to fetch ticker for %s: %s", signal_id, exc)
                # Handle rate limit errors in monitoring
                if self._is_rate_limit_error(exc):
                    self._register_backoff("binanceusdm")
                    logger.warning("Rate limit hit during monitoring; backing off")
                continue

            if not isinstance(price_raw, (int, float, str)):
                continue
            try:
                price = float(price_raw)
            except (TypeError, ValueError):
                continue

            direction_val = payload_obj.get("direction")
            if not isinstance(direction_val, str):
                self.state.remove_signal(signal_id)
                if self.stats:
                    self.stats.discard(signal_id)
                continue
            direction = direction_val

            entry_raw = payload_obj.get("entry")
            tp1_raw = payload_obj.get("take_profit_1")
            tp2_raw = payload_obj.get("take_profit_2")
            sl_raw = payload_obj.get("stop_loss")

            if not isinstance(entry_raw, (int, float, str)):
                self.state.remove_signal(signal_id)
                if self.stats:
                    self.stats.discard(signal_id)
                continue
            if not isinstance(tp1_raw, (int, float, str)):
                self.state.remove_signal(signal_id)
                if self.stats:
                    self.stats.discard(signal_id)
                continue
            if not isinstance(tp2_raw, (int, float, str)):
                self.state.remove_signal(signal_id)
                if self.stats:
                    self.stats.discard(signal_id)
                continue
            if not isinstance(sl_raw, (int, float, str)):
                self.state.remove_signal(signal_id)
                if self.stats:
                    self.stats.discard(signal_id)
                continue

            try:
                entry = float(entry_raw)
                tp1 = float(tp1_raw)
                tp2 = float(tp2_raw)
                sl = float(sl_raw)
            except (TypeError, ValueError):
                self.state.remove_signal(signal_id)
                if self.stats:
                    self.stats.discard(signal_id)
                continue

            # Check TP/SL hits based on direction
            # Priority order: TP2 > TP1 > SL (maximize profit if multiple levels hit)
            if direction == "BULLISH":
                hit_tp2 = price >= tp2
                hit_tp1 = price >= tp1 and not hit_tp2  # Only TP1 if TP2 not hit
                hit_sl = price <= sl
            else:  # BEARISH
                hit_tp2 = price <= tp2
                hit_tp1 = price <= tp1 and not hit_tp2  # Only TP1 if TP2 not hit
                hit_sl = price >= sl

            result = None
            # Check in priority order: TP2 (best) > TP1 > SL (worst)
            if hit_tp2:
                result = "TP2"
            elif hit_tp1:
                result = "TP1"
            elif hit_sl:
                result = "SL"

            if result:
                summary_message = None
                if self.stats:
                    stats_record = self.stats.record_close(
                        signal_id,
                        exit_price=price,
                        result=result,
                    )
                    if stats_record:
                        summary_message = self.stats.build_summary_message(stats_record)
                    else:
                        self.stats.discard(signal_id)

                if summary_message:
                    self._dispatch(summary_message)
                else:
                    pattern = payload_obj.get("pattern", "STRAT")
                    # Clean symbol format - avoid double /USDT
                    clean_sym = symbol.replace("/USDT", "") + "/USDT"
                    message = (
                        f"🎯 {clean_sym} {pattern} {direction} {result} hit!\n"
                        f"Entry <code>{entry:.6f}</code> | Last <code>{price:.6f}</code>\n"
                        f"TP1 <code>{tp1:.6f}</code> | TP2 <code>{tp2:.6f}</code> | SL <code>{sl:.6f}</code>"
                    )
                    self._dispatch(message)

                self.state.remove_signal(signal_id)

    def _dispatch(self, message: str) -> None:
        """Send message via Telegram."""
        if self.notifier:
            if self.notifier.send_message(message):
                logger.info("Alert sent to Telegram")
            else:
                logger.error("Failed to send Telegram message")
                logger.info("%s", message)
        else:
            logger.info("Alert:\n%s", message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STRAT Bot")
    parser.add_argument("--once", action="store_true", help="Run once and exit (default: loop forever)")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between cycles")
    parser.add_argument("--cooldown", type=int, default=5, help="Default cooldown minutes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bot = STRATBot(interval=args.interval, default_cooldown=args.cooldown)
    # Run in loop mode by default, unless --once is specified
    bot.run(loop=not args.once)


if __name__ == "__main__":
    main()
