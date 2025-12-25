#!/usr/bin/env python3
"""
Opening Range Breakout (ORB) Bot - Gold Standard Edition.

Features:
- Multi-Exchange Support (Binance, Bybit, MEXC) via CCXT
- Robust State Management with File Locking
- Signal Reversal Detection
- Rate Limit Protection
- Enhanced Logging
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
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import FrameType
from typing import Any, Dict, List, Optional, cast

import ccxt  # type: ignore[import-untyped]
import numpy as np

# Graceful shutdown handling
shutdown_requested = False


def signal_handler(signum: int, frame: Optional[FrameType]) -> None:
    """Handle shutdown signals (SIGINT, SIGTERM) gracefully."""
    global shutdown_requested
    shutdown_requested = True
    logger.info("Received %s, shutting down gracefully...", signal.Signals(signum).name)

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
STATE_FILE = BASE_DIR / "orb_state.json"
WATCHLIST_FILE = BASE_DIR / "orb_watchlist.json"
STATS_FILE = LOG_DIR / "orb_stats.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# Add parent directory to path for shared modules
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))

# Required imports (fail fast if missing)
from message_templates import format_signal_message, format_result_message
from notifier import TelegramNotifier
from signal_stats import SignalStats
from tp_sl_calculator import TPSLCalculator, calculate_atr
from trade_config import get_config_manager

# Optional imports (safe fallback)
from safe_import import safe_import
HealthMonitor = safe_import('health_monitor', 'HealthMonitor')
RateLimitHandler = safe_import('rate_limit_handler', 'RateLimitHandler')
RateLimitedExchange = safe_import('rate_limit_handler', 'RateLimitedExchange')

def load_json_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return cast(Dict[str, Any], json.loads(path.read_text()))
    except Exception:
        return {}

def setup_logging(log_level: str = "INFO", enable_detailed: bool = False) -> logging.Logger:
    """Setup enhanced logging."""
    from logging.handlers import RotatingFileHandler
    detailed_format = "%(asctime)s | %(levelname)-8s | [%(name)s:%(funcName)s:%(lineno)d] | %(message)s"
    simple_format = "%(asctime)s | %(levelname)-8s | %(message)s"
    formatter = logging.Formatter(detailed_format if enable_detailed else simple_format)
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root_logger.handlers.clear()
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    file_handler = RotatingFileHandler(
        LOG_DIR / "orb_bot.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    error_handler = RotatingFileHandler(
        LOG_DIR / "orb_errors.log", maxBytes=5 * 1024 * 1024, backupCount=3
    )
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    return logging.getLogger("orb_bot")

logger = logging.getLogger("orb_bot")

# Result notification toggle - set to False to disable separate TP/SL hit alerts
ENABLE_RESULT_NOTIFICATIONS = False

# Price tolerance for TP/SL hit detection (0.5% tolerance for slippage)
PRICE_TOLERANCE = 0.005

EXCHANGE_CONFIG = {
    "binanceusdm": {
        "factory": ccxt.binanceusdm,
        "params": {"enableRateLimit": True},
        "default_market": "swap",
    },
    "mexc": {
        "factory": ccxt.mexc,
        "params": {"enableRateLimit": True, "options": {"defaultType": "swap"}},
        "default_market": "swap",
    },
    "bybit": {
        "factory": ccxt.bybit,
        "params": {"enableRateLimit": True, "options": {"defaultType": "swap"}},
        "default_market": "swap",
    },
}

def resolve_symbol(symbol: str, market_type: str = "swap") -> str:
    cleaned = symbol.upper().replace(" ", "")
    if "/" not in cleaned:
        if cleaned.endswith("USDT"):
            base = cleaned[:-4]
            cleaned = f"{base}/USDT"
        else:
            cleaned = f"{cleaned}/USDT"
    if market_type == "swap" and ":USDT" not in cleaned:
        cleaned = cleaned.replace("/USDT", "/USDT:USDT")
    if market_type == "spot" and ":USDT" in cleaned:
        cleaned = cleaned.replace(":USDT", "")
    return cleaned

@dataclass
class ORBSignal:
    symbol: str
    direction: str
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    timestamp: str
    timeframe: str
    exchange: str
    breakout_type: str
    orb_high: float
    orb_low: float
    range_pct: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ORBLevel:
    high: float
    low: float
    mid: float
    range_pct: float

    @property
    def range_dollars(self) -> float:
        return self.high - self.low

class ORBAnalyzer:
    """Detects Opening Range Breakouts."""

    def __init__(self, config: Dict[str, Any], risk_config: Optional[Any] = None) -> None:
        self.config = config
        self.buffer_pct = config.get("analysis", {}).get("breakout_buffer_pct", 0.001)
        self.window_minutes = config.get("analysis", {}).get("orb_window_minutes", 60)
        self.use_tpsl_calculator = config.get("tp_sl", {}).get("use_tpsl_calculator", True)
        self.risk_config = risk_config

        # Initialize TPSLCalculator if available
        self.tpsl_calc: Optional[Any] = None
        if TPSLCalculator is not None and self.use_tpsl_calculator:
            min_rr = config.get("risk", {}).get("min_risk_reward_ratio", 0.8)
            self.tpsl_calc = TPSLCalculator(min_risk_reward=min_rr, min_risk_reward_tp2=1.5)

    def calculate_orb(self, ohlcv: List[List[float]], session_start_ts: int) -> Optional[ORBLevel]:
        """Calculate High/Low/Mid for the defined opening window."""
        # Filter candles that fall within the ORB window
        # session_start_ts is in milliseconds
        window_end_ts = session_start_ts + (self.window_minutes * 60 * 1000)

        relevant_candles = [c for c in ohlcv if session_start_ts <= c[0] < window_end_ts]

        # Debug logging
        if ohlcv:
            first_ts = ohlcv[0][0] if ohlcv else 0
            last_ts = ohlcv[-1][0] if ohlcv else 0
            logger.info("ORB DEBUG: window=%d-%d, data=%d-%d, found=%d candles",
                       session_start_ts, window_end_ts, first_ts, last_ts, len(relevant_candles))

        if not relevant_candles:
            return None

        high = max(c[2] for c in relevant_candles)
        low = min(c[3] for c in relevant_candles)

        return ORBLevel(
            high=high,
            low=low,
            mid=(high + low) / 2,
            range_pct=((high - low) / low) * 100 if low > 0 else 0
        )

    def detect_breakout(self, current_price: float, orb: ORBLevel) -> Optional[str]:
        """Check if price breaks the ORB level."""
        # Simple breakout logic
        if current_price > orb.high * (1 + self.buffer_pct):
            return "BULLISH"
        if current_price < orb.low * (1 - self.buffer_pct):
            return "BEARISH"
        return None

    def calculate_targets(self, entry: float, direction: str, orb: ORBLevel, ohlcv: Optional[List[List[float]]] = None) -> Dict[str, float]:
        """Calculate TP/SL based on ORB range projection or TPSLCalculator."""
        orb_range = orb.range_dollars

        # Try using TPSLCalculator with ATR-based calculation if available
        if self.tpsl_calc is not None and ohlcv and calculate_atr is not None:
            atr = calculate_atr(ohlcv, period=14)
            if atr:
                # Get multipliers from risk_config or use defaults
                if self.risk_config:
                    sl_mult = self.risk_config.sl_atr_multiplier
                    tp1_mult = self.risk_config.tp1_atr_multiplier
                    tp2_mult = self.risk_config.tp2_atr_multiplier
                else:
                    sl_mult = self.config.get("tp_sl", {}).get("atr_sl_multiplier", 1.5)
                    tp1_mult = self.config.get("tp_sl", {}).get("atr_tp1_multiplier", 2.0)
                    tp2_mult = self.config.get("tp_sl", {}).get("atr_tp2_multiplier", 3.5)

                levels = self.tpsl_calc.calculate(
                    entry=entry,
                    direction=direction,
                    atr=atr,
                    sl_multiplier=sl_mult,
                    tp1_multiplier=tp1_mult,
                    tp2_multiplier=tp2_mult,
                    swing_high=orb.high,
                    swing_low=orb.low,
                )

                if levels.is_valid:
                    logger.debug(f"TPSLCalculator: SL={levels.stop_loss:.6f}, TP1={levels.take_profit_1:.6f}, TP2={levels.take_profit_2:.6f}, RR1={levels.risk_reward_1:.2f}")
                    return {"entry": entry, "sl": levels.stop_loss, "tp1": levels.take_profit_1, "tp2": levels.take_profit_2}
                else:
                    logger.debug(f"TPSLCalculator rejected: {levels.rejection_reason}, falling back to ORB range")

        # Fallback to ORB range-based calculation
        if direction == "BULLISH":
            sl = orb.low  # Stop at bottom of range (conservative) or mid
            tp1 = entry + orb_range  # Measured move
            tp2 = entry + (orb_range * 2)
        else:  # BEARISH
            sl = orb.high
            tp1 = entry - orb_range
            tp2 = entry - (orb_range * 2)

        return {"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2}

class SignalTracker:
    """Robust signal tracker with file locking."""

    def __init__(self, stats: Optional[Any] = None, config: Optional[Dict[str, Any]] = None) -> None:
        self.stats = stats
        self.config = config or {}
        self.state_lock = threading.Lock()
        self.state = self._load_state()

    def _empty_state(self) -> Dict[str, Any]:
        return {"last_alerts": {}, "open_signals": {}, "closed_signals": {}, "last_result_notifications": {}}

    def _load_state(self) -> Dict[str, Any]:
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, 'r') as f:
                    return cast(Dict[str, Any], json.load(f))
            except Exception:
                return self._empty_state()
        return self._empty_state()

    def _save_state(self) -> None:
        with self.state_lock:
            temp_file = STATE_FILE.with_suffix('.tmp')
            try:
                with open(temp_file, 'w') as f:
                    if HAS_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    json.dump(self.state, f, indent=2)
                    if HAS_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                temp_file.replace(STATE_FILE)
            except Exception as e:
                logger.error(f"Failed to save state: {e}")
                # Clean up temp file on error
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except OSError:
                        pass

    def is_duplicate(self, symbol: str, exchange: str, timestamp: int) -> bool:
        # ORB is daily/session based. We assume one signal per session per symbol per direction is enough?
        # Better: Check if we have an open signal for this symbol+exchange
        open_signals = self.state.get("open_signals", {})
        for sig in open_signals.values():
            if sig.get("symbol") == symbol and sig.get("exchange") == exchange:
                # We already have an open trade for this session
                # Unless we want to support multiple breakouts? Usually risky.
                return True
        return False

    def add_signal(self, signal: ORBSignal) -> None:
        signals = self.state.setdefault("open_signals", {})
        sig_id = f"{signal.symbol}-{signal.timestamp}"
        signals[sig_id] = signal.as_dict()
        self._save_state()

        if self.stats:
            self.stats.record_open(
                sig_id, signal.symbol, signal.direction, signal.entry, signal.timestamp,
                extra={"exchange": signal.exchange, "orb_range": f"{signal.orb_low}-{signal.orb_high}"}
            )

    def check_open_signals(self, client_map: Dict[str, Any], notifier: Optional[Any]) -> None:
        signals = self.state.get("open_signals", {})
        if not signals: return

        # Result Notification Cooldown
        last_notifs = self.state.setdefault("last_result_notifications", {})
        cooldown_mins = self.config.get("signal", {}).get("result_notification_cooldown_minutes", 15)

        updated = False
        for sig_id, payload in list(signals.items()):
            symbol = payload.get("symbol")
            display_symbol = symbol if "/" in symbol else f"{symbol}/USDT"
            exchange = payload.get("exchange", "mexc")
            direction = payload.get("direction")

            # Cooldown check - only affects notification, not TP/SL detection
            in_cooldown = False
            last_ts = last_notifs.get(display_symbol)
            if last_ts:
                try:
                    last_dt = datetime.fromisoformat(last_ts)
                    if last_dt.tzinfo is None: last_dt = last_dt.replace(tzinfo=timezone.utc)
                    if (datetime.now(timezone.utc) - last_dt) < timedelta(minutes=cooldown_mins):
                        in_cooldown = True
                except ValueError as exc:
                    logger.debug("Invalid timestamp format for %s cooldown check: %s", symbol, exc)

            try:
                client = client_map.get(exchange)
                if not client: continue
                ticker = client.fetch_ticker(resolve_symbol(symbol))
                price = ticker.get("last")
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                logger.warning(f"Exchange error fetching {symbol}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error fetching {symbol}: {e}")
                continue

            if not price: continue

            tp1, tp2 = payload.get("take_profit_1"), payload.get("take_profit_2")
            sl = payload.get("stop_loss")
            entry = payload.get("entry")

            # Check TP/SL hits with price tolerance for slippage
            res = None
            if direction == "BULLISH":
                # With tolerance for slippage (allow slightly lower prices)
                hit_tp2 = price >= (tp2 * (1 - PRICE_TOLERANCE))
                hit_tp1 = price >= (tp1 * (1 - PRICE_TOLERANCE)) and price < (tp2 * (1 - PRICE_TOLERANCE))
                hit_sl = price <= (sl * (1 + PRICE_TOLERANCE))
            else:
                # For BEARISH/SHORT: TP is below entry, SL is above entry
                # Allow slightly HIGHER prices for TP (price doesn't drop quite as far)
                # Allow slightly LOWER prices for SL (price doesn't rise quite as far)
                hit_tp2 = price <= (tp2 * (1 + PRICE_TOLERANCE))
                hit_tp1 = price <= (tp1 * (1 + PRICE_TOLERANCE)) and price > (tp2 * (1 + PRICE_TOLERANCE))
                hit_sl = price >= (sl * (1 - PRICE_TOLERANCE))

            if hit_tp2: res = "TP2"
            elif hit_tp1: res = "TP1"
            elif hit_sl: res = "SL"

            if res:
                msg = format_result_message(
                    symbol=display_symbol,
                    direction=direction,
                    result=res,
                    entry=entry,
                    exit_price=price,
                    stop_loss=sl,
                    tp1=tp1,
                    tp2=tp2,
                    signal_id=sig_id,
                )

                if notifier and ENABLE_RESULT_NOTIFICATIONS and not in_cooldown:
                    notifier.send_message(msg)
                    last_notifs[display_symbol] = datetime.now(timezone.utc).isoformat()
                elif not ENABLE_RESULT_NOTIFICATIONS:
                    logger.info("Result notification skipped (disabled): %s %s", sig_id, res)
                elif in_cooldown:
                    logger.info("Result notification skipped (cooldown): %s %s", sig_id, res)

                if self.stats: self.stats.record_close(sig_id, price, res)

                # Archive to closed_signals instead of deleting
                closed = self.state.setdefault("closed_signals", {})
                closed_signal = payload.copy()
                closed_signal["closed_at"] = datetime.now(timezone.utc).isoformat()
                closed_signal["exit_price"] = price
                closed_signal["result"] = res
                closed[sig_id] = closed_signal
                del signals[sig_id]
                updated = True

        if updated: self._save_state()

    def check_reversal(self, symbol: str, new_direction: str, notifier: Optional[Any]) -> None:
        signals = self.state.get("open_signals", {})
        for sig_id, payload in signals.items():
            if payload.get("symbol") == symbol:
                old_dir = payload.get("direction")
                # Only check reversal if old_dir exists and differs from new direction
                if old_dir and old_dir != new_direction:
                    msg = f"⚠️ ORB FAKEOUT REVERSAL: {symbol}\nOpen: {old_dir} | New: {new_direction}\nLikely a Fakeout!"
                    if notifier: notifier.send_message(msg)

    def cleanup_stale_signals(self, max_age_hours: int = 24) -> int:
        """Remove signals older than max_age_hours and cleanup closed_signals."""
        signals = self.state.get("open_signals", {})
        closed = self.state.setdefault("closed_signals", {})
        current_time = datetime.now(timezone.utc)
        removed_count = 0
        signal_ids_to_remove = []

        for sig_id, payload in list(signals.items()):
            if not isinstance(payload, dict):
                signal_ids_to_remove.append(sig_id)
                continue

            timestamp_str = payload.get("timestamp")
            if not isinstance(timestamp_str, str):
                signal_ids_to_remove.append(sig_id)
                continue

            try:
                created_time = datetime.fromisoformat(timestamp_str)
                if created_time.tzinfo is None:
                    created_time = created_time.replace(tzinfo=timezone.utc)
                age = current_time - created_time

                if age >= timedelta(hours=max_age_hours):
                    closed[sig_id] = {**payload, "closed_reason": "TIMEOUT", "closed_at": current_time.isoformat()}
                    signal_ids_to_remove.append(sig_id)
                    removed_count += 1
                    logger.info("Stale signal removed: %s (age: %.1f hours)", sig_id, age.total_seconds() / 3600)
            except (ValueError, TypeError):
                signal_ids_to_remove.append(sig_id)

        for sig_id in signal_ids_to_remove:
            if sig_id in signals:
                del signals[sig_id]

        # Also cleanup old closed signals to prevent unbounded growth
        closed_pruned = 0
        max_closed_signals = self.config.get("signal", {}).get("max_closed_signals", 100)
        if isinstance(closed, dict) and len(closed) > max_closed_signals:
            sorted_closed = sorted(
                closed.items(),
                key=lambda x: x[1].get("closed_at", "") if isinstance(x[1], dict) else "",
                reverse=True
            )
            self.state["closed_signals"] = dict(sorted_closed[:max_closed_signals])
            closed_pruned = len(closed) - max_closed_signals
            logger.info("Pruned %d old closed signals (keeping %d)", closed_pruned, max_closed_signals)

        if removed_count > 0 or closed_pruned > 0:
            self._save_state()

        return removed_count

class ORBBot:
    """Refactored ORB Bot."""

    def __init__(self, config_path: Path) -> None:
        self.config = load_json_config(config_path)
        self.watchlist = self._load_watchlist()

        # Load centralized config if available
        self.config_manager: Optional[Any] = None
        self.risk_config: Optional[Any] = None
        if get_config_manager is not None:
            try:
                self.config_manager = get_config_manager()
                self.risk_config = self.config_manager.get_effective_risk("orb_bot", "default")
                logger.info("Loaded centralized config from TradeConfigManager")
            except Exception as e:
                logger.warning(f"Could not load centralized config: {e}")

        self.analyzer = ORBAnalyzer(self.config, risk_config=self.risk_config)
        self.tracker = SignalTracker(SignalStats("ORB Bot", STATS_FILE), self.config)
        self.notifier = self._init_notifier()

        heartbeat = self.config.get("execution", {}).get("health_check_interval_seconds", 3600)
        self.health_monitor = HealthMonitor("ORB Bot", self.notifier, heartbeat_interval=heartbeat) if HealthMonitor and self.notifier else None

        # Initialize rate limit handler from config
        rate_cfg = self.config.get("rate_limit", {})
        self.rate_handler = None
        if RateLimitHandler:
            self.rate_handler = RateLimitHandler(
                base_delay=rate_cfg.get("base_delay_seconds", 0.5),
                max_retries=rate_cfg.get("max_retries", 5),
                backoff_factor=rate_cfg.get("backoff_multiplier", 2.0),
                max_backoff=rate_cfg.get("rate_limit_backoff_max", 30.0)
            )
            logger.info("RateLimitHandler initialized for API protection")

        # Initialize exchange clients with rate limiting
        self.clients = {}
        for name, cfg in EXCHANGE_CONFIG.items():
            try:
                raw_client = cfg["factory"](cfg["params"])
                # Wrap with rate limiting if available
                if RateLimitedExchange and self.rate_handler:
                    self.clients[name] = RateLimitedExchange(raw_client, self.rate_handler)
                    logger.debug(f"Exchange {name} wrapped with RateLimitedExchange")
                else:
                    self.clients[name] = raw_client
            except Exception as e:
                logger.error(f"Failed to init {name}: {e}")

    def _load_watchlist(self) -> List[Dict[str, Any]]:
        if not WATCHLIST_FILE.exists(): return []
        try:
            return cast(List[Dict[str, Any]], json.loads(WATCHLIST_FILE.read_text()))
        except Exception: return []

    def _init_notifier(self) -> Optional[Any]:
        if TelegramNotifier is None: return None
        token = os.getenv("TELEGRAM_BOT_TOKEN_ORB") or os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        return TelegramNotifier(bot_token=token, chat_id=chat_id, signals_log_file=str(LOG_DIR/"orb_signals.json")) if token and chat_id else None

    def get_session_start_ts(self) -> int:
        """Get 00:00 UTC timestamp for today in milliseconds."""
        now = datetime.now(timezone.utc)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return int(start.timestamp() * 1000)

    def run(self, run_once: bool = False) -> None:
        global shutdown_requested
        logger.info("Starting Refactored ORB Bot...")

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        if self.health_monitor: self.health_monitor.send_startup_message()

        try:
            while not shutdown_requested:
                try:
                    session_start = self.get_session_start_ts()

                    for item in self.watchlist:
                        # Check for shutdown during watchlist scan
                        if shutdown_requested:
                            logger.info("Shutdown requested during watchlist scan")
                            break

                        symbol: Optional[str] = item.get("symbol")
                        exchange: str = item.get("exchange", "mexc")
                        timeframe = "5m"  # Default for monitoring

                        if not symbol:
                            continue

                        client = self.clients.get(exchange)
                        if not client:
                            logger.debug(f"No client for {exchange}, skipping {symbol}")
                            continue

                        try:
                            # Fetch OHLCV since session start (plus buffer)
                            # We need enough candles to form the ORB window (e.g. 60 mins)
                            limit = 200  # Should cover 12h+ of 5m candles
                            ohlcv = client.fetch_ohlcv(resolve_symbol(symbol), timeframe, limit=limit)
                            ticker = client.fetch_ticker(resolve_symbol(symbol))
                            price = ticker.get('last')
                        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                            logger.warning(f"Exchange error for {symbol}: {e}")
                            continue
                        except Exception as e:
                            logger.error(f"Unexpected error fetching {symbol}: {e}")
                            continue

                        if not price:
                            continue

                        # Validate OHLCV data for NaN/inf values
                        if ohlcv:
                            ohlcv_arr = np.array(ohlcv)
                            if np.any(np.isnan(ohlcv_arr[:, 1:5])) or np.any(np.isinf(ohlcv_arr[:, 1:5])):
                                logger.warning(f"Invalid OHLCV data (NaN/inf) for {symbol}, skipping")
                                continue
                            # Filter out candles with zero/negative prices or invalid high < low
                            valid_mask = (ohlcv_arr[:, 2] >= ohlcv_arr[:, 3]) & (ohlcv_arr[:, 3] > 0)
                            if not np.all(valid_mask):
                                logger.debug(f"Filtering {np.sum(~valid_mask)} invalid candles for {symbol}")
                                ohlcv = [c for c, v in zip(ohlcv, valid_mask) if v]

                        # 1. Calculate ORB Level
                        orb = self.analyzer.calculate_orb(ohlcv, session_start)
                        logger.info("ORB: %s - ORB range: %s, price: %.6f", symbol,
                                   f"H:{orb.high:.6f} L:{orb.low:.6f} ({orb.range_pct:.2f}%)" if orb else "None", price)

                        # Only proceed if we have a valid ORB range established
                        if orb:
                            # 2. Detect Breakout
                            direction = self.analyzer.detect_breakout(price, orb)
                            logger.info("ORB: %s - Breakout check: %s (price=%.6f, high=%.6f, low=%.6f)",
                                       symbol, direction or "NO BREAKOUT", price, orb.high, orb.low)

                            if direction:
                                # 3. Deduplicate (Check if we already alerted this session)
                                is_dup = self.tracker.is_duplicate(symbol, exchange, session_start)
                                logger.info("ORB: %s - Duplicate check: %s", symbol, "DUPLICATE" if is_dup else "NEW")
                                if not is_dup:

                                    # Update risk config for specific symbol if using centralized config
                                    if self.config_manager:
                                        try:
                                            symbol_risk = self.config_manager.get_effective_risk("orb_bot", symbol)
                                            self.analyzer.risk_config = symbol_risk
                                        except Exception:
                                            pass  # Use existing risk_config

                                    # 4. Targets (pass ohlcv for ATR-based TPSLCalculator)
                                    targets = self.analyzer.calculate_targets(price, direction, orb, ohlcv=ohlcv)

                                    orb_signal = ORBSignal(
                                        symbol=symbol, direction=direction,
                                        entry=price, stop_loss=targets['sl'],
                                        take_profit_1=targets['tp1'], take_profit_2=targets['tp2'],
                                        timestamp=datetime.now(timezone.utc).isoformat(),
                                        timeframe=timeframe, exchange=exchange,
                                        breakout_type="ORB Session Breakout",
                                        orb_high=orb.high, orb_low=orb.low, range_pct=orb.range_pct
                                    )

                                    self.tracker.check_reversal(symbol, direction, self.notifier)
                                    self.tracker.add_signal(orb_signal)
                                    self._send_alert(orb_signal)

                    self.tracker.check_open_signals(self.clients, self.notifier)

                    # Cleanup stale signals every cycle
                    max_age = self.config.get("signal", {}).get("max_signal_age_hours", 24)
                    stale_count = self.tracker.cleanup_stale_signals(max_age_hours=max_age)
                    if stale_count > 0:
                        logger.info("Cleaned up %d stale signals", stale_count)

                    if self.health_monitor:
                        self.health_monitor.record_cycle()

                    if shutdown_requested:
                        break

                    logger.info("Cycle complete; sleeping 60s")
                    if run_once:
                        break
                    # Sleep in 1-second chunks to respond quickly to shutdown signals
                    for _ in range(60):
                        if shutdown_requested:
                            break
                        time.sleep(1)

                except Exception as e:
                    logger.error(f"Cycle error: {e}")
                    if self.health_monitor:
                        self.health_monitor.record_error(str(e))
                    if run_once:
                        raise
                    if shutdown_requested:
                        break
                    time.sleep(10)

        finally:
            if self.health_monitor:
                self.health_monitor.send_shutdown_message()

    def _send_alert(self, signal: ORBSignal) -> None:
        # Build extra info with ORB-specific data
        extra_info = f"ORB Range: {signal.range_pct:.2f}% (${signal.orb_low:.4f} - ${signal.orb_high:.4f})"

        msg = format_signal_message(
            bot_name="ORB",
            symbol=signal.symbol,
            direction=signal.direction,
            entry=signal.entry,
            stop_loss=signal.stop_loss,
            tp1=signal.take_profit_1,
            tp2=signal.take_profit_2,
            exchange=signal.exchange.upper(),
            timeframe=signal.timeframe,
            extra_info=extra_info,
        )
        if self.notifier:
            self.notifier.send_message(msg)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="orb_config.json")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    global logger
    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logging(log_level=log_level)

    try:
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / ".env", override=True)
        load_dotenv(BASE_DIR.parent / ".env", override=True)
    except ImportError:
        logger.debug("python-dotenv not installed, skipping .env loading")

    bot = ORBBot(BASE_DIR / args.config)
    bot.run(run_once=args.once)

if __name__ == "__main__":
    main()
