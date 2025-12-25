#!/usr/bin/env python3
"""Funding Rate & Open Interest Bot - Gold Standard Edition.

Features:
- Multi-Exchange Support (Binance, Bybit, MEXC) via CCXT
- Robust State Management with File Locking
- Signal Reversal Detection
- Enhanced Logging & Error Handling
- Dynamic Configuration
"""

from __future__ import annotations

import argparse
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    # Windows doesn't have fcntl - use a no-op fallback
    HAS_FCNTL = False
import html
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
from typing import Any, Dict, List, Optional, cast

import ccxt  # type: ignore
import numpy as np

# Graceful shutdown handling
shutdown_requested = False


def signal_handler(signum: int, frame: Any) -> None:
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    sig_name = signal.Signals(signum).name
    logger.info(f"Received {sig_name}, initiating graceful shutdown...")
    shutdown_requested = True

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
STATE_FILE = BASE_DIR / "funding_state.json"
WATCHLIST_FILE = BASE_DIR / "funding_watchlist.json"
STATS_FILE = LOG_DIR / "funding_stats.json"

# Price tolerance for TP/SL detection (0.1% = 0.001)
# This accounts for spread, slippage, and minor price variations
PRICE_TOLERANCE = 0.001

LOG_DIR.mkdir(parents=True, exist_ok=True)

# Add parent directory to path for shared modules
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))

# Required imports (fail fast if missing)
from message_templates import format_signal_message, format_result_message
from notifier import TelegramNotifier
from signal_stats import SignalStats
from tp_sl_calculator import TPSLCalculator
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
        LOG_DIR / "funding_bot.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    error_handler = RotatingFileHandler(
        LOG_DIR / "funding_errors.log", maxBytes=5 * 1024 * 1024, backupCount=3
    )
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    return logging.getLogger("funding_bot")

logger = logging.getLogger("funding_bot")

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
class FundingSignal:
    symbol: str
    direction: str
    funding_rate: float
    predicted_price: float
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    timestamp: str
    timeframe: str
    exchange: str
    reasons: List[str]
    funding_timestamp: Optional[int] = None # For deduplication
    open_interest: Optional[float] = None
    oi_change_24h: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

class FundingAnalyzer:
    """Analyzes Funding Rates and Open Interest for signals."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.funding_thresholds = config.get("analysis", {}).get("funding_thresholds", {})
        self.oi_thresholds = config.get("analysis", {}).get("open_interest_thresholds", {})

    def analyze(self, symbol: str, funding_rate: float, oi: Optional[float], oi_change: Optional[float], price: float, ema: Optional[float]) -> Optional[Dict[str, Any]]:
        score = 0
        reasons = []

        # Funding Rate Logic (Contrarian)
        extreme_pos = self.funding_thresholds.get("extreme_positive", 0.0005)
        extreme_neg = self.funding_thresholds.get("extreme_negative", -0.0005)

        direction = "NEUTRAL"

        if funding_rate >= extreme_pos:
            score -= 2
            reasons.append(f"Extreme Positive Funding ({funding_rate*100:.4f}%)")
            if ema and price < ema:
                score -= 1
                reasons.append("Downtrend alignment (Price < EMA)")

        elif funding_rate <= extreme_neg:
            score += 2
            reasons.append(f"Extreme Negative Funding ({funding_rate*100:.4f}%)")
            if ema and price > ema:
                score += 1
                reasons.append("Uptrend alignment (Price > EMA)")

        if oi_change:
            limit = self.oi_thresholds.get("strong_increase_pct", 5.0)
            if oi_change >= limit:
                reasons.append(f"Strong OI Increase (+{oi_change:.1f}%)")
                if abs(score) >= 2:
                    score = int(score * 1.5)

        # WIN RATE FIX: Require higher score threshold for stronger signals
        if score >= 3:
            direction = "BULLISH"
        elif score <= -3:
            direction = "BEARISH"

        # WIN RATE FIX: Require EMA trend alignment (price must align with trend)
        if direction == "BULLISH" and ema and price < ema:
            logger.debug("Skipping BULLISH - price below EMA (counter-trend)")
            return None
        if direction == "BEARISH" and ema and price > ema:
            logger.debug("Skipping BEARISH - price above EMA (counter-trend)")
            return None

        if direction == "NEUTRAL":
            return None

        return {
            "direction": direction,
            "reasons": reasons,
            "score": score
        }

    def calculate_targets(self, entry: float, direction: str, atr: float, symbol: str = "") -> Optional[Dict[str, float]]:
        """Calculate TP/SL using centralized TPSLCalculator with config from trade_config."""
        # Get risk settings from central config
        config_mgr = get_config_manager()
        risk_config = config_mgr.get_effective_risk("funding_bot", symbol)

        dir_normalized = "LONG" if direction == "BULLISH" else "SHORT"
        # WIN RATE FIX: Higher min R:R for better profitability
        calculator = TPSLCalculator(min_risk_reward=max(1.5, risk_config.min_risk_reward))
        levels = calculator.calculate(
            entry=entry,
            direction=dir_normalized,
            atr=atr,
            sl_multiplier=risk_config.sl_atr_multiplier,
            tp1_multiplier=risk_config.tp1_atr_multiplier,
            tp2_multiplier=risk_config.tp2_atr_multiplier,
        )
        if not levels.is_valid:
            return None
        return {"entry": entry, "sl": levels.stop_loss, "tp1": levels.take_profit_1, "tp2": levels.take_profit_2}

class SignalTracker:
    """Robust signal tracker with file locking."""

    def __init__(self, stats: Optional[Any] = None, config: Optional[Dict[str, Any]] = None) -> None:
        self.stats = stats
        self.config: Dict[str, Any] = config or {}
        self.state_lock = threading.Lock()
        self.state = self._load_state()

    def _empty_state(self) -> Dict[str, Any]:
        return {"last_alerts": {}, "open_signals": {}, "closed_signals": {}, "signal_history": {}, "last_result_notifications": {}}

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
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                except Exception:
                    pass  # Ignore cleanup errors

    def should_alert(self, symbol: str, exchange: str, cooldown_mins: int, funding_ts: Optional[int] = None) -> bool:
        """Check if we should alert, using either cooldown or funding timestamp."""
        if funding_ts:
            history = self.state.setdefault("signal_history", {})
            key = f"{symbol}-{exchange}"
            if str(funding_ts) in history.get(key, []):
                return False

        key = f"{symbol}-{exchange}"
        last_ts = self.state.get("last_alerts", {}).get(key)
        if not last_ts: return True
        try:
            last = datetime.fromisoformat(last_ts)
            if last.tzinfo is None: last = last.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - last) >= timedelta(minutes=cooldown_mins)
        except Exception: return True

    def add_signal(self, signal: FundingSignal) -> None:
        # Mark history for deduplication
        if signal.funding_timestamp:
            history = self.state.setdefault("signal_history", {})
            key = f"{signal.symbol}-{signal.exchange}"
            if key not in history: history[key] = []
            ts_str = str(signal.funding_timestamp)
            if ts_str not in history[key]:
                history[key].append(ts_str)
                if len(history[key]) > 10: history[key] = history[key][-10:]

        signals = self.state.setdefault("open_signals", {})
        sig_id = f"{signal.symbol}-{signal.timestamp}"
        signals[sig_id] = signal.as_dict()

        last_alerts = self.state.setdefault("last_alerts", {})
        last_alerts[f"{signal.symbol}-{signal.exchange}"] = signal.timestamp

        self._save_state()

        if self.stats:
            self.stats.record_open(
                sig_id, signal.symbol, signal.direction, signal.entry, signal.timestamp,
                extra={"exchange": signal.exchange}
            )

    def check_open_signals(self, client_map: Dict[str, Any], notifier: Optional[Any]) -> None:
        """Check open signals for TP/SL hits."""
        signals = self.state.get("open_signals", {})
        if not signals:
            return

        # Result Notification Cooldown
        last_notifs = self.state.setdefault("last_result_notifications", {})
        cooldown_mins = self.config.get("signal", {}).get("result_notification_cooldown_minutes", 15)

        updated = False
        for sig_id, payload in list(signals.items()):
            symbol = payload.get("symbol")
            # Validate symbol is not empty
            if not isinstance(symbol, str) or not symbol or not symbol.strip():
                logger.warning(f"Removing signal {sig_id} with invalid symbol: {symbol}")
                signals.pop(sig_id, None)
                updated = True
                continue

            display_symbol = symbol if "/" in symbol else f"{symbol}/USDT"
            exchange = payload.get("exchange", "mexc")
            direction = payload.get("direction")

            # Validate direction
            if direction not in ["BULLISH", "BEARISH"]:
                logger.warning(f"Removing signal {sig_id} with invalid direction: {direction}")
                signals.pop(sig_id, None)
                updated = True
                continue

            # Cooldown Check - only affects notification, not TP/SL detection
            in_cooldown = False
            last_ts = last_notifs.get(display_symbol)
            if last_ts:
                try:
                    last_dt = datetime.fromisoformat(last_ts)
                    if last_dt.tzinfo is None:
                        last_dt = last_dt.replace(tzinfo=timezone.utc)
                    if (datetime.now(timezone.utc) - last_dt) < timedelta(minutes=cooldown_mins):
                        in_cooldown = True
                except ValueError:
                    pass  # Invalid timestamp, proceed with check

            try:
                client = client_map.get(exchange)
                if not client:
                    continue
                ticker = client.fetch_ticker(resolve_symbol(symbol))
                price = ticker.get("last")
            except Exception:
                continue

            if not price or not isinstance(price, (int, float)):
                continue

            tp1, tp2 = payload.get("take_profit_1"), payload.get("take_profit_2")
            sl = payload.get("stop_loss")
            entry = payload.get("entry")

            # Validate TP/SL values
            if not all(isinstance(v, (int, float)) for v in [tp1, tp2, sl, entry] if v is not None):
                logger.warning(f"Signal {sig_id} has invalid TP/SL values, removing")
                signals.pop(sig_id, None)
                updated = True
                continue

            # Check TP/SL hits with price tolerance
            res = None
            if direction == "BULLISH":
                # BULLISH: TPs are ABOVE entry, SL is BELOW entry
                if price >= (tp2 * (1 - PRICE_TOLERANCE)):
                    res = "TP2"
                elif price >= (tp1 * (1 - PRICE_TOLERANCE)):
                    res = "TP1"
                elif price <= (sl * (1 + PRICE_TOLERANCE)):
                    res = "SL"
            else:  # BEARISH
                # BEARISH: TPs are BELOW entry, SL is ABOVE entry
                if price <= (tp2 * (1 + PRICE_TOLERANCE)):
                    res = "TP2"
                elif price <= (tp1 * (1 + PRICE_TOLERANCE)):
                    res = "TP1"
                elif price >= (sl * (1 - PRICE_TOLERANCE)):
                    res = "SL"

            if res:
                pnl = (price - entry) / entry * 100 if direction == "BULLISH" else (entry - price) / entry * 100

                # Check if result notifications are enabled (cooldown only affects notification)
                enable_result_notifs = self.config.get("telegram", {}).get("enable_result_notifications", True)
                if enable_result_notifs and notifier and not in_cooldown:
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
                    notifier.send_message(msg)
                    last_notifs[display_symbol] = datetime.now(timezone.utc).isoformat()
                elif in_cooldown:
                    logger.info("Result notification skipped (cooldown): %s %s", sig_id, res)

                if self.stats:
                    self.stats.record_close(sig_id, price, res)

                # Archive to closed_signals instead of deleting
                closed = self.state.setdefault("closed_signals", {})
                closed_signal = payload.copy()
                closed_signal["closed_at"] = datetime.now(timezone.utc).isoformat()
                closed_signal["exit_price"] = price
                closed_signal["result"] = res
                closed_signal["pnl_percent"] = pnl
                closed[sig_id] = closed_signal
                del signals[sig_id]
                updated = True

        if updated:
            self._save_state()

    def cleanup_stale_signals(self, max_age_hours: int = 24) -> int:
        """Remove signals older than max_age_hours and cleanup closed_signals."""
        signals = self.state.get("open_signals", {})
        now = datetime.now(timezone.utc)
        stale_ids = []

        for sig_id, payload in list(signals.items()):
            if not isinstance(payload, dict):
                stale_ids.append(sig_id)
                continue

            timestamp_str = payload.get("timestamp")
            if not isinstance(timestamp_str, str):
                stale_ids.append(sig_id)
                continue

            try:
                created_dt = datetime.fromisoformat(timestamp_str)
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                age = now - created_dt
                if age >= timedelta(hours=max_age_hours):
                    stale_ids.append(sig_id)
                    logger.info("Stale signal removed: %s (age: %.1f hours)", sig_id, age.total_seconds() / 3600)
            except (ValueError, TypeError):
                stale_ids.append(sig_id)

        for sig_id in stale_ids:
            if sig_id in signals:
                # Archive to closed_signals
                closed = self.state.setdefault("closed_signals", {})
                closed_signal = signals[sig_id].copy()
                closed_signal["closed_at"] = now.isoformat()
                closed_signal["result"] = "EXPIRED"
                closed[sig_id] = closed_signal
                del signals[sig_id]

        # Also cleanup old closed signals to prevent unbounded growth
        closed_pruned = 0
        closed = self.state.get("closed_signals", {})
        max_closed_signals = 100
        if isinstance(closed, dict) and len(closed) > max_closed_signals:
            sorted_closed = sorted(
                closed.items(),
                key=lambda x: x[1].get("closed_at", "") if isinstance(x[1], dict) else "",
                reverse=True
            )
            self.state["closed_signals"] = dict(sorted_closed[:max_closed_signals])
            closed_pruned = len(closed) - max_closed_signals
            logger.info("Pruned %d old closed signals (keeping %d)", closed_pruned, max_closed_signals)

        if stale_ids or closed_pruned > 0:
            self._save_state()

        return len(stale_ids)

    def check_reversal(self, symbol: str, new_direction: str, notifier: Optional[Any]) -> None:
        signals = self.state.get("open_signals", {})
        for sig_id, payload in signals.items():
            if payload.get("symbol") == symbol:
                old_dir = payload.get("direction")
                # Skip if old_dir is None or same as new direction
                if old_dir is None or old_dir == new_direction:
                    continue
                # Safely escape values for HTML
                safe_symbol = html.escape(str(symbol) if symbol else "")
                safe_old = html.escape(str(old_dir) if old_dir else "")
                safe_new = html.escape(str(new_direction) if new_direction else "")
                msg = (
                    f"⚠️ <b>FUNDING REVERSAL</b> ⚠️\n\n"
                    f"<b>Symbol:</b> {safe_symbol}\n"
                    f"<b>Open:</b> {safe_old}\n"
                    f"<b>New:</b> {safe_new}\n\n"
                    f"💡 Check your position!"
                )
                if notifier:
                    notifier.send_message(msg, parse_mode="HTML")

class FundingBot:
    """Refactored Funding Bot."""

    def __init__(self, config_path: Path):
        self.config = load_json_config(config_path)
        self.watchlist = self._load_watchlist()
        self.analyzer = FundingAnalyzer(self.config)
        self.tracker = SignalTracker(SignalStats("Funding Bot", STATS_FILE), self.config)
        self.notifier = self._init_notifier()

        # Init Health Monitor
        heartbeat = self.config.get("execution", {}).get("health_check_interval_seconds", 3600)
        self.health_monitor = HealthMonitor("Funding Bot", self.notifier, heartbeat_interval=heartbeat) if HealthMonitor and self.notifier else None

        # Init Rate Limiter
        rate_cfg = self.config.get("rate_limit", {})
        self.rate_handler = None
        if RateLimitHandler:
            self.rate_handler = RateLimitHandler(
                base_delay=rate_cfg.get("base_delay_seconds", 0.5),
                max_retries=rate_cfg.get("max_retries", 5),
                backoff_factor=rate_cfg.get("backoff_multiplier", 2.0),
                max_backoff=rate_cfg.get("max_backoff_seconds", 30.0)
            )
            logger.info("RateLimitHandler initialized for API protection")

        # Init Exchange Clients with rate limiting
        self.clients: Dict[str, Any] = {}
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
        token = os.getenv("TELEGRAM_BOT_TOKEN_FUNDING") or os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        return TelegramNotifier(bot_token=token, chat_id=chat_id, signals_log_file=str(LOG_DIR/"funding_signals.json")) if token and chat_id else None

    def run(self, run_once: bool = False) -> None:
        global shutdown_requested
        logger.info("Starting Refactored Funding Bot...")

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        if self.health_monitor:
            self.health_monitor.send_startup_message()

        try:
            while not shutdown_requested:
                try:
                    logger.info("Starting cycle - processing %d watchlist items", len(self.watchlist))
                    for item in self.watchlist:
                        # Check shutdown during watchlist scan
                        if shutdown_requested:
                            logger.info("Shutdown requested during watchlist scan")
                            break

                        symbol_raw = item.get("symbol")
                        if not symbol_raw or not isinstance(symbol_raw, str):
                            continue
                        symbol: str = symbol_raw
                        exchange: str = str(item.get("exchange", "mexc"))
                        timeframe: str = str(item.get("timeframe", "8h"))

                        client = self.clients.get(exchange)
                        if not client:
                            continue

                        # 1. Fetch Data via CCXT
                        try:
                            # Funding Rate
                            fr_data = client.fetch_funding_rate(resolve_symbol(symbol))
                            funding_rate = fr_data.get('fundingRate')
                            funding_ts = fr_data.get('fundingTimestamp')  # Unique ID for this funding interval

                            # Ticker (Price)
                            ticker = client.fetch_ticker(resolve_symbol(symbol))
                            price = ticker.get('last')

                            # EMA (needs history)
                            ohlcv = client.fetch_ohlcv(resolve_symbol(symbol), timeframe, limit=50)

                            # Validate OHLCV data
                            if not ohlcv or len(ohlcv) < 20:
                                logger.debug(f"Insufficient OHLCV data for {symbol}: {len(ohlcv) if ohlcv else 0}")
                                continue

                            closes = [c[4] for c in ohlcv]
                            closes_arr = np.array(closes, dtype=np.float64)

                            # Validate closes are not NaN/inf
                            if np.any(np.isnan(closes_arr)) or np.any(np.isinf(closes_arr)):
                                logger.warning(f"{symbol}: OHLCV contains NaN/inf values, skipping")
                                continue

                            if np.any(closes_arr <= 0):
                                logger.warning(f"{symbol}: OHLCV contains zero/negative prices, skipping")
                                continue

                            ema = float(np.mean(closes_arr[-20:])) if len(closes) >= 20 else None

                            # Open Interest (Try fetch, handle if not supported)
                            oi = None
                            oi_change = None
                            try:
                                if hasattr(client, 'fetch_open_interest'):
                                    oi_data = client.fetch_open_interest(resolve_symbol(symbol))
                                    oi = oi_data.get('openInterestAmount')
                            except Exception as exc:
                                logger.debug("Open interest not available for %s: %s", symbol, exc)

                        except Exception as e:
                            logger.error(f"Error fetching {symbol}: {e}")
                            continue

                        # 2. Analyze
                        if funding_rate is None or price is None:
                            continue

                        result = self.analyzer.analyze(symbol, funding_rate, oi, oi_change, price, ema)

                        if result:
                            # Check direction filter
                            allowed_dirs = self.config.get("signal", {}).get("allowed_directions", ["BULLISH", "BEARISH"])
                            if result['direction'] not in allowed_dirs:
                                logger.debug(f"Skipping {result['direction']} signal for {symbol} - not in allowed_directions")
                                continue

                            # 3. Targets
                            atr = price * 0.02  # Simple proxy if calc fails
                            targets = self.analyzer.calculate_targets(price, result['direction'], atr, symbol)
                            if targets is None:
                                continue

                            funding_signal = FundingSignal(
                                symbol=symbol, direction=result['direction'],
                                funding_rate=funding_rate, predicted_price=0,
                                entry=price, stop_loss=targets['sl'],
                                take_profit_1=targets['tp1'], take_profit_2=targets['tp2'],
                                timestamp=datetime.now(timezone.utc).isoformat(),
                                funding_timestamp=funding_ts,
                                timeframe=timeframe, exchange=exchange, reasons=result['reasons']
                            )

                            # 4. Alert & Track
                            cooldown = self.config.get("signal", {}).get("cooldown_minutes", 60)
                            if self.tracker.should_alert(symbol, exchange, cooldown, funding_ts=funding_ts):
                                self.tracker.check_reversal(symbol, funding_signal.direction, self.notifier)
                                self.tracker.add_signal(funding_signal)
                                self._send_alert(funding_signal)

                    self.tracker.check_open_signals(self.clients, self.notifier)

                    # Cleanup stale signals every cycle
                    max_age = self.config.get("signal", {}).get("max_signal_age_hours", 24)
                    stale_count = self.tracker.cleanup_stale_signals(max_age_hours=max_age)
                    if stale_count > 0:
                        logger.info("Cleaned up %d stale signals", stale_count)

                    if self.health_monitor:
                        self.health_monitor.record_cycle()

                    if run_once or shutdown_requested:
                        break

                    logger.info("Cycle complete; sleeping 60s")
                    # Responsive sleep that checks for shutdown
                    for _ in range(60):
                        if shutdown_requested:
                            logger.info("Shutdown requested during sleep")
                            break
                        time.sleep(1)

                except Exception as e:
                    logger.error(f"Cycle error: {e}")
                    if self.health_monitor:
                        self.health_monitor.record_error(str(e))
                    if run_once:
                        raise
                    time.sleep(10)

        finally:
            if self.health_monitor:
                self.health_monitor.send_shutdown_message()

    def _send_alert(self, signal: FundingSignal) -> None:
        msg = format_signal_message(
            bot_name="FUNDING",
            symbol=signal.symbol,
            direction=signal.direction,
            entry=signal.entry,
            stop_loss=signal.stop_loss,
            tp1=signal.take_profit_1,
            funding_rate=signal.funding_rate,
            reasons=signal.reasons,
            timeframe=signal.timeframe,
        )
        if self.notifier: self.notifier.send_message(msg)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="funding_config.json")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    global logger
    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logging(log_level=log_level)

    # Load dotenv here too
    try:
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / ".env", override=True)
        load_dotenv(BASE_DIR.parent / ".env", override=True)
    except ImportError:
        logger.debug("python-dotenv not installed, skipping .env loading")

    bot = FundingBot(BASE_DIR / args.config)
    bot.run(run_once=args.once)

if __name__ == "__main__":
    main()
