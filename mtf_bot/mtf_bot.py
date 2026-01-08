#!/usr/bin/env python3
"""
Multi-Timeframe Bot - Detects confluence across multiple timeframes
Analyzes trend alignment between lower and higher timeframes
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
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
STATE_FILE = BASE_DIR / "mtf_state.json"
WATCHLIST_FILE = BASE_DIR / "mtf_watchlist.json"
STATS_FILE = LOG_DIR / "mtf_stats.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "mtf_bot.log"),
    ],
)

logger = logging.getLogger("mtf_bot")

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
from tp_sl_calculator import TPSLCalculator, calculate_atr
from trade_config import get_config_manager

# NEW: Import unified signal system
from core.bot_signal_mixin import BotSignalMixin, create_price_fetcher

# Optional imports (safe fallback)
from safe_import import safe_import
from file_lock import safe_read_json, safe_write_json
HealthMonitor = safe_import('health_monitor', 'HealthMonitor')
RateLimitHandler = safe_import('rate_limit_handler', 'RateLimitHandler')

# Result notification toggle - set to False to disable separate TP/SL hit alerts
ENABLE_RESULT_NOTIFICATIONS = True

# Bot configuration (previously hardcoded values)
class BotConfigDict(TypedDict):
    max_open_signals: int
    max_signal_age_hours: int
    price_tolerance: float
    exchange: str

BOT_CONFIG: BotConfigDict = {
    "max_open_signals": 50,
    "max_signal_age_hours": 24,
    "price_tolerance": 0.005,  # 0.5% tolerance for slippage
    "exchange": "binanceusdm",
}


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol to always have /USDT suffix (without duplication).

    Handles cases where symbol may already contain /USDT to avoid
    creating malformed symbols like 'NIGHT/USDT/USDT'.
    """
    base = symbol.replace("/USDT", "").replace("_USDT", "")
    return f"{base}/USDT"


class WatchItem(TypedDict, total=False):
    symbol: str
    timeframe: str
    cooldown_minutes: int
    exchange: str
    market_type: str


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
    for item in data:
        if not isinstance(item, dict):
            continue
        symbol_val = item.get("symbol")
        if not isinstance(symbol_val, str):
            continue
        # Use timeframe field (standardized across all bots)
        timeframe_val = item.get("timeframe", "5m")
        timeframe = timeframe_val if isinstance(timeframe_val, str) else "5m"
        cooldown_val = item.get("cooldown_minutes", 30)
        try:
            cooldown = int(cooldown_val)
        except Exception:
            cooldown = 30
        normalized.append({
            "symbol": symbol_val.upper(),
            "timeframe": timeframe,
            "cooldown_minutes": cooldown,
        })
    return normalized


def human_ts() -> str:
    """Return human-readable UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class MultiTimeframeAnalyzer:
    """Analyzes trends across multiple timeframes."""

    # Timeframe hierarchy for confluence analysis
    TIMEFRAME_MAP = {
        "5m": ["15m", "1h"],
        "15m": ["1h", "4h"],
        "30m": ["1h", "4h"],
        "1h": ["4h", "1d"],
        "4h": ["1d", "1w"],
    }

    def __init__(self) -> None:
        pass

    def get_higher_timeframes(self, base_tf: str) -> List[str]:
        """Get higher timeframes for analysis."""
        return self.TIMEFRAME_MAP.get(base_tf, ["1h", "4h"])

    def calculate_ema(self, closes: npt.NDArray[np.floating[Any]], period: int) -> npt.NDArray[np.floating[Any]]:
        """Calculate EMA."""
        ema = np.zeros_like(closes)
        multiplier = 2 / (period + 1)
        ema[0] = closes[0]
        for i in range(1, len(closes)):
            ema[i] = (closes[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema

    @staticmethod
    def calculate_sma(closes: npt.NDArray[np.floating[Any]], period: int) -> npt.NDArray[np.floating[Any]]:
        sma = np.zeros_like(closes)
        for i in range(len(closes)):
            start = max(0, i - period + 1)
            sma[i] = float(np.mean(closes[start:i+1]))
        return sma

    @staticmethod
    def calculate_rsi(closes: npt.NDArray[np.floating[Any]], period: int = 14) -> npt.NDArray[np.floating[Any]]:
        # Use tolerance for division by zero checks
        MIN_DIVISOR = 1e-10
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.zeros_like(closes)
        avg_loss = np.zeros_like(closes)
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])
        for i in range(period + 1, len(closes)):
            avg_gain[i] = ((avg_gain[i-1] * (period - 1)) + gains[i-1]) / period
            avg_loss[i] = ((avg_loss[i-1] * (period - 1)) + losses[i-1]) / period
        # Use tolerance instead of exact equality for division check
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss > MIN_DIVISOR)
        rsi = 100 - (100 / (1 + rs))
        rsi[:period] = 50
        # Handle NaN values
        rsi = np.where(np.isnan(rsi), 50, rsi)
        return rsi

    def detect_trend(self, ohlcv: List[Any]) -> Tuple[str, float]:
        """
        Detect trend direction and strength.
        Returns: (direction, confidence_score)
        """
        if len(ohlcv) < 50:
            return ("NEUTRAL", 0.0)

        closes = np.array([x[4] for x in ohlcv], dtype=np.float64)
        rsi = self.calculate_rsi(closes, period=14)
        sma50 = self.calculate_sma(closes, 50)
        current_close = closes[-1]
        current_rsi = rsi[-1]
        current_sma = sma50[-1]

        # Protect against division by zero or near-zero SMA
        MIN_DIVISOR = 1e-10
        if abs(current_sma) < MIN_DIVISOR:
            return ("NEUTRAL", 0.0)

        if current_close > current_sma and current_rsi > 50:
            direction = "BULLISH"
            confidence = min((current_close - current_sma) / current_sma * 2, 1.0)
        elif current_close < current_sma and current_rsi < 50:
            direction = "BEARISH"
            confidence = min((current_sma - current_close) / current_sma * 2, 1.0)
        else:
            direction = "NEUTRAL"
            confidence = 0.0

        return (direction, confidence)

    def analyze_confluence(
        self,
        base_ohlcv: List[Any],
        higher_ohlcvs: Dict[str, List[Any]]
    ) -> Optional[Dict[str, object]]:
        """
        Analyze confluence across timeframes.
        Returns signal if strong confluence detected.
        """
        # Detect trend on base timeframe
        base_trend, base_conf = self.detect_trend(base_ohlcv)

        if base_trend == "NEUTRAL" or base_conf < 0.02:  # loosened from 0.05 to allow more signals
            logger.info("MTF: Base trend=%s, conf=%.2f - SKIPPED (neutral or low conf)", base_trend, base_conf)
            return None

        higher_trends = {}
        score = 0
        ordered_tfs = list(higher_ohlcvs.keys())
        for idx, tf in enumerate(ordered_tfs):
            ohlcv = higher_ohlcvs[tf]
            trend, conf = self.detect_trend(ohlcv)
            higher_trends[tf] = (trend, conf)
            if trend == base_trend:
                weight = 2 if idx == 0 else 1
                score += weight

        if score < 2:
            logger.info("MTF: Score=%d (need 2+), trends=%s - SKIPPED (low confluence)", score, higher_trends)
            return None
        total_possible = 2 + max(0, len(higher_trends) - 1)
        confluence_pct = (score / total_possible) * 100
        if score >= 2:
            strength = "STRONG"
        elif score == 2:
            strength = "MODERATE"
        else:
            strength = "WEAK"

        return {
            "direction": base_trend,
            "strength": strength,
            "confluence_pct": confluence_pct,
            "base_conf": base_conf,
            "higher_trends": higher_trends,
        }

    def calculate_targets(
        self,
        direction: str,
        current_price: float,
        atr: float,
        symbol: str = "",
        strength: str = "MODERATE",
    ) -> Optional[Dict[str, float]]:
        """Calculate entry, stop loss, and targets."""
        entry = current_price

        # Adjust SL multiplier based on signal strength
        sl_adjust = {"STRONG": 3.0, "MODERATE": 2.0, "WEAK": 1.5}
        strength_mult = sl_adjust.get(strength, 2.0)

        # Get risk config and calculate TP/SL using centralized calculator
        config_mgr = get_config_manager()
        risk_config = config_mgr.get_effective_risk("mtf_bot", symbol)
        base_sl_mult = risk_config.sl_atr_multiplier
        sl_mult = base_sl_mult * (strength_mult / 2.0)
        tp1_mult = risk_config.tp1_atr_multiplier
        tp2_mult = risk_config.tp2_atr_multiplier

        calculator = TPSLCalculator(min_risk_reward=max(1.2, risk_config.min_risk_reward * 0.8), min_risk_reward_tp2=1.3)  # loosened
        levels = calculator.calculate(
            entry=entry,
            direction=direction,
            atr=atr,
            sl_multiplier=sl_mult,
            tp1_multiplier=tp1_mult,
            tp2_multiplier=tp2_mult,
        )

        if not levels.is_valid:
            logger.info("%s: TPSLCalculator REJECTED - %s", symbol, levels.rejection_reason)
            return None

        logger.debug("%s %s MTF signal | Entry: %.6f | SL: %.6f | TP1: %.6f | TP2: %.6f",
                   symbol, direction, entry, levels.stop_loss, levels.take_profit_1, levels.take_profit_2)

        return {
            "entry": entry,
            "stop_loss": levels.stop_loss,
            "take_profit_1": levels.take_profit_1,
            "take_profit_2": levels.take_profit_2,
        }

    # Note: ATR calculation moved to shared tp_sl_calculator.calculate_atr()


class BinanceClient:
    """Binance USDM Futures exchange client wrapper."""

    def __init__(self) -> None:
        self.exchange: Any = ccxt.binanceusdm({
            "enableRateLimit": True,
            "options": {"defaultType": "swap"}
        })
        self.exchange.load_markets()
        self.rate_limiter: Any = RateLimitHandler(base_delay=0.5, max_retries=5) if RateLimitHandler else None

    @staticmethod
    def _swap_symbol(symbol: str) -> str:
        """Convert symbol to Binance USDM format (e.g., FHE/USDT -> FHE/USDT:USDT)."""
        sym = symbol.upper().replace("/USDT", "")
        return f"{sym}/USDT:USDT"

    def fetch_ohlcv(self, symbol: str, timeframe: str = "5m", limit: int = 100) -> List[Any]:
        """Fetch OHLCV data with rate limiting and network error retry."""
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
        """Fetch ticker data with rate limiting and network error retry."""
        if self.rate_limiter:
            return cast(Dict[str, Any], self.rate_limiter.execute(self.exchange.fetch_ticker, self._swap_symbol(symbol)))
        return cast(Dict[str, Any], self.exchange.fetch_ticker(self._swap_symbol(symbol)))


# Graceful shutdown handling
shutdown_requested = False


def signal_handler(signum: int, frame: Optional[FrameType]) -> None:  # pragma: no cover - signal path
    """Handle shutdown signals (SIGINT, SIGTERM) gracefully."""
    global shutdown_requested
    shutdown_requested = True
    logger.info("Received %s, shutting down gracefully...", signal.Signals(signum).name)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@dataclass
class MTFSignal:
    """Represents a multi-timeframe signal."""
    symbol: str
    direction: str
    strength: str
    timestamp: str
    base_timeframe: str
    confluence_pct: float
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    current_price: Optional[float] = None


OpenSignals = Dict[str, Dict[str, Any]]


class BotState:
    """Manages bot state persistence with file locking for thread safety."""

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
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt

    def _empty_state(self) -> Dict[str, Any]:
        return {"last_alert": {}, "open_signals": {}, "closed_signals": {}}

    def _load(self) -> Dict[str, Any]:
        """Load state from file with file locking."""
        default = self._empty_state()
        data = safe_read_json(self.path, default)
        if not isinstance(data, dict):
            return default
        last_alert = data.get("last_alert")
        open_signals = data.get("open_signals")
        closed_signals = data.get("closed_signals")
        return {
            "last_alert": last_alert if isinstance(last_alert, dict) else {},
            "open_signals": open_signals if isinstance(open_signals, dict) else {},
            "closed_signals": closed_signals if isinstance(closed_signals, dict) else {},
        }

    def save(self) -> None:
        """Save state to file with file locking."""
        safe_write_json(self.path, self.data)

    def can_alert(self, symbol: str, cooldown_minutes: int) -> bool:
        """Check if enough time has passed since last alert for this symbol."""
        last_map = self.data.setdefault("last_alert", {})
        if not isinstance(last_map, dict):
            self.data["last_alert"] = {}
            return True
        last_ts = last_map.get(symbol)
        if not isinstance(last_ts, str):
            return True

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
                    closed[signal_id] = {**payload, "closed_reason": "TIMEOUT", "closed_at": current_time.isoformat()}
                    signal_ids_to_remove.append(signal_id)
                    removed_count += 1
                    logger.info("Stale signal removed: %s (age: %.1f hours)", signal_id, age.total_seconds() / 3600)
            except (ValueError, TypeError):
                signal_ids_to_remove.append(signal_id)

        for signal_id in signal_ids_to_remove:
            if signal_id in signals:
                signals.pop(signal_id)

        # Also cleanup old closed signals to prevent unbounded growth
        closed_pruned = 0
        max_closed_signals = 100  # Keep only the most recent 100 closed signals
        if isinstance(closed, dict) and len(closed) > max_closed_signals:
            # Sort by closed_at and keep only the most recent
            sorted_closed = sorted(
                closed.items(),
                key=lambda x: x[1].get("closed_at", "") if isinstance(x[1], dict) else "",
                reverse=True
            )
            # Keep only the most recent max_closed_signals
            self.data["closed_signals"] = dict(sorted_closed[:max_closed_signals])
            closed_pruned = len(closed) - max_closed_signals
            logger.info("Pruned %d old closed signals (keeping %d)", closed_pruned, max_closed_signals)

        if removed_count > 0 or closed_pruned > 0:
            self.save()

        return removed_count

    def add_signal(self, signal_id: str, payload: Dict[str, Any]) -> None:
        signals = self.iter_signals()
        signals[signal_id] = payload
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


class MTFBot(BotSignalMixin):
    """Main Multi-Timeframe Bot with unified signal management."""

    def __init__(self, interval: int = 60, default_cooldown: int = 5):
        self.interval = interval
        self.default_cooldown = default_cooldown
        self.watchlist: List[WatchItem] = load_watchlist()
        self.client = BinanceClient()
        self.analyzer = MultiTimeframeAnalyzer()
        self.state = BotState(STATE_FILE)
        self.notifier = self._init_notifier()
        self.stats = SignalStats("MTF Bot", STATS_FILE)
        self.health_monitor = (
            HealthMonitor("MTF Bot", self.notifier, heartbeat_interval=3600)
            if HealthMonitor else None
        )
        self.rate_limiter = (
            RateLimitHandler(base_delay=0.5, max_retries=5)
            if RateLimitHandler else None
        )
        
        # NEW: Initialize unified signal adapter
        self._init_signal_adapter(
            bot_name="mtf_bot",
            notifier=self.notifier,
            exchange="Binance",
            default_timeframe="1m",
            notification_mode="signal_only",
        )

    def _init_notifier(self) -> Optional[Any]:
        if TelegramNotifier is None:
            logger.warning("Telegram notifier unavailable")
            return None
        token = os.getenv("TELEGRAM_BOT_TOKEN_MTF") or os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            logger.warning("Telegram credentials missing")
            return None
        return TelegramNotifier(
            bot_token=token,
            chat_id=chat_id,
            signals_log_file=str(LOG_DIR / "mtf_signals.json")
        )

    def run(self, loop: bool = False) -> None:
        if not self.watchlist:
            logger.error("Empty watchlist; exiting")
            return

        logger.info("Starting Multi-Timeframe Bot for %d symbols", len(self.watchlist))

        if self.health_monitor:
            self.health_monitor.send_startup_message()

        try:
            while not shutdown_requested:
                try:
                    self._run_cycle()

                    # Cleanup stale signals every cycle (using config)
                    stale_count = self.state.cleanup_stale_signals(max_age_hours=BOT_CONFIG["max_signal_age_hours"])
                    if stale_count > 0:
                        logger.info("Cleaned up %d stale signals", stale_count)

                    self._monitor_open_signals()

                    if self.health_monitor:
                        self.health_monitor.record_cycle()

                    if not loop:
                        break

                    if shutdown_requested:
                        break
                    logger.info("Cycle complete; sleeping %ds", self.interval)
                    # Sleep in 1-second chunks to respond quickly to shutdown signals
                    for _ in range(self.interval):
                        if shutdown_requested:
                            logger.info("Shutdown requested during sleep")
                            break
                        time.sleep(1)
                except Exception as exc:
                    logger.error("Error in cycle: %s", exc)
                    if self.health_monitor:
                        self.health_monitor.record_error(str(exc))
                    if not loop:
                        raise
                    if shutdown_requested:
                        break
                    time.sleep(10)
        finally:
            if self.health_monitor:
                self.health_monitor.send_shutdown_message()

    def _run_cycle(self) -> None:
        for item in self.watchlist:
            # Check for shutdown during watchlist scan
            if shutdown_requested:
                logger.info("Shutdown requested during watchlist scan")
                break

            symbol_val = item.get("symbol") if isinstance(item, dict) else None
            if not isinstance(symbol_val, str):
                continue
            symbol = symbol_val
            timeframe_val = item.get("timeframe") if isinstance(item, dict) else None
            timeframe = timeframe_val if isinstance(timeframe_val, str) else "5m"
            cooldown_raw = item.get("cooldown_minutes") if isinstance(item, dict) else None
            try:
                cooldown = int(cooldown_raw) if cooldown_raw is not None else self.default_cooldown
            except Exception:
                cooldown = self.default_cooldown

            try:
                signal = self._analyze_symbol(symbol, timeframe)
            except (ccxt.NetworkError, ccxt.ExchangeError) as exc:
                logger.warning("Exchange error for %s: %s", symbol, exc)
                if self.health_monitor:
                    self.health_monitor.record_error(f"Exchange error for {symbol}: {exc}")
                continue
            except Exception as exc:
                logger.error("Failed to analyze %s: %s", symbol, exc)
                if self.health_monitor:
                    self.health_monitor.record_error(f"Analysis error for {symbol}: {exc}")
                continue

            if signal is None:
                logger.debug("%s: No confluence detected", symbol)
                continue

            # Alert on STRONG or MODERATE signals (relaxed from STRONG-only)
            if signal.strength not in ("STRONG", "MODERATE"):
                logger.debug("%s: Confluence not strong enough (%s)", symbol, signal.strength)
                continue

            if not self.state.can_alert(symbol, cooldown):
                logger.debug("Cooldown active for %s", symbol)
                continue

            # Max open signals limit (from config)
            max_signals = BOT_CONFIG["max_open_signals"]
            current_open = len(self.state.iter_signals())
            if current_open >= max_signals:
                logger.info(
                    "Max open signals limit reached (%d/%d). Skipping %s",
                    current_open, max_signals, symbol
                )
                continue

            # Send alert
            message = self._format_message(signal)
            self._dispatch(message)
            self.state.mark_alert(symbol)

            # Track signal
            signal_id = f"{symbol}-MTF-{signal.timestamp}"
            exchange_name = BOT_CONFIG["exchange"]
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
                "exchange": exchange_name,
                "strength": signal.strength,
            }
            self.state.add_signal(signal_id, trade_data)

            if self.stats:
                self.stats.record_open(
                    signal_id=signal_id,
                    symbol=normalize_symbol(symbol),
                    direction=signal.direction,
                    entry=signal.entry,
                    created_at=signal.timestamp,
                    extra={
                        "timeframe": timeframe,
                        "exchange": exchange_name,
                        "strategy": "Multi-Timeframe",
                    },
                )

            time.sleep(0.5)

    def _analyze_symbol(self, symbol: str, timeframe: str) -> Optional[MTFSignal]:
        """Analyze symbol across multiple timeframes."""
        # Fetch base timeframe data
        base_ohlcv = self.client.fetch_ohlcv(symbol, timeframe, limit=100)

        if len(base_ohlcv) < 50:
            return None

        # Validate OHLCV data for NaN/inf values
        try:
            closes_check = np.array([x[4] for x in base_ohlcv], dtype=np.float64)
            if np.any(np.isnan(closes_check)) or np.any(np.isinf(closes_check)):
                logger.warning("%s: OHLCV contains NaN/inf values, skipping", symbol)
                return None
            if np.any(closes_check <= 0):
                logger.warning("%s: OHLCV contains zero or negative prices, skipping", symbol)
                return None
        except (IndexError, TypeError, ValueError) as e:
            logger.warning("%s: Failed to validate OHLCV: %s", symbol, e)
            return None

        # Fetch higher timeframes
        higher_tfs = self.analyzer.get_higher_timeframes(timeframe)
        higher_ohlcvs: Dict[str, List[Any]] = {}

        for htf in higher_tfs:
            try:
                higher_ohlcvs[htf] = self.client.fetch_ohlcv(symbol, htf, limit=100)
            except Exception as exc:
                logger.warning("Failed to fetch %s %s: %s", symbol, htf, exc)

        if not higher_ohlcvs:
            return None

        # Analyze confluence
        logger.info("MTF: Analyzing %s on %s timeframe...", symbol, timeframe)
        result = self.analyzer.analyze_confluence(base_ohlcv, higher_ohlcvs)

        if result is None or not isinstance(result, dict):
            logger.info("MTF: %s - No confluence result", symbol)
            return None
        direction_val = result.get("direction")
        strength_val = result.get("strength")
        confluence_val = result.get("confluence_pct")
        if not isinstance(direction_val, str) or not isinstance(strength_val, str):
            return None
        if isinstance(confluence_val, (int, float)):
            confluence_pct = float(confluence_val)
        else:
            return None

        # Calculate ATR
        # Use shared ATR calculation from tp_sl_calculator
        atr_value = calculate_atr(base_ohlcv)
        atr = float(atr_value) if atr_value is not None else 0.0

        # Get current price
        ticker = self.client.fetch_ticker(symbol)
        current_price_raw = ticker.get("last") if isinstance(ticker, dict) else None
        if current_price_raw is None and isinstance(ticker, dict):
            current_price_raw = ticker.get("close")
        if not isinstance(current_price_raw, (int, float, str)):
            return None
        try:
            current_price = float(current_price_raw)
        except (TypeError, ValueError):
            return None

        # Calculate targets
        targets = self.analyzer.calculate_targets(
            direction_val,
            current_price,
            atr,
            symbol=symbol,
            strength=strength_val,
        )

        if targets is None:
            logger.info("MTF: %s - Targets calculation failed", symbol)
            return None

        logger.info("MTF: %s SIGNAL GENERATED - %s %.1f%% confluence", symbol, direction_val, confluence_pct)
        return MTFSignal(
            symbol=symbol,
            direction=direction_val,
            strength=strength_val,
            timestamp=human_ts(),
            base_timeframe=timeframe,
            confluence_pct=confluence_pct,
            entry=targets["entry"],
            stop_loss=targets["stop_loss"],
            take_profit_1=targets["take_profit_1"],
            take_profit_2=targets["take_profit_2"],
            current_price=current_price,
        )

    def _format_message(self, signal: MTFSignal) -> str:
        """Format Telegram message for signal using centralized template."""
        # Map strength to a numeric value for the template
        strength_map = {"STRONG": 0.9, "MODERATE": 0.7, "WEAK": 0.5}
        strength_val = strength_map.get(signal.strength, 0.6)

        # Get performance stats (ALWAYS included)
        tp1_count = 0
        tp2_count = 0
        sl_count = 0
        if self.stats is not None:
            symbol_key = normalize_symbol(signal.symbol)
            counts = self.stats.symbol_tp_sl_counts(symbol_key)
            tp1_count = counts.get("TP1", 0)
            tp2_count = counts.get("TP2", 0)
            sl_count = counts.get("SL", 0)
        total = tp1_count + tp2_count + sl_count
        perf_stats = {
            "tp1": tp1_count,
            "tp2": tp2_count,
            "sl": sl_count,
            "wins": tp1_count + tp2_count,
            "total": total,
        }

        # Build extra info with MTF-specific data
        extra_info = f"MTF Confluence: {signal.confluence_pct:.0f}% | {signal.strength} | Base TF: {signal.base_timeframe}"

        return format_signal_message(
            bot_name="MTF",
            symbol=normalize_symbol(signal.symbol),
            direction=signal.direction,
            entry=signal.entry,
            stop_loss=signal.stop_loss,
            tp1=signal.take_profit_1,
            tp2=signal.take_profit_2,
            strength=strength_val,
            exchange=BOT_CONFIG["exchange"],
            timeframe=signal.base_timeframe,
            current_price=signal.current_price,
            performance_stats=perf_stats,
            extra_info=extra_info,
        )

    def _monitor_open_signals(self) -> None:
        """Monitor open signals for TP/SL hits."""
        signals = self.state.iter_signals()
        if not signals:
            return

        for signal_id, payload in list(signals.items()):
            if not isinstance(payload, dict):
                self.state.remove_signal(signal_id)
                continue

            symbol_val = payload.get("symbol")
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
                continue

            if not isinstance(price_raw, (int, float, str)):
                continue
            try:
                price = float(price_raw)
            except (TypeError, ValueError):
                continue

            direction_val = payload.get("direction")
            if not isinstance(direction_val, str):
                self.state.remove_signal(signal_id)
                if self.stats:
                    self.stats.discard(signal_id)
                continue
            direction = direction_val

            entry_raw = payload.get("entry")
            tp1_raw = payload.get("take_profit_1")
            tp2_raw = payload.get("take_profit_2")
            sl_raw = payload.get("stop_loss")
            if not all(isinstance(v, (int, float, str)) for v in (entry_raw, tp1_raw, tp2_raw, sl_raw)):
                self.state.remove_signal(signal_id)
                if self.stats:
                    self.stats.discard(signal_id)
                continue
            try:
                entry = float(entry_raw)  # type: ignore[arg-type]
                tp1 = float(tp1_raw)  # type: ignore[arg-type]
                tp2 = float(tp2_raw)  # type: ignore[arg-type]
                sl = float(sl_raw)    # type: ignore[arg-type]
            except (TypeError, ValueError):
                self.state.remove_signal(signal_id)
                if self.stats:
                    self.stats.discard(signal_id)
                continue

            # Check for TP/SL hits with price tolerance (from config)
            price_tol = BOT_CONFIG["price_tolerance"]

            if direction == "BULLISH":
                # With tolerance for slippage (allow slightly lower prices)
                hit_tp2 = price >= (tp2 * (1 - price_tol))
                hit_tp1 = price >= (tp1 * (1 - price_tol)) and price < (tp2 * (1 - price_tol))
                hit_sl = price <= (sl * (1 + price_tol))
            else:
                # For BEARISH/SHORT: TP is below entry, SL is above entry
                # Allow slightly HIGHER prices for TP (price doesn't drop quite as far)
                # Allow slightly LOWER prices for SL (price doesn't rise quite as far)
                hit_tp2 = price <= (tp2 * (1 + price_tol))
                hit_tp1 = price <= (tp1 * (1 + price_tol)) and price > (tp2 * (1 + price_tol))
                hit_sl = price >= (sl * (1 - price_tol))

            result = None
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
                        logger.info("Trade closed: %s | %s | Entry: %.6f | Exit: %.6f | Result: %s | P&L: %.2f%%",
                                   signal_id, symbol, entry, price, result, stats_record.pnl_pct)
                    else:
                        self.stats.discard(signal_id)

                if ENABLE_RESULT_NOTIFICATIONS:
                    if summary_message:
                        self._dispatch(summary_message)
                    else:
                        message = format_result_message(
                            symbol=symbol,
                            direction=direction,
                            result=result,
                            entry=entry,
                            exit_price=price,
                            stop_loss=sl,
                            tp1=tp1,
                            tp2=tp2,
                            signal_id=signal_id,
                        )
                        self._dispatch(message)
                else:
                    logger.info("Result notification skipped (disabled): %s %s", signal_id, result)

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
    parser = argparse.ArgumentParser(description="Multi-Timeframe Bot")
    parser.add_argument("--once", action="store_true", help="Run only one cycle")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between cycles")
    parser.add_argument("--cooldown", type=int, default=5, help="Default cooldown minutes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bot = MTFBot(interval=args.interval, default_cooldown=args.cooldown)
    bot.run(loop=not args.once)


if __name__ == "__main__":
    main()
