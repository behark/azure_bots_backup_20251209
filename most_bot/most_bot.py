#!/usr/bin/env python3
"""
MOST Bot - Moving Stop Loss (MOST) Indicator Strategy
EMA with percentage-based trailing stop by Anıl ÖZEKŞİ
Signals: BUY when EMA crosses above MOST, SELL when EMA crosses below MOST
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
STATE_FILE = BASE_DIR / "most_state.json"
WATCHLIST_FILE = BASE_DIR / "most_watchlist.json"
STATS_FILE = LOG_DIR / "most_stats.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "most_bot.log"),
    ],
)

logger = logging.getLogger("most_bot")

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
    load_dotenv(BASE_DIR.parent / ".env")
except ImportError:
    pass

sys.path.append(str(BASE_DIR.parent))

# Required imports (fail fast if missing)
from message_templates import format_signal_message, format_result_message
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
    for item in data:
        if not isinstance(item, dict):
            continue
        symbol_val = item.get("symbol")
        if not isinstance(symbol_val, str):
            continue
        period_val = item.get("period", "5m")
        period = period_val if isinstance(period_val, str) else "5m"
        cooldown_val = item.get("cooldown_minutes", 30)
        try:
            cooldown = int(cooldown_val)
        except Exception:
            cooldown = 30
        normalized.append({
            "symbol": symbol_val.upper(),
            "period": period,
            "cooldown_minutes": cooldown,
        })
    return normalized


def human_ts() -> str:
    """Return human-readable UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class MOSTAnalyzer:
    """MOST (Moving Stop Loss) indicator analyzer."""
    
    def __init__(self, ema_length: int = 9, atr_period: int = 14, atr_multiplier: float = 2.0):
        self.ema_length = ema_length
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
    
    @staticmethod
    def calculate_ema(prices: npt.NDArray[np.floating[Any]], period: int) -> npt.NDArray[np.floating[Any]]:
        """Calculate EMA."""
        ema = np.zeros(len(prices))
        if len(prices) < period:
            return ema
        multiplier = 2 / (period + 1)
        ema[period-1] = np.mean(prices[:period])
        for i in range(period, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema
    
    def calculate_most(self, highs: npt.NDArray[np.floating[Any]], lows: npt.NDArray[np.floating[Any]], closes: npt.NDArray[np.floating[Any]]) -> Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
        """
        Calculate MOST indicator.
        Returns: (most_line, trend) where trend is 1 (bullish) or -1 (bearish)
        """
        n = len(closes)
        ema = self.calculate_ema(closes, self.ema_length)
        atr = self.calculate_atr(highs, lows, closes, period=self.atr_period)
        atr = atr if atr > 0 else float(np.std(closes[-self.atr_period:]))
        
        most = np.zeros(n)
        trend = np.zeros(n)
        
        # Initialize
        start = self.ema_length
        stop_offset = atr * self.atr_multiplier
        most[start] = ema[start] - stop_offset
        trend[start] = 1
        
        for i in range(start + 1, n):
            stop_offset = atr * self.atr_multiplier
            stop_long = ema[i] - stop_offset
            stop_short = ema[i] + stop_offset
            
            if trend[i-1] == 1:  # Was bullish
                # Raise stop if EMA goes higher
                most[i] = max(most[i-1], stop_long)
                
                # Check for reversal (EMA crosses below MOST)
                if ema[i] < most[i]:
                    trend[i] = -1
                    most[i] = stop_short
                else:
                    trend[i] = 1
            else:  # Was bearish
                # Lower stop if EMA goes lower
                most[i] = min(most[i-1], stop_short)
                
                # Check for reversal (EMA crosses above MOST)
                if ema[i] > most[i]:
                    trend[i] = 1
                    most[i] = stop_long
                else:
                    trend[i] = -1
        
        return most, trend, ema
    
    def detect_signal(self, highs: npt.NDArray[np.floating[Any]], lows: npt.NDArray[np.floating[Any]], closes: npt.NDArray[np.floating[Any]]) -> Optional[Tuple[str, float, str]]:
        """
        Detect MOST crossover signals.
        Returns (direction, most_value) or None.
        """
        most, trend, ema = self.calculate_most(highs, lows, closes)
        
        # Check for recent reversal (within last 3 candles)
        if len(trend) < self.ema_length + 3:
            return None
        
        recent_trends = trend[-3:]
        current_trend = trend[-1]
        prev_trend = trend[-2]
        
        # Detect reversal
        if current_trend != prev_trend:
            direction = "BULLISH" if current_trend == 1 else "BEARISH"
            return (direction, most[-1], 'REVERSAL')
        
        # Pullback entry: price tags EMA and bounces back with same trend
        prev_close = closes[-2]
        current_close = closes[-1]
        ema_val = ema[-1]
        pullback = False
        if current_trend == 1:
            touched = lows[-1] <= ema_val <= closes[-1]
            pullback = touched and current_close > ema_val and current_close > prev_close
        else:
            touched = highs[-1] >= ema_val >= closes[-1]
            pullback = touched and current_close < ema_val and current_close < prev_close
        if pullback:
            direction = "BULLISH" if current_trend == 1 else "BEARISH"
            return (direction, most[-1], 'PULLBACK')
        
        return None
    
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
    """MEXC exchange client wrapper."""

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
class MOSTSignal:
    """Represents a MOST indicator signal."""
    symbol: str
    direction: str
    timestamp: str
    most_value: float
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


class MOSTBot:
    """Main MOST Bot."""
    
    def __init__(self, interval: int = 60, default_cooldown: int = 5):
        self.interval = interval
        self.default_cooldown = default_cooldown
        self.watchlist: List[WatchItem] = load_watchlist()
        self.client = MexcClient()
        self.analyzer = MOSTAnalyzer()
        self.state = BotState(STATE_FILE)
        self.notifier = self._init_notifier()
        self.stats = SignalStats("MOST Bot", STATS_FILE)
        self.health_monitor = (
            HealthMonitor("MOST Bot", self.notifier, heartbeat_interval=3600)
            if HealthMonitor and self.notifier else None
        )
        self.rate_limiter = (
            RateLimiter(calls_per_minute=60, backoff_file=LOG_DIR / "rate_limiter.json")
            if RateLimiter else None
        )
        self.exchange_backoff: Dict[str, float] = {}
        self.exchange_delay: Dict[str, float] = {}
    
    def _init_notifier(self) -> Optional[Any]:
        if TelegramNotifier is None:
            logger.warning("Telegram notifier unavailable")
            return None
        token = os.getenv("TELEGRAM_BOT_TOKEN_MOST") or os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            logger.warning("Telegram credentials missing")
            return None
        return TelegramNotifier(
            bot_token=token,
            chat_id=chat_id,
            signals_log_file=str(LOG_DIR / "most_signals.json")
        )

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        msg = str(exc)
        return "Requests are too frequent" in msg or "429" in msg or "403 Forbidden" in msg

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
    
    def run(self, loop: bool = False) -> None:
        if not self.watchlist:
            logger.error("Empty watchlist; exiting")
            return
        
        logger.info("Starting MOST Bot for %d symbols", len(self.watchlist))
        
        if self.health_monitor:
            self.health_monitor.send_startup_message()
        
        try:
            while not shutdown_requested:
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

                    if shutdown_requested:
                        break
                    logger.info("Cycle complete; sleeping %ds", self.interval)
                    # Sleep in 1-second chunks to respond quickly to shutdown signals
                    for _ in range(self.interval):
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
            
            # Exchange-level backoff for rate limits
            if self._backoff_active("mexc"):
                logger.debug("Backoff active for mexc; skipping %s", symbol)
                continue

            if self.rate_limiter:
                self.rate_limiter.wait_if_needed()
            
            try:
                signal = self._analyze_symbol(symbol, period)
                if self.rate_limiter:
                    self.rate_limiter.record_success(f"mexc_{symbol}")
            except Exception as exc:
                logger.error("Failed to analyze %s: %s", symbol, exc)
                if self._is_rate_limit_error(exc):
                    self._register_backoff("mexc")
                if self.rate_limiter:
                    self.rate_limiter.record_error(f"mexc_{symbol}")
                if self.health_monitor:
                    self.health_monitor.record_error(f"Analysis error for {symbol}: {exc}")
                continue
            
            if signal is None:
                logger.debug("%s: No MOST signal", symbol)
                continue
            
            if not self.state.can_alert(symbol, cooldown):
                logger.debug("Cooldown active for %s", symbol)
                continue
            
            # Max open signals limit
            MAX_OPEN_SIGNALS = 50
            current_open = len(self.state.iter_signals())
            if current_open >= MAX_OPEN_SIGNALS:
                logger.info(
                    "Max open signals limit reached (%d/%d). Skipping %s",
                    current_open, MAX_OPEN_SIGNALS, symbol
                )
                continue
            
            # Send alert
            message = self._format_message(signal)
            self._dispatch(message)
            self.state.mark_alert(symbol)
            
            # Track signal
            signal_id = f"{symbol}-MOST-{signal.timestamp}"
            trade_data = {
                "id": signal_id,
                "symbol": symbol,
                "direction": signal.direction,
                "entry": signal.entry,
                "stop_loss": signal.stop_loss,
                "take_profit_1": signal.take_profit_1,
                "take_profit_2": signal.take_profit_2,
                "created_at": signal.timestamp,
                "timeframe": period,
                "exchange": "MEXC",
            }
            self.state.add_signal(signal_id, trade_data)
            
            if self.stats:
                self.stats.record_open(
                    signal_id=signal_id,
                    symbol=f"{symbol}/USDT",
                    direction=signal.direction,
                    entry=signal.entry,
                    created_at=signal.timestamp,
                    extra={
                        "timeframe": period,
                        "exchange": "MEXC",
                        "strategy": "MOST",
                    },
                )
            
            time.sleep(0.5)
    
    def _analyze_symbol(self, symbol: str, timeframe: str) -> Optional[MOSTSignal]:
        """Analyze symbol for MOST signals."""
        # Fetch OHLCV data
        ohlcv = self.client.fetch_ohlcv(symbol, timeframe, limit=100)
        
        if len(ohlcv) < 50:
            return None
        
        highs = np.array([x[2] for x in ohlcv])
        lows = np.array([x[3] for x in ohlcv])
        closes = np.array([x[4] for x in ohlcv])
        
        # Detect signal
        result = self.analyzer.detect_signal(highs, lows, closes)
        
        if result is None:
            return None
        
        direction, most_val_raw, setup_type = result
        direction = str(direction)
        most_value = float(most_val_raw)
        
        # Calculate ATR
        atr = float(self.analyzer.calculate_atr(highs, lows, closes))
        
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
        
        entry = current_price
        
        # Validate BEARISH/BULLISH setup before TP/SL calculation
        if direction == "BULLISH":
            # For BULLISH: MOST should be below entry (support level)
            if most_value >= entry:
                logger.error("Invalid BULLISH setup: MOST %.6f >= entry %.6f (direction contradiction)", most_value, entry)
                return None
        else:  # BEARISH
            # For BEARISH: MOST should be above entry (resistance level)
            if most_value <= entry:
                logger.error("Invalid BEARISH setup: MOST %.6f <= entry %.6f (direction contradiction)", most_value, entry)
                return None
        
        # Get risk config and calculate TP/SL using centralized calculator
        config_mgr = get_config_manager()
        risk_config = config_mgr.get_effective_risk("most_bot", symbol)
        tp1_mult = risk_config.tp1_atr_multiplier
        tp2_mult = risk_config.tp2_atr_multiplier
        min_rr = risk_config.min_risk_reward

        calculator = TPSLCalculator(min_risk_reward=0.8, min_risk_reward_tp2=1.5)
        levels = calculator.calculate(
            entry=entry,
            direction=direction,
            atr=atr,
            tp1_multiplier=tp1_mult,
            tp2_multiplier=tp2_mult,
            custom_sl=most_value,  # Use MOST as trailing stop
        )

        if not levels.is_valid:
            logger.info("Signal rejected for %s: %s", symbol, levels.rejection_reason)
            return None

        sl = most_value  # Keep MOST as trailing stop
        tp1 = levels.take_profit_1
        tp2 = levels.take_profit_2

        logger.debug("%s %s signal | Entry: %.6f | SL: %.6f | TP1: %.6f | TP2: %.6f | R:R: 1:%.2f / 1:%.2f",
                   symbol, direction, entry, sl, tp1, tp2, levels.risk_reward_1, levels.risk_reward_2)
        
        signal = MOSTSignal(
            symbol=symbol,
            direction=direction,
            timestamp=human_ts(),
            most_value=most_value,
            entry=entry,
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp2,
            current_price=current_price,
        )
        
        logger.info("MOST signal created: %s | %s | Entry: %.6f | SL: %.6f | TP1: %.6f | TP2: %.6f",
                   symbol, direction, entry, sl, tp1, tp2)
        return signal
    
    def _format_message(self, signal: MOSTSignal) -> str:
        """Format Telegram message for signal using centralized template."""
        # Get performance stats
        perf_stats = None
        if self.stats is not None:
            symbol_key = signal.symbol if "/" in signal.symbol else f"{signal.symbol}/USDT"
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

        return format_signal_message(
            bot_name="MOST",
            symbol=f"{signal.symbol}/USDT",
            direction=signal.direction,
            entry=signal.entry,
            stop_loss=signal.stop_loss,
            tp1=signal.take_profit_1,
            tp2=signal.take_profit_2,
            indicator_value=signal.most_value,
            indicator_name="MOST",
            exchange="MEXC",
            timeframe="15m",
            current_price=signal.current_price,
            performance_stats=perf_stats,
            extra_info="MOST acts as trailing stop - adjust as it moves",
        )

    def _symbol_perf_line(self, symbol: str) -> Optional[str]:
        """Build a compact TP/SL history line for the symbol."""
        if not self.stats:
            return None
        symbol_key = symbol if "/" in symbol else f"{symbol}/USDT"
        counts = self.stats.symbol_tp_sl_counts(symbol_key)
        tp1 = counts.get("TP1", 0)
        tp2 = counts.get("TP2", 0)
        sl = counts.get("SL", 0)
        total = tp1 + tp2 + sl
        if total == 0:
            return "📈 <b>History:</b> No previous trades for this symbol"
        win_rate = (tp1 + tp2) / total * 100.0
        return (
            f"📈 <b>History:</b> TP1 {tp1} | TP2 {tp2} | SL {sl} "
            f"(Win rate: {win_rate:.1f}%)"
        )
    
    def _monitor_open_signals(self) -> None:
        """Monitor open signals for TP/SL hits with improved tolerance."""
        signals = self.state.iter_signals()
        if not signals:
            return
        
        # Price tolerance for partial fills/slippage (0.5%)
        PRICE_TOLERANCE = 0.005
        
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
            
            # Check for TP/SL with tolerance for slippage
            result = None
            if direction == "BULLISH":
                # With tolerance for slippage (allow slightly lower prices)
                hit_tp2 = price >= (tp2 * (1 - PRICE_TOLERANCE))
                hit_tp1 = price >= (tp1 * (1 - PRICE_TOLERANCE)) and price < (tp2 * (1 - PRICE_TOLERANCE))
                hit_sl = price <= (sl * (1 + PRICE_TOLERANCE))
            else:
                # For BEARISH, allow slightly higher prices
                hit_tp2 = price <= (tp2 * (1 + PRICE_TOLERANCE))
                hit_tp1 = price <= (tp1 * (1 + PRICE_TOLERANCE)) and price > (tp2 * (1 + PRICE_TOLERANCE))
                hit_sl = price >= (sl * (1 - PRICE_TOLERANCE))
            
            # Priority: TP2 > TP1 > SL
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
                
                if summary_message:
                    self._dispatch(summary_message)
                else:
                    message = (
                        f"🎯 {symbol}/USDT MOST {direction} {result} hit!\n"
                        f"Entry <code>{entry:.6f}</code> | Exit <code>{price:.6f}</code>\n"
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
    parser = argparse.ArgumentParser(description="MOST Bot")
    parser.add_argument("--once", action="store_true", help="Run only one cycle")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between cycles")
    parser.add_argument("--cooldown", type=int, default=5, help="Default cooldown minutes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bot = MOSTBot(interval=args.interval, default_cooldown=args.cooldown)
    bot.run(loop=not args.once)


if __name__ == "__main__":
    main()
