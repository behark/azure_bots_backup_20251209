#!/usr/bin/env python3
"""
Fibonacci Reversal Bot - Speed-based Fibonacci retracement strategy
Detects swing highs/lows and Fibonacci reversal zones
Speed modes: scalp (fast), fast, medium, safe (slow)
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
from typing import Any, Dict, List, Optional, TypedDict, cast

import ccxt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
STATE_FILE = BASE_DIR / "fib_reversal_state.json"
WATCHLIST_FILE = BASE_DIR / "fib_reversal_watchlist.json"
STATS_FILE = LOG_DIR / "fib_reversal_stats.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "fib_reversal_bot.log"),
    ],
)

logger = logging.getLogger("fib_reversal_bot")

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
    load_dotenv(BASE_DIR.parent / ".env")
except ImportError:
    pass

sys.path.append(str(BASE_DIR.parent))
try:
    from notifier import TelegramNotifier
    from signal_stats import SignalStats
    from health_monitor import HealthMonitor, RateLimiter
    from tp_sl_calculator import TPSLCalculator, CalculationMethod
    from trade_config import get_config_manager
    from rate_limit_handler import RateLimitHandler
except ImportError:
    TelegramNotifier = None
    SignalStats = None
    HealthMonitor = None
    RateLimiter = None
    TPSLCalculator = None
    CalculationMethod = None
    get_config_manager = None
    RateLimitHandler = None


class WatchItem(TypedDict, total=False):
    symbol: str
    period: str
    speed: str
    cooldown_minutes: int


# Speed presets
SPEED_PRESETS = {
    'fast': {'lookback': 5, 'confirmation': 2},
    'medium': {'lookback': 10, 'confirmation': 3},
    'safe': {'lookback': 15, 'confirmation': 4},
}


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
        symbol_val = item.get("symbol")
        if not isinstance(symbol_val, str):
            continue
        period_val = item.get("period", "5m")
        speed_val = item.get("speed", "medium")
        cooldown_raw = item.get("cooldown_minutes", 30)
        try:
            cooldown = int(cooldown_raw)
        except Exception:
            cooldown = 30
        normalized.append(
            {
                "symbol": symbol_val.upper(),
                "period": period_val if isinstance(period_val, str) else "5m",
                "speed": speed_val if isinstance(speed_val, str) else "medium",
                "cooldown_minutes": cooldown,
            }
        )
    return normalized


def human_ts() -> str:
    """Return human-readable UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class FibonacciAnalyzer:
    """Analyzes Fibonacci retracements from swing points."""
    
    def __init__(self, speed: str = 'medium'):
        self.speed = speed
        self.lookback = SPEED_PRESETS[speed]['lookback']
        self.confirmation = SPEED_PRESETS[speed]['confirmation']
    
    def find_swing_high(self, highs: np.ndarray, index: int) -> bool:
        """Check if index is a swing high."""
        if index < self.lookback or index >= len(highs) - self.lookback:
            return False
        window_start = index - self.lookback
        window_end = index + self.lookback + 1
        return highs[index] == max(highs[window_start:window_end])
    
    def find_swing_low(self, lows: np.ndarray, index: int) -> bool:
        """Check if index is a swing low."""
        if index < self.lookback or index >= len(lows) - self.lookback:
            return False
        window_start = index - self.lookback
        window_end = index + self.lookback + 1
        return lows[index] == min(lows[window_start:window_end])
    
    def detect_swing_points(self, highs: np.ndarray, lows: np.ndarray) -> tuple:
        """Detect most recent swing high and low."""
        swing_high_idx, swing_high_price = None, None
        swing_low_idx, swing_low_price = None, None
        
        # Search backwards for most recent swing points
        for i in range(len(highs) - self.lookback - 1, self.lookback, -1):
            if swing_high_idx is None and self.find_swing_high(highs, i):
                swing_high_idx = i
                swing_high_price = highs[i]
            if swing_low_idx is None and self.find_swing_low(lows, i):
                swing_low_idx = i
                swing_low_price = lows[i]
            if swing_high_idx is not None and swing_low_idx is not None:
                break
        
        return (swing_high_idx, swing_high_price), (swing_low_idx, swing_low_price)
    
    def check_confirmation(self, lows: np.ndarray, swing_low_idx: int) -> bool:
        """Check if swing low has been confirmed (price hasn't broken it)."""
        if swing_low_idx is None:
            return False
        
        bars_since = len(lows) - 1 - swing_low_idx
        if bars_since < self.confirmation:
            return False
        
        swing_low_price = lows[swing_low_idx]
        recent_lows = lows[swing_low_idx + 1:]
        
        # Confirmed if no bar has gone below swing low
        return all(low >= swing_low_price for low in recent_lows)
    
    def detect_fib_reversal(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        opens: np.ndarray,
    ) -> Optional[tuple]:
        """
        Detect Fibonacci reversal signals.
        Returns (direction, fib_level, swing_high, swing_low) or None.
        """
        (swing_high_idx, swing_high), (swing_low_idx, swing_low) = self.detect_swing_points(highs, lows)
        
        if swing_high is None or swing_low is None:
            return None
        
        current_price = closes[-1]
        current_open = opens[-1]
        current_high = highs[-1]
        current_low = lows[-1]

        def _candle_supports(direction: str) -> bool:
            body = abs(current_price - current_open)
            upper_wick = max(0.0, current_high - max(current_price, current_open))
            lower_wick = max(0.0, min(current_price, current_open) - current_low)
            if direction == 'BULLISH':
                return current_price >= current_open or (body > 0 and lower_wick >= 2 * body)
            if direction == 'BEARISH':
                return current_price <= current_open or (body > 0 and upper_wick >= 2 * body)
            return False
        
        # Determine trend: which swing point is more recent?
        if swing_high_idx > swing_low_idx:
            # Downtrend - look for bullish reversal from swing low
            if not self.check_confirmation(lows, swing_low_idx):
                return None
            
            fib_range = swing_high - swing_low
            fib_levels = {
                '0.236': swing_low + (fib_range * 0.236),
                '0.382': swing_low + (fib_range * 0.382),
                '0.5': swing_low + (fib_range * 0.5),
                '0.618': swing_low + (fib_range * 0.618),
                '0.65': swing_low + (fib_range * 0.65),
                '0.786': swing_low + (fib_range * 0.786),
            }

            tolerance = fib_range * 0.05
            gp_tolerance = fib_range * 0.02
            golden_low = fib_levels['0.618']
            golden_high = fib_levels['0.65']

            if (golden_low - gp_tolerance) <= current_price <= (golden_high + gp_tolerance):
                if not _candle_supports('BULLISH'):
                    return None
                return ('BULLISH', 0.618, swing_high, swing_low, fib_levels, 'GOLDEN_POCKET')

            if abs(current_price - fib_levels['0.618']) < tolerance:
                if not _candle_supports('BULLISH'):
                    return None
                return ('BULLISH', 0.618, swing_high, swing_low, fib_levels, 'STANDARD')
            if abs(current_price - fib_levels['0.5']) < tolerance:
                if not _candle_supports('BULLISH'):
                    return None
                return ('BULLISH', 0.5, swing_high, swing_low, fib_levels, 'STANDARD')
            if abs(current_price - fib_levels['0.382']) < tolerance:
                if not _candle_supports('BULLISH'):
                    return None
                return ('BULLISH', 0.382, swing_high, swing_low, fib_levels, 'STANDARD')
        
        else:
            # Uptrend - look for bearish reversal from swing high
            bars_since_high = len(highs) - 1 - swing_high_idx
            if bars_since_high < self.confirmation:
                return None
            
            swing_high_price = highs[swing_high_idx]
            recent_highs = highs[swing_high_idx + 1:]
            confirmed = all(high <= swing_high_price for high in recent_highs)
            
            if not confirmed:
                return None
            
            fib_range = swing_high - swing_low
            fib_levels = {
                '0.236': swing_high - (fib_range * 0.236),
                '0.382': swing_high - (fib_range * 0.382),
                '0.5': swing_high - (fib_range * 0.5),
                '0.618': swing_high - (fib_range * 0.618),
                '0.65': swing_high - (fib_range * 0.65),
                '0.786': swing_high - (fib_range * 0.786),
            }
            tolerance = fib_range * 0.05
            gp_tolerance = fib_range * 0.02
            golden_high = fib_levels['0.618']
            golden_low = fib_levels['0.65']

            if (golden_low - gp_tolerance) <= current_price <= (golden_high + gp_tolerance):
                if not _candle_supports('BEARISH'):
                    return None
                return ('BEARISH', 0.618, swing_high, swing_low, fib_levels, 'GOLDEN_POCKET')

            if abs(current_price - fib_levels['0.618']) < tolerance:
                if not _candle_supports('BEARISH'):
                    return None
                return ('BEARISH', 0.618, swing_high, swing_low, fib_levels, 'STANDARD')
            if abs(current_price - fib_levels['0.5']) < tolerance:
                if not _candle_supports('BEARISH'):
                    return None
                return ('BEARISH', 0.5, swing_high, swing_low, fib_levels, 'STANDARD')
            if abs(current_price - fib_levels['0.382']) < tolerance:
                if not _candle_supports('BEARISH'):
                    return None
                return ('BEARISH', 0.382, swing_high, swing_low, fib_levels, 'STANDARD')
        
        return None
    
    @staticmethod
    def calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
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
    
    def __init__(self):
        self.exchange = ccxt.mexc({  # type: ignore[call-arg]
            "enableRateLimit": True,
            "options": {"defaultType": "swap"}
        })
        self.exchange.load_markets()
        self.rate_limiter = RateLimitHandler(base_delay=0.5, max_retries=5) if RateLimitHandler else None
    
    @staticmethod
    def _swap_symbol(symbol: str) -> str:
        return f"{symbol.upper()}/USDT:USDT"
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = "5m", limit: int = 100) -> list:
        """Fetch OHLCV data."""
        if self.rate_limiter:
            return self.rate_limiter.execute(
                self.exchange.fetch_ohlcv,
                self._swap_symbol(symbol),
                timeframe=timeframe,
                limit=limit
            )
        return self.exchange.fetch_ohlcv(
            self._swap_symbol(symbol),
            timeframe=timeframe,
            limit=limit
        )
    
    def fetch_ticker(self, symbol: str) -> dict:
        """Fetch ticker data."""
        if self.rate_limiter:
            return self.rate_limiter.execute(self.exchange.fetch_ticker, self._swap_symbol(symbol))
        return self.exchange.fetch_ticker(self._swap_symbol(symbol))


# Graceful shutdown handling
shutdown_requested = False


def signal_handler(signum, frame) -> None:  # pragma: no cover - signal path
    """Handle shutdown signals (SIGINT, SIGTERM) gracefully."""
    global shutdown_requested
    shutdown_requested = True
    logger.info("Received %s, shutting down gracefully...", signal.Signals(signum).name)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@dataclass
class FibReversalSignal:
    """Represents a Fibonacci reversal signal."""
    symbol: str
    direction: str
    timestamp: str
    fib_level: float
    swing_high: float
    swing_low: float
    speed: str
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    current_price: Optional[float] = None
    setup_type: str = "STANDARD"
    fib_levels: Optional[Dict[str, float]] = None


class BotState:
    """Manages bot state persistence."""
    
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, Any] = self._load()
    
    @staticmethod
    def _parse_ts(value: str) -> datetime:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    
    def _empty_state(self) -> Dict[str, Any]:
        return {"last_alert": {}, "open_signals": {}}

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return self._empty_state()
        try:
            data = json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return self._empty_state()
        if not isinstance(data, dict):
            return self._empty_state()
        last = data.get("last_alert")
        if not isinstance(last, dict):
            data["last_alert"] = {}
        open_signals = data.get("open_signals")
        if not isinstance(open_signals, dict):
            data["open_signals"] = {}
        return data
    
    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2))
    
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
    
    def add_signal(self, signal_id: str, payload: Dict[str, Any]) -> None:
        signals = self.data.setdefault("open_signals", {})
        if not isinstance(signals, dict):
            signals = {}
            self.data["open_signals"] = signals
        signals[signal_id] = payload
        self.save()
    
    def remove_signal(self, signal_id: str) -> None:
        signals = self.data.setdefault("open_signals", {})
        if not isinstance(signals, dict):
            self.data["open_signals"] = {}
            return
        if signal_id in signals:
            signals.pop(signal_id)
            self.save()
    
    def iter_signals(self) -> Dict[str, Dict[str, Any]]:
        signals = self.data.setdefault("open_signals", {})
        if not isinstance(signals, dict):
            signals = {}
            self.data["open_signals"] = signals
        return cast(Dict[str, Dict[str, Any]], signals)


class FibReversalBot:
    """Main Fibonacci Reversal Bot."""
    
    def __init__(self, interval: int = 300, default_cooldown: int = 30):
        self.interval = interval
        self.default_cooldown = default_cooldown
        self.watchlist: List[WatchItem] = load_watchlist()
        self.client = MexcClient()
        self.state = BotState(STATE_FILE)
        self.notifier = self._init_notifier()
        self.stats = SignalStats("Fib Reversal Bot", STATS_FILE) if SignalStats else None
        self.health_monitor = (
            HealthMonitor("Fib Reversal Bot", self.notifier, heartbeat_interval=3600)
            if HealthMonitor and self.notifier else None
        )
        self.rate_limiter = (
            RateLimiter(calls_per_minute=60, backoff_file=LOG_DIR / "rate_limiter.json")
            if RateLimiter else None
        )
    
    def _init_notifier(self):
        if TelegramNotifier is None:
            logger.warning("Telegram notifier unavailable")
            return None
        token = os.getenv("TELEGRAM_BOT_TOKEN_FIB_REVERSAL") or os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            logger.warning("Telegram credentials missing")
            return None
        return TelegramNotifier(
            bot_token=token,
            chat_id=chat_id,
            signals_log_file=str(LOG_DIR / "fib_reversal_signals.json")
        )
    
    def run(self, loop: bool = False) -> None:
        if not self.watchlist:
            logger.error("Empty watchlist; exiting")
            return
        
        logger.info("Starting Fibonacci Reversal Bot for %d symbols", len(self.watchlist))
        
        if self.health_monitor:
            self.health_monitor.send_startup_message()
        
        try:
            while not shutdown_requested:
                try:
                    self._run_cycle()
                    self._monitor_open_signals()
                    
                    if self.health_monitor:
                        self.health_monitor.record_cycle()
                    
                    if not loop:
                        break

                    if shutdown_requested:
                        break
                    logger.info("Cycle complete; sleeping %ds", self.interval)
                    time.sleep(self.interval)
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
            period_val = item.get("period", "5m") if isinstance(item, dict) else "5m"
            speed_val = item.get("speed", "medium") if isinstance(item, dict) else "medium"
            cooldown_raw = item.get("cooldown_minutes", self.default_cooldown) if isinstance(item, dict) else self.default_cooldown
            period = period_val if isinstance(period_val, str) else "5m"
            speed = speed_val if isinstance(speed_val, str) else "medium"
            try:
                cooldown = int(cooldown_raw)
            except Exception:
                cooldown = self.default_cooldown
            
            if self.rate_limiter:
                self.rate_limiter.wait_if_needed()
            
            try:
                signal = self._analyze_symbol(symbol, period, speed)
                if self.rate_limiter:
                    self.rate_limiter.record_success(f"mexc_{symbol}")
            except Exception as exc:
                logger.error("Failed to analyze %s: %s", symbol, exc)
                if self.rate_limiter:
                    self.rate_limiter.record_error(f"mexc_{symbol}")
                if self.health_monitor:
                    self.health_monitor.record_error(f"Analysis error for {symbol}: {exc}")
                continue
            
            if signal is None:
                logger.debug("%s: No Fib reversal signal", symbol)
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
            signal_id = f"{symbol}-FibReversal-{signal.timestamp}"
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
                "fib_level": signal.fib_level,
                "setup_type": signal.setup_type,
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
                        "strategy": "Fibonacci Reversal",
                        "fib_level": signal.fib_level,
                        "speed": signal.speed,
                        "setup_type": signal.setup_type,
                    },
                )
            
            time.sleep(0.5)
    
    def _analyze_symbol(self, symbol: str, timeframe: str, speed: str) -> Optional[FibReversalSignal]:
        """Analyze symbol for Fibonacci reversals."""
        # Fetch OHLCV data
        ohlcv = self.client.fetch_ohlcv(symbol, timeframe, limit=100)
        
        if len(ohlcv) < 50:
            return None
        
        try:
            opens = np.array([x[1] for x in ohlcv], dtype=float)
            highs = np.array([x[2] for x in ohlcv], dtype=float)
            lows = np.array([x[3] for x in ohlcv], dtype=float)
            closes = np.array([x[4] for x in ohlcv], dtype=float)
        except Exception:
            return None
        
        # Create analyzer with speed preset
        analyzer = FibonacciAnalyzer(speed)
        
        # Detect Fib reversal
        result = analyzer.detect_fib_reversal(highs, lows, closes, opens)
        
        if result is None:
            return None
        
        direction, fib_level, swing_high, swing_low, fib_levels, setup_type = result
        fib_levels = fib_levels or {}
        
        # Calculate ATR
        atr = float(analyzer.calculate_atr(highs, lows, closes))

        def tiered_stop_loss() -> float:
            if direction == "BULLISH":
                if abs(fib_level - 0.382) < 1e-6 and '0.5' in fib_levels:
                    return float(fib_levels['0.5'] * 0.995)
                if abs(fib_level - 0.5) < 1e-6 and '0.618' in fib_levels:
                    return float(fib_levels['0.618'] * 0.995)
                if fib_level >= 0.618:
                    ref = fib_levels.get('0.786', swing_low)
                    return float(ref * 0.997)
                return float(swing_low * 0.995)
            else:
                if abs(fib_level - 0.382) < 1e-6 and '0.5' in fib_levels:
                    return float(fib_levels['0.5'] * 1.005)
                if abs(fib_level - 0.5) < 1e-6 and '0.618' in fib_levels:
                    return float(fib_levels['0.618'] * 1.005)
                if fib_level >= 0.618:
                    ref = fib_levels.get('0.786', swing_high)
                    return float(ref * 1.003)
                return float(swing_high * 1.005)
        
        # Get current price
        ticker = self.client.fetch_ticker(symbol)
        current_price = ticker.get("last") if isinstance(ticker, dict) else None
        if current_price is None and isinstance(ticker, dict):
            current_price = ticker.get("close")
        if not isinstance(current_price, (int, float)):
            return None
        entry = float(current_price)
        
        if TPSLCalculator is not None and CalculationMethod is not None:
            tp1_mult = 2.0
            tp2_mult = 3.5
            min_rr = 1.5
            if get_config_manager is not None:
                try:
                    config_mgr = get_config_manager()
                    risk_config = config_mgr.get_effective_risk("fib_reversal_bot", symbol)
                    tp1_mult = risk_config.tp1_atr_multiplier
                    tp2_mult = risk_config.tp2_atr_multiplier
                    min_rr = risk_config.min_risk_reward
                except Exception:
                    pass

            calculator = TPSLCalculator(min_risk_reward=min_rr)
            levels = calculator.calculate(
                entry=entry,
                direction=direction,
                swing_high=swing_high if direction == "BEARISH" else None,
                swing_low=swing_low if direction == "BULLISH" else None,
                tp1_multiplier=tp1_mult,
                tp2_multiplier=tp2_mult,
                method=CalculationMethod.STRUCTURE,
            )

            if levels.is_valid:
                sl = float(levels.stop_loss)
                tp1 = float(levels.take_profit_1)
                tp2 = float(levels.take_profit_2)
            else:
                logger.info("Signal rejected for %s: %s", symbol, levels.rejection_reason)
                return None
        else:
            if direction == "BULLISH":
                sl = tiered_stop_loss()
                tp1 = float(entry + (atr * 2.0))
                tp2 = float(entry + (atr * 3.5))
            else:
                sl = tiered_stop_loss()
                tp1 = float(entry - (atr * 2.0))
                tp2 = float(entry - (atr * 3.5))
        
        return FibReversalSignal(
            symbol=symbol,
            direction=direction,
            timestamp=human_ts(),
            fib_level=float(fib_level),
            swing_high=float(swing_high),
            swing_low=float(swing_low),
            speed=speed,
            entry=entry,
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp2,
            current_price=float(current_price),
            setup_type=setup_type,
            fib_levels={k: float(v) for k, v in fib_levels.items()},
        )
    
    def _format_message(self, signal: FibReversalSignal) -> str:
        """Format Telegram message for signal."""
        direction = signal.direction
        emoji = "🟢" if direction == "BULLISH" else "🔴"
        
        current_price = signal.current_price if isinstance(signal.current_price, (int, float)) else None
        lines = [
            f"{emoji} <b>{direction} FIB REVERSAL - {signal.symbol}/USDT</b>",
            "",
            "📊 <b>Strategy:</b> Fibonacci Retracement",
            f"📐 <b>Fib Level:</b> {signal.fib_level:.3f}",
            f"⚡ <b>Speed:</b> {signal.speed.upper()}",
            f"🧭 <b>Setup:</b> {signal.setup_type.replace('_', ' ')}",
        ]
        if current_price is not None:
            lines.append(f"💰 <b>Current Price:</b> <code>{current_price:.6f}</code>")
        lines.extend([
            "",
            f"📍 Swing High: <code>{signal.swing_high:.6f}</code>",
            f"📍 Swing Low: <code>{signal.swing_low:.6f}</code>",
            "",
            "<b>🎯 TRADE LEVELS:</b>",
            f"🟢 Entry: <code>{signal.entry:.6f}</code>",
            f"🛑 Stop Loss: <code>{signal.stop_loss:.6f}</code>",
            f"🎯 TP1: <code>{signal.take_profit_1:.6f}</code>",
            f"🚀 TP2: <code>{signal.take_profit_2:.6f}</code>",
        ])
        
        # Calculate R:R
        risk = abs(signal.entry - signal.stop_loss)
        reward1 = abs(signal.take_profit_1 - signal.entry)
        reward2 = abs(signal.take_profit_2 - signal.entry)
        
        if risk > 0:
            rr1 = reward1 / risk
            rr2 = reward2 / risk
            lines.append("")
            lines.append(f"⚖️ <b>Risk/Reward:</b> 1:{rr1:.2f} (TP1) | 1:{rr2:.2f} (TP2)")

        # Historical TP/SL stats per symbol
        if self.stats is not None:
            symbol_key = f"{signal.symbol}/USDT"
            counts = self.stats.symbol_tp_sl_counts(symbol_key)
            tp1 = counts.get("TP1", 0)
            tp2 = counts.get("TP2", 0)
            sl = counts.get("SL", 0)
            total = tp1 + tp2 + sl
            if total > 0:
                win_rate = (tp1 + tp2) / total * 100.0
                lines.append("")
                lines.append(
                    f"📈 <b>History:</b> TP1 {tp1} | TP2 {tp2} | SL {sl} "
                    f"(Win rate: {win_rate:.1f}%)"
                )
        
        lines.extend([
            "",
            f"⏱️ {signal.timestamp}",
        ])
        
        return "\n".join(lines)
    
    def _monitor_open_signals(self) -> None:
        """Monitor open signals for TP/SL hits."""
        signals = self.state.iter_signals()
        if not signals:
            return
        
        for signal_id, payload in list(signals.items()):
            if not isinstance(payload, dict):
                self.state.remove_signal(signal_id)
                continue
            symbol = payload.get("symbol")
            if not isinstance(symbol, str):
                self.state.remove_signal(signal_id)
                continue
            
            try:
                ticker = self.client.fetch_ticker(symbol)
                price = ticker.get("last") if isinstance(ticker, dict) else None
                if price is None and isinstance(ticker, dict):
                    price = ticker.get("close")
            except Exception as exc:
                logger.warning("Failed to fetch ticker for %s: %s", signal_id, exc)
                continue
            
            if not isinstance(price, (int, float)):
                continue
            
            direction = payload.get("direction")
            entry_raw = payload.get("entry")
            tp1_raw = payload.get("take_profit_1")
            tp2_raw = payload.get("take_profit_2")
            sl_raw = payload.get("stop_loss")
            if not isinstance(direction, str):
                self.state.remove_signal(signal_id)
                if self.stats:
                    self.stats.discard(signal_id)
                continue
            if not all(isinstance(v, (int, float, str)) for v in (entry_raw, tp1_raw, tp2_raw, sl_raw)):
                self.state.remove_signal(signal_id)
                if self.stats:
                    self.stats.discard(signal_id)
                continue
            try:
                entry = float(entry_raw)  # type: ignore[arg-type]
                tp1 = float(tp1_raw)      # type: ignore[arg-type]
                tp2 = float(tp2_raw)      # type: ignore[arg-type]
                sl = float(sl_raw)        # type: ignore[arg-type]
            except Exception:
                self.state.remove_signal(signal_id)
                if self.stats:
                    self.stats.discard(signal_id)
                continue
            
            if direction == "BULLISH":
                hit_tp2 = price >= tp2
                hit_tp1 = price >= tp1
                hit_sl = price <= sl
            else:
                hit_tp2 = price <= tp2
                hit_tp1 = price <= tp1
                hit_sl = price >= sl
            
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
                    else:
                        self.stats.discard(signal_id)
                
                if summary_message:
                    self._dispatch(summary_message)
                else:
                    message = (
                        f"🎯 {symbol}/USDT Fib Reversal {direction} {result} hit!\n"
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
    parser = argparse.ArgumentParser(description="Fibonacci Reversal Bot")
    parser.add_argument("--loop", action="store_true", help="Run indefinitely")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between cycles")
    parser.add_argument("--cooldown", type=int, default=30, help="Default cooldown minutes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bot = FibReversalBot(interval=args.interval, default_cooldown=args.cooldown)
    bot.run(loop=args.loop)


if __name__ == "__main__":
    main()
