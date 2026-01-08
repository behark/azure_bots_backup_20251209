#!/usr/bin/env python3
"""
Fibonacci Swing Trading Bot
Based on Fibonacci retracement + Swing confirmation strategy

Detects:
- Swing High/Low pivot points
- Fibonacci retracement levels (38.2%, 50%, 61.8%)
- Swing Low Confirmation (held for X candles)
- Entry signals: Premium, Confirmed, Standard

Entry Types:
- PREMIUM:   Fibonacci level + Swing confirmation (BEST!)
- CONFIRMED: Swing low confirmed + near bottom
- STANDARD:  At Fibonacci level in uptrend
"""


import sys
import os
import signal
import json
import time
import logging
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    # Windows doesn't have fcntl - use a no-op fallback
    HAS_FCNTL = False
from pathlib import Path
from threading import Lock
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from types import FrameType
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import ccxt  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

# Ensure logs directory exists before configuring logging
_LOGS_DIR = Path(__file__).parent / "logs"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Logging setup (must be before any logger use)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(_LOGS_DIR / "fib_swing_bot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
_parent_path = str(Path(__file__).parent.parent)
if _parent_path not in sys.path:
    sys.path.insert(0, _parent_path)

# Required imports (fail fast if missing)
from message_templates import format_signal_message, format_result_message
from notifier import TelegramNotifier
from signal_stats import SignalStats
from tp_sl_calculator import TPSLCalculator, CalculationMethod
from trade_config import get_config_manager

# NEW: Import unified signal system
from core.bot_signal_mixin import BotSignalMixin, create_price_fetcher

# Optional imports (safe fallback)
from safe_import import safe_import
HealthMonitor = safe_import('health_monitor', 'HealthMonitor')
RateLimiter = None  # Disabled for testing
RateLimitHandler = safe_import('rate_limit_handler', 'RateLimitHandler')

# Result notification toggle - using signal_only mode now
ENABLE_RESULT_NOTIFICATIONS = False


# Graceful shutdown handling
shutdown_requested = False


def signal_handler(signum: int, frame: Optional[FrameType]) -> None:  # pragma: no cover - signal path
    """Handle shutdown signals (SIGINT, SIGTERM) gracefully."""
    global shutdown_requested
    shutdown_requested = True
    logger.info("Received %s, shutting down gracefully...", signal.Signals(signum).name)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


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

STATS_FILE = Path(__file__).parent / "logs" / "fib_stats.json"
STATE_FILE = Path(__file__).parent / "logs" / "fib_state.json"


@dataclass
class FibSignal:
    """Fibonacci swing signal"""
    signal_id: str
    symbol: str
    timeframe: str
    direction: str
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    quality: str  # PREMIUM, CONFIRMED, STANDARD

    swing_high: float
    swing_low: float
    fib_level: str  # 38.2%, 50.0%, 61.8%
    fib_price: float

    swing_confirmed: bool
    bars_since_swing: int
    uptrend: bool
    high_volume: bool

    created_at: str
    exchange: str = "binanceusdm"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "direction": self.direction,
            "entry": float(self.entry),
            "stop_loss": float(self.stop_loss),
            "tp1": float(self.tp1),
            "tp2": float(self.tp2),
            "tp3": float(self.tp3),
            "quality": self.quality,
            "swing_high": float(self.swing_high),
            "swing_low": float(self.swing_low),
            "fib_level": self.fib_level,
            "fib_price": float(self.fib_price),
            "swing_confirmed": bool(self.swing_confirmed),
            "bars_since_swing": int(self.bars_since_swing),
            "uptrend": bool(self.uptrend),
            "high_volume": bool(self.high_volume),
            "created_at": self.created_at,
            "exchange": self.exchange,
        }


class FibSwingDetector:
    """Detects swing points and calculates Fibonacci levels"""

    def __init__(self, lookback: int = 10):
        self.lookback = lookback

    @staticmethod
    def calculate_ema(prices: npt.NDArray[np.floating[Any]], period: int) -> npt.NDArray[np.floating[Any]]:
        """Calculate Exponential Moving Average (optimized with local variables)"""
        n = len(prices)
        ema = np.zeros(n)

        if n < period:
            return ema

        multiplier = 2.0 / (period + 1)
        decay = 1.0 - multiplier

        # Initialize with SMA
        ema[period - 1] = np.mean(prices[:period])

        # Pre-multiply prices for fewer operations in loop
        weighted_prices = prices[period:] * multiplier

        # Optimized loop with local variable (avoids array indexing overhead)
        prev_ema = ema[period - 1]
        for i in range(period, n):
            prev_ema = weighted_prices[i - period] + prev_ema * decay
            ema[i] = prev_ema

        return ema
    
    @staticmethod
    def calculate_atr(highs: npt.NDArray[np.floating[Any]], 
                      lows: npt.NDArray[np.floating[Any]], 
                      closes: npt.NDArray[np.floating[Any]], 
                      period: int = 14) -> float:
        """Calculate Average True Range (ATR)"""
        if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
            return 0.0
        
        # Calculate True Range
        high_low = highs[1:] - lows[1:]
        high_close = np.abs(highs[1:] - closes[:-1])
        low_close = np.abs(lows[1:] - closes[:-1])
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        
        # Calculate ATR as simple moving average of True Range
        if len(true_range) < period:
            return float(np.mean(true_range))
        
        atr = float(np.mean(true_range[-period:]))
        return atr

    def find_swing_high(self, highs: npt.NDArray[np.floating[Any]], index: int) -> bool:
        """Check if index is a swing high"""
        if index < self.lookback or index >= len(highs) - self.lookback:
            return False
        window = highs[index - self.lookback : index + self.lookback + 1]
        return bool(highs[index] == max(window))

    def find_swing_low(self, lows: npt.NDArray[np.floating[Any]], index: int) -> bool:
        """Check if index is a swing low"""
        if index < self.lookback or index >= len(lows) - self.lookback:
            return False
        window = lows[index - self.lookback : index + self.lookback + 1]
        return bool(lows[index] == min(window))

    def detect_swing_points(
        self, highs: npt.NDArray[np.floating[Any]], lows: npt.NDArray[np.floating[Any]]
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """Detect all swing highs and lows (optimized with vectorized rolling window)"""
        n = len(highs)
        window_size = 2 * self.lookback + 1

        if n < window_size:
            return [], []

        # Create rolling windows for highs and lows (uses module-level import)
        high_windows = sliding_window_view(highs, window_size)
        low_windows = sliding_window_view(lows, window_size)

        # Find max/min in each window (vectorized)
        window_max = np.max(high_windows, axis=1)
        window_min = np.min(low_windows, axis=1)

        # Center values (potential swing points)
        center_highs = highs[self.lookback : n - self.lookback]
        center_lows = lows[self.lookback : n - self.lookback]

        # Use np.isclose for robust floating-point comparison (fixes precision issues)
        is_swing_high = np.isclose(center_highs, window_max, rtol=1e-9)
        is_swing_low = np.isclose(center_lows, window_min, rtol=1e-9)

        # Get indices and values
        swing_high_indices = np.where(is_swing_high)[0] + self.lookback
        swing_low_indices = np.where(is_swing_low)[0] + self.lookback

        swing_highs = [(int(i), float(highs[i])) for i in swing_high_indices]
        swing_lows = [(int(i), float(lows[i])) for i in swing_low_indices]

        return swing_highs, swing_lows

    @staticmethod
    def check_swing_confirmation(
        lows: npt.NDArray[np.floating[Any]],
        swing_low_idx: int,
        swing_low_price: float,
        confirmation_candles: int = 5
    ) -> Tuple[bool, int]:
        """Check if swing low has been confirmed (held for X candles)"""
        if swing_low_idx + confirmation_candles >= len(lows):
            bars_since = len(lows) - 1 - swing_low_idx
            return False, bars_since

        bars_since = len(lows) - 1 - swing_low_idx

        # Check if all candles stayed above swing low
        for i in range(1, min(confirmation_candles + 1, bars_since + 1)):
            if swing_low_idx + i < len(lows):
                if lows[swing_low_idx + i] < swing_low_price:
                    return False, bars_since

        return bars_since >= confirmation_candles, bars_since

    @staticmethod
    def check_swing_high_confirmation(
        highs: npt.NDArray[np.floating[Any]],
        swing_high_idx: int,
        swing_high_price: float,
        confirmation_candles: int = 5
    ) -> Tuple[bool, int]:
        """Check if swing high has been confirmed (held for X candles)"""
        if swing_high_idx + confirmation_candles >= len(highs):
            bars_since = len(highs) - 1 - swing_high_idx
            return False, bars_since

        bars_since = len(highs) - 1 - swing_high_idx

        # Check if all candles stayed below swing high
        for i in range(1, min(confirmation_candles + 1, bars_since + 1)):
            if swing_high_idx + i < len(highs):
                if highs[swing_high_idx + i] > swing_high_price:
                    return False, bars_since

        return bars_since >= confirmation_candles, bars_since

    @staticmethod
    def calculate_fibonacci_levels(swing_high: float, swing_low: float, bullish: bool) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels for bullish or bearish swings"""
        diff = swing_high - swing_low
        levels: Dict[str, float] = {}
        ratios = {
            '0.0%': 0.0,
            '23.6%': 0.236,
            '38.2%': 0.382,
            '50.0%': 0.5,
            '61.8%': 0.618,
            '65.0%': 0.65,
            '78.6%': 0.786,
            '100%': 1.0,
        }
        for name, ratio in ratios.items():
            if bullish:
                price = swing_high - diff * ratio
            else:
                price = swing_low + diff * ratio
            levels[name] = price
        return levels


class FibSignalEvaluator:
    """Evaluates Fibonacci signals and determines entry conditions"""

    def __init__(self, confirmation_candles: int = 5):
        self.confirmation_candles = confirmation_candles
        self.detector = FibSwingDetector()

    def analyze(
        self,
        symbol: str,
        timeframe: str,
        ohlcv: List[Any]
    ) -> Optional[FibSignal]:
        """Analyze market data and generate Fibonacci signal if conditions met"""

        if len(ohlcv) < 100:
            return None

        # Extract and validate OHLCV data
        try:
            closes = np.array([x[4] for x in ohlcv], dtype=np.float64)
            highs = np.array([x[2] for x in ohlcv], dtype=np.float64)
            lows = np.array([x[3] for x in ohlcv], dtype=np.float64)
            volumes = np.array([x[5] for x in ohlcv], dtype=np.float64)

            # Validate no NaN or invalid values
            if np.any(np.isnan(closes)) or np.any(np.isnan(highs)) or np.any(np.isnan(lows)):
                logger.warning("%s: OHLCV contains NaN values, skipping", symbol)
                return None
            if np.any(closes <= 0) or np.any(highs <= 0) or np.any(lows <= 0):
                logger.warning("%s: OHLCV contains invalid (<=0) prices, skipping", symbol)
                return None
        except (IndexError, TypeError, ValueError) as e:
            logger.warning("%s: Failed to parse OHLCV data: %s", symbol, e)
            return None

        current_price = closes[-1]

        # Calculate EMAs
        ema_fast = self.detector.calculate_ema(closes, 20)
        ema_slow = self.detector.calculate_ema(closes, 50)

        uptrend = ema_fast[-1] > ema_slow[-1]

        # Volume analysis
        volume_ma = np.mean(volumes[-20:])
        high_volume = volumes[-1] > volume_ma

        # Detect swing points
        swing_highs, swing_lows = self.detector.detect_swing_points(highs, lows)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None

        signal_direction = None
        swing_high = swing_low = 0.0

        if uptrend:
            # Use previous swing low as anchor, recent swing high as reference
            swing_low_idx, swing_low_price = swing_lows[-2]
            swing_high_idx, swing_high_price = swing_highs[-1]
            if swing_high_idx <= swing_low_idx:
                return None
            swing_low = swing_low_price
            swing_high = swing_high_price
            signal_direction = "LONG"
        else:
            # Downtrend: previous swing high anchor, recent swing low reference
            swing_high_idx, swing_high_price = swing_highs[-2]
            swing_low_idx, swing_low_price = swing_lows[-1]
            if swing_low_idx <= swing_high_idx:
                return None
            swing_high = swing_high_price
            swing_low = swing_low_price
            signal_direction = "SHORT"

        fib_levels = self.detector.calculate_fibonacci_levels(swing_high, swing_low, bullish=uptrend)
        diff = abs(swing_high - swing_low)

        # Guard against division by zero / zero tolerance when swing_high == swing_low
        if diff < 1e-10:
            logger.debug("%s: swing_high equals swing_low, skipping", symbol)
            return None

        tolerance = diff * 0.02

        fib_names = {
            '38.2%': fib_levels['38.2%'],
            '50.0%': fib_levels['50.0%'],
            '61.8%': fib_levels['61.8%'],
            '65.0%': fib_levels['65.0%'],
        }

        fib_level_name = None
        fib_level_price = None
        for name in ['65.0%', '61.8%', '50.0%', '38.2%']:
            target = fib_names.get(name)
            if target is not None and abs(current_price - target) <= tolerance:
                fib_level_name = name
                fib_level_price = target
                break

        if fib_level_name is None:
            return None

        if uptrend:
            confirmation_idx, confirmation_price = swing_low_idx, swing_low
            is_confirmed, bars_since = self.detector.check_swing_confirmation(
                lows, confirmation_idx, confirmation_price, self.confirmation_candles
            )
            in_zone = current_price >= swing_low and current_price <= swing_high
        else:
            confirmation_idx, confirmation_price = swing_high_idx, swing_high
            is_confirmed, bars_since = self.detector.check_swing_high_confirmation(
                highs, confirmation_idx, confirmation_price, self.confirmation_candles
            )
            in_zone = current_price <= swing_high and current_price >= swing_low

        if not in_zone:
            return None

        if not uptrend and signal_direction == "LONG":
            return None
        if uptrend and signal_direction == "SHORT":
            return None

        quality = self._determine_quality(fib_level_name)
        if quality == "WEAK" and not high_volume:
            return None

        # Calculate trade setup using centralized TPSLCalculator
        direction = signal_direction
        config_mgr = get_config_manager()
        risk_config = config_mgr.get_effective_risk("fib_swing_bot", symbol)

        calculator = TPSLCalculator(min_risk_reward=risk_config.min_risk_reward, min_risk_reward_tp2=1.5)
        levels = calculator.calculate(
            entry=current_price,
            direction=direction,
            swing_high=swing_high if direction == "SHORT" else None,
            swing_low=swing_low if direction == "LONG" else None,
            method=CalculationMethod.STRUCTURE,
        )

        if not levels.is_valid:
            logger.debug("%s: TPSLCalculator rejected - %s", symbol, levels.rejection_reason)
            return None

        stop_loss = levels.stop_loss
        tp1 = levels.take_profit_1
        tp2 = levels.take_profit_2
        
        # CRITICAL FIX: Add maximum stop loss limit to prevent catastrophic losses
        MAX_SL_PERCENT = 2.5  # Maximum 2.5% stop loss
        risk_pct = abs(stop_loss - current_price) / current_price * 100
        if risk_pct > MAX_SL_PERCENT:
            logger.debug("%s: Stop loss too wide: %.2f%% (max %.2f%%), rejecting signal", 
                        symbol, risk_pct, MAX_SL_PERCENT)
            return None
        
        # CRITICAL FIX: Validate minimum R:R ratio
        MIN_RR_RATIO = 1.5  # Minimum 1.5:1 risk/reward
        rr_ratio = abs(tp1 - current_price) / abs(stop_loss - current_price) if abs(stop_loss - current_price) > 0 else 0
        if rr_ratio < MIN_RR_RATIO:
            logger.debug("%s: Risk/reward too low: 1:%.2f (min 1:%.2f), rejecting signal",
                        symbol, rr_ratio, MIN_RR_RATIO)
            return None
        # CRITICAL FIX: TP3 calculation using ATR for more realistic targets
        if levels.take_profit_3:
            tp3 = levels.take_profit_3
        else:
            # Calculate ATR for realistic TP3 extension
            atr = self.detector.calculate_atr(highs, lows, closes, period=14)
            if direction == "LONG":
                tp3 = tp2 + (atr * 2.0)  # Extend by 2x ATR beyond TP2
            else:
                tp3 = tp2 - (atr * 2.0)  # Extend by 2x ATR below TP2 for SHORT

        # Create signal (use timezone-aware datetime, not deprecated utcnow())
        now_utc = datetime.now(timezone.utc)
        signal_id = f"{symbol}-{timeframe}-{now_utc.isoformat()}"

        signal = FibSignal(
            signal_id=signal_id,
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            entry=current_price,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            quality=quality,
            swing_high=swing_high,
            swing_low=swing_low,
            fib_level=fib_level_name or "N/A",
            fib_price=fib_level_price or 0.0,
            swing_confirmed=is_confirmed,
            bars_since_swing=bars_since,
            uptrend=uptrend,
            high_volume=high_volume,
            created_at=now_utc.isoformat(),
        )

        return signal

    def _determine_quality(self, fib_level: Optional[str]) -> str:
        if fib_level in ("61.8%", "65.0%"):
            return "PREMIUM"
        if fib_level == "50.0%":
            return "CONFIRMED"
        if fib_level == "38.2%":
            return "WEAK"
        return "STANDARD"

    # NOTE: _fallback_levels was removed as dead code - it was never called.
    # TP/SL calculation is now handled by TPSLCalculator.


class SignalTracker:
    """Tracks open signals and monitors TP/SL hits"""

    def __init__(self, stats: Optional[SignalStats] = None, exchange: Optional[Any] = None, rate_limiter: Optional[Any] = None):
        self.state_file = STATE_FILE
        self.state_lock = Lock()
        self.state: Dict[str, Any] = self._load_state()
        self.stats = stats
        self.exchange = exchange
        self.rate_limiter = rate_limiter

    def _load_state(self) -> Dict[str, Any]:
        """Load state from file"""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                return {
                    "open_signals": data.get("open_signals", {}),
                    "last_alerts": data.get("last_alerts", {}),
                    "closed_signals": data.get("closed_signals", {}),
                }
            except json.JSONDecodeError:
                return {"open_signals": {}, "last_alerts": {}, "closed_signals": {}}
        return {"open_signals": {}, "last_alerts": {}, "closed_signals": {}}

    def _save_state(self) -> None:
        """Save state to file with file locking for concurrent access safety"""
        with self.state_lock:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file = self.state_file.with_suffix('.tmp')
            try:
                with open(temp_file, 'w') as f:
                    if HAS_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    json.dump(self.state, f, indent=2)
                    if HAS_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                temp_file.replace(self.state_file)
            except Exception as e:
                logger.error("Failed to save state: %s", e)
                if temp_file.exists():
                    temp_file.unlink()

    def can_alert(self, symbol: str, timeframe: str, cooldown_minutes: int) -> bool:
        """Check if enough time has passed since last alert"""
        key = f"{symbol}-{timeframe}"
        last_alerts = self.state.get("last_alerts", {})
        last_ts = last_alerts.get(key)

        if not last_ts:
            return True

        try:
            last_dt = datetime.fromisoformat(last_ts)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            else:
                last_dt = last_dt.astimezone(timezone.utc)
            current_time = datetime.now(timezone.utc)
            return current_time - last_dt >= timedelta(minutes=cooldown_minutes)
        except (ValueError, TypeError):
            return True

    def mark_alert(self, symbol: str, timeframe: str) -> None:
        """Mark that an alert was sent"""
        key = f"{symbol}-{timeframe}"
        self.state.setdefault("last_alerts", {})[key] = datetime.now(timezone.utc).isoformat()
        self._save_state()

    def cleanup_stale_signals(self, max_age_hours: int = 24) -> int:
        """Remove signals older than max_age_hours and move to closed_signals."""
        signals = self.state.setdefault("open_signals", {})
        closed = self.state.setdefault("closed_signals", {})

        current_time = datetime.now(timezone.utc)
        removed_count = 0
        signal_ids_to_remove = []

        for signal_id, signal_data in list(signals.items()):
            if not isinstance(signal_data, dict):
                signal_ids_to_remove.append(signal_id)
                continue

            created_at_str = signal_data.get("created_at")
            if not isinstance(created_at_str, str):
                signal_ids_to_remove.append(signal_id)
                continue

            try:
                created_time = datetime.fromisoformat(created_at_str)
                if created_time.tzinfo is None:
                    created_time = created_time.replace(tzinfo=timezone.utc)
                else:
                    created_time = created_time.astimezone(timezone.utc)

                age = current_time - created_time

                if age >= timedelta(hours=max_age_hours):
                    closed[signal_id] = {**signal_data, "closed_reason": "TIMEOUT", "closed_at": current_time.isoformat()}
                    signal_ids_to_remove.append(signal_id)
                    removed_count += 1
                    logger.info("Stale signal removed: %s (age: %.1f hours)", signal_id, age.total_seconds() / 3600)
            except (ValueError, TypeError) as exc:
                logger.warning("Invalid timestamp for signal %s, removing: %s", signal_id, exc)
                signal_ids_to_remove.append(signal_id)

        for signal_id in signal_ids_to_remove:
            if signal_id in signals:
                signals.pop(signal_id)

        # Prune closed_signals to prevent unbounded growth (keep max 100)
        max_closed_signals = 100
        if len(closed) > max_closed_signals:
            # Sort by closed_at and keep the most recent
            sorted_closed = sorted(
                closed.items(),
                key=lambda x: x[1].get("closed_at", "") if isinstance(x[1], dict) else "",
                reverse=True,
            )
            # Keep only the most recent entries
            self.state["closed_signals"] = dict(sorted_closed[:max_closed_signals])
            logger.info("Pruned closed_signals from %d to %d entries", len(closed), max_closed_signals)

        if removed_count > 0 or len(closed) > max_closed_signals:
            self._save_state()

        return removed_count

    def add_signal(self, signal: FibSignal) -> None:
        """Add signal to tracking"""
        signals = self.state.setdefault("open_signals", {})
        signals[signal.signal_id] = signal.as_dict()
        self._save_state()

        # Record in stats
        if self.stats:
            try:
                self.stats.record_open(
                    signal_id=signal.signal_id,
                    symbol=normalize_symbol(signal.symbol),
                    direction=signal.direction,
                    entry=signal.entry,
                    created_at=signal.created_at,
                    extra={
                        "timeframe": signal.timeframe,
                        "quality": signal.quality,
                        "fib_level": signal.fib_level,
                        "swing_confirmed": signal.swing_confirmed,
                    },
                )
            except Exception as exc:
                logger.warning("Failed to record stats for signal %s: %s", signal.signal_id, exc)

    def check_open_signals(self, notifier: Optional[TelegramNotifier]) -> None:
        """Check all open signals for TP/SL hits"""
        signals = self.state.get("open_signals", {})
        if not signals:
            return

        # Use stored exchange or create a fallback
        if self.exchange is None:
            self.exchange = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "swap"}})
        updated = False

        for signal_id, signal_data in list(signals.items()):
            if not isinstance(signal_data, dict):
                signals.pop(signal_id, None)
                updated = True
                continue
            symbol = signal_data.get("symbol")
            if not isinstance(symbol, str):
                signals.pop(signal_id, None)
                updated = True
                continue
            try:
                market_symbol = f"{symbol.replace("/USDT", "")}/USDT:USDT"
                if self.rate_limiter:
                    ticker = self.rate_limiter.execute(self.exchange.fetch_ticker, market_symbol)
                else:
                    ticker = self.exchange.fetch_ticker(market_symbol)
                current_price = ticker.get("last") if isinstance(ticker, dict) else None
            except Exception as exc:
                logger.warning("Ticker fetch failed for %s: %s", symbol, exc)
                continue
            if not isinstance(current_price, (int, float)):
                continue
            entry_raw = signal_data.get("entry")
            tp1_raw = signal_data.get("tp1")
            tp2_raw = signal_data.get("tp2")
            tp3_raw = signal_data.get("tp3")
            sl_raw = signal_data.get("stop_loss")
            if not all(isinstance(v, (int, float, str)) for v in (entry_raw, tp1_raw, tp2_raw, tp3_raw, sl_raw)):
                signals.pop(signal_id, None)
                updated = True
                continue
            try:
                entry = float(entry_raw)  # type: ignore[arg-type]
                tp1 = float(tp1_raw)      # type: ignore[arg-type]
                tp2 = float(tp2_raw)      # type: ignore[arg-type]
                tp3 = float(tp3_raw)      # type: ignore[arg-type]
                sl = float(sl_raw)        # type: ignore[arg-type]
            except (ValueError, TypeError) as exc:
                logger.warning("Invalid signal data for %s, removing: %s", signal_id, exc)
                signals.pop(signal_id, None)
                updated = True
                continue

            # Check TP/SL hits with price tolerance for slippage
            PRICE_TOLERANCE = 0.005  # 0.5% tolerance
            direction = signal_data.get("direction", "LONG")

            # Direction-aware TP/SL detection
            if direction == "LONG":
                # LONG: TP when price goes UP, SL when price goes DOWN
                # Allow hitting TP slightly below target (1 - tolerance)
                # Allow hitting SL slightly above target (1 + tolerance) for safety
                hit_tp3 = current_price >= (tp3 * (1 - PRICE_TOLERANCE))
                hit_tp2 = current_price >= (tp2 * (1 - PRICE_TOLERANCE)) and not hit_tp3
                hit_tp1 = current_price >= (tp1 * (1 - PRICE_TOLERANCE)) and not hit_tp2 and not hit_tp3
                hit_sl = current_price <= (sl * (1 + PRICE_TOLERANCE))
            else:
                # SHORT: TP when price goes DOWN, SL when price goes UP
                # BUG FIX: For SHORT, TPs are BELOW entry, so we want to trigger
                # when price drops to or slightly ABOVE the TP target (1 + tolerance)
                # SL is ABOVE entry, so trigger when price rises to or slightly BELOW SL (1 - tolerance)
                hit_tp3 = current_price <= (tp3 * (1 + PRICE_TOLERANCE))
                hit_tp2 = current_price <= (tp2 * (1 + PRICE_TOLERANCE)) and not hit_tp3
                hit_tp1 = current_price <= (tp1 * (1 + PRICE_TOLERANCE)) and not hit_tp2 and not hit_tp3
                hit_sl = current_price >= (sl * (1 - PRICE_TOLERANCE))

            result = None
            if hit_tp3:
                result = "TP3"
            elif hit_tp2:
                result = "TP2"
            elif hit_tp1:
                result = "TP1"
            elif hit_sl:
                result = "SL"

            if result:
                # Record close
                summary_message = None
                if self.stats:
                    stats_record = self.stats.record_close(
                        signal_id,
                        exit_price=current_price,
                        result=result,
                    )
                    if stats_record:
                        summary_message = self.stats.build_summary_message(stats_record)
                        logger.info("Trade closed: %s | %s | Entry: %.6f | Exit: %.6f | Result: %s | P&L: %.2f%%",
                                   signal_id, symbol, entry, current_price, result, stats_record.pnl_pct)
                    else:
                        self.stats.discard(signal_id)

                # Send message
                if summary_message:
                    message = summary_message
                else:
                    # Convert LONG/SHORT to BULLISH/BEARISH for message template
                    msg_direction = "BULLISH" if direction == "LONG" else "BEARISH"
                    message = format_result_message(
                        symbol=symbol,
                        direction=msg_direction,
                        result=result,
                        entry=entry,
                        exit_price=current_price,
                        stop_loss=sl,
                        tp1=tp1,
                        tp2=tp2,
                        signal_id=signal_id,
                    )

                if notifier and ENABLE_RESULT_NOTIFICATIONS:
                    notifier.send_message(message)
                elif not ENABLE_RESULT_NOTIFICATIONS:
                    logger.info("Result notification skipped (disabled): %s %s", signal_id, result)

                logger.info("Signal %s closed with %s", signal_id, result)

                # Archive to closed_signals before removing
                closed = self.state.setdefault("closed_signals", {})
                closed_signal = signal_data.copy()
                closed_signal["closed_at"] = datetime.now(timezone.utc).isoformat()
                closed_signal["exit_price"] = current_price
                closed_signal["result"] = result
                pnl = ((current_price - entry) / entry) * 100 if direction == "LONG" else ((entry - current_price) / entry) * 100
                closed_signal["pnl_percent"] = pnl
                closed[signal_id] = closed_signal

                signals.pop(signal_id)
                updated = True

        if updated:
            self._save_state()


class FibSwingBot(BotSignalMixin):
    """Main Fibonacci Swing Bot with unified signal management."""

    def __init__(self, watchlist_file: Path, interval: int = 60):
        self.watchlist_file = watchlist_file
        self.interval = interval
        self.watchlist: List[WatchItem] = self._load_watchlist()
        self.notifier = self._build_notifier()
        self.stats = SignalStats("Fib Swing Bot", STATS_FILE)
        self.evaluator = FibSignalEvaluator(confirmation_candles=5)
        self.health_monitor = self._build_health_monitor()
        self.rate_limiter = RateLimiter(calls_per_minute=30) if RateLimiter else None
        self.exchange = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "swap"}})
        self.rate_limiter_handler = RateLimitHandler(base_delay=0.5, max_retries=5) if RateLimitHandler else None
        # Pass exchange and rate limiter to tracker for monitoring
        self.tracker = SignalTracker(self.stats, self.exchange, self.rate_limiter_handler)

        # NEW: Initialize unified signal adapter
        self._init_signal_adapter(
            bot_name="fib_swing_bot",
            notifier=self.notifier,
            exchange="Binance",
            default_timeframe="1m",
            notification_mode="signal_only",
        )

        logger.info("Fib Swing Bot initialized with %d symbols", len(self.watchlist))

    def _load_watchlist(self) -> List[WatchItem]:
        """Load watchlist from JSON file"""
        if not self.watchlist_file.exists():
            logger.error("Watchlist file not found: %s", self.watchlist_file)
            return []

        try:
            data = json.loads(self.watchlist_file.read_text())
        except json.JSONDecodeError as e:
            logger.error("Failed to parse watchlist: %s", e)
            return []
        result: List[WatchItem] = []
        for row in data if isinstance(data, list) else []:
            if not isinstance(row, dict):
                continue
            symbol_val = row.get("symbol")
            if not isinstance(symbol_val, str):
                continue
            timeframe_val = row.get("timeframe", "5m")
            timeframe = timeframe_val if isinstance(timeframe_val, str) else "5m"
            cooldown_raw = row.get("cooldown_minutes", 30)
            try:
                cooldown = int(cooldown_raw)
            except (ValueError, TypeError):
                logger.debug("Invalid cooldown value '%s' for %s, using default 30", cooldown_raw, symbol_val)
                cooldown = 30
            result.append({"symbol": symbol_val.upper(), "timeframe": timeframe, "cooldown_minutes": cooldown})
        return result

    def _build_notifier(self) -> Optional[TelegramNotifier]:
        """Build Telegram notifier"""
        from dotenv import load_dotenv
        load_dotenv()

        token = os.getenv("TELEGRAM_BOT_TOKEN_FIB")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if token and chat_id:
            return TelegramNotifier(token, chat_id)

        logger.warning("Telegram credentials not found for Fib bot")
        return None

    def _build_health_monitor(self) -> Optional[Any]:
        """Build health monitor"""
        if self.notifier and HealthMonitor:
            return HealthMonitor(
                bot_name="Fib Swing Bot",
                notifier=self.notifier,
                heartbeat_interval=3600,  # 1 hour in seconds
            )
        return None

    def _format_signal_message(self, signal: FibSignal) -> str:
        """Format Fibonacci signal for Telegram using centralized template."""
        # Get performance stats (with error handling for corrupted stats)
        # Get performance stats (ALWAYS included)
        tp1_count = 0
        tp2_count = 0
        sl_count = 0
        if self.stats is not None:
            try:
                symbol_key = normalize_symbol(signal.symbol)
                counts = self.stats.symbol_tp_sl_counts(symbol_key)
                tp1_count = counts.get("TP1", 0)
                tp2_count = counts.get("TP2", 0)
                sl_count = counts.get("SL", 0)
            except Exception as e:
                logger.warning("Failed to get performance stats for %s: %s", signal.symbol, e)
        total = tp1_count + tp2_count + sl_count
        perf_stats = {
            "tp1": tp1_count,
            "tp2": tp2_count,
            "sl": sl_count,
            "wins": tp1_count + tp2_count,
            "total": total,
        }

        # Build extra info with Fibonacci-specific data
        fib_info = f"Fib: {signal.fib_level}" if signal.fib_level != "N/A" else ""
        swing_info = f"Swing: ${signal.swing_low:.4f}-${signal.swing_high:.4f}"
        quality_info = f"Quality: {signal.quality}"
        extra_info = " | ".join(filter(None, [fib_info, swing_info, quality_info]))

        return format_signal_message(
            bot_name="FIB SWING",
            symbol=normalize_symbol(signal.symbol),
            direction=signal.direction,
            entry=signal.entry,
            stop_loss=signal.stop_loss,
            tp1=signal.tp1,
            tp2=signal.tp2,
            tp3=signal.tp3,
            pattern_name=f"{signal.quality} Fibonacci Entry",
            exchange="binanceusdm",
            timeframe=signal.timeframe,
            current_price=signal.entry,
            performance_stats=perf_stats,
            extra_info=extra_info,
        )

    def run(self, run_once: bool = False) -> None:
        """Main bot loop"""
        logger.info("Starting Fib Swing Bot for %d symbols", len(self.watchlist))

        # Send startup message
        if self.health_monitor:
            self.health_monitor.send_startup_message()

        try:
            while not shutdown_requested:
                try:
                    # Check open signals
                    self.tracker.check_open_signals(self.notifier)

                    # Cleanup stale signals every cycle
                    stale_count = self.tracker.cleanup_stale_signals(max_age_hours=24)
                    if stale_count > 0:
                        logger.info("Cleaned up %d stale signals", stale_count)

                    # Scan watchlist
                    for item in self.watchlist:
                        # Check for shutdown between symbols for faster response
                        if shutdown_requested:
                            logger.info("Shutdown requested during watchlist scan")
                            break

                        symbol_val = item.get("symbol") if isinstance(item, dict) else None
                        if not isinstance(symbol_val, str):
                            continue
                        symbol = symbol_val
                        timeframe_val = item.get("timeframe", "5m") if isinstance(item, dict) else "5m"
                        timeframe = timeframe_val if isinstance(timeframe_val, str) else "5m"
                        cooldown_raw = item.get("cooldown_minutes", 30) if isinstance(item, dict) else 30
                        try:
                            cooldown = int(cooldown_raw)
                        except (ValueError, TypeError):
                            logger.debug("Invalid cooldown value '%s' for %s, using default 30", cooldown_raw, symbol)
                            cooldown = 30

                        # Check cooldown
                        if not self.tracker.can_alert(symbol, timeframe, cooldown):
                            logger.debug("Cooldown active for %s %s", symbol, timeframe)
                            continue

                        # Rate limiting
                        if self.rate_limiter:
                            self.rate_limiter.wait_if_needed()

                        # Fetch data
                        try:
                            if self.rate_limiter_handler:
                                ohlcv = self.rate_limiter_handler.execute(
                                    self.exchange.fetch_ohlcv,
                                    f"{symbol.replace("/USDT", "")}/USDT:USDT",
                                    timeframe=timeframe,
                                    limit=200
                                )
                            else:
                                ohlcv = self.exchange.fetch_ohlcv(
                                    f"{symbol.replace("/USDT", "")}/USDT:USDT",
                                    timeframe,
                                    limit=200
                                )
                        except Exception as exc:
                            logger.warning("Failed to fetch %s %s: %s", symbol, timeframe, exc)
                            continue

                        # Analyze
                        signal = self.evaluator.analyze(symbol, timeframe, ohlcv)

                        if signal:
                            # MAX OPEN SIGNALS LIMIT
                            MAX_OPEN_SIGNALS = 45
                            current_open = len(self.tracker.state.get("open_signals", {}))
                            if current_open >= MAX_OPEN_SIGNALS:
                                logger.info(
                                    "Max open signals limit reached (%d/%d). Skipping %s",
                                    current_open, MAX_OPEN_SIGNALS, symbol
                                )
                                continue
                            
                            # Check for duplicates using unified adapter
                            if self._has_open_signal(signal.symbol):
                                logger.debug("Already have open signal for %s - skipping", signal.symbol)
                                continue

                            # NEW: Send via unified signal adapter
                            created = self._send_signal(
                                symbol=signal.symbol,
                                direction=signal.direction,
                                entry=signal.entry,
                                stop_loss=signal.stop_loss,
                                take_profit_1=signal.take_profit_1,
                                take_profit_2=signal.take_profit_2,
                                take_profit_3=getattr(signal, 'take_profit_3', 0),
                                pattern_name=f"Fib {signal.fib_level}",
                                timeframe=timeframe,
                                confidence=signal.confidence * 100 if hasattr(signal, 'confidence') else 70,
                                reasons=[f"Quality: {signal.quality}", f"Fib {signal.fib_level}"],
                            )

                            logger.info(
                                "%s %s signal: %s at %.6f (Fib %s)",
                                signal.symbol, signal.quality, signal.direction,
                                signal.entry, signal.fib_level
                            )

                            # Track signal in legacy tracker for compatibility
                            if created:
                                self.tracker.add_signal(signal)
                                self.tracker.mark_alert(symbol, timeframe)

                        time.sleep(1)  # Small delay between symbols

                    # Record successful cycle
                    if self.health_monitor:
                        self.health_monitor.record_cycle()

                    if run_once:
                        break

                    logger.info("Cycle complete; sleeping %d seconds", self.interval)
                    # Sleep in 1-second chunks to respond quickly to shutdown signals
                    for _ in range(self.interval):
                        if shutdown_requested:
                            logger.info("Shutdown requested during sleep")
                            break
                        time.sleep(1)

                except Exception as exc:
                    logger.error("Error in cycle: %s", exc, exc_info=True)
                    if self.health_monitor:
                        self.health_monitor.record_error(str(exc))
                    if run_once:
                        raise
                    time.sleep(10)

        finally:
            # Send shutdown message
            if self.health_monitor:
                self.health_monitor.send_shutdown_message()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fibonacci Swing Bot")
    parser.add_argument("--once", action="store_true", help="Run only one cycle")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    args = parser.parse_args()

    watchlist_file = Path(__file__).parent / "fib_watchlist.json"
    bot = FibSwingBot(watchlist_file, interval=args.interval)
    bot.run(run_once=args.once)


if __name__ == "__main__":
    main()
