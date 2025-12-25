#!/usr/bin/env python3
"""
Harmonic Pattern Bot - Non-Repainting Implementation

This bot identifies harmonic patterns (Gartley, Bat, Butterfly, Crab, Shark, ABCD)
using CONFIRMED swing points only (no repainting).

Key improvements over TradingView version:
1. Non-repainting ZigZag using confirmed swings
2. Proper R:R management (minimum 1.5:1)
3. ATR-based stop loss for better risk control
4. Pattern quality scoring
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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import FrameType
from typing import Any, Dict, List, Optional, Tuple

import ccxt
import numpy as np
from dotenv import load_dotenv

# Graceful shutdown handling
shutdown_requested = False

# Price tolerance for TP/SL detection (0.5% = 0.005)
# This accounts for spread, slippage, and minor price variations
PRICE_TOLERANCE = 0.005

# Result notifications disabled - history included in next signal instead
ENABLE_RESULT_NOTIFICATIONS = False


def signal_handler(signum: int, frame: Optional[FrameType]) -> None:
    """Handle shutdown signals (SIGINT, SIGTERM) gracefully."""
    global shutdown_requested
    shutdown_requested = True
    logger.info("Received %s, shutting down gracefully...", signal.Signals(signum).name)

# =========================================================
# PATHS / LOGGING
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
LOG_DIR = ROOT_DIR / "logs"
WATCHLIST_FILE = BASE_DIR / "harmonic_watchlist.json"
STATE_FILE = BASE_DIR / "harmonic_state.json"
STATS_FILE = LOG_DIR / "harmonic_pattern_stats.json"
CONFIG_FILE = BASE_DIR / "harmonic_config.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "harmonic_pattern_bot.log"),
    ],
)
logger = logging.getLogger("harmonic_pattern_bot")

load_dotenv(ROOT_DIR / ".env")
sys.path.insert(0, str(ROOT_DIR))

from message_templates import format_result_message
from notifier import TelegramNotifier
from signal_stats import SignalStats

# =========================================================
# CONFIGURATION
# =========================================================
@dataclass
class HarmonicConfig:
    """Configuration for harmonic pattern detection."""
    # ZigZag settings
    zigzag_depth: int = 12  # Minimum bars between swings
    zigzag_deviation: float = 5.0  # Minimum % deviation for new swing

    # Pattern tolerances (how much deviation from ideal ratios)
    fib_tolerance: float = 0.05  # 5% tolerance on fib ratios

    # Risk management
    min_risk_reward: float = 1.5  # Minimum R:R ratio
    atr_period: int = 14
    atr_sl_multiplier: float = 1.5  # SL = ATR * multiplier
    tp1_rr: float = 1.5  # TP1 at 1.5R
    tp2_rr: float = 2.5  # TP2 at 2.5R
    max_stop_pct: float = 3.0  # Maximum stop loss %

    # Signal management
    cooldown_minutes: int = 60
    min_pattern_score: float = 70.0  # Minimum pattern quality score (0-100)


def load_config() -> HarmonicConfig:
    """Load configuration from file or use defaults."""
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text())
            return HarmonicConfig(**data.get("harmonic", {}))
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
    return HarmonicConfig()


# =========================================================
# SWING POINT DETECTION (Non-Repainting)
# =========================================================
@dataclass
class SwingPoint:
    """Represents a confirmed swing high or low."""
    index: int
    price: float
    timestamp: datetime
    swing_type: str  # "high" or "low"


def find_swing_points(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    timestamps: List[datetime],
    depth: int = 12,
    deviation_pct: float = 5.0
) -> List[SwingPoint]:
    """
    Find confirmed swing points (non-repainting).

    A swing is confirmed only after 'depth' bars have passed,
    ensuring the point won't change with new data.
    """
    swings = []
    n = len(highs)

    if n < depth * 2:
        return swings

    # Find swing highs
    for i in range(depth, n - depth):
        is_swing_high = True
        for j in range(i - depth, i + depth + 1):
            if j != i and highs[j] >= highs[i]:
                is_swing_high = False
                break

        if is_swing_high:
            # Check minimum deviation from neighbors
            left_low = min(lows[i-depth:i])
            right_low = min(lows[i+1:i+depth+1])
            deviation = (highs[i] - min(left_low, right_low)) / highs[i] * 100

            if deviation >= deviation_pct:
                swings.append(SwingPoint(
                    index=i,
                    price=highs[i],
                    timestamp=timestamps[i],
                    swing_type="high"
                ))

    # Find swing lows
    for i in range(depth, n - depth):
        is_swing_low = True
        for j in range(i - depth, i + depth + 1):
            if j != i and lows[j] <= lows[i]:
                is_swing_low = False
                break

        if is_swing_low:
            # Check minimum deviation from neighbors
            left_high = max(highs[i-depth:i])
            right_high = max(highs[i+1:i+depth+1])
            deviation = (max(left_high, right_high) - lows[i]) / lows[i] * 100

            if deviation >= deviation_pct:
                swings.append(SwingPoint(
                    index=i,
                    price=lows[i],
                    timestamp=timestamps[i],
                    swing_type="low"
                ))

    # Sort by index and filter alternating swings
    swings.sort(key=lambda x: x.index)

    # Keep only alternating high-low sequence
    filtered = []
    for swing in swings:
        if not filtered:
            filtered.append(swing)
        elif swing.swing_type != filtered[-1].swing_type:
            filtered.append(swing)
        elif swing.swing_type == "high" and swing.price > filtered[-1].price:
            filtered[-1] = swing
        elif swing.swing_type == "low" and swing.price < filtered[-1].price:
            filtered[-1] = swing

    return filtered


# =========================================================
# HARMONIC PATTERN DETECTION
# =========================================================
@dataclass
class HarmonicPattern:
    """Represents a detected harmonic pattern."""
    name: str
    direction: str  # "bullish" or "bearish"
    points: Dict[str, SwingPoint]  # X, A, B, C, D
    ratios: Dict[str, float]  # XAB, ABC, BCD, XAD
    score: float  # Pattern quality 0-100
    entry_zone: Tuple[float, float]  # (lower, upper)
    stop_loss: float
    take_profit_1: float
    take_profit_2: float


# Ideal Fibonacci ratios for each pattern
PATTERN_RATIOS = {
    "gartley": {
        "xab": (0.618, 0.618),
        "abc": (0.382, 0.886),
        "bcd": (1.272, 1.618),
        "xad": (0.786, 0.786),
    },
    "bat": {
        "xab": (0.382, 0.500),
        "abc": (0.382, 0.886),
        "bcd": (1.618, 2.618),
        "xad": (0.886, 0.886),
    },
    "butterfly": {
        "xab": (0.786, 0.786),
        "abc": (0.382, 0.886),
        "bcd": (1.618, 2.618),
        "xad": (1.272, 1.618),
    },
    "crab": {
        "xab": (0.382, 0.618),
        "abc": (0.382, 0.886),
        "bcd": (2.240, 3.618),
        "xad": (1.618, 1.618),
    },
    "shark": {
        "xab": (0.446, 0.618),
        "abc": (1.130, 1.618),
        "bcd": (1.618, 2.240),
        "xad": (0.886, 1.130),
    },
    "abcd": {
        "abc": (0.382, 0.886),
        "bcd": (1.130, 2.618),
    },
}


def calculate_ratio(p1: float, p2: float, p3: float) -> float:
    """Calculate Fibonacci ratio between three points."""
    # Use tolerance instead of exact equality to prevent division by near-zero
    MIN_DIFF = 1e-10
    diff = abs(p1 - p2)
    if diff < MIN_DIFF:
        return 0
    return abs(p2 - p3) / diff


def score_ratio(actual: float, ideal_range: Tuple[float, float], tolerance: float = 0.05) -> float:
    """
    Score how close a ratio is to the ideal range.
    Returns 0-100 score.
    """
    ideal_min, ideal_max = ideal_range
    ideal_mid = (ideal_min + ideal_max) / 2

    # Perfect score if within range
    if ideal_min <= actual <= ideal_max:
        # Higher score for closer to midpoint
        distance = abs(actual - ideal_mid) / (ideal_max - ideal_min + 0.001)
        return 100 - (distance * 20)  # 80-100 range

    # Reduced score if within tolerance
    extended_min = ideal_min * (1 - tolerance)
    extended_max = ideal_max * (1 + tolerance)

    if extended_min <= actual <= extended_max:
        if actual < ideal_min:
            distance = (ideal_min - actual) / (ideal_min * tolerance)
        else:
            distance = (actual - ideal_max) / (ideal_max * tolerance)
        return 80 - (distance * 30)  # 50-80 range

    return 0  # Outside tolerance


def detect_pattern(
    swings: List[SwingPoint],
    current_price: float,
    config: HarmonicConfig
) -> Optional[HarmonicPattern]:
    """
    Detect harmonic patterns from the last 5 swing points.
    Returns the best matching pattern if found.
    """
    if len(swings) < 5:
        return None

    # Get last 5 swings (X, A, B, C, D)
    x, a, b, c, d = swings[-5:]

    # Determine pattern direction
    if d.swing_type == "low":
        direction = "bullish"
        # For bullish: X=high, A=low, B=high, C=low, D=low
        if not (x.swing_type == "high" and a.swing_type == "low" and
                b.swing_type == "high" and c.swing_type == "low"):
            return None
    else:
        direction = "bearish"
        # For bearish: X=low, A=high, B=low, C=high, D=high
        if not (x.swing_type == "low" and a.swing_type == "high" and
                b.swing_type == "low" and c.swing_type == "high"):
            return None

    # Calculate ratios
    xab = calculate_ratio(x.price, a.price, b.price)
    abc = calculate_ratio(a.price, b.price, c.price)
    bcd = calculate_ratio(b.price, c.price, d.price)
    xad = calculate_ratio(x.price, a.price, d.price)

    ratios = {"xab": xab, "abc": abc, "bcd": bcd, "xad": xad}

    # Score each pattern
    best_pattern = None
    best_score = 0

    for pattern_name, ideal_ratios in PATTERN_RATIOS.items():
        scores = []

        for ratio_name, ideal_range in ideal_ratios.items():
            if ratio_name in ratios:
                score = score_ratio(ratios[ratio_name], ideal_range, config.fib_tolerance)
                scores.append(score)

        if scores and all(s > 0 for s in scores):
            avg_score = sum(scores) / len(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_pattern = pattern_name

    if best_pattern is None or best_score < config.min_pattern_score:
        return None

    # Calculate entry zone, SL, and TPs
    cd_range = abs(c.price - d.price)

    if direction == "bullish":
        # Entry zone: D to 0.382 retracement of CD
        entry_lower = d.price
        entry_upper = d.price + (cd_range * 0.382)

        # Stop loss below D
        stop_loss = d.price - (cd_range * 0.236)

        # Take profits based on CD retracement
        take_profit_1 = d.price + (cd_range * 0.618)
        take_profit_2 = d.price + (cd_range * 1.0)
    else:
        # Bearish
        entry_lower = d.price - (cd_range * 0.382)
        entry_upper = d.price

        stop_loss = d.price + (cd_range * 0.236)

        take_profit_1 = d.price - (cd_range * 0.618)
        take_profit_2 = d.price - (cd_range * 1.0)

    # Calculate R:R and adjust if needed
    entry_mid = (entry_lower + entry_upper) / 2
    risk = abs(entry_mid - stop_loss)
    reward = abs(take_profit_1 - entry_mid)

    if risk > 0:
        rr_ratio = reward / risk

        # If R:R is below minimum, adjust TPs
        if rr_ratio < config.min_risk_reward:
            # Extend TP1 to meet minimum R:R
            if direction == "bullish":
                take_profit_1 = entry_mid + (risk * config.tp1_rr)
                take_profit_2 = entry_mid + (risk * config.tp2_rr)
            else:
                take_profit_1 = entry_mid - (risk * config.tp1_rr)
                take_profit_2 = entry_mid - (risk * config.tp2_rr)

    return HarmonicPattern(
        name=best_pattern,
        direction=direction,
        points={"X": x, "A": a, "B": b, "C": c, "D": d},
        ratios=ratios,
        score=best_score,
        entry_zone=(entry_lower, entry_upper),
        stop_loss=stop_loss,
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
    )


# =========================================================
# ATR CALCULATION
# =========================================================
def calculate_atr(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = 14
) -> float:
    """Calculate Average True Range."""
    if len(closes) < period + 1:
        return 0.0

    true_ranges = []
    for i in range(1, len(closes)):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i - 1])
        low_close = abs(lows[i] - closes[i - 1])
        true_ranges.append(max(high_low, high_close, low_close))

    if len(true_ranges) < period:
        return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0

    atr = sum(true_ranges[:period]) / period
    for i in range(period, len(true_ranges)):
        atr = (atr * (period - 1) + true_ranges[i]) / period

    return atr


# =========================================================
# STATE MANAGEMENT
# =========================================================
class StateManager:
    """Thread-safe state management for signal tracking."""

    def __init__(self) -> None:
        self.state_lock = threading.Lock()
        self.state = self._load_state()

    def _empty_state(self) -> Dict[str, Any]:
        return {"open_signals": {}, "closed_signals": {}, "last_alerts": {}}

    def _load_state(self) -> Dict[str, Any]:
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, 'r') as f:
                    data = json.load(f)
                    for key in self._empty_state():
                        if key not in data:
                            data[key] = {}
                    return data
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
                    json.dump(self.state, f, indent=2, default=str)
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

    def should_alert(self, symbol: str, cooldown_mins: int = 60) -> bool:
        last_ts = self.state.get("last_alerts", {}).get(symbol)
        if not last_ts:
            return True
        try:
            last = datetime.fromisoformat(last_ts)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - last) >= timedelta(minutes=cooldown_mins)
        except Exception:
            return True

    def add_signal(self, sig_id: str, signal_data: Dict[str, Any], symbol: str) -> None:
        signals = self.state.setdefault("open_signals", {})
        signal_data["created_at"] = datetime.now(timezone.utc).isoformat()
        signals[sig_id] = signal_data

        last_alerts = self.state.setdefault("last_alerts", {})
        last_alerts[symbol] = datetime.now(timezone.utc).isoformat()

        self._save_state()
        logger.info(f"Signal added: {sig_id}")

    def close_signal(self, sig_id: str, exit_price: float, result: str) -> None:
        signals = self.state.get("open_signals", {})
        if sig_id not in signals:
            return

        closed = self.state.setdefault("closed_signals", {})
        closed_signal = signals[sig_id].copy()
        closed_signal["closed_at"] = datetime.now(timezone.utc).isoformat()
        closed_signal["exit_price"] = exit_price
        closed_signal["result"] = result

        entry = closed_signal.get("entry", 0)
        direction = closed_signal.get("direction", "bullish")
        if entry > 0:
            if direction == "bullish":
                pnl = (exit_price - entry) / entry * 100
            else:
                pnl = (entry - exit_price) / entry * 100
            closed_signal["pnl_percent"] = pnl

        closed[sig_id] = closed_signal
        del signals[sig_id]
        self._save_state()
        logger.info(f"Signal closed: {sig_id} -> {result}")

    def get_open_signals(self) -> Dict[str, Any]:
        return self.state.get("open_signals", {})

    def cleanup_stale_signals(self, max_age_hours: int = 24) -> int:
        """Remove signals older than max_age_hours and cleanup closed_signals."""
        signals = self.state.get("open_signals", {})
        now = datetime.now(timezone.utc)
        stale_ids = []

        for sig_id, payload in signals.items():
            if not isinstance(payload, dict):
                stale_ids.append(sig_id)
                continue

            created_at = payload.get("created_at")
            if not isinstance(created_at, str):
                stale_ids.append(sig_id)
                continue

            try:
                created_dt = datetime.fromisoformat(created_at)
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


# =========================================================
# BOT CLASS
# =========================================================
class HarmonicPatternBot:
    """Main bot class for harmonic pattern detection."""

    def __init__(self):
        self.config = load_config()
        self.watchlist = self._load_watchlist()
        self.client = ccxt.binanceusdm({"enableRateLimit": True})
        self.notifier = TelegramNotifier(
            os.getenv("TELEGRAM_BOT_TOKEN_HARMONIC"),
            os.getenv("TELEGRAM_CHAT_ID"),
            str(LOG_DIR / "harmonic_signals.json"),
        )
        self.state = StateManager()
        self.stats = SignalStats("Harmonic Pattern Bot", STATS_FILE)

    def _load_watchlist(self) -> List[Dict]:
        if WATCHLIST_FILE.exists():
            return json.loads(WATCHLIST_FILE.read_text())
        # Default watchlist
        return [
            {"symbol": "BTC/USDT", "timeframe": "1h"},
            {"symbol": "ETH/USDT", "timeframe": "1h"},
        ]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[List]:
        """Fetch OHLCV data from exchange."""
        try:
            full_symbol = f"{symbol.replace('/USDT', '')}/USDT:USDT"
            return self.client.fetch_ohlcv(full_symbol, timeframe, limit=limit)
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None

    def analyze_symbol(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Analyze a symbol for harmonic patterns."""
        ohlcv = self.fetch_ohlcv(symbol, timeframe)
        if not ohlcv or len(ohlcv) < 100:
            return None

        # Parse OHLCV data
        timestamps = [datetime.fromtimestamp(x[0] / 1000, tz=timezone.utc) for x in ohlcv]
        highs = [float(x[2]) for x in ohlcv]
        lows = [float(x[3]) for x in ohlcv]
        closes = [float(x[4]) for x in ohlcv]

        # Validate OHLCV values are not NaN/inf and are positive
        highs_arr = np.array(highs, dtype=np.float64)
        lows_arr = np.array(lows, dtype=np.float64)
        closes_arr = np.array(closes, dtype=np.float64)

        if np.any(np.isnan(highs_arr)) or np.any(np.isnan(lows_arr)) or np.any(np.isnan(closes_arr)):
            logger.warning(f"{symbol}: OHLCV contains NaN values, skipping")
            return None

        if np.any(np.isinf(highs_arr)) or np.any(np.isinf(lows_arr)) or np.any(np.isinf(closes_arr)):
            logger.warning(f"{symbol}: OHLCV contains infinite values, skipping")
            return None

        if np.any(closes_arr <= 0) or np.any(highs_arr <= 0) or np.any(lows_arr <= 0):
            logger.warning(f"{symbol}: OHLCV contains zero or negative prices, skipping")
            return None

        # Validate OHLCV logic: high >= low
        if np.any(highs_arr < lows_arr):
            logger.warning(f"{symbol}: OHLCV has high < low, skipping")
            return None

        current_price = closes[-1]

        # Find swing points
        swings = find_swing_points(
            highs, lows, closes, timestamps,
            depth=self.config.zigzag_depth,
            deviation_pct=self.config.zigzag_deviation
        )

        if len(swings) < 5:
            logger.debug(f"{symbol} | Not enough swings: {len(swings)}")
            return None

        # Detect pattern
        pattern = detect_pattern(swings, current_price, self.config)

        if not pattern:
            logger.debug(f"{symbol} | No pattern detected")
            return None

        # Check if price is in entry zone
        entry_lower, entry_upper = pattern.entry_zone
        if not (entry_lower <= current_price <= entry_upper):
            logger.debug(f"{symbol} | Price {current_price:.4f} outside entry zone [{entry_lower:.4f}, {entry_upper:.4f}]")
            return None

        # Calculate ATR for stop adjustment
        atr = calculate_atr(highs, lows, closes, self.config.atr_period)

        # Adjust stop loss with ATR if it exceeds max %
        entry = current_price
        sl = pattern.stop_loss
        risk_pct = abs(entry - sl) / entry * 100

        if risk_pct > self.config.max_stop_pct:
            if pattern.direction == "bullish":
                sl = entry * (1 - self.config.max_stop_pct / 100)
            else:
                sl = entry * (1 + self.config.max_stop_pct / 100)

        # Recalculate TPs based on actual risk
        risk = abs(entry - sl)
        if pattern.direction == "bullish":
            tp1 = entry + (risk * self.config.tp1_rr)
            tp2 = entry + (risk * self.config.tp2_rr)
        else:
            tp1 = entry - (risk * self.config.tp1_rr)
            tp2 = entry - (risk * self.config.tp2_rr)

        # Final R:R check
        reward = abs(tp1 - entry)
        if risk > 0 and reward / risk < self.config.min_risk_reward:
            logger.debug(f"{symbol} | R:R {reward/risk:.2f} below minimum {self.config.min_risk_reward}")
            return None

        logger.info(f"{symbol} | {pattern.name.upper()} {pattern.direction} | Score: {pattern.score:.1f}")

        return {
            "pattern": pattern.name,
            "direction": pattern.direction,
            "score": pattern.score,
            "entry": entry,
            "stop_loss": sl,
            "take_profit_1": tp1,
            "take_profit_2": tp2,
            "ratios": pattern.ratios,
            "points": {k: {"price": v.price, "type": v.swing_type} for k, v in pattern.points.items()},
        }

    def check_open_signals(self) -> None:
        """Check open signals for TP/SL hits."""
        signals = self.state.get_open_signals()
        if not signals:
            return

        for sig_id, payload in list(signals.items()):
            symbol = payload.get("symbol")
            direction = payload.get("direction")

            try:
                full_symbol = f"{symbol.replace('/USDT', '')}/USDT:USDT"
                ticker = self.client.fetch_ticker(full_symbol)
                price = ticker.get("last")
            except Exception as e:
                logger.debug(f"Error fetching {symbol}: {e}")
                continue

            if not price:
                continue

            tp1, tp2 = payload.get("take_profit_1"), payload.get("take_profit_2")
            sl = payload.get("stop_loss")

            # Use module-level PRICE_TOLERANCE for consistency
            result = None
            if direction == "bullish":
                # With tolerance for slippage (allow slightly lower prices for TPs)
                if price >= (tp2 * (1 - PRICE_TOLERANCE)):
                    result = "TP2"
                elif price >= (tp1 * (1 - PRICE_TOLERANCE)):
                    result = "TP1"
                elif price <= (sl * (1 + PRICE_TOLERANCE)):
                    result = "SL"
            else:  # bearish
                # Allow slightly higher prices for TPs (price doesn't drop quite as far)
                # Allow slightly lower prices for SL (price doesn't rise quite as far)
                if price <= (tp2 * (1 + PRICE_TOLERANCE)):
                    result = "TP2"
                elif price <= (tp1 * (1 + PRICE_TOLERANCE)):
                    result = "TP1"
                elif price >= (sl * (1 - PRICE_TOLERANCE)):
                    result = "SL"

            if result:
                entry = payload.get("entry", 0)
                # Record stats before closing
                if self.stats:
                    self.stats.record_close(sig_id, price, result)

                # Only send result notification if enabled
                if ENABLE_RESULT_NOTIFICATIONS:
                    msg_direction = "BULLISH" if direction == "bullish" else "BEARISH"
                    msg = format_result_message(
                        symbol=symbol,
                        direction=msg_direction,
                        result=result,
                        entry=entry,
                        exit_price=price,
                        stop_loss=sl,
                        tp1=tp1,
                        tp2=tp2,
                        signal_id=sig_id,
                    )
                    self.notifier.send_message(msg)
                else:
                    logger.info("Result notification skipped (disabled): %s %s", sig_id, result)

                self.state.close_signal(sig_id, price, result)

    def send_signal(self, symbol: str, timeframe: str, signal: Dict[str, Any]) -> None:
        """Send signal notification and save to state."""
        direction_emoji = "🟢" if signal["direction"] == "bullish" else "🔴"
        pattern_name = signal["pattern"].upper()

        # Calculate R:R
        entry = signal["entry"]
        sl = signal["stop_loss"]
        tp1 = signal["take_profit_1"]
        risk = abs(entry - sl)
        reward = abs(tp1 - entry)
        rr = reward / risk if risk > 0 else 0

        sig_id = f"{symbol}-{timeframe}-{datetime.now(timezone.utc).isoformat()}"

        # Get performance stats for this symbol
        perf_line = ""
        if self.stats:
            counts = self.stats.symbol_tp_sl_counts(symbol)
            tp1_count = counts.get("TP1", 0)
            tp2_count = counts.get("TP2", 0)
            sl_count = counts.get("SL", 0)
            total = tp1_count + tp2_count + sl_count
            if total > 0:
                win_rate = (tp1_count + tp2_count) / total * 100
                perf_line = f"\n📊 <b>History:</b> {win_rate:.0f}% Win ({tp1_count + tp2_count}/{total}) | TP:{tp1_count + tp2_count} SL:{sl_count}"

        msg = (
            f"{direction_emoji} <b>HARMONIC: {pattern_name}</b>\n"
            f"{symbol} [{timeframe}]\n\n"
            f"<b>Direction:</b> {signal['direction'].upper()}\n"
            f"<b>Score:</b> {signal['score']:.1f}/100\n\n"
            f"🎯 <b>Setup:</b>\n"
            f"Entry: {entry:.4f}\n"
            f"TP1 ({self.config.tp1_rr}R): {tp1:.4f}\n"
            f"TP2 ({self.config.tp2_rr}R): {signal['take_profit_2']:.4f}\n"
            f"SL: {sl:.4f}\n"
            f"R:R: {rr:.2f}\n\n"
            f"📊 <b>Ratios:</b>\n"
            f"XAB: {signal['ratios'].get('xab', 0):.3f}\n"
            f"ABC: {signal['ratios'].get('abc', 0):.3f}\n"
            f"BCD: {signal['ratios'].get('bcd', 0):.3f}\n"
            f"XAD: {signal['ratios'].get('xad', 0):.3f}"
            f"{perf_line}\n\n"
            f"⏱️ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )

        self.notifier.send_message(msg)

        # Save to state
        signal_data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "pattern": signal["pattern"],
            "direction": signal["direction"],
            "score": signal["score"],
            "entry": entry,
            "stop_loss": sl,
            "take_profit_1": tp1,
            "take_profit_2": signal["take_profit_2"],
            "ratios": signal["ratios"],
        }
        self.state.add_signal(sig_id, signal_data, symbol)

        # Record to stats
        if self.stats:
            direction_upper = "BULLISH" if signal["direction"] == "bullish" else "BEARISH"
            self.stats.record_open(
                sig_id, symbol, direction_upper, entry,
                datetime.now(timezone.utc).isoformat(),
                extra={"pattern": signal["pattern"], "timeframe": timeframe}
            )

    def run_cycle(self):
        """Run one analysis cycle."""
        global shutdown_requested

        # Check open signals first
        self.check_open_signals()

        # Cleanup stale signals
        stale_count = self.state.cleanup_stale_signals(max_age_hours=24)
        if stale_count > 0:
            logger.info("Cleaned up %d stale signals", stale_count)

        # Scan for new patterns
        for item in self.watchlist:
            # Check for shutdown during watchlist scan
            if shutdown_requested:
                logger.info("Shutdown requested during watchlist scan")
                break

            symbol = item["symbol"]
            timeframe = item.get("timeframe", "1h")

            if not self.state.should_alert(symbol, self.config.cooldown_minutes):
                continue

            signal_result = self.analyze_symbol(symbol, timeframe)
            if signal_result:
                self.send_signal(symbol, timeframe, signal_result)


# =========================================================
# MAIN
# =========================================================
def main():
    global shutdown_requested

    parser = argparse.ArgumentParser(description="Harmonic Pattern Bot")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    bot = HarmonicPatternBot()
    logger.info("Harmonic Pattern Bot Started")
    logger.info(f"Config: min_score={bot.config.min_pattern_score}, min_rr={bot.config.min_risk_reward}")

    if args.once:
        bot.run_cycle()
        return

    try:
        while not shutdown_requested:
            try:
                bot.run_cycle()

                if shutdown_requested:
                    break

                logger.info("Cycle complete; sleeping 300s (5m)")
                # Sleep in 1-second chunks to respond quickly to shutdown signals
                for _ in range(300):
                    if shutdown_requested:
                        break
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                if shutdown_requested:
                    break
                time.sleep(30)
    finally:
        logger.info("Harmonic Pattern Bot stopped")


if __name__ == "__main__":
    main()
