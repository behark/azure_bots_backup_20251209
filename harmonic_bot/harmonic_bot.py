#!/usr/bin/env python3
# pyright: ignore-all
"""
Harmonic Pattern Bot - Automated harmonic pattern detection and alerts
Detects: Bat, Butterfly, Gartley, Crab, Shark, ABCD patterns and more
"""

from __future__ import annotations

import argparse
import fcntl
import html
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import ccxt  # type: ignore[import-untyped]
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
STATE_FILE = BASE_DIR / "harmonic_state.json"
WATCHLIST_FILE = BASE_DIR / "harmonic_watchlist.json"
STATS_FILE = LOG_DIR / "harmonic_stats.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env", override=True)
    load_dotenv(BASE_DIR.parent / ".env", override=True)
except ImportError:
    pass

# Add parent directory to path for shared modules
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))

# Required imports (fail fast if missing)
from message_templates import format_signal_message
from notifier import TelegramNotifier
from signal_stats import SignalStats
from trade_config import get_config_manager

# Optional imports (safe fallback)
from safe_import import safe_import
HealthMonitor = safe_import('health_monitor', 'HealthMonitor')
RateLimiter = None  # Disabled for testing
RateLimitHandler = safe_import('rate_limit_handler', 'RateLimitHandler')

# Import configuration module (Same config structure as Volume Bot)
# We assume we can load the JSON config similarly
def load_json_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return cast(Dict[str, Any], json.loads(path.read_text()))
    except Exception:
        return {}

def setup_logging(log_level: str = "INFO", enable_detailed: bool = False) -> logging.Logger:
    """Setup enhanced logging with rotation and detailed formatting."""
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
        LOG_DIR / "harmonic_bot.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    error_handler = RotatingFileHandler(
        LOG_DIR / "harmonic_errors.log", maxBytes=5 * 1024 * 1024, backupCount=3
    )
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    return logging.getLogger("harmonic_bot")

logger = logging.getLogger("harmonic_bot")

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
    """Resolve symbol format for CCXT."""
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
class HarmonicSignal:
    symbol: str
    pattern_name: str
    direction: str
    timestamp: str
    d_timestamp: str
    x: float
    a: float
    b: float
    c: float
    d: float
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    error_score: float
    current_price: float
    timeframe: str
    exchange: str
    take_profit_3: Optional[float] = None
    xab: float = 0.0
    abc: float = 0.0
    bcd: float = 0.0
    xad: float = 0.0
    confidence: float = 0.0  # Confidence score 0-1 (from Volume Bot)
    prz_low: float = 0.0     # PRZ range
    prz_high: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

class HarmonicPatternDetector:
    """Core logic for Harmonic Pattern Detection (unchanged, just imported/encapsulated)."""
    
    PATTERNS = {
        "Cypher": {"xab": (0.382, 0.618), "abc": (1.13, 1.414), "bcd": (1.272, 2.0), "xad": (0.76, 0.80)},
        "Deep Crab": {"xab": (0.88, 0.895), "abc": (0.382, 0.886), "bcd": (2.0, 3.618), "xad": (1.60, 1.63)},
        "Bat": {"xab": (0.382, 0.5), "abc": (0.382, 0.886), "bcd": (1.618, 2.618), "xad": (0.0, 0.886)},
        "Butterfly": {"xab": (0.0, 0.786), "abc": (0.382, 0.886), "bcd": (1.618, 2.618), "xad": (1.27, 1.618)},
        "Gartley": {"xab": (0.61, 0.625), "abc": (0.382, 0.886), "bcd": (1.13, 2.618), "xad": (0.75, 0.875)},
        "Crab": {"xab": (0.5, 0.875), "abc": (0.382, 0.886), "bcd": (2.0, 5.0), "xad": (1.382, 5.0)},
        "Shark": {"xab": (0.5, 0.875), "abc": (1.13, 1.618), "bcd": (1.27, 2.24), "xad": (0.886, 1.13)},
        "ABCD": {"xab": (0.0, 999.0), "abc": (0.382, 0.886), "bcd": (1.13, 2.618), "xad": (0.0, 999.0)},
    }
    
    PATTERN_IDEALS = {
        "Cypher": {"xab": 0.5, "abc": 1.272, "bcd": 1.414, "xad": 0.786},
        "Deep Crab": {"xab": 0.886, "abc": 0.618, "bcd": 2.618, "xad": 1.618},
        "Crab": {"xab": 0.618, "abc": 0.618, "bcd": 2.618, "xad": 1.618},
        "Butterfly": {"xab": 0.786, "abc": 0.618, "bcd": 2.24, "xad": 1.41},
        "Gartley": {"xab": 0.618, "abc": 0.618, "bcd": 1.272, "xad": 0.786},
        "Bat": {"xab": 0.5, "abc": 0.618, "bcd": 2.0, "xad": 0.886},
        "Shark": {"xab": 0.618, "abc": 1.27, "bcd": 1.618, "xad": 1.0},
        "ABCD": {"abc": 0.618, "bcd": 1.27},
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.error_thresholds = self.config.get("analysis", {}).get("pattern_error_thresholds", {
            "Cypher": 0.08, "Shark": 0.08, "Gartley": 0.10, "Bat": 0.10,
            "Butterfly": 0.12, "Crab": 0.12, "Deep Crab": 0.12, "ABCD": 0.15
        })

    def calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        if len(closes) < period + 1:
            return float(np.mean(np.array(highs) - np.array(lows))) if highs and lows else 0.0
        trs = []
        for i in range(1, len(closes)):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            trs.append(tr)
        return float(np.mean(trs[-period:])) if len(trs) >= period else float(np.mean(trs) if trs else 0.0)

    def calculate_rsi(self, closes: List[float], period: int = 14) -> float:
        if len(closes) < period + 1: return 50.0
        deltas = np.diff(closes)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        if down == 0: return 100.0
        rs = up / down
        return float(100 - (100 / (1 + rs)))

    def find_pivots(self, highs: List[float], lows: List[float], closes: List[float], opens: List[float], atr: float) -> List[Tuple[int, float, str]]:
        # Increased ZigZag multiplier (Pro Upgrade)
        mult = self.config.get("analysis", {}).get("zigzag_atr_multiplier", 3.0)
        atr_threshold = max(atr * mult, 1e-8)
        
        # ... (ZigZag Logic)
        highs_arr = np.asarray(highs, dtype=float)
        lows_arr = np.asarray(lows, dtype=float)
        closes_arr = np.asarray(closes, dtype=float)
        opens_arr = np.asarray(opens, dtype=float)
        n = closes_arr.size
        if n < 2: return []
        
        candle_dirs = np.where(closes_arr >= opens_arr, 1, -1)
        change_points = np.flatnonzero(candle_dirs[1:] != candle_dirs[:-1]) + 1
        if change_points.size == 0: return []
        segments = np.split(np.arange(n, dtype=int), change_points)
        segments = [seg for seg in segments if seg.size > 0]
        if len(segments) < 2: return []

        pivots: List[Tuple[int, float, str]] = []
        segment_dirs = [int(candle_dirs[seg[0]]) for seg in segments]
        for seg_idx in range(1, len(segments)):
            prev_seg = segments[seg_idx - 1]
            prev_dir = segment_dirs[seg_idx - 1]
            if prev_dir == 1:
                local_idx = prev_seg[np.argmax(highs_arr[prev_seg])]
                price = float(highs_arr[local_idx])
                pivot_type = 'H'
            else:
                local_idx = prev_seg[np.argmin(lows_arr[prev_seg])]
                price = float(lows_arr[local_idx])
                pivot_type = 'L'
            if not pivots or abs(price - pivots[-1][1]) >= atr_threshold:
                pivots.append((int(local_idx), price, pivot_type))
        return pivots

    def detect(self, ohlcv: List[List[Any]], symbol: str, timeframe: str, current_price: float, exchange: str = "binanceusdm") -> Optional[HarmonicSignal]:
        # Validate OHLCV data
        if not ohlcv or not isinstance(ohlcv, list):
            logger.debug(f"Invalid OHLCV data for {symbol}: empty or not a list")
            return None

        if len(ohlcv) < 50:
            logger.debug(f"Insufficient OHLCV data for {symbol}: {len(ohlcv)} candles (need 50+)")
            return None

        # Look-ahead bias prevention (from Volume Bot):
        # Exclude current (incomplete) candle to avoid using future data
        # Use only CLOSED candles for calculations
        if len(ohlcv) > 1:
            ohlcv = ohlcv[:-1]  # Remove last incomplete candle
            logger.debug(f"{symbol}: Using {len(ohlcv)} closed candles (excluded incomplete candle)")

        # Validate OHLCV structure
        try:
            opens = [x[1] for x in ohlcv]
            highs = [x[2] for x in ohlcv]
            lows = [x[3] for x in ohlcv]
            closes = [x[4] for x in ohlcv]
        except (IndexError, TypeError) as e:
            logger.error(f"Malformed OHLCV data for {symbol}: {e}")
            return None
        
        atr = self.calculate_atr(highs, lows, closes)
        if atr <= 0:
            # Fix 1.9: Use average high-low range instead of std deviation
            fallback_period = self.config.get("analysis", {}).get("atr_fallback_period", 50)
            highs_arr = np.array(highs[-fallback_period:])
            lows_arr = np.array(lows[-fallback_period:])
            atr = max(float(np.mean(highs_arr - lows_arr)), 1e-6)
        
        pivots = self.find_pivots(highs, lows, closes, opens, atr)
        if len(pivots) < 5:
            logger.debug(f"{symbol}: Insufficient pivots ({len(pivots)}/5 needed)")
            return None
        
        # Use last 5 pivots
        x_idx, x, x_type = pivots[-5]
        a_idx, a, a_type = pivots[-4]
        b_idx, b, b_type = pivots[-3]
        c_idx, c, c_type = pivots[-2]
        d_idx, d, d_type = pivots[-1]
        
        # Check separation with configurable minimum leg candles (Fix 1.3)
        min_leg_candles = self.config.get("analysis", {}).get("min_leg_candles", 3)
        if any((p2 - p1) < min_leg_candles for p1, p2 in [(x_idx, a_idx), (a_idx, b_idx), (b_idx, c_idx), (c_idx, d_idx)]):
            logger.debug(f"{symbol}: Pivots too close together (need {min_leg_candles}+ candles separation)")
            return None
        
        # Check patterns
        ratios = self._get_ratios(x, a, b, c, d)
        if not ratios:
            logger.debug(f"{symbol}: Invalid ratios (division by zero)")
            return None
        xab, xad, abc, bcd = ratios
        
        # Determine direction from XA leg (Fix 1.2)
        # Bullish: A < X (price moved down from X to A, expecting reversal up)
        # Bearish: A > X (price moved up from X to A, expecting reversal down)
        direction = "BULLISH" if a < x else "BEARISH"
        
        best_pattern = None
        for name, ranges in self.PATTERNS.items():
            # Range checks - validate ALL ratios including BCD (Fix 1.1)
            if not (ranges["xab"][0] <= xab <= ranges["xab"][1] and
                    ranges["abc"][0] <= abc <= ranges["abc"][1] and
                    ranges["bcd"][0] <= bcd <= ranges["bcd"][1] and
                    ranges["xad"][0] <= xad <= ranges["xad"][1]):
                continue
            
            # Error score
            ideals = self.PATTERN_IDEALS.get(name, {})
            score = 0.0
            score += abs(xab - ideals.get("xab", xab))
            score += abs(abc - ideals.get("abc", abc))
            score += abs(bcd - ideals.get("bcd", bcd))
            score += abs(xad - ideals.get("xad", xad))
            
            thresh = self.error_thresholds.get(name, 0.15)
            if score <= thresh:
                if best_pattern is None or score < best_pattern[2]:
                    best_pattern = (name, direction, score)
        
        if not best_pattern:
            logger.debug(f"{symbol}: No pattern match (ratios: XAB={xab:.3f}, ABC={abc:.3f}, BCD={bcd:.3f}, XAD={xad:.3f})")
            return None

        # RSI Check - now optional with enable_rsi_filter (Fix 1.4)
        rsi = self.calculate_rsi(closes)
        enable_rsi_filter = self.config.get("analysis", {}).get("enable_rsi_filter", True)
        if enable_rsi_filter:
            rsi_overbought = self.config.get("analysis", {}).get("rsi_overbought", 70)
            rsi_oversold = self.config.get("analysis", {}).get("rsi_oversold", 30)
            if direction == "BULLISH" and rsi > rsi_overbought:
                logger.debug(f"{symbol}: BULLISH pattern rejected - RSI too high ({rsi:.1f} > {rsi_overbought})")
                return None
            if direction == "BEARISH" and rsi < rsi_oversold:
                logger.debug(f"{symbol}: BEARISH pattern rejected - RSI too low ({rsi:.1f} < {rsi_oversold})")
                return None

        # Stale Check
        d_age = (len(closes) - 1) - d_idx
        if d_age > self.config.get("analysis", {}).get("max_pattern_age_candles", 3):
            logger.debug(f"{symbol}: Pattern too old ({d_age} candles, max 3)")
            return None
        
        # Targets
        targets = self._calculate_targets(c, d, direction, atr, symbol)

        # PRZ Validation - check if current price is within PRZ zone (Fix 1.5)
        enable_prz_validation = self.config.get("analysis", {}).get("enable_prz_validation", True)
        if enable_prz_validation:
            prz_low = targets.get("prz_low", 0)
            prz_high = targets.get("prz_high", float('inf'))
            if not (prz_low <= current_price <= prz_high):
                logger.debug(f"{symbol}: Price {current_price:.6f} outside PRZ [{prz_low:.6f}, {prz_high:.6f}]")
                return None

        d_ts_ms = ohlcv[d_idx][0]
        d_ts_str = datetime.fromtimestamp(d_ts_ms/1000, timezone.utc).isoformat()

        # Compute confidence score (from Volume Bot)
        prz_dist_pct = abs(current_price - d) / d * 100 if d > 0 else 0
        confidence = self._compute_confidence(
            pattern_name=best_pattern[0],
            error_score=best_pattern[2],
            rsi=rsi,
            direction=direction,
            d_age=d_age,
            prz_dist_pct=prz_dist_pct
        )

        logger.debug(f"{symbol}: ✓ {best_pattern[0]} pattern detected! {direction}, RSI={rsi:.1f}, Error={best_pattern[2]:.4f}, Confidence={confidence:.2f}")

        return HarmonicSignal(
            symbol=symbol, pattern_name=best_pattern[0], direction=direction,
            timestamp=datetime.now(timezone.utc).isoformat(), d_timestamp=d_ts_str,
            x=x, a=a, b=b, c=c, d=d, xab=xab, abc=abc, bcd=bcd, xad=xad,
            entry=targets["entry"], stop_loss=targets["stop_loss"],
            take_profit_1=targets["take_profit_1"], take_profit_2=targets["take_profit_2"],
            take_profit_3=targets["take_profit_3"],
            error_score=best_pattern[2], current_price=current_price,
            timeframe=timeframe, exchange=exchange,
            confidence=confidence,
            prz_low=targets.get("prz_low", 0),
            prz_high=targets.get("prz_high", 0)
        )

    def _get_ratios(self, x: float, a: float, b: float, c: float, d: float) -> Optional[Tuple[float, float, float, float]]:
        # Check denominators: XAB uses (x-a), ABC uses (a-b), BCD uses (b-c), XAD uses (x-a)
        if x == a or a == b or b == c: return None
        xab = abs(b - a) / abs(x - a)
        xad = abs(a - d) / abs(x - a)
        abc = abs(b - c) / abs(a - b)
        bcd = abs(c - d) / abs(b - c)
        return xab, xad, abc, bcd

    def _compute_confidence(self, pattern_name: str, error_score: float, rsi: float,
                           direction: str, d_age: int, prz_dist_pct: float) -> float:
        """
        Compute confidence score 0-1 based on pattern quality (from Volume Bot).

        Factors weighted:
        - Error score (lower is better): 40%
        - Pattern age (fresher is better): 20%
        - RSI alignment: 20%
        - PRZ distance (closer is better): 20%
        """
        # Error score component (0-1, lower error = higher confidence)
        # Typical error scores range 0.0 - 0.15
        error_conf = max(0, 1 - (error_score / 0.15))

        # Pattern age component (0-1, fresher = higher)
        max_age = self.config.get("analysis", {}).get("max_pattern_age_candles", 3)
        age_conf = max(0, 1 - (d_age / max_age))

        # RSI alignment component
        if direction == "BULLISH":
            # Bullish: RSI < 50 is good (oversold conditions)
            rsi_conf = max(0, (50 - rsi) / 50) if rsi < 50 else max(0, (100 - rsi) / 100)
        else:
            # Bearish: RSI > 50 is good (overbought conditions)
            rsi_conf = max(0, (rsi - 50) / 50) if rsi > 50 else max(0, rsi / 100)

        # PRZ distance component (0-1, closer = higher)
        # prz_dist_pct is the % distance from current price to D point
        prz_conf = max(0, 1 - (prz_dist_pct / 2))  # 2% = 0 confidence

        # Weighted average
        confidence = (
            error_conf * 0.40 +
            age_conf * 0.20 +
            rsi_conf * 0.20 +
            prz_conf * 0.20
        )

        return float(round(min(1.0, max(0.0, confidence)), 3))

    def _calculate_targets(self, c: float, d: float, direction: str, atr: float, symbol: str) -> Dict[str, float]:
        # Get risk config from centralized trade_config
        config_mgr = get_config_manager()
        risk_config = config_mgr.get_effective_risk("harmonic_bot", symbol)

        # Harmonic-specific config (from local config)
        entry_buffer_mult = self.config.get("analysis", {}).get("entry_buffer_atr_multiplier", 0.05)
        sl_fib_mult = self.config.get("analysis", {}).get("sl_fib_range_multiplier", 0.5)
        prz_atr_mult = self.config.get("analysis", {}).get("prz_atr_multiplier", 0.5)

        # Standard risk config from centralized config
        tp1_mult = risk_config.tp1_atr_multiplier
        tp2_mult = risk_config.tp2_atr_multiplier
        tp3_mult = getattr(risk_config, 'tp3_atr_multiplier', 4.5)

        fib_range = abs(d - c)
        entry_buffer = atr * entry_buffer_mult
        prz_buffer = atr * prz_atr_mult  # PRZ range around D (Fix 1.5)

        if direction == "BULLISH":
            entry = d + entry_buffer
            # PRZ zone: price should be within this range for valid entry
            prz_low = d - prz_buffer
            prz_high = d + prz_buffer
            sl_default = d - (fib_range * sl_fib_mult)
            if sl_default >= entry: sl_default = entry - (atr * 2)
            risk = abs(entry - sl_default)
            tp1 = entry + (risk * tp1_mult)
            tp2 = entry + (risk * tp2_mult)
            tp3 = entry + (risk * tp3_mult)
            sl = sl_default
        else:
            entry = d - entry_buffer
            # PRZ zone for bearish
            prz_low = d - prz_buffer
            prz_high = d + prz_buffer
            sl_default = d + (fib_range * sl_fib_mult)
            if sl_default <= entry: sl_default = entry + (atr * 2)
            risk = abs(entry - sl_default)
            tp1 = entry - (risk * tp1_mult)
            tp2 = entry - (risk * tp2_mult)
            tp3 = entry - (risk * tp3_mult)
            sl = sl_default

        return {
            "entry": entry,
            "stop_loss": sl,
            "take_profit_1": tp1,
            "take_profit_2": tp2,
            "take_profit_3": tp3,
            "prz_low": prz_low,
            "prz_high": prz_high
        }


class SignalTracker:
    """Robust signal tracker with file locking and reversal detection (Ported from Volume Bot)."""

    STATE_VERSION = 2  # Increment when state format changes (Fix 1.10)

    def __init__(self, stats: Optional[Any] = None, config: Optional[Dict[str, Any]] = None) -> None:
        self.stats = stats
        self.config: Dict[str, Any] = config or {}
        self.state_lock = threading.Lock()
        self.state = self._load_state()

    def _empty_state(self) -> Dict[str, Any]:
        return {
            "version": self.STATE_VERSION,  # Fix 1.10: Add version field
            "last_alerts": {},
            "open_signals": {},
            "signal_history": {},
            "last_result_notifications": {}
        }

    def _load_state(self) -> Dict[str, Any]:
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, 'r') as f:
                    loaded_state = cast(Dict[str, Any], json.load(f))

                # Fix 1.10: Version validation and migration
                loaded_version = loaded_state.get("version", 1)
                if loaded_version < self.STATE_VERSION:
                    logger.warning(f"State version mismatch (file: {loaded_version}, expected: {self.STATE_VERSION}). Migrating...")
                    # Migrate old state to new format
                    new_state = self._empty_state()
                    # Preserve compatible fields
                    for key in ["last_alerts", "open_signals", "signal_history", "last_result_notifications"]:
                        if key in loaded_state:
                            new_state[key] = loaded_state[key]
                    new_state["version"] = self.STATE_VERSION
                    logger.info(f"State migrated from v{loaded_version} to v{self.STATE_VERSION}")
                    # Save migrated state immediately to prevent re-migration on next restart
                    try:
                        with open(STATE_FILE, 'w') as f:
                            json.dump(new_state, f, indent=2)
                        logger.info("Migrated state saved to disk")
                    except Exception as e:
                        logger.warning(f"Could not save migrated state: {e}")
                    return new_state

                return loaded_state
            except json.JSONDecodeError as e:
                logger.error(f"State file has invalid JSON: {e}, rebuilding from scratch")
                return self._empty_state()
            except IOError as e:
                logger.error(f"Failed to read state file: {e}, rebuilding from scratch")
                return self._empty_state()
            except Exception as e:
                logger.error(f"Unexpected error loading state: {e}, rebuilding from scratch")
                return self._empty_state()
        return self._empty_state()

    def _save_state(self) -> None:
        with self.state_lock:
            temp_file = STATE_FILE.with_suffix('.tmp')
            try:
                with open(temp_file, 'w') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    json.dump(self.state, f, indent=2)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                temp_file.replace(STATE_FILE)
            except Exception as e:
                logger.error(f"Failed to save state: {e}")

    def is_duplicate(self, symbol: str, pattern: str, d_ts: str, exchange: str = "binanceusdm", timeframe: str = "1h") -> bool:
        """Check duplicate using D-Timestamp with exchange+timeframe scope (Fix 1.6)."""
        history = self.state.setdefault("signal_history", {})
        # Include exchange and timeframe in key to allow same pattern on different TFs/exchanges
        check_exchange = self.config.get("exchange_settings", {}).get("check_exchange_for_duplicates", True)
        check_timeframe = self.config.get("exchange_settings", {}).get("check_timeframe_for_duplicates", True)

        key_parts = [symbol, pattern]
        if check_exchange:
            key_parts.append(exchange)
        if check_timeframe:
            key_parts.append(timeframe)
        key = "|".join(key_parts)
        return d_ts in history.get(key, [])

    def add_signal(self, signal: HarmonicSignal) -> None:
        # Mark history with exchange+timeframe aware key (Fix 1.6)
        history = self.state.setdefault("signal_history", {})
        check_exchange = self.config.get("exchange_settings", {}).get("check_exchange_for_duplicates", True)
        check_timeframe = self.config.get("exchange_settings", {}).get("check_timeframe_for_duplicates", True)

        key_parts = [signal.symbol, signal.pattern_name]
        if check_exchange:
            key_parts.append(signal.exchange)
        if check_timeframe:
            key_parts.append(signal.timeframe)
        key = "|".join(key_parts)

        history_limit = self.config.get("analysis", {}).get("signal_history_limit", 10)
        if key not in history: history[key] = []
        if signal.d_timestamp not in history[key]:
            history[key].append(signal.d_timestamp)
            if len(history[key]) > history_limit: history[key] = history[key][-history_limit:]
            
        # Add open signal
        signals = self.state.setdefault("open_signals", {})
        sig_id = f"{signal.symbol}-{signal.pattern_name}-{signal.timestamp}"
        signals[sig_id] = signal.as_dict()
        self._save_state()
        
        if self.stats:
            self.stats.record_open(sig_id, signal.symbol, signal.direction, signal.entry, signal.timestamp)

    def check_open_signals(self, client_map: Dict[str, Any], notifier: Optional[Any], rate_limiter: Optional[Any] = None) -> None:
        """Check open signals with rate limiting support (Fix 2.1)."""
        signals = self.state.get("open_signals", {})
        if not signals: return

        updated = False
        symbol_delay = self.config.get("execution", {}).get("symbol_delay_seconds", 1)

        for sig_id, payload in list(signals.items()):
            # Validate signal data
            symbol = payload.get("symbol")
            exchange = payload.get("exchange", "mexc")
            direction = payload.get("direction")

            if not isinstance(symbol, str) or not symbol or not symbol.strip():
                logger.warning(f"Removing signal {sig_id} with invalid symbol: {symbol}")
                signals.pop(sig_id, None)
                updated = True
                continue

            if not isinstance(exchange, str) or exchange not in ["binanceusdm", "mexc", "bybit"]:
                logger.warning(f"Invalid exchange '{exchange}' for {sig_id}, using mexc")
                exchange = "mexc"

            if direction not in ["BULLISH", "BEARISH"]:
                logger.warning(f"Removing signal {sig_id} with invalid direction: {direction}")
                signals.pop(sig_id, None)
                updated = True
                continue

            # Fix 2.1: Use rate limiter before API call
            if rate_limiter:
                rate_limiter.wait_if_needed()

            # Fetch price with specific exception handling
            price = None
            try:
                client = client_map.get(exchange) or client_map.get("binanceusdm")
                if not client:
                    logger.warning(f"No client for exchange {exchange}, skipping {symbol}")
                    continue
                ticker = client.fetch_ticker(resolve_symbol(symbol))
                price = ticker.get("last")

                if rate_limiter:
                    rate_limiter.record_success(f"check_{exchange}_{symbol}")
            except ccxt.NetworkError as e:
                logger.warning(f"Network error fetching {symbol}: {e}")
                if rate_limiter:
                    rate_limiter.record_error(f"check_{exchange}_{symbol}")
                continue
            except ccxt.RateLimitExceeded as e:
                logger.warning(f"Rate limit exceeded for {symbol}: {e}")
                if rate_limiter:
                    rate_limiter.record_error(f"check_{exchange}_{symbol}")
                time.sleep(5)  # Longer backoff on rate limit
                continue
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error for {symbol}: {e}")
                if rate_limiter:
                    rate_limiter.record_error(f"check_{exchange}_{symbol}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error fetching {symbol}: {e}")
                continue

            # Add delay between API calls to prevent bursting
            time.sleep(symbol_delay * 0.5)

            # Validate price
            if price is None or not isinstance(price, (int, float)) or price <= 0:
                logger.debug(f"Invalid price for {symbol}: {price}")
                continue

            # Validate and check TP/SL - now including TP3 (Fix 1.7)
            tp1, tp2 = payload.get("take_profit_1"), payload.get("take_profit_2")
            tp3 = payload.get("take_profit_3")  # TP3 support
            sl = payload.get("stop_loss")
            entry = payload.get("entry")

            # Track which TPs have been hit (partial TP tracking)
            tp_hits = payload.get("tp_hits", {"tp1": False, "tp2": False, "tp3": False})

            # Ensure all required values are valid numbers
            if not all(isinstance(v, (int, float)) for v in [tp1, tp2, sl, entry]):
                logger.warning(f"Signal {sig_id} has invalid TP/SL values, removing")
                signals.pop(sig_id, None)
                updated = True
                continue

            # Check for TP hits with partial tracking (Fix 1.7)
            res = None
            close_signal = False

            if direction == "BULLISH":
                # Check TP3 first (highest), then TP2, then TP1
                if tp3 and isinstance(tp3, (int, float)) and price >= tp3:
                    if not tp_hits.get("tp3"):
                        res = "TP3"
                        tp_hits["tp3"] = True
                        close_signal = True  # TP3 closes the trade
                elif price >= tp2:
                    if not tp_hits.get("tp2"):
                        res = "TP2"
                        tp_hits["tp2"] = True
                        tp_hits["tp1"] = True  # Implied
                elif price >= tp1:
                    if not tp_hits.get("tp1"):
                        res = "TP1"
                        tp_hits["tp1"] = True
                elif price <= sl:
                    res = "SL"
                    close_signal = True
            else:
                # BEARISH - inverted logic
                if tp3 and isinstance(tp3, (int, float)) and price <= tp3:
                    if not tp_hits.get("tp3"):
                        res = "TP3"
                        tp_hits["tp3"] = True
                        close_signal = True
                elif price <= tp2:
                    if not tp_hits.get("tp2"):
                        res = "TP2"
                        tp_hits["tp2"] = True
                        tp_hits["tp1"] = True
                elif price <= tp1:
                    if not tp_hits.get("tp1"):
                        res = "TP1"
                        tp_hits["tp1"] = True
                elif price >= sl:
                    res = "SL"
                    close_signal = True

            # Update tp_hits in payload if we hit something
            if res and not close_signal:
                payload["tp_hits"] = tp_hits
                updated = True

            if res:
                # Check result notification cooldown (from Volume Bot)
                result_cooldown = self.config.get("signal", {}).get("result_notification_cooldown_minutes", 15)
                should_notify = self._should_notify_result(symbol, result_cooldown)

                # Notify with HTML escaping
                pnl = (price - entry)/entry * 100 if direction == "BULLISH" else (entry - price)/entry * 100
                safe_symbol = html.escape(symbol)
                safe_pattern = html.escape(payload.get('pattern_name', ''))
                emoji = "🎯" if res in ["TP1", "TP2", "TP3"] else "🛑"

                # Show partial TP status
                tp_status = ""
                if not close_signal and res.startswith("TP"):
                    remaining_tps = []
                    if not tp_hits.get("tp2"): remaining_tps.append("TP2")
                    if tp3 and not tp_hits.get("tp3"): remaining_tps.append("TP3")
                    if remaining_tps:
                        tp_status = f"\n📊 <b>Remaining:</b> {', '.join(remaining_tps)}"

                if should_notify:
                    msg = (
                        f"{emoji} <b>{safe_symbol} {safe_pattern}</b>\n\n"
                        f"<b>Result:</b> {res} HIT!{' (PARTIAL)' if not close_signal else ''}\n"
                        f"💰 <b>Entry:</b> <code>{entry:.6f}</code>\n"
                        f"💵 <b>{'Exit' if close_signal else 'Current'}:</b> <code>{price:.6f}</code>\n"
                        f"📈 <b>PnL:</b> {pnl:+.2f}%{tp_status}"
                    )
                    if notifier:
                        notifier.send_message(msg, parse_mode="HTML")
                        self._mark_result_notified(symbol)
                        logger.info(f"📤 Result notification sent for {sig_id}")
                else:
                    logger.info(f"⏭️ Skipping duplicate result notification for {symbol} (within {result_cooldown}m cooldown)")

                # Stats - only record close on final TP or SL
                if close_signal:
                    if self.stats: self.stats.record_close(sig_id, price, res)
                    del signals[sig_id]
                    updated = True
        
        if updated: self._save_state()

    def should_alert(self, symbol: str, pattern: str, cooldown_minutes: int) -> bool:
        """Check if enough time has passed since last alert for this symbol-pattern pair."""
        last_alerts = self.state.setdefault("last_alerts", {})
        key = f"{symbol}|{pattern}"
        last_time_str = last_alerts.get(key)

        if not last_time_str:
            return True

        try:
            last_time = datetime.fromisoformat(last_time_str)
            elapsed = datetime.now(timezone.utc) - last_time
            return elapsed.total_seconds() >= (cooldown_minutes * 60)
        except Exception:
            return True

    def mark_alert(self, symbol: str, pattern: str) -> None:
        """Mark that we just sent an alert for this symbol-pattern pair."""
        last_alerts = self.state.setdefault("last_alerts", {})
        key = f"{symbol}|{pattern}"
        last_alerts[key] = datetime.now(timezone.utc).isoformat()
        self._save_state()

    def cleanup_stale_signals(self, max_age_hours: int = 24) -> int:
        """Remove signals older than max_age_hours and archive to stats (from Volume Bot)."""
        signals = self.state.get("open_signals", {})
        if not signals:
            return 0

        removed = []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

        for sig_id, payload in list(signals.items()):
            timestamp_str = payload.get("timestamp")
            if not timestamp_str:
                # Archive to stats as expired before removing
                if self.stats:
                    self.stats.record_close(
                        sig_id,
                        exit_price=payload.get("entry", 0),
                        result="EXPIRED"
                    )
                removed.append(sig_id)
                continue

            try:
                sig_time = datetime.fromisoformat(timestamp_str)
                if sig_time.tzinfo is None:
                    sig_time = sig_time.replace(tzinfo=timezone.utc)

                age_hours = (datetime.now(timezone.utc) - sig_time).total_seconds() / 3600

                if sig_time < cutoff:
                    # Archive to stats as expired before removing
                    if self.stats:
                        self.stats.record_close(
                            sig_id,
                            exit_price=payload.get("entry", 0),
                            result="EXPIRED"
                        )
                    removed.append(sig_id)
                    logger.info(f"Removed stale signal {sig_id} (age: {age_hours:.1f}h)")
            except Exception:
                removed.append(sig_id)

        for sig_id in removed:
            del signals[sig_id]

        if removed:
            self._save_state()
            logger.info(f"Cleaned up {len(removed)} stale signals")

        return len(removed)

    def _should_notify_result(self, symbol: str, cooldown_minutes: int) -> bool:
        """Check if we should send result notification (from Volume Bot)."""
        last_notifs = self.state.setdefault("last_result_notifications", {})
        if not isinstance(last_notifs, dict):
            last_notifs = {}
            self.state["last_result_notifications"] = last_notifs

        last_ts = last_notifs.get(symbol)
        if not isinstance(last_ts, str):
            return True

        try:
            last_dt = datetime.fromisoformat(last_ts)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return True

        now = datetime.now(timezone.utc)
        return now - last_dt >= timedelta(minutes=cooldown_minutes)

    def _mark_result_notified(self, symbol: str) -> None:
        """Mark that we sent a result notification for this symbol (from Volume Bot)."""
        last_notifs = self.state.setdefault("last_result_notifications", {})
        if not isinstance(last_notifs, dict):
            last_notifs = {}
            self.state["last_result_notifications"] = last_notifs
        last_notifs[symbol] = datetime.now(timezone.utc).isoformat()
        self._save_state()

    def check_reversal(self, symbol: str, new_direction: str, notifier: Optional[Any]) -> None:
        signals = self.state.get("open_signals", {})
        for sig_id, payload in signals.items():
            if payload.get("symbol") == symbol:
                old_dir = payload.get("direction")
                if old_dir != new_direction:
                    safe_symbol = html.escape(symbol)
                    safe_old = html.escape(old_dir)
                    safe_new = html.escape(new_direction)
                    msg = (
                        f"⚠️ <b>SIGNAL REVERSAL</b> ⚠️\n\n"
                        f"<b>Symbol:</b> {safe_symbol}\n"
                        f"<b>Open:</b> {safe_old}\n"
                        f"<b>New:</b> {safe_new}\n\n"
                        f"💡 Check your position!"
                    )
                    if notifier: notifier.send_message(msg, parse_mode="HTML")

class HarmonicBot:
    """Refactored Harmonic Bot with Volume Bot features."""

    def __init__(self, config_path: Path):
        self.config = load_json_config(config_path)

        # Init components
        self.watchlist = self._load_watchlist()
        self.analyzer = HarmonicPatternDetector(self.config)
        self.stats = SignalStats("Harmonic Bot", STATS_FILE)
        self.tracker = SignalTracker(self.stats, self.config)
        self.notifier = self._init_notifier()

        # Init Health Monitor
        heartbeat_interval = self.config.get("execution", {}).get("health_check_interval_seconds", 3600)
        self.health_monitor = HealthMonitor("Harmonic Bot", self.notifier, heartbeat_interval=heartbeat_interval) if HealthMonitor and self.notifier else None

        # Init Rate Limiter
        calls_per_min = self.config.get("rate_limit", {}).get("calls_per_minute", 60)
        self.rate_limiter = RateLimiter(calls_per_minute=calls_per_min, backoff_file=LOG_DIR / "rate_limiter.json") if RateLimiter else None
        if self.rate_limiter:
            logger.info(f"Rate limiter initialized: {calls_per_min} calls/min")

        # Exchange backoff tracking (from Volume Bot)
        self.exchange_backoff: Dict[str, float] = {}
        self.exchange_delay: Dict[str, float] = {}

        # Init Exchanges with timeout
        request_timeout = self.config.get("analysis", {}).get("request_timeout_seconds", 30)
        self.clients = {}
        for name, cfg in EXCHANGE_CONFIG.items():
            try:
                params = dict(cfg["params"])
                params['timeout'] = request_timeout * 1000  # ccxt uses milliseconds
                self.clients[name] = cfg["factory"](params)
            except Exception as e:
                logger.error(f"Failed to init {name}: {e}")

        # Validate exchanges on startup
        self._validate_exchanges()

        # Sync existing signals to stats on startup
        self._sync_signals_to_stats()

    def _validate_exchanges(self) -> None:
        """Validate exchange connections on startup (from Volume Bot)."""
        for exchange_name, client in self.clients.items():
            try:
                # Try to load markets to verify connection
                client.load_markets()
                logger.info(f"✅ {exchange_name} connection validated")
            except ccxt.NetworkError as e:
                logger.warning(f"⚠️ Network error connecting to {exchange_name}: {e}")
            except ccxt.ExchangeError as e:
                logger.warning(f"⚠️ Exchange error for {exchange_name}: {e}")
            except Exception as e:
                logger.warning(f"⚠️ Could not validate {exchange_name}: {e}")

    def _sync_signals_to_stats(self) -> None:
        """Sync open signals from state to stats on startup (from Volume Bot)."""
        if not self.stats:
            return

        signals = self.tracker.state.get("open_signals", {})
        if not isinstance(signals, dict):
            return

        stats_data = self.stats.data if isinstance(self.stats.data, dict) else {}
        stats_open_raw = stats_data.get("open", {})
        stats_open: Dict[str, Any] = stats_open_raw if isinstance(stats_open_raw, dict) else {}
        synced = 0

        for signal_id, payload in signals.items():
            if signal_id not in stats_open and isinstance(payload, dict):
                self.stats.record_open(
                    signal_id=signal_id,
                    symbol=payload.get("symbol", ""),
                    direction=payload.get("direction", "BULLISH"),
                    entry=payload.get("entry", 0),
                    created_at=payload.get("timestamp", ""),
                    extra={
                        "timeframe": payload.get("timeframe", ""),
                        "exchange": payload.get("exchange", ""),
                        "pattern": payload.get("pattern_name", ""),
                    },
                )
                synced += 1

        if synced > 0:
            logger.info(f"Synced {synced} existing signals to stats tracker")

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
        signals = self.tracker.state.get("open_signals", {})
        if not isinstance(signals, dict):
            return 0

        # Normalize symbol for comparison
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

    def _load_watchlist(self) -> List[Dict[str, Any]]:
        """Load and validate watchlist entries (Fix 2.3)."""
        if not WATCHLIST_FILE.exists():
            logger.warning("Watchlist file not found")
            return []
        try:
            raw_watchlist = json.loads(WATCHLIST_FILE.read_text())
        except Exception as e:
            logger.error(f"Failed to load watchlist: {e}")
            return []

        # Validate watchlist entries
        valid_exchanges = ["binanceusdm", "mexc", "bybit"]
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"]
        valid_entries = []
        invalid_count = 0

        for item in raw_watchlist:
            if not isinstance(item, dict):
                logger.warning(f"Invalid watchlist entry (not a dict): {item}")
                invalid_count += 1
                continue

            symbol = item.get("symbol")
            exchange = item.get("exchange", "mexc")
            timeframe = item.get("timeframe", "1h")

            # Validate symbol
            if not symbol or not isinstance(symbol, str):
                logger.warning(f"Invalid symbol in watchlist: {symbol}")
                invalid_count += 1
                continue

            # Validate exchange
            if exchange not in valid_exchanges:
                logger.warning(f"Invalid exchange '{exchange}' for {symbol}, using mexc")
                item["exchange"] = "mexc"

            # Validate timeframe
            if timeframe not in valid_timeframes:
                logger.warning(f"Invalid timeframe '{timeframe}' for {symbol}, using 1h")
                item["timeframe"] = "1h"

            valid_entries.append(item)

        if invalid_count > 0:
            logger.warning(f"Skipped {invalid_count} invalid watchlist entries")

        logger.info(f"Loaded {len(valid_entries)} valid watchlist entries")
        return valid_entries

    def _init_notifier(self) -> Optional[Any]:
        if TelegramNotifier is None: return None
        token = os.getenv("TELEGRAM_BOT_TOKEN_HARMONIC") or os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        return TelegramNotifier(bot_token=token, chat_id=chat_id, signals_log_file=str(LOG_DIR/"signals.json")) if token and chat_id else None

    def run(self, run_once: bool = False, track_only: bool = False) -> None:
        logger.info("Starting Refactored Harmonic Bot...")

        # Get configuration values
        symbol_delay = self.config.get("execution", {}).get("symbol_delay_seconds", 1)
        max_open_signals = self.config.get("risk", {}).get("max_open_signals", 30)
        max_same_symbol = self.config.get("risk", {}).get("max_same_symbol_signals", 2)
        cooldown_minutes = self.config.get("signal", {}).get("cooldown_minutes", 5)
        max_slippage_pct = self.config.get("risk", {}).get("max_slippage_pct", 0.5)
        confidence_threshold = self.config.get("analysis", {}).get("confidence_threshold", 0.0)
        cycle_interval = self.config.get("execution", {}).get("cycle_interval_seconds", 60)

        logger.info(f"Config: Max signals={max_open_signals}, Max/symbol={max_same_symbol}, Cooldown={cooldown_minutes}min")
        logger.info(f"Config: Symbol delay={symbol_delay}s, Max slippage={max_slippage_pct}%, Confidence threshold={confidence_threshold}")

        if self.health_monitor:
            self.health_monitor.send_startup_message()

        # Track-only mode: just check open signals
        if track_only:
            logger.info("Running in track-only mode - checking open signals...")
            self.tracker.check_open_signals(self.clients, self.notifier, self.rate_limiter)
            logger.info("Track-only check complete")
            return

        # Fix 2.2: No-signal heartbeat tracking
        last_signal_time = datetime.now(timezone.utc)
        no_signal_hours = self.config.get("signal", {}).get("no_signal_alert_hours", 6)
        last_heartbeat_time = datetime.now(timezone.utc)

        try:
            while True:
                try:
                    signals_this_cycle = 0  # Track signals per cycle
                    for item in self.watchlist:
                        symbol_raw = item.get("symbol")
                        if not isinstance(symbol_raw, str) or not symbol_raw:
                            continue
                        symbol: str = symbol_raw
                        exchange_raw = item.get("exchange", "mexc")
                        exchange: str = exchange_raw if isinstance(exchange_raw, str) else "mexc"
                        timeframe_raw = item.get("timeframe", "1h")
                        timeframe: str = timeframe_raw if isinstance(timeframe_raw, str) else "1h"

                        # Check exchange backoff (from Volume Bot)
                        if self._backoff_active(exchange):
                            logger.debug(f"Backoff active for {exchange}; skipping {symbol}")
                            continue

                        # 1. Fetch Data (with rate limiting)
                        client = self.clients.get(exchange)
                        if not client: continue

                        try:
                            # Use rate limiter if available
                            if self.rate_limiter:
                                self.rate_limiter.wait_if_needed()

                            ohlcv = cast(List[List[Any]], client.fetch_ohlcv(resolve_symbol(symbol), timeframe, limit=200))
                            ticker = cast(Dict[str, Any], client.fetch_ticker(resolve_symbol(symbol)))
                            price_raw = ticker.get("last")
                            if not isinstance(price_raw, (int, float)):
                                logger.debug(f"Invalid price for {symbol}: {price_raw}")
                                continue
                            price: float = float(price_raw)

                            # Record success if rate limiter is available
                            if self.rate_limiter:
                                self.rate_limiter.record_success(f"{exchange}_{symbol}")
                        except ccxt.RateLimitExceeded as e:
                            logger.warning(f"Rate limit hit for {symbol}, backing off...")
                            self._register_backoff(exchange)
                            if self.health_monitor: self.health_monitor.record_error(f"Rate Limit {symbol}")
                            if self.rate_limiter:
                                self.rate_limiter.record_error(f"{exchange}_{symbol}")
                            continue
                        except ccxt.NetworkError as e:
                            logger.warning(f"Network error for {symbol}: {e}")
                            if self.rate_limiter:
                                self.rate_limiter.record_error(f"{exchange}_{symbol}")
                            continue
                        except Exception as e:
                            logger.error(f"Error fetching {symbol}: {e}")
                            if self._is_rate_limit_error(e):
                                self._register_backoff(exchange)
                            if self.health_monitor: self.health_monitor.record_error(f"API Error {symbol}: {e}")
                            if self.rate_limiter:
                                self.rate_limiter.record_error(f"{exchange}_{symbol}")
                            continue

                        # 2. Analyze
                        signal = self.analyzer.detect(ohlcv, symbol, timeframe, price, exchange)

                        if signal:
                            # 2.3 Confidence threshold check (from Volume Bot)
                            if hasattr(signal, 'confidence') and signal.confidence < confidence_threshold:
                                logger.debug(f"{symbol}: Confidence too low ({signal.confidence:.2f} < {confidence_threshold})")
                                continue

                            # 2.5 Slippage protection (Fix 1.8)
                            if signal.entry > 0:
                                slippage = abs(price - signal.entry) / signal.entry * 100
                                if slippage > max_slippage_pct:
                                    logger.debug(f"{symbol}: Slippage too high ({slippage:.2f}% > {max_slippage_pct}%), skipping")
                                    continue

                            # 3. Check Max Open Signals BEFORE adding
                            current_open = len(self.tracker.state.get("open_signals", {}))
                            if current_open >= max_open_signals:
                                logger.info(f"Max signals limit reached ({current_open}/{max_open_signals}). Skipping {symbol}")
                                continue

                            # 3.5 Check Max Same Symbol Signals (from Volume Bot)
                            symbol_count = self._count_symbol_signals(symbol)
                            if symbol_count >= max_same_symbol:
                                logger.debug(f"{symbol}: Max same symbol signals reached ({symbol_count}/{max_same_symbol})")
                                continue

                            # 4. Check Cooldown
                            if not self.tracker.should_alert(symbol, signal.pattern_name, cooldown_minutes):
                                logger.debug(f"Cooldown active for {symbol} {signal.pattern_name}")
                                continue

                            # 5. Deduplicate (Fix 1.6 - pass exchange and timeframe)
                            if not self.tracker.is_duplicate(symbol, signal.pattern_name, signal.d_timestamp, exchange, timeframe):
                                # 6. Check Reversal
                                self.tracker.check_reversal(symbol, signal.direction, self.notifier)

                                # 7. Alert & Track
                                self.tracker.add_signal(signal)
                                self._send_alert(signal)
                                self.tracker.mark_alert(symbol, signal.pattern_name)
                                signals_this_cycle += 1
                                last_signal_time = datetime.now(timezone.utc)  # Fix 2.2

                        # CRITICAL: Add delay between symbols to prevent rate limiting
                        time.sleep(symbol_delay)

                    # 6. Cleanup stale signals FIRST (before checking prices)
                    max_age_hours = self.config.get("signal", {}).get("max_signal_age_hours", 24)
                    self.tracker.cleanup_stale_signals(max_age_hours)

                    # 7. Monitor Open Signals (after cleanup) - with rate limiting (Fix 2.1)
                    self.tracker.check_open_signals(self.clients, self.notifier, self.rate_limiter)

                    if self.health_monitor:
                        self.health_monitor.record_cycle()

                    # Fix 2.2: No-signal heartbeat notification
                    hours_since_signal = (datetime.now(timezone.utc) - last_signal_time).total_seconds() / 3600
                    hours_since_heartbeat = (datetime.now(timezone.utc) - last_heartbeat_time).total_seconds() / 3600

                    if hours_since_signal >= no_signal_hours and hours_since_heartbeat >= no_signal_hours:
                        if self.notifier:
                            open_signals = len(self.tracker.state.get("open_signals", {}))
                            msg = (
                                f"💓 <b>Harmonic Bot Heartbeat</b>\n\n"
                                f"⏰ No new signals for {hours_since_signal:.1f} hours\n"
                                f"📊 Open signals: {open_signals}\n"
                                f"👁️ Watching: {len(self.watchlist)} symbols\n"
                                f"✅ Bot is running normally"
                            )
                            self.notifier.send_message(msg, parse_mode="HTML")
                            last_heartbeat_time = datetime.now(timezone.utc)
                            logger.info(f"Heartbeat sent - no signals for {hours_since_signal:.1f} hours")

                    if run_once:
                        break
                    time.sleep(cycle_interval)
                    
                except Exception as e:
                    logger.error(f"Cycle error: {e}")
                    if self.health_monitor: self.health_monitor.record_error(f"Cycle Error: {e}")
                    if run_once: raise
                    time.sleep(10)
        finally:
            if self.health_monitor:
                self.health_monitor.send_shutdown_message()

    def _get_symbol_performance(self, symbol: str) -> Dict[str, Any]:
        """Get performance statistics for a specific symbol."""
        if not self.tracker.stats or not isinstance(self.tracker.stats.data, dict):
            return {"total": 0, "wins": 0, "tp1": 0, "tp2": 0, "sl": 0, "avg_pnl": 0.0}

        history = self.tracker.stats.data.get("history", [])
        if not isinstance(history, list):
            return {"total": 0, "wins": 0, "tp1": 0, "tp2": 0, "sl": 0, "avg_pnl": 0.0}

        # Normalize symbol for comparison
        symbol_key = symbol.split(":")[0].upper()

        tp1_count = 0
        tp2_count = 0
        sl_count = 0
        pnl_sum = 0.0
        count = 0

        for entry in history:
            if not isinstance(entry, dict):
                continue

            entry_symbol = entry.get("symbol", "").split(":")[0].upper()
            if entry_symbol != symbol_key:
                continue

            result = entry.get("result", "")
            pnl = entry.get("pnl_pct", 0.0)

            if result == "TP1":
                tp1_count += 1
            elif result == "TP2":
                tp2_count += 1
            elif result == "SL":
                sl_count += 1

            if isinstance(pnl, (int, float)):
                pnl_sum += float(pnl)
                count += 1

        total = tp1_count + tp2_count + sl_count
        wins = tp1_count + tp2_count
        avg_pnl = (pnl_sum / count) if count > 0 else 0.0

        return {
            "total": total,
            "wins": wins,
            "tp1": tp1_count,
            "tp2": tp2_count,
            "sl": sl_count,
            "avg_pnl": avg_pnl
        }

    def _send_alert(self, signal: HarmonicSignal) -> None:
        # Get symbol performance stats
        symbol_stats = self._get_symbol_performance(signal.symbol)

        msg = format_signal_message(
            bot_name="HARMONIC",
            symbol=signal.symbol,
            direction=signal.direction,
            entry=signal.entry,
            stop_loss=signal.stop_loss,
            tp1=signal.take_profit_1,
            tp2=signal.take_profit_2,
            pattern_name=signal.pattern_name,
            exchange=signal.exchange.upper(),
            timeframe=signal.timeframe,
            performance_stats=symbol_stats if symbol_stats.get("total", 0) > 0 else None,
        )
        if self.notifier:
            self.notifier.send_message(msg, parse_mode="HTML")
            logger.info(f"Signal sent: {signal.symbol} {signal.pattern_name} {signal.direction}")

def validate_environment() -> bool:
    """Validate all required environment variables are set (from Volume Bot)."""
    required = {
        "TELEGRAM_BOT_TOKEN": "Telegram bot token (or TELEGRAM_BOT_TOKEN_HARMONIC)",
        "TELEGRAM_CHAT_ID": "Telegram chat ID",
    }

    missing = []

    # Check Telegram (at least one token must exist)
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN_HARMONIC") or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token:
        missing.append("  - TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN_HARMONIC: Telegram bot token")
    if not chat_id:
        missing.append("  - TELEGRAM_CHAT_ID: Telegram chat ID")

    if missing:
        logger.critical("❌ Missing required environment variables:")
        for item in missing:
            logger.critical(item)
        logger.critical("\nPlease create a .env file with these variables.")
        return False

    logger.info("✅ All required environment variables present")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harmonic Pattern Bot - Automated harmonic pattern detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python harmonic_bot.py --once              # Run one cycle
  python harmonic_bot.py --track             # Check open signals only
  python harmonic_bot.py --debug             # Enable debug logging
  python harmonic_bot.py --config custom.json  # Use custom config file
        """
    )
    parser.add_argument("--config", default="harmonic_config.json", help="Path to configuration JSON file")
    parser.add_argument("--once", action="store_true", help="Run only one cycle")
    parser.add_argument("--track", action="store_true", help="Only run tracker checks (from Volume Bot)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip environment validation (not recommended)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level (overrides config)",
    )
    args = parser.parse_args()

    # Load config first to get logging settings
    config_path = BASE_DIR / args.config
    config = load_json_config(config_path)

    # Determine log level (CLI > config > default)
    if args.log_level:
        log_level = args.log_level
    elif args.debug:
        log_level = "DEBUG"
    else:
        log_level = config.get("execution", {}).get("log_level", "INFO")
    enable_detailed = config.get("execution", {}).get("enable_detailed_logging", False)

    global logger
    logger = setup_logging(log_level=log_level, enable_detailed=enable_detailed)

    logger.info("=" * 60)
    logger.info("🤖 Harmonic Pattern Bot Starting...")
    logger.info("=" * 60)
    logger.info(f"📝 Log level: {log_level} | Detailed: {enable_detailed}")
    logger.info(f"📁 Working directory: {BASE_DIR}")
    logger.info(f"⚙️  Config file: {config_path}")

    # Validate environment unless skipped
    if not args.skip_validation:
        if not validate_environment():
            logger.critical("❌ Environment validation failed - exiting")
            logger.critical("Use --skip-validation to bypass (not recommended)")
            sys.exit(1)

    bot = HarmonicBot(config_path)
    bot.run(run_once=args.once, track_only=args.track)


if __name__ == "__main__":
    main()