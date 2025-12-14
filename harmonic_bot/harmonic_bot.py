#!/usr/bin/env python3
# pyright: ignore-all
"""
Harmonic Pattern Bot - Automated harmonic pattern detection and alerts
Detects: Bat, Butterfly, Gartley, Crab, Shark, ABCD patterns and more
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import ccxt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
STATE_FILE = BASE_DIR / "harmonic_state.json"
WATCHLIST_FILE = BASE_DIR / "harmonic_watchlist.json"
STATS_FILE = LOG_DIR / "harmonic_stats.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "harmonic_bot.log"),
    ],
)

logger = logging.getLogger("harmonic_bot")

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
    from tp_sl_calculator import TPSLCalculator
    from trade_config import get_config_manager
    from rate_limit_handler import RateLimitHandler
except ImportError:
    TelegramNotifier = None
    SignalStats = None
    HealthMonitor = None
    RateLimitHandler = None
    RateLimiter = None
    TPSLCalculator = None
    get_config_manager = None


def load_watchlist() -> List[Dict[str, object]]:
    """Load watchlist from JSON file."""
    if not WATCHLIST_FILE.exists():
        logger.error("Watchlist file missing: %s", WATCHLIST_FILE)
        return []
    
    try:
        data = json.loads(WATCHLIST_FILE.read_text())
    except json.JSONDecodeError as exc:
        logger.error("Invalid watchlist JSON: %s", exc)
        return []
    
    normalized = []
    for item in data:
        normalized.append({
            "symbol": item["symbol"].upper(),
            "period": item.get("period", "5m"),
            "cooldown_minutes": int(item.get("cooldown_minutes", 30)),
        })
    return normalized


def human_ts() -> str:
    """Return human-readable UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class HarmonicPatternDetector:
    """Detects harmonic patterns from price data."""
    
    # Pattern definitions (XAB, ABC, BCD, XAD ranges)
    PATTERNS = {
        "Cypher": {
            "xab": (0.382, 0.618),
            "abc": (1.13, 1.414),
            "bcd": (1.272, 2.0),
            "xad": (0.76, 0.80),
        },
        "Deep Crab": {
            "xab": (0.88, 0.895),
            "abc": (0.382, 0.886),
            "bcd": (2.0, 3.618),
            "xad": (1.60, 1.63),
        },
        "Bat": {
            "xab": (0.382, 0.5),
            "abc": (0.382, 0.886),
            "bcd": (1.618, 2.618),
            "xad": (0.0, 0.886),
        },
        "Butterfly": {
            "xab": (0.0, 0.786),
            "abc": (0.382, 0.886),
            "bcd": (1.618, 2.618),
            "xad": (1.27, 1.618),
        },
        "Gartley": {
            "xab": (0.61, 0.625),
            "abc": (0.382, 0.886),
            "bcd": (1.13, 2.618),
            "xad": (0.75, 0.875),
        },
        "Crab": {
            "xab": (0.5, 0.875),
            "abc": (0.382, 0.886),
            "bcd": (2.0, 5.0),
            "xad": (1.382, 5.0),
        },
        "Shark": {
            "xab": (0.5, 0.875),
            "abc": (1.13, 1.618),
            "bcd": (1.27, 2.24),
            "xad": (0.886, 1.13),
        },
        "ABCD": {
            "xab": (0.0, 999.0),  # Not used for ABCD
            "abc": (0.382, 0.886),
            "bcd": (1.13, 2.618),
            "xad": (0.0, 999.0),  # Not used for ABCD
        },
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
    
    def find_zigzag_pivots(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        opens: List[float],
        atr_value: float,
    ) -> List[Tuple[int, float, str]]:
        """Find ZigZag pivot points."""
        highs_arr = np.asarray(highs, dtype=float)
        lows_arr = np.asarray(lows, dtype=float)
        closes_arr = np.asarray(closes, dtype=float)
        opens_arr = np.asarray(opens, dtype=float)

        n = closes_arr.size
        if n < 2:
            return []

        atr_threshold = max(atr_value * 2.0, 1e-8)

        # 1 for bullish candle, -1 for bearish (ties treated as bullish to match prior behavior)
        candle_dirs = np.where(closes_arr >= opens_arr, 1, -1)
        change_points = np.flatnonzero(candle_dirs[1:] != candle_dirs[:-1]) + 1
        if change_points.size == 0:
            return []

        segments = np.split(np.arange(n, dtype=int), change_points)
        # Remove potential empty segments (shouldn't happen but keeps logic safe)
        segments = [seg for seg in segments if seg.size > 0]
        if len(segments) < 2:
            return []

        segment_dirs = [int(candle_dirs[seg[0]]) for seg in segments]
        pivots: List[Tuple[int, float, str]] = []

        for seg_idx in range(1, len(segments)):
            prev_seg = segments[seg_idx - 1]
            prev_dir = segment_dirs[seg_idx - 1]

            if prev_dir == 1:
                # Up-segment completed -> pivot high
                local_idx = prev_seg[np.argmax(highs_arr[prev_seg])]
                price = float(highs_arr[local_idx])
                pivot_type = 'H'
            else:
                # Down-segment completed -> pivot low
                local_idx = prev_seg[np.argmin(lows_arr[prev_seg])]
                price = float(lows_arr[local_idx])
                pivot_type = 'L'

            if not pivots or abs(price - pivots[-1][1]) >= atr_threshold:
                pivots.append((int(local_idx), price, pivot_type))

        return pivots

    @staticmethod
    def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        if len(closes) < period + 1:
            return float(np.mean(np.array(highs) - np.array(lows))) if highs and lows else 0.0
        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            trs.append(tr)
        if not trs:
            return 0.0
        if len(trs) >= period:
            return float(np.mean(trs[-period:]))
        return float(np.mean(trs))
    
    def get_xabcd_ratios(
        self,
        x: float,
        a: float,
        b: float,
        c: float,
        d: float,
    ) -> Tuple[float, float, float, float]:
        """Calculate XABCD Fibonacci ratios."""
        xab = abs(b - a) / abs(x - a) if abs(x - a) != 0 else 0
        xad = abs(a - d) / abs(x - a) if abs(x - a) != 0 else 0
        abc = abs(b - c) / abs(a - b) if abs(a - b) != 0 else 0
        bcd = abs(c - d) / abs(b - c) if abs(b - c) != 0 else 0
        
        return xab, xad, abc, bcd
    
    def detect_patterns(
        self,
        x: float,
        a: float,
        b: float,
        c: float,
        d: float,
    ) -> List[Tuple[str, str, float]]:
        """Detect harmonic patterns based on Fibonacci ratios with error scoring."""
        xab, xad, abc, bcd = self.get_xabcd_ratios(x, a, b, c, d)
        direction = "BULLISH" if d < c else "BEARISH"

        priority_order = [
            "Cypher",
            "Shark",
            "Deep Crab",
            "Butterfly",
            "Bat",
            "Crab",
            "Gartley",
            "ABCD",
        ]

        for pattern_name in priority_order:
            ranges = self.PATTERNS.get(pattern_name)
            if not ranges:
                continue

            xab_min, xab_max = ranges["xab"]
            abc_min, abc_max = ranges["abc"]
            bcd_min, bcd_max = ranges["bcd"]
            xad_min, xad_max = ranges["xad"]

            if not (
                xab_min <= xab <= xab_max
                and abc_min <= abc <= abc_max
                and bcd_min <= bcd <= bcd_max
                and xad_min <= xad <= xad_max
            ):
                continue

            ideals = self.PATTERN_IDEALS.get(pattern_name, {})
            error_score = 0.0
            for key, actual in {
                "xab": xab,
                "abc": abc,
                "bcd": bcd,
                "xad": xad,
            }.items():
                ideal_val = ideals.get(key)
                if ideal_val is not None:
                    error_score += abs(actual - ideal_val)

            if error_score > 0.15:
                continue

            return [(pattern_name, direction, error_score)]

        return []
    
    def calculate_targets(
        self,
        c: float,
        d: float,
        direction: str,
        symbol: str = "",
    ) -> Dict[str, float]:
        """Calculate entry, stop loss, and take profit levels."""
        fib_range = abs(d - c)
        
        if direction == "BULLISH":
            entry = d + (fib_range * 0.236)
            swing_low = d
        else:
            entry = d - (fib_range * 0.236)
            swing_high = d
        
        if TPSLCalculator is not None:
            from tp_sl_calculator import CalculationMethod
            if get_config_manager is not None:
                config_mgr = get_config_manager()
                risk_config = config_mgr.get_effective_risk("harmonic_bot", symbol)
                min_rr = risk_config.min_risk_reward
            else:
                min_rr = 2.0
            
            calculator = TPSLCalculator(min_risk_reward=min_rr)
            levels = calculator.calculate(
                entry=entry,
                direction=direction,
                swing_high=d if direction == "BEARISH" else None,
                swing_low=d if direction == "BULLISH" else None,
                method=CalculationMethod.FIBONACCI,
            )
            
            if levels.is_valid:
                tp3_val = levels.take_profit_3 or levels.take_profit_2 or levels.take_profit_1
                return {
                    "entry": float(entry),
                    "stop_loss": float(levels.stop_loss),
                    "take_profit_1": float(levels.take_profit_1),
                    "take_profit_2": float(levels.take_profit_2),
                    "take_profit_3": float(tp3_val),
                }
        
        # Fallback to original calculation
        if direction == "BULLISH":
            sl = d * 0.98
            tp1 = d + (fib_range * 0.618)
            tp2 = d + (fib_range * 1.0)
            tp3 = d + (fib_range * 1.618)
        else:
            sl = d * 1.02
            tp1 = d - (fib_range * 0.618)
            tp2 = d - (fib_range * 1.0)
            tp3 = d - (fib_range * 1.618)
        
        return {
            "entry": entry,
            "stop_loss": sl,
            "take_profit_1": tp1,
            "take_profit_2": tp2,
            "take_profit_3": tp3,
        }


class MexcClient:
    """MEXC exchange client wrapper."""
    
    def __init__(self):
        self.exchange = ccxt.mexc({  # type: ignore[arg-type]
            "enableRateLimit": True,
            "options": {"defaultType": "swap"}
        })
        self.exchange.load_markets()
        self.rate_limiter = RateLimitHandler(base_delay=0.5, max_retries=5) if RateLimitHandler else None
    
    @staticmethod
    def _swap_symbol(symbol: str) -> str:
        return f"{symbol.upper()}/USDT:USDT"
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = "5m", limit: int = 300) -> list:
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


@dataclass
class HarmonicSignal:
    """Represents a harmonic pattern signal."""
    symbol: str
    pattern_name: str
    direction: str
    timestamp: str
    x: float
    a: float
    b: float
    c: float
    d: float
    xab: float
    abc: float
    bcd: float
    xad: float
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: Optional[float] = None
    current_price: Optional[float] = None
    error_score: float = 0.0


class BotState:
    """Manages bot state persistence."""
    
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, Dict[str, object]] = self._load()
    
    @staticmethod
    def _parse_ts(value: str) -> datetime:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    
    def _load(self) -> Dict[str, Dict[str, object]]:
        if not self.path.exists():
            return {"last_alert": {}, "open_signals": {}}
        try:
            return json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return {"last_alert": {}, "open_signals": {}}
    
    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2))
    
    def can_alert(self, symbol: str, cooldown_minutes: int) -> bool:
        last_map = cast(Dict[str, str], self.data.setdefault("last_alert", {}))
        last_ts = last_map.get(symbol)
        if not last_ts:
            return True
        delta = datetime.now(timezone.utc) - self._parse_ts(last_ts)
        return delta >= timedelta(minutes=cooldown_minutes)
    
    def mark_alert(self, symbol: str) -> None:
        cast(Dict[str, str], self.data.setdefault("last_alert", {}))[symbol] = datetime.now(timezone.utc).isoformat()
        self.save()
    
    def add_signal(self, signal_id: str, payload: Dict[str, object]) -> None:
        cast(Dict[str, Dict[str, object]], self.data.setdefault("open_signals", {}))[signal_id] = payload
        self.save()
    
    def remove_signal(self, signal_id: str) -> None:
        signals = cast(Dict[str, Dict[str, object]], self.data.setdefault("open_signals", {}))
        if signal_id in signals:
            signals.pop(signal_id)
            self.save()
    
    def iter_signals(self) -> Dict[str, Dict[str, object]]:
        return cast(Dict[str, Dict[str, object]], self.data.setdefault("open_signals", {}))


class HarmonicBot:
    """Main Harmonic Pattern Bot."""
    
    def __init__(self, interval: int = 300, default_cooldown: int = 30):
        self.interval = interval
        self.default_cooldown = default_cooldown
        self.watchlist = load_watchlist()
        self.client = MexcClient()
        self.detector = HarmonicPatternDetector()
        self.state = BotState(STATE_FILE)
        self.notifier = self._init_notifier()
        self.stats = SignalStats("Harmonic Bot", STATS_FILE) if SignalStats else None
        self.health_monitor = (
            HealthMonitor("Harmonic Bot", self.notifier, heartbeat_interval=3600)
            if HealthMonitor else None
        )
        self.rate_limiter = (
            RateLimiter(calls_per_minute=60, backoff_file=LOG_DIR / "rate_limiter.json")
            if RateLimiter else None
        )
    
    def _init_notifier(self):
        if TelegramNotifier is None:
            logger.warning("Telegram notifier unavailable")
            return None
        token = os.getenv("TELEGRAM_BOT_TOKEN_HARMONIC") or os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            logger.warning("Telegram credentials missing")
            return None
        return TelegramNotifier(
            bot_token=token,
            chat_id=chat_id,
            signals_log_file=str(LOG_DIR / "harmonic_signals.json")
        )
    
    def run(self, loop: bool = False) -> None:
        if not self.watchlist:
            logger.error("Empty watchlist; exiting")
            return
        
        logger.info("Starting Harmonic Pattern Bot for %d symbols", len(self.watchlist))
        
        if self.health_monitor:
            self.health_monitor.send_startup_message()
        
        try:
            while True:
                try:
                    self._run_cycle()
                    self._monitor_open_signals()
                    
                    if self.health_monitor:
                        self.health_monitor.record_cycle()
                    
                    if not loop:
                        break
                    logger.info("Cycle complete; sleeping %ds", self.interval)
                    time.sleep(self.interval)
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
            period_val = item.get("period", "5m") if isinstance(item, dict) else "5m"
            period = period_val if isinstance(period_val, str) else "5m"
            cooldown_raw = item.get("cooldown_minutes", self.default_cooldown) if isinstance(item, dict) else self.default_cooldown
            if isinstance(cooldown_raw, (int, float, str)):
                try:
                    cooldown = int(cooldown_raw)
                except Exception:
                    cooldown = self.default_cooldown
            else:
                cooldown = self.default_cooldown
            
            if self.rate_limiter:
                self.rate_limiter.wait_if_needed()
            
            try:
                signal = self._analyze_symbol(symbol, period)
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
                logger.debug("%s: No pattern detected", symbol)
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
            signal_id = f"{symbol}-{signal.pattern_name}-{signal.timestamp}"
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
                "exchange": "MEXC",
                "error_score": signal.error_score,
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
                        "pattern": signal.pattern_name,
                    },
                )
            
            time.sleep(0.5)
    
    def _analyze_symbol(self, symbol: str, timeframe: str) -> Optional[HarmonicSignal]:
        """Analyze symbol for harmonic patterns."""
        # Fetch OHLCV data
        ohlcv = self.client.fetch_ohlcv(symbol, timeframe, limit=300)
        
        if len(ohlcv) < 100:
            return None
        
        opens = [x[1] for x in ohlcv]
        highs = [x[2] for x in ohlcv]
        lows = [x[3] for x in ohlcv]
        closes = [x[4] for x in ohlcv]
        
        atr_value = self.detector.calculate_atr(highs, lows, closes)
        if atr_value <= 0:
            atr_value = max(np.std(closes[-50:]), 1e-6)
        # Find ZigZag pivots
        pivots = self.detector.find_zigzag_pivots(highs, lows, closes, opens, atr_value)
        
        if len(pivots) < 5:
            return None
        
        # Get last 5 pivots for XABCD
        x_idx, x, _ = pivots[-5]
        a_idx, a, _ = pivots[-4]
        b_idx, b, _ = pivots[-3]
        c_idx, c, _ = pivots[-2]
        d_idx, d, _ = pivots[-1]
        
        # Detect patterns
        patterns = self.detector.detect_patterns(x, a, b, c, d)
        
        if not patterns:
            return None
        
        # Use first (strongest) pattern
        pattern_name, direction, error_score = patterns[0]
        
        # Calculate ratios
        xab, xad, abc, bcd = self.detector.get_xabcd_ratios(x, a, b, c, d)
        
        # Calculate targets
        targets = self.detector.calculate_targets(c, d, direction)
        
        # Get current price
        ticker = self.client.fetch_ticker(symbol)
        current_price = ticker.get("last") or ticker.get("close")
        
        return HarmonicSignal(
            symbol=symbol,
            pattern_name=pattern_name,
            direction=direction,
            timestamp=human_ts(),
            x=x, a=a, b=b, c=c, d=d,
            xab=xab, abc=abc, bcd=bcd, xad=xad,
            entry=targets["entry"],
            stop_loss=targets["stop_loss"],
            take_profit_1=targets["take_profit_1"],
            take_profit_2=targets["take_profit_2"],
            take_profit_3=targets.get("take_profit_3"),
            current_price=current_price,
            error_score=error_score,
        )
    
    def _format_message(self, signal: HarmonicSignal) -> str:
        """Format Telegram message for signal."""
        direction = signal.direction
        emoji = "🟢" if direction == "BULLISH" else "🔴"
        perf_line = self._symbol_perf_line(signal.symbol)

        lines = [
            f"{emoji} <b>{direction} {signal.pattern_name.upper()} PATTERN - {signal.symbol}/USDT</b>",
            "",
            f"📐 <b>Pattern:</b> {signal.pattern_name}",
            f"💰 <b>Current Price:</b> <code>{signal.current_price:.6f}</code>",
        ]

        if perf_line:
            lines.append(perf_line)
            lines.append("")

        lines.extend([
            "<b>🎯 TRADE LEVELS:</b>",
            f"🟢 Entry: <code>{signal.entry:.6f}</code>",
            f"🛑 Stop Loss: <code>{signal.stop_loss:.6f}</code>",
            f"🎯 TP1: <code>{signal.take_profit_1:.6f}</code>",
            f"🚀 TP2: <code>{signal.take_profit_2:.6f}</code>",
        ])
        
        if signal.take_profit_3:
            lines.append(f"🌟 TP3: <code>{signal.take_profit_3:.6f}</code>")
        
        # Calculate R:R
        risk = abs(signal.entry - signal.stop_loss)
        reward1 = abs(signal.take_profit_1 - signal.entry)
        reward2 = abs(signal.take_profit_2 - signal.entry)
        
        if risk > 0:
            rr1 = reward1 / risk
            rr2 = reward2 / risk
            lines.append("")
            lines.append(f"⚖️ <b>Risk/Reward:</b> 1:{rr1:.2f} (TP1) | 1:{rr2:.2f} (TP2)")
        
        lines.extend([
            "",
            "<b>📊 FIBONACCI RATIOS:</b>",
            f"XAB: <code>{signal.xab:.3f}</code> | ABC: <code>{signal.abc:.3f}</code>",
            f"BCD: <code>{signal.bcd:.3f}</code> | XAD: <code>{signal.xad:.3f}</code>",
            f"🎯 Pattern Error Score: <code>{signal.error_score:.3f}</code>",
            "",
            f"⏱️ {signal.timestamp}",
        ])
        
        return "\n".join(lines)

    def _symbol_perf_line(self, symbol: str) -> Optional[str]:
        if not self.stats:
            return None
        symbol_key = symbol if "/" in symbol else f"{symbol}/USDT"
        counts = self.stats.symbol_tp_sl_counts(symbol_key)
        tp1 = counts.get("TP1", 0)
        tp2 = counts.get("TP2", 0)
        sl = counts.get("SL", 0)
        total = tp1 + tp2 + sl
        if total == 0:
            return None
        win_rate = (tp1 + tp2) / total * 100.0
        return (
            f"📈 <b>History:</b> TP1 {tp1} | TP2 {tp2} | SL {sl} "
            f"(Win rate: {win_rate:.1f}%)"
        )
    
    def _monitor_open_signals(self) -> None:
        """Monitor open signals for TP/SL hits."""
        signals = self.state.iter_signals()
        if not signals:
            return
        
        for signal_id, payload in list(signals.items()):
            symbol_val = payload.get("symbol") if isinstance(payload, dict) else None
            if not isinstance(symbol_val, str):
                self.state.remove_signal(signal_id)
                continue
            symbol = symbol_val
            
            try:
                ticker = self.client.fetch_ticker(symbol)
                price = ticker.get("last") or ticker.get("close")
            except Exception as exc:
                logger.warning("Failed to fetch ticker for %s: %s", signal_id, exc)
                continue
            
            if price is None:
                continue
            
            direction = payload.get("direction")
            entry = payload.get("entry")
            tp1 = payload.get("take_profit_1")
            tp2 = payload.get("take_profit_2")
            sl = payload.get("stop_loss")
            
            if None in (direction, entry, tp1, tp2, sl):
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
                    pattern = payload.get("pattern", "Pattern")
                    message = (
                        f"🎯 {symbol}/USDT {pattern} {direction} {result} hit!\n"
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
    parser = argparse.ArgumentParser(description="Harmonic Pattern Bot")
    parser.add_argument("--loop", action="store_true", help="Run indefinitely")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between cycles")
    parser.add_argument("--cooldown", type=int, default=30, help="Default cooldown minutes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bot = HarmonicBot(interval=args.interval, default_cooldown=args.cooldown)
    bot.run(loop=args.loop)


if __name__ == "__main__":
    main()
