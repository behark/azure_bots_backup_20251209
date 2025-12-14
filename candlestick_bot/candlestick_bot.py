#!/usr/bin/env python3
"""
Candlestick Pattern Bot - Automated candlestick pattern detection
Detects: Hammer, Shooting Star, Engulfing, Morning/Evening Star, and more
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
from typing import Any, Dict, List, Optional, Tuple, TypedDict, cast

import ccxt

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
STATE_FILE = BASE_DIR / "candlestick_state.json"
WATCHLIST_FILE = BASE_DIR / "candlestick_watchlist.json"
STATS_FILE = LOG_DIR / "candlestick_stats.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "candlestick_bot.log"),
    ],
)

logger = logging.getLogger("candlestick_bot")

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
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        symbol_val = item.get("symbol")
        if not isinstance(symbol_val, str):
            continue
        period_val = item.get("period", "5m")
        cooldown_raw = item.get("cooldown_minutes", 30)
        try:
            cooldown = int(cooldown_raw)
        except Exception:
            cooldown = 30
        normalized.append(
            {
                "symbol": symbol_val.upper(),
                "period": period_val if isinstance(period_val, str) else "5m",
                "cooldown_minutes": cooldown,
            }
        )
    return normalized


def human_ts() -> str:
    """Return human-readable UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Candle:
    """Represents a single candlestick."""
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)
    
    @property
    def upper_wick(self) -> float:
        return self.high - max(self.close, self.open)
    
    @property
    def lower_wick(self) -> float:
        return min(self.close, self.open) - self.low
    
    @property
    def total_range(self) -> float:
        return self.high - self.low
    
    @property
    def is_bullish(self) -> bool:
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        return self.close < self.open


class CandlestickPatternDetector:
    """Detects candlestick patterns."""
    
    def __init__(self):
        self.patterns = {
            "hammer": self._is_hammer,
            "shooting_star": self._is_shooting_star,
            "bullish_engulfing": self._is_bullish_engulfing,
            "bearish_engulfing": self._is_bearish_engulfing,
            "morning_star": self._is_morning_star,
            "evening_star": self._is_evening_star,
        }
    
    def detect(self, candles: List[Candle]) -> Optional[Tuple[str, str]]:
        """Detect patterns from recent candles. Returns (pattern_name, direction)."""
        if len(candles) < 4:
            return None

        current = candles[-2]
        prev1 = candles[-3]
        prev2 = candles[-4]
        
        # Single candle patterns
        if self._is_hammer(current):
            return ("Hammer", "BULLISH")
        if self._is_shooting_star(current):
            return ("Shooting Star", "BEARISH")
        
        # Two candle patterns
        if self._is_bullish_engulfing(prev1, current):
            return ("Bullish Engulfing", "BULLISH")
        if self._is_bearish_engulfing(prev1, current):
            return ("Bearish Engulfing", "BEARISH")
        
        # Three candle patterns
        if self._is_morning_star(prev2, prev1, current):
            return ("Morning Star", "BULLISH")
        if self._is_evening_star(prev2, prev1, current):
            return ("Evening Star", "BEARISH")
        
        return None
    
    def _is_hammer(self, candle: Candle) -> bool:
        """Check if candle is a hammer (bullish)."""
        return (candle.is_bullish and
                candle.lower_wick > candle.body_size * 2 and
                candle.upper_wick < candle.body_size * 0.3)
    
    def _is_shooting_star(self, candle: Candle) -> bool:
        """Check if candle is a shooting star (bearish)."""
        return (candle.is_bearish and
                candle.upper_wick > candle.body_size * 2 and
                candle.lower_wick < candle.body_size * 0.3)
    
    def _is_bullish_engulfing(self, prev: Candle, current: Candle) -> bool:
        """Check if pattern is bullish engulfing."""
        return (prev.is_bearish and current.is_bullish and
                current.open <= prev.close and
                current.close >= prev.open and
                current.body_size > prev.body_size * 0.8 and
                current.volume > prev.volume)
    
    def _is_bearish_engulfing(self, prev: Candle, current: Candle) -> bool:
        """Check if pattern is bearish engulfing."""
        return (prev.is_bullish and current.is_bearish and
                current.open >= prev.close and
                current.close <= prev.open and
                current.body_size > prev.body_size * 0.8 and
                current.volume > prev.volume)
    
    def _is_morning_star(self, first: Candle, middle: Candle, last: Candle) -> bool:
        """Check if pattern is morning star (bullish)."""
        avg_body = (first.body_size + last.body_size) / 2
        return (first.is_bearish and
                middle.body_size < avg_body * 0.5 and
                last.is_bullish and
                last.close > (first.close + first.open) / 2)
    
    def _is_evening_star(self, first: Candle, middle: Candle, last: Candle) -> bool:
        """Check if pattern is evening star (bearish)."""
        avg_body = (first.body_size + last.body_size) / 2
        return (first.is_bullish and
                middle.body_size < avg_body * 0.5 and
                last.is_bearish and
                last.close < (first.close + first.open) / 2)
    
    def calculate_targets(
        self,
        pattern_name: str,
        direction: str,
        candles: List[Candle],
        symbol: str = "",
    ) -> Dict[str, float]:
        """Calculate entry, stop loss, and take profit levels."""
        current = candles[-1]
        entry = current.close
        
        # Calculate structure-based SL
        if direction == "BULLISH":
            if pattern_name == "Hammer":
                structure_sl = current.low - (current.total_range * 0.1)
            else:
                structure_sl = min(c.low for c in candles[-3:]) * 0.99
            swing_low = structure_sl
            swing_high = None
        else:
            if pattern_name == "Shooting Star":
                structure_sl = current.high + (current.total_range * 0.1)
            else:
                structure_sl = max(c.high for c in candles[-3:]) * 1.005
            swing_high = structure_sl
            swing_low = None
        
        if TPSLCalculator is not None:
            from tp_sl_calculator import CalculationMethod
            if get_config_manager is not None:
                config_mgr = get_config_manager()
                risk_config = config_mgr.get_effective_risk("candlestick_bot", symbol)
                tp1_mult = risk_config.tp1_atr_multiplier
                tp2_mult = risk_config.tp2_atr_multiplier
                min_rr = risk_config.min_risk_reward
            else:
                tp1_mult, tp2_mult, min_rr = 1.5, 2.5, 1.5

            calculator = TPSLCalculator(min_risk_reward=min_rr)
            levels = calculator.calculate(
                entry=entry,
                direction=direction,
                swing_high=swing_high,
                swing_low=swing_low,
                tp1_multiplier=tp1_mult,
                tp2_multiplier=tp2_mult,
                method=CalculationMethod.STRUCTURE,
            )

            if levels.is_valid:
                return {
                    "entry": entry,
                    "stop_loss": levels.stop_loss,
                    "take_profit_1": levels.take_profit_1,
                    "take_profit_2": levels.take_profit_2,
                }
        
        # Fallback to original calculation
        sl = structure_sl
        risk = abs(entry - sl)
        if direction == "BULLISH":
            tp1 = entry + risk * 1.5
            tp2 = entry + risk * 2.5
        else:
            tp1 = entry - risk * 1.5
            tp2 = entry - risk * 2.5
        
        return {
            "entry": entry,
            "stop_loss": sl,
            "take_profit_1": tp1,
            "take_profit_2": tp2,
        }


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
class CandlestickSignal:
    """Represents a candlestick pattern signal."""
    symbol: str
    pattern_name: str
    direction: str
    timestamp: str
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    current_price: Optional[float] = None


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
        if not isinstance(data.get("last_alert"), dict):
            data["last_alert"] = {}
        if not isinstance(data.get("open_signals"), dict):
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


class CandlestickBot:
    """Main Candlestick Pattern Bot."""
    
    def __init__(self, interval: int = 300, default_cooldown: int = 30):
        self.interval = interval
        self.default_cooldown = default_cooldown
        self.watchlist: List[WatchItem] = load_watchlist()
        self.client = MexcClient()
        self.detector = CandlestickPatternDetector()
        self.state = BotState(STATE_FILE)
        self.notifier = self._init_notifier()
        self.stats = SignalStats("Candlestick Bot", STATS_FILE) if SignalStats else None
        self.health_monitor = (
            HealthMonitor("Candlestick Bot", self.notifier, heartbeat_interval=3600)
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
        token = os.getenv("TELEGRAM_BOT_TOKEN_CANDLESTICK") or os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            logger.warning("Telegram credentials missing")
            return None
        return TelegramNotifier(
            bot_token=token,
            chat_id=chat_id,
            signals_log_file=str(LOG_DIR / "candlestick_signals.json")
        )
    
    def run(self, loop: bool = False) -> None:
        if not self.watchlist:
            logger.error("Empty watchlist; exiting")
            return
        
        logger.info("Starting Candlestick Bot for %d symbols", len(self.watchlist))
        
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
            cooldown_raw = item.get("cooldown_minutes", self.default_cooldown) if isinstance(item, dict) else self.default_cooldown
            period = period_val if isinstance(period_val, str) else "5m"
            try:
                cooldown = int(cooldown_raw)
            except Exception:
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
            signal_id = f"{symbol}-{signal.pattern_name.replace(' ', '')}-{signal.timestamp}"
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
    
    def _analyze_symbol(self, symbol: str, timeframe: str) -> Optional[CandlestickSignal]:
        """Analyze symbol for candlestick patterns."""
        # Fetch OHLCV data
        ohlcv = self.client.fetch_ohlcv(symbol, timeframe, limit=50)
        
        if len(ohlcv) < 10:
            return None
        
        try:
            candles = [
                Candle(
                    open=float(x[1]),
                    high=float(x[2]),
                    low=float(x[3]),
                    close=float(x[4]),
                    volume=float(x[5]),
                )
                for x in ohlcv
            ]
        except Exception:
            return None

        closes = [c.close for c in candles]
        sma_period = 50
        if len(closes) < sma_period:
            return None
        sma_50 = sum(closes[-sma_period:]) / sma_period
        signal_candle = candles[-2]
        
        # Detect pattern
        result = self.detector.detect(candles)
        
        if result is None:
            return None
        
        pattern_name, direction = result

        if direction == "BULLISH" and signal_candle.close <= sma_50:
            return None
        if direction == "BEARISH" and signal_candle.close >= sma_50:
            return None
        
        # Calculate targets
        targets = self.detector.calculate_targets(pattern_name, direction, candles, symbol)
        
        # Get current price
        ticker = self.client.fetch_ticker(symbol)
        current_price = ticker.get("last") if isinstance(ticker, dict) else None
        if current_price is None and isinstance(ticker, dict):
            current_price = ticker.get("close")
        if not isinstance(current_price, (int, float)):
            return None
        entry = float(targets["entry"])
        return CandlestickSignal(
            symbol=symbol,
            pattern_name=pattern_name,
            direction=direction,
            timestamp=human_ts(),
            entry=entry,
            stop_loss=float(targets["stop_loss"]),
            take_profit_1=float(targets["take_profit_1"]),
            take_profit_2=float(targets["take_profit_2"]),
            current_price=float(current_price),
        )
    
    def _format_message(self, signal: CandlestickSignal) -> str:
        """Format Telegram message for signal."""
        direction = signal.direction
        emoji = "🟢" if direction == "BULLISH" else "🔴"
        
        current_price = signal.current_price if isinstance(signal.current_price, (int, float)) else None
        lines = [
            f"{emoji} <b>{direction} {signal.pattern_name.upper()} - {signal.symbol}/USDT</b>",
            "",
            f"📊 <b>Pattern:</b> {signal.pattern_name}",
        ]
        if current_price is not None:
            lines.append(f"💰 <b>Current Price:</b> <code>{current_price:.6f}</code>")
        lines.extend([
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
    parser = argparse.ArgumentParser(description="Candlestick Pattern Bot")
    parser.add_argument("--loop", action="store_true", help="Run indefinitely")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between cycles")
    parser.add_argument("--cooldown", type=int, default=30, help="Default cooldown minutes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bot = CandlestickBot(interval=args.interval, default_cooldown=args.cooldown)
    bot.run(loop=args.loop)


if __name__ == "__main__":
    main()
