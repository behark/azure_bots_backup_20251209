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
from typing import Any, Dict, List, Optional, TypedDict, cast
import ccxt
import numpy as np

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
    RateLimiter = None
    TPSLCalculator = None
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
    def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA."""
        ema = np.zeros(len(prices))
        if len(prices) < period:
            return ema
        multiplier = 2 / (period + 1)
        ema[period-1] = np.mean(prices[:period])
        for i in range(period, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema
    
    def calculate_most(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> tuple:
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
    
    def detect_signal(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Optional[tuple]:
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
        self.exchange = ccxt.mexc({
            "enableRateLimit": True,
            "options": {"defaultType": "swap"}
        })  # type: ignore[arg-type]
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
            if not isinstance(data, dict):
                return self._empty_state()
            last_alert = data.get("last_alert")
            open_signals = data.get("open_signals")
            return {
                "last_alert": last_alert if isinstance(last_alert, dict) else {},
                "open_signals": open_signals if isinstance(open_signals, dict) else {},
            }
        except json.JSONDecodeError:
            return self._empty_state()
    
    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2))
    
    def can_alert(self, symbol: str, cooldown_minutes: int) -> bool:
        last_map = self.data.setdefault("last_alert", {})
        if not isinstance(last_map, dict):
            self.data["last_alert"] = {}
            return True
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
    
    def __init__(self, interval: int = 300, default_cooldown: int = 30):
        self.interval = interval
        self.default_cooldown = default_cooldown
        self.watchlist: List[WatchItem] = load_watchlist()
        self.client = MexcClient()
        self.analyzer = MOSTAnalyzer()
        self.state = BotState(STATE_FILE)
        self.notifier = self._init_notifier()
        self.stats = SignalStats("MOST Bot", STATS_FILE) if SignalStats else None
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
    
    def _init_notifier(self):
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
        
        if TPSLCalculator is not None:
            if get_config_manager is not None:
                config_mgr = get_config_manager()
                risk_config = config_mgr.get_effective_risk("most_bot", symbol)
                tp1_mult = risk_config.tp1_atr_multiplier
                tp2_mult = risk_config.tp2_atr_multiplier
                min_rr = risk_config.min_risk_reward
            else:
                tp1_mult, tp2_mult, min_rr = 2.0, 3.5, 1.5

            calculator = TPSLCalculator(min_risk_reward=min_rr)
            levels = calculator.calculate(
                entry=entry,
                direction=direction,
                atr=atr,
                tp1_multiplier=tp1_mult,
                tp2_multiplier=tp2_mult,
                custom_sl=most_value,  # Use MOST as trailing stop
            )

            if levels.is_valid:
                sl = most_value  # Keep MOST as trailing stop
                tp1 = levels.take_profit_1
                tp2 = levels.take_profit_2
            else:
                logger.info("Signal rejected for %s: %s", symbol, levels.rejection_reason)
                return None
        else:
            sl = most_value  # Use MOST as trailing stop
            if direction == "BULLISH":
                tp1 = entry + (atr * 2.0)
                tp2 = entry + (atr * 3.5)
            else:
                tp1 = entry - (atr * 2.0)
                tp2 = entry - (atr * 3.5)
        
        return MOSTSignal(
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
    
    def _format_message(self, signal: MOSTSignal) -> str:
        """Format Telegram message for signal."""
        direction = signal.direction
        emoji = "🟢" if direction == "BULLISH" else "🔴"
        perf_line = self._symbol_perf_line(signal.symbol)

        lines = [
            f"{emoji} <b>{direction} MOST SIGNAL - {signal.symbol}/USDT</b>",
            "",
            "📊 <b>Strategy:</b> MOST (Moving Stop Loss)",
            f"💰 <b>Current Price:</b> <code>{signal.current_price:.6f}</code>" if signal.current_price is not None else "💰 <b>Current Price:</b> N/A",
            f"📍 <b>MOST Level:</b> <code>{signal.most_value:.6f}</code>",
        ]

        if perf_line:
            lines.append(perf_line)

        lines.extend([
            "",
            "<b>🎯 TRADE LEVELS:</b>",
            f"🟢 Entry: <code>{signal.entry:.6f}</code>",
            f"🛑 Stop Loss (MOST): <code>{signal.stop_loss:.6f}</code>",
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
        
        lines.extend([
            "",
            "💡 <b>Note:</b> MOST acts as trailing stop - adjust as it moves",
            f"⏱️ {signal.timestamp}",
        ])
        
        return "\n".join(lines)

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
                        f"🎯 {symbol}/USDT MOST {direction} {result} hit!\n"
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
    parser = argparse.ArgumentParser(description="MOST Bot")
    parser.add_argument("--loop", action="store_true", help="Run indefinitely")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between cycles")
    parser.add_argument("--cooldown", type=int, default=30, help="Default cooldown minutes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bot = MOSTBot(interval=args.interval, default_cooldown=args.cooldown)
    bot.run(loop=args.loop)


if __name__ == "__main__":
    main()
