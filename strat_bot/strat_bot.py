#!/usr/bin/env python3
"""
STRAT Bot - Rob Smith's 1-2-3 Price Action System
Detects inside bars (1), directional bars (2u/2d), outside bars (3),
and actionable STRAT combos for high-probability trades
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
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, cast

import ccxt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
STATE_FILE = BASE_DIR / "strat_state.json"
WATCHLIST_FILE = BASE_DIR / "strat_watchlist.json"
STATS_FILE = LOG_DIR / "strat_stats.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "strat_bot.log"),
    ],
)

logger = logging.getLogger("strat_bot")

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
        symbol_val = item.get("symbol") if isinstance(item, dict) else None
        if not isinstance(symbol_val, str):
            continue
        period_val = item.get("period", "5m") if isinstance(item, dict) else "5m"
        period = period_val if isinstance(period_val, str) else "5m"
        cooldown_val = item.get("cooldown_minutes", 30) if isinstance(item, dict) else 30
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


class STRATAnalyzer:
    """Analyzes STRAT 1-2-3 bar patterns and combos."""
    
    @staticmethod
    def get_bar_type(high: float, low: float, prev_high: float, prev_low: float) -> str:
        """
        Determine bar type:
        1 = Inside bar (high <= prev_high AND low >= prev_low)
        2u = 2 Up (high > prev_high, no lower low)
        2d = 2 Down (low < prev_low, no higher high)
        3 = Outside bar (high > prev_high AND low < prev_low)
        """
        is_higher_high = high > prev_high
        is_lower_low = low < prev_low
        
        if is_higher_high and is_lower_low:
            return '3'
        elif is_higher_high and not is_lower_low:
            return '2u'
        elif is_lower_low and not is_higher_high:
            return '2d'
        else:
            return '1'
    
    def classify_candles(self, highs: np.ndarray, lows: np.ndarray) -> List[str]:
        """Classify all candles as 1, 2u, 2d, or 3."""
        bar_types = ['1']  # First bar is neutral
        
        for i in range(1, len(highs)):
            bar_type = self.get_bar_type(highs[i], lows[i], highs[i-1], lows[i-1])
            bar_types.append(bar_type)
        
        return bar_types
    
    def detect_actionable_patterns(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> Optional[Tuple[str, str]]:
        """Detect actionable Hammer/Shooter patterns using last closed bar."""
        if len(closes) < 3:
            return None
        current = len(closes) - 2
        candle_range = highs[current] - lows[current]
        if candle_range == 0:
            return None
        
        wick_threshold = candle_range * 0.75
        shooter_top = highs[current] - wick_threshold
        hammer_bottom = lows[current] + wick_threshold
        
        # Shooter: open and close in bottom 25% (bearish)
        if opens[current] < shooter_top and closes[current] < shooter_top:
            return ('SHOOTER', 'BEARISH')
        
        # Hammer: open and close in top 25% (bullish)
        if opens[current] > hammer_bottom and closes[current] > hammer_bottom:
            return ('HAMMER', 'BULLISH')
        
        return None
    
    def detect_strat_combos(self, bar_types: List[str]) -> List[Tuple[str, str]]:
        """
        Detect STRAT combos from bar sequence.
        Returns list of (combo_name, direction) tuples.
        """
        if len(bar_types) < 4:
            return []
        closed = bar_types[:-1]
        if len(closed) < 3:
            return []
        combos: List[Tuple[str, str]] = []
        b0, b1, b2 = closed[-1], closed[-2], closed[-3]
        
        # 222 Continuations (strong momentum)
        if b0 == '2d' and b1 == '2d' and b2 == '2d':
            combos.append(('222 Bearish Continuation', 'BEARISH'))
        if b0 == '2u' and b1 == '2u' and b2 == '2u':
            combos.append(('222 Bullish Continuation', 'BULLISH'))
        
        # 212 Measured Move (consolidation then continuation)
        if b0 == '2d' and b1 == '1' and b2 == '2d':
            combos.append(('212 Bearish Measured Move', 'BEARISH'))
        if b0 == '2u' and b1 == '1' and b2 == '2u':
            combos.append(('212 Bullish Measured Move', 'BULLISH'))
        
        # 212 Reversals
        if b0 == '2d' and b1 == '1' and b2 == '2u':
            combos.append(('212 Bearish Reversal', 'BEARISH'))
        if b0 == '2u' and b1 == '1' and b2 == '2d':
            combos.append(('212 Bullish Reversal', 'BULLISH'))
        
        # 22 Reversals (simple direction change)
        if b0 == '2d' and b1 == '2u':
            combos.append(('22 Bearish Reversal', 'BEARISH'))
        if b0 == '2u' and b1 == '2d':
            combos.append(('22 Bullish Reversal', 'BULLISH'))
        
        # 322 Reversals (outside bar followed by reversal)
        if b0 == '2d' and b1 == '2u' and b2 == '3':
            combos.append(('322 Bearish Reversal', 'BEARISH'))
        if b0 == '2u' and b1 == '2d' and b2 == '3':
            combos.append(('322 Bullish Reversal', 'BULLISH'))
        
        # 32 Reversals (outside bar then directional)
        if b0 == '2d' and b1 == '3':
            combos.append(('32 Bearish Reversal', 'BEARISH'))
        if b0 == '2u' and b1 == '3':
            combos.append(('32 Bullish Reversal', 'BULLISH'))
        
        return combos
    
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
class STRATSignal:
    """Represents a STRAT pattern signal."""
    symbol: str
    pattern_name: str
    direction: str
    timestamp: str
    bar_sequence: str
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
        self.data.setdefault("open_signals", {})[signal_id] = payload
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


class STRATBot:
    """Main STRAT Bot."""
    
    def __init__(self, interval: int = 300, default_cooldown: int = 30):
        self.interval = interval
        self.default_cooldown = default_cooldown
        self.watchlist: List[WatchItem] = load_watchlist()
        self.client = MexcClient()
        self.analyzer = STRATAnalyzer()
        self.state = BotState(STATE_FILE)
        self.notifier = self._init_notifier()
        self.stats = SignalStats("STRAT Bot", STATS_FILE) if SignalStats else None
        self.health_monitor = (
            HealthMonitor("STRAT Bot", self.notifier, heartbeat_interval=3600)
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
        token = os.getenv("TELEGRAM_BOT_TOKEN_STRAT") or os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            logger.warning("Telegram credentials missing")
            return None
        return TelegramNotifier(
            bot_token=token,
            chat_id=chat_id,
            signals_log_file=str(LOG_DIR / "strat_signals.json")
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
        
        logger.info("Starting STRAT Bot for %d symbols", len(self.watchlist))
        
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
            
            backoff_until = self.exchange_backoff.get("mexc")
            if backoff_until and time.time() < backoff_until:
                logger.debug("Backoff active for mexc; skipping %s", symbol)
                continue

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
                # Handle rate limit errors (consolidated check)
                if self._is_rate_limit_error(exc):
                    self._register_backoff("mexc")
                    logger.warning("MEXC rate limit hit; backing off for 60s")
                if self.rate_limiter:
                    self.rate_limiter.record_error(f"mexc_{symbol}")
                if self.health_monitor:
                    self.health_monitor.record_error(f"Analysis error for {symbol}: {exc}")
                continue
            
            if signal is None:
                logger.debug("%s: No STRAT pattern", symbol)
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
            signal_id = f"{symbol}-STRAT-{signal.timestamp}"
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
                        "strategy": "STRAT",
                        "pattern": signal.pattern_name,
                    },
                )
            
            time.sleep(0.5)
    
    def _analyze_symbol(self, symbol: str, timeframe: str) -> Optional[STRATSignal]:
        """Analyze symbol for STRAT patterns."""
        # Fetch OHLCV data
        ohlcv = self.client.fetch_ohlcv(symbol, timeframe, limit=50)
        
        if len(ohlcv) < 10:
            return None
        
        opens = np.array([x[1] for x in ohlcv])
        highs = np.array([x[2] for x in ohlcv])
        lows = np.array([x[3] for x in ohlcv])
        closes = np.array([x[4] for x in ohlcv])
        if len(highs) < 3:
            return None
        closes_for_sma = closes[:-1] if len(closes) > 1 else closes
        sma50 = float(np.mean(closes_for_sma[-50:])) if len(closes_for_sma) >= 50 else None
        
        # Classify bars
        bar_types = self.analyzer.classify_candles(highs, lows)
        
        # Detect STRAT combos
        combos = self.analyzer.detect_strat_combos(bar_types)
        
        # Detect Hammer/Shooter
        actionable = self.analyzer.detect_actionable_patterns(opens, highs, lows, closes)
        
        # Prioritize STRAT combos, then Hammer/Shooter
        if combos:
            pattern_name, direction = combos[0]  # Use first combo
        elif actionable:
            pattern_name, direction = actionable
        else:
            return None
        
        # Get current price
        ticker = self.client.fetch_ticker(symbol)
        current_price_val = ticker.get("last") or ticker.get("close")
        if current_price_val is None:
            return None
        try:
            current_price = float(current_price_val)
        except (TypeError, ValueError):
            return None
        
        if sma50 is None:
            return None
        if direction == "BULLISH" and current_price <= sma50:
            return None
        if direction == "BEARISH" and current_price >= sma50:
            return None

        setup_high = float(highs[-2])
        setup_low = float(lows[-2])
        trigger_price = setup_high if direction == "BULLISH" else setup_low
        if direction == "BULLISH" and current_price <= trigger_price:
            return None
        if direction == "BEARISH" and current_price >= trigger_price:
            return None

        stop_loss = setup_low if direction == "BULLISH" else setup_high
        entry = current_price
        atr = float(self.analyzer.calculate_atr(highs, lows, closes))
        risk = abs(entry - stop_loss)
        if risk == 0:
            return None
        if direction == "BULLISH" and stop_loss >= entry:
            return None
        if direction == "BEARISH" and stop_loss <= entry:
            return None
        risk_unit = max(risk, atr) if atr > 0 else risk
        direction_factor = 1 if direction == "BULLISH" else -1
        tp1 = entry + direction_factor * risk_unit * 2
        tp2 = entry + direction_factor * risk_unit * 3
        
        closed_seq = bar_types[:-1]
        if len(closed_seq) >= 4:
            bar_sequence = '-'.join(closed_seq[-4:])
        else:
            bar_sequence = '-'.join(closed_seq)
        
        return STRATSignal(
            symbol=symbol,
            pattern_name=pattern_name,
            direction=direction,
            timestamp=human_ts(),
            bar_sequence=bar_sequence,
            entry=entry,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            current_price=current_price,
        )
    
    def _format_message(self, signal: STRATSignal) -> str:
        """Format Telegram message for signal."""
        direction = signal.direction
        emoji = "🟢" if direction == "BULLISH" else "🔴"
        
        lines = [
            f"{emoji} <b>{direction} STRAT PATTERN - {signal.symbol}/USDT</b>",
            "",
            f"📊 <b>Pattern:</b> {signal.pattern_name}",
            f"📐 <b>Bar Sequence:</b> {signal.bar_sequence}",
            f"💰 <b>Current Price:</b> <code>{signal.current_price:.6f}</code>",
            "",
            "<b>🎯 TRADE LEVELS:</b>",
            f"🟢 Entry: <code>{signal.entry:.6f}</code>",
            f"🛑 Stop Loss: <code>{signal.stop_loss:.6f}</code>",
            f"🎯 TP1: <code>{signal.take_profit_1:.6f}</code>",
            f"🚀 TP2: <code>{signal.take_profit_2:.6f}</code>",
        ]
        
        # Calculate R:R
        risk = abs(signal.entry - signal.stop_loss)
        reward1 = abs(signal.take_profit_1 - signal.entry)
        reward2 = abs(signal.take_profit_2 - signal.entry)
        
        if risk > 0:
            rr1 = reward1 / risk
            rr2 = reward2 / risk
            lines.append("")
            lines.append(f"⚖️ <b>Risk/Reward:</b> 1:{rr1:.2f} (TP1) | 1:{rr2:.2f} (TP2)")

        # Historical TP/SL stats
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
        
        for signal_id, payload_obj in list(signals.items()):
            if not isinstance(payload_obj, dict):
                self.state.remove_signal(signal_id)
                continue

            symbol_val = payload_obj.get("symbol")
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
            
            direction_val = payload_obj.get("direction")
            if not isinstance(direction_val, str):
                self.state.remove_signal(signal_id)
                if self.stats:
                    self.stats.discard(signal_id)
                continue
            direction = direction_val

            entry_raw = payload_obj.get("entry")
            tp1_raw = payload_obj.get("take_profit_1")
            tp2_raw = payload_obj.get("take_profit_2")
            sl_raw = payload_obj.get("stop_loss")

            if not isinstance(entry_raw, (int, float, str)):
                self.state.remove_signal(signal_id)
                if self.stats:
                    self.stats.discard(signal_id)
                continue
            if not isinstance(tp1_raw, (int, float, str)):
                self.state.remove_signal(signal_id)
                if self.stats:
                    self.stats.discard(signal_id)
                continue
            if not isinstance(tp2_raw, (int, float, str)):
                self.state.remove_signal(signal_id)
                if self.stats:
                    self.stats.discard(signal_id)
                continue
            if not isinstance(sl_raw, (int, float, str)):
                self.state.remove_signal(signal_id)
                if self.stats:
                    self.stats.discard(signal_id)
                continue

            try:
                entry = float(entry_raw)
                tp1 = float(tp1_raw)
                tp2 = float(tp2_raw)
                sl = float(sl_raw)
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
                    pattern = payload_obj.get("pattern", "STRAT")
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
    parser = argparse.ArgumentParser(description="STRAT Bot")
    parser.add_argument("--loop", action="store_true", help="Run indefinitely")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between cycles")
    parser.add_argument("--cooldown", type=int, default=30, help="Default cooldown minutes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bot = STRATBot(interval=args.interval, default_cooldown=args.cooldown)
    bot.run(loop=args.loop)


if __name__ == "__main__":
    main()
