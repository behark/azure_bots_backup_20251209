#!/usr/bin/env python3
"""
PSAR Trend Bot - Parabolic SAR trend following system
Detects strong trends and provides trailing stops
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict, cast

import ccxt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
STATE_FILE = BASE_DIR / "psar_state.json"
WATCHLIST_FILE = BASE_DIR / "psar_watchlist.json"
STATS_FILE = LOG_DIR / "psar_stats.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "psar_bot.log"),
    ],
)

logger = logging.getLogger("psar_bot")

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


class PSARAnalyzer:
    """Parabolic SAR indicator analyzer."""
    
    def __init__(self, af_start: float = 0.02, af_increment: float = 0.02, af_max: float = 0.2):
        self.af_start = af_start
        self.af_increment = af_increment
        self.af_max = af_max
    
    def calculate_psar(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Parabolic SAR.
        Returns: (psar_values, trend) where trend is 1 for uptrend, -1 for downtrend
        """
        n = len(closes)
        psar = np.zeros(n)
        trend = np.zeros(n)
        ep = np.zeros(n)  # Extreme point
        af = np.zeros(n)  # Acceleration factor
        
        # Initialize
        psar[0] = closes[0]
        trend[0] = 1
        ep[0] = highs[0]
        af[0] = self.af_start
        
        for i in range(1, n):
            # Calculate SAR
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            
            # Check for trend reversal
            if trend[i-1] == 1:  # Uptrend
                # Ensure SAR is below last two lows
                psar[i] = min(psar[i], lows[i-1], lows[i-2] if i > 1 else lows[i-1])
                
                if lows[i] < psar[i]:
                    # Reversal to downtrend
                    trend[i] = -1
                    psar[i] = ep[i-1]
                    ep[i] = lows[i]
                    af[i] = self.af_start
                else:
                    # Continue uptrend
                    trend[i] = 1
                    if highs[i] > ep[i-1]:
                        ep[i] = highs[i]
                        af[i] = min(af[i-1] + self.af_increment, self.af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            
            else:  # Downtrend
                # Ensure SAR is above last two highs
                psar[i] = max(psar[i], highs[i-1], highs[i-2] if i > 1 else highs[i-1])
                
                if highs[i] > psar[i]:
                    # Reversal to uptrend
                    trend[i] = 1
                    psar[i] = ep[i-1]
                    ep[i] = highs[i]
                    af[i] = self.af_start
                else:
                    # Continue downtrend
                    trend[i] = -1
                    if lows[i] < ep[i-1]:
                        ep[i] = lows[i]
                        af[i] = min(af[i-1] + self.af_increment, self.af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
        
        return psar, trend
    
    def detect_signal(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> Optional[Tuple[str, float, float, float]]:
        """
        Detect PSAR trend signals using closed candles only.
        Returns: (direction, psar_value, strength, adx_value) or None
        """
        if len(closes) < 20:
            return None
        psar, trend = self.calculate_psar(highs, lows, closes)
        adx = self.calculate_adx(highs, lows, closes, period=14)
        signal_idx = len(closes) - 2  # last fully closed candle
        prev_idx = signal_idx - 1
        if signal_idx <= 0 or prev_idx < 0:
            return None
        adx_value = adx[signal_idx]
        if adx_value < 20:
            return None
        current_trend = trend[signal_idx]
        prev_trend = trend[prev_idx]
        if current_trend == prev_trend:
            return None
        direction = "BULLISH" if current_trend == 1 else "BEARISH"
        candle_open = opens[signal_idx]
        candle_close = closes[signal_idx]
        if direction == "BULLISH" and candle_close <= candle_open:
            return None
        if direction == "BEARISH" and candle_close >= candle_open:
            return None
        strength = 1.0
        psar_value = psar[signal_idx]
        return (direction, psar_value, strength, float(adx_value))

    
    def calculate_targets(
        self,
        direction: str,
        current_price: float,
        psar_value: float,
        atr: float,
        symbol: str = "",
    ) -> Dict[str, float]:
        """Calculate entry, stop loss, and targets."""
        entry = current_price
        
        if TPSLCalculator is not None:
            if get_config_manager is not None:
                config_mgr = get_config_manager()
                risk_config = config_mgr.get_effective_risk("psar_bot", symbol)
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
                custom_sl=psar_value,  # Use PSAR as stop loss
            )

            if levels.is_valid:
                return {
                    "entry": entry,
                    "stop_loss": psar_value,  # Keep PSAR as trailing stop
                    "take_profit_1": levels.take_profit_1,
                    "take_profit_2": levels.take_profit_2,
                }
        
        # Fallback to original calculation
        sl = psar_value
        if direction == "BULLISH":
            tp1 = entry + (atr * 2.0)
            tp2 = entry + (atr * 3.5)
        else:
            tp1 = entry - (atr * 2.0)
            tp2 = entry - (atr * 3.5)
        
        return {
            "entry": entry,
            "stop_loss": sl,
            "take_profit_1": tp1,
            "take_profit_2": tp2,
        }
    
    @staticmethod
    def calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Calculate ATR."""
        if len(closes) <= period:
            return float(np.mean(highs - lows))
        
        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            trs.append(tr)
        
        if len(trs) >= period:
            return float(np.mean(trs[-period:]))
        return float(np.mean(trs)) if trs else 0.0

    @staticmethod
    def calculate_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate ADX values."""
        length = len(closes)
        adx = np.zeros(length)
        if length <= period:
            return adx
        tr = np.zeros(length)
        plus_dm_raw = np.zeros(length)
        minus_dm_raw = np.zeros(length)
        for i in range(1, length):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            plus_dm_raw[i] = up_move if (up_move > down_move and up_move > 0) else 0
            minus_dm_raw[i] = down_move if (down_move > up_move and down_move > 0) else 0
        atr = np.zeros(length)
        plus_dm = np.zeros(length)
        minus_dm = np.zeros(length)
        plus_di = np.zeros(length)
        minus_di = np.zeros(length)
        dx_values = np.zeros(length)
        atr[period] = np.mean(tr[1:period+1])
        plus_dm[period] = np.mean(plus_dm_raw[1:period+1])
        minus_dm[period] = np.mean(minus_dm_raw[1:period+1])
        if atr[period] != 0:
            plus_di[period] = 100 * (plus_dm[period] / atr[period])
            minus_di[period] = 100 * (minus_dm[period] / atr[period])
            di_sum = plus_di[period] + minus_di[period]
            if di_sum != 0:
                dx_values[period] = 100 * abs(plus_di[period] - minus_di[period]) / di_sum
        for i in range(period + 1, length):
            atr[i] = atr[i-1] - (atr[i-1] / period) + tr[i]
            plus_dm[i] = plus_dm[i-1] - (plus_dm[i-1] / period) + plus_dm_raw[i]
            minus_dm[i] = minus_dm[i-1] - (minus_dm[i-1] / period) + minus_dm_raw[i]
            if atr[i] == 0:
                continue
            plus_di[i] = 100 * (plus_dm[i] / atr[i])
            minus_di[i] = 100 * (minus_dm[i] / atr[i])
            di_sum = plus_di[i] + minus_di[i]
            if di_sum == 0:
                continue
            dx_values[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
        first_adx_idx = period * 2
        if first_adx_idx >= length:
            first_adx_idx = length - 1
        valid_dx = dx_values[period:first_adx_idx+1]
        if len(valid_dx) == 0:
            return adx
        adx[first_adx_idx] = np.mean(valid_dx)
        for i in range(first_adx_idx + 1, length):
            adx[i] = ((adx[i-1] * (period - 1)) + dx_values[i]) / period
        return adx


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


@dataclass
class PSARSignal:
    """Represents a PSAR trend signal."""
    symbol: str
    direction: str
    timestamp: str
    psar_value: float
    strength: float
    adx: float
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


class PSARBot:
    """Main PSAR Trend Bot."""
    
    def __init__(self, interval: int = 300, default_cooldown: int = 30):
        self.interval = interval
        self.default_cooldown = default_cooldown
        self.watchlist: List[WatchItem] = load_watchlist()
        self.client = MexcClient()
        self.analyzer = PSARAnalyzer()
        self.state = BotState(STATE_FILE)
        self.notifier = self._init_notifier()
        self.stats = SignalStats("PSAR Bot", STATS_FILE) if SignalStats else None
        self.health_monitor = (
            HealthMonitor("PSAR Bot", self.notifier, heartbeat_interval=3600)
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
        token = os.getenv("TELEGRAM_BOT_TOKEN_PSAR") or os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            logger.warning("Telegram credentials missing")
            return None
        return TelegramNotifier(
            bot_token=token,
            chat_id=chat_id,
            signals_log_file=str(LOG_DIR / "psar_signals.json")
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
        
        logger.info("Starting PSAR Trend Bot for %d symbols", len(self.watchlist))
        
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
            period_val = item.get("period") if isinstance(item, dict) else None
            period = period_val if isinstance(period_val, str) else "5m"
            cooldown_raw = item.get("cooldown_minutes") if isinstance(item, dict) else None
            try:
                cooldown = int(cooldown_raw) if cooldown_raw is not None else self.default_cooldown
            except Exception:
                cooldown = self.default_cooldown

            # Check backoff status before making API calls
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
                logger.debug("%s: No PSAR signal", symbol)
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
            signal_id = f"{symbol}-PSAR-{signal.timestamp}"
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
                        "strategy": "PSAR Trend",
                    },
                )
            
            time.sleep(0.5)
    
    def _analyze_symbol(self, symbol: str, timeframe: str) -> Optional[PSARSignal]:
        """Analyze symbol for PSAR signals."""
        # Fetch OHLCV data
        ohlcv = self.client.fetch_ohlcv(symbol, timeframe, limit=100)
        
        if len(ohlcv) < 50:
            return None
        
        opens = np.array([x[1] for x in ohlcv])
        highs = np.array([x[2] for x in ohlcv])
        lows = np.array([x[3] for x in ohlcv])
        closes = np.array([x[4] for x in ohlcv])
        
        # Detect signal
        result = self.analyzer.detect_signal(opens, highs, lows, closes)
        
        if result is None:
            return None
        
        direction, psar_value, strength, adx_value = result
        
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
        
        # Calculate targets
        targets = self.analyzer.calculate_targets(direction, current_price, psar_value, atr)
        
        return PSARSignal(
            symbol=symbol,
            direction=direction,
            timestamp=human_ts(),
            psar_value=psar_value,
            strength=strength,
            adx=adx_value,
            entry=targets["entry"],
            stop_loss=targets["stop_loss"],
            take_profit_1=targets["take_profit_1"],
            take_profit_2=targets["take_profit_2"],
            current_price=current_price,
        )
    
    def _format_message(self, signal: PSARSignal) -> str:
        """Format Telegram message for signal."""
        direction = signal.direction
        emoji = "🟢" if direction == "BULLISH" else "🔴"
        
        lines = [
            f"{emoji} <b>{direction} PSAR TREND - {signal.symbol}/USDT</b>",
            "",
            "📊 <b>Strategy:</b> Parabolic SAR Trend Following",
            f"💰 <b>Current Price:</b> <code>{signal.current_price:.6f}</code>" if signal.current_price is not None else "💰 <b>Current Price:</b> N/A",
            f"📍 <b>PSAR Level:</b> <code>{signal.psar_value:.6f}</code>",
            f"📈 <b>ADX:</b> {signal.adx:.1f}",
            "",
            "<b>🎯 TRADE LEVELS:</b>",
            f"🟢 Entry: <code>{signal.entry:.6f}</code>",
            f"🛑 Stop Loss (PSAR): <code>{signal.stop_loss:.6f}</code>",
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
            "💡 <b>Note:</b> PSAR acts as trailing stop - adjust as it moves",
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
                        f"🎯 {symbol}/USDT PSAR {direction} {result} hit!\n"
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
    parser = argparse.ArgumentParser(description="PSAR Trend Bot")
    parser.add_argument("--loop", action="store_true", help="Run indefinitely")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between cycles")
    parser.add_argument("--cooldown", type=int, default=30, help="Default cooldown minutes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bot = PSARBot(interval=args.interval, default_cooldown=args.cooldown)
    bot.run(loop=args.loop)


if __name__ == "__main__":
    main()
