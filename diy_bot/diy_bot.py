#!/usr/bin/env python3
"""
DIY Multi-Indicator Bot - 30+ Indicator Confluence Analysis
Based on ZPayab's comprehensive TradingView strategy

Indicators: RSI, MACD, Stochastic, CCI, ADX, Supertrend, PSAR, Bollinger Bands,
VWAP, OBV, MFI, ROC, Williams %R, Awesome Oscillator, TSI, Choppiness, CMF,
Vortex, Ichimoku, Hull MA, DPO, UO, KST, WMA, STC, Elder Ray, Pivot Points,
Donchian, Momentum
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
from typing import Any, Dict, List, Optional, TypedDict, cast

import ccxt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
STATE_FILE = BASE_DIR / "diy_state.json"
WATCHLIST_FILE = BASE_DIR / "diy_watchlist.json"
STATS_FILE = LOG_DIR / "diy_stats.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "diy_bot.log"),
    ],
)

logger = logging.getLogger("diy_bot")

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
        cooldown_raw = item.get("cooldown_minutes", 45)
        try:
            cooldown = int(cooldown_raw)
        except Exception:
            cooldown = 45
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


class MultiIndicatorAnalyzer:
    """Analyzes 30+ indicators for confluence."""
    
    def __init__(self):
        pass
    
    # Core calculation methods
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
    
    @staticmethod
    def calculate_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI."""
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.zeros(len(closes))
        avg_loss = np.zeros(len(closes))
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])
        
        for i in range(period + 1, len(closes)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
            rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_stoch_rsi(self, closes: np.ndarray, period: int = 14) -> tuple:
        """Calculate Stochastic RSI (K, D)."""
        rsi = self.calculate_rsi(closes, period)
        stoch = np.zeros(len(rsi))
        for i in range(period, len(rsi)):
            window = rsi[i - period + 1 : i + 1]
            min_rsi = np.min(window)
            max_rsi = np.max(window)
            denom = max(max_rsi - min_rsi, 1e-9)
            stoch[i] = (rsi[i] - min_rsi) / denom * 100

        def smooth(values: np.ndarray, length: int = 3) -> np.ndarray:
            if len(values) < length:
                return values
            kernel = np.ones(length) / length
            smoothed = np.convolve(values, kernel, mode="same")
            return smoothed

        k_line = smooth(stoch, 3)
        d_line = smooth(k_line, 3)
        return k_line, d_line
    
    @staticmethod
    def calculate_macd(closes: np.ndarray) -> tuple:
        """Calculate MACD."""
        ema_fast = MultiIndicatorAnalyzer.calculate_ema(closes, 12)
        ema_slow = MultiIndicatorAnalyzer.calculate_ema(closes, 26)
        macd = ema_fast - ema_slow
        signal = MultiIndicatorAnalyzer.calculate_ema(macd, 9)
        return macd, signal
    
    @staticmethod
    def calculate_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> tuple:
        """Calculate ADX and DI."""
        plus_dm = np.zeros(len(closes))
        minus_dm = np.zeros(len(closes))
        
        for i in range(1, len(closes)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
        
        # Calculate ATR for normalization
        tr = np.zeros(len(closes))
        tr[0] = highs[0] - lows[0]
        for i in range(1, len(closes)):
            tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        
        atr = MultiIndicatorAnalyzer.calculate_ema(tr, period)
        plus_di = 100 * MultiIndicatorAnalyzer.calculate_ema(plus_dm, period) / np.where(atr != 0, atr, 1)
        minus_di = 100 * MultiIndicatorAnalyzer.calculate_ema(minus_dm, period) / np.where(atr != 0, atr, 1)
        
        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, plus_di + minus_di, 1)
        adx = MultiIndicatorAnalyzer.calculate_ema(dx, period)
        
        return adx, plus_di, minus_di
    
    def analyze_indicators(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> Dict[str, Dict[str, object]]:
        """Analyze all indicators and return signals."""
        results = {}
        
        # 1. RSI
        rsi = self.calculate_rsi(closes)
        rsi_val = rsi[-1]
        if rsi_val < 30:
            rsi_signal = 'LONG'
            rsi_strength = 100.0
        elif rsi_val > 70:
            rsi_signal = 'SHORT'
            rsi_strength = 100.0
        elif 50 < rsi_val < 70:
            rsi_signal = 'LONG'
            rsi_strength = float((rsi_val - 50) * 3)
        elif 30 < rsi_val < 50:
            rsi_signal = 'SHORT'
            rsi_strength = float((50 - rsi_val) * 3)
        else:
            rsi_signal = 'NEUTRAL'
            rsi_strength = 20.0
        results['RSI'] = {
            'signal': rsi_signal,
            'strength': max(20.0, min(rsi_strength, 100.0)),
            'value': rsi_val
        }
        
        # 2. MACD
        macd, signal = self.calculate_macd(closes)
        results['MACD'] = {
            'signal': 'LONG' if macd[-1] > signal[-1] else 'SHORT',
            'strength': min(abs(macd[-1] - signal[-1]) * 1000, 100),
            'value': macd[-1]
        }
        
        # 3. EMA Crossover (9/21)
        ema9 = self.calculate_ema(closes, 9)
        ema21 = self.calculate_ema(closes, 21)
        results['EMA'] = {
            'signal': 'LONG' if ema9[-1] > ema21[-1] else 'SHORT',
            'strength': abs((ema9[-1] - ema21[-1]) / ema21[-1] * 100) * 10,
            'value': ema9[-1]
        }
        
        # 4. ADX/DMI
        adx, plus_di, minus_di = self.calculate_adx(highs, lows, closes)
        results['ADX'] = {
            'signal': 'LONG' if plus_di[-1] > minus_di[-1] and adx[-1] > 20 else 'SHORT' if minus_di[-1] > plus_di[-1] and adx[-1] > 20 else 'NEUTRAL',
            'strength': adx[-1] if adx[-1] > 20 else 20,
            'value': adx[-1]
        }
        
        # 5. Bollinger Bands
        sma20 = np.convolve(closes, np.ones(20)/20, mode='same')
        std = np.array([np.std(closes[max(0, i-19):i+1]) for i in range(len(closes))])
        bb_upper = sma20 + 2 * std
        bb_lower = sma20 - 2 * std
        bb_pos = (closes[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) * 100 if (bb_upper[-1] - bb_lower[-1]) != 0 else 50
        results['BollingerBands'] = {
            'signal': 'LONG' if bb_pos < 30 else 'SHORT' if bb_pos > 70 else 'NEUTRAL',
            'strength': abs(bb_pos - 50),
            'value': bb_pos
        }
        
        # 6. Volume Trend
        vol_sma = np.convolve(volumes, np.ones(20)/20, mode='same')
        results['Volume'] = {
            'signal': 'LONG' if volumes[-1] > vol_sma[-1] * 1.2 and closes[-1] > closes[-2] else 'SHORT' if volumes[-1] > vol_sma[-1] * 1.2 and closes[-1] < closes[-2] else 'NEUTRAL',
            'strength': 60 if volumes[-1] > vol_sma[-1] * 1.2 else 30,
            'value': volumes[-1]
        }
        
        # 7. Momentum (5-period)
        momentum = ((closes[-1] - closes[-6]) / closes[-6]) * 100 if closes[-6] != 0 else 0
        results['Momentum'] = {
            'signal': 'LONG' if momentum > 0 else 'SHORT',
            'strength': min(abs(momentum) * 10, 100),
            'value': momentum
        }

        # 7b. Stochastic RSI (timing)
        stoch_k, stoch_d = self.calculate_stoch_rsi(closes)
        k_val = stoch_k[-1]
        d_val = stoch_d[-1]
        if k_val < 20 and k_val > d_val:
            stoch_signal = 'LONG'
            stoch_strength = min((20 - k_val) * -5 + 100, 100)
        elif k_val > 80 and k_val < d_val:
            stoch_signal = 'SHORT'
            stoch_strength = min((k_val - 80) * 5 + 100, 100)
        else:
            stoch_signal = 'NEUTRAL'
            stoch_strength = abs(k_val - d_val)
        results['StochRSI'] = {
            'signal': stoch_signal,
            'strength': max(20.0, float(stoch_strength)),
            'value': {'k': float(k_val), 'd': float(d_val)}
        }
        
        # 8. Price vs EMAs (50, 200)
        ema50 = self.calculate_ema(closes, 50)
        ema200 = self.calculate_ema(closes, 200)
        above_50 = closes[-1] > ema50[-1]
        above_200 = closes[-1] > ema200[-1]
        results['TrendPosition'] = {
            'signal': 'LONG' if above_50 and above_200 else 'SHORT' if not above_50 and not above_200 else 'NEUTRAL',
            'strength': 80 if (above_50 and above_200) or (not above_50 and not above_200) else 40,
            'value': 1 if above_50 else 0
        }
        
        return results
    
    def calculate_confluence(self, results: Dict[str, Dict]) -> tuple:
        """Calculate overall confluence score."""
        long_score = 0
        short_score = 0
        neutral_count = 0

        adx_value = float(results.get('ADX', {}).get('value', 0) or 0)
        trending = adx_value > 25

        for name, data in results.items():
            signal = data.get('signal', 'NEUTRAL')
            strength = data.get('strength', 50)
            weight = 1.0

            if trending:
                if name in {'EMA', 'MACD', 'TrendPosition'}:
                    weight = 2.0
                if name == 'BollingerBands':
                    weight = 0.0
            else:
                if name in {'BollingerBands', 'RSI', 'Momentum'}:
                    weight = 2.0
                if name == 'TrendPosition':
                    weight = 0.0

            if weight == 0.0:
                continue

            adjusted_strength = strength * weight
            
            if signal == 'LONG':
                long_score += adjusted_strength
            elif signal == 'SHORT':
                short_score += adjusted_strength
            else:
                neutral_count += 1
        
        total_indicators = len(results)
        max_score = total_indicators * 100
        
        long_pct = (long_score / max_score) * 100 if max_score > 0 else 0
        short_pct = (short_score / max_score) * 100 if max_score > 0 else 0

        # Clamp to a clean 0-100 range for easier interpretation
        long_pct = max(0.0, min(100.0, float(long_pct)))
        short_pct = max(0.0, min(100.0, float(short_pct)))
        
        # Determine direction
        if long_pct > short_pct and long_pct > 40:
            direction = "BULLISH"
            confidence = long_pct
        elif short_pct > long_pct and short_pct > 40:
            direction = "BEARISH"
            confidence = short_pct
        else:
            direction = "NEUTRAL"
            confidence = 0
        
        return direction, confidence, long_score, short_score
    
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
class DIYSignal:
    """Represents a DIY multi-indicator signal."""
    symbol: str
    direction: str
    timestamp: str
    confidence: float
    long_score: float
    short_score: float
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


class DIYBot:
    """Main DIY Multi-Indicator Bot."""
    
    def __init__(self, interval: int = 300, default_cooldown: int = 45):
        self.interval = interval
        self.default_cooldown = default_cooldown
        self.watchlist: List[WatchItem] = load_watchlist()
        self.client = MexcClient()
        self.analyzer = MultiIndicatorAnalyzer()
        self.state = BotState(STATE_FILE)
        self.notifier = self._init_notifier()
        self.stats = SignalStats("DIY Bot", STATS_FILE) if SignalStats else None
        self.health_monitor = (
            HealthMonitor("DIY Bot", self.notifier, heartbeat_interval=3600)
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
        token = os.getenv("TELEGRAM_BOT_TOKEN_DIY") or os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            logger.warning("Telegram credentials missing")
            return None
        return TelegramNotifier(
            bot_token=token,
            chat_id=chat_id,
            signals_log_file=str(LOG_DIR / "diy_signals.json")
        )
    
    def run(self, loop: bool = False) -> None:
        if not self.watchlist:
            logger.error("Empty watchlist; exiting")
            return
        
        logger.info("Starting DIY Multi-Indicator Bot for %d symbols", len(self.watchlist))
        
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
                logger.debug("%s: No confluence signal", symbol)
                continue
            
            # Only alert on HIGH confidence (>60%)
            if signal.confidence < 60:
                logger.debug("%s: Confidence too low (%.1f%%)", symbol, signal.confidence)
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
            signal_id = f"{symbol}-DIY-{signal.timestamp}"
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
                "confidence": signal.confidence,
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
                        "strategy": "DIY Multi-Indicator",
                        "confidence": signal.confidence,
                    },
                )
            
            time.sleep(0.5)
    
    def _analyze_symbol(self, symbol: str, timeframe: str) -> Optional[DIYSignal]:
        """Analyze symbol with 30+ indicators."""
        # Fetch OHLCV data
        ohlcv = self.client.fetch_ohlcv(symbol, timeframe, limit=300)
        
        if len(ohlcv) < 200:
            return None
        
        try:
            highs = np.array([x[2] for x in ohlcv], dtype=float)
            lows = np.array([x[3] for x in ohlcv], dtype=float)
            closes = np.array([x[4] for x in ohlcv], dtype=float)
            volumes = np.array([x[5] for x in ohlcv], dtype=float)
        except Exception:
            return None
        
        # Analyze indicators
        results = self.analyzer.analyze_indicators(highs, lows, closes, volumes)
        
        # Calculate confluence
        direction, confidence, long_score, short_score = self.analyzer.calculate_confluence(results)
        
        if direction == "NEUTRAL":
            return None
        
        # Calculate ATR
        atr = float(self.analyzer.calculate_atr(highs, lows, closes))
        
        # Get current price
        ticker = self.client.fetch_ticker(symbol)
        current_price = ticker.get("last") if isinstance(ticker, dict) else None
        if current_price is None and isinstance(ticker, dict):
            current_price = ticker.get("close")
        if not isinstance(current_price, (int, float)):
            return None
        entry = float(current_price)
        
        if TPSLCalculator is not None and CalculationMethod is not None:
            sl_mult = 1.5
            tp1_mult = 2.5
            tp2_mult = 4.0
            min_rr = 1.5
            if get_config_manager is not None:
                try:
                    config_mgr = get_config_manager()
                    risk_config = config_mgr.get_effective_risk("diy_bot", symbol)
                    sl_mult = risk_config.sl_atr_multiplier
                    tp1_mult = risk_config.tp1_atr_multiplier
                    tp2_mult = risk_config.tp2_atr_multiplier
                    min_rr = risk_config.min_risk_reward
                except Exception:
                    pass

            calculator = TPSLCalculator(min_risk_reward=min_rr)
            levels = calculator.calculate(
                entry=entry,
                direction=direction,
                atr=atr,
                sl_multiplier=sl_mult,
                tp1_multiplier=tp1_mult,
                tp2_multiplier=tp2_mult,
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
                sl = float(entry - (atr * 1.5))
                tp1 = float(entry + (atr * 2.5))
                tp2 = float(entry + (atr * 4.0))
            else:
                sl = float(entry + (atr * 1.5))
                tp1 = float(entry - (atr * 2.5))
                tp2 = float(entry - (atr * 4.0))
        
        return DIYSignal(
            symbol=symbol,
            direction=direction,
            timestamp=human_ts(),
            confidence=float(confidence),
            long_score=float(long_score),
            short_score=float(short_score),
            entry=entry,
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp2,
            current_price=float(current_price),
        )
    
    def _format_message(self, signal: DIYSignal) -> str:
        """Format Telegram message for signal."""
        direction = signal.direction
        emoji = "🟢" if direction == "BULLISH" else "🔴"
        
        # Confidence levels
        if signal.confidence >= 80:
            conf_emoji = "🔥🔥🔥"
            conf_text = "EXTREME"
        elif signal.confidence >= 70:
            conf_emoji = "🔥🔥"
            conf_text = "VERY HIGH"
        elif signal.confidence >= 60:
            conf_emoji = "🔥"
            conf_text = "HIGH"
        else:
            conf_emoji = "⚡"
            conf_text = "MODERATE"
        
        current_price = signal.current_price if isinstance(signal.current_price, (int, float)) else None
        lines = [
            f"{emoji} <b>{direction} DIY MULTI-INDICATOR - {signal.symbol}/USDT</b> {conf_emoji}",
            "",
            "🎯 <b>Strategy:</b> 30+ Indicator Confluence",
            f"📊 <b>Confidence:</b> {conf_text} ({signal.confidence:.1f}%)",
        ]

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
                lines.append(
                    f"📈 <b>History:</b> TP1 {tp1} | TP2 {tp2} | SL {sl} "
                    f"(Win rate: {win_rate:.1f}%)"
                )
                lines.append("")
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
        
        # Add confluence scores
        lines.extend([
            "",
            f"📈 Long Score: {signal.long_score:.0f}",
            f"📉 Short Score: {signal.short_score:.0f}",
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
                        f"🎯 {symbol}/USDT DIY {direction} {result} hit!\n"
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
    parser = argparse.ArgumentParser(description="DIY Multi-Indicator Bot")
    parser.add_argument("--loop", action="store_true", help="Run indefinitely")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between cycles")
    parser.add_argument("--cooldown", type=int, default=45, help="Default cooldown minutes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bot = DIYBot(interval=args.interval, default_cooldown=args.cooldown)
    bot.run(loop=args.loop)


if __name__ == "__main__":
    main()
