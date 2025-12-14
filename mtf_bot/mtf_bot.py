#!/usr/bin/env python3
"""
Multi-Timeframe Bot - Detects confluence across multiple timeframes
Analyzes trend alignment between lower and higher timeframes
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
STATE_FILE = BASE_DIR / "mtf_state.json"
WATCHLIST_FILE = BASE_DIR / "mtf_watchlist.json"
STATS_FILE = LOG_DIR / "mtf_stats.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "mtf_bot.log"),
    ],
)

logger = logging.getLogger("mtf_bot")

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


class MultiTimeframeAnalyzer:
    """Analyzes trends across multiple timeframes."""
    
    # Timeframe hierarchy for confluence analysis
    TIMEFRAME_MAP = {
        "5m": ["15m", "1h"],
        "15m": ["1h", "4h"],
        "30m": ["1h", "4h"],
        "1h": ["4h", "1d"],
        "4h": ["1d", "1w"],
    }
    
    def __init__(self):
        pass
    
    def get_higher_timeframes(self, base_tf: str) -> List[str]:
        """Get higher timeframes for analysis."""
        return self.TIMEFRAME_MAP.get(base_tf, ["1h", "4h"])
    
    def calculate_ema(self, closes: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA."""
        ema = np.zeros_like(closes)
        multiplier = 2 / (period + 1)
        ema[0] = closes[0]
        for i in range(1, len(closes)):
            ema[i] = (closes[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema

    @staticmethod
    def calculate_sma(closes: np.ndarray, period: int) -> np.ndarray:
        sma = np.zeros_like(closes)
        for i in range(len(closes)):
            start = max(0, i - period + 1)
            sma[i] = float(np.mean(closes[start:i+1]))
        return sma

    @staticmethod
    def calculate_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.zeros_like(closes)
        avg_loss = np.zeros_like(closes)
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])
        for i in range(period + 1, len(closes)):
            avg_gain[i] = ((avg_gain[i-1] * (period - 1)) + gains[i-1]) / period
            avg_loss[i] = ((avg_loss[i-1] * (period - 1)) + losses[i-1]) / period
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
        rsi = 100 - (100 / (1 + rs))
        rsi[:period] = 50
        rsi[-1] = 50 if np.isnan(rsi[-1]) else rsi[-1]
        return rsi
    
    def detect_trend(self, ohlcv: list) -> Tuple[str, float]:
        """
        Detect trend direction and strength.
        Returns: (direction, confidence_score)
        """
        if len(ohlcv) < 50:
            return ("NEUTRAL", 0.0)
        
        closes = np.array([x[4] for x in ohlcv])
        rsi = self.calculate_rsi(closes, period=14)
        sma50 = self.calculate_sma(closes, 50)
        current_close = closes[-1]
        current_rsi = rsi[-1]
        current_sma = sma50[-1]

        if current_close > current_sma and current_rsi > 50:
            direction = "BULLISH"
            confidence = min((current_close - current_sma) / current_sma * 2, 1.0)
        elif current_close < current_sma and current_rsi < 50:
            direction = "BEARISH"
            confidence = min((current_sma - current_close) / current_sma * 2, 1.0)
        else:
            direction = "NEUTRAL"
            confidence = 0.0
        
        return (direction, confidence)
    
    def analyze_confluence(
        self,
        base_ohlcv: list,
        higher_ohlcvs: Dict[str, list]
    ) -> Optional[Dict[str, object]]:
        """
        Analyze confluence across timeframes.
        Returns signal if strong confluence detected.
        """
        # Detect trend on base timeframe
        base_trend, base_conf = self.detect_trend(base_ohlcv)
        
        if base_trend == "NEUTRAL" or base_conf < 0.3:
            return None
        
        higher_trends = {}
        score = 0
        ordered_tfs = list(higher_ohlcvs.keys())
        for idx, tf in enumerate(ordered_tfs):
            ohlcv = higher_ohlcvs[tf]
            trend, conf = self.detect_trend(ohlcv)
            higher_trends[tf] = (trend, conf)
            if trend == base_trend:
                weight = 2 if idx == 0 else 1
                score += weight
        
        if score < 2:
            return None
        total_possible = 2 + max(0, len(higher_trends) - 1)
        confluence_pct = (score / total_possible) * 100
        if score >= 3:
            strength = "STRONG"
        elif score == 2:
            strength = "MODERATE"
        else:
            strength = "WEAK"
        
        return {
            "direction": base_trend,
            "strength": strength,
            "confluence_pct": confluence_pct,
            "base_conf": base_conf,
            "higher_trends": higher_trends,
        }
    
    def calculate_targets(
        self,
        direction: str,
        current_price: float,
        atr: float,
        symbol: str = "",
        strength: str = "MODERATE",
    ) -> Dict[str, float]:
        """Calculate entry, stop loss, and targets."""
        entry = current_price
        
        sl_adjust = {"STRONG": 3.0, "MODERATE": 2.0, "WEAK": 1.5}
        fallback_sl_mult = sl_adjust.get(strength, 2.0)
        if TPSLCalculator is not None:
            if get_config_manager is not None:
                config_mgr = get_config_manager()
                risk_config = config_mgr.get_effective_risk("mtf_bot", symbol)
                base_sl_mult = risk_config.sl_atr_multiplier
                sl_mult = base_sl_mult * (fallback_sl_mult / 2.0)
                tp1_mult = risk_config.tp1_atr_multiplier
                tp2_mult = risk_config.tp2_atr_multiplier
                min_rr = risk_config.min_risk_reward
            else:
                sl_mult, tp1_mult, tp2_mult, min_rr = fallback_sl_mult, 2.0, 3.5, 1.8

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
                return {
                    "entry": entry,
                    "stop_loss": levels.stop_loss,
                    "take_profit_1": levels.take_profit_1,
                    "take_profit_2": levels.take_profit_2,
                }
        
        # Fallback to original calculation
        if direction == "BULLISH":
            sl = entry - (atr * 1.5)
            tp1 = entry + (atr * 2.0)
            tp2 = entry + (atr * 3.5)
        else:
            sl = entry + (atr * 1.5)
            tp1 = entry - (atr * 2.0)
            tp2 = entry - (atr * 3.5)
        
        return {
            "entry": entry,
            "stop_loss": sl,
            "take_profit_1": tp1,
            "take_profit_2": tp2,
        }
    
    @staticmethod
    def calculate_atr(ohlcv: list, period: int = 14) -> float:
        """Calculate ATR."""
        if len(ohlcv) <= period:
            highs = [x[2] for x in ohlcv]
            lows = [x[3] for x in ohlcv]
            return float(np.mean([high - low for high, low in zip(highs, lows)]))
        
        highs = [x[2] for x in ohlcv]
        lows = [x[3] for x in ohlcv]
        closes = [x[4] for x in ohlcv]
        
        trs = []
        for i in range(1, len(ohlcv)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            trs.append(tr)
        
        if len(trs) >= period:
            return float(np.mean(trs[-period:]))
        return float(np.mean(trs)) if trs else 0.0


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
class MTFSignal:
    """Represents a multi-timeframe signal."""
    symbol: str
    direction: str
    strength: str
    timestamp: str
    base_timeframe: str
    confluence_pct: float
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


class MTFBot:
    """Main Multi-Timeframe Bot."""
    
    def __init__(self, interval: int = 300, default_cooldown: int = 45):
        self.interval = interval
        self.default_cooldown = default_cooldown
        self.watchlist: List[WatchItem] = load_watchlist()
        self.client = MexcClient()
        self.analyzer = MultiTimeframeAnalyzer()
        self.state = BotState(STATE_FILE)
        self.notifier = self._init_notifier()
        self.stats = SignalStats("MTF Bot", STATS_FILE) if SignalStats else None
        self.health_monitor = (
            HealthMonitor("MTF Bot", self.notifier, heartbeat_interval=3600)
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
        token = os.getenv("TELEGRAM_BOT_TOKEN_MTF") or os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            logger.warning("Telegram credentials missing")
            return None
        return TelegramNotifier(
            bot_token=token,
            chat_id=chat_id,
            signals_log_file=str(LOG_DIR / "mtf_signals.json")
        )
    
    def run(self, loop: bool = False) -> None:
        if not self.watchlist:
            logger.error("Empty watchlist; exiting")
            return
        
        logger.info("Starting Multi-Timeframe Bot for %d symbols", len(self.watchlist))
        
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
                logger.debug("%s: No confluence detected", symbol)
                continue
            
            # Only alert on STRONG signals
            if signal.strength != "STRONG":
                logger.debug("%s: Confluence not strong enough (%s)", symbol, signal.strength)
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
            signal_id = f"{symbol}-MTF-{signal.timestamp}"
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
                "strength": signal.strength,
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
                        "strategy": "Multi-Timeframe",
                    },
                )
            
            time.sleep(0.5)
    
    def _analyze_symbol(self, symbol: str, timeframe: str) -> Optional[MTFSignal]:
        """Analyze symbol across multiple timeframes."""
        # Fetch base timeframe data
        base_ohlcv = self.client.fetch_ohlcv(symbol, timeframe, limit=100)
        
        if len(base_ohlcv) < 50:
            return None
        
        # Fetch higher timeframes
        higher_tfs = self.analyzer.get_higher_timeframes(timeframe)
        higher_ohlcvs: Dict[str, list] = {}
        
        for htf in higher_tfs:
            try:
                higher_ohlcvs[htf] = self.client.fetch_ohlcv(symbol, htf, limit=100)
            except Exception as exc:
                logger.warning("Failed to fetch %s %s: %s", symbol, htf, exc)
        
        if not higher_ohlcvs:
            return None
        
        # Analyze confluence
        result = self.analyzer.analyze_confluence(base_ohlcv, higher_ohlcvs)
        
        if result is None or not isinstance(result, dict):
            return None
        direction_val = result.get("direction")
        strength_val = result.get("strength")
        confluence_val = result.get("confluence_pct")
        if not isinstance(direction_val, str) or not isinstance(strength_val, str):
            return None
        if isinstance(confluence_val, (int, float)):
            confluence_pct = float(confluence_val)
        else:
            return None
        
        # Calculate ATR
        atr = float(self.analyzer.calculate_atr(base_ohlcv))
        
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
        targets = self.analyzer.calculate_targets(
            direction_val,
            current_price,
            atr,
            symbol=symbol,
            strength=strength_val,
        )
        
        return MTFSignal(
            symbol=symbol,
            direction=direction_val,
            strength=strength_val,
            timestamp=human_ts(),
            base_timeframe=timeframe,
            confluence_pct=confluence_pct,
            entry=targets["entry"],
            stop_loss=targets["stop_loss"],
            take_profit_1=targets["take_profit_1"],
            take_profit_2=targets["take_profit_2"],
            current_price=current_price,
        )
    
    def _format_message(self, signal: MTFSignal) -> str:
        """Format Telegram message for signal."""
        direction = signal.direction
        emoji = "🟢" if direction == "BULLISH" else "🔴"
        
        # Strength emoji
        if signal.strength == "STRONG":
            strength_emoji = "💪💪💪"
        elif signal.strength == "MODERATE":
            strength_emoji = "💪💪"
        else:
            strength_emoji = "💪"
        
        perf_line = None
        if self.stats is not None:
            symbol_key = f"{signal.symbol}/USDT"
            counts = self.stats.symbol_tp_sl_counts(symbol_key)
            tp1 = counts.get("TP1", 0)
            tp2 = counts.get("TP2", 0)
            sl = counts.get("SL", 0)
            total = tp1 + tp2 + sl
            if total > 0:
                win_rate = (tp1 + tp2) / total * 100.0
                perf_line = (
                    f"📈 <b>History:</b> TP1 {tp1} | TP2 {tp2} | SL {sl} "
                    f"(Win rate: {win_rate:.1f}%)"
                )

        lines = [
            f"{emoji} <b>{direction} MTF CONFLUENCE - {signal.symbol}/USDT</b> {strength_emoji}",
            "",
            "📊 <b>Strategy:</b> Multi-Timeframe Analysis",
            f"⚡ <b>Strength:</b> {signal.strength}",
            f"📈 <b>Confluence:</b> {signal.confluence_pct:.0f}%",
            f"💰 <b>Current Price:</b> <code>{signal.current_price:.6f}</code>" if signal.current_price is not None else "💰 <b>Current Price:</b> N/A",
        ]

        if perf_line:
            lines.append(perf_line)

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
        
        lines.extend([
            "",
            f"⏱️ Base TF: {signal.base_timeframe}",
            f"🕐 {signal.timestamp}",
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
                        f"🎯 {symbol}/USDT MTF {direction} {result} hit!\n"
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
    parser = argparse.ArgumentParser(description="Multi-Timeframe Bot")
    parser.add_argument("--loop", action="store_true", help="Run indefinitely")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between cycles")
    parser.add_argument("--cooldown", type=int, default=45, help="Default cooldown minutes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bot = MTFBot(interval=args.interval, default_cooldown=args.cooldown)
    bot.run(loop=args.loop)


if __name__ == "__main__":
    main()
