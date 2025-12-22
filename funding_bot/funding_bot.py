#!/usr/bin/env python3
"""Funding Rate & Open Interest Bot - Gold Standard Edition.

Features:
- Multi-Exchange Support (Binance, Bybit, MEXC) via CCXT
- Robust State Management with File Locking
- Signal Reversal Detection
- Enhanced Logging & Error Handling
- Dynamic Configuration
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
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import ccxt  # type: ignore
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
STATE_FILE = BASE_DIR / "funding_state.json"
WATCHLIST_FILE = BASE_DIR / "funding_watchlist.json"
STATS_FILE = LOG_DIR / "funding_stats.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# Add parent directory to path for shared modules
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))

# Required imports (fail fast if missing)
from message_templates import format_signal_message
from notifier import TelegramNotifier
from signal_stats import SignalStats
from tp_sl_calculator import TPSLCalculator
from trade_config import get_config_manager

# Optional imports (safe fallback)
from safe_import import safe_import
HealthMonitor = safe_import('health_monitor', 'HealthMonitor')
RateLimitHandler = safe_import('rate_limit_handler', 'RateLimitHandler')
RateLimitedExchange = safe_import('rate_limit_handler', 'RateLimitedExchange')

def load_json_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return cast(Dict[str, Any], json.loads(path.read_text()))
    except Exception:
        return {}

def setup_logging(log_level: str = "INFO", enable_detailed: bool = False) -> logging.Logger:
    """Setup enhanced logging."""
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
        LOG_DIR / "funding_bot.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    error_handler = RotatingFileHandler(
        LOG_DIR / "funding_errors.log", maxBytes=5 * 1024 * 1024, backupCount=3
    )
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    return logging.getLogger("funding_bot")

logger = logging.getLogger("funding_bot")

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
class FundingSignal:
    symbol: str
    direction: str
    funding_rate: float
    predicted_price: float
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    timestamp: str
    timeframe: str
    exchange: str
    reasons: List[str]
    funding_timestamp: Optional[int] = None # For deduplication
    open_interest: Optional[float] = None
    oi_change_24h: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

class FundingAnalyzer:
    """Analyzes Funding Rates and Open Interest for signals."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.funding_thresholds = config.get("analysis", {}).get("funding_thresholds", {})
        self.oi_thresholds = config.get("analysis", {}).get("open_interest_thresholds", {})

    def analyze(self, symbol: str, funding_rate: float, oi: Optional[float], oi_change: Optional[float], price: float, ema: Optional[float]) -> Optional[Dict[str, Any]]:
        score = 0
        reasons = []
        
        # Funding Rate Logic (Contrarian)
        extreme_pos = self.funding_thresholds.get("extreme_positive", 0.0005)
        extreme_neg = self.funding_thresholds.get("extreme_negative", -0.0005)
        
        direction = "NEUTRAL"
        
        if funding_rate >= extreme_pos:
            score -= 2
            reasons.append(f"Extreme Positive Funding ({funding_rate*100:.4f}%)")
            if ema and price < ema:
                score -= 1
                reasons.append("Downtrend alignment (Price < EMA)")
        
        elif funding_rate <= extreme_neg:
            score += 2
            reasons.append(f"Extreme Negative Funding ({funding_rate*100:.4f}%)")
            if ema and price > ema:
                score += 1
                reasons.append("Uptrend alignment (Price > EMA)")
                
        if oi_change:
            limit = self.oi_thresholds.get("strong_increase_pct", 5.0)
            if oi_change >= limit:
                reasons.append(f"Strong OI Increase (+{oi_change:.1f}%)")
                if abs(score) >= 2:
                    score = int(score * 1.5)
        
        if score >= 2:
            direction = "BULLISH"
        elif score <= -2:
            direction = "BEARISH"
            
        if direction == "NEUTRAL":
            return None
            
        return {
            "direction": direction,
            "reasons": reasons,
            "score": score
        }

    def calculate_targets(self, entry: float, direction: str, atr: float, symbol: str = "") -> Optional[Dict[str, float]]:
        """Calculate TP/SL using centralized TPSLCalculator with config from trade_config."""
        # Get risk settings from central config
        config_mgr = get_config_manager()
        risk_config = config_mgr.get_effective_risk("funding_bot", symbol)

        dir_normalized = "LONG" if direction == "BULLISH" else "SHORT"
        calculator = TPSLCalculator(min_risk_reward=risk_config.min_risk_reward)
        levels = calculator.calculate(
            entry=entry,
            direction=dir_normalized,
            atr=atr,
            sl_multiplier=risk_config.sl_atr_multiplier,
            tp1_multiplier=risk_config.tp1_atr_multiplier,
            tp2_multiplier=risk_config.tp2_atr_multiplier,
        )
        if not levels.is_valid:
            return None
        return {"entry": entry, "sl": levels.stop_loss, "tp1": levels.take_profit_1, "tp2": levels.take_profit_2}

class SignalTracker:
    """Robust signal tracker with file locking."""

    def __init__(self, stats: Optional[Any] = None, config: Optional[Dict[str, Any]] = None) -> None:
        self.stats = stats
        self.config: Dict[str, Any] = config or {}
        self.state_lock = threading.Lock()
        self.state = self._load_state()

    def _empty_state(self) -> Dict[str, Any]:
        return {"last_alerts": {}, "open_signals": {}, "signal_history": {}, "last_result_notifications": {}}

    def _load_state(self) -> Dict[str, Any]:
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, 'r') as f:
                    return cast(Dict[str, Any], json.load(f))
            except Exception:
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

    def should_alert(self, symbol: str, exchange: str, cooldown_mins: int, funding_ts: Optional[int] = None) -> bool:
        """Check if we should alert, using either cooldown or funding timestamp."""
        if funding_ts:
            history = self.state.setdefault("signal_history", {})
            key = f"{symbol}-{exchange}"
            if str(funding_ts) in history.get(key, []):
                return False

        key = f"{symbol}-{exchange}"
        last_ts = self.state.get("last_alerts", {}).get(key)
        if not last_ts: return True
        try:
            last = datetime.fromisoformat(last_ts)
            if last.tzinfo is None: last = last.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - last) >= timedelta(minutes=cooldown_mins)
        except Exception: return True

    def add_signal(self, signal: FundingSignal) -> None:
        # Mark history for deduplication
        if signal.funding_timestamp:
            history = self.state.setdefault("signal_history", {})
            key = f"{signal.symbol}-{signal.exchange}"
            if key not in history: history[key] = []
            ts_str = str(signal.funding_timestamp)
            if ts_str not in history[key]:
                history[key].append(ts_str)
                if len(history[key]) > 10: history[key] = history[key][-10:]

        signals = self.state.setdefault("open_signals", {})
        sig_id = f"{signal.symbol}-{signal.timestamp}"
        signals[sig_id] = signal.as_dict()
        
        last_alerts = self.state.setdefault("last_alerts", {})
        last_alerts[f"{signal.symbol}-{signal.exchange}"] = signal.timestamp
        
        self._save_state()
        
        if self.stats:
            self.stats.record_open(
                sig_id, signal.symbol, signal.direction, signal.entry, signal.timestamp,
                extra={"exchange": signal.exchange}
            )

    def check_open_signals(self, client_map: Dict[str, Any], notifier: Optional[Any]) -> None:
        """Check open signals for TP/SL hits."""
        signals = self.state.get("open_signals", {})
        if not signals:
            return

        # Result Notification Cooldown
        last_notifs = self.state.setdefault("last_result_notifications", {})
        cooldown_mins = self.config.get("signal", {}).get("result_notification_cooldown_minutes", 15)

        updated = False
        for sig_id, payload in list(signals.items()):
            symbol = payload.get("symbol")
            display_symbol = symbol if "/" in symbol else f"{symbol}/USDT"
            exchange = payload.get("exchange", "mexc")
            direction = payload.get("direction")

            # Cooldown Check
            last_ts = last_notifs.get(display_symbol)
            if last_ts:
                last_dt = datetime.fromisoformat(last_ts)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                if (datetime.now(timezone.utc) - last_dt) < timedelta(minutes=cooldown_mins):
                    continue

            try:
                client = client_map.get(exchange)
                if not client:
                    continue
                ticker = client.fetch_ticker(resolve_symbol(symbol))
                price = ticker.get("last")
            except Exception:
                continue

            if not price:
                continue

            tp1, tp2 = payload.get("take_profit_1"), payload.get("take_profit_2")
            sl = payload.get("stop_loss")
            entry = payload.get("entry")

            res = None
            if direction == "BULLISH":
                if price >= tp2:
                    res = "TP2"
                elif price >= tp1:
                    res = "TP1"
                elif price <= sl:
                    res = "SL"
            else:  # BEARISH
                if price <= tp2:
                    res = "TP2"
                elif price <= tp1:
                    res = "TP1"
                elif price >= sl:
                    res = "SL"

            if res:
                pnl = (price - entry) / entry * 100 if direction == "BULLISH" else (entry - price) / entry * 100
                msg = f"🎯 {display_symbol} FUNDING {res} HIT!\n💰 PnL: {pnl:.2f}%"

                if notifier:
                    notifier.send_message(msg)
                    last_notifs[display_symbol] = datetime.now(timezone.utc).isoformat()

                if self.stats:
                    self.stats.record_close(sig_id, price, res)
                del signals[sig_id]
                updated = True

        if updated:
            self._save_state()

    def check_reversal(self, symbol: str, new_direction: str, notifier: Optional[Any]) -> None:
        signals = self.state.get("open_signals", {})
        for sig_id, payload in signals.items():
            if payload.get("symbol") == symbol:
                old_dir = payload.get("direction")
                if old_dir != new_direction:
                    msg = f"⚠️ FUNDING REVERSAL: {symbol}\nOpen: {old_dir} | New: {new_direction}\nCheck position!"
                    if notifier: notifier.send_message(msg)

class FundingBot:
    """Refactored Funding Bot."""
    
    def __init__(self, config_path: Path):
        self.config = load_json_config(config_path)
        self.watchlist = self._load_watchlist()
        self.analyzer = FundingAnalyzer(self.config)
        self.tracker = SignalTracker(SignalStats("Funding Bot", STATS_FILE), self.config)
        self.notifier = self._init_notifier()
        
        # Init Health Monitor
        heartbeat = self.config.get("execution", {}).get("health_check_interval_seconds", 3600)
        self.health_monitor = HealthMonitor("Funding Bot", self.notifier, heartbeat_interval=heartbeat) if HealthMonitor and self.notifier else None

        # Init Rate Limiter
        rate_cfg = self.config.get("rate_limit", {})
        self.rate_handler = None
        if RateLimitHandler:
            self.rate_handler = RateLimitHandler(
                base_delay=rate_cfg.get("base_delay_seconds", 0.5),
                max_retries=rate_cfg.get("max_retries", 5),
                backoff_factor=rate_cfg.get("backoff_multiplier", 2.0),
                max_backoff=rate_cfg.get("max_backoff_seconds", 30.0)
            )
            logger.info("RateLimitHandler initialized for API protection")

        # Init Exchange Clients with rate limiting
        self.clients: Dict[str, Any] = {}
        for name, cfg in EXCHANGE_CONFIG.items():
            try:
                raw_client = cfg["factory"](cfg["params"])
                # Wrap with rate limiting if available
                if RateLimitedExchange and self.rate_handler:
                    self.clients[name] = RateLimitedExchange(raw_client, self.rate_handler)
                    logger.debug(f"Exchange {name} wrapped with RateLimitedExchange")
                else:
                    self.clients[name] = raw_client
            except Exception as e:
                logger.error(f"Failed to init {name}: {e}")

    def _load_watchlist(self) -> List[Dict[str, Any]]:
        if not WATCHLIST_FILE.exists(): return []
        try:
            return cast(List[Dict[str, Any]], json.loads(WATCHLIST_FILE.read_text()))
        except Exception: return []

    def _init_notifier(self) -> Optional[Any]:
        if TelegramNotifier is None: return None
        token = os.getenv("TELEGRAM_BOT_TOKEN_FUNDING") or os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        return TelegramNotifier(bot_token=token, chat_id=chat_id, signals_log_file=str(LOG_DIR/"funding_signals.json")) if token and chat_id else None

    def run(self, run_once: bool = False) -> None:
        logger.info("Starting Refactored Funding Bot...")
        if self.health_monitor: self.health_monitor.send_startup_message()
        
        while True:
            try:
                for item in self.watchlist:
                    symbol_raw = item.get("symbol")
                    if not symbol_raw or not isinstance(symbol_raw, str):
                        continue
                    symbol: str = symbol_raw
                    exchange: str = str(item.get("exchange", "mexc"))
                    timeframe: str = str(item.get("timeframe", "8h"))

                    client = self.clients.get(exchange)
                    if not client: continue
                    
                    # 1. Fetch Data via CCXT
                    try:
                        # Funding Rate
                        fr_data = client.fetch_funding_rate(resolve_symbol(symbol))
                        funding_rate = fr_data.get('fundingRate')
                        funding_ts = fr_data.get('fundingTimestamp') # Unique ID for this funding interval
                        
                        # Ticker (Price)
                        ticker = client.fetch_ticker(resolve_symbol(symbol))
                        price = ticker.get('last')
                        
                        # EMA (needs history)
                        ohlcv = client.fetch_ohlcv(resolve_symbol(symbol), timeframe, limit=50)
                        closes = [c[4] for c in ohlcv]
                        ema = sum(closes[-20:]) / 20 if len(closes) >= 20 else None
                        
                        # Open Interest (Try fetch, handle if not supported)
                        oi = None
                        oi_change = None
                        try:
                            if hasattr(client, 'fetch_open_interest'):
                                oi_data = client.fetch_open_interest(resolve_symbol(symbol))
                                oi = oi_data.get('openInterestAmount')
                        except Exception: pass
                        
                    except Exception as e:
                        logger.error(f"Error fetching {symbol}: {e}")
                        continue
                        
                    # 2. Analyze
                    if funding_rate is None or price is None: continue
                    
                    result = self.analyzer.analyze(symbol, funding_rate, oi, oi_change, price, ema)
                    
                    if result:
                        # Check direction filter
                        allowed_dirs = self.config.get("signal", {}).get("allowed_directions", ["BULLISH", "BEARISH"])
                        if result['direction'] not in allowed_dirs:
                            logger.debug(f"Skipping {result['direction']} signal for {symbol} - not in allowed_directions")
                            continue

                        # 3. Targets
                        atr = price * 0.02 # Simple proxy if calc fails
                        targets = self.analyzer.calculate_targets(price, result['direction'], atr, symbol)
                        if targets is None:
                            continue

                        signal = FundingSignal(
                            symbol=symbol, direction=result['direction'],
                            funding_rate=funding_rate, predicted_price=0,
                            entry=price, stop_loss=targets['sl'],
                            take_profit_1=targets['tp1'], take_profit_2=targets['tp2'],
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            funding_timestamp=funding_ts,
                            timeframe=timeframe, exchange=exchange, reasons=result['reasons']
                        )
                        
                        # 4. Alert & Track
                        cooldown = self.config.get("signal", {}).get("cooldown_minutes", 60)
                        if self.tracker.should_alert(symbol, exchange, cooldown, funding_ts=funding_ts):
                            self.tracker.check_reversal(symbol, signal.direction, self.notifier)
                            self.tracker.add_signal(signal)
                            self._send_alert(signal)
                            
                self.tracker.check_open_signals(self.clients, self.notifier)
                
                if self.health_monitor: self.health_monitor.record_cycle()
                
                if run_once: break
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                if self.health_monitor: self.health_monitor.record_error(str(e))
                if run_once: raise
                time.sleep(10)
        
        if self.health_monitor: self.health_monitor.send_shutdown_message()

    def _send_alert(self, signal: FundingSignal) -> None:
        msg = format_signal_message(
            bot_name="FUNDING",
            symbol=signal.symbol,
            direction=signal.direction,
            entry=signal.entry,
            stop_loss=signal.stop_loss,
            tp1=signal.take_profit_1,
            funding_rate=signal.funding_rate,
            reasons=signal.reasons,
            timeframe="15m",
        )
        if self.notifier: self.notifier.send_message(msg)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="funding_config.json")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    global logger
    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logging(log_level=log_level)
    
    # Load dotenv here too
    try:
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / ".env", override=True)
        load_dotenv(BASE_DIR.parent / ".env", override=True)
    except ImportError: pass
    
    bot = FundingBot(BASE_DIR / args.config)
    bot.run(run_once=args.once)

if __name__ == "__main__":
    main()