#!/usr/bin/env python3
"""Volume Node (VN) bot built on top of volume_profile.py without modifying it."""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict, cast

import ccxt  # type: ignore
import numpy as np
import sys

BASE_DIR = Path(__file__).resolve().parent

# Add parent directory to path for shared modules
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(BASE_DIR / ".env")
    load_dotenv(BASE_DIR.parent / ".env")
except ImportError:
    pass

import volume_profile as vp  # noqa: E402  pylint: disable=wrong-import-position

try:
    from notifier import TelegramNotifier  # type: ignore  # noqa: E402
    from health_monitor import HealthMonitor, RateLimiter  # type: ignore  # noqa: E402
    from signal_stats import SignalStats  # type: ignore  # noqa: E402
    from tp_sl_calculator import TPSLCalculator  # type: ignore  # noqa: E402
    from trade_config import get_config_manager  # type: ignore  # noqa: E402
    from rate_limit_handler import RateLimitHandler  # type: ignore  # noqa: E402
except ImportError as e:  # pragma: no cover - notifier is optional for local dry runs
    print(f"Import error: {e}")
    TelegramNotifier = None  # type: ignore
    TPSLCalculator = None  # type: ignore
    get_config_manager = None  # type: ignore
    HealthMonitor = None  # type: ignore
    RateLimiter = None  # type: ignore
    SignalStats = None  # type: ignore
    RateLimitHandler = None  # type: ignore


LOG_DIR = BASE_DIR / "logs"
STATE_FILE = BASE_DIR / "volume_vn_state.json"
SIGNALS_FILE = BASE_DIR / "volume_vn_signals.json"
WATCHLIST_FILE = BASE_DIR / "volume_watchlist.json"
STATS_FILE = LOG_DIR / "volume_stats.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "volume_vn_bot.log"),
    ],
)

logger = logging.getLogger("volume_vn")


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

class WatchItem(TypedDict, total=False):
    symbol: str
    timeframe: str
    exchange: str
    market_type: str


DEFAULT_WATCHLIST: List[WatchItem] = []  # populated via file or user-provided list


def ensure_watchlist() -> List[WatchItem]:
    if WATCHLIST_FILE.exists():
        try:
            data = json.loads(WATCHLIST_FILE.read_text())
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON in watchlist file: %s", exc)
            data = []
        normalized: List[WatchItem] = []
        for item in data if isinstance(data, list) else []:
            if not isinstance(item, dict):
                continue
            symbol_val = item.get("symbol")
            if not isinstance(symbol_val, str):
                continue
            exchange_val = item.get("exchange", "binanceusdm")
            exchange = exchange_val if isinstance(exchange_val, str) else "binanceusdm"
            cfg = EXCHANGE_CONFIG.get(exchange, EXCHANGE_CONFIG["binanceusdm"])
            market_type_val = item.get("market_type", cfg.get("default_market", "swap"))
            market_type = market_type_val if isinstance(market_type_val, str) else cfg.get("default_market", "swap")
            timeframe_val = item.get("timeframe", "5m")
            timeframe = timeframe_val if isinstance(timeframe_val, str) else "5m"
            normalized.append(
                {
                    "symbol": symbol_val.upper(),
                    "timeframe": timeframe,
                    "exchange": exchange,
                    "market_type": market_type,
                }
            )
        if normalized:
            return normalized

    WATCHLIST_FILE.write_text(json.dumps(DEFAULT_WATCHLIST, indent=2))
    return cast(List[WatchItem], DEFAULT_WATCHLIST)


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
class VolumeSignal:
    symbol: str  # e.g., "RIVER/USDT"
    exchange: str
    timeframe: str
    market_type: str
    direction: str
    entry: float
    stop_loss: float
    take_profit_primary: float
    take_profit_secondary: float
    confidence: float
    rationale: List[str]
    created_at: str

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


class VolumeAnalyzer:
    def __init__(self, candle_limit: int = 200):
        self.candle_limit = candle_limit
        self.clients: Dict[str, ccxt.Exchange] = {}
        self.rate_limiter = RateLimitHandler(base_delay=0.5, max_retries=5) if RateLimitHandler else None

    def get_client(self, exchange_key: str, market_type: str) -> ccxt.Exchange:
        if exchange_key not in EXCHANGE_CONFIG:
            raise ValueError(f"Unsupported exchange: {exchange_key}")

        client_key = f"{exchange_key}:{market_type}"
        if client_key not in self.clients:
            cfg = EXCHANGE_CONFIG[exchange_key]
            params = json.loads(json.dumps(cfg["params"])) if isinstance(cfg["params"], dict) else {}
            options = params.get("options", {}).copy()
            options["defaultType"] = market_type
            params["options"] = options
            self.clients[client_key] = cfg["factory"](params)  # type: ignore[call-arg]
        return self.clients[client_key]

    def analyze(self, symbol: str, timeframe: str, exchange_key: str, market_type: str) -> Dict[str, Any]:
        client = self.get_client(exchange_key, market_type)
        market_symbol = resolve_symbol(symbol, market_type)
        logger.debug("Fetching OHLCV for %s %s on %s", market_symbol, timeframe, exchange_key)
        if self.rate_limiter:
            ohlcv = self.rate_limiter.execute(client.fetch_ohlcv, market_symbol, timeframe, limit=self.candle_limit)
        else:
            ohlcv = client.fetch_ohlcv(market_symbol, timeframe, limit=self.candle_limit)

        try:
            opens = [float(x[1]) for x in ohlcv]
            highs = [float(x[2]) for x in ohlcv]
            lows = [float(x[3]) for x in ohlcv]
            closes = [float(x[4]) for x in ohlcv]
            volumes = [float(x[5]) for x in ohlcv]
        except Exception:
            return {}

        if len(closes) < 20:
            return {}

        current_price = closes[-1]
        ema20 = sum(closes[-20:]) / 20
        vp_result_raw = vp.calculate_volume_profile(highs[-100:], lows[-100:], closes[-100:], volumes[-100:])
        vp_result = cast(Dict[str, Any], vp_result_raw)
        rsi = float(vp.calculate_rsi(np.array(closes)))
        pattern_raw = vp.detect_candlestick_pattern(opens, highs, lows, closes)
        pattern = pattern_raw if isinstance(pattern_raw, str) else None

        avg_volume = sum(volumes[-20:]) / 20
        current_volume = volumes[-1]
        volume_spike = current_volume > avg_volume * 1.5

        poc = float(vp_result.get("poc", current_price)) if isinstance(vp_result.get("poc"), (int, float)) else current_price
        vah = float(vp_result.get("vah", current_price)) if isinstance(vp_result.get("vah"), (int, float)) else current_price
        val = float(vp_result.get("val", current_price)) if isinstance(vp_result.get("val"), (int, float)) else current_price
        row_height = float(vp_result.get("row_height", 0.0)) if isinstance(vp_result.get("row_height"), (int, float)) else 0.0

        price_vs_poc = "ABOVE" if current_price > poc else "BELOW"
        in_value_area = val <= current_price <= vah

        nearest_hvn, near_hvn = self._nearest_hvn(current_price, vp_result, row_height)

        recent_candles = list(zip(opens[-20:], closes[-20:], volumes[-20:]))

        long_factors, short_factors = self._build_factors(
            current_price,
            ema20,
            rsi,
            pattern,
            volume_spike,
            near_hvn,
            vp_result,
            recent_candles,
        )

        signal, setup = self._build_signal(
            current_price,
            vp_result,
            long_factors,
            short_factors,
            symbol,
        )

        return {
            "symbol": market_symbol,
            "base_symbol": symbol.split("/")[0] if "/" in symbol else symbol,
            "watch_symbol": symbol,
            "timeframe": timeframe,
            "current_price": current_price,
            "ema20": ema20,
            "rsi": rsi,
            "current_volume": current_volume,
            "volume_spike": volume_spike,
            "avg_volume": avg_volume,
            "pattern": pattern,
            "price_vs_poc": price_vs_poc,
            "in_value_area": in_value_area,
            "nearest_hvn": nearest_hvn,
            "volume_profile": vp_result,
            "long_factors": long_factors,
            "short_factors": short_factors,
            "signal": signal,
            "setup": setup,
            "exchange": exchange_key,
            "market_type": market_type,
            "row_height": row_height,
        }

    @staticmethod
    def _nearest_hvn(current_price: float, vp_result: Dict[str, Any], row_height: float):
        nearest: Optional[float] = None
        min_dist = float("inf")
        hvn_levels = vp_result.get("hvn_levels")
        if isinstance(hvn_levels, list):
            for entry in hvn_levels:
                if isinstance(entry, (list, tuple)) and len(entry) >= 1:
                    hvn_price = entry[0]
                    if isinstance(hvn_price, (int, float)):
                        dist = abs(current_price - hvn_price)
                        if dist < min_dist:
                            min_dist = dist
                            nearest = float(hvn_price)
        near_hvn = min_dist < row_height * 2 if nearest is not None else False
        return nearest, near_hvn

    @staticmethod
    def _build_factors(
        current_price: float,
        ema20: float,
        rsi: float,
        pattern: Optional[str],
        volume_spike: bool,
        near_hvn: bool,
        vp_result: Dict[str, Any],
        recent_candles: List[Tuple[float, float, float]],
    ) -> tuple:
        long_factors: List[str] = []
        short_factors: List[str] = []

        if current_price > ema20:
            long_factors.append("Price > EMA20")
        else:
            short_factors.append("Price < EMA20")

        if 45 < rsi < 80:
            long_factors.append(f"RSI favorable ({rsi:.1f})")
        elif 20 < rsi < 55:
            short_factors.append(f"RSI favorable ({rsi:.1f})")

        if rsi > 50 and current_price > ema20:
            long_factors.append("Bullish Momentum (RSI Trend)")

        poc_val = vp_result.get("poc")
        poc = float(poc_val) if isinstance(poc_val, (int, float)) else current_price

        if near_hvn:
            if current_price < poc:
                long_factors.append("Near HVN below POC (support)")
            else:
                short_factors.append("Near HVN above POC (resistance)")

        if pattern:
            if "BULLISH" in pattern:
                long_factors.append(pattern)
            else:
                short_factors.append(pattern)

        if volume_spike:
            long_factors.append("Volume spike")

        if VolumeAnalyzer._has_buying_pressure(recent_candles):
            long_factors.append("Buying Pressure Dominant")

        return long_factors, short_factors

    @staticmethod
    def _has_buying_pressure(
        candles: List[Tuple[float, float, float]],
        sample_size: int = 3,
        threshold: float = 1.2,
    ) -> bool:
        if len(candles) < sample_size * 2:
            return False

        greens: List[float] = []
        reds: List[float] = []
        for open_price, close_price, volume in reversed(candles):
            if close_price > open_price:
                greens.append(volume)
            elif close_price < open_price:
                reds.append(volume)
            if len(greens) >= sample_size and len(reds) >= sample_size:
                break

        if len(greens) < sample_size or len(reds) < sample_size:
            return False

        green_avg = sum(greens[:sample_size]) / sample_size
        red_avg = sum(reds[:sample_size]) / sample_size
        return green_avg > red_avg * threshold

    @staticmethod
    def _build_signal(
        current_price: float,
        vp_result: Dict[str, Any],
        long_factors: List[str],
        short_factors: List[str],
        symbol: str = "",
    ):
        if len(long_factors) >= 3 and len(long_factors) > len(short_factors):
            val = float(vp_result.get("val", current_price)) if isinstance(vp_result.get("val"), (int, float)) else current_price
            row_height = float(vp_result.get("row_height", 0.0)) if isinstance(vp_result.get("row_height"), (int, float)) else 0.0
            poc = float(vp_result.get("poc", current_price)) if isinstance(vp_result.get("poc"), (int, float)) else current_price
            raw_sl = val - row_height
            # Wider stop loss - was too tight at 0.1%, now 1.5% to avoid premature stops
            custom_sl = min(raw_sl, current_price * 0.985)  # Changed from 0.999 (0.1% risk) to 0.985 (1.5% risk)
            
            if TPSLCalculator is not None:
                if get_config_manager is not None:
                    try:
                        config_mgr = get_config_manager()
                        risk_config = config_mgr.get_effective_risk("volume_bot", symbol)
                        tp1_mult = risk_config.tp1_atr_multiplier
                        tp2_mult = risk_config.tp2_atr_multiplier
                        min_rr = risk_config.min_risk_reward
                    except Exception:
                        tp1_mult, tp2_mult, min_rr = 2.0, 3.0, 1.5
                else:
                    tp1_mult, tp2_mult, min_rr = 2.0, 3.0, 1.5
                
                # Increased minimum risk from 0.3% to 1.0% for more reasonable position sizing
                risk = max(current_price - custom_sl, current_price * 0.01)
                calculator = TPSLCalculator(min_risk_reward=min_rr)
                levels = calculator.calculate(
                    entry=current_price,
                    direction="LONG",
                    atr=risk,  # Use risk as ATR proxy
                    tp1_multiplier=tp1_mult,
                    tp2_multiplier=tp2_mult,
                    custom_sl=custom_sl,
                )
                
                if levels.is_valid:
                    tp1 = max(poc, float(levels.take_profit_1))
                    tp2 = max(tp1 + risk, float(levels.take_profit_2))
                    return "LONG", {"entry": current_price, "sl": custom_sl, "tp1": tp1, "tp2": tp2}
            
            # Fallback with improved risk sizing
            risk = max(current_price - custom_sl, current_price * 0.01)
            tp1 = max(poc, current_price + risk * 2)  # 2R for TP1
            tp2 = max(tp1 + risk, current_price + risk * 3)  # 3R for TP2
            return "LONG", {"entry": current_price, "sl": custom_sl, "tp1": tp1, "tp2": tp2}

        if len(short_factors) >= 3 and len(short_factors) > len(long_factors):
            vah = float(vp_result.get("vah", current_price)) if isinstance(vp_result.get("vah"), (int, float)) else current_price
            row_height = float(vp_result.get("row_height", 0.0)) if isinstance(vp_result.get("row_height"), (int, float)) else 0.0
            poc = float(vp_result.get("poc", current_price)) if isinstance(vp_result.get("poc"), (int, float)) else current_price
            raw_sl = vah + row_height
            # Wider stop loss for SHORT - was too tight at 0.1%, now 1.5% to avoid premature stops
            custom_sl = max(raw_sl, current_price * 1.015)  # Changed from 1.001 (0.1% risk) to 1.015 (1.5% risk)
            
            if TPSLCalculator is not None:
                if get_config_manager is not None:
                    try:
                        config_mgr = get_config_manager()
                        risk_config = config_mgr.get_effective_risk("volume_bot", symbol)
                        tp1_mult = risk_config.tp1_atr_multiplier
                        tp2_mult = risk_config.tp2_atr_multiplier
                        min_rr = risk_config.min_risk_reward
                    except Exception:
                        tp1_mult, tp2_mult, min_rr = 2.0, 3.0, 1.5
                else:
                    tp1_mult, tp2_mult, min_rr = 2.0, 3.0, 1.5
                
                # Increased minimum risk from 0.3% to 1.0% for more reasonable position sizing
                risk = max(custom_sl - current_price, current_price * 0.01)
                calculator = TPSLCalculator(min_risk_reward=min_rr)
                levels = calculator.calculate(
                    entry=current_price,
                    direction="SHORT",
                    atr=risk,
                    tp1_multiplier=tp1_mult,
                    tp2_multiplier=tp2_mult,
                    custom_sl=custom_sl,
                )
                
                if levels.is_valid:
                    tp1 = min(poc, float(levels.take_profit_1))
                    tp2 = min(tp1 - risk, float(levels.take_profit_2))
                    return "SHORT", {"entry": current_price, "sl": custom_sl, "tp1": tp1, "tp2": tp2}
            
            # Fallback with improved risk sizing
            risk = max(custom_sl - current_price, current_price * 0.01)
            tp1 = min(poc, current_price - risk * 2)  # 2R for TP1
            tp2 = min(tp1 - risk, current_price - risk * 3)  # 3R for TP2
            return "SHORT", {"entry": current_price, "sl": custom_sl, "tp1": tp1, "tp2": tp2}

        return "NEUTRAL", None


class SignalTracker:
    def __init__(self, analyzer: VolumeAnalyzer, stats=None):
        self.analyzer = analyzer
        self.stats = stats
        self.state: Dict[str, Any] = self._load_state()
        # Sync existing open signals to stats on startup
        self._sync_signals_to_stats()

    def _empty_state(self) -> Dict[str, Any]:
        return {"last_alerts": {}, "open_signals": {}}

    def _load_state(self) -> Dict[str, Any]:
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
            except json.JSONDecodeError:
                logger.warning("State file invalid, rebuilding from scratch")
                return self._empty_state()
            if isinstance(data, dict):
                if not isinstance(data.get("last_alerts"), dict):
                    data["last_alerts"] = {}
                if not isinstance(data.get("open_signals"), dict):
                    data["open_signals"] = {}
                return data
        return self._empty_state()

    def _save_state(self) -> None:
        STATE_FILE.write_text(json.dumps(self.state, indent=2))

    def _sync_signals_to_stats(self) -> None:
        """Sync open signals from state to stats (for signals created before stats integration)."""
        if not self.stats:
            return
        
        signals = self.state.get("open_signals", {}) if isinstance(self.state.get("open_signals", {}), dict) else {}
        stats_open = self.stats.data.get("open", {}) if isinstance(self.stats.data.get("open", {}), dict) else {}
        synced = 0
        
        for signal_id, payload in signals.items():
            if signal_id not in stats_open and isinstance(payload, dict):
                self.stats.record_open(
                    signal_id=signal_id,
                    symbol=payload.get("symbol", ""),
                    direction=payload.get("direction", "LONG"),
                    entry=payload.get("entry", 0),
                    created_at=payload.get("created_at", ""),
                    extra={
                        "timeframe": payload.get("timeframe", ""),
                        "exchange": payload.get("exchange", ""),
                        "display_symbol": payload.get("symbol", ""),
                    },
                )
                synced += 1
        
        if synced > 0:
            logger.info("Synced %d existing signals to stats tracker", synced)

    def should_alert(self, symbol: str, timeframe: str, exchange: str, cooldown_minutes: int) -> bool:
        key = f"{symbol}-{timeframe}-{exchange}"
        last_alerts = self.state.get("last_alerts")
        if not isinstance(last_alerts, dict):
            self.state["last_alerts"] = {}
            return True
        last_ts = last_alerts.get(key)
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

    def mark_alert(self, symbol: str, timeframe: str, exchange: str) -> None:
        key = f"{symbol}-{timeframe}-{exchange}"
        last_alerts = self.state.setdefault("last_alerts", {})
        if not isinstance(last_alerts, dict):
            last_alerts = {}
            self.state["last_alerts"] = last_alerts
        last_alerts[key] = datetime.now(timezone.utc).isoformat()
        self._save_state()

    def has_open_signal(self, symbol: str) -> bool:
        """Check if there's already an open signal for this symbol."""
        signals = self.state.get("open_signals", {}) if isinstance(self.state.get("open_signals", {}), dict) else {}
        for signal_data in signals.values():
            if signal_data.get("symbol") == symbol:
                return True
        return False

    def cleanup_stale_signals(self, max_age_hours: int = 24) -> int:
        """Stale cleanup disabled: retain signals for full stats history."""
        return 0

    def add_signal(self, signal: VolumeSignal) -> bool:
        """Add a new signal. Returns False if duplicate exists for symbol."""
        # Check for duplicate
        if self.has_open_signal(signal.symbol):
            logger.info("Skipping duplicate signal for %s - already have open position", signal.symbol)
            return False
        
        signals = self.state.setdefault("open_signals", {})
        if not isinstance(signals, dict):
            signals = {}
            self.state["open_signals"] = signals
        signal_id = f"{signal.symbol}-{signal.timeframe}-{signal.exchange}-{signal.created_at}"
        signals[signal_id] = signal.as_dict()
        self._save_state()
        
        # Record signal opening in stats
        if self.stats:
            self.stats.record_open(
                signal_id=signal_id,
                symbol=signal.symbol,
                direction=signal.direction,
                entry=signal.entry,
                created_at=signal.created_at,
                extra={
                    "timeframe": signal.timeframe,
                    "exchange": signal.exchange,
                    "display_symbol": signal.symbol,
                },
            )
        
        return True

    def check_open_signals(self, notifier) -> None:
        # First, cleanup stale signals (older than 24 hours)
        stale_count = self.cleanup_stale_signals(max_age_hours=24)
        if stale_count > 0:
            logger.info("Cleaned up %d stale signals", stale_count)
        
        signals = self.state.get("open_signals", {}) if isinstance(self.state.get("open_signals", {}), dict) else {}
        if not signals:
            return

        updated = False
        for signal_id, payload in list(signals.items()):
            if not isinstance(payload, dict):
                signals.pop(signal_id, None)
                updated = True
                continue
            market_type_val = payload.get("market_type", "swap")
            market_type = market_type_val if isinstance(market_type_val, str) else "swap"
            exchange_val = payload.get("exchange", "binanceusdm")
            if not isinstance(exchange_val, str):
                signals.pop(signal_id, None)
                updated = True
                continue
            symbol_val = payload.get("symbol")
            if not isinstance(symbol_val, str):
                signals.pop(signal_id, None)
                updated = True
                continue
            market_symbol = resolve_symbol(symbol_val, market_type)
            try:
                client = self.analyzer.get_client(exchange_val, market_type)
                if self.analyzer.rate_limiter:
                    ticker = self.analyzer.rate_limiter.execute(client.fetch_ticker, market_symbol)
                else:
                    ticker = client.fetch_ticker(market_symbol)
                current_price = ticker.get("last") if isinstance(ticker, dict) else None
                if current_price is None and isinstance(ticker, dict):
                    current_price = ticker.get("close")
            except Exception as exc:  # pragma: no cover - network
                logger.warning("Ticker fetch failed for %s: %s", market_symbol, exc)
                continue

            if not isinstance(current_price, (int, float)):
                continue

            direction = payload.get("direction")
            tp1_raw = payload.get("take_profit_primary")
            tp2_raw = payload.get("take_profit_secondary")
            sl_raw = payload.get("stop_loss")
            entry_raw = payload.get("entry")
            if not isinstance(direction, str):
                signals.pop(signal_id, None)
                updated = True
                continue
            if not all(isinstance(v, (int, float, str)) for v in (tp1_raw, tp2_raw, sl_raw, entry_raw)):
                signals.pop(signal_id, None)
                updated = True
                continue
            try:
                tp1 = float(tp1_raw)  # type: ignore[arg-type]
                tp2 = float(tp2_raw)  # type: ignore[arg-type]
                sl = float(sl_raw)    # type: ignore[arg-type]
                entry = float(entry_raw)  # type: ignore[arg-type]
            except Exception:
                signals.pop(signal_id, None)
                updated = True
                continue

            if direction == "LONG":
                hit_tp1 = current_price >= tp1
                hit_tp2 = current_price >= tp2
                hit_sl = current_price <= sl
            else:
                hit_tp1 = current_price <= tp1
                hit_tp2 = current_price <= tp2
                hit_sl = current_price >= sl

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
                        exit_price=current_price,
                        result=result,
                    )
                    if stats_record:
                        summary_message = self.stats.build_summary_message(stats_record)
                    else:
                        self.stats.discard(signal_id)
                
                if summary_message:
                    message = summary_message
                else:
                    message = (
                        f"🎯 {symbol_val} {payload.get('timeframe', '')} {direction} {result} hit!\n"
                        f"Entry {entry:.6f} | Last {current_price:.6f}\n"
                        f"SL {sl:.6f} | TP1 {tp1:.6f} | TP2 {tp2:.6f}"
                    )
                
                if notifier:
                    notifier.send_message(message)
                logger.info("Signal %s closed with %s", signal_id, result)
                signals.pop(signal_id, None)
                updated = True

        if updated:
            self._save_state()


class VolumeVNBOT:
    def __init__(self, cooldown_minutes: int = 30):
        self.watchlist: List[WatchItem] = ensure_watchlist()
        self.analyzer = VolumeAnalyzer()
        self.cooldown = cooldown_minutes
        self.notifier = self._build_notifier()
        self.stats = SignalStats("Volume Bot", STATS_FILE) if SignalStats else None
        self.tracker = SignalTracker(self.analyzer, stats=self.stats)
        self.health_monitor = HealthMonitor("Volume Bot", self.notifier, heartbeat_interval=3600) if HealthMonitor and self.notifier else None
        self.rate_limiter = RateLimiter(calls_per_minute=60, backoff_file=LOG_DIR / "rate_limiter.json") if RateLimiter else None
        self.exchange_backoff: Dict[str, float] = {}
        self.exchange_delay: Dict[str, float] = {}

    def _build_notifier(self):
        if TelegramNotifier is None:
            logger.warning("Telegram notifier unavailable; running in console-only mode")
            return None

        bot_token = os.getenv("TELEGRAM_BOT_TOKEN_VOLUME") or os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not bot_token or not chat_id:
            logger.warning("Telegram credentials missing; alerts will only log locally")
            return None

        signals_log = str(SIGNALS_FILE)
        return TelegramNotifier(bot_token=bot_token, chat_id=chat_id, signals_log_file=signals_log)

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

    def run_cycle(self, run_once: bool = True, delay_seconds: int = 5) -> None:
        logger.info("Starting Volume VN cycle for %d pairs", len(self.watchlist))
        
        # Send startup notification
        if self.health_monitor:
            self.health_monitor.send_startup_message()
        
        try:
            while True:
                try:
                    for item in self.watchlist:
                        symbol_val = item.get("symbol") if isinstance(item, dict) else None
                        if not isinstance(symbol_val, str):
                            continue
                        timeframe_val = item.get("timeframe", "5m") if isinstance(item, dict) else "5m"
                        exchange_val = item.get("exchange", "binanceusdm") if isinstance(item, dict) else "binanceusdm"
                        market_type_val = item.get("market_type") if isinstance(item, dict) else None
                        exchange = exchange_val if isinstance(exchange_val, str) else "binanceusdm"
                        market_type = (
                            market_type_val
                            if isinstance(market_type_val, str)
                            else EXCHANGE_CONFIG.get(exchange, {}).get("default_market", "swap")
                        )
                        symbol = symbol_val
                        timeframe = timeframe_val if isinstance(timeframe_val, str) else "5m"

                        if self._backoff_active(exchange):
                            logger.debug("Backoff active for %s; skipping %s", exchange, symbol)
                            continue

                        # Apply rate limiting
                        if self.rate_limiter:
                            self.rate_limiter.wait_if_needed()

                        try:
                            snapshot = self.analyzer.analyze(symbol, timeframe, exchange, market_type)
                            if self.rate_limiter:
                                self.rate_limiter.record_success(f"{exchange}_{symbol}")
                        except Exception as exc:  # pragma: no cover - network
                            logger.error("Failed to analyze %s %s on %s: %s", symbol, timeframe, exchange, exc)
                            if self._is_rate_limit_error(exc):
                                self._register_backoff(exchange)
                            if self.rate_limiter:
                                self.rate_limiter.record_error(f"{exchange}_{symbol}")
                            if self.health_monitor:
                                self.health_monitor.record_error(f"API error for {symbol}: {exc}")
                            continue

                        if not isinstance(snapshot, dict) or snapshot.get("signal") == "NEUTRAL":
                            logger.debug("%s %s neutral", symbol, timeframe)
                            continue

                        if not self.tracker.should_alert(symbol, timeframe, exchange, self.cooldown):
                            logger.debug("Cooldown active for %s %s %s", symbol, timeframe, exchange)
                            continue

                        # MAX OPEN SIGNALS LIMIT: Prevent overexposure
                        MAX_OPEN_SIGNALS = 50
                        open_signals = self.tracker.state.get("open_signals", {}) if isinstance(self.tracker.state.get("open_signals", {}), dict) else {}
                        current_open = len(open_signals)
                        if current_open >= MAX_OPEN_SIGNALS:
                            logger.info(
                                "Max open signals limit reached (%d/%d). Skipping new signal for %s",
                                current_open, MAX_OPEN_SIGNALS, symbol
                            )
                            continue
                        
                        signal_payload = self._build_signal(snapshot)
                        
                        # SIGNAL REVERSAL DETECTION: Warn if opposite direction signal detected
                        self._check_signal_reversal(symbol, signal_payload.direction)
                        
                        self._dispatch_signal(signal_payload, snapshot)
                        self.tracker.mark_alert(symbol, timeframe, exchange)
                        self.tracker.add_signal(signal_payload)

                        time.sleep(delay_seconds)

                    self.tracker.check_open_signals(self.notifier)
                    
                    # Record successful cycle
                    if self.health_monitor:
                        self.health_monitor.record_cycle()

                    if run_once:
                        break
                    logger.info("Cycle complete; sleeping 60 seconds")
                    time.sleep(60)
                except Exception as exc:
                    logger.error(f"Error in cycle: {exc}")
                    if self.health_monitor:
                        self.health_monitor.record_error(str(exc))
                    if run_once:
                        raise
                    time.sleep(10)  # Brief pause before retry
        finally:
            # Send shutdown notification
            if self.health_monitor:
                self.health_monitor.send_shutdown_message()

    @staticmethod
    def _compute_confidence(rationale: List[str]) -> float:
        weight = 0.0
        for factor in rationale:
            text = factor.lower()
            if "price > ema20" in text or "price < ema20" in text:
                weight += 1.0
            elif "near hvn" in text:
                weight += 2.0
            elif "volume spike" in text:
                weight += 2.0
            elif "buying pressure" in text:
                weight += 2.0
            else:
                weight += 1.0
        return round(weight, 2)

    def _build_signal(self, snapshot: Dict[str, Any]) -> VolumeSignal:
        setup = snapshot.get("setup") or {}
        watch_symbol = snapshot.get("watch_symbol")
        base_symbol = snapshot.get("base_symbol", "")
        display_symbol = watch_symbol if isinstance(watch_symbol, str) else base_symbol
        exchange = snapshot.get("exchange", "")
        timeframe = snapshot.get("timeframe", "")
        market_type = snapshot.get("market_type", "swap")
        direction = snapshot.get("signal", "")
        entry_val = setup.get("entry", snapshot.get("current_price", 0.0)) if isinstance(setup, dict) else snapshot.get("current_price", 0.0)
        sl_val = setup.get("sl", snapshot.get("current_price", 0.0)) if isinstance(setup, dict) else snapshot.get("current_price", 0.0)
        tp1_val = setup.get("tp1", snapshot.get("current_price", 0.0)) if isinstance(setup, dict) else snapshot.get("current_price", 0.0)
        tp2_val = setup.get("tp2", snapshot.get("current_price", 0.0)) if isinstance(setup, dict) else snapshot.get("current_price", 0.0)
        long_factors = snapshot.get("long_factors", []) if isinstance(snapshot.get("long_factors", []), list) else []
        short_factors = snapshot.get("short_factors", []) if isinstance(snapshot.get("short_factors", []), list) else []
        rationale = long_factors if direction == "LONG" else short_factors
        confidence_score = self._compute_confidence(rationale)

        return VolumeSignal(
            symbol=str(display_symbol),
            exchange=str(exchange),
            timeframe=str(timeframe),
            market_type=str(market_type),
            direction=str(direction),
            entry=float(entry_val),
            stop_loss=float(sl_val),
            take_profit_primary=float(tp1_val),
            take_profit_secondary=float(tp2_val),
            confidence=confidence_score,
            rationale=[str(r) for r in rationale],
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def _dispatch_signal(self, signal: VolumeSignal, snapshot: Dict[str, object]) -> None:
        message = self._format_message(signal, snapshot)
        if self.notifier:
            self.notifier.send_message(message)
        else:
            logger.info("Alert:\n%s", message)

    def _format_message(self, signal: VolumeSignal, snapshot: Dict[str, Any]) -> str:
        emoji = "🟢" if signal.direction == "LONG" else "🔴"
        metrics = snapshot.get("volume_profile", {})
        poc = metrics.get("poc", 0.0) if isinstance(metrics, dict) else 0.0
        vah = metrics.get("vah", 0.0) if isinstance(metrics, dict) else 0.0
        val = metrics.get("val", 0.0) if isinstance(metrics, dict) else 0.0
        perf_line = self._symbol_perf_line(signal.symbol)
        tp_sl_line = self._symbol_tp_sl_line(signal.symbol)

        message = [
            f"{emoji} <b>{signal.direction} {signal.symbol}</b>",
            "",
            f"💰 Entry: <code>{signal.entry:.6f}</code>",
            f"🛑 Stop Loss: <code>{signal.stop_loss:.6f}</code>",
            f"🎯 Take Profit 1: <code>{signal.take_profit_primary:.6f}</code>",
            f"🎯 Take Profit 2: <code>{signal.take_profit_secondary:.6f}</code>",
            "",
            f"🕒 Timeframe: {signal.timeframe} | 🏦 Exchange: {signal.exchange.upper()}",
            f"📊 POC/VAH/VAL: <code>{float(poc):.6f}</code> / <code>{float(vah):.6f}</code> / <code>{float(val):.6f}</code>",
        ]

        if perf_line:
            message.append(perf_line)

        if tp_sl_line:
            message.append(tp_sl_line)

        if signal.rationale:
            message.append("📝 Factors: " + ", ".join(signal.rationale))

        return "\n".join(message)

    def _symbol_perf_line(self, symbol: str) -> Optional[str]:
        history = self.stats.data.get("history", []) if self.stats and isinstance(self.stats.data, dict) else []
        if not isinstance(history, list) or not history:
            return None
        symbol_key = symbol if "/" in symbol else f"{symbol}/USDT"
        tp = 0
        sl = 0
        for entry in history:
            if not isinstance(entry, dict):
                continue
            if entry.get("symbol") != symbol_key:
                continue
            result = str(entry.get("result", ""))
            if result.startswith("TP"):
                tp += 1
            elif result == "SL":
                sl += 1
        
        # Calculate win rate
        total = tp + sl
        win_rate = (tp / total * 100) if total > 0 else 0.0
        
        return f"📈 {symbol_key} history: TP {tp} / SL {sl} (Win rate: {win_rate:.1f}%)"

    def _symbol_tp_sl_line(self, symbol: str) -> Optional[str]:
        if not self.stats or not isinstance(self.stats.data, dict):
            return None
        history = self.stats.data.get("history", [])
        if not isinstance(history, list) or not history:
            return None
        symbol_key = symbol if "/" in symbol else f"{symbol}/USDT"
        tp1 = tp2 = sl = 0
        for entry in history:
            if not isinstance(entry, dict):
                continue
            if entry.get("symbol") != symbol_key:
                continue
            result = str(entry.get("result", ""))
            if result == "TP1":
                tp1 += 1
            elif result == "TP2":
                tp2 += 1
            elif result == "SL":
                sl += 1
        if tp1 == 0 and tp2 == 0 and sl == 0:
            return None
        
        # Calculate win rate
        total = tp1 + tp2 + sl
        win_rate = ((tp1 + tp2) / total * 100) if total > 0 else 0.0
        
        return f"📊 TP/SL history: TP1 {tp1} | TP2 {tp2} | SL {sl} (Win rate: {win_rate:.1f}%)"
    
    def _check_signal_reversal(self, symbol: str, new_direction: str) -> None:
        """Check if new signal is opposite direction to open position and warn user."""
        open_signals = self.tracker.state.get("open_signals", {}) if isinstance(self.tracker.state.get("open_signals", {}), dict) else {}
        for signal_id, signal_data in open_signals.items():
            if not isinstance(signal_data, dict):
                continue
            signal_symbol = signal_data.get("symbol", "")
            if signal_symbol == symbol or signal_symbol == f"{symbol}/USDT":
                open_direction = signal_data.get("direction", "")
                opposite = (
                    (new_direction == "LONG" and open_direction == "SHORT") or
                    (new_direction == "SHORT" and open_direction == "LONG")
                )
                if opposite:
                    warning_msg = (
                        f"⚠️ <b>SIGNAL REVERSAL DETECTED</b> ⚠️\n\n"
                        f"<b>Symbol:</b> {symbol}\n"
                        f"<b>Open Position:</b> {open_direction}\n"
                        f"<b>New Signal:</b> {new_direction}\n\n"
                        f"💡 <b>Action:</b> Consider exiting your {open_direction} position!\n"
                        f"🔄 <b>Market may be reversing</b>\n\n"
                        f"🆔 Open Signal: {signal_id}\n"
                        f"⏰ {datetime.now(timezone.utc).isoformat()}"
                    )
                    if self.notifier:
                        self.notifier.send_message(warning_msg)
                    logger.info("Signal reversal detected for %s: %s -> %s", symbol, open_direction, new_direction)
                    break


def main() -> None:
    parser = argparse.ArgumentParser(description="Volume VN bot")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument(
        "--cooldown",
        type=int,
        default=30,
        help="Minutes before repeating alert per pair",
    )
    parser.add_argument("--track", action="store_true", help="Only run tracker checks")
    args = parser.parse_args()

    bot = VolumeVNBOT(cooldown_minutes=args.cooldown)

    if args.track:
        bot.tracker.check_open_signals(bot.notifier)
        return

    bot.run_cycle(run_once=not args.loop)


if __name__ == "__main__":
    main()
