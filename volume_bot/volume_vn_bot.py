#!/usr/bin/env python3
"""Volume Node (VN) bot built on top of volume_profile.py without modifying it.

REFACTORED VERSION:
- Fixed look-ahead bias in volume calculations
- Added proper error handling and timeouts
- Implemented file locking for state persistence
- Fixed signal duplicate detection logic
- Enabled stale signal cleanup with archiving
- Externalized configuration to config.py
- Added exchange credential validation
"""

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
from types import FrameType
from typing import Any, Dict, List, Optional, Tuple, TypedDict, cast

import ccxt  # type: ignore[import-untyped]
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

# Add parent directory to path for shared modules
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))

try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
    load_dotenv(BASE_DIR.parent / ".env")
except ImportError:
    pass

import volume_profile as vp  # noqa: E402  pylint: disable=wrong-import-position

# Safe imports with independent error handling
from safe_import import safe_import_multiple, safe_import
from message_templates import format_signal_message, format_result_message


# Required imports (fail fast if missing)
from notifier import TelegramNotifier
from signal_stats import SignalStats
from tp_sl_calculator import TPSLCalculator
from trade_config import get_config_manager

# Optional imports (safe fallback)
from safe_import import safe_import
HealthMonitor = safe_import('health_monitor', 'HealthMonitor')
RateLimiter = None  # Disabled for testing
RateLimitHandler = safe_import('rate_limit_handler', 'RateLimitHandler')

# Import configuration module
VolumeConfig = safe_import('config', 'VolumeConfig')
load_config = safe_import('config', 'load_config')


LOG_DIR = BASE_DIR / "logs"
STATE_FILE = BASE_DIR / "volume_vn_state.json"
SIGNALS_FILE = BASE_DIR / "volume_vn_signals.json"
WATCHLIST_FILE = BASE_DIR / "volume_watchlist.json"
STATS_FILE = LOG_DIR / "volume_stats.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

def setup_logging(log_level: str = "INFO", enable_detailed: bool = False) -> logging.Logger:
    """Setup enhanced logging with rotation and detailed formatting."""
    from logging.handlers import RotatingFileHandler

    # Create formatters
    detailed_format = "%(asctime)s | %(levelname)-8s | [%(name)s:%(funcName)s:%(lineno)d] | %(message)s"
    simple_format = "%(asctime)s | %(levelname)-8s | %(message)s"

    formatter = logging.Formatter(detailed_format if enable_detailed else simple_format)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler with color-coded levels
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # File handler with rotation (10MB max, keep 5 backups)
    file_handler = RotatingFileHandler(
        LOG_DIR / "volume_vn_bot.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file

    # Error file handler (separate file for errors)
    error_handler = RotatingFileHandler(
        LOG_DIR / "volume_vn_errors.log",
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3
    )
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)

    return logging.getLogger("volume_vn")

# Default setup (will be reconfigured in main())
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
    return DEFAULT_WATCHLIST


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
    def __init__(self, config: Optional[Any] = None) -> None:
        self.config: Optional[Any] = config
        self.candle_limit: int = getattr(getattr(config, 'analysis', None), 'candle_limit', 200) if config else 200
        self.request_timeout: int = getattr(getattr(config, 'analysis', None), 'request_timeout_seconds', 30) if config else 30
        self.clients: Dict[str, Any] = {}
        self.rate_limiter: Optional[Any] = RateLimitHandler(
            base_delay=getattr(getattr(config, 'rate_limit', None), 'base_delay_seconds', 0.5) if config else 0.5,
            max_retries=getattr(getattr(config, 'rate_limit', None), 'max_retries', 5) if config else 5
        ) if RateLimitHandler else None

    def get_client(self, exchange_key: str, market_type: str) -> ccxt.Exchange:
        if exchange_key not in EXCHANGE_CONFIG:
            raise ValueError(f"Unsupported exchange: {exchange_key}")

        client_key = f"{exchange_key}:{market_type}"
        if client_key not in self.clients:
            cfg = EXCHANGE_CONFIG[exchange_key]
            params = json.loads(json.dumps(cfg["params"])) if isinstance(cfg["params"], dict) else {}
            
            # Add timeout configuration (FIX: Issue #3 - Add request timeouts)
            params['timeout'] = self.request_timeout * 1000  # ccxt uses milliseconds
            
            options = params.get("options", {}).copy()
            options["defaultType"] = market_type
            params["options"] = options
            
            # Add credentials if available
            if self.config:
                get_creds = getattr(self.config, 'get_exchange_credentials', None)
                if get_creds:
                    creds = get_creds(exchange_key)
                    if getattr(creds, 'is_configured', False):
                        params['apiKey'] = getattr(creds, 'api_key', None)
                        params['secret'] = getattr(creds, 'secret', None)

            self.clients[client_key] = cfg["factory"](params)
        return self.clients[client_key]
    
    def validate_exchange_credentials(self, exchange_key: str, market_type: str = "swap") -> bool:
        """Validate exchange API credentials (FIX: Security - API Key Validation)."""
        try:
            client = self.get_client(exchange_key, market_type)
            # Try to fetch balance (requires authentication)
            balance = client.fetch_balance()
            logger.info(f"✅ {exchange_key} credentials validated successfully")
            return True
        except ccxt.AuthenticationError as e:
            logger.error(f"❌ Invalid API credentials for {exchange_key}: {e}")
            return False
        except ccxt.NetworkError as e:
            logger.warning(f"⚠️ Network error validating {exchange_key} credentials: {e}")
            return False
        except Exception as e:
            logger.warning(f"⚠️ Could not validate {exchange_key} credentials: {e}")
            return False

    def analyze(self, symbol: str, timeframe: str, exchange_key: str, market_type: str) -> Dict[str, Any]:
        """Analyze symbol and generate trading signal."""
        logger.debug("🔍 Analyzing %s on %s (%s, %s)", symbol, exchange_key, timeframe, market_type)

        client = self.get_client(exchange_key, market_type)
        market_symbol = resolve_symbol(symbol, market_type)
        logger.debug("📊 Fetching OHLCV for %s %s on %s", market_symbol, timeframe, exchange_key)
        
        # FIX: Issue #3 - Add specific exception handling for network requests
        try:
            if self.rate_limiter:
                ohlcv = self.rate_limiter.execute(client.fetch_ohlcv, market_symbol, timeframe, limit=self.candle_limit)
            else:
                ohlcv = client.fetch_ohlcv(market_symbol, timeframe, limit=self.candle_limit)
            logger.debug("✅ Successfully fetched %d candles for %s", len(ohlcv) if ohlcv else 0, market_symbol)
        except ccxt.NetworkError as e:
            logger.error(f"🌐 Network error fetching OHLCV for {market_symbol}: {e}")
            return {}
        except ccxt.ExchangeError as e:
            logger.error(f"⚠️  Exchange error fetching OHLCV for {market_symbol}: {e}")
            return {}
        except Exception as e:
            logger.error(f"❌ Unexpected error fetching OHLCV for {market_symbol}: {e}", exc_info=True)
            return {}

        try:
            opens = [float(x[1]) for x in ohlcv]
            highs = [float(x[2]) for x in ohlcv]
            lows = [float(x[3]) for x in ohlcv]
            closes = [float(x[4]) for x in ohlcv]
            volumes = [float(x[5]) for x in ohlcv]
        except (IndexError, ValueError, TypeError) as e:
            logger.error(f"Error parsing OHLCV data for {market_symbol}: {e}")
            return {}

        if len(closes) < 20:
            logger.debug(f"Insufficient candle data for {market_symbol}: {len(closes)} candles")
            return {}

        current_price = closes[-1]
        ema20 = sum(closes[-20:]) / 20
        vp_result_raw = vp.calculate_volume_profile(highs[-100:], lows[-100:], closes[-100:], volumes[-100:])
        vp_result: Dict[str, Any] = vp_result_raw
        rsi = float(vp.calculate_rsi(np.array(closes)))
        
        # FIX: Issue #6 - Look-ahead bias: Use only CLOSED candles for pattern detection
        pattern_raw = vp.detect_candlestick_pattern(opens[:-1], highs[:-1], lows[:-1], closes[:-1])
        pattern = pattern_raw if isinstance(pattern_raw, str) else None

        # FIX: Issue #6 - Look-ahead bias: Use only CLOSED candles for volume analysis
        # Exclude current incomplete candle from calculations
        if len(volumes) >= 21:
            avg_volume = sum(volumes[-21:-1]) / 20  # Use last 20 CLOSED candles
            last_closed_volume = volumes[-2]  # Use previous CLOSED candle
            volume_spike_threshold = getattr(getattr(self.config, 'analysis', None), 'volume_spike_threshold', 1.5) if self.config else 1.5
            volume_spike = last_closed_volume > avg_volume * volume_spike_threshold
        else:
            avg_volume = sum(volumes[:-1]) / max(1, len(volumes) - 1)
            last_closed_volume = volumes[-2] if len(volumes) >= 2 else volumes[-1]
            volume_spike = False

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
            "current_volume": last_closed_volume,
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
    def _nearest_hvn(current_price: float, vp_result: Dict[str, Any], row_height: float) -> Tuple[Optional[float], bool]:
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
    ) -> Tuple[List[str], List[str]]:
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

        # Note: This needs to be called on instance if config is needed
        # For now, keep static behavior for backward compatibility
        if len(recent_candles) >= 6:
            greens = [v for o, c, v in reversed(recent_candles) if c > o][:3]
            reds = [v for o, c, v in reversed(recent_candles) if c < o][:3]
            if len(greens) >= 3 and len(reds) >= 3:
                if sum(greens) / 3 > (sum(reds) / 3) * 1.2:
                    long_factors.append("Buying Pressure Dominant")

        return long_factors, short_factors

    def _has_buying_pressure(
        self,
        candles: List[Tuple[float, float, float]],
        sample_size: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> bool:
        """Check if buying pressure is dominant (configurable thresholds)."""
        if sample_size is None:
            sample_size = getattr(getattr(self.config, 'analysis', None), 'buying_pressure_sample_size', 3) if self.config else 3
        if threshold is None:
            threshold = getattr(getattr(self.config, 'analysis', None), 'buying_pressure_threshold', 1.2) if self.config else 1.2
        
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
    ) -> Tuple[str, Optional[Dict[str, float]]]:
        if len(long_factors) >= 3 and len(long_factors) > len(short_factors):
            val = float(vp_result.get("val", current_price)) if isinstance(vp_result.get("val"), (int, float)) else current_price
            row_height = float(vp_result.get("row_height", 0.0)) if isinstance(vp_result.get("row_height"), (int, float)) else 0.0
            poc = float(vp_result.get("poc", current_price)) if isinstance(vp_result.get("poc"), (int, float)) else current_price
            raw_sl = val - row_height
            # Use configurable stop loss percentage (FIX: Externalized configuration)
            # Default is now 1.5% to avoid premature stops
            custom_sl = min(raw_sl, current_price * (1 - 0.015))  # 1.5% stop loss
            
            try:
                config_mgr = get_config_manager()
                risk_config = config_mgr.get_effective_risk("volume_bot", symbol)
                tp1_mult = risk_config.tp1_atr_multiplier
                tp2_mult = risk_config.tp2_atr_multiplier
                min_rr = risk_config.min_risk_reward
            except Exception:
                tp1_mult, tp2_mult, min_rr = 2.0, 3.0, 1.5

            # Increased minimum risk from 0.3% to 1.0% for more reasonable position sizing
            risk = max(current_price - custom_sl, current_price * 0.01)
            calculator = TPSLCalculator(min_risk_reward=0.8, min_risk_reward_tp2=1.5)
            levels = calculator.calculate(
                entry=current_price,
                direction="LONG",
                atr=risk,  # Use risk as ATR proxy
                tp1_multiplier=tp1_mult,
                tp2_multiplier=tp2_mult,
                custom_sl=custom_sl,
            )

            if levels.is_valid:
                # For LONG: TP1 should be lower than TP2 (TP1 is closer to entry)
                tp1 = float(levels.take_profit_1)
                tp2 = float(levels.take_profit_2)
                # Ensure TP2 > TP1 > Entry for longs
                if tp2 <= tp1:
                    tp2 = tp1 + risk  # Make TP2 further from entry
                return "LONG", {"entry": current_price, "sl": custom_sl, "tp1": tp1, "tp2": tp2}

            # If levels not valid, return None
            return "NEUTRAL", None

        if len(short_factors) >= 3 and len(short_factors) > len(long_factors):
            vah = float(vp_result.get("vah", current_price)) if isinstance(vp_result.get("vah"), (int, float)) else current_price
            row_height = float(vp_result.get("row_height", 0.0)) if isinstance(vp_result.get("row_height"), (int, float)) else 0.0
            poc = float(vp_result.get("poc", current_price)) if isinstance(vp_result.get("poc"), (int, float)) else current_price
            raw_sl = vah + row_height
            # Use configurable stop loss percentage (FIX: Externalized configuration)
            # Default is now 1.5% to avoid premature stops
            custom_sl = max(raw_sl, current_price * (1 + 0.015))  # 1.5% stop loss
            
            try:
                config_mgr = get_config_manager()
                risk_config = config_mgr.get_effective_risk("volume_bot", symbol)
                tp1_mult = risk_config.tp1_atr_multiplier
                tp2_mult = risk_config.tp2_atr_multiplier
                min_rr = risk_config.min_risk_reward
            except Exception:
                tp1_mult, tp2_mult, min_rr = 2.0, 3.0, 1.5

            # Increased minimum risk from 0.3% to 1.0% for more reasonable position sizing
            risk = max(custom_sl - current_price, current_price * 0.01)
            calculator = TPSLCalculator(min_risk_reward=0.8, min_risk_reward_tp2=1.5)
            levels = calculator.calculate(
                entry=current_price,
                direction="SHORT",
                atr=risk,
                tp1_multiplier=tp1_mult,
                tp2_multiplier=tp2_mult,
                custom_sl=custom_sl,
            )

            if levels.is_valid:
                # For SHORT: TP1 should be higher than TP2 (TP1 is closer to entry)
                tp1 = float(levels.take_profit_1)
                tp2 = float(levels.take_profit_2)
                # Ensure TP2 < TP1 < Entry for shorts
                if tp2 >= tp1:
                    tp2 = tp1 - risk  # Make TP2 further from entry
                return "SHORT", {"entry": current_price, "sl": custom_sl, "tp1": tp1, "tp2": tp2}

            # If levels not valid, return None
            return "NEUTRAL", None

        return "NEUTRAL", None


class SignalTracker:
    def __init__(self, analyzer: VolumeAnalyzer, stats: Optional[Any] = None, config: Optional[Any] = None) -> None:
        self.analyzer = analyzer
        self.stats: Optional[Any] = stats
        self.config: Optional[Any] = config
        self.state_lock = threading.Lock()  # FIX: Issue #5 - Add thread lock
        self.state = self._load_state()
        # Sync existing open signals to stats on startup
        self._sync_signals_to_stats()

    def _empty_state(self) -> Dict[str, Any]:
        return {"last_alerts": {}, "open_signals": {}, "last_result_notifications": {}}

    def _load_state(self) -> Dict[str, Any]:
        """Load state with proper error handling (FIX: Issue #3)."""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, 'r') as f:
                    data = json.load(f)
                
                # Validate structure
                if not isinstance(data, dict):
                    logger.warning(f"State file has invalid structure (expected dict, got {type(data)}), rebuilding")
                    return self._empty_state()
                
                if not isinstance(data.get("last_alerts"), dict):
                    data["last_alerts"] = {}
                if not isinstance(data.get("open_signals"), dict):
                    data["open_signals"] = {}
                if not isinstance(data.get("last_result_notifications"), dict):
                    data["last_result_notifications"] = {}
                return data
                
            except json.JSONDecodeError as e:
                logger.error(f"State file has invalid JSON: {e}, rebuilding from scratch")
                return self._empty_state()
            except IOError as e:
                logger.error(f"Failed to read state file: {e}")
                return self._empty_state()
        return self._empty_state()

    def _save_state(self) -> None:
        """Thread-safe state persistence with file locking (FIX: Issue #5)."""
        with self.state_lock:
            temp_file = STATE_FILE.with_suffix('.tmp')
            try:
                # Write to temp file first for atomic operation
                with open(temp_file, 'w') as f:
                    # Acquire exclusive lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    json.dump(self.state, f, indent=2)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
                # Atomic rename
                temp_file.replace(STATE_FILE)
            except Exception as e:
                logger.error(f"Failed to save state: {e}")
                if temp_file.exists():
                    temp_file.unlink()

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

    def has_open_signal(self, symbol: str, exchange: Optional[str] = None, timeframe: Optional[str] = None) -> bool:
        """Check if there's already an open signal (FIX: Issue #1 - Proper duplicate detection)."""
        signals = self.state.get("open_signals", {}) if isinstance(self.state.get("open_signals", {}), dict) else {}
        
        # Normalize symbol for comparison (remove :USDT suffix)
        normalized_symbol = symbol.split(":")[0]
        
        check_exchange = (self.config.signal.check_exchange_for_duplicates 
                         if self.config else True)
        check_timeframe = (self.config.signal.check_timeframe_for_duplicates 
                          if self.config else False)
        
        for signal_data in signals.values():
            if not isinstance(signal_data, dict):
                continue
            
            stored_symbol = signal_data.get("symbol", "").split(":")[0]
            
            # Basic symbol match
            if stored_symbol != normalized_symbol:
                continue
            
            # Check exchange if required and provided
            if check_exchange and exchange:
                if signal_data.get("exchange") != exchange:
                    continue
            
            # Check timeframe if required and provided
            if check_timeframe and timeframe:
                if signal_data.get("timeframe") != timeframe:
                    continue
            
            # Found a match
            return True
        
        return False

    def cleanup_stale_signals(self, max_age_hours: Optional[int] = None) -> int:
        """Remove stale signals and archive to stats (FIX: Issue #2 - Enable cleanup)."""
        if max_age_hours is None:
            max_age_hours = self.config.signal.max_signal_age_hours if self.config else 24
        
        signals = self.state.get("open_signals", {})
        if not isinstance(signals, dict) or not signals:
            return 0
        
        now = datetime.now(timezone.utc)
        removed = []
        
        for signal_id, payload in list(signals.items()):
            if not isinstance(payload, dict):
                signals.pop(signal_id, None)
                removed.append(signal_id)
                continue
            
            created_str = payload.get("created_at")
            if not isinstance(created_str, str):
                signals.pop(signal_id, None)
                removed.append(signal_id)
                continue
            
            try:
                created_dt = datetime.fromisoformat(created_str)
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                
                age_hours = (now - created_dt).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    # Archive to stats as expired before removing
                    if self.stats:
                        self.stats.record_close(
                            signal_id,
                            exit_price=payload.get("entry", 0),
                            result="EXPIRED"
                        )
                    
                    removed.append(signal_id)
                    signals.pop(signal_id)
                    logger.info(f"Removed stale signal {signal_id} (age: {age_hours:.1f}h)")
                    
            except (ValueError, AttributeError) as e:
                logger.warning(f"Invalid timestamp in signal {signal_id}: {e}")
                signals.pop(signal_id, None)
                removed.append(signal_id)
                continue
        
        if removed:
            self._save_state()
            logger.info(f"Cleaned up {len(removed)} stale signals")
        
        return len(removed)

    def add_signal(self, signal: VolumeSignal) -> bool:
        """Add a new signal. Returns False if duplicate exists for symbol (FIX: Issue #1)."""
        # Safety check for duplicate (should be caught earlier, but kept as failsafe)
        if self.has_open_signal(signal.symbol, exchange=signal.exchange, timeframe=signal.timeframe):
            logger.debug("Duplicate signal caught in add_signal() for %s (%s, %s) - failsafe triggered",
                       signal.symbol, signal.exchange, signal.timeframe)
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

    def _should_notify_result(self, symbol: str, signal_id: str, cooldown_minutes: int) -> bool:
        """Check if we should send result notification (prevent duplicates within cooldown)."""
        last_notifs = self.state.setdefault("last_result_notifications", {})
        if not isinstance(last_notifs, dict):
            last_notifs = {}
            self.state["last_result_notifications"] = last_notifs

        # Use symbol as key to prevent spam for same symbol
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

    def _mark_result_notified(self, symbol: str, signal_id: str) -> None:
        """Mark that we sent a result notification for this symbol."""
        last_notifs = self.state.setdefault("last_result_notifications", {})
        if not isinstance(last_notifs, dict):
            last_notifs = {}
            self.state["last_result_notifications"] = last_notifs
        last_notifs[symbol] = datetime.now(timezone.utc).isoformat()
        self._save_state()

    def _build_result_message(self, symbol: str, direction: str, result: str, entry: float,
                              exit_price: float, sl: float, tp1: float, tp2: float,
                              timeframe: str, exchange: str, signal_id: str) -> str:
        """Build enhanced result message with per-symbol performance history."""
        # Calculate PnL
        if direction == "LONG":
            pnl_pct = ((exit_price - entry) / entry) * 100
            emoji = "🟢" if result in ["TP1", "TP2"] else "🔴"
        else:
            pnl_pct = ((entry - exit_price) / entry) * 100
            emoji = "🟢" if result in ["TP1", "TP2"] else "🔴"

        # Get symbol performance history
        symbol_stats = self._get_symbol_performance(symbol)

        # Build message
        lines = [
            f"{emoji} <b>{direction} {symbol} - {result} HIT!</b>",
            "",
            f"💰 Entry: <code>{entry:.6f}</code>",
            f"🎯 Exit: <code>{exit_price:.6f}</code>",
            f"📊 PnL: <b>{pnl_pct:+.2f}%</b>",
            "",
            f"🛑 SL: <code>{sl:.6f}</code>",
            f"🎯 TP1: <code>{tp1:.6f}</code>",
            f"🎯 TP2: <code>{tp2:.6f}</code>",
            "",
            f"🕒 Timeframe: {timeframe} | 🏦 Exchange: {exchange.upper()}",
        ]

        # Add symbol performance section
        if symbol_stats["total"] > 0:
            win_rate = (symbol_stats["wins"] / symbol_stats["total"]) * 100
            lines.extend([
                "",
                f"📈 <b>{symbol} Performance History:</b>",
                f"   TP1: {symbol_stats['tp1']} | TP2: {symbol_stats['tp2']} | SL: {symbol_stats['sl']}",
                f"   Win Rate: <b>{win_rate:.1f}%</b> ({symbol_stats['wins']}/{symbol_stats['total']})",
                f"   Avg PnL: <b>{symbol_stats['avg_pnl']:+.2f}%</b>",
            ])

        lines.append(f"\n🆔 <code>{signal_id}</code>")
        return "\n".join(lines)

    def _get_symbol_performance(self, symbol: str) -> Dict[str, Any]:
        """Get performance statistics for a specific symbol."""
        if not self.stats or not isinstance(self.stats.data, dict):
            return {"total": 0, "wins": 0, "tp1": 0, "tp2": 0, "sl": 0, "avg_pnl": 0.0}

        history = self.stats.data.get("history", [])
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

    def check_open_signals(self, notifier: Optional[Any]) -> None:
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
            # FIX: Validate exchange is not empty
            if not isinstance(exchange_val, str) or not exchange_val or not exchange_val.strip():
                logger.warning(f"Removing signal {signal_id} with invalid exchange: {exchange_val}")
                signals.pop(signal_id, None)
                updated = True
                continue
            symbol_val = payload.get("symbol")
            # FIX: Validate symbol is not empty
            if not isinstance(symbol_val, str) or not symbol_val or not symbol_val.strip():
                logger.warning(f"Removing signal {signal_id} with invalid symbol: {symbol_val}")
                signals.pop(signal_id, None)
                updated = True
                continue
            market_symbol = resolve_symbol(symbol_val, market_type)
            # FIX: Issue #3 - Specific exception handling for ticker fetch
            try:
                client = self.analyzer.get_client(exchange_val, market_type)
                if self.analyzer.rate_limiter:
                    ticker = self.analyzer.rate_limiter.execute(client.fetch_ticker, market_symbol)
                else:
                    ticker = client.fetch_ticker(market_symbol)
                current_price = ticker.get("last") if isinstance(ticker, dict) else None
                if current_price is None and isinstance(ticker, dict):
                    current_price = ticker.get("close")
            except ccxt.NetworkError as exc:
                logger.warning(f"Network error fetching ticker for {market_symbol}: {exc}")
                continue
            except ccxt.ExchangeError as exc:
                logger.error(f"Exchange error for {market_symbol}: {exc}")
                continue
            except Exception as exc:
                logger.critical(f"Unexpected error fetching ticker for {market_symbol}: {exc}")
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
                # Check if we should notify (prevent duplicates within cooldown period)
                cooldown_minutes = self.config.signal.result_notification_cooldown_minutes if self.config and hasattr(self.config.signal, 'result_notification_cooldown_minutes') else 15
                should_notify = self._should_notify_result(symbol_val, signal_id, cooldown_minutes)

                logger.info("✅ Signal %s closed: %s hit! Entry: %.6f | Exit: %.6f | PnL: %.2f%%",
                           signal_id, result, entry, current_price,
                           ((current_price - entry) / entry * 100) if direction == "LONG" else ((entry - current_price) / entry * 100))

                # Update stats
                if self.stats:
                    stats_record = self.stats.record_close(
                        signal_id,
                        exit_price=current_price,
                        result=result,
                    )

                # Send notification if not duplicate
                if should_notify and notifier:
                    message = self._build_result_message(symbol_val, direction, result, entry, current_price,
                                                        sl, tp1, tp2, payload.get('timeframe', ''),
                                                        exchange_val, signal_id)
                    notifier.send_message(message)
                    self._mark_result_notified(symbol_val, signal_id)
                    logger.info("📤 Result notification sent for %s", signal_id)
                elif not should_notify:
                    logger.info("⏭️  Skipping duplicate result notification for %s (within %dm cooldown)", symbol_val, cooldown_minutes)

                signals.pop(signal_id, None)
                updated = True

        if updated:
            self._save_state()


class VolumeVNBOT:
    def __init__(self, cooldown_minutes: Optional[int] = None, config: Optional[Any] = None) -> None:
        # Load configuration
        self.config = config if config else (load_config() if load_config else None)
        
        self.watchlist = ensure_watchlist()
        self.analyzer = VolumeAnalyzer(config=self.config)
        self.cooldown = cooldown_minutes if cooldown_minutes is not None else (
            int(self.config.signal.cooldown_minutes) if self.config and hasattr(self.config, 'signal') and hasattr(self.config.signal, 'cooldown_minutes') and self.config.signal.cooldown_minutes is not None else 5
        )
        self.notifier = self._build_notifier()
        self.stats = SignalStats("Volume Bot", STATS_FILE)
        self.tracker = SignalTracker(self.analyzer, stats=self.stats, config=self.config)
        
        heartbeat_interval = int(self.config.execution.health_check_interval_seconds) if self.config and hasattr(self.config, 'execution') and hasattr(self.config.execution, 'health_check_interval_seconds') and self.config.execution.health_check_interval_seconds is not None else 3600
        self.health_monitor = HealthMonitor("Volume Bot", self.notifier, heartbeat_interval=heartbeat_interval) if HealthMonitor and self.notifier else None
        
        calls_per_min = int(self.config.rate_limit.calls_per_minute) if self.config and hasattr(self.config, 'rate_limit') and hasattr(self.config.rate_limit, 'calls_per_minute') and self.config.rate_limit.calls_per_minute is not None else 60
        self.rate_limiter = RateLimiter(calls_per_minute=calls_per_min, backoff_file=LOG_DIR / "rate_limiter.json") if RateLimiter else None
        
        self.exchange_backoff: Dict[str, float] = {}
        self.exchange_delay: Dict[str, float] = {}
        
        # Validate exchange credentials on startup (FIX: Security - API Key Validation)
        self._validate_exchanges()

    def _validate_exchanges(self) -> None:
        """Validate exchange credentials for all exchanges in watchlist (FIX: Security)."""
        if not self.config:
            logger.warning("No config available, skipping exchange validation")
            return
        
        exchanges = set(item.get("exchange", "binanceusdm") for item in self.watchlist)
        for exchange in exchanges:
            creds = self.config.get_exchange_credentials(exchange)
            if creds.is_configured:
                self.analyzer.validate_exchange_credentials(exchange, "swap")
            else:
                logger.warning(f"No credentials configured for {exchange}")

    def _build_notifier(self) -> Optional[Any]:
        """Build Telegram notifier with validation (FIX: Security - Telegram validation)."""
        if TelegramNotifier is None:
            logger.warning("Telegram notifier unavailable; running in console-only mode")
            return None

        if self.config:
            telegram = self.config.get_telegram_config()
            is_valid, error = telegram.validate()
            if not is_valid:
                logger.warning(f"Telegram configuration invalid: {error}")
                return None
            bot_token = telegram.bot_token
            chat_id = telegram.chat_id
        else:
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

    def _register_backoff(self, exchange: str, base_delay: Optional[float] = None, max_delay: Optional[float] = None) -> None:
        """Register rate limit backoff with configurable delays."""
        if base_delay is None:
            base_delay = self.config.rate_limit.rate_limit_backoff_base if self.config else 60
        if max_delay is None:
            max_delay = self.config.rate_limit.rate_limit_backoff_max if self.config else 300
        
        prev = self.exchange_delay.get(exchange, base_delay)
        multiplier = self.config.rate_limit.backoff_multiplier if self.config and self.config.rate_limit and hasattr(self.config.rate_limit, 'backoff_multiplier') else 2.0
        # Ensure prev, multiplier, max_delay are not None
        if prev is None:
            prev = 60.0
        if multiplier is None:
            multiplier = 2.0
        if max_delay is None:
            max_delay = 300.0
        try:
            prev_f = float(prev) if prev is not None else 60.0
            multiplier_f = float(multiplier) if multiplier is not None else 2.0
            max_delay_f = float(max_delay) if max_delay is not None else 300.0
            base_delay_f = float(base_delay) if base_delay is not None else 60.0
            new_delay = min(prev_f * multiplier_f, max_delay_f) if exchange in self.exchange_backoff else base_delay_f
        except Exception:
            new_delay = 60.0
        until = time.time() + new_delay
        self.exchange_backoff[exchange] = until
        self.exchange_delay[exchange] = new_delay
        logger.warning("%s rate limit; backing off for %.0fs", exchange, new_delay)

    def run_cycle(self, run_once: bool = True, delay_seconds: Optional[int] = None) -> None:
        """Main bot cycle with improved error handling and configuration."""
        if delay_seconds is None:
            delay_seconds = self.config.execution.symbol_delay_seconds if self.config else 5
        
        cycle_interval = self.config.execution.cycle_interval_seconds if self.config else 60
        max_open_signals = self.config.risk.max_open_signals if self.config else 50
        
        logger.info("Starting Volume VN cycle for %d pairs", len(self.watchlist))
        logger.info(f"Config: Max open signals={max_open_signals}, Cooldown={self.cooldown}min, Cycle interval={cycle_interval}s")
        
        # Send startup notification
        if self.health_monitor and self.config and self.config.execution.enable_startup_notification:
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

                        # MAX OPEN SIGNALS LIMIT: Prevent overexposure (now configurable)
                        open_signals = self.tracker.state.get("open_signals", {}) if isinstance(self.tracker.state.get("open_signals", {}), dict) else {}
                        current_open = len(open_signals)
                        if current_open >= max_open_signals:
                            logger.info(
                                "Max open signals limit reached (%d/%d). Skipping new signal for %s",
                                current_open, max_open_signals, symbol
                            )
                            continue
                        
                        try:
                            signal_payload = self._build_signal(snapshot)

                            # DUPLICATE CHECK: Check if we already have this signal BEFORE sending
                            if self.tracker.has_open_signal(signal_payload.symbol,
                                                           exchange=signal_payload.exchange,
                                                           timeframe=signal_payload.timeframe):
                                logger.info("Skipping duplicate signal for %s (%s, %s) - already have open position",
                                          signal_payload.symbol, signal_payload.exchange, signal_payload.timeframe)
                                continue

                            # SIGNAL REVERSAL DETECTION: Warn if opposite direction signal detected
                            self._check_signal_reversal(symbol, signal_payload.direction)

                            self._dispatch_signal(signal_payload, snapshot)
                            self.tracker.mark_alert(symbol, timeframe, exchange)
                            self.tracker.add_signal(signal_payload)
                        except ValueError as e:
                            logger.error(f"Failed to build signal for {symbol}: {e}")
                            continue

                        time.sleep(float(delay_seconds) if delay_seconds is not None else 1.0)

                    self.tracker.check_open_signals(self.notifier)
                    
                    # Record successful cycle
                    if self.health_monitor:
                        self.health_monitor.record_cycle()

                    if run_once:
                        break
                    logger.info(f"Cycle complete; sleeping {cycle_interval} seconds")
                    # Sleep in 1-second chunks to respond quickly to shutdown signals
                    for _ in range(int(cycle_interval)):
                        time.sleep(1.0)
                except Exception as exc:
                    logger.error(f"Error in cycle: {exc}")
                    if self.health_monitor:
                        self.health_monitor.record_error(str(exc))
                    if run_once:
                        raise
                    time.sleep(10)  # Brief pause before retry
        finally:
            # Send shutdown notification
            if self.health_monitor and self.config and self.config.execution.enable_shutdown_notification:
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
        display_symbol = watch_symbol if isinstance(watch_symbol, str) and watch_symbol else base_symbol
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

        # FIX: Validate critical fields before creating signal
        if not display_symbol or not str(display_symbol).strip():
            raise ValueError(f"Cannot create signal with empty symbol. Snapshot: {snapshot}")
        if not exchange or not str(exchange).strip():
            raise ValueError(f"Cannot create signal with empty exchange for {display_symbol}")
        if not timeframe or not str(timeframe).strip():
            raise ValueError(f"Cannot create signal with empty timeframe for {display_symbol}")
        if not direction or not str(direction).strip():
            raise ValueError(f"Cannot create signal with empty direction for {display_symbol}")

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
        """Format volume signal using centralized template."""
        # Convert LONG/SHORT to BULLISH/BEARISH
        direction = "BULLISH" if signal.direction == "LONG" else "BEARISH"

        # Get volume profile metrics for extra info
        metrics = snapshot.get("volume_profile", {})
        poc = metrics.get("poc", 0.0) if isinstance(metrics, dict) else 0.0
        vah = metrics.get("vah", 0.0) if isinstance(metrics, dict) else 0.0
        val = metrics.get("val", 0.0) if isinstance(metrics, dict) else 0.0
        extra_info = f"POC: {float(poc):.4f} | VAH: {float(vah):.4f} | VAL: {float(val):.4f}"

        # Get performance stats
        perf_stats = None
        symbol_key = signal.symbol if "/" in signal.symbol else f"{signal.symbol}/USDT"
        if self.stats and isinstance(self.stats.data, dict):
            history = self.stats.data.get("history", [])
            if isinstance(history, list):
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

                total = tp1 + tp2 + sl
                if total > 0:
                    perf_stats = {
                        "tp1": tp1,
                        "tp2": tp2,
                        "sl": sl,
                        "wins": tp1 + tp2,
                        "total": total,
                    }

        # Format reasons
        reasons = [html.escape(str(r)) for r in signal.rationale] if signal.rationale else None

        return format_signal_message(
            bot_name="VOLUME",
            symbol=symbol_key,
            direction=direction,
            entry=signal.entry,
            stop_loss=signal.stop_loss,
            tp1=signal.take_profit_primary,
            tp2=signal.take_profit_secondary,
            reasons=reasons,
            exchange=signal.exchange.upper(),
            timeframe=signal.timeframe,
            current_price=signal.entry,
            performance_stats=perf_stats,
            extra_info=extra_info,
        )

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
    
    @staticmethod
    def _normalize_symbol_for_comparison(symbol: str) -> str:
        """Normalize symbol to base form for comparison (FIX: Issue #4)."""
        # Remove /USDT and :USDT suffixes, keep only base currency
        return symbol.upper().split("/")[0].replace("USDT", "").replace(":USDT", "")
    
    def _check_signal_reversal(self, symbol: str, new_direction: str) -> None:
        """Check if new signal is opposite direction to open position (FIX: Issue #4)."""
        open_signals = self.tracker.state.get("open_signals", {})
        if not isinstance(open_signals, dict):
            return
        
        normalized_new = self._normalize_symbol_for_comparison(symbol)
        
        for signal_id, signal_data in open_signals.items():
            if not isinstance(signal_data, dict):
                continue
            
            signal_symbol = signal_data.get("symbol", "")
            normalized_open = self._normalize_symbol_for_comparison(signal_symbol)
            
            # Check if same base symbol (e.g., both BTC)
            if normalized_new != normalized_open:
                continue
            
            open_direction = signal_data.get("direction", "")
            opposite = (
                (new_direction == "LONG" and open_direction == "SHORT") or
                (new_direction == "SHORT" and open_direction == "LONG")
            )
            
            if opposite:
                # Escape HTML special characters to prevent Telegram API errors
                safe_symbol = html.escape(symbol)
                safe_signal_id = html.escape(signal_id)
                safe_open_dir = html.escape(open_direction)
                safe_new_dir = html.escape(new_direction)

                warning_msg = (
                    f"⚠️ <b>SIGNAL REVERSAL DETECTED</b> ⚠️\n\n"
                    f"<b>Symbol:</b> {safe_symbol}\n"
                    f"<b>Open Position:</b> {safe_open_dir}\n"
                    f"<b>New Signal:</b> {safe_new_dir}\n\n"
                    f"💡 <b>Action:</b> Consider exiting your {safe_open_dir} position!\n"
                    f"🔄 <b>Market may be reversing</b>\n\n"
                    f"🆔 Open Signal: {safe_signal_id}\n"
                    f"⏰ {datetime.now(timezone.utc).isoformat()}"
                )
                if self.notifier:
                    self.notifier.send_message(warning_msg)
                logger.info("Signal reversal detected for %s: %s -> %s", symbol, open_direction, new_direction)
                break


def validate_environment() -> bool:
    """Validate all required environment variables are set (FIX: Environment validation)."""
    required = {
        "TELEGRAM_BOT_TOKEN": "Telegram bot token (or TELEGRAM_BOT_TOKEN_VOLUME)",
        "TELEGRAM_CHAT_ID": "Telegram chat ID",
    }
    
    optional_warning = {
        "BINANCEUSDM_API_KEY": "Binance USDⓈ-M API key",
        "BINANCEUSDM_SECRET": "Binance USDⓈ-M secret",
    }
    
    missing = []
    
    # Check Telegram (at least one token must exist)
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN_VOLUME") or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token:
        missing.append("  - TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN_VOLUME: Telegram bot token")
    if not chat_id:
        missing.append("  - TELEGRAM_CHAT_ID: Telegram chat ID")
    
    if missing:
        logger.critical("❌ Missing required environment variables:")
        for item in missing:
            logger.critical(item)
        logger.critical("\nPlease create a .env file with these variables.")
        logger.critical("See .env.example for a template.")
        return False
    
    logger.info("✅ All required environment variables present")
    
    # Check optional (at least one exchange should be configured)
    has_exchange = False
    for exchange in ['binanceusdm', 'mexc', 'bybit']:
        api_key = os.getenv(f'{exchange.upper()}_API_KEY')
        secret = os.getenv(f'{exchange.upper()}_SECRET')
        if api_key and secret:
            has_exchange = True
            logger.info(f"✅ {exchange.upper()} credentials configured")
    
    if not has_exchange:
        logger.warning("⚠️ No exchange credentials configured")
        logger.warning("   The bot will run but cannot execute trades")
        logger.warning("   Configure at least one: BINANCEUSDM, MEXC, or BYBIT")
    
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Volume VN Bot - Refactored version with fixes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python volume_vn_bot.py --once              # Run one cycle
  python volume_vn_bot.py --track             # Check open signals only
  python volume_vn_bot.py --cooldown 10       # 10 minute cooldown
  python volume_vn_bot.py --config config.json  # Use custom config file
        """
    )
    parser.add_argument("--once", action="store_true", help="Run only one cycle")
    parser.add_argument(
        "--cooldown",
        type=int,
        default=None,
        help="Minutes before repeating alert per pair (overrides config)",
    )
    parser.add_argument("--track", action="store_true", help="Only run tracker checks")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration JSON file",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip environment validation (not recommended)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level (default: INFO)",
    )
    parser.add_argument(
        "--detailed-logging",
        action="store_true",
        help="Enable detailed logging with function names and line numbers",
    )
    args = parser.parse_args()

    # Load configuration first to get log settings
    config_file = Path(args.config) if args.config else (BASE_DIR / "config.json")
    config = load_config(config_file) if load_config and config_file.exists() else None

    # Setup enhanced logging
    log_level = args.log_level
    enable_detailed = args.detailed_logging
    if config and hasattr(config, 'execution'):
        log_level = getattr(config.execution, 'log_level', log_level)
        enable_detailed = getattr(config.execution, 'enable_detailed_logging', enable_detailed)

    global logger
    logger = setup_logging(log_level=log_level, enable_detailed=enable_detailed)
    logger.info("=" * 60)
    logger.info("🤖 Volume VN Bot Starting...")
    logger.info("=" * 60)
    logger.info(f"📝 Log level: {log_level} | Detailed: {enable_detailed}")
    logger.info(f"📁 Working directory: {BASE_DIR}")
    logger.info(f"⚙️  Config file: {config_file if config else 'Using defaults'}")

    # Validate environment unless skipped
    if not args.skip_validation:
        if not validate_environment():
            logger.critical("❌ Environment validation failed - exiting")
            logger.critical("Use --skip-validation to bypass (not recommended)")
            sys.exit(1)

    # Reload configuration (config may have been loaded earlier for log settings)
    config_path: Optional[Path] = Path(args.config) if args.config else None
    config = load_config(config_path) if load_config else None

    if config:
        logger.info("Configuration loaded successfully")
        # Additional validation from config
        is_valid, errors = config.validate_environment()
        if not is_valid and not args.skip_validation:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            sys.exit(1)
    
    # Initialize bot
    bot = VolumeVNBOT(cooldown_minutes=args.cooldown, config=config)

    if args.track:
        bot.tracker.check_open_signals(bot.notifier)
        return

    bot.run_cycle(run_once=args.once)


if __name__ == "__main__":
    main()
