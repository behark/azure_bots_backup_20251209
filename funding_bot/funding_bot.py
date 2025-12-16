#!/usr/bin/env python3
"""
Automated funding-rate/open-interest monitor built from funding_oi.py.

PHASE 3 ENHANCEMENTS:
- Structured JSON logging with correlation IDs
- Retry logic with exponential backoff for API calls
- Circuit breaker pattern to prevent cascading failures
- Enhanced type safety with common.types
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, cast

import ccxt
import requests

# Phase 3: Import structured logging, resilience, and types
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from common.logging_config import get_logger, generate_correlation_id, ContextLogger
    from common.resilience import retry_with_backoff, CircuitBreaker, RetryError
    from common.types import SignalDirection
    PHASE3_AVAILABLE = True
except ImportError:
    # Fallback if common module not available
    PHASE3_AVAILABLE = False
    import logging

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
STATE_FILE = BASE_DIR / "funding_state.json"
WATCHLIST_FILE = BASE_DIR / "funding_watchlist.json"
STATS_FILE = LOG_DIR / "funding_stats.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# Phase 3: Use structured logging with JSON format for production
if PHASE3_AVAILABLE:
    logger = get_logger(
        "funding_bot",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_dir=LOG_DIR,
        json_logs=os.getenv("JSON_LOGS", "true").lower() == "true",
        console_output=True,
        max_bytes=10 * 1024 * 1024,  # 10MB
        backup_count=5
    )
    logger.info("Phase 3 structured logging initialized", extra={
        "json_logs": os.getenv("JSON_LOGS", "true").lower() == "true",
        "log_dir": str(LOG_DIR)
    })
else:
    # Fallback to basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "funding_bot.log"),
        ],
    )
    logger = logging.getLogger("funding_bot")
    logger.warning("Phase 3 modules not available, using basic logging")

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(BASE_DIR / ".env")
    load_dotenv(BASE_DIR.parent / ".env")
except ImportError:
    pass

sys.path.append(str(BASE_DIR.parent))
try:
    from notifier import TelegramNotifier  # type: ignore
    from signal_stats import SignalStats  # type: ignore
    from health_monitor import HealthMonitor, RateLimiter  # type: ignore
    from tp_sl_calculator import TPSLCalculator, TradeLevels  # type: ignore
    from trade_config import get_config_manager  # type: ignore
    from rate_limit_handler import RateLimitHandler  # type: ignore
except ImportError:
    TelegramNotifier = None  # type: ignore
    SignalStats = None  # type: ignore
    HealthMonitor = None  # type: ignore
    RateLimiter = None  # type: ignore
    TPSLCalculator = None  # type: ignore
    TradeLevels = None  # type: ignore
    get_config_manager = None  # type: ignore
    RateLimitHandler = None  # type: ignore


# Graceful shutdown handling
shutdown_requested = False


def signal_handler(signum, frame) -> None:  # pragma: no cover - signal path
    """Handle shutdown signals (SIGINT, SIGTERM) gracefully."""
    global shutdown_requested
    shutdown_requested = True
    logger.info("Received %s, shutting down gracefully...", signal.Signals(signum).name)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class WatchItem(TypedDict, total=False):
    symbol: str
    period: str
    cooldown_minutes: int


def load_watchlist() -> List[WatchItem]:
    if not WATCHLIST_FILE.exists():
        logger.error("Watchlist file missing: %s", WATCHLIST_FILE)
        return []

    try:
        data = json.loads(WATCHLIST_FILE.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover
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
        normalized.append(
            {
                "symbol": symbol_val.upper(),
                "period": period,
                "cooldown_minutes": cooldown,
            }
        )
    return normalized


def human_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class MexcFundingClient:
    REST_BASE = "https://contract.mexc.com/api/v1/contract"

    def __init__(self):
        self.session = requests.Session()
        self.exchange = ccxt.mexc({"enableRateLimit": True, "options": {"defaultType": "swap"}})  # type: ignore[arg-type]
        self.exchange.load_markets()
        # Initialize rate limit handler with conservative settings
        self.rate_limiter = RateLimitHandler(base_delay=0.5, max_retries=5) if RateLimitHandler else None

        # Phase 3: Circuit breakers for MEXC API endpoints
        if PHASE3_AVAILABLE:
            self.api_breaker = CircuitBreaker(
                failure_threshold=5,
                timeout=60,
                name="mexc_api"
            )
            self.exchange_breaker = CircuitBreaker(
                failure_threshold=3,
                timeout=30,
                name="mexc_exchange"
            )
            logger.info("Circuit breakers initialized for MEXC API", extra={
                "api_threshold": 5,
                "exchange_threshold": 3
            })
        else:
            self.api_breaker = None
            self.exchange_breaker = None

    @staticmethod
    def _swap_symbol(symbol: str) -> str:
        return f"{symbol.upper()}/USDT:USDT"

    @staticmethod
    def _contract_symbol(symbol: str) -> str:
        return f"{symbol.upper()}_USDT"

    def funding_history(self, symbol: str, limit: int = 10) -> list:
        params = {"symbol": self._contract_symbol(symbol), "page_num": 1, "page_size": limit}
        resp = self.session.get(f"{self.REST_BASE}/funding_rate/history", params=params, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("data", {}).get("resultList", [])

    def funding_rate(self, symbol: str) -> dict:
        rest_symbol = self._contract_symbol(symbol).lower()
        resp = self.session.get(f"{self.REST_BASE}/funding_rate/{rest_symbol}", timeout=10)
        resp.raise_for_status()
        return resp.json().get("data", {})

    def ticker(self, symbol: str) -> dict:
        """
        Fetch ticker data with retry logic and circuit breaker protection.

        Phase 3: Demonstrates resilience patterns - retries transient errors
        and fails fast when circuit breaker is open.
        """
        def _fetch():
            if self.rate_limiter:
                return self.rate_limiter.execute(self.exchange.fetch_ticker, self._swap_symbol(symbol))
            return self.exchange.fetch_ticker(self._swap_symbol(symbol))

        # Phase 3: Use circuit breaker if available
        if PHASE3_AVAILABLE and self.exchange_breaker:
            result = self.exchange_breaker.call(_fetch)
            if result is None:
                raise Exception("Circuit breaker open for exchange API")
            return result

        # Fallback to direct call with retry
        if PHASE3_AVAILABLE:
            @retry_with_backoff(max_attempts=3, base_delay=1.0)
            def _with_retry():
                return _fetch()
            return _with_retry()

        return _fetch()

    def trades(self, symbol: str, limit: int = 500) -> list:
        """
        Fetch trades data with retry logic and circuit breaker protection.

        Phase 3: Demonstrates resilience patterns for API calls.
        """
        def _fetch():
            if self.rate_limiter:
                return self.rate_limiter.execute(self.exchange.fetch_trades, self._swap_symbol(symbol), limit=limit)
            return self.exchange.fetch_trades(self._swap_symbol(symbol), limit=limit)

        # Phase 3: Use circuit breaker if available
        if PHASE3_AVAILABLE and self.exchange_breaker:
            result = self.exchange_breaker.call(_fetch)
            if result is None:
                raise Exception("Circuit breaker open for exchange API")
            return result

        # Fallback to direct call with retry
        if PHASE3_AVAILABLE:
            @retry_with_backoff(max_attempts=3, base_delay=1.0)
            def _with_retry():
                return _fetch()
            return _with_retry()

        return _fetch()

    def ohlcv(self, symbol: str, timeframe: str = "5m", limit: int = 100) -> list:
        """
        Fetch OHLCV data with retry logic and circuit breaker protection.

        Phase 3: Demonstrates resilience patterns - retries up to 3 times
        with exponential backoff on transient failures.
        """
        def _fetch():
            if self.rate_limiter:
                return self.rate_limiter.execute(self.exchange.fetch_ohlcv, self._swap_symbol(symbol), timeframe=timeframe, limit=limit)
            return self.exchange.fetch_ohlcv(self._swap_symbol(symbol), timeframe=timeframe, limit=limit)

        # Phase 3: Use circuit breaker if available
        if PHASE3_AVAILABLE and self.exchange_breaker:
            result = self.exchange_breaker.call(_fetch)
            if result is None:
                raise Exception("Circuit breaker open for exchange API")
            return result

        # Fallback to direct call with retry
        if PHASE3_AVAILABLE:
            @retry_with_backoff(max_attempts=3, base_delay=1.0)
            def _with_retry():
                return _fetch()
            return _with_retry()

        return _fetch()


@dataclass
class FundingSnapshot:
    symbol: str
    timestamp: str
    price: Optional[float] = None
    ema_20: Optional[float] = None
    funding_rate: Optional[float] = None  # percent
    avg_funding: Optional[float] = None  # percent
    funding_samples: List[Dict[str, object]] = field(default_factory=list)
    open_interest: Optional[float] = None
    oi_change_pct: Optional[float] = None
    oi_trend: str = "NEUTRAL"
    ls_ratio: Optional[float] = None
    long_pct: Optional[float] = None
    short_pct: Optional[float] = None
    taker_ratio: Optional[float] = None
    taker_buy_vol: Optional[float] = None
    taker_sell_vol: Optional[float] = None
    volume_24h: Optional[float] = None  # Added missing attribute


class FundingSignalEvaluator:
    FUNDING_EXTREME = 0.05
    FUNDING_HIGH = 0.02
    FUNDING_LOW = -0.02
    FUNDING_EXTREME_LOW = -0.05
    OI_STRONG = 2.0
    OI_WEAK = -2.0
    LS_HIGH = 1.8
    LS_LOW = 0.6
    TAKER_HIGH = 1.2
    TAKER_LOW = 0.8

    def evaluate(
        self,
        snapshot: FundingSnapshot,
        current_price: Optional[float] = None,
        ema_20: Optional[float] = None,
    ) -> Dict[str, object]:
        bias_score = 0
        reasons: List[str] = []
        price = current_price if isinstance(current_price, (int, float)) else snapshot.price
        ema = ema_20 if isinstance(ema_20, (int, float)) else snapshot.ema_20

        def _trend_supports(direction: str) -> bool:
            if price is None or ema is None:
                return False
            return price < ema if direction == "BEARISH" else price > ema

        if snapshot.funding_rate is not None:
            fr = snapshot.funding_rate

            if fr >= self.FUNDING_EXTREME:
                if _trend_supports("BEARISH"):
                    bias_score -= 3
                    reasons.append("Extremely high funding → long squeeze risk")
                else:
                    logger.debug(
                        "Ignoring extreme positive funding for %s: price/EMA misaligned",
                        snapshot.symbol,
                    )
            elif fr >= self.FUNDING_HIGH:
                if _trend_supports("BEARISH"):
                    bias_score -= 2
                    reasons.append("High positive funding → longs paying heavily")
                else:
                    logger.debug(
                        "Ignoring positive funding for %s: price/EMA misaligned",
                        snapshot.symbol,
                    )
            elif fr <= self.FUNDING_EXTREME_LOW:
                if _trend_supports("BULLISH"):
                    bias_score += 3
                    reasons.append("Extremely negative funding → short squeeze risk")
                else:
                    logger.debug(
                        "Ignoring extreme negative funding for %s: price/EMA misaligned",
                        snapshot.symbol,
                    )
            elif fr <= self.FUNDING_LOW:
                if _trend_supports("BULLISH"):
                    bias_score += 2
                    reasons.append("Negative funding → shorts paying")
                else:
                    logger.debug(
                        "Ignoring negative funding for %s: price/EMA misaligned",
                        snapshot.symbol,
                    )

        if snapshot.oi_change_pct is not None:
            if snapshot.oi_change_pct >= self.OI_STRONG:
                bias_score += 1 if bias_score > 0 else -1
                reasons.append(f"OI rising {snapshot.oi_change_pct:+.1f}% ({snapshot.oi_trend})")
            elif snapshot.oi_change_pct <= self.OI_WEAK:
                bias_score -= 1 if bias_score < 0 else 1
                reasons.append(f"OI falling {snapshot.oi_change_pct:+.1f}% ({snapshot.oi_trend})")

        if snapshot.ls_ratio is not None:
            if snapshot.ls_ratio >= self.LS_HIGH:
                bias_score -= 2
                reasons.append(f"Long/Short ratio {snapshot.ls_ratio:.2f} → crowded longs")
            elif snapshot.ls_ratio <= self.LS_LOW:
                bias_score += 2
                reasons.append(f"Long/Short ratio {snapshot.ls_ratio:.2f} → crowded shorts")

        if snapshot.taker_ratio is not None:
            if snapshot.taker_ratio >= self.TAKER_HIGH:
                bias_score += 1
                reasons.append("Taker flow net buying")
            elif snapshot.taker_ratio <= self.TAKER_LOW:
                bias_score -= 1
                reasons.append("Taker flow net selling")

        if bias_score > 1:
            direction = "BULLISH"
        elif bias_score < -1:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        return {
            "direction": direction,
            "score": bias_score,
            "reasons": reasons,
            "should_alert": direction != "NEUTRAL" and reasons,
        }


class FundingState:
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
        except json.JSONDecodeError:  # pragma: no cover
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

    def iter_signals(self) -> Dict[str, Dict[str, Any]]:
        signals = self.data.setdefault("open_signals", {})
        if not isinstance(signals, dict):
            signals = {}
            self.data["open_signals"] = signals
        return cast(Dict[str, Dict[str, Any]], signals)


class FundingBot:
    def __init__(self, interval: int = 300, default_cooldown: int = 45):
        self.interval = interval
        self.default_cooldown = default_cooldown
        self.watchlist: List[WatchItem] = load_watchlist()
        self.client = MexcFundingClient()
        self.evaluator = FundingSignalEvaluator()
        self.state = FundingState(STATE_FILE)
        self.notifier = self._init_notifier()
        self.stats = SignalStats("Funding Bot", STATS_FILE) if SignalStats else None
        self.atr_cache: Dict[str, tuple[float, float]] = {}
        self.health_monitor = HealthMonitor("Funding Bot", self.notifier, heartbeat_interval=3600) if HealthMonitor else None
        self.rate_limiter = RateLimiter(calls_per_minute=60, backoff_file=LOG_DIR / "rate_limiter.json") if RateLimiter else None

    def _init_notifier(self):
        if TelegramNotifier is None:
            logger.warning("Telegram notifier unavailable; alerts will log locally only")
            return None
        token = os.getenv("TELEGRAM_BOT_TOKEN_FUNDING") or os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            logger.warning("Telegram credentials missing; alerts will not be sent")
            return None
        return TelegramNotifier(bot_token=token, chat_id=chat_id, signals_log_file=str(LOG_DIR / "funding_signals.json"))

    def run(self, loop: bool = False) -> None:
        if not self.watchlist:
            logger.error("Empty watchlist; exiting")
            return

        logger.info("Starting funding bot for %d symbols", len(self.watchlist))
        
        # Send startup notification
        if self.health_monitor:
            self.health_monitor.send_startup_message()
        
        try:
            while not shutdown_requested:
                try:
                    self._run_cycle()
                    self._monitor_open_signals()

                    # Record successful cycle
                    if self.health_monitor:
                        self.health_monitor.record_cycle()

                    if not loop:
                        break

                    if shutdown_requested:
                        break

                    logger.info("Cycle complete; sleeping %ds", self.interval)
                    time.sleep(self.interval)
                except Exception as exc:
                    logger.error(f"Error in cycle: {exc}")
                    if self.health_monitor:
                        self.health_monitor.record_error(str(exc))
                    if not loop:
                        raise
                    if shutdown_requested:
                        break
                    time.sleep(10)  # Brief pause before retry
        finally:
            # Send shutdown notification
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
            
            # Apply rate limiting
            if self.rate_limiter:
                self.rate_limiter.wait_if_needed()
            
            try:
                snapshot = self._collect(symbol, period)
                if self.rate_limiter:
                    self.rate_limiter.record_success(f"mexc_{symbol}")
            except (ccxt.NetworkError, ccxt.ExchangeError) as exc:
                logger.error("Exchange API error for %s: %s", symbol, exc)
                if self.rate_limiter:
                    self.rate_limiter.record_error(f"mexc_{symbol}")
                if self.health_monitor:
                    self.health_monitor.record_error(f"API error for {symbol}: {exc}")
                continue
            except ValueError as exc:
                logger.error("Invalid data for %s: %s", symbol, exc)
                if self.health_monitor:
                    self.health_monitor.record_error(f"Data validation error for {symbol}: {exc}")
                continue
            except Exception as exc:
                # Catch-all for unexpected errors - log and continue
                logger.exception("Unexpected error collecting data for %s: %s", symbol, exc)
                if self.health_monitor:
                    self.health_monitor.record_error(f"Unexpected error for {symbol}: {exc}")
                continue

            evaluation = self.evaluator.evaluate(snapshot, snapshot.price, snapshot.ema_20)
            if not evaluation["should_alert"]:
                logger.debug("%s no alert", symbol)
                continue

            if not self.state.can_alert(symbol, cooldown):
                logger.debug("Cooldown active for %s", symbol)
                continue

            trade_levels = self._build_trade_levels(snapshot, evaluation, period)

            direction_str = evaluation.get("direction")
            if isinstance(direction_str, str) and direction_str in {"BULLISH", "BEARISH"}:
                self._check_signal_reversal(symbol, direction_str)
            
            message = self._format_message(snapshot, evaluation, trade_levels)
            self._dispatch(message)
            self.state.mark_alert(symbol)
            if trade_levels:
                signal_id_val = trade_levels.get("id") if isinstance(trade_levels, dict) else None
                direction_val = trade_levels.get("direction") if isinstance(trade_levels, dict) else None
                entry_val = trade_levels.get("entry") if isinstance(trade_levels, dict) else None
                created_at_val = trade_levels.get("created_at") if isinstance(trade_levels, dict) else None
                if not (
                    isinstance(signal_id_val, str)
                    and isinstance(direction_val, str)
                    and isinstance(entry_val, (int, float))
                    and isinstance(created_at_val, str)
                ):
                    continue
                # MAX OPEN SIGNALS LIMIT: Prevent overexposure
                MAX_OPEN_SIGNALS = 50
                current_open = len(self.state.iter_signals())
                if current_open >= MAX_OPEN_SIGNALS:
                    logger.info(
                        "Max open signals limit reached (%d/%d). Skipping new signal for %s",
                        current_open, MAX_OPEN_SIGNALS, symbol
                    )
                    continue
                
                trade_levels["timeframe"] = period
                trade_levels["exchange"] = "MEXC"
                self.state.add_signal(signal_id_val, trade_levels)
                if self.stats:
                    self.stats.record_open(
                        signal_id=signal_id_val,
                        symbol=f"{symbol}/USDT",
                        direction=direction_val,
                        entry=float(entry_val),
                        created_at=created_at_val,
                        extra={
                            "timeframe": period,
                            "exchange": "MEXC",
                            "display_symbol": f"{symbol}/USDT",
                        },
                    )
            time.sleep(0.2)  # light throttle

    def _collect(self, symbol: str, period: str) -> FundingSnapshot:
        snapshot = FundingSnapshot(symbol=symbol, timestamp=human_ts())

        try:
            funding_history = self.client.funding_history(symbol)
            snapshot.funding_samples = funding_history
            if funding_history:
                rates = [float(entry.get("fundingRate", entry.get("rate", 0))) for entry in funding_history]
                snapshot.funding_rate = rates[-1] * 100
                snapshot.avg_funding = (sum(rates) / len(rates)) * 100
        except Exception as exc:
            logger.warning("Failed to fetch funding history for %s: %s", symbol, exc)

        try:
            fr = self.client.funding_rate(symbol)
            if fr:
                snapshot.funding_rate = float(fr.get("fundingRate", fr.get("info", {}).get("fundingRate", 0))) * 100
        except Exception as exc:
            logger.warning("Failed to fetch current funding for %s: %s", symbol, exc)

        try:
            ticker = self.client.ticker(symbol)
            if isinstance(ticker, dict):
                snapshot.price = ticker.get("last") or ticker.get("close")
                snapshot.volume_24h = ticker.get("quoteVolume")
        except Exception as exc:
            logger.warning("Failed to fetch ticker for %s: %s", symbol, exc)

        # MEXC does not expose open interest/L-S ratio publicly via ccxt; leave as None
        snapshot.open_interest = None
        snapshot.oi_change_pct = None
        snapshot.oi_trend = "NEUTRAL"

        try:
            trades = self.client.trades(symbol)
        except Exception as exc:
            logger.warning("Failed to fetch trades for %s: %s", symbol, exc)
            trades = []

        buy_volume = 0.0
        sell_volume = 0.0
        for trade in trades:
            price = trade.get("price") or trade.get("info", {}).get("p")
            amount = trade.get("amount") or trade.get("info", {}).get("v")
            if price is None or amount is None:
                continue
            value = float(price) * float(amount)
            side = trade.get("side") or ("buy" if not trade.get("info", {}).get("m") else "sell")
            if side == "buy":
                buy_volume += value
            elif side == "sell":
                sell_volume += value

        total_flow = buy_volume + sell_volume
        snapshot.taker_buy_vol = buy_volume
        snapshot.taker_sell_vol = sell_volume
        if total_flow > 0:
            snapshot.taker_ratio = buy_volume / sell_volume if sell_volume else None
        else:
            snapshot.taker_ratio = None

        try:
            ohlcv = self.client.ohlcv(symbol, timeframe=period, limit=50)
            closes = [float(candle[4]) for candle in ohlcv if isinstance(candle, (list, tuple)) and len(candle) >= 5]
            if len(closes) >= 20:
                snapshot.ema_20 = self._calculate_ema(closes, 20)
        except Exception as exc:
            logger.debug("Failed to compute EMA20 for %s: %s", symbol, exc)

        return snapshot

    def _build_trade_levels(
        self,
        snapshot: FundingSnapshot,
        evaluation: Dict[str, object],
        period: str,
    ) -> Optional[Dict[str, object]]:
        direction_val = evaluation.get("direction")
        if not isinstance(direction_val, str) or direction_val not in {"BULLISH", "BEARISH"}:
            return None
        direction = direction_val

        price_raw = snapshot.price
        if not isinstance(price_raw, (int, float)):
            return None
        price = float(price_raw)

        period_val = period if isinstance(period, str) else "5m"

        atr_val = self._get_atr(snapshot.symbol, period_val)
        atr = float(atr_val) if isinstance(atr_val, (int, float)) else None
        fallback = price * 0.005
        atr = atr or fallback

        # Use new TP/SL calculator if available
        if TPSLCalculator is not None:
            # Get config for this bot and symbol
            if get_config_manager is not None:
                config_mgr = get_config_manager()
                risk_config = config_mgr.get_effective_risk("funding_bot", snapshot.symbol)
                sl_mult = risk_config.sl_atr_multiplier
                tp1_mult = risk_config.tp1_atr_multiplier
                tp2_mult = risk_config.tp2_atr_multiplier
                min_rr = risk_config.min_risk_reward
            else:
                sl_mult, tp1_mult, tp2_mult, min_rr = 1.5, 2.5, 4.0, 1.5
            
            calculator = TPSLCalculator(min_risk_reward=min_rr)
            levels = calculator.calculate(
                entry=price,
                direction=direction,
                atr=atr,
                sl_multiplier=sl_mult,
                tp1_multiplier=tp1_mult,
                tp2_multiplier=tp2_mult,
            )
            
            if not levels.is_valid:
                logger.info(
                    "Signal rejected for %s: %s",
                    snapshot.symbol, levels.rejection_reason
                )
                return None
            
            sl = levels.stop_loss
            tp1 = levels.take_profit_1
            tp2 = levels.take_profit_2
        else:
            # Fallback to original calculation
            risk = max(atr, price * 0.002)
            if direction == "BULLISH":
                sl = price - 1.5 * risk
                tp1 = price + 2.5 * risk
                tp2 = price + 4 * risk
            else:
                sl = price + 1.5 * risk
                tp1 = price - 2.5 * risk
                tp2 = price - 4 * risk

        signal_id = f"{snapshot.symbol}-{snapshot.timestamp}"
        return {
            "id": signal_id,
            "symbol": snapshot.symbol,
            "direction": direction,
            "entry": price,
            "stop_loss": sl,
            "take_profit_1": tp1,
            "take_profit_2": tp2,
            "created_at": snapshot.timestamp,
            "timeframe": period_val,
        }

    def _get_atr(self, symbol: str, timeframe: str) -> Optional[float]:
        key = f"{symbol}:{timeframe}"
        now_ts = time.time()
        cached = self.atr_cache.get(key)
        if cached and now_ts - cached[1] < self.interval:
            return cached[0]

        try:
            ohlcv = self.client.ohlcv(symbol, timeframe=timeframe, limit=50)
        except Exception as exc:
            logger.warning("Failed to fetch OHLCV for %s %s: %s", symbol, timeframe, exc)
            return None

        atr = self._calculate_atr(ohlcv, period=14)
        if atr:
            self.atr_cache[key] = (atr, now_ts)
        return atr

    @staticmethod
    def _calculate_atr(ohlcv: List[List[float]], period: int = 14) -> Optional[float]:
        if len(ohlcv) <= period:
            return None
        trs: List[float] = []
        prev_close = ohlcv[0][4]
        for candle in ohlcv[1:]:
            high = candle[2]
            low = candle[3]
            close = candle[4]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
            prev_close = close
        if len(trs) < period:
            return sum(trs) / len(trs) if trs else None
        return sum(trs[-period:]) / period

    @staticmethod
    def _calculate_ema(values: List[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        ema = values[0]
        multiplier = 2 / (period + 1)
        for price in values[1:]:
            ema = (price - ema) * multiplier + ema
        return ema

    def _format_message(
        self,
        snapshot: FundingSnapshot,
        evaluation: Dict[str, object],
        trade_levels: Optional[Dict[str, object]] = None,
    ) -> str:
        direction = evaluation["direction"]
        emoji = "🟢" if direction == "BULLISH" else "🔴" if direction == "BEARISH" else "⚪"

        lines = [f"{emoji} <b>{direction} FUNDING ALERT - {snapshot.symbol}/USDT</b>", ""]

        # Historical TP/SL stats per symbol
        if self.stats is not None:
            symbol_key = f"{snapshot.symbol}/USDT"
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
        if snapshot.price is not None:
            lines.append(f"💰 Price: <code>{snapshot.price:.6f}</code>")
        if snapshot.funding_rate is not None:
            avg = snapshot.avg_funding or 0.0
            lines.append(f"💹 Funding: <code>{snapshot.funding_rate:+.4f}%</code> (avg {avg:+.4f}%)")
        if snapshot.open_interest is not None:
            delta = snapshot.oi_change_pct
            delta_str = f"{delta:+.2f}%" if delta is not None else "N/A"
            lines.append(
                f"📊 OI: <code>{snapshot.open_interest:,.0f}</code> ({snapshot.oi_trend}, {delta_str})"
            )
        if snapshot.ls_ratio is not None and snapshot.long_pct is not None and snapshot.short_pct is not None:
            lines.append(
                f"⚖️ L/S Ratio: <code>{snapshot.ls_ratio:.2f}</code> (Longs {snapshot.long_pct:.1f}%, Shorts {snapshot.short_pct:.1f}%)"
            )
        if snapshot.taker_ratio is not None:
            buy = f"{snapshot.taker_buy_vol:,.0f}" if snapshot.taker_buy_vol else "N/A"
            sell = f"{snapshot.taker_sell_vol:,.0f}" if snapshot.taker_sell_vol else "N/A"
            lines.append(f"🫧 Taker Flow: <code>{snapshot.taker_ratio:.2f}</code> (Buy {buy} / Sell {sell})")

        reasons_val = evaluation.get("reasons", [])
        reasons: List[str] = [r for r in reasons_val if isinstance(r, str)] if isinstance(reasons_val, list) else []
        if reasons:
            lines.append("")
            lines.append("📝 Factors: " + "; ".join(reasons))

        if trade_levels:
            lines.append("")
            try:
                tp1 = float(cast(float, trade_levels.get("take_profit_1")))
                tp2 = float(cast(float, trade_levels.get("take_profit_2")))
                sl = float(cast(float, trade_levels.get("stop_loss")))
                lines.append(
                    "🎯 Targets: "
                    f"TP1 <code>{tp1:.6f}</code> | "
                    f"TP2 <code>{tp2:.6f}</code>"
                )
                lines.append(f"🛑 Stop: <code>{sl:.6f}</code>")
            except Exception as e:
                logger.error(f"Failed to calculate TP/SL for {snapshot.symbol}: {e}")

        lines.append("")
        lines.append(f"⏱️ Updated {snapshot.timestamp}")
        return "\n".join(lines)

    def _monitor_open_signals(self) -> None:
        signals = self.state.iter_signals()
        if not signals:
            return

        for signal_id, payload in list(signals.items()):
            symbol = payload.get("symbol")
            if not symbol:
                self.state.remove_signal(signal_id)
                continue

            try:
                ticker = self.client.ticker(symbol)
                price = ticker.get("last") or ticker.get("close")
            except Exception as exc:
                logger.warning("Failed to fetch ticker for open signal %s: %s", signal_id, exc)
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
                    message = (
                        f"🎯 {symbol}/USDT {direction} {result} hit!\n"
                        f"Entry <code>{entry:.6f}</code> | Last <code>{price:.6f}</code>\n"
                        f"TP1 <code>{tp1:.6f}</code> | TP2 <code>{tp2:.6f}</code> | SL <code>{sl:.6f}</code>"
                    )
                    self._dispatch(message)

                self.state.remove_signal(signal_id)

    def _dispatch(self, message: str) -> None:
        if self.notifier:
            if self.notifier.send_message(message):
                logger.info("Alert sent to Telegram")
            else:
                logger.error("Failed to send Telegram message; logging locally")
                logger.info("%s", message)
        else:
            logger.info("Alert:\n%s", message)

    def _check_signal_reversal(self, symbol: str, new_direction: str) -> None:
        """Warn if a new signal opposes an existing open signal for the same symbol."""
        open_signals = self.state.iter_signals()
        for signal_id, payload in open_signals.items():
            if not isinstance(payload, dict):
                continue
            signal_symbol = payload.get("symbol")
            if signal_symbol not in (symbol, f"{symbol}/USDT"):
                continue
            open_direction = payload.get("direction")
            if not isinstance(open_direction, str):
                continue
            opposite = (
                (new_direction == "BULLISH" and open_direction == "BEARISH")
                or (new_direction == "BEARISH" and open_direction == "BULLISH")
            )
            if opposite:
                warning_msg = (
                    "⚠️ <b>SIGNAL REVERSAL DETECTED</b> ⚠️\n\n"
                    f"<b>Symbol:</b> {symbol}\n"
                    f"<b>Open Position:</b> {open_direction}\n"
                    f"<b>New Signal:</b> {new_direction}\n\n"
                    "💡 <b>Action:</b> Consider exiting your open position!\n"
                    "🔄 <b>Market may be reversing</b>\n\n"
                    f"🆔 Open Signal: {signal_id}\n"
                    f"⏰ {datetime.now(timezone.utc).isoformat()}"
                )
                if self.notifier:
                    self.notifier.send_message(warning_msg)
                logger.info("Signal reversal detected for %s: %s -> %s", symbol, open_direction, new_direction)
                break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Funding/OI monitoring bot")
    parser.add_argument("--loop", action="store_true", help="Run indefinitely")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between cycles")
    parser.add_argument("--cooldown", type=int, default=45, help="Default cooldown minutes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bot = FundingBot(interval=args.interval, default_cooldown=args.cooldown)
    bot.run(loop=args.loop)


if __name__ == "__main__":
    main()
