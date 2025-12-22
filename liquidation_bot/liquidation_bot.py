#!/usr/bin/env python3
"""Automated liquidation/orderflow alert bot based on liquidations.py."""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, pstdev
from threading import Lock
from types import FrameType
from typing import Any, Dict, List, Optional, TypedDict, cast

import ccxt  # type: ignore[import-untyped]

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
STATE_FILE = BASE_DIR / "liquidation_state.json"
WATCHLIST_FILE = BASE_DIR / "liquidation_watchlist.json"
STATS_FILE = LOG_DIR / "liquidation_stats.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "liquidation_bot.log"),
    ],
)

logger = logging.getLogger("liquidation_bot")

try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
    load_dotenv(BASE_DIR.parent / ".env")
except ImportError:
    pass

sys_path_added = False
if str(BASE_DIR.parent) not in sys.path:
    sys.path.append(str(BASE_DIR.parent))
    sys_path_added = True

# Required imports (fail fast if missing)
from message_templates import format_signal_message, format_result_message
from notifier import TelegramNotifier
from signal_stats import SignalStats
from tp_sl_calculator import TPSLCalculator
from trade_config import get_config_manager

# Optional imports (safe fallback)
from safe_import import safe_import
HealthMonitor = safe_import('health_monitor', 'HealthMonitor')
RateLimiter = None  # Disabled for testing
RateLimitHandler = safe_import('rate_limit_handler', 'RateLimitHandler')


# Graceful shutdown handling
shutdown_requested = False


def signal_handler(signum: int, frame: Optional[FrameType]) -> None:  # pragma: no cover - signal path
    """Handle shutdown signals (SIGINT, SIGTERM) gracefully."""
    global shutdown_requested
    shutdown_requested = True
    logger.info("Received %s, shutting down gracefully...", signal.Signals(signum).name)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol to always have /USDT suffix (without duplication).

    Handles cases where symbol may already contain /USDT to avoid
    creating malformed symbols like 'NIGHT/USDT/USDT'.
    """
    base = symbol.replace("/USDT", "").replace("_USDT", "")
    return f"{base}/USDT"


class WatchItem(TypedDict, total=False):
    symbol: str
    cooldown_minutes: int
    timeframe: str


def load_watchlist() -> List[WatchItem]:
    if not WATCHLIST_FILE.exists():
        logger.error("Watchlist file missing: %s", WATCHLIST_FILE)
        return []
    try:
        data = json.loads(WATCHLIST_FILE.read_text())
    except json.JSONDecodeError as exc:
        logger.error("Invalid watchlist JSON: %s", exc)
        return []
    result: List[WatchItem] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        symbol_val = item.get("symbol")
        if not isinstance(symbol_val, str):
            continue
        cooldown_val = item.get("cooldown_minutes", 30)
        try:
            cooldown = int(cooldown_val)
        except Exception:
            cooldown = 30
        timeframe_val = item.get("timeframe") or item.get("period") or "5m"
        timeframe = timeframe_val if isinstance(timeframe_val, str) else "5m"
        result.append(
            {
                "symbol": symbol_val.upper(),
                "cooldown_minutes": cooldown,
                "timeframe": timeframe,
            }
        )
    return result


class MexcOrderflowClient:
    def __init__(self) -> None:
        self.exchange = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "swap"}})
        self.exchange.load_markets()
        # Initialize rate limit handler
        self.rate_limiter = RateLimitHandler(base_delay=0.5, max_retries=5) if RateLimitHandler else None

    @staticmethod
    def _symbol(symbol: str) -> str:
        return f"{symbol.replace("/USDT", "")}/USDT:USDT"

    def ticker(self, symbol: str) -> Dict[str, Any]:
        if self.rate_limiter:
            return cast(Dict[str, Any], self.rate_limiter.execute(self.exchange.fetch_ticker, self._symbol(symbol)))
        return cast(Dict[str, Any], self.exchange.fetch_ticker(self._symbol(symbol)))

    def orderbook(self, symbol: str, limit: int = 50) -> Dict[str, Any]:
        if self.rate_limiter:
            return cast(Dict[str, Any], self.rate_limiter.execute(self.exchange.fetch_order_book, self._symbol(symbol), limit=limit))
        return cast(Dict[str, Any], self.exchange.fetch_order_book(self._symbol(symbol), limit=limit))

    def trades(self, symbol: str, limit: int = 1000) -> List[Any]:
        if self.rate_limiter:
            return cast(List[Any], self.rate_limiter.execute(self.exchange.fetch_trades, self._symbol(symbol), limit=limit))
        return cast(List[Any], self.exchange.fetch_trades(self._symbol(symbol), limit=limit))

    def ohlcv(self, symbol: str, timeframe: str = "5m", limit: int = 100) -> List[Any]:
        if self.rate_limiter:
            return cast(List[Any], self.rate_limiter.execute(self.exchange.fetch_ohlcv, self._symbol(symbol), timeframe=timeframe, limit=limit))
        return cast(List[Any], self.exchange.fetch_ohlcv(self._symbol(symbol), timeframe=timeframe, limit=limit))


@dataclass
class LiquidationSnapshot:
    symbol: str
    timestamp: str
    price: Optional[float] = None
    price_change_pct: Optional[float] = None
    volume_24h: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    liquidations: List[Dict[str, Any]] = field(default_factory=list)
    orderbook: Optional[Dict[str, Any]] = None
    trades: List[Dict[str, Any]] = field(default_factory=list)
    long_liq_value: float = 0.0
    short_liq_value: float = 0.0
    bid_value: float = 0.0
    ask_value: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_middle: Optional[float] = None
    large_trade_threshold: Optional[float] = None


class LiquidationEvaluator:
    LIQ_IMBALANCE = 1.5
    ORDERBOOK_THRESHOLD = 1.3
    FLOW_DELTA = 0.6  # >60% buy or sell

    def evaluate(self, snapshot: LiquidationSnapshot) -> Dict[str, object]:
        reasons: List[str] = []
        score = 0
        has_liq_factor = False

        # More balanced scoring - BEARISH signals were getting over-penalized
        if snapshot.short_liq_value > snapshot.long_liq_value * self.LIQ_IMBALANCE:
            score += 2  # Reduced from 3 to balance with other factors
            reasons.append("Large short liquidations (bullish)")
            has_liq_factor = True
        elif snapshot.long_liq_value > snapshot.short_liq_value * self.LIQ_IMBALANCE:
            score -= 2  # Reduced from -3 to balance with other factors
            reasons.append("Large long liquidations (bearish)")
            has_liq_factor = True

        if snapshot.bid_value and snapshot.ask_value:
            ratio = snapshot.bid_value / snapshot.ask_value if snapshot.ask_value > 0 else 0
            if ratio >= self.ORDERBOOK_THRESHOLD:
                score += 1
                reasons.append("Orderbook support > resistance")
            elif ratio <= 1 / self.ORDERBOOK_THRESHOLD:
                score -= 1
                reasons.append("Orderbook resistance > support")

        total_flow = snapshot.buy_volume + snapshot.sell_volume
        if total_flow > 0:
            buy_ratio = snapshot.buy_volume / total_flow
            if buy_ratio >= self.FLOW_DELTA:
                score += 1
                reasons.append("Aggressive buyers dominate")
            elif buy_ratio <= 1 - self.FLOW_DELTA:
                score -= 1
                reasons.append("Aggressive sellers dominate")

        if not has_liq_factor and abs(score) > 1:
            score = 1 if score > 0 else -1

        price = snapshot.price
        upper = snapshot.bb_upper
        lower = snapshot.bb_lower
        location_state: Optional[str] = None
        if isinstance(price, (int, float)) and isinstance(upper, (int, float)) and isinstance(lower, (int, float)):
            if price <= lower:
                location_state = "BULLISH"
                reasons.append("Price at lower Bollinger band")
            elif price >= upper:
                location_state = "BEARISH"
                reasons.append("Price at upper Bollinger band")
            else:
                location_state = "INSIDE"
                reasons.append("Price inside Bollinger bands (no reversal edge)")
                score = 0

        # Adjusted thresholds with confluence requirements
        direction = "NEUTRAL"
        if score >= 2 and has_liq_factor and (location_state in (None, "BULLISH")):
            direction = "BULLISH"
        elif score <= -2 and has_liq_factor and (location_state in (None, "BEARISH")):
            direction = "BEARISH"
        else:
            score = 0 if direction == "NEUTRAL" else score

        return {
            "direction": direction,
            "score": score,
            "reasons": reasons,
            "should_alert": direction != "NEUTRAL" and bool(reasons),
        }


class LiquidationState:
    def __init__(self, path: Path):
        self.path = path
        self.state_lock = Lock()
        self.data: Dict[str, Any] = self._load()

    def _empty_state(self) -> Dict[str, Any]:
        return {"last_alert": {}, "open_signals": {}, "closed_signals": {}}

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return self._empty_state()
        try:
            data = json.loads(self.path.read_text())
            if not isinstance(data, dict):
                return self._empty_state()
            last_alert = data.get("last_alert")
            open_signals = data.get("open_signals")
            closed_signals = data.get("closed_signals")
            return {
                "last_alert": last_alert if isinstance(last_alert, dict) else {},
                "open_signals": open_signals if isinstance(open_signals, dict) else {},
                "closed_signals": closed_signals if isinstance(closed_signals, dict) else {},
            }
        except json.JSONDecodeError:
            return self._empty_state()

    def save(self) -> None:
        """Save state to file with file locking for concurrent access safety."""
        with self.state_lock:
            temp_file = self.path.with_suffix('.tmp')
            try:
                with open(temp_file, 'w') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    json.dump(self.data, f, indent=2)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                temp_file.replace(self.path)
            except Exception as e:
                logger.error("Failed to save state: %s", e)
                if temp_file.exists():
                    temp_file.unlink()

    @staticmethod
    def _parse_ts(value: str) -> datetime:
        """Parse ISO format timestamp, ensuring UTC timezone."""
        try:
            dt = datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt

    def can_alert(self, symbol: str, cooldown_minutes: int) -> bool:
        last_map = self.data.setdefault("last_alert", {})
        if not isinstance(last_map, dict):
            self.data["last_alert"] = {}
            return True
        last_ts = last_map.get(symbol)
        if not isinstance(last_ts, str):
            return True
        delta = utc_now() - self._parse_ts(last_ts)
        return delta.total_seconds() >= cooldown_minutes * 60

    def mark_alert(self, symbol: str) -> None:
        last_map = self.data.setdefault("last_alert", {})
        if not isinstance(last_map, dict):
            last_map = {}
            self.data["last_alert"] = last_map
        last_map[symbol] = utc_now().isoformat()
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

    def cleanup_stale_signals(self, max_age_hours: int = 24) -> int:
        """Remove signals older than max_age_hours and move to closed_signals."""
        signals = self.iter_signals()
        closed = self.data.setdefault("closed_signals", {})
        if not isinstance(closed, dict):
            closed = {}
            self.data["closed_signals"] = closed

        current_time = datetime.now(timezone.utc)
        removed_count = 0
        signal_ids_to_remove: List[str] = []

        for signal_id, payload in list(signals.items()):
            if not isinstance(payload, dict):
                signal_ids_to_remove.append(signal_id)
                continue

            created_at_str = payload.get("created_at")
            if not isinstance(created_at_str, str):
                signal_ids_to_remove.append(signal_id)
                continue

            try:
                created_time = self._parse_ts(created_at_str)
                age = current_time - created_time

                if age >= timedelta(hours=max_age_hours):
                    # Move to closed signals with timeout status
                    closed[signal_id] = {**payload, "closed_reason": "TIMEOUT", "closed_at": current_time.isoformat()}
                    signal_ids_to_remove.append(signal_id)
                    removed_count += 1
                    logger.info("Stale signal removed: %s (age: %.1f hours)", signal_id, age.total_seconds() / 3600)
            except (ValueError, TypeError):
                signal_ids_to_remove.append(signal_id)

        for signal_id in signal_ids_to_remove:
            if signal_id in signals:
                signals.pop(signal_id)

        if removed_count > 0:
            self.save()

        return removed_count


class LiquidationBot:
    def __init__(self, interval: int = 60):
        self.interval = interval
        self.watchlist: List[WatchItem] = load_watchlist()
        self.client = MexcOrderflowClient()
        self.evaluator = LiquidationEvaluator()
        self.state = LiquidationState(STATE_FILE)
        self.notifier = self._init_notifier()
        self.stats = SignalStats("Liquidation Bot", STATS_FILE)
        self.atr_cache: Dict[str, tuple[float, float]] = {}
        self.health_monitor = HealthMonitor("Liquidation Bot", self.notifier, heartbeat_interval=3600) if HealthMonitor else None
        self.rate_limiter = RateLimiter(calls_per_minute=60, backoff_file=LOG_DIR / "rate_limiter.json") if RateLimiter else None
        self.large_trade_multiplier = float(os.getenv("LIQ_LARGE_TRADE_MULTIPLIER", "20"))

    def _init_notifier(self) -> Optional[Any]:
        if TelegramNotifier is None:
            logger.warning("Telegram notifier unavailable")
            return None
        token = os.getenv("TELEGRAM_BOT_TOKEN_LIQUIDATION") or os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            logger.warning("Telegram credentials missing")
            return None
        return TelegramNotifier(token, chat_id, signals_log_file=str(LOG_DIR / "liquidation_signals.json"))

    def run(self, loop: bool = False) -> None:
        if not self.watchlist:
            logger.error("Watchlist is empty")
            return

        logger.info("Starting liquidation bot for %d symbols", len(self.watchlist))

        # Send startup notification
        if self.health_monitor:
            self.health_monitor.send_startup_message()

        try:
            while not shutdown_requested:
                try:
                    self._run_cycle()

                    # Cleanup stale signals every cycle
                    stale_count = self.state.cleanup_stale_signals(max_age_hours=24)
                    if stale_count > 0:
                        logger.info("Cleaned up %d stale signals", stale_count)

                    self._monitor_open_signals()

                    # Record successful cycle
                    if self.health_monitor:
                        self.health_monitor.record_cycle()

                    if not loop:
                        break

                    if shutdown_requested:
                        break

                    logger.info("Cycle complete; sleeping %ds", self.interval)
                    # Sleep in 1-second chunks to respond quickly to shutdown signals
                    for _ in range(self.interval):
                        time.sleep(1)
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
            cooldown_raw = item.get("cooldown_minutes") if isinstance(item, dict) else None
            try:
                cooldown = int(cooldown_raw) if cooldown_raw is not None else 30
            except Exception:
                cooldown = 30
            timeframe_val = item.get("timeframe") if isinstance(item, dict) else None
            timeframe = timeframe_val if isinstance(timeframe_val, str) else "5m"

            # Apply rate limiting
            if self.rate_limiter:
                self.rate_limiter.wait_if_needed()

            try:
                snapshot = self._collect(symbol, timeframe)
                if self.rate_limiter:
                    self.rate_limiter.record_success(f"mexc_{symbol}")
            except Exception as exc:
                logger.error("Failed to collect data for %s: %s", symbol, exc)
                if self.rate_limiter:
                    self.rate_limiter.record_error(f"mexc_{symbol}")
                if self.health_monitor:
                    self.health_monitor.record_error(f"API error for {symbol}: {exc}")
                continue

            evaluation = self.evaluator.evaluate(snapshot)
            if not evaluation["should_alert"]:
                continue

            if not self.state.can_alert(symbol, cooldown):
                continue

            trade_levels = self._build_trade_levels(snapshot, evaluation, timeframe)

            message = self._format_message(snapshot, evaluation, trade_levels)
            self._dispatch(message)
            self.state.mark_alert(symbol)
            if trade_levels and isinstance(trade_levels, dict):
                signal_id_val = trade_levels.get("id")
                direction_val = trade_levels.get("direction")
                entry_val = trade_levels.get("entry")
                created_at_val = trade_levels.get("created_at")
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

                trade_levels["exchange"] = "MEXC"
                trade_levels.setdefault("timeframe", timeframe)
                self.state.add_signal(signal_id_val, trade_levels)
                if self.stats:
                    self.stats.record_open(
                        signal_id=signal_id_val,
                        symbol=normalize_symbol(symbol),
                        direction=direction_val,
                        entry=float(entry_val),
                        created_at=created_at_val,
                        extra={
                            "timeframe": trade_levels.get("timeframe", timeframe),
                            "exchange": "MEXC",
                            "display_symbol": normalize_symbol(symbol),
                        },
                    )
            time.sleep(0.2)

    def _collect(self, symbol: str, timeframe: str) -> LiquidationSnapshot:
        snapshot = LiquidationSnapshot(symbol=symbol, timestamp=utc_now().strftime("%Y-%m-%dT%H:%M:%SZ"))

        ticker = self.client.ticker(symbol)
        if isinstance(ticker, dict):
            snapshot.price = ticker.get("last") or ticker.get("close")
            snapshot.price_change_pct = ticker.get("percentage")
            snapshot.volume_24h = ticker.get("quoteVolume")
            snapshot.high_24h = ticker.get("high")
            snapshot.low_24h = ticker.get("low")

        try:
            ohlcv = self.client.ohlcv(symbol, timeframe=timeframe, limit=60)
            closes = [float(candle[4]) for candle in ohlcv if isinstance(candle, list) and len(candle) >= 5]
            if len(closes) >= 20:
                window = closes[-20:]
                mid = mean(window)
                std_dev = pstdev(window)
                snapshot.bb_middle = mid
                snapshot.bb_upper = mid + 2 * std_dev
                snapshot.bb_lower = mid - 2 * std_dev
        except Exception as exc:  # pragma: no cover - network variability
            logger.debug("Failed to compute Bollinger Bands for %s: %s", symbol, exc)

        snapshot.orderbook = self.client.orderbook(symbol)
        bids = snapshot.orderbook.get("bids", []) if isinstance(snapshot.orderbook, dict) else []
        asks = snapshot.orderbook.get("asks", []) if isinstance(snapshot.orderbook, dict) else []

        def _level_value(levels: List[List[float]]) -> float:
            total = 0.0
            for level in levels:
                if len(level) < 2:
                    continue
                price, amount = level[0], level[1]
                total += float(price) * float(amount)
            return total

        snapshot.bid_value = _level_value(bids)
        snapshot.ask_value = _level_value(asks)

        trades = self.client.trades(symbol)
        snapshot.trades = trades if isinstance(trades, list) else []

        parsed_trades: List[Dict[str, Any]] = []
        for trade in snapshot.trades:
            info = trade.get("info", {}) if isinstance(trade, dict) else {}
            price = trade.get("price") or info.get("p")
            amount = trade.get("amount") or info.get("v")
            if price is None or amount is None:
                continue
            try:
                price_f = float(price)
                amount_f = float(amount)
            except (TypeError, ValueError):
                continue
            value = price_f * amount_f
            side = trade.get("side") or ("buy" if not info.get("m") else "sell")
            parsed_trades.append({
                "value": value,
                "side": side,
                "price": price_f,
                "timestamp": trade.get("timestamp"),
            })

        sample_values = [item["value"] for item in parsed_trades[-100:]]
        avg_trade_value = (sum(sample_values) / len(sample_values)) if sample_values else None
        large_trade_threshold = 10_000.0
        if avg_trade_value and avg_trade_value > 0:
            large_trade_threshold = max(avg_trade_value * self.large_trade_multiplier, 1_000.0)
        snapshot.large_trade_threshold = large_trade_threshold

        buy_volume = sell_volume = 0.0
        long_liq_value = short_liq_value = 0.0
        inferred_liqs: List[Dict[str, Any]] = []

        for trade in parsed_trades:
            value = trade["value"]
            side = trade["side"]
            price = trade["price"]
            timestamp = trade.get("timestamp")

            if side == "buy":
                buy_volume += value
                if value >= large_trade_threshold:
                    short_liq_value += value
                    inferred_liqs.append({"side": "BUY", "price": price, "value": value, "timestamp": timestamp})
            elif side == "sell":
                sell_volume += value
                if value >= large_trade_threshold:
                    long_liq_value += value
                    inferred_liqs.append({"side": "SELL", "price": price, "value": value, "timestamp": timestamp})

        snapshot.buy_volume = buy_volume
        snapshot.sell_volume = sell_volume
        snapshot.short_liq_value = short_liq_value
        snapshot.long_liq_value = long_liq_value
        snapshot.liquidations = inferred_liqs

        return snapshot

    def _build_trade_levels(
        self, snapshot: LiquidationSnapshot, evaluation: Dict[str, object], timeframe: str
    ) -> Optional[Dict[str, object]]:
        direction_val = evaluation.get("direction")
        if not isinstance(direction_val, str) or direction_val not in {"BULLISH", "BEARISH"}:
            return None
        direction = direction_val

        price_raw = snapshot.price
        if not isinstance(price_raw, (int, float)):
            return None
        price = float(price_raw)

        timeframe_val = timeframe if isinstance(timeframe, str) else "5m"

        atr_val = self._get_atr(snapshot.symbol, timeframe_val)
        atr = float(atr_val) if isinstance(atr_val, (int, float)) else None
        fallback = price * 0.0075
        atr = atr or fallback

        # Get risk config and calculate TP/SL using centralized calculator
        config_mgr = get_config_manager()
        risk_config = config_mgr.get_effective_risk("liquidation_bot", snapshot.symbol)
        sl_mult = risk_config.sl_atr_multiplier
        tp1_mult = risk_config.tp1_atr_multiplier
        tp2_mult = risk_config.tp2_atr_multiplier
        min_rr = risk_config.min_risk_reward

        calculator = TPSLCalculator(min_risk_reward=0.8, min_risk_reward_tp2=1.5)
        levels = calculator.calculate(
            entry=price,
            direction=direction,
            atr=atr,
            sl_multiplier=sl_mult,
            tp1_multiplier=tp1_mult,
            tp2_multiplier=tp2_mult,
        )

        if not levels.is_valid:
            logger.info("Signal rejected for %s: %s", snapshot.symbol, levels.rejection_reason)
            return None

        sl = levels.stop_loss
        tp1 = levels.take_profit_1
        tp2 = levels.take_profit_2

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
            "timeframe": timeframe_val,
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

        atr = self._calculate_atr(ohlcv)
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

    def _format_message(
        self,
        snapshot: LiquidationSnapshot,
        evaluation: Dict[str, object],
        trade_levels: Optional[Dict[str, object]] = None,
    ) -> str:
        direction = str(evaluation.get("direction", "NEUTRAL"))
        if direction not in ("BULLISH", "BEARISH"):
            direction = "BULLISH"  # fallback

        # Build extra info with liquidation-specific data
        extra_lines = []
        if snapshot.long_liq_value or snapshot.short_liq_value:
            extra_lines.append(f"Liqs: Longs ${snapshot.long_liq_value:,.0f} | Shorts ${snapshot.short_liq_value:,.0f}")
        if snapshot.bid_value or snapshot.ask_value:
            ratio = (snapshot.bid_value / snapshot.ask_value) if snapshot.ask_value else 0
            extra_lines.append(f"Orderbook ratio: {ratio:.2f}")
        if snapshot.buy_volume or snapshot.sell_volume:
            total = snapshot.buy_volume + snapshot.sell_volume
            if total > 0:
                buy_pct = snapshot.buy_volume / total * 100
                extra_lines.append(f"Flow: Buy {buy_pct:.1f}%")

        extra_info = " | ".join(extra_lines) if extra_lines else None

        # Get reasons from evaluation
        reasons_val = evaluation.get("reasons", [])
        reasons = [r for r in reasons_val if isinstance(r, str)] if isinstance(reasons_val, list) else []

        # Get performance stats
        perf_stats = None
        if self.stats is not None:
            symbol_key = normalize_symbol(snapshot.symbol)
            counts = self.stats.symbol_tp_sl_counts(symbol_key)
            tp1_count = counts.get("TP1", 0)
            tp2_count = counts.get("TP2", 0)
            sl_count = counts.get("SL", 0)
            total_count = tp1_count + tp2_count + sl_count
            if total_count > 0:
                perf_stats = {
                    "tp1": tp1_count,
                    "tp2": tp2_count,
                    "sl": sl_count,
                    "wins": tp1_count + tp2_count,
                    "total": total_count,
                }

        # Build signal message using centralized template
        entry = float(cast(float, trade_levels.get("entry", snapshot.price or 0))) if trade_levels else float(snapshot.price or 0)
        sl = float(cast(float, trade_levels.get("stop_loss", entry * 0.98))) if trade_levels else entry * 0.98
        tp1 = float(cast(float, trade_levels.get("take_profit_1", entry * 1.03))) if trade_levels else entry * 1.03
        tp2 = float(cast(float, trade_levels.get("take_profit_2", entry * 1.05))) if trade_levels else entry * 1.05
        timeframe = str(trade_levels.get("timeframe", "15m")) if trade_levels else "15m"

        return format_signal_message(
            bot_name="LIQUIDATION",
            symbol=normalize_symbol(snapshot.symbol),
            direction=direction,
            entry=entry,
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
            reasons=reasons if reasons else None,
            exchange="MEXC",
            timeframe=timeframe,
            current_price=float(snapshot.price) if snapshot.price else None,
            performance_stats=perf_stats,
            extra_info=extra_info,
        )

    def _dispatch(self, message: str) -> None:
        if self.notifier:
            if self.notifier.send_message(message):
                logger.info("Alert sent to Telegram")
            else:
                logger.error("Failed to send Telegram message; logging locally")
                logger.info("%s", message)
        else:
            logger.info("Alert:\n%s", message)

    def _monitor_open_signals(self) -> None:
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
                ticker = self.client.ticker(symbol)
                price_raw = ticker.get("last") if isinstance(ticker, dict) else None
                if price_raw is None and isinstance(ticker, dict):
                    price_raw = ticker.get("close")
            except Exception as exc:
                logger.warning("Failed to fetch ticker for open liquidation signal %s: %s", signal_id, exc)
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

            # Check for TP/SL hits with price tolerance
            PRICE_TOLERANCE = 0.005  # 0.5% tolerance for slippage

            if direction == "BULLISH":
                # With tolerance for slippage (allow slightly lower prices)
                hit_tp2 = price >= (tp2 * (1 - PRICE_TOLERANCE))
                hit_tp1 = price >= (tp1 * (1 - PRICE_TOLERANCE)) and price < (tp2 * (1 - PRICE_TOLERANCE))
                hit_sl = price <= (sl * (1 + PRICE_TOLERANCE))
            else:
                # For BEARISH, allow slightly higher prices
                hit_tp2 = price <= (tp2 * (1 + PRICE_TOLERANCE))
                hit_tp1 = price <= (tp1 * (1 + PRICE_TOLERANCE)) and price > (tp2 * (1 + PRICE_TOLERANCE))
                hit_sl = price >= (sl * (1 - PRICE_TOLERANCE))

            result = None
            if hit_tp2:
                result = "TP2"
            elif hit_tp1:
                result = "TP1"
            elif hit_sl:
                result = "SL"

            if result:
                # Get updated performance stats
                perf_stats = None
                if self.stats:
                    stats_record = self.stats.record_close(
                        signal_id,
                        exit_price=price,
                        result=result,
                    )
                    if stats_record:
                        logger.info("Trade closed: %s | %s | Entry: %.6f | Exit: %.6f | Result: %s | P&L: %.2f%%",
                                   signal_id, symbol, entry, price, result, stats_record.pnl_pct)
                        # Build perf stats for result message
                        symbol_key = normalize_symbol(symbol)
                        counts = self.stats.symbol_tp_sl_counts(symbol_key)
                        tp1_count = counts.get("TP1", 0)
                        tp2_count = counts.get("TP2", 0)
                        sl_count = counts.get("SL", 0)
                        total_count = tp1_count + tp2_count + sl_count
                        if total_count > 0:
                            perf_stats = {
                                "tp1": tp1_count,
                                "tp2": tp2_count,
                                "sl": sl_count,
                                "wins": tp1_count + tp2_count,
                                "total": total_count,
                            }
                    else:
                        self.stats.discard(signal_id)

                # Use centralized result template
                message = format_result_message(
                    symbol=normalize_symbol(symbol),
                    direction=direction,
                    result=result,
                    entry=entry,
                    exit_price=price,
                    stop_loss=sl,
                    tp1=tp1,
                    tp2=tp2,
                    signal_id=signal_id,
                    performance_stats=perf_stats,
                )
                self._dispatch(message)
                self.state.remove_signal(signal_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Liquidation/orderflow monitoring bot")
    parser.add_argument("--once", action="store_true", help="Run only one cycle")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between cycles")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bot = LiquidationBot(interval=args.interval)
    bot.run(loop=not args.once)


if __name__ == "__main__":
    main()
