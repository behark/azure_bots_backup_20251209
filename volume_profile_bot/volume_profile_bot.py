#!/usr/bin/env python3
"""Volume Profile Pivot-Anchored bot.

This bot ports the TradingView "Volume Profile, Pivot Anchored" indicator logic into
Python so we can alert whenever price interacts with Point of Control, Value Area
High/Low, or when a high-volume bar appears. Alerts are sent via Telegram and
respect per-symbol cooldowns from the watchlist file.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast
from types import FrameType
from uuid import uuid4

import ccxt  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
LOG_DIR = ROOT_DIR / "logs"
STATE_FILE = BASE_DIR / "state.json"
WATCHLIST_FILE = BASE_DIR / "watchlist.json"
STATS_FILE = LOG_DIR / "volume_profile_stats.json"

# Exchange configuration - actual exchange used for data
EXCHANGE_NAME = "Binance Futures"

# Module-level lock for state file access
_state_lock = Lock()

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "volume_profile.log"),
    ],
)

logger = logging.getLogger("volume_profile_bot")


load_dotenv(ROOT_DIR / ".env")
load_dotenv(BASE_DIR / ".env")

sys.path.insert(0, str(ROOT_DIR))

# Required imports (fail fast if missing)
from message_templates import format_signal_message
from notifier import TelegramNotifier
from signal_stats import SignalStats
from tp_sl_calculator import TPSLCalculator, CalculationMethod

# Optional imports (safe fallback)
from safe_import import safe_import
HealthMonitor = safe_import('health_monitor', 'HealthMonitor', logger_instance=logger)
RateLimitHandler = safe_import('rate_limit_handler', 'RateLimitHandler', logger_instance=logger)


shutdown_requested = False


def signal_handler(signum: int, frame: Optional[FrameType]) -> None:  # pragma: no cover - signal path
    global shutdown_requested
    shutdown_requested = True
    logger.info("Received %s, shutting down soon", signal.Signals(signum).name)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol to always have /USDT suffix (without duplication).

    Handles cases where symbol may already contain /USDT to avoid
    creating malformed symbols like 'NIGHT/USDT/USDT'.
    """
    base = symbol.replace("/USDT", "").replace("_USDT", "")
    return f"{base}/USDT"


def load_watchlist() -> List[Dict[str, object]]:
    if not WATCHLIST_FILE.exists():
        raise FileNotFoundError(f"Watchlist file missing: {WATCHLIST_FILE}")
    return cast(List[Dict[str, object]], json.loads(WATCHLIST_FILE.read_text()))


def load_state() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        return {"last_alerts": {}, "open_positions": {}}
    try:
        data = json.loads(STATE_FILE.read_text() or "{}")
    except json.JSONDecodeError:
        data = {}
    data.setdefault("last_alerts", {})
    data.setdefault("open_positions", {})
    return cast(Dict[str, Any], data)


def save_state(state: Dict[str, Any]) -> None:
    """Save state to file with file locking for concurrent access safety."""
    with _state_lock:
        temp_file = STATE_FILE.with_suffix('.tmp')
        try:
            with open(temp_file, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(state, f, indent=2)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            temp_file.replace(STATE_FILE)
        except Exception as e:
            logger.error("Failed to save state: %s", e)
            if temp_file.exists():
                temp_file.unlink()


@dataclass
class IndicatorConfig:
    pivot_length: int = 20
    profile_bins: int = 25
    value_area_pct: float = 0.68
    volume_ma_length: int = 89
    high_volume_mult: float = 1.618


@dataclass
class TradeSettings:
    range_risk_fraction: float = 0.5  # fraction of VA range used to size risk
    buffer_fraction: float = 0.05  # extra padding beyond VA boundaries
    tp1_rr: float = 1.2  # risk/reward multiplier for TP1
    tp2_rr: float = 1.8  # risk/reward multiplier for TP2
    max_positions_per_symbol: int = 40


class PivotVolumeProfile:
    def __init__(self, config: IndicatorConfig):
        self.config = config

    def compute(self, highs: npt.NDArray[np.floating[Any]], lows: npt.NDArray[np.floating[Any]], closes: npt.NDArray[np.floating[Any]], volumes: npt.NDArray[np.floating[Any]]) -> Optional[Dict[str, float]]:
        events = self._detect_pivots(highs, lows)
        if len(events) < 2:
            return None
        start_idx, end_idx = events[-2][1], events[-1][1]
        if start_idx == end_idx:
            return None
        lo = min(start_idx, end_idx)
        hi = max(start_idx, end_idx)
        segment_high = float(np.max(highs[lo : hi + 1]))
        segment_low = float(np.min(lows[lo : hi + 1]))
        if segment_high <= segment_low:
            return None
        bins = self.config.profile_bins
        step = (segment_high - segment_low) / bins
        if step <= 0:
            return None
        vol_profile = np.zeros(bins)
        prices = (highs + lows) / 2.0
        for idx in range(lo, hi + 1):
            price = prices[idx]
            bin_idx = int((price - segment_low) / step)
            bin_idx = max(0, min(bins - 1, bin_idx))
            vol_profile[bin_idx] += volumes[idx]

        total_volume = float(np.sum(vol_profile))
        if total_volume <= 0:
            return None
        poc_idx = int(np.argmax(vol_profile))
        value_bins = {poc_idx}
        cumulative = vol_profile[poc_idx]
        target = total_volume * self.config.value_area_pct
        offset = 1
        while cumulative < target and (poc_idx - offset >= 0 or poc_idx + offset < bins):
            left_idx = poc_idx - offset
            right_idx = poc_idx + offset
            left_vol = vol_profile[left_idx] if left_idx >= 0 else -1.0
            right_vol = vol_profile[right_idx] if right_idx < bins else -1.0
            if right_vol >= left_vol and right_idx < bins:
                value_bins.add(right_idx)
                cumulative += right_vol
            if cumulative >= target:
                break
            if left_idx >= 0:
                value_bins.add(left_idx)
                cumulative += max(left_vol, 0.0)
            offset += 1

        poc_price = segment_low + (poc_idx + 0.5) * step
        vah_idx = max(value_bins)
        val_idx = min(value_bins)
        vah_price = segment_low + (vah_idx + 1.0) * step
        val_price = segment_low + val_idx * step

        vols = volumes.astype(float)
        if len(vols) < self.config.volume_ma_length + 1:
            return {
                "poc": poc_price,
                "vah": vah_price,
                "val": val_price,
                "high_volume": False,
            }
        window = vols[-self.config.volume_ma_length :]
        vol_ma = float(np.mean(window))
        high_volume = vols[-1] > vol_ma * self.config.high_volume_mult
        return {
            "poc": poc_price,
            "vah": vah_price,
            "val": val_price,
            "high_volume": high_volume,
        }

    def _detect_pivots(self, highs: npt.NDArray[np.floating[Any]], lows: npt.NDArray[np.floating[Any]]) -> List[Tuple[str, int, float]]:
        length = self.config.pivot_length
        events: List[Tuple[str, int, float]] = []
        for idx in range(length, len(highs) - length):
            segment_high = np.max(highs[idx - length : idx + length + 1])
            segment_low = np.min(lows[idx - length : idx + length + 1])
            if highs[idx] >= segment_high:
                events.append(("H", idx, float(highs[idx])))
            if lows[idx] <= segment_low:
                events.append(("L", idx, float(lows[idx])))
        events.sort(key=lambda item: item[1])
        return events


class VolumeProfileBot:
    def __init__(self, interval: int = 60) -> None:
        self.interval = interval
        self.watchlist = load_watchlist()
        self.state = load_state()
        self.client = ccxt.binanceusdm({
            "enableRateLimit": True,
            "options": {"defaultType": "swap"}
        })
        self.notifier = self._init_notifier()
        self.indicator = PivotVolumeProfile(IndicatorConfig())
        self.trade_settings = TradeSettings()
        self.stats = SignalStats("Volume Profile Bot", STATS_FILE)
        self.exchange = EXCHANGE_NAME
        self.rate_limiter = (
            RateLimitHandler(base_delay=0.5, max_retries=5)
            if RateLimitHandler else None
        )
        self.health_monitor = (
            HealthMonitor("Volume Profile Bot", self.notifier, heartbeat_interval=3600)
            if 'HealthMonitor' in globals() and HealthMonitor is not None
            else None
        )

    def _init_notifier(self) -> Optional[TelegramNotifier]:
        if TelegramNotifier is None:
            logger.warning("Telegram notifier unavailable")
            return None
        token = (
            os.getenv("TELEGRAM_BOT_TOKEN_VOLUME_PROFILE")
            or os.getenv("TELEGRAM_BOT_TOKEN_VOLUME")
            or os.getenv("TELEGRAM_BOT_TOKEN")
        )
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            logger.warning("Telegram credentials missing; alerts will be logged only")
            return None
        return TelegramNotifier(
            bot_token=token,
            chat_id=chat_id,
            signals_log_file=str(LOG_DIR / "volume_profile_signals.json"),
        )

    def _record_alert(self, symbol: str, event_name: str) -> None:
        timestamp = utcnow().isoformat()
        self.state.setdefault("last_alerts", {}).setdefault(symbol, {})[event_name] = timestamp

    def run_loop(self) -> None:
        logger.info("Starting Volume Profile bot for %d symbols", len(self.watchlist))
        monitor = self.health_monitor
        if monitor:
            monitor.send_startup_message()
        while not shutdown_requested:
            start_time = time.time()
            self.run_cycle()
            elapsed = time.time() - start_time
            sleep_for = max(0, self.interval - elapsed)
            if shutdown_requested:
                break
            if sleep_for:
                logger.info("Cycle complete; sleeping %.1fs", sleep_for)
                # Sleep in 1-second chunks to respond quickly to shutdown signals
                for _ in range(int(sleep_for)):
                    if shutdown_requested:
                        break
                    time.sleep(1)
                # Sleep remaining fractional seconds
                if not shutdown_requested and sleep_for % 1 > 0:
                    time.sleep(sleep_for % 1)
        logger.info("Shutdown requested, exiting main loop")
        if monitor:
            monitor.send_shutdown_message()

    def run_cycle(self) -> None:
        monitor = self.health_monitor
        for entry in self.watchlist:
            symbol = str(entry.get("symbol"))
            timeframe = str(entry.get("period", "5m"))
            cooldown_val = entry.get("cooldown_minutes", 5)
            cooldown_minutes = int(cooldown_val) if isinstance(cooldown_val, (int, float, str)) else 5
            try:
                candles = self.fetch_ohlcv(symbol, timeframe)
                if not candles:
                    continue
                closes = np.array([c[4] for c in candles], dtype=float)
                if closes.size == 0:
                    continue
                curr_close = float(closes[-1])
                levels = self.indicator.compute(
                    highs=np.array([c[2] for c in candles], dtype=float),
                    lows=np.array([c[3] for c in candles], dtype=float),
                    closes=closes,
                    volumes=np.array([c[5] for c in candles], dtype=float),
                )
                if levels and closes.size >= 2:
                    prev_close = float(closes[-2])
                    self.evaluate_events(
                        symbol,
                        timeframe,
                        prev_close,
                        curr_close,
                        levels,
                        cooldown_minutes,
                    )
                self._check_open_positions(symbol, curr_close)
            except Exception as exc:
                logger.error("Failed to process %s: %s", symbol, exc, exc_info=True)
                if monitor:
                    monitor.record_error(str(exc))

        if monitor:
            monitor.record_cycle()

        save_state(self.state)

    def fetch_ohlcv(self, symbol: str, timeframe: str) -> Sequence[Sequence[float]]:
        market = f"{symbol.replace("/USDT", "")}/USDT:USDT"
        if self.rate_limiter:
            candles = self.rate_limiter.execute(
                self.client.fetch_ohlcv,
                market,
                timeframe=timeframe,
                limit=500
            )
        else:
            candles = self.client.fetch_ohlcv(market, timeframe=timeframe, limit=500)
        return cast(Sequence[Sequence[float]], candles)

    def evaluate_events(
        self,
        symbol: str,
        timeframe: str,
        prev_close: float,
        curr_close: float,
        levels: Dict[str, float],
        cooldown_minutes: int,
    ) -> None:
        events = [
            ("poc", levels["poc"]),
            ("vah", levels["vah"]),
            ("val", levels["val"]),
        ]
        for key, level in events:
            direction = self._cross_direction(prev_close, curr_close, level)
            if not direction:
                continue
            event_name = f"{key}_{direction}"
            if not self._can_alert(symbol, event_name, cooldown_minutes):
                continue
            outcome = self._maybe_open_trade(
                symbol=symbol,
                timeframe=timeframe,
                event_name=event_name,
                trigger_level=level,
                prev_close=prev_close,
                curr_close=curr_close,
                levels=levels,
            )
            if outcome:
                continue
            if outcome is None:
                continue
            message = (
                f"{symbol} ({timeframe}) price {direction}crossed {key.upper()} level at {level:.4f}.\n"
                f"Close: {curr_close:.4f}, Previous: {prev_close:.4f}"
            )
            self._dispatch(symbol, event_name, message)

        if levels.get("high_volume"):
            if self._can_alert(symbol, "high_volume", cooldown_minutes):
                message = (
                    f"{symbol} ({timeframe}) printed a high-volume bar (>"
                    f"{self.indicator.config.high_volume_mult}× SMA).\nClose: {curr_close:.4f}"
                )
                self._dispatch(symbol, "high_volume", message)

    def _cross_direction(self, prev_close: float, curr_close: float, level: float) -> Optional[str]:
        if prev_close < level <= curr_close:
            return "up"
        if prev_close > level >= curr_close:
            return "down"
        return None

    def _can_alert(self, symbol: str, event_name: str, cooldown_minutes: int) -> bool:
        last_alerts = self.state.setdefault("last_alerts", {}).setdefault(symbol, {})
        last_time_str = last_alerts.get(event_name)
        if last_time_str:
            last_time = datetime.fromisoformat(last_time_str.replace("Z", "+00:00"))
            if (utcnow() - last_time).total_seconds() < cooldown_minutes * 60:
                return False
        return True

    def _dispatch(self, symbol: str, event_name: str, message: str) -> None:
        self._record_alert(symbol, event_name)
        logger.info("Alert %s: %s", symbol, message)

        # Enhanced message format
        formatted_msg = f"📊 <b>Volume Profile Bot</b>\n\n{message}\n\n⏱️ {datetime.now(timezone.utc).isoformat()}"
        self._send_message(formatted_msg)

    def _send_message(self, message: str) -> None:
        if self.notifier:
            try:
                self.notifier.send_message(message)
            except Exception as exc:
                logger.error("Failed to send Telegram alert: %s", exc)
        else:
            logger.info("Notification (no Telegram configured): %s", message)

    def _open_positions(self) -> Dict[str, Dict[str, Any]]:
        return cast(Dict[str, Dict[str, Any]], self.state.setdefault("open_positions", {}))

    def _symbol_open_count(self, symbol: str) -> int:
        positions = self._open_positions()
        return sum(1 for trade in positions.values() if trade.get("symbol") == symbol)

    def _maybe_open_trade(
        self,
        symbol: str,
        timeframe: str,
        event_name: str,
        trigger_level: float,
        prev_close: float,
        curr_close: float,
        levels: Dict[str, float],
    ) -> Optional[bool]:
        direction = "LONG" if event_name.endswith("_up") else "SHORT"
        if self._symbol_open_count(symbol) >= self.trade_settings.max_positions_per_symbol:
            return None

        va_range = abs(levels["vah"] - levels["val"])
        if va_range <= 0:
            return False

        # Use TPSLCalculator with STRUCTURE method (VAL/VAH as swing levels)
        calculator = TPSLCalculator(min_risk_reward=0.8, min_risk_reward_tp2=1.5)
        calc_levels = calculator.calculate(
            entry=curr_close,
            direction=direction,
            method=CalculationMethod.STRUCTURE,
            swing_high=levels["vah"],
            swing_low=levels["val"],
            tp1_multiplier=self.trade_settings.tp1_rr,
            tp2_multiplier=self.trade_settings.tp2_rr,
        )

        if not calc_levels.is_valid:
            logger.debug("%s: TPSLCalculator rejected - %s", symbol, calc_levels.rejection_reason)
            return False

        sl = calc_levels.stop_loss
        tp1 = calc_levels.take_profit_1
        tp2 = calc_levels.take_profit_2

        signal_id = f"VP-{symbol}-{int(time.time())}-{uuid4().hex[:4]}"
        created_at = utcnow().isoformat()
        positions = self._open_positions()
        positions[signal_id] = {
            "symbol": symbol,
            "pair": normalize_symbol(symbol),
            "direction": direction,
            "entry": curr_close,
            "sl": sl,
            "tp_targets": [tp1, tp2],
            "timeframe": timeframe,
            "event": event_name,
            "trigger_level": trigger_level,
            "created_at": created_at,
        }

        extra = {
            "display_symbol": normalize_symbol(symbol),
            "timeframe": timeframe,
            "exchange": self.exchange,
            "event": event_name,
            "sl": sl,
            "tp_targets": [tp1, tp2],
        }
        try:
            self.stats.record_open(
                signal_id=signal_id,
                symbol=normalize_symbol(symbol),
                direction=direction,
                entry=curr_close,
                created_at=created_at,
                extra=extra,
            )
        except Exception as exc:
            logger.error("Failed to record open trade %s: %s", signal_id, exc)

        self._record_alert(symbol, event_name)
        save_state(self.state)

        entry_message = self._build_entry_message(
            signal_id=signal_id,
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            event_name=event_name,
            entry=curr_close,
            sl=sl,
            tp_targets=[tp1, tp2],
            prev_close=prev_close,
            trigger_level=trigger_level,
        )
        self._send_message(entry_message)
        return True

    def _build_entry_message(
        self,
        signal_id: str,
        symbol: str,
        timeframe: str,
        direction: str,
        event_name: str,
        entry: float,
        sl: float,
        tp_targets: List[float],
        prev_close: float,
        trigger_level: float,
    ) -> str:
        """Build entry message using centralized template."""
        # Convert LONG/SHORT to BULLISH/BEARISH
        msg_direction = "BULLISH" if direction == "LONG" else "BEARISH"

        # Get historical stats
        symbol_key = normalize_symbol(symbol)
        counts = self.stats.symbol_tp_sl_counts(symbol_key)
        tp1_count = counts.get("TP1", 0)
        tp2_count = counts.get("TP2", 0)
        sl_count = counts.get("SL", 0)
        total = tp1_count + tp2_count + sl_count

        perf_stats = None
        if total > 0:
            perf_stats = {
                "tp1": tp1_count,
                "tp2": tp2_count,
                "sl": sl_count,
                "wins": tp1_count + tp2_count,
                "total": total,
            }

        # Build extra info
        extra_info = f"Trigger: {event_name.upper()} at {trigger_level:.4f}"

        return format_signal_message(
            bot_name="VOLUME PROFILE",
            symbol=symbol_key,
            direction=msg_direction,
            entry=entry,
            stop_loss=sl,
            tp1=tp_targets[0],
            tp2=tp_targets[1],
            exchange=self.exchange.upper(),
            timeframe=timeframe,
            current_price=entry,
            signal_id=signal_id,
            performance_stats=perf_stats,
            extra_info=extra_info,
        )

    def _check_open_positions(self, symbol: str, current_price: float) -> None:
        positions = self._open_positions()
        for signal_id, trade in list(positions.items()):
            if trade.get("symbol") != symbol:
                continue

            direction = trade.get("direction", "LONG")
            sl = float(trade.get("sl", trade.get("entry", 0.0)))
            tp_targets = trade.get("tp_targets", [])
            if len(tp_targets) < 2:
                tp_targets = [trade.get("entry", current_price), trade.get("entry", current_price)]
            tp1, tp2 = float(tp_targets[0]), float(tp_targets[1])

            if direction == "LONG":
                if current_price <= sl:
                    self._close_trade(signal_id, current_price, "SL")
                    continue
                if current_price >= tp2:
                    self._close_trade(signal_id, current_price, "TP2")
                    continue
                if current_price >= tp1:
                    self._close_trade(signal_id, current_price, "TP1")
            else:
                if current_price >= sl:
                    self._close_trade(signal_id, current_price, "SL")
                    continue
                if current_price <= tp2:
                    self._close_trade(signal_id, current_price, "TP2")
                    continue
                if current_price <= tp1:
                    self._close_trade(signal_id, current_price, "TP1")

    def _close_trade(self, signal_id: str, exit_price: float, result: str) -> None:
        positions = self._open_positions()
        trade = positions.pop(signal_id, None)
        if not trade:
            return

        try:
            record = self.stats.record_close(signal_id, exit_price, result)
        except Exception as exc:
            logger.error("Failed to record close for %s: %s", signal_id, exc)
            record = None

        save_state(self.state)

        if record:
            summary = self.stats.build_summary_message(record)
            self._send_message(summary)
        else:
            fallback = (
                f"{trade.get('symbol')} {trade.get('direction')} closed at {exit_price:.4f}"
                f" with result {result}"
            )
            self._send_message(f"📊 Volume Profile Bot\n{fallback}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Volume Profile bot")
    parser.add_argument("--interval", type=int, default=60, help="Loop interval seconds")
    parser.add_argument("--once", action="store_true", help="Run only one cycle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bot = VolumeProfileBot(interval=args.interval)
    if not args.once:
        bot.run_loop()
    else:
        bot.run_cycle()


if __name__ == "__main__":
    main()
