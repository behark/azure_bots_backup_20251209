#!/usr/bin/env python3
"""
Consensus Bot - Multi-Signal Aggregator (patched)
Monitors multiple trading bots and alerts when multiple bots agree on same signal.

Patched improvements:
- Robust timestamp parsing (ISO, common formats, unix ms/s).
- Safe numeric coercion for entry/sl/tp.
- Rolling seen_signals cache with expiry to avoid memory leak and allow updates.
- Grouping includes timeframe if available to avoid mixing timeframes.
- TP/SL default calculations only used when sensible.
- R:R filtering uses TP1 or TP2 (not only TP2).
- Atomic writes for performance file.
- Clear warnings when notifier/health monitor unavailable.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
CONSENSUS_STATE_FILE = BASE_DIR / "consensus_state.json"
PERFORMANCE_FILE = LOG_DIR / "consensus_performance.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# Add parent directory to path (so local notifier/health_monitor modules can be imported)
sys.path.insert(0, str(BASE_DIR.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "consensus_bot.log"),
    ],
)

logger = logging.getLogger("consensus_bot")

# Load environment if dotenv available (non-fatal)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(BASE_DIR.parent / ".env")
except Exception:
    pass

# Safe imports with independent error handling
from safe_import import safe_import

TelegramNotifier = safe_import('notifier', 'TelegramNotifier', logger_instance=logger)
HealthMonitor = safe_import('health_monitor', 'HealthMonitor', logger_instance=logger)


def safe_float(value: Any, fallback: float = 0.0) -> float:
    """Convert a value to float safely, handling strings and None."""
    try:
        if value is None:
            return fallback
        if isinstance(value, (int, float)):
            return float(value)
        s = str(value).strip()
        if s == "":
            return fallback
        return float(s)
    except Exception:
        return fallback


def parse_timestamp(value: Any) -> datetime:
    """
    Attempts to parse many common timestamp formats:
    - datetime objects (returned unchanged)
    - ISO 8601 strings, with or without 'Z'
    - common formats like "%Y-%m-%d %H:%M:%S"
    - unix timestamps in seconds or milliseconds (int/str)
    Fallback: current UTC time (and logs warning)
    """
    if isinstance(value, datetime):
        # Ensure tz-aware
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    if value is None:
        return datetime.now(timezone.utc)

    s = str(value).strip()

    # Check numeric unix timestamps (ms or s)
    try:
        if s.isdigit():
            num = int(s)
            # Heuristic: > 10**12 => ms
            if num > 10**12:
                return datetime.fromtimestamp(num / 1000.0, tz=timezone.utc)
            # > 10**9 => seconds
            if num > 10**9:
                return datetime.fromtimestamp(num, tz=timezone.utc)
    except Exception:
        pass

    # Try ISO handling 'Z'
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
        if "+" in s or "T" in s:
            return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        pass

    # Try common strptime formats
    common_formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in common_formats:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue

    logger.debug(f"parse_timestamp: unknown format '{value}', using now() as fallback")
    return datetime.now(timezone.utc)


def stable_hash(obj: Any) -> str:
    """Small stable hash for signal content used to detect updates."""
    try:
        dumped = json.dumps(obj, sort_keys=True, default=str)
    except Exception:
        dumped = str(obj)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


class Signal:
    """Represents a trading signal from a bot."""

    def __init__(self, bot_name: str, symbol: str, direction: str,
                 entry: Any, sl: Any, tp1: Any, tp2: Any,
                 timestamp: Any, raw_data: Optional[dict] = None):
        self.bot_name = bot_name
        # normalize symbol to canonical format (strip /USDT etc.)
        self.symbol = str(symbol).replace("/USDT", "").replace(":USDT", "").strip().upper()
        self.direction = self._normalize_direction(direction)
        self.entry = safe_float(entry, 0.0)
        self.sl = safe_float(sl, 0.0)
        self.tp1 = safe_float(tp1, 0.0)
        self.tp2 = safe_float(tp2, 0.0)
        self.timestamp = parse_timestamp(timestamp)
        self.raw_data = raw_data or {}

    def _normalize_direction(self, direction: Any) -> str:
        if direction is None:
            return "UNKNOWN"
        d = str(direction).strip().upper()
        if d in ("BULLISH", "LONG", "BUY"):
            return "LONG"
        if d in ("BEARISH", "SHORT", "SELL"):
            return "SHORT"
        return d

    def age_minutes(self) -> float:
        now = datetime.now(timezone.utc)
        # ensure timezone aware
        ts = self.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (now - ts.astimezone(timezone.utc)).total_seconds() / 60.0

    def to_dict(self) -> dict:
        return {
            "bot_name": self.bot_name,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry": self.entry,
            "sl": self.sl,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "timestamp": self.timestamp.isoformat(),
            "raw_data": self.raw_data,
        }

    def __repr__(self):
        return f"Signal({self.bot_name}, {self.symbol}, {self.direction}, {self.entry})"


class SignalMonitor:
    """Monitors signal files from all bots."""

    def __init__(self, base_dir: Path, seen_retention_minutes: int = 60 * 24):
        self.base_dir = base_dir
        self.signal_files = {
            "liquidation": base_dir / "liquidation_bot" / "liquidation_state.json",
            "funding": base_dir / "funding_bot" / "funding_state.json",
            "volume": base_dir / "volume_bot" / "volume_vn_state.json",
            "harmonic": base_dir / "harmonic_bot" / "harmonic_state.json",
            "candlestick": base_dir / "candlestick_bot" / "candlestick_state.json",
            "mtf": base_dir / "mtf_bot" / "mtf_state.json",
            "psar": base_dir / "psar_bot" / "psar_state.json",
            "diy": base_dir / "diy_bot" / "diy_state.json",
            "strat": base_dir / "strat_bot" / "strat_state.json",
            "fib_reversal": base_dir / "fib_reversal_bot" / "fib_reversal_state.json",
            "fib_swing": base_dir / "fib_swing_bot" / "logs" / "fib_state.json",
            "most": base_dir / "most_bot" / "most_state.json",
        }
        # seen_signals: unique_id -> (hash, first_seen_datetime)
        self.seen_signals: Dict[str, Tuple[str, datetime]] = {}
        self.seen_retention = timedelta(minutes=seen_retention_minutes)

    def _prune_seen(self):
        """Remove seen signals older than retention window."""
        now = datetime.now(timezone.utc)
        expired = [uid for uid, (_, ts) in self.seen_signals.items() if (now - ts) > self.seen_retention]
        for uid in expired:
            del self.seen_signals[uid]

    def get_recent_signals(self, max_age_minutes: int = 30) -> List[Signal]:
        """Get all recent signals from all bots. Allows updated signals to be reprocessed."""
        self._prune_seen()
        signals: List[Signal] = []
        max_age_td = timedelta(minutes=max_age_minutes)
        now = datetime.now(timezone.utc)

        for bot_name, file_path in self.signal_files.items():
            if not file_path.exists():
                continue
            try:
                with open(file_path, "r") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON for {bot_name} at {file_path}: {e}")
                        continue

                open_signals = data.get("open_signals", {}) or {}

                for signal_id, signal_data in open_signals.items():
                    unique_id = f"{bot_name}_{signal_id}"
                    # Normalize fields and coerce numeric types
                    symbol = signal_data.get("symbol", "") or ""
                    direction = signal_data.get("direction", "") or ""
                    entry = signal_data.get("entry", 0)
                    sl = signal_data.get("stop_loss", signal_data.get("sl", 0))
                    tp1 = signal_data.get("take_profit_1") or signal_data.get("tp1") or signal_data.get("take_profit_primary") or 0
                    tp2 = signal_data.get("take_profit_2") or signal_data.get("tp2") or signal_data.get("take_profit_secondary") or 0
                    timestamp = signal_data.get("created_at", signal_data.get("timestamp", datetime.now(timezone.utc).isoformat()))

                    sig = Signal(bot_name, symbol, direction, entry, sl, tp1, tp2, timestamp, signal_data)

                    # Only include if not too old
                    if now - sig.timestamp > max_age_td:
                        continue

                    # Check if we've seen this exact content before
                    content_hash = stable_hash({
                        "symbol": sig.symbol,
                        "direction": sig.direction,
                        "entry": sig.entry,
                        "sl": sig.sl,
                        "tp1": sig.tp1,
                        "tp2": sig.tp2,
                        "timestamp": sig.timestamp.isoformat(),
                    })

                    prev = self.seen_signals.get(unique_id)
                    if prev is not None and prev[0] == content_hash:
                        # exact same signal content already processed, skip
                        continue
                    # Otherwise add/replace seen entry and include signal
                    self.seen_signals[unique_id] = (content_hash, datetime.now(timezone.utc))
                    signals.append(sig)

            except Exception as e:
                logger.warning(f"Error reading signals from {bot_name}: {e}")

        return signals


class ConsensusDetector:
    """Detects consensus signals across multiple bots."""

    BASE_BOT_WEIGHTS: Dict[str, float] = {
        "harmonic": 2.0,
        "liquidation": 1.75,
        "volume": 1.5,
        "strat": 1.5,
        "mtf": 1.5,
        "fib_reversal": 1.25,
        "fib_swing": 1.25,
        "diy": 1.0,
        "candlestick": 1.0,
        "funding": 1.0,
        "most": 0.75,
        "psar": 0.75,
    }
    BOT_WEIGHTS: Dict[str, float] = {
        **BASE_BOT_WEIGHTS,
        **{f"{name}_bot": weight for name, weight in BASE_BOT_WEIGHTS.items()},
    }
    DEFAULT_BOT_WEIGHT: float = 1.0

    def __init__(self, time_window_minutes: int = 30, min_risk_reward: float = 1.7, min_bots: int = 3):
        self.time_window_minutes = time_window_minutes
        self.min_risk_reward = min_risk_reward
        self.min_bots = min_bots
        # alerted_consensus: id -> {"bots": set(...), "first_seen": datetime}
        self.alerted_consensus: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def _bot_weight(cls, bot_name: str) -> float:
        key = bot_name.lower().replace(" ", "_")
        if key in cls.BOT_WEIGHTS:
            return cls.BOT_WEIGHTS[key]
        if key.endswith("_bot"):
            trimmed = key[:-4]
            if trimmed in cls.BOT_WEIGHTS:
                return cls.BOT_WEIGHTS[trimmed]
        alt_key = key.replace("bot", "_bot")
        if alt_key in cls.BOT_WEIGHTS:
            return cls.BOT_WEIGHTS[alt_key]
        return cls.DEFAULT_BOT_WEIGHT

    def _effective_min_rr(self, confidence_level: float) -> float:
        base = self.min_risk_reward
        if confidence_level >= 7.0:
            return min(base, 1.1)
        if confidence_level >= 5.0:
            return min(base, 1.3)
        if confidence_level >= 3.5:
            return min(base, 1.4)
        return base

    @staticmethod
    def _recency_weight(signal: Signal) -> float:
        age = signal.age_minutes()
        if age <= 5:
            return 2.0
        if age >= 15:
            return 1.0
        return 1.5

    def find_consensus(self, signals: List[Signal]) -> List[Dict]:
        consensus_signals: List[Dict] = []
        now = datetime.now(timezone.utc)

        # Group signals by (symbol, direction, timeframe) where timeframe is optional
        groups: Dict[Tuple[str, str, Optional[str]], List[Signal]] = defaultdict(list)
        for s in signals:
            timeframe = None
            # look for common timeframe keys in raw_data
            for key in ("timeframe", "tf", "interval", "granularity"):
                if key in s.raw_data:
                    timeframe = str(s.raw_data.get(key))
                    break
            groups[(s.symbol, s.direction, timeframe)].append(s)

        for (symbol, direction, timeframe), group in groups.items():
            if len(group) < 2:
                continue

            # filter by recent within detector time window
            valid_group = [s for s in group if s.age_minutes() <= self.time_window_minutes]
            if len(valid_group) < 2:
                continue

            # unique bots
            unique_bots = sorted({s.bot_name for s in valid_group})
            current_bots = set(unique_bots)
            weighted_score = sum(self._bot_weight(bot) for bot in current_bots)

            if weighted_score < self.min_bots:
                continue

            base_consensus_id = f"{symbol}_{direction}" + (f"_{timeframe}" if timeframe else "")

            prev = self.alerted_consensus.get(base_consensus_id, {"bots": set(), "first_seen": now})
            prev_bots = set(prev.get("bots", set()))
            new_bots = current_bots - prev_bots

            # If nothing new and we've already alerted, skip
            if not new_bots and prev_bots:
                continue

            is_update = bool(prev_bots)
            confidence_level = weighted_score
            bot_count = len(current_bots)
            if confidence_level >= 7.0:
                confidence_label = "LEGENDARY"
            elif confidence_level >= 5.0:
                confidence_label = "EXTREME"
            elif confidence_level >= 3.5:
                confidence_label = "HIGH"
            elif confidence_level >= 2.0:
                confidence_label = "MODERATE"
            else:
                confidence_label = "LOW"

            # compute averages on available values (always use floats)
            entries = [
                (s.entry, self._recency_weight(s))
                for s in valid_group
                if s.entry and s.entry > 0
            ]
            sls = [s.sl for s in valid_group if s.sl and s.sl > 0]

            if not entries or not sls:
                # not enough numeric info to compute risk properly
                logger.info(f"Skipping {symbol} {direction} (missing entry or sl values)")
                continue

            total_entry_weight = sum(weight for _, weight in entries)
            if total_entry_weight <= 0:
                logger.info(f"Skipping {symbol} {direction} due to zero entry weighting")
                continue

            consensus_entry = sum(value * weight for value, weight in entries) / total_entry_weight
            consensus_sl = sum(sls) / len(sls)

            # compute TP averages if present
            tp1_list = [s.tp1 for s in valid_group if s.tp1 and s.tp1 > 0]
            tp2_list = [s.tp2 for s in valid_group if s.tp2 and s.tp2 > 0]

            # If no TP provided, compute defaults only if sl and entry are valid and not equal
            risk = abs(consensus_entry - consensus_sl)
            if not tp1_list:
                if risk > 0:
                    if direction.upper() == "LONG":
                        consensus_tp1 = consensus_entry + risk  # 1:1
                    else:
                        consensus_tp1 = consensus_entry - risk
                    logger.info(f"No valid TP1 from bots for {symbol}, using calculated TP1={consensus_tp1:.8f}")
                else:
                    # cannot compute sensible TP
                    logger.info(f"Skipping {symbol} {direction} due to zero risk (entry==sl)")
                    continue
            else:
                consensus_tp1 = sum(tp1_list) / len(tp1_list)

            if not tp2_list:
                if risk > 0:
                    if direction.upper() == "LONG":
                        consensus_tp2 = consensus_entry + (risk * 2)
                    else:
                        consensus_tp2 = consensus_entry - (risk * 2)
                    logger.info(f"No valid TP2 from bots for {symbol}, using calculated TP2={consensus_tp2:.8f}")
                else:
                    logger.info(f"Skipping {symbol} {direction} due to zero risk (entry==sl)")
                    continue
            else:
                consensus_tp2 = sum(tp2_list) / len(tp2_list)

            # Ensure TP direction correctness
            is_long = direction.upper() == "LONG"
            if is_long:
                if consensus_tp1 <= consensus_entry:
                    consensus_tp1 = consensus_entry + risk
                if consensus_tp2 <= consensus_entry:
                    consensus_tp2 = consensus_entry + (risk * 2)
                # Validate LONG setup
                assert consensus_sl < consensus_entry, f"Consensus LONG: SL {consensus_sl} should be < entry {consensus_entry}"
                assert consensus_tp1 > consensus_entry, f"Consensus LONG: TP1 {consensus_tp1} should be > entry {consensus_entry}"
                assert consensus_tp2 > consensus_entry, f"Consensus LONG: TP2 {consensus_tp2} should be > entry {consensus_entry}"
            else:
                if consensus_tp1 >= consensus_entry:
                    consensus_tp1 = consensus_entry - risk
                if consensus_tp2 >= consensus_entry:
                    consensus_tp2 = consensus_entry - (risk * 2)
                # Validate SHORT setup
                assert consensus_sl > consensus_entry, f"Consensus SHORT: SL {consensus_sl} should be > entry {consensus_entry}"
                assert consensus_tp1 < consensus_entry, f"Consensus SHORT: TP1 {consensus_tp1} should be < entry {consensus_entry}"
                assert consensus_tp2 < consensus_entry, f"Consensus SHORT: TP2 {consensus_tp2} should be < entry {consensus_entry}"
            
            logger.debug("%s %s Consensus | Entry: %.6f | SL: %.6f | TP1: %.6f | TP2: %.6f | Bots: %d",
                       symbol, direction, consensus_entry, consensus_sl, consensus_tp1, consensus_tp2, len(valid_group))

            # Compute R:R for TP1 and TP2 and require at least one to meet min_risk_reward
            reward1 = abs(consensus_tp1 - consensus_entry)
            reward2 = abs(consensus_tp2 - consensus_entry)
            rr1 = (reward1 / risk) if risk > 0 else 0.0
            rr2 = (reward2 / risk) if risk > 0 else 0.0

            effective_min_rr = self._effective_min_rr(confidence_level)
            if risk <= 0 or (rr1 < effective_min_rr and rr2 < effective_min_rr):
                logger.info(f"Skipping {symbol} {direction} - R:R too low (1:{rr1:.2f}, 1:{rr2:.2f})")
                continue

            consensus_signals.append({
                "id": base_consensus_id,
                "symbol": symbol,
                "direction": direction,
                "timeframe": timeframe,
                "confidence_level": confidence_level,
                "bot_count": bot_count,
                "confidence_label": confidence_label,
                "bots": unique_bots,
                "new_bots": sorted(list(new_bots)),
                "is_update": is_update,
                "signals": valid_group,
                "consensus_entry": consensus_entry,
                "consensus_sl": consensus_sl,
                "consensus_tp1": consensus_tp1,
                "consensus_tp2": consensus_tp2,
                "timestamp": now.isoformat(),
            })

            # Update tracking with current bots and timestamp
            self.alerted_consensus[base_consensus_id] = {"bots": current_bots, "first_seen": now}

        return consensus_signals

    def cleanup_old_alerts(self, max_age_hours: int = 2):
        """Remove alerts older than max_age_hours to prevent memory growth."""
        now = datetime.now(timezone.utc)
        expiry = timedelta(hours=max_age_hours)
        to_delete = []
        for cid, info in self.alerted_consensus.items():
            first = info.get("first_seen", now)
            if (now - first) > expiry:
                to_delete.append(cid)
        for cid in to_delete:
            del self.alerted_consensus[cid]


class ConsensusPerformanceTracker:
    """Track performance of consensus signals with atomic writes."""

    def __init__(self, performance_file: Path = PERFORMANCE_FILE):
        self.performance_file = performance_file
        self.data = self._load()

    def _load(self) -> Dict:
        if self.performance_file.exists():
            try:
                with open(self.performance_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load performance file: {e}")
        return {"consensus_signals": [], "stats": {}}

    def _atomic_save(self):
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(prefix="perf_", dir=str(self.performance_file.parent))
            with os.fdopen(tmp_fd, "w") as tmpf:
                json.dump(self.data, tmpf, indent=2, default=str)
                tmpf.flush()
                os.fsync(tmpf.fileno())
            os.replace(tmp_path, str(self.performance_file))
        except Exception as e:
            logger.error(f"Failed to save performance data atomically: {e}")

    def record_consensus(self, consensus: Dict):
        record = {
            "id": consensus["id"],
            "symbol": consensus["symbol"],
            "direction": consensus["direction"],
            "timeframe": consensus.get("timeframe"),
            "confidence_level": consensus["confidence_level"],
            "bots": consensus["bots"],
            "entry": consensus["consensus_entry"],
            "sl": consensus["consensus_sl"],
            "tp1": consensus["consensus_tp1"],
            "tp2": consensus["consensus_tp2"],
            "timestamp": consensus["timestamp"],
            "status": "open"
        }
        self.data.setdefault("consensus_signals", []).append(record)
        self._atomic_save()
        logger.info(f"Recorded consensus signal: {record['id']}")

    def record_outcome(self, signal_id: str, exit_price: float, result: str, closed_at: Optional[str] = None):
        """Update a consensus signal with outcome and basic PnL stats."""
        signals = self.data.setdefault("consensus_signals", [])
        for rec in signals:
            if rec.get("id") != signal_id or rec.get("status") != "open":
                continue
            entry = float(rec.get("entry", 0.0) or 0.0)
            if entry == 0:
                pnl_pct = 0.0
            else:
                direction = str(rec.get("direction", "LONG")).upper()
                if direction == "SHORT":
                    pnl_pct = (entry - exit_price) / entry * 100.0
                else:
                    pnl_pct = (exit_price - entry) / entry * 100.0
            rec["status"] = "closed"
            rec["result"] = result
            rec["exit"] = exit_price
            rec["pnl_pct"] = pnl_pct
            rec["closed_at"] = closed_at or datetime.now(timezone.utc).isoformat()
            break
        self._recompute_stats()
        self._atomic_save()

    def _recompute_stats(self):
        """Rebuild aggregate stats from closed consensus signals."""
        signals = self.data.get("consensus_signals", []) or []
        closed = [s for s in signals if s.get("status") == "closed"]
        total = len(closed)
        if total == 0:
            self.data["stats"] = {}
            return
        tp_hits = sum(1 for s in closed if str(s.get("result", "")).startswith("TP"))
        sl_hits = sum(1 for s in closed if s.get("result") == "SL")
        total_pnl = sum(float(s.get("pnl_pct", 0.0) or 0.0) for s in closed)
        win_rate = (tp_hits / total) * 100.0
        self.data["stats"] = {
            "closed_signals": total,
            "tp_hits": tp_hits,
            "sl_hits": sl_hits,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            # Per (symbol, direction) stats for quick lookup
            "by_symbol_direction": self._build_symbol_direction_stats(closed),
        }

    @staticmethod
    def _build_symbol_direction_stats(closed: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Build per (symbol, direction) stats from closed consensus signals."""
        stats: Dict[str, Dict[str, float]] = {}
        for rec in closed:
            symbol = str(rec.get("symbol", "")).upper()
            direction = str(rec.get("direction", "")).upper()
            if not symbol or not direction:
                continue
            key = f"{symbol}:{direction}"
            bucket = stats.setdefault(key, {
                "count": 0.0,
                "tp_hits": 0.0,
                "sl_hits": 0.0,
                "total_pnl": 0.0,
            })
            bucket["count"] += 1.0
            result = str(rec.get("result", ""))
            if result.startswith("TP"):
                bucket["tp_hits"] += 1.0
            elif result == "SL":
                bucket["sl_hits"] += 1.0
            bucket["total_pnl"] += float(rec.get("pnl_pct", 0.0) or 0.0)
        # compute win_rate and avg_pnl for each key
        for key, bucket in stats.items():
            count = bucket["count"] or 1.0
            bucket["win_rate"] = (bucket["tp_hits"] / count) * 100.0
            bucket["avg_pnl"] = bucket["total_pnl"] / count
        return stats

    def symbol_tp_sl_counts(self, symbol: str, direction: str, timeframe: Optional[str] = None) -> Optional[Dict[str, int]]:
        """Return TP1/TP2/SL counts for a given symbol/direction/(optional) timeframe."""
        signals = self.data.get("consensus_signals", []) or []
        sym = str(symbol).upper()
        dir_norm = str(direction).upper()
        tf_norm = str(timeframe) if timeframe else None
        tp1 = tp2 = sl = 0
        found = False
        for rec in signals:
            if rec.get("status") != "closed":
                continue
            if str(rec.get("symbol", "")).upper() != sym:
                continue
            if str(rec.get("direction", "")).upper() != dir_norm:
                continue
            rec_tf = rec.get("timeframe")
            if tf_norm is not None and rec_tf != tf_norm:
                continue
            result = str(rec.get("result", ""))
            found = True
            if result == "TP1":
                tp1 += 1
            elif result == "TP2":
                tp2 += 1
            elif result == "SL":
                sl += 1
        if not found:
            return None
        return {"tp1": tp1, "tp2": tp2, "sl": sl}


class ConsensusBot:
    """Main consensus bot class."""

    def __init__(self, check_interval: int = 30, time_window: int = 30,
                 min_risk_reward: float = 1.7, min_bots: int = 3):
        self.check_interval = check_interval
        self.time_window = time_window
        self.min_risk_reward = min_risk_reward
        self.min_bots = min_bots

        self.monitor = SignalMonitor(BASE_DIR.parent)
        self.detector = ConsensusDetector(time_window_minutes=time_window, min_risk_reward=min_risk_reward, min_bots=min_bots)
        self.notifier = self._init_notifier()
        self.health_monitor = HealthMonitor("Consensus Bot", self.notifier, heartbeat_interval=3600) if HealthMonitor else None
        self.performance_tracker = ConsensusPerformanceTracker()

    def _init_notifier(self):
        if TelegramNotifier is None:
            logger.warning("Telegram notifier unavailable - alerts will only be logged")
            return None
        token = os.getenv("TELEGRAM_BOT_TOKEN_CONSENSUS")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            logger.warning("Telegram credentials missing for consensus bot - alerts will only be logged")
            return None
        try:
            return TelegramNotifier(
                bot_token=token,
                chat_id=chat_id,
                signals_log_file=str(LOG_DIR / "consensus_signals.json")
            )
        except Exception as e:
            logger.warning(f"Failed to initialize TelegramNotifier: {e}")
            return None

    def run(self, loop: bool = True):
        logger.info("Starting Consensus Bot")
        logger.info(f"Check interval: {self.check_interval}s - Time window: {self.time_window}min - Min R:R: {self.min_risk_reward} - Min bots: {self.min_bots}")

        if self.health_monitor:
            try:
                self.health_monitor.send_startup_message()
            except Exception as e:
                logger.warning(f"Health monitor startup failed: {e}")

        try:
            while True:
                try:
                    self._check_consensus()
                    if self.health_monitor:
                        try:
                            self.health_monitor.record_cycle()
                        except Exception as e:
                            logger.debug(f"Health monitor record_cycle failed: {e}")
                    if not loop:
                        break
                    # Sleep in 1-second chunks to respond quickly to shutdown signals
                    for _ in range(self.check_interval):
                        time.sleep(1)
                except Exception as exc:
                    logger.error(f"Error in consensus check: {exc}")
                    if self.health_monitor:
                        try:
                            self.health_monitor.record_error(str(exc))
                        except Exception:
                            pass
                    time.sleep(10)
        finally:
            if self.health_monitor:
                try:
                    self.health_monitor.send_shutdown_message()
                except Exception:
                    pass

    def _check_consensus(self):
        signals = self.monitor.get_recent_signals(max_age_minutes=self.time_window)
        if not signals:
            logger.debug("No recent signals found")
            return

        logger.debug(f"Found {len(signals)} recent signals")
        consensus_signals = self.detector.find_consensus(signals)

        if consensus_signals:
            logger.info(f"Found {len(consensus_signals)} consensus signals")
            for consensus in consensus_signals:
                try:
                    self._send_consensus_alert(consensus)
                except Exception as e:
                    logger.error(f"Failed to send consensus alert: {e}")
                try:
                    self.performance_tracker.record_consensus(consensus)
                except Exception as e:
                    logger.error(f"Failed to record consensus: {e}")

        # Cleanup
        self.detector.cleanup_old_alerts()

        # Update outcomes for any open consensus signals using latest prices
        try:
            self._update_performance_with_market_prices()
        except Exception as e:
            logger.error(f"Failed to update consensus performance: {e}")

    def _update_performance_with_market_prices(self):
        """Fetch current prices and close consensus signals that hit TP1/TP2/SL."""
        try:
            import ccxt  # type: ignore
        except Exception as e:
            logger.debug(f"ccxt not available for consensus tracking: {e}")
            return

        signals = self.performance_tracker.data.get("consensus_signals", []) or []
        open_signals = [s for s in signals if s.get("status") == "open"]
        if not open_signals:
            return

        # Use MEXC USDT‑M futures by default for pricing
        try:
            client = ccxt.mexc({"enableRateLimit": True, "options": {"defaultType": "swap"}})
        except Exception as e:
            logger.error(f"Failed to init ccxt client for consensus tracking: {e}")
            return

        for rec in open_signals:
            symbol = str(rec.get("symbol", "")).upper()
            if not symbol:
                continue
            market_symbol = f"{symbol}/USDT:USDT"
            try:
                ticker = client.fetch_ticker(market_symbol)
            except Exception as e:
                logger.debug(f"Ticker fetch failed for {market_symbol}: {e}")
                continue
            last = ticker.get("last") if isinstance(ticker, dict) else None
            if last is None and isinstance(ticker, dict):
                last = ticker.get("close")
            if not isinstance(last, (int, float)):
                continue

            direction = str(rec.get("direction", "LONG")).upper()
            entry = float(rec.get("entry", 0.0) or 0.0)
            sl = float(rec.get("sl", 0.0) or 0.0)
            tp1 = float(rec.get("tp1", 0.0) or 0.0)
            tp2 = float(rec.get("tp2", 0.0) or 0.0)
            if not all([entry, sl, tp1, tp2]):
                continue

            if direction == "SHORT":
                hit_tp2 = last <= tp2
                hit_tp1 = last <= tp1
                hit_sl = last >= sl
            else:
                hit_tp2 = last >= tp2
                hit_tp1 = last >= tp1
                hit_sl = last <= sl

            result: Optional[str] = None
            if hit_tp2:
                result = "TP2"
            elif hit_tp1:
                result = "TP1"
            elif hit_sl:
                result = "SL"

            if result:
                exit_price = float(last)
                self.performance_tracker.record_outcome(rec["id"], exit_price, result)
                pnl_pct = 0.0
                if entry:
                    if direction == "SHORT":
                        pnl_pct = (entry - exit_price) / entry * 100.0
                    else:
                        pnl_pct = (exit_price - entry) / entry * 100.0
                stats = self.performance_tracker.data.get("stats", {}) if self.performance_tracker else {}
                win_rate = stats.get("win_rate") if isinstance(stats, dict) else None
                message = self.build_summary_message(
                    symbol=symbol,
                    direction=direction,
                    entry=entry,
                    exit_price=exit_price,
                    result=result,
                    pnl_pct=pnl_pct,
                    total_win_rate=win_rate,
                )
                if self.notifier and message:
                    try:
                        self.notifier.send_message(message)
                    except Exception as e:
                        logger.error(f"Failed to send consensus close alert: {e}")
                logger.info(f"Consensus {rec['id']} closed with {result} at {last}")

    def build_summary_message(
        self,
        *,
        symbol: str,
        direction: str,
        entry: float,
        exit_price: float,
        result: str,
        pnl_pct: float,
        total_win_rate: Optional[float] = None,
    ) -> str:
        """Build a rich summary message for closed consensus trades."""
        header = "🎯 CONSENSUS TRADE CLOSED" if result.startswith("TP") else "⛔ CONSENSUS STOPPED"
        body = [header, ""]
        body.append(f"Symbol: <b>{symbol}/USDT</b>")
        body.append(f"Direction: <b>{direction}</b>")
        body.append(f"Entry: <code>{entry:.6f}</code>")
        body.append(f"Exit: <code>{exit_price:.6f}</code>")
        body.append(f"Result: <b>{result}</b>")
        body.append(f"P&L: <b>{pnl_pct:+.2f}%</b>")
        if total_win_rate is not None:
            body.append(f"Total Consensus Win Rate: <b>{total_win_rate:.1f}%</b>")
        return "\n".join(body)

    def _send_consensus_alert(self, consensus: Dict):
        symbol = consensus["symbol"]
        direction = consensus["direction"]
        confidence = float(consensus["confidence_level"])
        bot_count = int(consensus.get("bot_count", len(consensus.get("bots", []))))
        is_update = consensus.get("is_update", False)
        new_bots = consensus.get("new_bots", [])
        timeframe = consensus.get("timeframe")

        total_bots_tracked = len(self.monitor.signal_files)

        # Derive a weighted historical quality score for this symbol/direction
        quality_text = "N/A"
        quality_bucket = None
        stats_root = self.performance_tracker.data.get("stats", {}) if self.performance_tracker else {}
        by_sd = stats_root.get("by_symbol_direction") if isinstance(stats_root, dict) else None
        if isinstance(by_sd, dict):
            key = f"{symbol.upper()}:{direction.upper()}"
            sd = by_sd.get(key)
            if isinstance(sd, dict) and sd.get("count", 0) >= 5:
                win_rate = float(sd.get("win_rate", 0.0) or 0.0)
                avg_pnl = float(sd.get("avg_pnl", 0.0) or 0.0)
                quality_bucket = (win_rate, avg_pnl)
                # Simple textual quality label
                if win_rate >= 65 and avg_pnl >= 0:
                    quality_text = f"STRONG ({win_rate:.1f}% win, avg {avg_pnl:+.2f}%)"
                elif win_rate >= 55 and avg_pnl >= -0.1:
                    quality_text = f"DECENT ({win_rate:.1f}% win, avg {avg_pnl:+.2f}%)"
                else:
                    quality_text = f"WEAK ({win_rate:.1f}% win, avg {avg_pnl:+.2f}%)"

        label = (consensus.get("confidence_label") or "").upper()
        if label == "LEGENDARY":
            confidence_emoji = "🔥🔥🔥🔥🔥"
            confidence_text = "LEGENDARY CONSENSUS"
            position_multiplier = "4-5x"
        elif label == "EXTREME":
            confidence_emoji = "🔥🔥🔥🔥"
            confidence_text = "EXTREME CONSENSUS"
            position_multiplier = "3-4x"
        elif label == "HIGH":
            confidence_emoji = "🔥🔥🔥"
            confidence_text = "HIGH CONSENSUS"
            position_multiplier = "2-3x"
        elif label == "MODERATE":
            confidence_emoji = "🔥🔥"
            confidence_text = "MODERATE CONSENSUS"
            position_multiplier = "1.5-2x"
        else:
            confidence_emoji = "⚡"
            confidence_text = "LOW CONFIDENCE"
            position_multiplier = "0.5-1x"

        # Build message with historical stats first (like funding bot)
        if is_update:
            message = "🔄 <b>CONSENSUS UPDATE</b> 🔄\n"
            message += f"➕ New bot(s) joined: <b>{', '.join(new_bots)}</b>\n\n"
        else:
            emoji = "🟢" if direction.upper() in ("BULLISH", "LONG") else "🔴"
            message = f"{emoji} <b>{direction.upper()} CONSENSUS SIGNAL - {symbol}/USDT</b>\n\n"

        # Historical TP/SL stats at the top (like funding bot format)
        if self.performance_tracker:
            counts = self.performance_tracker.symbol_tp_sl_counts(symbol, direction, timeframe)
            if counts and (counts['tp1'] + counts['tp2'] + counts['sl']) > 0:
                total = counts['tp1'] + counts['tp2'] + counts['sl']
                win_rate = ((counts['tp1'] + counts['tp2']) / total) * 100.0
                message += (
                    f"📈 <b>History:</b> TP1 {counts['tp1']} | TP2 {counts['tp2']} | SL {counts['sl']} "
                    f"(Win rate: {win_rate:.1f}%)\n\n"
                )

        tf_text = f" ({timeframe})" if timeframe else ""
        message += f"💰 Symbol: <b>{symbol}/USDT{tf_text}</b>\n"
        message += f"📍 Direction: <b>{direction}</b>\n"
        message += f"⚡ Confidence: <b>{confidence_text}</b>\n"
        if quality_bucket is not None:
            message += f"📊 Quality: <b>{quality_text}</b>\n"
        star_count = max(1, min(bot_count, 5)) if bot_count > 0 else 1
        message += (
            f"🤖 Bots Agree: <b>{bot_count}/{total_bots_tracked}</b> (weight {confidence:.1f}) "
            f"{'⭐' * star_count}\n\n"
        )

        message += "<b>📊 SUPPORTING SIGNALS:</b>\n"
        for signal in consensus["signals"]:
            age = int(signal.age_minutes())
            new_marker = "➕ " if signal.bot_name in new_bots else "✅ "
            message += f"{new_marker}<b>{signal.bot_name.title()} Bot</b> ({age}min ago)\n"
            try:
                message += f"   Entry: {signal.entry:.8f}  SL: {signal.sl:.8f}\n"
            except Exception:
                message += f"   Entry: {signal.entry}  SL: {signal.sl}\n"

        message += "\n<b>🎯 CONSENSUS LEVELS:</b>\n"
        message += f"💰 Entry: <code>{consensus['consensus_entry']:.8f}</code>\n"
        message += f"🛑 Stop Loss: <code>{consensus['consensus_sl']:.8f}</code>\n"
        message += f"🎯 Targets: TP1 <code>{consensus['consensus_tp1']:.8f}</code> | TP2 <code>{consensus['consensus_tp2']:.8f}</code>\n"
        message += f"🛑 Stop: <code>{consensus['consensus_sl']:.8f}</code>\n\n"

        # Calculate R:R
        risk = abs(consensus['consensus_entry'] - consensus['consensus_sl'])
        reward1 = abs(consensus['consensus_tp1'] - consensus['consensus_entry'])
        reward2 = abs(consensus['consensus_tp2'] - consensus['consensus_entry'])
        if risk > 0:
            rr1 = reward1 / risk
            rr2 = reward2 / risk
            message += f"⚖️ Risk/Reward: 1:{rr1:.2f} (TP1) | 1:{rr2:.2f} (TP2)\n"

        expected_win_rate = 60 + (confidence - 2) * 6
        message += f"📈 Expected Win Rate: {min(expected_win_rate, 98)}-{min(expected_win_rate+8, 99)}%\n"
        message += f"💎 Suggested Position: <b>{position_multiplier}</b> normal size\n\n"

        message += f"⏰ Valid for: {self.time_window} minutes\n"
        message += f"🆔 {consensus['id']}"

        # Send alert
        if self.notifier:
            try:
                if self.notifier.send_message(message):
                    update_text = "UPDATE " if is_update else ""
                    logger.info(
                        f"Consensus {update_text}alert sent: {symbol} {direction} (weight {confidence:.1f}, {bot_count} bots)"
                    )
                else:
                    logger.error("Failed to send consensus alert via notifier")
            except Exception as e:
                logger.error(f"Notifier send failed: {e}")
        else:
            # Fallback: log the message
            logger.info(f"CONSENSUS ALERT (no notifier):\n{message}")


def main():
    parser = argparse.ArgumentParser(description="Consensus Bot - Multi-Signal Aggregator (patched)")
    parser.add_argument("--once", action="store_true", help="Run only one cycle")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    parser.add_argument("--window", type=int, default=30, help="Time window for consensus in minutes")
    parser.add_argument("--min-rr", type=float, default=1.7, help="Minimum R:R ratio to send alert (default: 1.7)")
    parser.add_argument("--min-bots", type=int, default=3, help="Minimum bots agreeing for consensus (default: 3)")
    args = parser.parse_args()

    bot = ConsensusBot(check_interval=args.interval, time_window=args.window, min_risk_reward=args.min_rr, min_bots=args.min_bots)
    bot.run(loop=not args.once)


if __name__ == "__main__":
    main()
