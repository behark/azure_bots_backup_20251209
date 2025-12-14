#!/usr/bin/env python3
"""Sync local bot state files into telemetry logs and refresh the dashboard DB."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

MAX_RECORDS_PER_BOT = 4000
DEFAULT_INTERVAL_SECONDS = 60
# Use environment variable or default to ~/bots_telemetry for portability
import os
DEFAULT_TELEMETRY_ROOT = Path(os.getenv("TELEMETRY_ROOT", str(Path.home() / "bots_telemetry")))


@dataclass(frozen=True)
class BotConfig:
    key: str
    name: str
    state_file: Path
    log_file: Path
    symbol_suffix: str = "/USDT"
    take_profit_keys: Sequence[str] = (
        "take_profit_1",
        "take_profit_primary",
        "take_profit",
        "tp1",
        "tp",
    )


def build_bot_configs(azure_root: Path, telemetry_root: Path) -> List[BotConfig]:
    sync_dir = telemetry_root / "synced_logs"
    sync_dir.mkdir(parents=True, exist_ok=True)
    return [
        BotConfig(
            "candlestick",
            "Candlestick Bot",
            azure_root / "candlestick_bot" / "candlestick_state.json",
            sync_dir / "candlestick_signals.json",
        ),
        BotConfig(
            "diy",
            "DIY Bot",
            azure_root / "diy_bot" / "diy_state.json",
            sync_dir / "diy_signals.json",
        ),
        BotConfig(
            "fib_reversal",
            "Fib Reversal Bot",
            azure_root / "fib_reversal_bot" / "fib_reversal_state.json",
            sync_dir / "fib_reversal_signals.json",
        ),
        BotConfig(
            "fib_swing",
            "Fib Swing Bot",
            azure_root / "fib_swing_bot" / "logs" / "fib_state.json",
            sync_dir / "fib_swing_signals.json",
            take_profit_keys=("tp1", "take_profit_1", "tp", "take_profit"),
        ),
        BotConfig(
            "funding",
            "Funding Bot",
            azure_root / "funding_bot" / "funding_state.json",
            sync_dir / "funding_signals.json",
        ),
        BotConfig(
            "harmonic",
            "Harmonic Bot",
            azure_root / "harmonic_bot" / "harmonic_state.json",
            sync_dir / "harmonic_signals.json",
        ),
        BotConfig(
            "liquidation",
            "Liquidation Bot",
            azure_root / "liquidation_bot" / "liquidation_state.json",
            sync_dir / "liquidation_signals.json",
        ),
        BotConfig(
            "most",
            "MOST Bot",
            azure_root / "most_bot" / "most_state.json",
            sync_dir / "most_signals.json",
        ),
        BotConfig(
            "mtf",
            "MTF Bot",
            azure_root / "mtf_bot" / "mtf_state.json",
            sync_dir / "mtf_signals.json",
        ),
        BotConfig(
            "psar",
            "PSAR Bot",
            azure_root / "psar_bot" / "psar_state.json",
            sync_dir / "psar_signals.json",
        ),
        BotConfig(
            "strat",
            "Strat Bot",
            azure_root / "strat_bot" / "strat_state.json",
            sync_dir / "strat_signals.json",
        ),
        BotConfig(
            "volume",
            "Volume Bot",
            azure_root / "volume_bot" / "volume_vn_state.json",
            sync_dir / "volume_signals.json",
            take_profit_keys=(
                "take_profit_primary",
                "take_profit_1",
                "take_profit",
                "tp1",
                "tp",
            ),
        ),
    ]


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        logging.debug("State file missing: %s", path)
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        logging.warning("Invalid JSON in %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def _load_log_entries(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        logging.warning("Telemetry log %s corrupted (%s); resetting", path, exc)
        backup = path.with_suffix(path.suffix + ".corrupt")
        try:
            path.rename(backup)
        except OSError:
            pass
        return []
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    return []


def _write_log_entries(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(records, indent=2))
    temp_path.replace(path)


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _first_float(payload: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key not in payload:
            continue
        val = _to_float(payload.get(key))
        if val is not None:
            return val
    return None


def _normalize_direction(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    mapping = {
        "BULLISH": "LONG",
        "LONG": "LONG",
        "BUY": "LONG",
        "BEARISH": "SHORT",
        "SHORT": "SHORT",
        "SELL": "SHORT",
    }
    normalized = mapping.get(value.strip().upper())
    return normalized


def _normalize_symbol(value: Any, suffix: str) -> Optional[str]:
    if not isinstance(value, str):
        return None
    token = value.strip().upper()
    if not token:
        return None
    if "/" not in token and suffix:
        token = f"{token}{suffix}"
    token = token.replace("//", "/")
    return token


def _normalize_timestamp(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        candidate = value.strip()
        try:
            dt = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        except ValueError:
            pass
        else:
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_record(bot: BotConfig, signal_id: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    symbol = _normalize_symbol(payload.get("symbol"), bot.symbol_suffix)
    direction = _normalize_direction(payload.get("direction"))
    entry = _to_float(payload.get("entry"))
    stop = _to_float(payload.get("stop_loss") or payload.get("sl"))
    take_profit = _first_float(payload, bot.take_profit_keys)
    timestamp = _normalize_timestamp(payload.get("created_at") or payload.get("timestamp"))

    if not symbol or direction is None or entry is None or stop is None or take_profit is None:
        return None

    confidence = _first_float(
        payload,
        (
            "confidence",
            "confidence_pct",
            "score",
            "confluence_pct",
        ),
    )

    metadata: Dict[str, Any] = {}
    for key in (
        "timeframe",
        "exchange",
        "pattern",
        "strategy",
        "strength",
        "quality",
        "fib_level",
    ):
        if key in payload and payload[key] is not None:
            metadata[key] = payload[key]
    if "rationale" in payload and isinstance(payload["rationale"], list):
        metadata["rationale"] = payload["rationale"]

    record: Dict[str, Any] = {
        "signal_id": str(signal_id),
        "bot": bot.name,
        "timestamp": timestamp,
        "symbol": symbol,
        "pair": symbol,
        "direction": direction,
        "action": direction,
        "entry": entry,
        "entry_price": entry,
        "stop": stop,
        "stop_loss": stop,
        "sl": stop,
        "tp": take_profit,
        "take_profit": take_profit,
    }

    if confidence is not None:
        record["confidence"] = confidence
    if metadata:
        record["metadata"] = metadata

    risk = abs(entry - stop)
    reward = abs(take_profit - entry)
    if risk > 0:
        record["rr_multiple"] = reward / risk

    return record


def sync_bot(bot: BotConfig, max_records: int) -> int:
    state = _load_json(bot.state_file)
    if not state:
        return 0
    signals = state.get("open_signals", {})
    if not isinstance(signals, dict) or not signals:
        return 0

    log_entries = _load_log_entries(bot.log_file)
    existing_ids = {
        str(entry.get("signal_id") or entry.get("id"))
        for entry in log_entries
        if isinstance(entry, dict) and (entry.get("signal_id") or entry.get("id"))
    }

    new_records: List[Dict[str, Any]] = []
    for signal_id, payload in signals.items():
        if not isinstance(payload, dict):
            continue
        record = _build_record(bot, signal_id, payload)
        if record is None:
            continue
        if record["signal_id"] in existing_ids:
            continue
        existing_ids.add(record["signal_id"])
        new_records.append(record)

    if not new_records:
        return 0

    log_entries.extend(new_records)
    log_entries.sort(key=lambda item: item.get("timestamp", ""))
    if len(log_entries) > max_records:
        log_entries = log_entries[-max_records:]

    _write_log_entries(bot.log_file, log_entries)
    logging.info("%s: recorded %d new signal(s)", bot.name, len(new_records))
    return len(new_records)


def sync_all(bot_configs: Sequence[BotConfig], max_records: int) -> int:
    total = 0
    for bot in bot_configs:
        total += sync_bot(bot, max_records)
    if total:
        logging.info("Synced %d new signal(s) across %d bots", total, len(bot_configs))
    else:
        logging.debug("No new signals detected across %d bots", len(bot_configs))
    return total


def run_collector(telemetry_root: Path, limit: Optional[int]) -> None:
    collector = telemetry_root / "telemetry_collector.py"
    if not collector.exists():
        logging.error("telemetry_collector.py not found at %s", collector)
        return
    cmd = ["python3", str(collector), "--skip-sync"]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    logging.info("Running telemetry_collector (%s)", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        logging.error("telemetry_collector failed (exit code %s)", exc.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync bot signals into telemetry logs")
    parser.add_argument(
        "--azure-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to the azure bots project root",
    )
    parser.add_argument(
        "--telemetry-root",
        type=Path,
        default=DEFAULT_TELEMETRY_ROOT,
        help="Path to the bots_telemetry project",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL_SECONDS,
        help="Seconds between sync cycles (ignored with --once)",
    )
    parser.add_argument("--once", action="store_true", help="Run a single sync pass")
    parser.add_argument("--no-collector", action="store_true", help="Skip running telemetry_collector")
    parser.add_argument(
        "--collector-limit",
        type=int,
        default=None,
        help="Limit passed to telemetry_collector (per bot)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=MAX_RECORDS_PER_BOT,
        help="Maximum records to retain per bot log",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    azure_root = args.azure_root.expanduser().resolve()
    telemetry_root = args.telemetry_root.expanduser().resolve()
    interval = max(5, args.interval)
    max_records = max(100, args.max_records)

    if not azure_root.exists():
        raise SystemExit(f"Azure root not found: {azure_root}")
    if not telemetry_root.exists():
        raise SystemExit(f"Telemetry root not found: {telemetry_root}")

    bot_configs = build_bot_configs(azure_root, telemetry_root)

    try:
        while True:
            new_records = sync_all(bot_configs, max_records)
            if new_records and not args.no_collector:
                run_collector(telemetry_root, args.collector_limit)
            if args.once:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        logging.info("Sync interrupted by user")


if __name__ == "__main__":
    main()
