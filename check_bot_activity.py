#!/usr/bin/env python3
"""Audit bot activity by inspecting log freshness, signal stats, and recent errors."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent


def discover_bots() -> List[Path]:
    return sorted(path for path in BASE_DIR.glob("*_bot") if path.is_dir())


def newest_file(files: List[Path]) -> Optional[Path]:
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def read_stats(stats_path: Path) -> Tuple[Optional[datetime], int]:
    try:
        data = json.loads(stats_path.read_text())
    except Exception:
        return None, 0
    history = data.get("history", []) or []
    if not history:
        return None, 0
    latest = history[-1]
    ts = latest.get("closed_at") or latest.get("created_at")
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")), len(history)
    except Exception:
        return None, len(history)


def tail_errors(log_path: Path, limit: int, minutes: int) -> List[str]:
    window = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    try:
        text = log_path.read_text()[-20000:]
    except Exception:
        return []
    errors = []
    for line in text.splitlines():
        if "ERROR" not in line and "Traceback" not in line and "CRITICAL" not in line:
            continue
        timestamp = extract_timestamp(line)
        if timestamp and timestamp < window:
            continue
        errors.append(line.strip())
        if len(errors) >= limit:
            break
    return errors


def extract_timestamp(line: str) -> Optional[datetime]:
    line = line.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            stamp = datetime.strptime(line[:19], fmt)
            return stamp.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def format_age(dt: Optional[datetime]) -> str:
    if not dt:
        return "never"
    delta = datetime.now(timezone.utc) - dt
    hours, remainder = divmod(int(delta.total_seconds()), 3600)
    minutes = remainder // 60
    return f"{hours}h {minutes}m ago"


def analyze_bot(
    bot_dir: Path,
    log_stale_minutes: int,
    signal_stale_hours: int,
    error_minutes: int,
    error_limit: int,
) -> Dict[str, object]:
    logs_dir = bot_dir / "logs"
    log_file = newest_file(list(logs_dir.glob("*.log"))) if logs_dir.exists() else None
    stats_file = newest_file(list(logs_dir.glob("*stats*.json"))) if logs_dir.exists() else None

    last_log_time = (
        datetime.fromtimestamp(log_file.stat().st_mtime, tz=timezone.utc)
        if log_file and log_file.exists()
        else None
    )
    last_signal_time, total_signals = (
        read_stats(stats_file) if stats_file else (None, 0)
    )

    log_stale = (
        last_log_time is None
        or datetime.now(timezone.utc) - last_log_time > timedelta(minutes=log_stale_minutes)
    )
    signal_stale = (
        last_signal_time is None
        or datetime.now(timezone.utc) - last_signal_time > timedelta(hours=signal_stale_hours)
    )

    errors = tail_errors(log_file, error_limit, error_minutes) if log_file else []

    return {
        "bot": bot_dir.name,
        "log_file": log_file,
        "stats_file": stats_file,
        "last_log": last_log_time,
        "last_signal": last_signal_time,
        "total_signals": total_signals,
        "log_stale": log_stale,
        "signal_stale": signal_stale,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check bots for recent activity and errors.")
    parser.add_argument(
        "--log-stale-minutes",
        type=int,
        default=15,
        help="Minutes without log updates before flagging bot as stale.",
    )
    parser.add_argument(
        "--signal-stale-hours",
        type=int,
        default=6,
        help="Hours without closed signals before flagging bot as inactive.",
    )
    parser.add_argument(
        "--error-window-minutes",
        type=int,
        default=120,
        help="Lookback window for collecting recent errors.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=5,
        help="Maximum number of recent error lines to display per bot.",
    )
    return parser.parse_args()


def print_report(results: List[Dict[str, object]]) -> None:
    print("\n=== BOT ACTIVITY REPORT ===\n")
    for result in results:
        bot = result["bot"]
        print(f"{bot}")
        print("-" * len(bot))
        log_status = "STALE" if result["log_stale"] else "OK"
        signal_status = "STALE" if result["signal_stale"] else "OK"
        print(f"Last log:    {format_age(result['last_log'])} ({log_status})")
        print(f"Last signal: {format_age(result['last_signal'])} ({signal_status})")
        print(f"Signals recorded: {result['total_signals']}")
        if errors := result["errors"]:
            print("Recent errors:")
            for err in errors:
                print(f"  • {err}")
        else:
            print("Recent errors: none")
        print()

    stale = [r for r in results if r["log_stale"] or r["signal_stale"]]
    if stale:
        print("Bots flagged:")
        for r in stale:
            reasons = []
            if r["log_stale"]:
                reasons.append("stale logs")
            if r["signal_stale"]:
                reasons.append("no signals")
            print(f"- {r['bot']}: {', '.join(reasons)}")
    else:
        print("All bots appear healthy based on log and signal activity.")


def main() -> int:
    args = parse_args()
    bots = discover_bots()
    if not bots:
        print("No bot directories found.")
        return 1

    results = [
        analyze_bot(
            bot,
            log_stale_minutes=args.log_stale_minutes,
            signal_stale_hours=args.signal_stale_hours,
            error_minutes=args.error_window_minutes,
            error_limit=args.max_errors,
        )
        for bot in bots
    ]
    print_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
