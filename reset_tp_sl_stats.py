#!/usr/bin/env python3
"""
Utility to reset TP/SL stats for all bots by clearing their stats JSON files.
Supports filtering, dry-run previews, and optional confirmation prompts.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

BASE_DIR = Path(__file__).resolve().parent


def discover_stats_files() -> Dict[str, List[Path]]:
    """Return mapping of bot name -> list of stats files under its logs directory."""
    mapping: Dict[str, List[Path]] = {}
    for bot_dir in sorted(BASE_DIR.glob("*_bot")):
        logs_dir = bot_dir / "logs"
        if not logs_dir.is_dir():
            continue
        files = sorted(p for p in logs_dir.glob("*stats*.json") if p.is_file())
        if files:
            mapping[bot_dir.name] = files
    return mapping


def normalize_name(name: str) -> str:
    return name.lower().replace("_bot", "")


def filter_bots(available: Dict[str, List[Path]], selectors: Iterable[str] | None) -> Dict[str, List[Path]]:
    if not selectors:
        return available
    selected = {}
    normalized = {normalize_name(n): n for n in available}
    for raw in selectors:
        key = normalize_name(raw)
        match = normalized.get(key)
        if match:
            selected[match] = available[match]
        else:
            print(f"[WARN] No stats files found for selector '{raw}'.", file=sys.stderr)
    return selected


def reset_stats_file(path: Path, dry_run: bool, backup: bool) -> None:
    payload = {"open": {}, "history": []}
    if dry_run:
        action = "would reset" if path.exists() else "would create"
        print(f"[DRY-RUN] {action} {path}")
        return
    if backup and path.exists():
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        backup_path = path.with_suffix(path.suffix + f".bak.{timestamp}")
        backup_path.write_bytes(path.read_bytes())
        print(f"[INFO] Backup created: {backup_path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"[OK] Reset {path}")


def confirm(prompt: str) -> bool:
    reply = input(f"{prompt} [y/N]: ").strip().lower()
    return reply in {"y", "yes"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset TP/SL stats for trading bots.")
    parser.add_argument(
        "--bot",
        dest="bots",
        action="append",
        help="Only reset specific bot(s) (e.g. 'volume' or 'volume_bot').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying any files.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt and reset immediately.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak backups before overwriting files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    available = discover_stats_files()
    if not available:
        print("No bot stats files discovered. Nothing to do.")
        return 0

    targets = filter_bots(available, args.bots)
    if not targets:
        print("No matching bots selected. Exiting.")
        return 1

    print("Bots to reset:")
    for bot, files in targets.items():
        for file in files:
            print(f"- {bot}: {file}")

    if not args.dry_run and not args.yes:
        if not confirm("Proceed with resetting these stats files?"):
            print("Aborted by user.")
            return 1

    backup = not args.no_backup
    for files in targets.values():
        for path in files:
            reset_stats_file(path, args.dry_run, backup)

    if args.dry_run:
        print("Dry-run complete. No files were modified.")
    else:
        print("Reset complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
