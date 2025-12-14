#!/usr/bin/env python3
"""Analyze each bot to see whether it sends detailed SignalStats summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent


def find_main_script(bot_dir: Path) -> Path | None:
    candidates = list(bot_dir.glob("*_bot.py"))
    if not candidates:
        return None
    preferred = bot_dir.name.replace("-", "_")
    for cand in candidates:
        if cand.stem == bot_dir.name.rstrip("/"):
            return cand
        if cand.stem == preferred:
            return cand
    return candidates[0]


def discover_bots() -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for bot_dir in sorted(BASE_DIR.glob("*_bot")):
        if not bot_dir.is_dir():
            continue
        main = find_main_script(bot_dir)
        if main:
            mapping[bot_dir.name] = main
    return mapping


def analyze_file(path: Path) -> Dict[str, bool]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    has_signal_stats = "SignalStats" in text
    uses_summary = "build_summary_message" in text
    records_close = "record_close" in text
    mentions_perf = "Performance Stats" in text
    return {
        "has_signal_stats": has_signal_stats,
        "uses_summary_message": uses_summary,
        "tracks_closed_signals": records_close,
        "mentions_performance_text": mentions_perf,
    }


def render_report(results: Dict[str, Dict[str, bool]], output_json: bool) -> None:
    if output_json:
        print(json.dumps(results, indent=2))
        return
    header = f"{'Bot':<18} {'SignalStats':<12} {'Summary':<10} {'PerfText':<10}"
    print(header)
    print("-" * len(header))
    for bot, info in results.items():
        line = f"{bot:<18} {str(info['has_signal_stats']):<12} {str(info['uses_summary_message']):<10} {str(info['mentions_performance_text']):<10}"
        print(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check which bots send detailed performance summaries.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of table output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bots = discover_bots()
    if not bots:
        print("No bot directories found.")
        return 1
    results: Dict[str, Dict[str, bool]] = {}
    for bot_name, script_path in bots.items():
        results[bot_name] = analyze_file(script_path)
    render_report(results, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
