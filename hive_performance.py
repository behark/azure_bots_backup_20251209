#!/usr/bin/env python3
"""Unified performance report for every trading bot in the hive."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import fnmatch

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "HIVE_PERFORMANCE.md"


def discover_stats_files() -> Dict[str, Path]:
    stats_map: Dict[str, Path] = {}
    for bot_dir in BASE_DIR.iterdir():
        if not bot_dir.is_dir() or not bot_dir.name.endswith("_bot"):
            continue
        logs_dir = bot_dir / "logs"
        if not logs_dir.exists():
            continue
        for candidate in logs_dir.iterdir():
            if candidate.is_file() and fnmatch.fnmatch(candidate.name, "*stats.json"):
                bot_name = bot_dir.name.replace("_", " ").title()
                stats_map[bot_name] = candidate
                break
    return stats_map


def load_history(stats_path: Path) -> List[Dict[str, Any]]:
    if not stats_path.exists():
        return []
    try:
        with stats_path.open("r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    return data.get("history", []) or []


def compute_bot_stats(bot_name: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(history)
    tp1 = sum(1 for s in history if s.get("result") == "TP1")
    tp2 = sum(1 for s in history if s.get("result") == "TP2")
    sl = sum(1 for s in history if s.get("result") == "SL")
    wins = tp1 + tp2
    win_rate = (wins / total * 100) if total else 0.0
    total_pnl = sum(s.get("pnl_pct", 0.0) for s in history)

    symbol_perf: Dict[str, Dict[str, float]] = defaultdict(lambda: {"pnl": 0.0, "signals": 0})
    for signal in history:
        symbol = str(signal.get("symbol", "")).replace("/USDT", "").replace(":USDT", "")
        if not symbol:
            continue
        symbol_perf[symbol]["pnl"] += signal.get("pnl_pct", 0.0)
        symbol_perf[symbol]["signals"] += 1

    return {
        "bot_name": bot_name,
        "total_trades": total,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "tp1": tp1,
        "tp2": tp2,
        "sl": sl,
        "symbol_perf": symbol_perf,
    }


def build_overall_table(bot_stats: List[Dict[str, Any]]) -> str:
    lines = ["## 🧠 Hive Performance Overview", "", "| Bot | Win Rate | Total P&L | Trades |", "| --- | ---: | ---: | ---: |"]
    for stats in sorted(bot_stats, key=lambda x: x["total_pnl"], reverse=True):
        lines.append(
            f"| {stats['bot_name']} | {stats['win_rate']:.2f}% | {stats['total_pnl']:+.2f}% | {stats['total_trades']} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_top_bot_deep_dive(bot_stats: List[Dict[str, Any]], top_n: int = 3) -> str:
    lines = ["## 🔍 Top Bot Deep Dive", ""]
    top_bots = sorted(bot_stats, key=lambda x: x["total_pnl"], reverse=True)[:top_n]

    for stats in top_bots:
        lines.append(f"### {stats['bot_name']}")
        if not stats["symbol_perf"]:
            lines.append("No closed trades yet.\n")
            continue
        lines.append("Best Symbols:")
        lines.append("```")
        best_symbols = sorted(
            stats["symbol_perf"].items(), key=lambda x: x[1]["pnl"], reverse=True
        )[:5]
        for symbol, perf in best_symbols:
            lines.append(
                f"{symbol:<10} {perf['pnl']:+7.2f}% over {int(perf['signals'])} trades"
            )
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def build_blacklist(bot_stats: List[Dict[str, Any]]) -> str:
    symbol_bot_losses: Dict[str, int] = defaultdict(int)
    symbol_pnl_totals: Dict[str, float] = defaultdict(float)

    for stats in bot_stats:
        for symbol, perf in stats["symbol_perf"].items():
            pnl = perf["pnl"]
            symbol_pnl_totals[symbol] += pnl
            if pnl < 0:
                symbol_bot_losses[symbol] += 1

    blacklist = [
        (symbol, symbol_pnl_totals[symbol], symbol_bot_losses[symbol])
        for symbol in symbol_pnl_totals
        if symbol_bot_losses[symbol] >= 2 and symbol_pnl_totals[symbol] < 0
    ]

    if not blacklist:
        return "## 🧹 Blacklist Recommendation\n\nNo symbols show recurring losses across multiple bots."

    lines = ["## 🧹 Blacklist Recommendation", "", "Symbols to consider removing across the hive:", ""]
    lines.append("| Symbol | Total P&L | Bots Losing |")
    lines.append("| --- | ---: | ---: |")
    for symbol, pnl, bots in sorted(blacklist, key=lambda x: x[1]):
        lines.append(f"| {symbol} | {pnl:+.2f}% | {bots} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    stats_files = discover_stats_files()
    if not stats_files:
        print("No bot stats files found. Exiting.")
        return

    bot_stats: List[Dict[str, Any]] = []
    for bot_name, stats_path in stats_files.items():
        history = load_history(stats_path)
        stats = compute_bot_stats(bot_name, history)
        bot_stats.append(stats)

    report_sections = ["# 🐝 Hive Performance Report", ""]
    report_sections.append(build_overall_table(bot_stats))
    report_sections.append(build_top_bot_deep_dive(bot_stats))
    report_sections.append(build_blacklist(bot_stats))

    markdown_output = "\n".join(report_sections)
    OUTPUT_FILE.write_text(markdown_output)

    print("✅ Hive performance report generated.")
    print(f"📄 Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
