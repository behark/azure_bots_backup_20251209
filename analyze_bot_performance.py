#!/usr/bin/env python3
"""Summarize performance metrics for every bot that maintains SignalStats JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent


@dataclass
class BotStats:
    bot: str
    total: int
    tp_hits: int
    sl_hits: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    last_trade: Optional[str]
    stats_file: Path


def discover_stats_files() -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for bot_dir in sorted(BASE_DIR.glob("*_bot")):
        logs_dir = bot_dir / "logs"
        if not logs_dir.is_dir():
            continue
        candidates = sorted(logs_dir.glob("*stats*.json"))
        if not candidates:
            continue
        mapping[bot_dir.name] = candidates[0]
    return mapping


def load_history(stats_file: Path) -> List[Dict[str, object]]:
    try:
        data = json.loads(stats_file.read_text())
    except Exception:
        return []
    history = data.get("history", [])
    return history if isinstance(history, list) else []


def compute_bot_stats(bot: str, stats_file: Path) -> Optional[BotStats]:
    history = load_history(stats_file)
    if not history:
        return None

    total = len(history)
    tp_hits = sum(1 for h in history if str(h.get("result", "")).startswith("TP"))
    sl_hits = sum(1 for h in history if str(h.get("result")) == "SL")
    win_rate = (tp_hits / total * 100) if total else 0.0
    pnl_values = [float(h.get("pnl_pct", 0.0)) for h in history]
    total_pnl = sum(pnl_values)
    win_pnls = [float(h.get("pnl_pct", 0.0)) for h in history if str(h.get("result", "")).startswith("TP")]
    loss_pnls = [float(h.get("pnl_pct", 0.0)) for h in history if str(h.get("result")) == "SL"]
    avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0.0
    avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0
    last_trade = history[-1].get("closed_at") if history else None

    return BotStats(
        bot=bot,
        total=total,
        tp_hits=tp_hits,
        sl_hits=sl_hits,
        win_rate=win_rate,
        total_pnl=total_pnl,
        avg_win=avg_win,
        avg_loss=avg_loss,
        last_trade=last_trade,
        stats_file=stats_file,
    )


def print_report(stats: List[BotStats]) -> None:
    if not stats:
        print("No bot stats available.")
        return

    stats_sorted = sorted(stats, key=lambda s: s.total_pnl, reverse=True)
    header = f"{'Bot':<20} {'Signals':<9} {'Win%':<8} {'TP Hits':<8} {'SL Hits':<8} {'Total P&L%':<12}"
    print(header)
    print("-" * len(header))
    for item in stats_sorted:
        print(
            f"{item.bot:<20} {item.total:<9} {item.win_rate:>6.1f}% "
            f"{item.tp_hits:<8} {item.sl_hits:<8} {item.total_pnl:>+11.2f}"
        )

    best = stats_sorted[0]
    worst = stats_sorted[-1]
    print()
    print(f"🥇 Best bot: {best.bot} ({best.total_pnl:+.2f}% P&L, {best.win_rate:.1f}% win rate)")
    print(f"⚠️  Lagging bot: {worst.bot} ({worst.total_pnl:+.2f}% P&L, {worst.win_rate:.1f}% win rate)")


def main() -> int:
    stats_files = discover_stats_files()
    results: List[BotStats] = []
    for bot, path in stats_files.items():
        bot_stats = compute_bot_stats(bot, path)
        if bot_stats:
            results.append(bot_stats)
    print_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
