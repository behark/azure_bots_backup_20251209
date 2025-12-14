#!/usr/bin/env python3
"""Periodic performance digest for VM bots."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any
import fnmatch

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore

BASE_DIR = Path(__file__).resolve().parent
HOME = Path.home()

if load_dotenv:
    load_dotenv(HOME / ".env")
    load_dotenv(BASE_DIR / ".env")

sys.path.append(str(HOME))

from signal_stats import SignalStats  # type: ignore  # noqa: E402
from notifier import TelegramNotifier  # type: ignore  # noqa: E402

def discover_bots() -> Dict[str, Dict[str, Any]]:
    bots: Dict[str, Dict[str, Any]] = {}
    for bot_dir in HOME.iterdir():
        if not bot_dir.is_dir() or not bot_dir.name.endswith("_bot"):
            continue
        logs_dir = bot_dir / "logs"
        if not logs_dir.exists() or not logs_dir.is_dir():
            continue
        stats_file = None
        for candidate in logs_dir.iterdir():
            if candidate.is_file() and fnmatch.fnmatch(candidate.name, "*stats.json"):
                stats_file = candidate
                break
        if not stats_file:
            continue
        key = bot_dir.name.replace("_bot", "")
        pretty_name = bot_dir.name.replace("_", " ").title()
        token_env = f"TELEGRAM_BOT_TOKEN_{key.upper()}"
        bots[key] = {
            "name": pretty_name,
            "stats_path": stats_file,
            "token_env": token_env,
        }
    return bots


def build_digest(bot_key: str, config: Dict[str, Dict[str, Any]]) -> str | None:
    cfg = config[bot_key]
    stats_file = cfg["stats_path"]
    if not stats_file.exists():
        return None

    stats = SignalStats(str(cfg["name"]), stats_file)
    summary = stats.get_summary()
    history: List[Dict[str, Any]] = stats.data.get("history", [])  # type: ignore[attr-defined]
    open_positions: Dict[str, Any] = stats.data.get("open", {})  # type: ignore[attr-defined]
    total_trades = len(history)
    open_count = len(open_positions)

    last_trade_info = "No closed trades yet"
    if history:
        last = history[-1]
        last_trade_info = (
            f"Last: {last.get('symbol')} {last.get('direction')} {last.get('result')}"
            f" @ {last.get('exit', 'N/A')} ({float(last.get('pnl_pct', 0)):+.2f}%)"
        )

    lines = [
        f"📊 {cfg['name']} Performance Digest",
        "",
        f"Total Signals: {total_trades}",
        f"Open Positions: {open_count}",
        f"Win Rate: {summary['win_rate']:.1f}%",
        f"TP Hits: {summary['tp_hits']} | SL Hits: {summary['sl_hits']}",
        f"Cumulative P&L: {summary['total_pnl']:+.2f}%",
        "",
        last_trade_info,
        "",
        f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
    ]

    return "\n".join(lines)


def send_digest(bot_key: str, config: Dict[str, Dict[str, Any]]) -> None:
    message = build_digest(bot_key, config)
    if not message:
        print(f"No stats found for {bot_key}; skipping")
        return

    cfg = config[bot_key]
    token = os.getenv(str(cfg["token_env"])) or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print(f"Missing Telegram credentials for {bot_key}; printing instead:\n{message}")
        return

    notifier = TelegramNotifier(token, chat_id)
    if notifier.send_message(message):
        print(f"Digest sent for {bot_key}")
    else:
        print(f"Failed to send digest for {bot_key}; message was:\n{message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Send bot performance digests to Telegram")
    parser.add_argument(
        "--bot",
        default="all",
        help="Which bot digest to send (e.g. 'harmonic' or 'all')",
    )
    args = parser.parse_args()
    config = discover_bots()
    if not config:
        print("No bots discovered. Exiting.")
        return
    if args.bot == "all":
        selected = config.keys()
    else:
        key = args.bot.lower()
        if key not in config:
            print(f"Unknown bot '{args.bot}'. Available: {', '.join(sorted(config.keys()))}")
            return
        selected = [key]
    for bot in selected:
        send_digest(bot, config)


if __name__ == "__main__":
    main()
