#!/usr/bin/env python3
"""Quick health report for Azure bots.

Scans running processes, tail of each bot log, and highlights the latest
ERROR/Traceback so we can see which bots are down without digging manually.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"

BOTS: Dict[str, Dict[str, str]] = {
    "harmonic": {"script": "harmonic_bot/harmonic_bot.py", "log": "harmonic.log"},
    "candlestick": {"script": "candlestick_bot/candlestick_bot.py", "log": "candlestick.log"},
    "mtf": {"script": "mtf_bot/mtf_bot.py", "log": "mtf.log"},
    "psar": {"script": "psar_bot/psar_bot.py", "log": "psar.log"},
    "diy": {"script": "diy_bot/diy_bot.py", "log": "diy.log"},
    "most": {"script": "most_bot/most_bot.py", "log": "most.log"},
    "strat": {"script": "strat_bot/strat_bot.py", "log": "strat.log"},
    "fib_reversal": {"script": "fib_reversal_bot/fib_reversal_bot.py", "log": "fib_reversal.log"},
    "fib_swing": {"script": "fib_swing_bot/fib_swing_bot.py", "log": "fib_swing.log"},
    "funding": {"script": "funding_bot/funding_bot.py", "log": "funding.log"},
    "liquidation": {"script": "liquidation_bot/liquidation_bot.py", "log": "liquidation.log"},
    "volume": {"script": "volume_bot/volume_vn_bot.py", "log": "volume.log"},
    "consensus": {"script": "consensus_bot/consensus_bot.py", "log": "consensus.log"},
}

EXTERNAL_BOTS: Dict[str, Dict[str, str]] = {
    "nf_harmonic": {
        "script": "/home/behar/Desktop/New Folder (3)/harmonic_bot.py",
        "log": "/home/behar/Desktop/New Folder (3)/logs/harmonic_live.log",
    }
}


def capture_process_table() -> List[str]:
    result = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True, text=True, check=True)
    return result.stdout.strip().splitlines()


def map_running_processes(lines: List[str]) -> Dict[str, Tuple[str, str]]:
    running: Dict[str, Tuple[str, str]] = {}
    for line in lines:
        for bot, meta in BOTS.items():
            if meta["script"] in line:
                running[bot] = tuple(line.strip().split(maxsplit=1))  # (pid, cmd)
    return running


def tail_lines(path: Path, limit: int = 200) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", errors="ignore") as handle:
        lines = handle.readlines()
    return lines[-limit:]


def extract_last_error(lines: List[str]) -> str:
    for line in reversed(lines):
        text = line.strip()
        if "ERROR" in text or "Traceback" in text or "Exception" in text:
            return text
    return "(no recent errors)"


def summarize_bot(bot: str, running: Dict[str, Tuple[str, str]]) -> str:
    meta = BOTS[bot]
    log_path = LOG_DIR / meta["log"]
    status = "RUNNING" if bot in running else "STOPPED"
    pid_info = running.get(bot, ("-", ""))[0]
    log_lines = tail_lines(log_path)
    last_line = log_lines[-1].strip() if log_lines else "(log missing)"
    last_error = extract_last_error(log_lines)
    return (
        f"{bot.upper():<13} | {status:<8} | PID {pid_info:<6} | "
        f"Last line: {last_line}\n    ↳ {last_error}"
    )


def main() -> None:
    try:
        ps_lines = capture_process_table()
    except subprocess.CalledProcessError as exc:
        print(f"Failed to read process table: {exc}")
        return

    running = map_running_processes(ps_lines)
    print("BOT HEALTH REPORT")
    print("=" * 70)
    for bot in BOTS:
        print(summarize_bot(bot, running))
        print("-" * 70)

    for bot, meta in EXTERNAL_BOTS.items():
        log_path = Path(meta["log"])
        status = "RUNNING"
        pid = "-"
        for line in ps_lines:
            if meta["script"] in line:
                pid = line.strip().split(maxsplit=1)[0]
                break
        log_lines = tail_lines(log_path, limit=200)
        last_line = log_lines[-1].strip() if log_lines else "(log missing)"
        last_error = extract_last_error(log_lines)
        print(
            f"{bot.upper():<13} | {status:<8} | PID {pid:<6} | Last line: {last_line}\n    ↳ {last_error}"
        )
        print("-" * 70)


if __name__ == "__main__":
    main()
