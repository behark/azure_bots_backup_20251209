#!/usr/bin/env python3
"""
TOXIC ASSET FORENSICS (BACKUP MODE)
Analyzes the .bak files created during the System Reset to see historical performance.
"""

import json
from pathlib import Path
from collections import defaultdict

# The list of suspects
TOXIC_SYMBOLS = [
    "FRANKLIN", "GAIX", "VVV", "PIPPIN", "LRC",
    "RLS", "SUI", "ASTER", "POWER", "DOGE", "IRYS"
]

BASE_DIR = Path(__file__).resolve().parent

def discover_backups():
    stats_map = {}
    for bot_dir in BASE_DIR.glob("*_bot"):
        if not bot_dir.is_dir(): continue
        logs = bot_dir / "logs"
        if logs.exists():
            # Look for the most recent backup file
            backups = sorted(logs.glob("*stats.json.bak*"), reverse=True)
            if backups:
                bot_name = bot_dir.name.replace("_bot", "").upper()
                stats_map[bot_name] = backups[0] # Take the newest backup
                print(f"found backup for {bot_name}: {backups[0].name}")
    return stats_map

def analyze_suspects():
    bots = discover_backups()

    if not bots:
        print("❌ No backup files found! Did you run the reset with --no-backup?")
        return

    # Structure: results[symbol][bot] = {pnl, trades, wins}
    results = defaultdict(dict)

    print(f"\n🕵️  Investigating backups for {len(TOXIC_SYMBOLS)} suspects across {len(bots)} bots...\n")

    for bot_name, file_path in bots.items():
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        history = data.get('history', [])

        for t in history:
            # Clean symbol name
            sym = t.get('symbol', '').replace('/USDT', '').replace(':USDT', '')

            if sym in TOXIC_SYMBOLS:
                if bot_name not in results[sym]:
                    results[sym][bot_name] = {'pnl': 0.0, 'wins': 0, 'total': 0}

                stats = results[sym][bot_name]
                stats['pnl'] += float(t.get('pnl_pct', 0))
                stats['total'] += 1
                if str(t.get('result', '')).startswith('TP'):
                    stats['wins'] += 1

    # Print Report
    print(f"{'SYMBOL':<10} | {'BOT NAME':<15} | {'TRADES':<6} | {'WIN %':<6} | {'P&L'}")
    print("-" * 65)

    for sym in TOXIC_SYMBOLS:
        if sym not in results:
            print(f"⚪ {sym} (No trades found in backups)")
            continue

        print(f"🔴 {sym}")
        bot_stats = results[sym]

        # Sort by P&L to see the worst offenders first
        sorted_bots = sorted(bot_stats.items(), key=lambda x: x[1]['pnl'])

        for bot, s in sorted_bots:
            win_rate = (s['wins'] / s['total']) * 100
            pnl_str = f"{s['pnl']:+.2f}%"

            # Color coding
            if s['pnl'] > 0:
                pnl_display = f"\033[92m{pnl_str}\033[0m" # Green
            else:
                pnl_display = f"\033[91m{pnl_str}\033[0m" # Red

            print(f"{'':<10} | {bot:<15} | {s['total']:<6} | {win_rate:>5.1f}% | {pnl_display}")
        print("-" * 65)

if __name__ == "__main__":
    analyze_suspects()
