#!/usr/bin/env python3
"""
SURGICAL STRIKE: Targeted Asset Removal
Based on forensic analysis of backup logs.
Removes toxic assets ONLY from the specific bots where they lose money.
"""

import json
import os
from pathlib import Path

# The Logic Map derived from your report
# Format: "SYMBOL": ["BOT_NAME_1", "BOT_NAME_2"] (Bots to REMOVE from)
REMOVAL_PLAN = {
    "FRANKLIN": ["most", "mtf", "volume", "liquidation"],
    "GAIX":     ["psar", "mtf", "liquidation", "diy", "most", "candlestick", "fib_reversal"],
    "VVV":      ["harmonic", "mtf", "candlestick", "fib_reversal", "psar", "diy", "most", "volume"],
    "PIPPIN":   ["liquidation", "diy", "most", "strat", "fib_reversal", "mtf", "candlestick", "volume"],
    "LRC":      ["mtf", "strat", "psar", "liquidation", "fib_reversal", "diy", "candlestick", "most"],
    "RLS":      ["diy", "mtf", "strat", "candlestick", "liquidation", "funding", "fib_reversal"],
    "SUI":      ["liquidation", "candlestick", "mtf", "strat", "most", "psar"],
    "ASTER":    ["most", "fib_reversal", "mtf", "diy", "psar", "candlestick", "volume", "liquidation"],
    "POWER":    ["mtf", "most", "candlestick", "volume", "fib_reversal"],
    "DOGE":     ["volume", "mtf", "most", "strat"],
    "IRYS":     ["harmonic", "volume", "candlestick", "funding", "strat", "most"]
}

BASE_DIR = Path(__file__).resolve().parent

def get_watchlist_path(bot_name):
    """Finds the watchlist file for a given bot name."""
    # Handle special naming cases if any, otherwise standard mapping
    # Standard: bot name "most" -> dir "most_bot" -> file "most_watchlist.json"
    dir_name = f"{bot_name}_bot"
    file_name = f"{bot_name}_watchlist.json"

    # Check inside bot folder first
    path = BASE_DIR / dir_name / file_name
    if path.exists():
        return path

    # Check root folder fallback
    path = BASE_DIR / file_name
    if path.exists():
        return path

    return None

def execute_strike():
    print(f"🚀 INITIATING SURGICAL STRIKE on {len(REMOVAL_PLAN)} Assets...\n")

    changes_made = 0

    for symbol, target_bots in REMOVAL_PLAN.items():
        print(f"🎯 Target: {symbol}")

        for bot in target_bots:
            path = get_watchlist_path(bot)

            if not path:
                print(f"   ⚠️  Could not find watchlist for {bot}")
                continue

            try:
                # Load Watchlist
                with open(path, 'r') as f:
                    watchlist = json.load(f)

                original_count = len(watchlist)

                # Filter out the toxic symbol
                # Checks for "SYMBOL" or "SYMBOL/USDT"
                new_watchlist = [
                    item for item in watchlist
                    if item['symbol'].replace('/USDT', '').replace(':USDT', '') != symbol
                ]

                if len(new_watchlist) < original_count:
                    # Save changes
                    with open(path, 'w') as f:
                        json.dump(new_watchlist, f, indent=2)
                    print(f"   ✅ Removed from {bot.upper()}")
                    changes_made += 1
                else:
                    print(f"   ⚪ Not found in {bot.upper()} (already gone?)")

            except Exception as e:
                print(f"   ❌ Error updating {bot}: {e}")
        print("-" * 40)

    print(f"\n✨ DONE. {changes_made} adjustments applied across the fleet.")
    print("   The Hive Mind is now optimized.")

if __name__ == "__main__":
    execute_strike()
