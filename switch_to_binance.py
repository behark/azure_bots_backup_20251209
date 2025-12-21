#!/usr/bin/env python3
"""
Script to switch all bots from MEXC to Binance
- Updates exchange in all config files
- Updates exchange in all watchlist files
- Removes BLUAI/USDT
- Adds new symbols: TRUTH, SOON, ASR, ZKJ, H, FHE, NIGHT
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent

# New symbols to add
NEW_SYMBOLS = [
    {"symbol": "TRUTH/USDT", "timeframe": "15m", "exchange": "binance", "market_type": "swap"},
    {"symbol": "SOON/USDT", "timeframe": "15m", "exchange": "binance", "market_type": "swap"},
    {"symbol": "ASR/USDT", "timeframe": "15m", "exchange": "binance", "market_type": "swap"},
    {"symbol": "ZKJ/USDT", "timeframe": "15m", "exchange": "binance", "market_type": "swap"},
    {"symbol": "H/USDT", "timeframe": "15m", "exchange": "binance", "market_type": "swap"},
    {"symbol": "FHE/USDT", "timeframe": "15m", "exchange": "binance", "market_type": "swap"},
    {"symbol": "NIGHT/USDT", "timeframe": "15m", "exchange": "binance", "market_type": "swap"},
]

# Symbol to remove
REMOVE_SYMBOL = "BLUAI/USDT"

# Watchlist files
WATCHLIST_FILES = [
    "diy_bot/diy_watchlist.json",
    "fib_swing_bot/fib_watchlist.json",
    "funding_bot/funding_watchlist.json",
    "harmonic_bot/harmonic_watchlist.json",
    "liquidation_bot/liquidation_watchlist.json",
    "most_bot/most_watchlist.json",
    "mtf_bot/mtf_watchlist.json",
    "orb_bot/orb_watchlist.json",
    "psar_bot/psar_watchlist.json",
    "strat_bot/strat_watchlist.json",
    "volume_bot/volume_watchlist.json",
    "volume_profile_bot/watchlist.json",
]

# Config files with exchange settings
CONFIG_FILES = [
    "diy_bot/diy_config.json",
    "funding_bot/funding_config.json",
    "harmonic_bot/harmonic_config.json",
    "orb_bot/orb_config.json",
    "volume_bot/config.json",
]


def update_watchlist(filepath: Path) -> bool:
    """Update a watchlist file."""
    try:
        with open(filepath, 'r') as f:
            watchlist = json.load(f)

        # Remove BLUAI
        original_count = len(watchlist)
        watchlist = [item for item in watchlist if item.get("symbol") != REMOVE_SYMBOL]
        removed = original_count - len(watchlist)

        # Update exchange to binance for existing items
        for item in watchlist:
            if item.get("exchange") in ["mexc", "bybit"]:
                item["exchange"] = "binance"

        # Get existing symbols to avoid duplicates
        existing_symbols = {item.get("symbol") for item in watchlist}

        # Add new symbols
        added = 0
        for new_sym in NEW_SYMBOLS:
            if new_sym["symbol"] not in existing_symbols:
                # Copy the structure from existing items if available
                if watchlist:
                    new_item = watchlist[0].copy()
                    new_item.update(new_sym)
                else:
                    new_item = new_sym.copy()
                watchlist.append(new_item)
                added += 1

        # Write back
        with open(filepath, 'w') as f:
            json.dump(watchlist, f, indent=2)

        print(f"  ✅ {filepath.name}: {len(watchlist)} symbols (removed {removed}, added {added})")
        return True

    except Exception as e:
        print(f"  ❌ {filepath.name}: {e}")
        return False


def update_config(filepath: Path) -> bool:
    """Update a config file to use binance."""
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)

        # Update exchange settings
        updated = False
        if "exchange_settings" in config:
            if config["exchange_settings"].get("active_exchange") in ["mexc", "bybit"]:
                config["exchange_settings"]["active_exchange"] = "binance"
                updated = True

        if "exchange" in config:
            if config["exchange"] in ["mexc", "bybit"]:
                config["exchange"] = "binance"
                updated = True

        # Write back
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

        status = "updated to binance" if updated else "no change needed"
        print(f"  ✅ {filepath.name}: {status}")
        return True

    except Exception as e:
        print(f"  ❌ {filepath.name}: {e}")
        return False


def main():
    print("=" * 60)
    print("Switching all bots from MEXC to Binance")
    print("=" * 60)

    print(f"\n📋 Changes:")
    print(f"  - Exchange: MEXC → Binance")
    print(f"  - Remove: {REMOVE_SYMBOL}")
    print(f"  - Add: {', '.join(s['symbol'] for s in NEW_SYMBOLS)}")

    print(f"\n📁 Updating {len(WATCHLIST_FILES)} watchlist files:")
    for wf in WATCHLIST_FILES:
        update_watchlist(BASE_DIR / wf)

    print(f"\n⚙️  Updating {len(CONFIG_FILES)} config files:")
    for cf in CONFIG_FILES:
        update_config(BASE_DIR / cf)

    print("\n" + "=" * 60)
    print("✅ Done! Next steps:")
    print("  1. Add Binance API keys to .env files")
    print("  2. Run: ./stop_all_bots.sh")
    print("  3. Run: ./start_all_bots.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()
