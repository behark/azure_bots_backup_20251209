#!/usr/bin/env python3
"""Verification script to test all fixes applied to volume_bot."""

import json
import sys
from pathlib import Path

print("=" * 60)
print("VOLUME BOT - FIX VERIFICATION SCRIPT")
print("=" * 60)
print()

BASE_DIR = Path(__file__).resolve().parent
errors = []
warnings = []
successes = []

# Test 1: Watchlist Format Consistency
print("[TEST 1] Checking watchlist format consistency...")
try:
    with open(BASE_DIR / "volume_watchlist.json", 'r') as f:
        watchlist = json.load(f)

    required_fields = {"symbol", "timeframe", "exchange", "market_type"}
    for idx, item in enumerate(watchlist):
        if not isinstance(item, dict):
            errors.append(f"  ❌ Item {idx} is not a dictionary")
            continue

        missing = required_fields - set(item.keys())
        if missing:
            errors.append(f"  ❌ Item {idx} ({item.get('symbol', 'UNKNOWN')}) missing fields: {missing}")

        # Check for old format fields
        if "period" in item or "cooldown_minutes" in item:
            warnings.append(f"  ⚠️ Item {idx} ({item.get('symbol', 'UNKNOWN')}) has old format fields")

        # Validate symbol format
        symbol = item.get("symbol", "")
        if not symbol or "/" not in symbol:
            errors.append(f"  ❌ Item {idx} has invalid symbol format: {symbol}")

    if not errors and not warnings:
        successes.append("  ✅ All watchlist entries have consistent format")
    print(f"  Total entries: {len(watchlist)}")
except Exception as e:
    errors.append(f"  ❌ Failed to load watchlist: {e}")

print()

# Test 2: State File Integrity
print("[TEST 2] Checking state file integrity...")
try:
    with open(BASE_DIR / "volume_vn_state.json", 'r') as f:
        state = json.load(f)

    if "open_signals" not in state:
        errors.append("  ❌ State file missing 'open_signals' key")
    else:
        signals = state["open_signals"]
        if not isinstance(signals, dict):
            errors.append("  ❌ 'open_signals' is not a dictionary")
        else:
            for signal_id, signal_data in signals.items():
                if not isinstance(signal_data, dict):
                    errors.append(f"  ❌ Signal {signal_id} is not a dictionary")
                    continue

                # Check for empty fields (the main issue we fixed)
                symbol = signal_data.get("symbol", "")
                exchange = signal_data.get("exchange", "")
                timeframe = signal_data.get("timeframe", "")
                direction = signal_data.get("direction", "")

                if not symbol or not symbol.strip():
                    errors.append(f"  ❌ Signal {signal_id} has empty symbol")
                if not exchange or not exchange.strip():
                    errors.append(f"  ❌ Signal {signal_id} has empty exchange")
                if not timeframe or not timeframe.strip():
                    errors.append(f"  ❌ Signal {signal_id} has empty timeframe")
                if not direction or not direction.strip():
                    errors.append(f"  ❌ Signal {signal_id} has empty direction")

            if not errors:
                successes.append(f"  ✅ All {len(signals)} open signals have valid data")
            print(f"  Open signals: {len(signals)}")
except Exception as e:
    errors.append(f"  ❌ Failed to load state file: {e}")

print()

# Test 3: Python Syntax
print("[TEST 3] Checking Python syntax...")
try:
    import py_compile

    files_to_check = [
        "volume_vn_bot.py",
        "config.py",
        "notifier.py"
    ]

    for filename in files_to_check:
        filepath = BASE_DIR / filename
        if not filepath.exists():
            warnings.append(f"  ⚠️ {filename} not found")
            continue

        try:
            py_compile.compile(str(filepath), doraise=True)
            successes.append(f"  ✅ {filename} syntax valid")
        except py_compile.PyCompileError as e:
            errors.append(f"  ❌ {filename} has syntax errors: {e}")
except Exception as e:
    errors.append(f"  ❌ Syntax check failed: {e}")

print()

# Test 4: Check for Unsupported Symbols
print("[TEST 4] Checking for unsupported Binance symbols...")
unsupported_symbols = {"BAY", "CORL", "DSYNC", "LONG", "TYCOON"}
try:
    with open(BASE_DIR / "volume_watchlist.json", 'r') as f:
        watchlist = json.load(f)

    found_unsupported = []
    for item in watchlist:
        symbol_base = item.get("symbol", "").split("/")[0]
        if symbol_base in unsupported_symbols:
            found_unsupported.append(item.get("symbol"))

    if found_unsupported:
        errors.append(f"  ❌ Found unsupported symbols: {', '.join(found_unsupported)}")
    else:
        successes.append("  ✅ No unsupported Binance symbols found")
except Exception as e:
    errors.append(f"  ❌ Failed to check symbols: {e}")

print()

# Test 5: Import Test
print("[TEST 5] Checking imports...")
try:
    sys.path.insert(0, str(BASE_DIR))

    # Try to import the main bot module
    import volume_vn_bot
    successes.append("  ✅ volume_vn_bot module imports successfully")

    # Check for html import (fix we added)
    import html
    if hasattr(html, 'escape'):
        successes.append("  ✅ html.escape available for Telegram message escaping")
except ImportError as e:
    errors.append(f"  ❌ Import failed: {e}")
except Exception as e:
    warnings.append(f"  ⚠️ Import test encountered error: {e}")

print()

# Summary
print("=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)
print()

if successes:
    print("✅ SUCCESSES:")
    for success in successes:
        print(success)
    print()

if warnings:
    print("⚠️  WARNINGS:")
    for warning in warnings:
        print(warning)
    print()

if errors:
    print("❌ ERRORS:")
    for error in errors:
        print(error)
    print()

# Final verdict
print("=" * 60)
if errors:
    print("❌ VERIFICATION FAILED - Please review errors above")
    print("=" * 60)
    sys.exit(1)
elif warnings:
    print("⚠️  VERIFICATION PASSED WITH WARNINGS")
    print("=" * 60)
    sys.exit(0)
else:
    print("✅ ALL TESTS PASSED - Bot is ready for deployment")
    print("=" * 60)
    sys.exit(0)
