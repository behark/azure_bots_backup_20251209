#!/usr/bin/env python3
"""
Shared Utilities Usage Checker v2.0
Analyzes which bots actually use the shared utility files.
Shows both imports AND actual usage in code.

Fixed: Now detects both import patterns:
  - safe_import('module', 'Class')
  - safe_import_multiple([('module', 'Class', None)])
  - Direct imports: from module import Class
"""

import re
from pathlib import Path
from typing import Dict, List

# Shared utilities to check
SHARED_UTILS = {
    "notifier.py": {
        "module": "notifier",
        "classes": ["TelegramNotifier", "generate_signal_id"],
        "usage_patterns": [
            r"TelegramNotifier\s*\(",
            r"\.send_message\s*\(",
            r"\.send_signal\s*\(",
            r"\.save_signal_to_json\s*\(",
            r"generate_signal_id\s*\(",
            r"self\.notifier\.",
        ]
    },
    "signal_stats.py": {
        "module": "signal_stats",
        "classes": ["SignalStats"],
        "usage_patterns": [
            r"SignalStats\s*\(",
            r"\.record_open\s*\(",
            r"\.record_close\s*\(",
            r"\.get_summary\s*\(",
            r"self\.stats\.",
        ]
    },
    "health_monitor.py": {
        "module": "health_monitor",
        "classes": ["HealthMonitor", "RateLimiter"],
        "usage_patterns": [
            r"HealthMonitor\s*\(",
            r"\.heartbeat\s*\(",
            r"\.send_heartbeat\s*\(",
            r"\.send_startup\s*\(",
            r"\.send_shutdown\s*\(",
            r"self\.health\.",
        ]
    },
    "tp_sl_calculator.py": {
        "module": "tp_sl_calculator",
        "classes": ["TPSLCalculator", "TradeLevels", "calculate_atr", "quick_calculate"],
        "usage_patterns": [
            r"TPSLCalculator\s*\(",
            r"TradeLevels\s*\(",
            r"\.calculate\s*\([^)]*entry",
            r"calculate_atr\s*\(",
            r"quick_calculate\s*\(",
        ]
    },
    "trade_config.py": {
        "module": "trade_config",
        "classes": ["get_config_manager", "TradeConfigManager", "RiskConfig"],
        "usage_patterns": [
            r"get_config_manager\s*\(",
            r"TradeConfigManager\s*\(",
            r"\.get_bot_config\s*\(",
            r"\.get_effective_risk\s*\(",
            r"\.get_symbol_config\s*\(",
        ]
    },
    "rate_limit_handler.py": {
        "module": "rate_limit_handler",
        "classes": ["RateLimitHandler", "RateLimitedExchange", "safe_api_call"],
        "usage_patterns": [
            r"RateLimitHandler\s*\(",
            r"RateLimitedExchange\s*\(",
            r"safe_api_call\s*\(",
            r"handler\.execute\s*\(",
        ]
    },
    "file_lock.py": {
        "module": "file_lock",
        "classes": ["file_lock", "safe_read_json", "safe_write_json", "SafeStateManager"],
        "usage_patterns": [
            r"with\s+file_lock\s*\(",
            r"safe_read_json\s*\(",
            r"safe_write_json\s*\(",
            r"SafeStateManager\s*\(",
        ]
    },
    "safe_import.py": {
        "module": "safe_import",
        "classes": ["safe_import", "safe_import_multiple", "import_bot_dependencies"],
        "usage_patterns": [
            r"safe_import\s*\(",
            r"safe_import_multiple\s*\(",
            r"import_bot_dependencies\s*\(",
        ]
    },
    "global_config.json": {
        "module": "global_config",
        "classes": ["global_config"],
        "usage_patterns": [
            r"global_config\.json",
            r"['\"]global_config['\"]",
        ]
    },
}


def find_bot_folders(base_path: Path) -> List[Path]:
    """Find all bot folders."""
    bot_folders = []
    for item in base_path.iterdir():
        if item.is_dir() and item.name.endswith("_bot"):
            bot_folders.append(item)
    return sorted(bot_folders)


def get_bot_main_file(bot_folder: Path) -> Path | None:
    """Find the main bot Python file."""
    candidates = []

    # Check for any .py file that's not __init__.py or test
    for f in bot_folder.glob("*.py"):
        if f.name != "__init__.py" and "test" not in f.name.lower() and "verify" not in f.name.lower():
            candidates.append(f)

    # Prefer files with 'bot' in name
    for candidate in candidates:
        if "bot" in candidate.name.lower():
            return candidate

    return candidates[0] if candidates else None


def check_imports(content: str, module: str, classes: List[str]) -> Dict[str, bool]:
    """
    Check which imports are present in the file.
    Detects three patterns:
    1. Direct: from module import Class
    2. Single safe_import: Class = safe_import('module', 'Class')
    3. Batch safe_import_multiple: ('module', 'Class', None) in list
    """
    found = {}

    for cls in classes:
        is_imported = False

        # Pattern 1: Direct import
        # from signal_stats import SignalStats
        # from notifier import TelegramNotifier, generate_signal_id
        direct_patterns = [
            rf"from\s+{module}\s+import\s+[^#\n]*\b{cls}\b",
            rf"import\s+{module}",
        ]

        # Pattern 2: Single safe_import
        # SignalStats = safe_import('signal_stats', 'SignalStats')
        single_patterns = [
            rf"{cls}\s*=\s*safe_import\s*\(\s*['\"][\w.]*{module}['\"]",
            rf"safe_import\s*\(\s*['\"][^'\"]*{module}['\"],\s*['\"]?{cls}['\"]?\s*\)",
        ]

        # Pattern 3: Batch safe_import_multiple
        # ('signal_stats', 'SignalStats', None),
        # ("notifier", "TelegramNotifier", None),
        batch_patterns = [
            rf"\(\s*['\"][\w.]*{module}['\"],\s*['\"]?{cls}['\"]?\s*,",
            rf"['\"][\w.]*{module}['\"],\s*['\"]?{cls}['\"]",
        ]

        # Pattern 4: Components dict access after safe_import_multiple
        # SignalStats = components['SignalStats']
        components_patterns = [
            rf"{cls}\s*=\s*components\s*\[\s*['\"]?{cls}['\"]?\s*\]",
        ]

        all_patterns = direct_patterns + single_patterns + batch_patterns + components_patterns

        for pattern in all_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                is_imported = True
                break

        found[cls] = is_imported

    return found


def check_usage(content: str, patterns: List[str]) -> List[str]:
    """Check which usage patterns are found in the file."""
    found = []
    for pattern in patterns:
        if re.search(pattern, content):
            found.append(pattern)
    return found


def analyze_bot(bot_folder: Path) -> Dict:
    """Analyze a single bot's usage of shared utilities."""
    result = {
        "name": bot_folder.name,
        "main_file": None,
        "utilities": {}
    }

    main_file = get_bot_main_file(bot_folder)
    if not main_file:
        result["error"] = "No main file found"
        return result

    result["main_file"] = main_file.name

    try:
        content = main_file.read_text()
    except Exception as e:
        result["error"] = f"Cannot read file: {e}"
        return result

    for util_name, util_info in SHARED_UTILS.items():
        imports_found = check_imports(content, util_info["module"], util_info["classes"])
        any_imported = any(imports_found.values())
        imported_classes = [cls for cls, imp in imports_found.items() if imp]

        usage_found = check_usage(content, util_info["usage_patterns"])

        result["utilities"][util_name] = {
            "imported": any_imported,
            "imports_found": imported_classes,
            "actually_used": len(usage_found) > 0,
            "usage_patterns_found": len(usage_found),
            "usage_details": usage_found[:3],  # First 3 for brevity
        }

    return result


def print_report(results: List[Dict]):
    """Print a formatted report."""
    print("=" * 90)
    print("SHARED UTILITIES USAGE REPORT v2.0")
    print("=" * 90)
    print()

    # Summary table header
    utils_short = {
        "notifier.py": "Notifier",
        "signal_stats.py": "Stats",
        "health_monitor.py": "Health",
        "tp_sl_calculator.py": "TP/SL",
        "trade_config.py": "Config",
        "rate_limit_handler.py": "RateLimit",
        "file_lock.py": "FileLock",
        "safe_import.py": "SafeImp",
        "global_config.json": "GlobCfg",
    }

    # Print header
    print(f"{'Bot':<18}", end="")
    for short_name in utils_short.values():
        print(f"{short_name:^9}", end="")
    print()
    print("-" * 18 + "-" * (9 * len(utils_short)))

    # Track totals
    totals_imported = {u: 0 for u in SHARED_UTILS}
    totals_used = {u: 0 for u in SHARED_UTILS}

    for result in results:
        bot_name = result["name"].replace("_bot", "")[:16]
        print(f"{bot_name:<18}", end="")

        if "error" in result:
            print(f"ERROR: {result['error']}")
            continue

        for util_name in SHARED_UTILS:
            util_data = result["utilities"].get(util_name, {})
            imported = util_data.get("imported", False)
            used = util_data.get("actually_used", False)

            if imported:
                totals_imported[util_name] += 1
            if used:
                totals_used[util_name] += 1

            if imported and used:
                symbol = "✅"  # Imported and used
            elif imported and not used:
                symbol = "📦"  # Imported but not used
            elif not imported and used:
                symbol = "🔧"  # Uses patterns but import not detected (internal?)
            else:
                symbol = "❌"  # Not imported, not used

            print(f"{symbol:^9}", end="")
        print()

    # Print totals
    print("-" * 18 + "-" * (9 * len(utils_short)))
    print(f"{'IMPORTED':<18}", end="")
    for util_name in SHARED_UTILS:
        print(f"{totals_imported[util_name]:^9}", end="")
    print()

    print(f"{'USED':<18}", end="")
    for util_name in SHARED_UTILS:
        print(f"{totals_used[util_name]:^9}", end="")
    print()

    # Legend
    print()
    print("LEGEND:")
    print("  ✅ = Imported AND used (perfect!)")
    print("  📦 = Imported but NOT used (dead import)")
    print("  🔧 = Usage detected but import not found (might be internal)")
    print("  ❌ = Not imported, not used")
    print()

    # Detailed findings
    print("=" * 90)
    print("DETAILED FINDINGS")
    print("=" * 90)

    # Find unused imports
    print("\n📦 DEAD IMPORTS (imported but never used):")
    found_unused = False
    for result in results:
        if "error" in result:
            continue
        for util_name, util_data in result["utilities"].items():
            if util_data.get("imported") and not util_data.get("actually_used"):
                classes = ", ".join(util_data.get("imports_found", []))
                print(f"  - {result['name']}: imports {util_name} ({classes}) but doesn't use it")
                found_unused = True
    if not found_unused:
        print("  ✅ None found! All imports are used.")

    # Find utilities not used by any bot
    print("\n❌ UTILITIES NOT USED BY ANY BOT:")
    found_unused_util = False
    for util_name in SHARED_UTILS:
        if totals_used[util_name] == 0:
            print(f"  - {util_name}")
            found_unused_util = True
    if not found_unused_util:
        print("  ✅ All utilities are used by at least one bot!")

    # Find bots not using key utilities
    print("\n⚠️  BOTS MISSING KEY UTILITIES:")
    key_utils = ["notifier.py", "signal_stats.py", "health_monitor.py"]
    for result in results:
        if "error" in result:
            continue
        missing = []
        for util in key_utils:
            if not result["utilities"].get(util, {}).get("actually_used"):
                missing.append(util.replace(".py", ""))
        if missing:
            print(f"  - {result['name']}: missing {', '.join(missing)}")

    # Summary stats
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    total_bots = len([r for r in results if "error" not in r])

    print(f"\n📊 Coverage by utility:")
    for util_name, short in utils_short.items():
        imported = totals_imported[util_name]
        used = totals_used[util_name]
        pct = (used / total_bots * 100) if total_bots > 0 else 0
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        status = "✅" if pct >= 80 else "⚠️" if pct >= 50 else "❌"
        print(f"  {short:<12} {bar} {pct:5.1f}% ({used}/{total_bots} bots) {status}")

    print()


def main():
    base_path = Path(__file__).parent

    print(f"Scanning: {base_path}")
    print()

    bot_folders = find_bot_folders(base_path)
    print(f"Found {len(bot_folders)} bot folders:")
    for bf in bot_folders:
        print(f"  - {bf.name}")
    print()

    results = []
    for bot_folder in bot_folders:
        result = analyze_bot(bot_folder)
        results.append(result)

    print_report(results)


if __name__ == "__main__":
    main()
