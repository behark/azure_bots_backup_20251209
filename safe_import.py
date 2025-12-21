#!/usr/bin/env python3
"""
Safe Import Utility for Trading Bots

Provides independent, logged imports of optional dependencies.
Allows bots to run with partial functionality if some imports fail.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Callable, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def safe_import(
    module_name: str,
    attribute: Optional[str] = None,
    fallback: Any = None,
    logger_instance: Optional[logging.Logger] = None,
    silent: bool = False,
) -> Any:
    """
    Safely import a module or attribute with detailed error logging.

    Args:
        module_name: Name of the module to import (e.g., 'notifier')
        attribute: Specific attribute to import from module (e.g., 'TelegramNotifier')
        fallback: Value to return if import fails (default: None)
        logger_instance: Logger to use for messages (default: module logger)
        silent: If True, don't log import failures (default: False)

    Returns:
        The imported module/attribute, or fallback value if import fails

    Examples:
        >>> notifier_module = safe_import('notifier')
        >>> TelegramNotifier = safe_import('notifier', 'TelegramNotifier')
        >>> get_config = safe_import('trade_config', 'get_config_manager', fallback=lambda: {})
    """
    log = logger_instance or logger

    try:
        # Import the module
        module = __import__(module_name, fromlist=[attribute] if attribute else [])

        # If no specific attribute requested, return the module
        if attribute is None:
            if not silent:
                log.debug(f"Successfully imported module: {module_name}")
            return module

        # Try to get the specific attribute
        if not hasattr(module, attribute):
            if not silent:
                log.warning(
                    f"Module '{module_name}' imported but attribute '{attribute}' not found. "
                    f"Available attributes: {dir(module)}"
                )
            return fallback

        result = getattr(module, attribute)
        if not silent:
            log.debug(f"Successfully imported {module_name}.{attribute}")
        return result

    except ImportError as e:
        if not silent:
            location = f"{module_name}.{attribute}" if attribute else module_name
            log.warning(
                f"Failed to import {location}: {e.__class__.__name__}: {str(e)}"
            )
        return fallback

    except Exception as e:
        if not silent:
            location = f"{module_name}.{attribute}" if attribute else module_name
            log.error(
                f"Unexpected error importing {location}: {e.__class__.__name__}: {str(e)}",
                exc_info=True
            )
        return fallback


def safe_import_multiple(
    imports: list[Tuple[str, Optional[str], Any]],
    logger_instance: Optional[logging.Logger] = None,
    silent: bool = False,
) -> dict[str, Any]:
    """
    Safely import multiple modules/attributes at once.

    Args:
        imports: List of (module_name, attribute, fallback) tuples
        logger_instance: Logger to use for messages
        silent: If True, don't log import failures

    Returns:
        Dictionary mapping attribute names to imported values

    Example:
        >>> imports = [
        ...     ('notifier', 'TelegramNotifier', None),
        ...     ('signal_stats', 'SignalStats', None),
        ...     ('health_monitor', 'HealthMonitor', None),
        ... ]
        >>> components = safe_import_multiple(imports)
        >>> TelegramNotifier = components['TelegramNotifier']
    """
    log = logger_instance or logger
    results = {}

    for module_name, attribute, fallback in imports:
        key = attribute if attribute else module_name
        results[key] = safe_import(
            module_name=module_name,
            attribute=attribute,
            fallback=fallback,
            logger_instance=log,
            silent=silent,
        )

    return results


def check_optional_dependencies(
    logger_instance: Optional[logging.Logger] = None
) -> dict[str, bool]:
    """
    Check availability of all optional bot dependencies.

    Args:
        logger_instance: Logger to use for messages

    Returns:
        Dictionary mapping component names to availability (True/False)
    """
    log = logger_instance or logger

    dependencies = {
        'TelegramNotifier': ('notifier', 'TelegramNotifier'),
        'SignalStats': ('signal_stats', 'SignalStats'),
        'HealthMonitor': ('health_monitor', 'HealthMonitor'),
        'RateLimiter': ('health_monitor', 'RateLimiter'),
        'TPSLCalculator': ('tp_sl_calculator', 'TPSLCalculator'),
        'TradeLevels': ('tp_sl_calculator', 'TradeLevels'),
        'get_config_manager': ('trade_config', 'get_config_manager'),
        'RateLimitHandler': ('rate_limit_handler', 'RateLimitHandler'),
    }

    availability = {}
    log.info("Checking optional dependencies:")

    for name, (module, attr) in dependencies.items():
        result = safe_import(module, attr, silent=True)
        is_available = result is not None
        availability[name] = is_available
        status = "✓" if is_available else "✗"
        log.info(f"  {status} {name}: {'available' if is_available else 'not available'}")

    return availability


def get_import_summary(
    imported_components: dict[str, Any],
    logger_instance: Optional[logging.Logger] = None
) -> None:
    """
    Log a summary of successfully imported vs failed components.

    Args:
        imported_components: Dictionary of component name -> imported value
        logger_instance: Logger to use for messages
    """
    log = logger_instance or logger

    available = [name for name, value in imported_components.items() if value is not None]
    unavailable = [name for name, value in imported_components.items() if value is None]

    total = len(imported_components)
    success_count = len(available)

    log.info(f"Import summary: {success_count}/{total} components available")

    if available:
        log.info(f"  Available: {', '.join(sorted(available))}")

    if unavailable:
        log.warning(f"  Unavailable: {', '.join(sorted(unavailable))}")
        log.warning("  Bot will run with reduced functionality")


# Convenience function for common bot imports
def import_bot_dependencies(
    logger_instance: Optional[logging.Logger] = None,
    include_telegram: bool = True,
    include_stats: bool = True,
    include_health: bool = True,
    include_tpsl: bool = True,
    include_config: bool = True,
    include_rate_limit: bool = True,
) -> dict[str, Any]:
    """
    Import all common bot dependencies with fine-grained control.

    Args:
        logger_instance: Logger to use for messages
        include_telegram: Import Telegram notifier
        include_stats: Import signal stats
        include_health: Import health monitor and rate limiter
        include_tpsl: Import TP/SL calculator
        include_config: Import config manager
        include_rate_limit: Import rate limit handler

    Returns:
        Dictionary of component name -> imported value

    Example:
        >>> deps = import_bot_dependencies(logger_instance=logger)
        >>> TelegramNotifier = deps['TelegramNotifier']
        >>> SignalStats = deps['SignalStats']
    """
    log = logger_instance or logger
    imports_list = []

    if include_telegram:
        imports_list.append(('notifier', 'TelegramNotifier', None))

    if include_stats:
        imports_list.append(('signal_stats', 'SignalStats', None))

    if include_health:
        imports_list.extend([
            ('health_monitor', 'HealthMonitor', None),
            ('health_monitor', 'RateLimiter', None),
        ])

    if include_tpsl:
        imports_list.extend([
            ('tp_sl_calculator', 'TPSLCalculator', None),
            ('tp_sl_calculator', 'TradeLevels', None),
        ])

    if include_config:
        imports_list.append(('trade_config', 'get_config_manager', None))

    if include_rate_limit:
        imports_list.append(('rate_limit_handler', 'RateLimitHandler', None))

    components = safe_import_multiple(imports_list, logger_instance=log)
    get_import_summary(components, logger_instance=log)

    return components


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    print("\n" + "="*70)
    print("Testing Safe Import Utility")
    print("="*70 + "\n")

    # Test individual imports
    print("1. Testing individual imports:")
    TelegramNotifier = safe_import('notifier', 'TelegramNotifier')
    print(f"   TelegramNotifier: {TelegramNotifier}\n")

    # Test multiple imports
    print("2. Testing multiple imports:")
    imports = [
        ('notifier', 'TelegramNotifier', None),
        ('signal_stats', 'SignalStats', None),
        ('health_monitor', 'HealthMonitor', None),
        ('nonexistent_module', 'NonExistent', None),
    ]
    results = safe_import_multiple(imports)
    print(f"   Results: {results}\n")

    # Test dependency check
    print("3. Checking all dependencies:")
    check_optional_dependencies()
    print()

    # Test convenience function
    print("4. Using convenience function:")
    deps = import_bot_dependencies()
    print(f"\n   Total components: {len(deps)}")
    print(f"   Available: {sum(1 for v in deps.values() if v is not None)}")
