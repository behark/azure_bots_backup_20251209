#!/usr/bin/env python3
"""
Bot Signal Mixin - Drop-in Upgrade for Existing Bots

This module provides a mixin class that existing bots can inherit from
to automatically use the new unified signal system with minimal changes.

Usage in existing bot:

    # 1. Add import at top of bot file
    from core.bot_signal_mixin import BotSignalMixin
    
    # 2. Add mixin to bot class
    class MyBot(BotSignalMixin):
        def __init__(self):
            # ... existing init ...
            
            # 3. Initialize the signal adapter (add this line)
            self._init_signal_adapter(
                bot_name="my_bot",
                notifier=self.notifier,
                exchange="MEXC",
            )
    
    # 4. Replace signal sending:
    # OLD: self.notifier.send_message(format_signal_message(...))
    # NEW: self._send_signal(symbol, direction, entry, sl, tp1, tp2, ...)
    
    # 5. Replace monitoring:
    # OLD: self._monitor_open_signals() with complex logic
    # NEW: self._check_signals(price_fetcher_func)

This allows bots to adopt the new system with just a few line changes!
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from .signal_adapter import SignalAdapter
from .signal_manager import Signal, SignalResult

logger = logging.getLogger(__name__)


class BotSignalMixin:
    """
    Mixin class that adds unified signal management to any bot.
    
    Provides:
    - _init_signal_adapter(): Initialize the signal system
    - _send_signal(): Send a signal with beautiful templates
    - _check_signals(): Check for TP/SL hits
    - _has_open_signal(): Check if signal exists
    - _get_performance_stats(): Get bot performance
    """
    
    # Will be set by _init_signal_adapter
    _signal_adapter: Optional[SignalAdapter] = None
    
    def _init_signal_adapter(
        self,
        bot_name: str,
        notifier: Optional[Any] = None,
        exchange: str = "MEXC",
        default_timeframe: str = "5m",
        notification_mode: str = "signal_only",
        enable_breakeven: bool = True,
    ) -> None:
        """
        Initialize the signal adapter.
        
        Call this in your bot's __init__ method.
        
        Args:
            bot_name: Name of the bot (e.g., "strat_bot")
            notifier: Telegram notifier instance
            exchange: Exchange name for messages
            default_timeframe: Default timeframe
            notification_mode: "signal_only" (default), "both", or "result_only"
            enable_breakeven: Auto-move SL to entry after profit
        """
        self._signal_adapter = SignalAdapter(
            bot_name=bot_name,
            notifier=notifier,
            exchange=exchange,
            default_timeframe=default_timeframe,
            notification_mode=notification_mode,
            enable_breakeven=enable_breakeven,
            enable_partial_exits=False,  # Keep it simple
        )
        
        logger.info("Signal adapter initialized for %s", bot_name)
    
    def _send_signal(
        self,
        symbol: str,
        direction: str,
        entry: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: float = 0.0,
        take_profit_3: float = 0.0,
        confidence: float = 0.0,
        pattern_name: Optional[str] = None,
        timeframe: Optional[str] = None,
        atr: Optional[float] = None,
        reasons: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Signal]:
        """
        Send a signal with the new unified system.
        
        Replaces:
            self.notifier.send_message(format_signal_message(...))
        
        With:
            self._send_signal(symbol, direction, entry, sl, tp1, tp2, ...)
        """
        if not self._signal_adapter:
            logger.error("Signal adapter not initialized! Call _init_signal_adapter first.")
            return None
        
        return self._signal_adapter.send_signal(
            symbol=symbol,
            direction=direction,
            entry=entry,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=take_profit_3,
            confidence=confidence,
            pattern_name=pattern_name,
            timeframe=timeframe,
            atr=atr,
            reasons=reasons,
            extra=extra,
        )
    
    def _check_signals(
        self,
        price_fetcher: Callable[[str], Optional[float]],
    ) -> List[SignalResult]:
        """
        Check all open signals for TP/SL hits.
        
        Replaces complex _monitor_open_signals() logic.
        
        Args:
            price_fetcher: Function that takes symbol and returns current price
            
        Returns:
            List of closed signals
            
        Example:
            def fetch_price(symbol):
                ticker = self.client.fetch_ticker(symbol)
                return ticker.get("last")
            
            results = self._check_signals(fetch_price)
        """
        if not self._signal_adapter:
            return []
        
        return self._signal_adapter.check_and_notify(price_fetcher)
    
    def _has_open_signal(self, symbol: str, direction: Optional[str] = None) -> bool:
        """Check if there's already an open signal for symbol/direction."""
        if not self._signal_adapter:
            return False
        return self._signal_adapter.has_open_signal(symbol, direction)
    
    def _get_open_signals(self) -> Dict[str, Signal]:
        """Get all open signals."""
        if not self._signal_adapter:
            return {}
        return self._signal_adapter.get_open_signals()
    
    def _get_performance_stats(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self._signal_adapter:
            return {}
        return self._signal_adapter.get_performance_stats(symbol)
    
    def _update_trailing_stop(self, signal_id: str, new_stop: float) -> bool:
        """Update trailing stop for a signal."""
        if not self._signal_adapter:
            return False
        return self._signal_adapter.update_trailing_stop(signal_id, new_stop)
    
    def _move_to_breakeven(self, signal_id: str) -> bool:
        """Move stop loss to entry (breakeven)."""
        if not self._signal_adapter:
            return False
        return self._signal_adapter.move_to_breakeven(signal_id)


# =============================================================================
# HELPER FUNCTIONS FOR GRADUAL MIGRATION
# =============================================================================

def create_price_fetcher(client: Any, symbol_resolver: Optional[Callable] = None):
    """
    Create a price fetcher function for a CCXT client.
    
    Args:
        client: CCXT client instance
        symbol_resolver: Optional function to resolve symbol format
        
    Returns:
        Function that takes symbol and returns current price
    """
    def fetch_price(symbol: str) -> Optional[float]:
        try:
            # Resolve symbol if needed
            resolved = symbol_resolver(symbol) if symbol_resolver else symbol
            ticker = client.fetch_ticker(resolved)
            price = ticker.get("last")
            if price is None:
                price = ticker.get("close")
            return float(price) if price else None
        except Exception as e:
            logger.debug("Failed to fetch price for %s: %s", symbol, e)
            return None
    
    return fetch_price


def migrate_existing_signals(
    old_state_file: str,
    bot_name: str,
) -> int:
    """
    Migrate signals from old state file to new SignalManager.
    
    Args:
        old_state_file: Path to old state file
        bot_name: Name of the bot
        
    Returns:
        Number of signals migrated
    """
    from pathlib import Path
    from .signal_adapter import migrate_state_file
    
    return migrate_state_file(Path(old_state_file), bot_name)

