#!/usr/bin/env python3
"""
Signal Adapter - Easy Migration Layer for Existing Bots

This module provides adapter functions that allow existing bots to use the new
unified SignalManager and enhanced templates with minimal code changes.

Usage (in existing bot):

    # 1. Import the adapter
    from core.signal_adapter import SignalAdapter
    
    # 2. Create adapter (once during bot init)
    self.signal_adapter = SignalAdapter(
        bot_name="strat_bot",
        notifier=self.notifier,
    )
    
    # 3. Replace signal sending with adapter
    # OLD: self.notifier.send_message(format_signal_message(...))
    # NEW:
    self.signal_adapter.send_signal(
        symbol=symbol,
        direction=direction,
        entry=entry,
        stop_loss=sl,
        take_profit_1=tp1,
        take_profit_2=tp2,
        pattern_name="2-1-2 Reversal",
        timeframe="5m",
    )
    
    # 4. Replace monitoring with adapter
    # OLD: self._monitor_open_signals() with manual TP/SL checks
    # NEW:
    def fetch_price(symbol):
        return self.client.fetch_ticker(symbol)["last"]
    
    self.signal_adapter.check_and_notify(fetch_price)
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .signal_manager import (
    Signal,
    SignalManager,
    SignalResult,
    SignalState,
    CloseReason,
    get_signal_manager,
)
from .enhanced_templates import (
    SignalMessageBuilder,
    ResultMessageBuilder,
    PartialExitMessageBuilder,
    build_streak_alert,
)
from .signal_analytics import SignalAnalytics

logger = logging.getLogger(__name__)


class SignalAdapter:
    """
    Adapter that bridges existing bot code with the new unified signal system.
    
    Provides simple methods that:
    - Create signals with the new Signal class
    - Send beautiful enhanced notifications
    - Track signals with full lifecycle support
    - Monitor for TP/SL hits automatically
    - Provide performance insights
    """
    
    def __init__(
        self,
        bot_name: str,
        notifier: Optional[Any] = None,
        exchange: str = "MEXC",
        default_timeframe: str = "5m",
        enable_breakeven: bool = True,
        enable_partial_exits: bool = False,  # Disabled by default for cleaner flow
        enable_streak_alerts: bool = True,
        streak_alert_threshold: int = 5,
        notification_mode: str = "signal_only",  # "signal_only", "result_only", "both"
    ):
        """
        Initialize the signal adapter.
        
        Args:
            bot_name: Name of the bot (used for state file naming)
            notifier: Telegram notifier with send_message(str) method
            exchange: Default exchange name
            default_timeframe: Default timeframe
            enable_breakeven: Move SL to entry after profit threshold
            enable_partial_exits: Close partial position at each TP
            enable_streak_alerts: Send alerts on win streaks
            streak_alert_threshold: Streak length to trigger alert
            notification_mode: "signal_only" (default), "result_only", or "both"
        """
        self.bot_name = bot_name
        self.notifier = notifier
        self.exchange = exchange
        self.default_timeframe = default_timeframe
        self.enable_streak_alerts = enable_streak_alerts
        self.streak_alert_threshold = streak_alert_threshold
        self.notification_mode = notification_mode
        
        # Initialize signal manager with settings
        self.manager = get_signal_manager(
            bot_name,
            enable_breakeven=enable_breakeven,
            partial_tp1_size=0.5 if enable_partial_exits else 0.0,
            partial_tp2_size=0.3 if enable_partial_exits else 0.0,
        )
        
        # Cache for analytics (lazy loaded)
        self._analytics: Optional[SignalAnalytics] = None
        
        logger.info("SignalAdapter initialized for %s (mode: %s)", bot_name, notification_mode)
    
    @property
    def analytics(self) -> SignalAnalytics:
        """Get analytics instance (lazy loaded)."""
        if self._analytics is None:
            self._analytics = SignalAnalytics(self.manager.history)
        return self._analytics
    
    def refresh_analytics(self) -> None:
        """Refresh analytics with latest history."""
        self._analytics = SignalAnalytics(self.manager.history)
    
    def send_signal(
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
        Create a signal, track it, and send a beautiful notification.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            direction: "LONG"/"SHORT" or "BULLISH"/"BEARISH"
            entry: Entry price
            stop_loss: Stop loss price
            take_profit_1: First take profit target
            take_profit_2: Second take profit target (optional)
            take_profit_3: Third take profit target (optional)
            confidence: Signal confidence 0-100 (optional)
            pattern_name: Pattern/setup name (optional)
            timeframe: Timeframe (default: self.default_timeframe)
            atr: ATR value for volatility context (optional)
            reasons: List of reasons for the signal (optional)
            extra: Extra data to store with signal (optional)
            
        Returns:
            Created Signal object, or None if failed
        """
        # Create signal
        signal = Signal(
            symbol=symbol,
            direction=direction,
            entry=entry,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2 or 0.0,
            take_profit_3=take_profit_3 or 0.0,
            confidence=confidence,
            pattern_name=pattern_name or "",
            exchange=self.exchange,
            timeframe=timeframe or self.default_timeframe,
            extra=extra or {},
        )
        
        # Add to manager
        self.manager.add_signal(signal)
        
        # Get performance stats for context
        stats = self.manager.get_performance_stats()
        symbol_stats = self.manager.get_symbol_stats(symbol)
        
        # Get last result for this symbol (to show in "signal_only" mode)
        last_result = self.manager.get_last_result(symbol)
        
        # Build beautiful message with last result embedded
        message = SignalMessageBuilder(
            bot_name=self.bot_name,
            symbol=symbol,
            direction=direction,
            entry=entry,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2 if take_profit_2 else None,
            take_profit_3=take_profit_3 if take_profit_3 else None,
            confidence=confidence,
            pattern_name=pattern_name,
            stats=stats,
            symbol_stats=symbol_stats,
            last_result=last_result,  # Shows previous signal result inline
            exchange=self.exchange,
            timeframe=timeframe or self.default_timeframe,
            atr=atr,
            signal_id=signal.signal_id,
            reasons=reasons,
        ).build()
        
        # Send notification (in signal_only mode, this is the only notification)
        if self.notification_mode in ("signal_only", "both"):
            self._send_message(message)
        
        logger.info("Signal sent: %s %s %s", symbol, direction, signal.signal_id)
        
        return signal
    
    def check_and_notify(
        self,
        price_fetcher: Callable[[str], Optional[float]],
    ) -> List[SignalResult]:
        """
        Check all open signals for TP/SL hits and send notifications.
        
        Args:
            price_fetcher: Function that takes symbol and returns current price
            
        Returns:
            List of closed signals
        """
        results = self.manager.check_signals(price_fetcher)
        
        for result in results:
            self._notify_result(result)
        
        return results
    
    def _notify_result(self, result: SignalResult) -> None:
        """Send notification for a signal result (respects notification_mode)."""
        signal = result.signal
        
        # In "signal_only" mode, we don't send result notifications
        # The result will be shown in the NEXT signal's history line
        if self.notification_mode == "signal_only":
            logger.info("Result recorded (no notification in signal_only mode): %s %s %.2f%%", 
                       signal.symbol, result.close_reason.value, result.pnl_pct)
            return
        
        # Get updated stats
        stats = self.manager.get_performance_stats()
        
        # Build result message
        message = ResultMessageBuilder(
            symbol=signal.symbol,
            direction=signal.direction,
            result=result.close_reason.value,
            entry=signal.entry,
            exit_price=result.exit_price,
            pnl_pct=result.pnl_pct,
            stop_loss=signal.stop_loss,
            take_profit_1=signal.take_profit_1,
            take_profit_2=signal.take_profit_2,
            stats=stats,
            duration_str=result.duration_str,
            mfe=result.mfe,
            mae=result.mae,
        ).build()
        
        self._send_message(message)
        
        # Check for streak alert (still send these even in signal_only mode for milestones)
        if self.enable_streak_alerts:
            streak = stats.get("current_streak", 0)
            if streak >= self.streak_alert_threshold and streak % self.streak_alert_threshold == 0:
                # Calculate streak P&L
                streak_pnl = self._calculate_streak_pnl(streak)
                streak_msg = build_streak_alert(self.bot_name, streak, streak_pnl)
                self._send_message(streak_msg)
        
        logger.info("Result notified: %s %s %.2f%%", 
                   signal.symbol, result.close_reason.value, result.pnl_pct)
    
    def _calculate_streak_pnl(self, streak: int) -> float:
        """Calculate P&L for the current streak."""
        if not self.manager.history:
            return 0.0
        
        total_pnl = 0.0
        for trade in self.manager.history[-streak:]:
            total_pnl += trade.get("realized_pnl_pct", 0)
        
        return total_pnl
    
    def notify_partial_exit(
        self,
        signal: Signal,
        result: str,
        exit_price: float,
        pnl_pct: float,
        moved_to_breakeven: bool = False,
    ) -> None:
        """Send notification for partial position exit."""
        message = PartialExitMessageBuilder(
            symbol=signal.symbol,
            direction=signal.direction,
            result=result,
            exit_price=exit_price,
            pnl_pct=pnl_pct,
            position_remaining=signal.position_remaining,
            moved_to_breakeven=moved_to_breakeven,
            new_stop_loss=signal.current_stop_loss if moved_to_breakeven else None,
        ).build()
        
        self._send_message(message)
    
    def update_trailing_stop(self, signal_id: str, new_stop: float) -> bool:
        """
        Update trailing stop for a signal.
        
        Args:
            signal_id: Signal ID
            new_stop: New stop loss price
            
        Returns:
            True if updated, False if not (stop wasn't better)
        """
        return self.manager.update_trailing_stop(signal_id, new_stop)
    
    def move_to_breakeven(self, signal_id: str) -> bool:
        """
        Move stop loss to entry (breakeven).
        
        Args:
            signal_id: Signal ID
            
        Returns:
            True if moved, False if already at breakeven
        """
        return self.manager.move_to_breakeven(signal_id)
    
    def has_open_signal(self, symbol: str, direction: Optional[str] = None) -> bool:
        """Check if there's an open signal for symbol/direction."""
        return self.manager.has_open_signal(symbol, direction)
    
    def get_open_signals(self) -> Dict[str, Signal]:
        """Get all open signals."""
        return self.manager.get_open_signals()
    
    def get_signal(self, signal_id: str) -> Optional[Signal]:
        """Get a specific signal by ID."""
        return self.manager.get_signal(signal_id)
    
    def get_performance_stats(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.manager.get_performance_stats(symbol)
    
    def get_insights_report(self) -> str:
        """Get a full insights report for Telegram."""
        self.refresh_analytics()
        return self.analytics.generate_insights_report()
    
    def _send_message(self, message: str) -> bool:
        """Send message via notifier."""
        if self.notifier:
            try:
                return self.notifier.send_message(message)
            except Exception as e:
                logger.error("Failed to send message: %s", e)
                return False
        else:
            logger.info("Message (no notifier):\n%s", 
                       message.replace("<b>", "").replace("</b>", ""))
            return True


# =============================================================================
# LEGACY ADAPTER FUNCTIONS
# =============================================================================

def create_adapter_for_bot(
    bot_name: str,
    notifier: Optional[Any] = None,
    **kwargs
) -> SignalAdapter:
    """
    Factory function to create an adapter for a specific bot.
    
    This is the main entry point for migrating existing bots.
    
    Args:
        bot_name: Name of the bot
        notifier: Telegram notifier instance
        **kwargs: Additional SignalAdapter options
        
    Returns:
        Configured SignalAdapter
    """
    return SignalAdapter(bot_name=bot_name, notifier=notifier, **kwargs)


def migrate_state_file(
    old_state_file: Path,
    bot_name: str,
) -> int:
    """
    Migrate signals from old state file format to new SignalManager format.
    
    Args:
        old_state_file: Path to old state file (e.g., strat_state.json)
        bot_name: Name of the bot
        
    Returns:
        Number of signals migrated
    """
    import json
    
    if not old_state_file.exists():
        return 0
    
    try:
        with open(old_state_file, 'r') as f:
            old_data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return 0
    
    manager = get_signal_manager(bot_name)
    migrated = 0
    
    # Migrate open signals
    open_signals = old_data.get("open_signals", {})
    for signal_id, data in open_signals.items():
        try:
            signal = Signal(
                signal_id=signal_id,
                symbol=data.get("symbol", ""),
                direction=data.get("direction", "LONG"),
                entry=float(data.get("entry", 0)),
                stop_loss=float(data.get("stop_loss", 0)),
                take_profit_1=float(data.get("take_profit_1", 0)),
                take_profit_2=float(data.get("take_profit_2", 0)),
                created_at=data.get("created_at", ""),
                exchange=data.get("exchange", "MEXC"),
                timeframe=data.get("timeframe", "5m"),
            )
            signal.state = SignalState.ACTIVE.value
            manager.signals[signal_id] = signal
            migrated += 1
        except (TypeError, ValueError, KeyError):
            pass
    
    if migrated > 0:
        manager._save_state()
        logger.info("Migrated %d signals from %s", migrated, old_state_file)
    
    return migrated


if __name__ == "__main__":
    # Example usage
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Create adapter
    adapter = SignalAdapter(
        bot_name="test_bot",
        notifier=None,  # Would be TelegramNotifier in real usage
        exchange="MEXC",
    )
    
    # Send a test signal
    signal = adapter.send_signal(
        symbol="BTC/USDT",
        direction="LONG",
        entry=42000.0,
        stop_loss=41500.0,
        take_profit_1=42500.0,
        take_profit_2=43000.0,
        confidence=75,
        pattern_name="2-1-2 Reversal",
        reasons=["Strong momentum", "Volume confirmation"],
    )
    
    print(f"\nCreated signal: {signal.signal_id}")
    print(f"Open signals: {len(adapter.get_open_signals())}")
    
    # Check for TP/SL (would use real price fetcher)
    def mock_price_fetcher(symbol: str) -> float:
        return 42600.0  # Simulates TP1 hit
    
    results = adapter.check_and_notify(mock_price_fetcher)
    print(f"Results: {len(results)}")

