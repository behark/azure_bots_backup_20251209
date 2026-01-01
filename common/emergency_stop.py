#!/usr/bin/env python3
"""
Emergency Stop Loss and Drawdown Protection Module

This module provides critical risk management features:
1. Emergency stop loss (max loss per trade)
2. Portfolio drawdown protection (max portfolio loss)
3. Automatic bot shutdown on critical losses

Usage:
    from common.emergency_stop import EmergencyStopLoss, DrawdownProtection
    
    # In bot initialization
    emergency_stop = EmergencyStopLoss(max_loss_percent=5.0)
    drawdown_protection = DrawdownProtection(max_drawdown_percent=25.0)
    
    # In monitoring loop
    if emergency_stop.should_close(signal, current_price):
        close_signal(signal_id, current_price, "EMERGENCY_SL")
    
    if drawdown_protection.should_stop_trading(total_pnl):
        logger.critical("Drawdown protection triggered - stopping trading")
        break
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class EmergencyStopLoss:
    """
    Emergency stop loss to prevent catastrophic losses on individual trades.
    
    This is a last-resort protection that triggers when a trade exceeds
    the maximum acceptable loss, regardless of the strategy's normal stop loss.
    """
    
    def __init__(self, max_loss_percent: float = 5.0):
        """
        Initialize emergency stop loss.
        
        Args:
            max_loss_percent: Maximum loss percentage per trade (default 5%)
        """
        self.max_loss_percent = max_loss_percent
        logger.info(f"Emergency stop loss initialized: {max_loss_percent}% max loss per trade")
    
    def should_close(self, signal: Dict[str, Any], current_price: float) -> bool:
        """
        Check if emergency stop should trigger for a signal.
        
        Args:
            signal: Signal dictionary with entry, direction, etc.
            current_price: Current market price
            
        Returns:
            True if emergency stop should trigger, False otherwise
        """
        try:
            entry = float(signal.get('entry', 0))
            direction = str(signal.get('direction', '')).upper()
            
            if entry == 0 or not direction:
                return False
            
            # Calculate current loss percentage
            if direction in ['LONG', 'BULLISH', 'BUY']:
                loss_pct = ((entry - current_price) / entry) * 100
            elif direction in ['SHORT', 'BEARISH', 'SELL']:
                loss_pct = ((current_price - entry) / entry) * 100
            else:
                return False
            
            # Check if loss exceeds emergency threshold
            if loss_pct >= self.max_loss_percent:
                logger.critical(
                    "🚨 EMERGENCY STOP TRIGGERED: %.2f%% loss on %s (max %.2f%%)",
                    loss_pct, signal.get('symbol', 'UNKNOWN'), self.max_loss_percent
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking emergency stop: {e}")
            return False
    
    def get_loss_percent(self, signal: Dict[str, Any], current_price: float) -> float:
        """
        Calculate current loss percentage for a signal.
        
        Args:
            signal: Signal dictionary
            current_price: Current market price
            
        Returns:
            Loss percentage (positive number means loss)
        """
        try:
            entry = float(signal.get('entry', 0))
            direction = str(signal.get('direction', '')).upper()
            
            if entry == 0:
                return 0.0
            
            if direction in ['LONG', 'BULLISH', 'BUY']:
                return ((entry - current_price) / entry) * 100
            elif direction in ['SHORT', 'BEARISH', 'SELL']:
                return ((current_price - entry) / entry) * 100
            
            return 0.0
            
        except Exception:
            return 0.0


class DrawdownProtection:
    """
    Portfolio-level drawdown protection to prevent catastrophic losses.
    
    Monitors total portfolio P&L and stops trading when drawdown
    exceeds the maximum acceptable threshold.
    """
    
    def __init__(self, max_drawdown_percent: float = 25.0):
        """
        Initialize drawdown protection.
        
        Args:
            max_drawdown_percent: Maximum portfolio drawdown (default 25%)
        """
        self.max_drawdown_percent = max_drawdown_percent
        self.peak_equity = 100.0  # Start at 100%
        self.is_active = False
        self.triggered_at: Optional[datetime] = None
        
        logger.info(f"Drawdown protection initialized: {max_drawdown_percent}% max drawdown")
    
    def update(self, current_pnl: float) -> bool:
        """
        Update equity tracking and check if drawdown limit exceeded.
        
        Args:
            current_pnl: Current total P&L percentage
            
        Returns:
            True if drawdown protection triggered, False otherwise
        """
        try:
            # Calculate current equity (start at 100%)
            current_equity = 100.0 + current_pnl
            
            # Update peak equity
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
                # Reset protection if we've recovered
                if self.is_active:
                    logger.info("✓ Drawdown protection reset - new equity peak: %.2f%%", current_equity)
                    self.is_active = False
                    self.triggered_at = None
            
            # Calculate drawdown from peak
            if self.peak_equity > 0:
                drawdown = ((self.peak_equity - current_equity) / self.peak_equity) * 100
            else:
                drawdown = 0.0
            
            # Check if drawdown exceeds threshold
            if drawdown >= self.max_drawdown_percent and not self.is_active:
                self.is_active = True
                self.triggered_at = datetime.now(timezone.utc)
                
                logger.critical(
                    "🛑 DRAWDOWN PROTECTION TRIGGERED: %.2f%% drawdown (max %.2f%%)",
                    drawdown, self.max_drawdown_percent
                )
                logger.critical(
                    "Peak equity: %.2f%%, Current equity: %.2f%%",
                    self.peak_equity, current_equity
                )
                logger.critical("🚨 STOPPING ALL NEW SIGNALS")
                
                return True
            
            # Log warning if approaching limit
            if drawdown >= self.max_drawdown_percent * 0.8 and not self.is_active:
                logger.warning(
                    "⚠️ Drawdown approaching limit: %.2f%% (limit %.2f%%)",
                    drawdown, self.max_drawdown_percent
                )
            
            return self.is_active
            
        except Exception as e:
            logger.error(f"Error updating drawdown protection: {e}")
            return self.is_active
    
    def should_stop_trading(self, current_pnl: float) -> bool:
        """
        Check if trading should be stopped due to drawdown.
        
        Args:
            current_pnl: Current total P&L percentage
            
        Returns:
            True if trading should stop, False otherwise
        """
        return self.update(current_pnl)
    
    def get_current_drawdown(self, current_pnl: float) -> float:
        """
        Calculate current drawdown percentage.
        
        Args:
            current_pnl: Current total P&L percentage
            
        Returns:
            Current drawdown percentage
        """
        try:
            current_equity = 100.0 + current_pnl
            if self.peak_equity > 0:
                return ((self.peak_equity - current_equity) / self.peak_equity) * 100
            return 0.0
        except Exception:
            return 0.0
    
    def reset(self):
        """Reset drawdown protection (use with caution)."""
        self.is_active = False
        self.triggered_at = None
        logger.warning("Drawdown protection manually reset")


class BreakevenStop:
    """
    Breakeven stop loss manager.
    
    Moves stop loss to breakeven (entry price) after a certain profit
    threshold is reached, protecting gains.
    """
    
    def __init__(self, trigger_profit_percent: float = 1.5):
        """
        Initialize breakeven stop manager.
        
        Args:
            trigger_profit_percent: Profit % to trigger breakeven (default 1.5%)
        """
        self.trigger_profit_percent = trigger_profit_percent
        logger.info(f"Breakeven stop initialized: triggers at {trigger_profit_percent}% profit")
    
    def should_move_to_breakeven(self, signal: Dict[str, Any], current_price: float) -> bool:
        """
        Check if stop should be moved to breakeven.
        
        Args:
            signal: Signal dictionary
            current_price: Current market price
            
        Returns:
            True if stop should move to breakeven, False otherwise
        """
        try:
            entry = float(signal.get('entry', 0))
            current_sl = float(signal.get('stop_loss', 0))
            direction = str(signal.get('direction', '')).upper()
            
            if entry == 0 or current_sl == 0:
                return False
            
            # Calculate current profit
            if direction in ['LONG', 'BULLISH', 'BUY']:
                profit_pct = ((current_price - entry) / entry) * 100
                # For LONG, breakeven means SL should be at or above entry
                already_at_breakeven = current_sl >= entry
            elif direction in ['SHORT', 'BEARISH', 'SELL']:
                profit_pct = ((entry - current_price) / entry) * 100
                # For SHORT, breakeven means SL should be at or below entry
                already_at_breakeven = current_sl <= entry
            else:
                return False
            
            # Check if profit threshold reached and not already at breakeven
            if profit_pct >= self.trigger_profit_percent and not already_at_breakeven:
                logger.info(
                    "✓ Moving to breakeven: %s at %.2f%% profit (trigger: %.2f%%)",
                    signal.get('symbol', 'UNKNOWN'), profit_pct, self.trigger_profit_percent
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking breakeven condition: {e}")
            return False
    
    def get_breakeven_stop(self, signal: Dict[str, Any]) -> Optional[float]:
        """
        Get the breakeven stop loss price.
        
        Args:
            signal: Signal dictionary
            
        Returns:
            Breakeven stop loss price (entry price)
        """
        try:
            return float(signal.get('entry', 0))
        except Exception:
            return None


# Convenience function to check all protections
def check_all_protections(
    signal: Dict[str, Any],
    current_price: float,
    total_pnl: float,
    emergency_stop: Optional[EmergencyStopLoss] = None,
    drawdown_protection: Optional[DrawdownProtection] = None,
    breakeven_stop: Optional[BreakevenStop] = None
) -> Dict[str, Any]:
    """
    Check all protection mechanisms at once.
    
    Args:
        signal: Signal dictionary
        current_price: Current market price
        total_pnl: Total portfolio P&L
        emergency_stop: Emergency stop instance (optional)
        drawdown_protection: Drawdown protection instance (optional)
        breakeven_stop: Breakeven stop instance (optional)
        
    Returns:
        Dictionary with protection status:
        {
            'emergency_triggered': bool,
            'drawdown_triggered': bool,
            'move_to_breakeven': bool,
            'breakeven_price': float or None
        }
    """
    result = {
        'emergency_triggered': False,
        'drawdown_triggered': False,
        'move_to_breakeven': False,
        'breakeven_price': None
    }
    
    if emergency_stop:
        result['emergency_triggered'] = emergency_stop.should_close(signal, current_price)
    
    if drawdown_protection:
        result['drawdown_triggered'] = drawdown_protection.should_stop_trading(total_pnl)
    
    if breakeven_stop:
        result['move_to_breakeven'] = breakeven_stop.should_move_to_breakeven(signal, current_price)
        if result['move_to_breakeven']:
            result['breakeven_price'] = breakeven_stop.get_breakeven_stop(signal)
    
    return result


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create protection instances
    emergency = EmergencyStopLoss(max_loss_percent=5.0)
    drawdown = DrawdownProtection(max_drawdown_percent=25.0)
    breakeven = BreakevenStop(trigger_profit_percent=1.5)
    
    # Example signal
    test_signal = {
        'symbol': 'BTC/USDT',
        'entry': 50000.0,
        'stop_loss': 49000.0,
        'direction': 'LONG'
    }
    
    # Test emergency stop (price drops 6%)
    print("\n=== Testing Emergency Stop ===")
    current_price = 47000.0  # 6% loss
    if emergency.should_close(test_signal, current_price):
        print(f"✓ Emergency stop triggered at {current_price}")
    
    # Test drawdown protection
    print("\n=== Testing Drawdown Protection ===")
    if drawdown.should_stop_trading(-30.0):  # 30% loss
        print("✓ Drawdown protection triggered")
    
    # Test breakeven stop
    print("\n=== Testing Breakeven Stop ===")
    current_price = 50800.0  # 1.6% profit
    if breakeven.should_move_to_breakeven(test_signal, current_price):
        print(f"✓ Moving to breakeven at {breakeven.get_breakeven_stop(test_signal)}")
    
    # Test all protections
    print("\n=== Testing All Protections ===")
    result = check_all_protections(
        test_signal, 47000.0, -30.0,
        emergency, drawdown, breakeven
    )
    print(f"Protection status: {result}")
