#!/usr/bin/env python3
"""
Advanced TP/SL Calculator
Centralized, configurable take profit and stop loss calculation for all trading bots.
Supports multiple calculation methods: ATR-based, percentage-based, structure-based.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class CalculationMethod(Enum):
    """Methods for calculating TP/SL levels."""
    ATR = "atr"                    # ATR-based (recommended)
    PERCENTAGE = "percentage"      # Fixed percentage from entry
    STRUCTURE = "structure"        # Based on swing highs/lows
    FIBONACCI = "fibonacci"        # Fibonacci retracement levels
    VOLATILITY = "volatility"      # Based on recent volatility


@dataclass
class TradeLevels:
    """Calculated trade levels."""
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: Optional[float] = None
    risk_reward_1: float = 0.0
    risk_reward_2: float = 0.0
    risk_amount: float = 0.0
    is_valid: bool = True
    rejection_reason: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "entry": self.entry,
            "stop_loss": self.stop_loss,
            "take_profit_1": self.take_profit_1,
            "take_profit_2": self.take_profit_2,
            "take_profit_3": self.take_profit_3,
            "risk_reward_1": self.risk_reward_1,
            "risk_reward_2": self.risk_reward_2,
            "risk_amount": self.risk_amount,
            "is_valid": self.is_valid,
        }


class TPSLCalculator:
    """
    Advanced TP/SL Calculator with multiple methods and validation.
    
    Usage:
        calc = TPSLCalculator()
        levels = calc.calculate(
            entry=100.0,
            direction="LONG",
            atr=2.5,
            sl_multiplier=1.5,
            tp1_multiplier=2.0,
            tp2_multiplier=3.5
        )
    """
    
    def __init__(
        self,
        min_risk_reward: float = 1.8,
        max_sl_percent: float = 5.0,
        min_sl_percent: float = 0.2,
        sl_buffer_percent: float = 0.75,
    ):
        """
        Initialize calculator with validation parameters.
        
        Args:
            min_risk_reward: Minimum R:R ratio to accept signal
            max_sl_percent: Maximum stop loss as % of entry (filter extreme SL)
            min_sl_percent: Minimum stop loss as % of entry (avoid too tight SL)
            sl_buffer_percent: Extra buffer added to SL to avoid wicks
        """
        self.min_risk_reward = min_risk_reward
        self.max_sl_percent = max_sl_percent
        self.min_sl_percent = min_sl_percent
        self.sl_buffer_percent = sl_buffer_percent
    
    def calculate(
        self,
        entry: float,
        direction: str,
        atr: Optional[float] = None,
        sl_multiplier: float = 1.5,
        tp1_multiplier: float = 2.0,
        tp2_multiplier: float = 3.5,
        tp3_multiplier: Optional[float] = None,
        method: CalculationMethod = CalculationMethod.ATR,
        swing_high: Optional[float] = None,
        swing_low: Optional[float] = None,
        volatility_percent: Optional[float] = None,
        custom_sl: Optional[float] = None,
    ) -> TradeLevels:
        """
        Calculate TP/SL levels using specified method.
        
        Args:
            entry: Entry price
            direction: "LONG" or "SHORT" (also accepts "BULLISH"/"BEARISH")
            atr: Average True Range value
            sl_multiplier: ATR multiplier for stop loss
            tp1_multiplier: ATR multiplier for TP1
            tp2_multiplier: ATR multiplier for TP2
            tp3_multiplier: ATR multiplier for TP3 (optional)
            method: Calculation method to use
            swing_high: Recent swing high (for STRUCTURE method)
            swing_low: Recent swing low (for STRUCTURE method)
            volatility_percent: Recent volatility % (for VOLATILITY method)
            custom_sl: Custom stop loss price (overrides calculation)
            
        Returns:
            TradeLevels with all calculated values
        """
        # Normalize direction
        direction = self._normalize_direction(direction)
        
        if method == CalculationMethod.ATR:
            return self._calculate_atr_based(
                entry, direction, atr, sl_multiplier,
                tp1_multiplier, tp2_multiplier, tp3_multiplier, custom_sl
            )
        elif method == CalculationMethod.PERCENTAGE:
            return self._calculate_percentage_based(
                entry, direction, volatility_percent or 1.0,
                tp1_multiplier, tp2_multiplier, tp3_multiplier
            )
        elif method == CalculationMethod.STRUCTURE:
            return self._calculate_structure_based(
                entry, direction, swing_high, swing_low,
                tp1_multiplier, tp2_multiplier, tp3_multiplier
            )
        elif method == CalculationMethod.FIBONACCI:
            return self._calculate_fibonacci_based(
                entry, direction, swing_high, swing_low
            )
        else:
            return self._calculate_atr_based(
                entry, direction, atr, sl_multiplier,
                tp1_multiplier, tp2_multiplier, tp3_multiplier, custom_sl
            )
    
    def _normalize_direction(self, direction: str) -> str:
        """Normalize direction to LONG or SHORT."""
        direction = direction.upper()
        if direction in ["BULLISH", "BUY", "LONG"]:
            return "LONG"
        return "SHORT"
    
    def _calculate_atr_based(
        self,
        entry: float,
        direction: str,
        atr: Optional[float],
        sl_mult: float,
        tp1_mult: float,
        tp2_mult: float,
        tp3_mult: Optional[float],
        custom_sl: Optional[float],
    ) -> TradeLevels:
        """Calculate levels based on ATR."""
        if atr is None or atr <= 0:
            # Fallback to percentage-based if no ATR
            atr = entry * 0.01  # 1% as fallback
            logger.warning("No ATR provided, using 1% fallback")
        
        # Apply buffer to ATR for SL
        buffered_atr = atr * (1 + self.sl_buffer_percent / 100)
        
        if direction == "LONG":
            sl = custom_sl if custom_sl else entry - (buffered_atr * sl_mult)
            tp1 = entry + (atr * tp1_mult)
            tp2 = entry + (atr * tp2_mult)
            tp3 = entry + (atr * tp3_mult) if tp3_mult else None
        else:
            sl = custom_sl if custom_sl else entry + (buffered_atr * sl_mult)
            tp1 = entry - (atr * tp1_mult)
            tp2 = entry - (atr * tp2_mult)
            tp3 = entry - (atr * tp3_mult) if tp3_mult else None
        
        return self._validate_and_build(entry, sl, tp1, tp2, tp3, direction)
    
    def _calculate_percentage_based(
        self,
        entry: float,
        direction: str,
        sl_percent: float,
        tp1_mult: float,
        tp2_mult: float,
        tp3_mult: Optional[float],
    ) -> TradeLevels:
        """Calculate levels based on fixed percentages."""
        sl_amount = entry * (sl_percent / 100)
        
        if direction == "LONG":
            sl = entry - sl_amount
            tp1 = entry + (sl_amount * tp1_mult)
            tp2 = entry + (sl_amount * tp2_mult)
            tp3 = entry + (sl_amount * tp3_mult) if tp3_mult else None
        else:
            sl = entry + sl_amount
            tp1 = entry - (sl_amount * tp1_mult)
            tp2 = entry - (sl_amount * tp2_mult)
            tp3 = entry - (sl_amount * tp3_mult) if tp3_mult else None
        
        return self._validate_and_build(entry, sl, tp1, tp2, tp3, direction)
    
    def _calculate_structure_based(
        self,
        entry: float,
        direction: str,
        swing_high: Optional[float],
        swing_low: Optional[float],
        tp1_mult: float,
        tp2_mult: float,
        tp3_mult: Optional[float],
    ) -> TradeLevels:
        """Calculate levels based on market structure (swing points)."""
        if direction == "LONG":
            if swing_low is None:
                return TradeLevels(
                    entry=entry, stop_loss=0, take_profit_1=0, take_profit_2=0,
                    is_valid=False, rejection_reason="No swing low for LONG structure"
                )
            # SL below swing low with buffer
            sl = swing_low * (1 - self.sl_buffer_percent / 100)
            risk = entry - sl
            tp1 = entry + (risk * tp1_mult)
            tp2 = entry + (risk * tp2_mult)
            tp3 = entry + (risk * tp3_mult) if tp3_mult else None
        else:
            if swing_high is None:
                return TradeLevels(
                    entry=entry, stop_loss=0, take_profit_1=0, take_profit_2=0,
                    is_valid=False, rejection_reason="No swing high for SHORT structure"
                )
            # SL above swing high with buffer
            sl = swing_high * (1 + self.sl_buffer_percent / 100)
            risk = sl - entry
            tp1 = entry - (risk * tp1_mult)
            tp2 = entry - (risk * tp2_mult)
            tp3 = entry - (risk * tp3_mult) if tp3_mult else None
        
        return self._validate_and_build(entry, sl, tp1, tp2, tp3, direction)
    
    def _calculate_fibonacci_based(
        self,
        entry: float,
        direction: str,
        swing_high: Optional[float],
        swing_low: Optional[float],
    ) -> TradeLevels:
        """Calculate levels based on Fibonacci retracements."""
        if swing_high is None or swing_low is None:
            return TradeLevels(
                entry=entry, stop_loss=0, take_profit_1=0, take_profit_2=0,
                is_valid=False, rejection_reason="Need both swing high and low for Fibonacci"
            )
        
        fib_range = abs(swing_high - swing_low)
        
        if direction == "LONG":
            sl = swing_low * (1 - self.sl_buffer_percent / 100)
            tp1 = entry + (fib_range * 0.618)
            tp2 = entry + (fib_range * 1.0)
            tp3 = entry + (fib_range * 1.618)
        else:
            sl = swing_high * (1 + self.sl_buffer_percent / 100)
            tp1 = entry - (fib_range * 0.618)
            tp2 = entry - (fib_range * 1.0)
            tp3 = entry - (fib_range * 1.618)
        
        return self._validate_and_build(entry, sl, tp1, tp2, tp3, direction)
    
    def _validate_and_build(
        self,
        entry: float,
        sl: float,
        tp1: float,
        tp2: float,
        tp3: Optional[float],
        direction: str,
    ) -> TradeLevels:
        """Validate levels and build TradeLevels object."""
        risk = abs(entry - sl)
        reward1 = abs(tp1 - entry)
        reward2 = abs(tp2 - entry)
        
        # Calculate R:R ratios
        rr1 = reward1 / risk if risk > 0 else 0
        rr2 = reward2 / risk if risk > 0 else 0
        
        # Validate SL is not too far or too close
        sl_percent = (risk / entry) * 100 if entry > 0 else 0
        
        rejection_reason = None
        is_valid = True
        
        if sl_percent > self.max_sl_percent:
            rejection_reason = f"SL too far ({sl_percent:.1f}% > {self.max_sl_percent}%)"
            is_valid = False
        elif sl_percent < self.min_sl_percent:
            rejection_reason = f"SL too tight ({sl_percent:.2f}% < {self.min_sl_percent}%)"
            is_valid = False
        elif rr1 < self.min_risk_reward:
            rejection_reason = f"R:R too low ({rr1:.2f} < {self.min_risk_reward})"
            is_valid = False
        
        # Validate direction consistency
        if direction == "LONG":
            if sl >= entry:
                rejection_reason = "Invalid LONG: SL >= Entry"
                is_valid = False
            if tp1 <= entry:
                rejection_reason = "Invalid LONG: TP1 <= Entry"
                is_valid = False
        else:
            if sl <= entry:
                rejection_reason = "Invalid SHORT: SL <= Entry"
                is_valid = False
            if tp1 >= entry:
                rejection_reason = "Invalid SHORT: TP1 >= Entry"
                is_valid = False
        
        return TradeLevels(
            entry=entry,
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            risk_reward_1=rr1,
            risk_reward_2=rr2,
            risk_amount=risk,
            is_valid=is_valid,
            rejection_reason=rejection_reason,
        )
    
    def calculate_position_size(
        self,
        account_balance: float,
        risk_percent: float,
        entry: float,
        stop_loss: float,
        leverage: float = 1.0,
    ) -> Dict[str, float]:
        """
        Calculate position size based on risk management.
        
        Args:
            account_balance: Total account balance in USDT
            risk_percent: Max risk per trade (e.g., 2.0 for 2%)
            entry: Entry price
            stop_loss: Stop loss price
            leverage: Trading leverage
            
        Returns:
            Dict with position sizing info
        """
        risk_amount = account_balance * (risk_percent / 100)
        price_risk = abs(entry - stop_loss)
        price_risk_percent = (price_risk / entry) * 100 if entry > 0 else 0
        
        # Position size in USDT
        position_size_usdt = (risk_amount / price_risk_percent) * 100 if price_risk_percent > 0 else 0
        
        # Adjust for leverage
        margin_required = position_size_usdt / leverage if leverage > 0 else position_size_usdt
        
        # Position size in coins
        position_size_coins = position_size_usdt / entry if entry > 0 else 0
        
        return {
            "risk_amount_usdt": risk_amount,
            "position_size_usdt": position_size_usdt,
            "position_size_coins": position_size_coins,
            "margin_required": margin_required,
            "risk_percent": risk_percent,
            "price_risk_percent": price_risk_percent,
        }
    
    def calculate_trailing_stop(
        self,
        entry: float,
        current_price: float,
        direction: str,
        atr: float,
        activation_atr: float = 1.0,
        trail_distance_atr: float = 0.5,
        current_sl: Optional[float] = None,
    ) -> Optional[float]:
        """
        Calculate trailing stop level.
        
        Args:
            entry: Original entry price
            current_price: Current market price
            direction: Trade direction
            atr: Current ATR value
            activation_atr: ATR units of profit before trailing activates
            trail_distance_atr: ATR distance for trailing stop
            current_sl: Current stop loss level
            
        Returns:
            New trailing stop level, or None if not activated
        """
        direction = self._normalize_direction(direction)
        activation_profit = atr * activation_atr
        trail_distance = atr * trail_distance_atr
        
        if direction == "LONG":
            profit = current_price - entry
            if profit >= activation_profit:
                new_sl = current_price - trail_distance
                if current_sl is None or new_sl > current_sl:
                    return new_sl
        else:  # SHORT
            profit = entry - current_price
            if profit >= activation_profit:
                new_sl = current_price + trail_distance
                if current_sl is None or new_sl < current_sl:
                    return new_sl
        
        return current_sl


def calculate_atr(candles: List[Dict], period: int = 14) -> Optional[float]:
    """
    Calculate Average True Range from candle data.
    
    Args:
        candles: List of candles with 'high', 'low', 'close' keys
        period: ATR period (default 14)
        
    Returns:
        ATR value or None if insufficient data
    """
    if len(candles) < period + 1:
        return None
    
    trs = []
    for i in range(1, len(candles)):
        high = candles[i].get("high", candles[i][2] if isinstance(candles[i], list) else 0)
        low = candles[i].get("low", candles[i][3] if isinstance(candles[i], list) else 0)
        prev_close = candles[i-1].get("close", candles[i-1][4] if isinstance(candles[i-1], list) else 0)
        
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        trs.append(tr)
    
    if len(trs) < period:
        return sum(trs) / len(trs) if trs else None
    
    return sum(trs[-period:]) / period


# Convenience function for quick calculations
def quick_calculate(
    entry: float,
    direction: str,
    atr: float,
    sl_mult: float = 1.5,
    tp1_mult: float = 2.0,
    tp2_mult: float = 3.5,
) -> TradeLevels:
    """Quick calculation with default settings."""
    calc = TPSLCalculator()
    return calc.calculate(
        entry=entry,
        direction=direction,
        atr=atr,
        sl_multiplier=sl_mult,
        tp1_multiplier=tp1_mult,
        tp2_multiplier=tp2_mult,
    )


if __name__ == "__main__":
    # Example usage
    calc = TPSLCalculator(min_risk_reward=1.5)
    
    # ATR-based calculation
    levels = calc.calculate(
        entry=100.0,
        direction="LONG",
        atr=2.5,
        sl_multiplier=1.5,
        tp1_multiplier=2.0,
        tp2_multiplier=3.5,
    )
    
    print("ATR-Based Calculation:")
    print(f"  Entry: {levels.entry:.2f}")
    print(f"  Stop Loss: {levels.stop_loss:.2f}")
    print(f"  TP1: {levels.take_profit_1:.2f} (R:R {levels.risk_reward_1:.2f})")
    print(f"  TP2: {levels.take_profit_2:.2f} (R:R {levels.risk_reward_2:.2f})")
    print(f"  Valid: {levels.is_valid}")
    
    # Position sizing
    sizing = calc.calculate_position_size(
        account_balance=1000,
        risk_percent=2.0,
        entry=levels.entry,
        stop_loss=levels.stop_loss,
        leverage=10,
    )
    
    print("\nPosition Sizing (2% risk, 10x leverage):")
    print(f"  Risk Amount: ${sizing['risk_amount_usdt']:.2f}")
    print(f"  Position Size: ${sizing['position_size_usdt']:.2f}")
    print(f"  Margin Required: ${sizing['margin_required']:.2f}")
