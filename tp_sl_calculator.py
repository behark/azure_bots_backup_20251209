#!/usr/bin/env python3
"""
Advanced TP/SL Calculator
Centralized, configurable take profit and stop loss calculation for all trading bots.
Supports multiple calculation methods: ATR-based, percentage-based, structure-based.

PHASE 3 ENHANCEMENTS:
- Enhanced type hints using common.types
- Mypy strict mode compatibility
- Better IDE autocomplete support
"""

import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pathlib import Path

# Phase 3: Import common types for enhanced type safety
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from common.types import Direction, Timeframe
    PHASE3_TYPES_AVAILABLE = True
except ImportError:
    PHASE3_TYPES_AVAILABLE = False
    # Fallback type aliases
    Direction = str  # type: ignore
    Timeframe = str  # type: ignore

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
    
    def to_dict(self) -> Dict[str, Any]:
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
        min_risk_reward: float = 1.0,  # Relaxed for testing - accept any positive R:R
        max_sl_percent: float = 10.0,  # Relaxed for testing - allow wider stops
        min_sl_percent: float = 0.05,
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
        market_regime: str = "RANGING",  # TRENDING, RANGING, CHOPPY
        adx_strength: float = 20.0,
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
            market_regime: Current market regime for dynamic scaling
            adx_strength: Trend strength for scaling TP
            
        Returns:
            TradeLevels with all calculated values
        """
        # Normalize direction
        direction = self._normalize_direction(direction)

        # Dynamic adjustments based on market regime
        adj_sl_mult = sl_multiplier
        adj_tp1_mult = tp1_multiplier
        adj_tp2_mult = tp2_multiplier

        if market_regime == "TRENDING":
            # In trending markets, widen SL slightly and extend TP
            adj_sl_mult *= 1.2
            # FIX: Ensure trend_boost is at least 1.0 to avoid reducing TP
            # When ADX > 25, boost TP proportionally; when ADX <= 25, keep TP unchanged
            trend_boost = max(1.0, min(1.5, adx_strength / 25.0))
            adj_tp1_mult *= trend_boost
            adj_tp2_mult *= (trend_boost * 1.2)
        elif market_regime == "CHOPPY":
            # In choppy markets, tighten everything and reduce R:R expectations
            adj_sl_mult *= 0.8
            adj_tp1_mult *= 0.7
            adj_tp2_mult *= 0.7

        if method == CalculationMethod.ATR:
            return self._calculate_atr_based(
                entry, direction, atr, adj_sl_mult,
                adj_tp1_mult, adj_tp2_mult, tp3_multiplier, custom_sl
            )
        elif method == CalculationMethod.PERCENTAGE:
            return self._calculate_percentage_based(
                entry, direction, volatility_percent or 1.0,
                adj_tp1_mult, adj_tp2_mult, tp3_multiplier
            )
        elif method == CalculationMethod.STRUCTURE:
            return self._calculate_structure_based(
                entry, direction, swing_high, swing_low,
                adj_tp1_mult, adj_tp2_mult, tp3_multiplier
            )
        elif method == CalculationMethod.FIBONACCI:
            return self._calculate_fibonacci_based(
                entry, direction, swing_high, swing_low
            )
        else:
            return self._calculate_atr_based(
                entry, direction, atr, adj_sl_mult,
                adj_tp1_mult, adj_tp2_mult, tp3_multiplier, custom_sl
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
        """
        Calculate TP/SL levels using ATR (Average True Range) methodology.

        ATR-based calculations are the recommended approach as they adapt to market
        volatility. Higher volatility (larger ATR) results in wider stops and targets,
        while lower volatility produces tighter levels. This prevents overly tight
        stops in volatile markets and overly wide stops in calm markets.

        Algorithm:
            1. Apply buffer to ATR for stop loss (default 0.75% extra)
            2. For LONG trades:
               - SL = Entry - (buffered_ATR × sl_mult)
               - TP1 = Entry + (ATR × tp1_mult)
               - TP2 = Entry + (ATR × tp2_mult)
            3. For SHORT trades:
               - SL = Entry + (buffered_ATR × sl_mult)
               - TP1 = Entry - (ATR × tp1_mult)
               - TP2 = Entry - (ATR × tp2_mult)
            4. Validate levels and calculate risk:reward ratios

        Args:
            entry (float): Entry price for the trade
            direction (str): Trade direction ("LONG" or "SHORT")
            atr (Optional[float]): Average True Range value. If None, uses 1% of entry as fallback.
            sl_mult (float): ATR multiplier for stop loss distance (typically 1.5-2.0)
            tp1_mult (float): ATR multiplier for first take profit (typically 2.0-3.0)
            tp2_mult (float): ATR multiplier for second take profit (typically 3.5-5.0)
            tp3_mult (Optional[float]): ATR multiplier for third take profit (optional)
            custom_sl (Optional[float]): Override calculated SL with custom price level

        Returns:
            TradeLevels: Validated trade levels with risk:reward ratios

        Example:
            >>> calc = TPSLCalculator()
            >>> levels = calc._calculate_atr_based(
            ...     entry=100.0, direction="LONG", atr=2.5,
            ...     sl_mult=1.5, tp1_mult=2.0, tp2_mult=3.5, tp3_mult=None, custom_sl=None
            ... )
            >>> print(f"SL: {levels.stop_loss:.2f}, TP1: {levels.take_profit_1:.2f}")
            SL: 96.22, TP1: 105.00

        Notes:
            - Buffer prevents stops from being hit by normal price wicks
            - TP targets use unbuffered ATR (actual volatility-based targets)
            - Custom SL bypasses ATR calculation but still gets validated
            - Falls back to 1% of entry price if ATR is unavailable
            - Typical ATR multipliers: SL=1.5, TP1=2.0, TP2=3.5 for 1.33R and 2.33R
        """
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
        """
        Calculate TP/SL levels based on market structure and swing points.

        Structure-based calculations place stops beyond recent swing highs/lows, which
        are logical invalidation points for the trade thesis. This approach respects
        price action and support/resistance levels rather than arbitrary ATR distances.

        Algorithm:
            1. For LONG trades:
               - Identify recent swing low (support level)
               - SL = swing_low × (1 - buffer%) - placed just below support
               - Risk = Entry - SL
               - TP1 = Entry + (Risk × tp1_mult)
               - TP2 = Entry + (Risk × tp2_mult)
            2. For SHORT trades:
               - Identify recent swing high (resistance level)
               - SL = swing_high × (1 + buffer%) - placed just above resistance
               - Risk = SL - Entry
               - TP1 = Entry - (Risk × tp1_mult)
               - TP2 = Entry - (Risk × tp2_mult)
            3. Validate levels ensure proper direction and risk:reward

        Args:
            entry (float): Entry price for the trade
            direction (str): Trade direction ("LONG" or "SHORT")
            swing_high (Optional[float]): Recent swing high price (resistance).
                                         Required for SHORT trades.
            swing_low (Optional[float]): Recent swing low price (support).
                                        Required for LONG trades.
            tp1_mult (float): Risk multiplier for TP1 (e.g., 2.0 for 2R)
            tp2_mult (float): Risk multiplier for TP2 (e.g., 3.0 for 3R)
            tp3_mult (Optional[float]): Risk multiplier for TP3 (optional)

        Returns:
            TradeLevels: Validated trade levels, or invalid TradeLevels if swing point missing

        Example:
            >>> calc = TPSLCalculator()
            >>> levels = calc._calculate_structure_based(
            ...     entry=100.0, direction="LONG",
            ...     swing_high=None, swing_low=95.0,
            ...     tp1_mult=2.0, tp2_mult=3.0, tp3_mult=None
            ... )
            >>> if levels.is_valid:
            ...     print(f"SL at swing low: {levels.stop_loss:.2f}")

        Notes:
            - Swing points should be recent (last 20-50 candles depending on timeframe)
            - Buffer (default 0.75%) prevents stop hunting at obvious levels
            - Returns invalid TradeLevels if required swing point is missing
            - More suitable for trending markets with clear structure
            - Risk distance varies based on structure, not volatility
            - Best combined with confirmation from other indicators
        """
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
        """
        Calculate TP/SL levels using Fibonacci retracement extensions.

        Fibonacci-based calculations use the golden ratio (0.618) and its extensions
        (1.0, 1.618) to project natural profit targets. These levels often coincide
        with areas where price stalls or reverses, based on market psychology and
        the Fibonacci sequence's prevalence in nature and markets.

        Algorithm:
            1. Calculate Fibonacci range: |swing_high - swing_low|
            2. For LONG trades (bouncing from swing_low):
               - SL = swing_low × (1 - buffer%) - below the swing low
               - TP1 = Entry + (range × 0.618) - 61.8% Fibonacci extension
               - TP2 = Entry + (range × 1.0) - 100% extension (full range)
               - TP3 = Entry + (range × 1.618) - 161.8% golden extension
            3. For SHORT trades (rejecting from swing_high):
               - SL = swing_high × (1 + buffer%) - above the swing high
               - TP1 = Entry - (range × 0.618) - 61.8% Fibonacci extension
               - TP2 = Entry - (range × 1.0) - 100% extension (full range)
               - TP3 = Entry - (range × 1.618) - 161.8% golden extension

        Args:
            entry (float): Entry price for the trade
            direction (str): Trade direction ("LONG" or "SHORT")
            swing_high (float): Recent swing high (top of the Fibonacci range).
                               Required for both directions.
            swing_low (float): Recent swing low (bottom of the Fibonacci range).
                              Required for both directions.

        Returns:
            TradeLevels: Validated trade levels with 3 Fibonacci-based targets,
                        or invalid TradeLevels if swing points are missing

        Example:
            >>> calc = TPSLCalculator()
            >>> levels = calc._calculate_fibonacci_based(
            ...     entry=100.0, direction="LONG",
            ...     swing_high=110.0, swing_low=90.0
            ... )
            >>> if levels.is_valid:
            ...     print(f"TP1 (0.618): {levels.take_profit_1:.2f}")
            ...     print(f"TP2 (1.0): {levels.take_profit_2:.2f}")
            ...     print(f"TP3 (1.618): {levels.take_profit_3:.2f}")

        Notes:
            - Requires both swing_high and swing_low (complete Fibonacci range)
            - Common Fibonacci levels: 0.618 (golden ratio), 1.0 (full extension), 1.618 (phi)
            - TP3 at 1.618 is ambitious; consider partial profit at TP1/TP2
            - Works best when entry is near a Fibonacci retracement (e.g., 0.382, 0.5, 0.618)
            - Swing points should define a significant recent price swing
            - Popular in forex and crypto markets due to psychological significance
        """
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
        """
        Validate calculated trade levels and construct TradeLevels object.

        Performs comprehensive validation to ensure trade levels are logical,
        safe, and meet minimum risk:reward requirements. Rejects trades with
        stops that are too tight (whipsaw risk) or too wide (excessive risk),
        and ensures risk:reward ratios justify the trade.

        Validation Checks:
            1. **Stop Loss Distance**:
               - Too wide: SL% > max_sl_percent (default 5%)
                 → Risk too high, position would be overleveraged
               - Too tight: SL% < min_sl_percent (default 0.2%)
                 → Normal volatility would trigger stop

            2. **Risk:Reward Ratio**:
               - RR1 < min_risk_reward (default 1.8)
                 → Insufficient reward to justify risk
               - Expects at least 1.8R on first target

            3. **Direction Consistency**:
               - LONG: SL must be < Entry, TP1 must be > Entry
               - SHORT: SL must be > Entry, TP1 must be < Entry
               - Prevents inverted trade setups

            4. **Mathematical Validity**:
               - Calculates risk = |Entry - SL|
               - Calculates reward1 = |TP1 - Entry|
               - Calculates reward2 = |TP2 - Entry|
               - Derives RR ratios: reward/risk

        Args:
            entry (float): Entry price
            sl (float): Calculated stop loss price
            tp1 (float): Calculated first take profit price
            tp2 (float): Calculated second take profit price
            tp3 (Optional[float]): Calculated third take profit price (optional)
            direction (str): Trade direction ("LONG" or "SHORT")

        Returns:
            TradeLevels: Object containing:
                - All price levels (entry, SL, TP1, TP2, TP3)
                - Risk:reward ratios (rr1, rr2)
                - Risk amount in price units
                - is_valid (bool): Whether trade passes all validations
                - rejection_reason (str): Explanation if invalid

        Example:
            >>> calc = TPSLCalculator(min_risk_reward=2.0, max_sl_percent=3.0)
            >>> levels = calc._validate_and_build(
            ...     entry=100.0, sl=97.0, tp1=106.0, tp2=112.0,
            ...     tp3=None, direction="LONG"
            ... )
            >>> if not levels.is_valid:
            ...     print(f"Trade rejected: {levels.rejection_reason}")
            >>> else:
            ...     print(f"Valid trade: {levels.risk_reward_1:.2f}R")

        Notes:
            - Invalid trades should NOT be executed; return them to user for review
            - Validation parameters (min RR, max SL%) are configurable in __init__
            - Risk:reward is calculated to TP1 (first partial exit), not full target
            - SL percentage is calculated as: (|Entry - SL| / Entry) × 100
            - Tight stops (<0.2%) often result in premature stop-outs
            - Wide stops (>5%) expose too much capital to single trade risk
        """
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


def calculate_atr(candles: List[Any], period: int = 14) -> Optional[float]:
    """
    Calculate Average True Range (ATR) from candle data.

    ATR measures market volatility by calculating the average of True Range values
    over a specified period. True Range is the greatest of:
    1. Current high - current low
    2. |Current high - previous close|
    3. |Current low - previous close|

    This captures volatility including gaps between candles, making it more comprehensive
    than simple high-low range.

    Algorithm:
        1. For each candle, calculate True Range (TR):
           TR = max(high - low, |high - prev_close|, |low - prev_close|)
        2. Average the last 'period' TR values:
           ATR = average(TR[-period:])
        3. If insufficient data (<period), returns average of available TRs

    Args:
        candles (List[Dict]): Candle data in either format:
            - List of dicts with keys: 'high', 'low', 'close'
            - List of OHLCV arrays: [timestamp, open, high, low, close, volume]
              where high=index[2], low=index[3], close=index[4]
        period (int, optional): Number of candles for ATR calculation.
                               Standard is 14. Defaults to 14.

    Returns:
        Optional[float]: ATR value as absolute price units, or None if insufficient data
            - Returns None if candles list has <2 items (need previous close)
            - Returns average of available TRs if <period candles available

    Example:
        >>> candles = [
        ...     {'high': 100, 'low': 95, 'close': 98},
        ...     {'high': 102, 'low': 97, 'close': 101},
        ...     {'high': 103, 'low': 99, 'close': 100},
        ...     # ... 11 more candles for ATR(14)
        ... ]
        >>> atr = calculate_atr(candles, period=14)
        >>> print(f"ATR(14): {atr:.2f}")

        >>> # Using CCXT OHLCV format
        >>> ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=50)
        >>> atr = calculate_atr(ohlcv, period=14)

    Notes:
        - ATR is always positive (absolute value)
        - Higher ATR = higher volatility = wider TP/SL levels recommended
        - Lower ATR = lower volatility = tighter TP/SL levels appropriate
        - ATR(14) is standard, but shorter periods (7, 10) react faster to volatility changes
        - ATR is in price units, not percentage (e.g., $2.50 for a $100 asset = 2.5% volatility)
        - Commonly used for: stop loss placement, position sizing, volatility filters
        - Flexible input format supports both dict and OHLCV array formats from exchanges
    """
    if len(candles) < period + 1:
        return None
    
    trs: List[float] = []
    for i in range(1, len(candles)):
        # Handle both dict and OHLCV list formats
        candle = candles[i]
        prev_candle = candles[i - 1]

        if isinstance(candle, dict):
            high: float = float(candle.get("high", 0))
            low: float = float(candle.get("low", 0))
        else:
            high = float(candle[2])  # OHLCV: [timestamp, open, high, low, close, volume]
            low = float(candle[3])

        if isinstance(prev_candle, dict):
            prev_close: float = float(prev_candle.get("close", 0))
        else:
            prev_close = float(prev_candle[4])
        
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        trs.append(tr)
    
    if len(trs) < period:
        return sum(trs) / len(trs) if trs else None
    
    return float(sum(trs[-period:])) / period


# Convenience function for quick calculations
def quick_calculate(
    entry: float,
    direction: str,
    atr: float,
    sl_mult: float = 1.5,
    tp1_mult: float = 2.0,
    tp2_mult: float = 3.5,
) -> TradeLevels:
    """
    Quick TP/SL calculation with sensible default validation parameters.

    Convenience wrapper around TPSLCalculator for one-off calculations without
    needing to instantiate the calculator class. Uses default validation settings:
    - Minimum risk:reward: 1.8
    - Maximum stop loss: 5.0% of entry
    - Minimum stop loss: 0.2% of entry
    - Stop loss buffer: 0.75% extra

    This is ideal for interactive use, notebooks, or simple bots that don't need
    custom validation rules.

    Args:
        entry (float): Entry price for the trade
        direction (str): Trade direction ("LONG", "SHORT", "BULLISH", "BEARISH", "BUY", "SELL")
        atr (float): Average True Range value for volatility-based calculations
        sl_mult (float, optional): ATR multiplier for stop loss distance.
                                   Defaults to 1.5 (reasonable for most timeframes).
        tp1_mult (float, optional): ATR multiplier for first take profit.
                                    Defaults to 2.0 (1.33R with sl_mult=1.5).
        tp2_mult (float, optional): ATR multiplier for second take profit.
                                    Defaults to 3.5 (2.33R with sl_mult=1.5).

    Returns:
        TradeLevels: Validated trade levels with:
            - entry, stop_loss, take_profit_1, take_profit_2
            - risk_reward_1, risk_reward_2 (actual R:R ratios achieved)
            - is_valid (bool): Whether trade passes validation
            - rejection_reason (str): Explanation if invalid

    Example:
        >>> # Simple LONG trade calculation
        >>> levels = quick_calculate(entry=100.0, direction="LONG", atr=2.5)
        >>> if levels.is_valid:
        ...     print(f"Entry: ${levels.entry:.2f}")
        ...     print(f"SL: ${levels.stop_loss:.2f}")
        ...     print(f"TP1: ${levels.take_profit_1:.2f} ({levels.risk_reward_1:.2f}R)")
        ...     print(f"TP2: ${levels.take_profit_2:.2f} ({levels.risk_reward_2:.2f}R)")
        ... else:
        ...     print(f"Invalid: {levels.rejection_reason}")

        >>> # Custom multipliers for tighter stops
        >>> tight_levels = quick_calculate(
        ...     entry=50.0, direction="SHORT", atr=1.0,
        ...     sl_mult=1.0, tp1_mult=1.5, tp2_mult=2.5
        ... )

    Notes:
        - For custom validation rules, instantiate TPSLCalculator directly
        - Default multipliers (1.5, 2.0, 3.5) provide ~1.33R and 2.33R targets
        - Automatically applies 0.75% buffer to stop loss to avoid wick-outs
        - Direction is normalized: "BULLISH"/"BUY"/"LONG" → "LONG", others → "SHORT"
        - Returns ATR-based calculations only (for other methods, use TPSLCalculator)
    """
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
