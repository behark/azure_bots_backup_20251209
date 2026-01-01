"""
Risk management utilities for trading bots.

Provides default risk parameters and helper functions for
price tolerance calculations used in TP/SL detection.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RiskDefaults:
    """Default risk parameters for trading bots."""
    
    # ATR multipliers for stop loss and take profit levels
    sl_atr_multiplier: float = 1.5
    tp1_atr_multiplier: float = 1.5
    tp2_atr_multiplier: float = 2.5
    tp3_atr_multiplier: float = 4.0
    
    # Risk percentage limits
    min_risk_pct: float = 0.003  # 0.3% minimum risk
    max_risk_pct: float = 0.025  # 2.5% maximum risk
    
    # Risk/reward requirements
    min_risk_reward: float = 1.5
    
    # Position sizing
    max_position_pct: float = 0.02  # 2% max position size
    
    # Price tolerance for TP/SL hit detection
    # This accounts for spread, slippage, and minor price variations
    price_tolerance_base: float = 0.001  # 0.1% base tolerance
    
    # Volatility limits
    max_volatility_pct: float = 0.08  # 8% max ATR/price ratio
    
    # Maximum leverage
    max_leverage: int = 10


# Default risk configuration instance
DEFAULT_RISK = RiskDefaults()


def get_price_tolerance(
    atr: Optional[float] = None,
    price: Optional[float] = None,
    base_tolerance: float = 0.001
) -> float:
    """
    Calculate price tolerance for TP/SL hit detection.
    
    The tolerance makes it slightly easier to detect TP/SL hits,
    accounting for spread, slippage, and minor price variations.
    
    Args:
        atr: Average True Range (optional, for dynamic tolerance)
        price: Current price (optional, for ATR-based calculation)
        base_tolerance: Base tolerance as decimal (default 0.1%)
    
    Returns:
        Price tolerance as a decimal (e.g., 0.001 = 0.1%)
    """
    # If ATR and price provided, use dynamic tolerance based on volatility
    if atr is not None and price is not None and price > 0:
        # Higher volatility = slightly higher tolerance
        volatility_ratio = atr / price
        # Scale tolerance: base + (volatility_ratio * 0.5), capped at 0.5%
        dynamic_tolerance = base_tolerance + (volatility_ratio * 0.5)
        return min(dynamic_tolerance, 0.005)  # Cap at 0.5%
    
    # Return base tolerance if no dynamic calculation possible
    return base_tolerance


def validate_risk_reward(
    entry: float,
    stop_loss: float,
    take_profit: float,
    direction: str,
    min_rr: float = 1.5
) -> bool:
    """
    Validate that a trade meets minimum risk/reward requirements.
    
    Args:
        entry: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
        direction: Trade direction ("LONG", "SHORT", "BULLISH", "BEARISH")
        min_rr: Minimum risk/reward ratio
    
    Returns:
        True if trade meets minimum R:R, False otherwise
    """
    if entry <= 0 or stop_loss <= 0 or take_profit <= 0:
        return False
    
    is_long = direction.upper() in ("LONG", "BULLISH")
    
    if is_long:
        risk = entry - stop_loss
        reward = take_profit - entry
    else:
        risk = stop_loss - entry
        reward = entry - take_profit
    
    if risk <= 0:
        return False
    
    rr_ratio = reward / risk
    return rr_ratio >= min_rr


def calculate_position_size(
    account_balance: float,
    risk_per_trade_pct: float,
    entry: float,
    stop_loss: float,
    max_position_pct: float = 0.02
) -> float:
    """
    Calculate position size based on risk management rules.
    
    Args:
        account_balance: Total account balance
        risk_per_trade_pct: Risk per trade as decimal (e.g., 0.01 = 1%)
        entry: Entry price
        stop_loss: Stop loss price
        max_position_pct: Maximum position as % of balance
    
    Returns:
        Position size in base currency units
    """
    if account_balance <= 0 or entry <= 0:
        return 0.0
    
    risk_amount = account_balance * risk_per_trade_pct
    risk_per_unit = abs(entry - stop_loss)
    
    if risk_per_unit <= 0:
        return 0.0
    
    position_size = risk_amount / risk_per_unit
    
    # Cap at maximum position size
    max_position = (account_balance * max_position_pct) / entry
    return min(position_size, max_position)

