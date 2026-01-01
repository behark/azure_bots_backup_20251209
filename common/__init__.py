"""
Common utilities package for cryptocurrency trading bots.

This package provides shared functionality across all bots including:
- Structured logging with contextual information
- Resilience patterns (retry logic, circuit breakers)
- Type definitions and data models
- Risk management utilities
- Emergency stop functionality
"""

__version__ = "2.0.0"
__all__ = [
    # Core utilities
    "logging_config",
    "resilience",
    "types",
    "risk",
    "emergency_stop",
]

# Lazy imports for better performance
def __getattr__(name: str):
    """Lazy module imports for commonly used classes."""
    if name == "DEFAULT_RISK":
        from common.risk import DEFAULT_RISK
        return DEFAULT_RISK
    if name == "get_price_tolerance":
        from common.risk import get_price_tolerance
        return get_price_tolerance
    if name == "RiskDefaults":
        from common.risk import RiskDefaults
        return RiskDefaults
    if name == "validate_risk_reward":
        from common.risk import validate_risk_reward
        return validate_risk_reward
    if name == "calculate_position_size":
        from common.risk import calculate_position_size
        return calculate_position_size
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
