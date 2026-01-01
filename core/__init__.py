"""
Core infrastructure for the trading bot framework.

This package provides centralized management for:
- Portfolio risk management (cross-bot coordination)
- Bot orchestration and health monitoring
- Daily P&L summaries and reporting
- Interactive Telegram commands
- Unified signal management with lifecycle tracking
- Enhanced message templates
- Signal analytics and performance insights

Version: 2.0.0
"""

__version__ = "2.0.0"

# =============================================================================
# LAZY IMPORTS - Avoid circular dependencies
# =============================================================================

def get_portfolio_manager():
    """Get the global PortfolioRiskManager instance."""
    from .portfolio_manager import get_portfolio_manager as _get
    return _get()

def get_orchestrator():
    """Get the global BotOrchestrator instance."""
    from .bot_orchestrator import get_orchestrator as _get
    return _get()

def get_pnl_reporter():
    """Get the global PnLReporter instance."""
    from .pnl_reporter import get_pnl_reporter as _get
    return _get()

def get_signal_manager(bot_name: str, **kwargs):
    """Get or create SignalManager for a bot."""
    from .signal_manager import get_signal_manager as _get
    return _get(bot_name, **kwargs)


# =============================================================================
# DIRECT IMPORTS - Commonly used classes and functions
# =============================================================================

# Signal management
from .signal_manager import (
    Signal,
    SignalState,
    SignalResult,
    CloseReason,
)

# Enhanced templates
from .enhanced_templates import (
    SignalMessageBuilder,
    ResultMessageBuilder,
    PartialExitMessageBuilder,
    format_signal_message,
    format_result_message,
    build_streak_alert,
    build_daily_summary,
    build_emergency_alert,
)

# Analytics
from .signal_analytics import (
    SignalAnalytics,
    SetupStats,
    SymbolRanking,
)

# Adapter for easy bot migration
from .signal_adapter import (
    SignalAdapter,
    create_adapter_for_bot,
    migrate_state_file,
)

# Mixin for quick bot integration
from .bot_signal_mixin import (
    BotSignalMixin,
    create_price_fetcher,
)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Version
    "__version__",
    
    # Manager getters
    "get_portfolio_manager",
    "get_orchestrator", 
    "get_pnl_reporter",
    "get_signal_manager",
    
    # Signal classes
    "Signal",
    "SignalState",
    "SignalResult",
    "CloseReason",
    
    # Message builders
    "SignalMessageBuilder",
    "ResultMessageBuilder",
    "PartialExitMessageBuilder",
    "format_signal_message",
    "format_result_message",
    "build_streak_alert",
    "build_daily_summary",
    "build_emergency_alert",
    
    # Analytics
    "SignalAnalytics",
    "SetupStats",
    "SymbolRanking",
    
    # Adapter
    "SignalAdapter",
    "create_adapter_for_bot",
    "migrate_state_file",
    
    # Mixin
    "BotSignalMixin",
    "create_price_fetcher",
]
