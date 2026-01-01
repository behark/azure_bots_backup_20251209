#!/usr/bin/env python3
"""
Common type definitions for trading bots.

Provides TypedDict definitions, enums, and data classes for type-safe
configuration, signal data, and API responses across all bots.

Benefits:
- Static type checking with mypy
- IDE autocomplete and type hints
- Runtime validation
- Self-documenting code
- Reduced bugs from type mismatches
"""

from typing import Any, Dict, TypedDict, List, Optional, Literal
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class SignalDirection(Enum):
    """Trading signal direction."""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class SignalType(Enum):
    """Type of trading signal."""
    FUNDING_RATE = "funding_rate"
    LIQUIDATION = "liquidation"
    VOLUME_SPIKE = "volume_spike"
    HARMONIC_PATTERN = "harmonic_pattern"
    CANDLESTICK_PATTERN = "candlestick_pattern"
    FIBONACCI_REVERSAL = "fibonacci_reversal"
    STRAT_PATTERN = "strat_pattern"
    FIBONACCI_SWING = "fibonacci_swing"
    MTF_ALIGNMENT = "mtf_alignment"
    PSAR_REVERSAL = "psar_reversal"
    ORB_BREAKOUT = "orb_breakout"
    VOLUME_PROFILE = "volume_profile"
    CONSENSUS = "consensus"


class BotStatus(Enum):
    """Bot operational status."""
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


# ============================================================================
# Configuration TypedDicts
# ============================================================================

class SymbolConfig(TypedDict, total=False):
    """
    Configuration for a single trading symbol.

    Attributes:
        symbol: Trading pair symbol (e.g., "BTC/USDT")
        period: Timeframe for analysis (e.g., "15m", "1h")
        cooldown_minutes: Minutes to wait between signals for this symbol
        enabled: Whether this symbol is actively monitored
    """
    symbol: str
    period: str
    cooldown_minutes: int
    enabled: bool


class BotConfig(TypedDict, total=False):
    """
    Configuration for a single bot.

    Attributes:
        enabled: Whether bot is active
        max_open_signals: Maximum concurrent signals allowed
        interval_seconds: Seconds between analysis cycles
        default_cooldown_minutes: Default cooldown for all symbols
        symbols: List of symbol configurations
        custom_params: Bot-specific parameters (varies by bot type)
    """
    enabled: bool
    max_open_signals: int
    interval_seconds: int
    default_cooldown_minutes: int
    symbols: List[SymbolConfig]
    custom_params: Dict[str, Any]


class GlobalConfig(TypedDict, total=False):
    """
    Global configuration for all bots.

    Loaded from global_config.json. Contains configuration for all 15 bots.

    Attributes:
        funding_bot: Configuration for funding rate bot
        liquidation_bot: Configuration for liquidation bot
        volume_bot: Configuration for volume spike bot
        harmonic_bot: Configuration for harmonic pattern bot
        candlestick_bot: Configuration for candlestick pattern bot
        fib_reversal_bot: Configuration for fibonacci reversal bot
        strat_bot: Configuration for STRAT pattern bot
        fib_swing_bot: Configuration for fibonacci swing bot
        mtf_bot: Configuration for multi-timeframe bot
        psar_bot: Configuration for parabolic SAR bot
        diy_bot: Configuration for DIY pattern bot
        most_bot: Configuration for MOST pattern bot
        orb_bot: Configuration for opening range breakout bot
        volume_profile_bot: Configuration for volume profile bot
        consensus_bot: Configuration for consensus bot
    """
    funding_bot: BotConfig
    liquidation_bot: BotConfig
    volume_bot: BotConfig
    harmonic_bot: BotConfig
    candlestick_bot: BotConfig
    fib_reversal_bot: BotConfig
    strat_bot: BotConfig
    fib_swing_bot: BotConfig
    mtf_bot: BotConfig
    psar_bot: BotConfig
    diy_bot: BotConfig
    most_bot: BotConfig
    orb_bot: BotConfig
    volume_profile_bot: BotConfig
    consensus_bot: BotConfig


# ============================================================================
# Signal Data TypedDicts
# ============================================================================

class BaseSignal(TypedDict, total=False):
    """
    Base signal structure shared by all signal types.

    Attributes:
        symbol: Trading pair symbol
        direction: Signal direction (LONG/SHORT)
        timestamp: ISO 8601 timestamp
        current_price: Current market price when signal generated
        entry: Suggested entry price
        stop_loss: Stop loss price
        take_profit_1: First take profit target
        take_profit_2: Second take profit target
        take_profit_3: Optional third take profit target
        risk_reward_ratio: Risk:reward ratio (reward/risk)
        correlation_id: UUID for tracking related operations
    """
    symbol: str
    direction: str  # "LONG" or "SHORT"
    timestamp: str
    current_price: float
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: Optional[float]
    risk_reward_ratio: float
    correlation_id: str


class FundingRateSignal(BaseSignal):
    """
    Funding rate signal data.

    Extends BaseSignal with funding rate specific fields.

    Attributes:
        funding_rate: Current funding rate (e.g., -0.0123 = -1.23%)
        open_interest: Open interest in USDT
        oi_change_pct: Open interest change percentage
    """
    funding_rate: float
    open_interest: float
    oi_change_pct: float


class LiquidationSignal(BaseSignal):
    """
    Liquidation signal data.

    Extends BaseSignal with liquidation specific fields.

    Attributes:
        liquidations_usdt: Total liquidations in USDT
        long_liquidations: Long liquidations in USDT
        short_liquidations: Short liquidations in USDT
        net_liquidations: Net liquidations (short - long)
    """
    liquidations_usdt: float
    long_liquidations: float
    short_liquidations: float
    net_liquidations: float


class PatternSignal(BaseSignal):
    """
    Pattern-based signal (harmonic, candlestick, STRAT, etc.).

    Extends BaseSignal with pattern identification.

    Attributes:
        pattern_name: Name of detected pattern
        pattern_type: Type of pattern (harmonic, candlestick, etc.)
        timeframe: Timeframe where pattern detected
        confidence: Confidence score (0.0-1.0)
    """
    pattern_name: str
    pattern_type: str
    timeframe: str
    confidence: float


# ============================================================================
# API Response TypedDicts
# ============================================================================

class OHLCVCandle(TypedDict):
    """
    OHLCV candle data from exchange API.

    Attributes:
        timestamp: Unix timestamp in milliseconds
        open: Opening price
        high: Highest price
        low: Lowest price
        close: Closing price
        volume: Trading volume
    """
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class TickerData(TypedDict, total=False):
    """
    Ticker data from exchange API.

    Attributes:
        symbol: Trading pair symbol
        last: Last traded price
        bid: Best bid price
        ask: Best ask price
        high: 24h highest price
        low: 24h lowest price
        volume: 24h trading volume
        timestamp: Unix timestamp in milliseconds
    """
    symbol: str
    last: float
    bid: float
    ask: float
    high: float
    low: float
    volume: float
    timestamp: int


class BalanceInfo(TypedDict, total=False):
    """
    Account balance information.

    Attributes:
        currency: Currency code (e.g., "USDT")
        free: Available balance
        used: Balance in use (open orders)
        total: Total balance (free + used)
    """
    currency: str
    free: float
    used: float
    total: float


# ============================================================================
# State Management TypedDicts
# ============================================================================

class SignalState(TypedDict, total=False):
    """
    Signal state tracking for cooldown management.

    Attributes:
        symbol: Trading pair symbol
        direction: Signal direction
        timestamp: ISO 8601 timestamp when signal was sent
        expiry: ISO 8601 timestamp when cooldown expires
        signal_id: Unique signal identifier
        notified: Whether Telegram notification was sent
    """
    symbol: str
    direction: str
    timestamp: str
    expiry: str
    signal_id: str
    notified: bool


class BotState(TypedDict, total=False):
    """
    Bot state persistence.

    Attributes:
        last_run: ISO 8601 timestamp of last run
        total_signals: Total signals generated
        active_signals: Currently active signals (within cooldown)
        errors_count: Count of errors since last restart
        status: Current bot status
    """
    last_run: str
    total_signals: int
    active_signals: List[SignalState]
    errors_count: int
    status: str


# ============================================================================
# Statistics TypedDicts
# ============================================================================

class SignalStats(TypedDict, total=False):
    """
    Signal performance statistics.

    Attributes:
        total_signals: Total signals generated
        winning_signals: Signals that hit take profit
        losing_signals: Signals that hit stop loss
        pending_signals: Signals still active
        win_rate: Win rate percentage (0-100)
        avg_risk_reward: Average risk:reward ratio
        total_profit_pct: Total profit in percentage
        sharpe_ratio: Risk-adjusted return metric
    """
    total_signals: int
    winning_signals: int
    losing_signals: int
    pending_signals: int
    win_rate: float
    avg_risk_reward: float
    total_profit_pct: float
    sharpe_ratio: float


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TradingSignal:
    """
    Trading signal data class with validation.

    More structured than TypedDict, with methods and validation.
    Use for signal processing and business logic.
    """
    symbol: str
    direction: SignalDirection
    signal_type: SignalType
    timestamp: str
    current_price: float
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: Optional[float] = None
    risk_reward_ratio: float = 0.0
    correlation_id: Optional[str] = None
    metadata: dict = None  # type: ignore

    def __post_init__(self) -> None:
        """Validate signal data after initialization."""
        if self.metadata is None:
            self.metadata = {}

        # Validate prices
        if self.current_price <= 0:
            raise ValueError("Current price must be positive")

        # Validate direction consistency
        if self.direction == SignalDirection.LONG:
            if self.stop_loss >= self.entry:
                raise ValueError("LONG signal: stop loss must be below entry")
            if self.take_profit_1 <= self.entry:
                raise ValueError("LONG signal: take profit must be above entry")
        elif self.direction == SignalDirection.SHORT:
            if self.stop_loss <= self.entry:
                raise ValueError("SHORT signal: stop loss must be above entry")
            if self.take_profit_1 >= self.entry:
                raise ValueError("SHORT signal: take profit must be below entry")

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary for JSON serialization."""
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "signal_type": self.signal_type.value,
            "timestamp": self.timestamp,
            "current_price": self.current_price,
            "entry": self.entry,
            "stop_loss": self.stop_loss,
            "take_profit_1": self.take_profit_1,
            "take_profit_2": self.take_profit_2,
            "take_profit_3": self.take_profit_3,
            "risk_reward_ratio": self.risk_reward_ratio,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }


# ============================================================================
# Risk Management TypedDicts
# ============================================================================

class RiskConfig(TypedDict, total=False):
    """
    Risk management configuration.
    
    Attributes:
        sl_atr_multiplier: ATR multiplier for stop loss calculation
        tp1_atr_multiplier: ATR multiplier for first take profit
        tp2_atr_multiplier: ATR multiplier for second take profit
        tp3_atr_multiplier: ATR multiplier for third take profit
        min_risk_reward: Minimum R:R ratio for TP1
        min_risk_reward_tp2: Minimum R:R ratio for TP2
        max_risk_percent: Maximum risk per trade (% of account)
        max_stop_loss_percent: Maximum stop loss distance (%)
        min_stop_loss_percent: Minimum stop loss distance (%)
        emergency_stop_percent: Emergency stop loss trigger (%)
        use_trailing_stop: Whether to use trailing stops
        trailing_stop_activation: Profit % to activate trailing
        trailing_stop_distance: Trail distance in ATR
        sl_buffer_percent: Buffer to add to stop loss
        max_drawdown_percent: Maximum portfolio drawdown
        breakeven_trigger_percent: Profit % to move to breakeven
        slippage_percent: Expected slippage buffer
        max_correlation_signals: Max signals in same direction
        price_tolerance: TP/SL hit detection tolerance
    """
    sl_atr_multiplier: float
    tp1_atr_multiplier: float
    tp2_atr_multiplier: float
    tp3_atr_multiplier: float
    min_risk_reward: float
    min_risk_reward_tp2: float
    max_risk_percent: float
    max_stop_loss_percent: float
    min_stop_loss_percent: float
    emergency_stop_percent: float
    use_trailing_stop: bool
    trailing_stop_activation: float
    trailing_stop_distance: float
    sl_buffer_percent: float
    max_drawdown_percent: float
    breakeven_trigger_percent: float
    slippage_percent: float
    max_correlation_signals: int
    price_tolerance: float


class WatchlistItem(TypedDict, total=False):
    """
    Watchlist item configuration.
    
    Attributes:
        symbol: Trading pair symbol
        timeframe: Candle timeframe
        period: Alternative name for timeframe
        cooldown_minutes: Cooldown after signal
        exchange: Exchange identifier
        market_type: "swap" or "spot"
        enabled: Whether symbol is active
    """
    symbol: str
    timeframe: str
    period: str
    cooldown_minutes: int
    exchange: str
    market_type: str
    enabled: bool


class MarketRegime(TypedDict):
    """
    Market regime detection result.
    
    Attributes:
        regime: TRENDING, RANGING, CHOPPY, TRANSITIONING
        adx: Average Directional Index value
        trend_strength: Trend strength 0-100
        volatility: Normalized volatility
        efficiency: Price efficiency ratio
        trend_direction: UP, DOWN, NEUTRAL
        confidence: Detection confidence 0-1
    """
    regime: str
    adx: float
    trend_strength: float
    volatility: float
    efficiency: float
    trend_direction: str
    confidence: float


class EquityUpdate(TypedDict):
    """
    Equity curve update result.
    
    Attributes:
        current_equity: Current equity percentage
        current_drawdown: Current drawdown percentage
        protection_triggered: Whether protection was just triggered
        protection_active: Whether protection is currently active
    """
    current_equity: float
    current_drawdown: float
    protection_triggered: bool
    protection_active: bool


class ValidationResult(TypedDict):
    """
    Signal validation result.
    
    Attributes:
        is_valid: Whether signal passed validation
        errors: List of error messages
        warnings: List of warning messages
        quality_score: Quality score 0-100
        adjusted_values: Dictionary of corrected values
    """
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    quality_score: float
    adjusted_values: Dict[str, float]


# ============================================================================
# Type Aliases
# ============================================================================

# Common type aliases for cleaner code
Timeframe = Literal["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"]
Direction = Literal["LONG", "SHORT"]
OrderSide = Literal["buy", "sell"]
OrderType = Literal["market", "limit", "stop", "stop_limit"]
MarketRegimeType = Literal["TRENDING", "RANGING", "CHOPPY", "TRANSITIONING", "UNKNOWN"]
SignalQuality = Literal["PREMIUM", "CONFIRMED", "STANDARD", "WEAK"]


# Example usage
if __name__ == "__main__":
    from datetime import datetime

    # Example 1: Using TypedDict for configuration
    symbol_config: SymbolConfig = {
        "symbol": "BTC/USDT",
        "period": "15m",
        "cooldown_minutes": 60,
        "enabled": True
    }

    # Example 2: Using data class for signal
    signal = TradingSignal(
        symbol="BTC/USDT",
        direction=SignalDirection.LONG,
        signal_type=SignalType.FUNDING_RATE,
        timestamp=datetime.utcnow().isoformat() + "Z",
        current_price=45000.0,
        entry=45050.0,
        stop_loss=44500.0,
        take_profit_1=46000.0,
        take_profit_2=47000.0,
        risk_reward_ratio=1.72,
        correlation_id="abc-123-def-456",
        metadata={"funding_rate": -0.0123}
    )

    print(f"Signal: {signal.symbol} {signal.direction.value}")
    print(f"Entry: {signal.entry}, SL: {signal.stop_loss}, TP1: {signal.take_profit_1}")
    print(f"Risk:Reward: {signal.risk_reward_ratio}")

    # Example 3: Type hints in function signatures
    def process_signal(signal_data: BaseSignal) -> bool:
        """Process a trading signal with type safety."""
        print(f"Processing {signal_data['symbol']} signal")
        return True

    # Example 4: Type-safe configuration loading
    def load_bot_config(config: BotConfig) -> None:
        """Load bot configuration with type checking."""
        if config.get('enabled', False):
            print(f"Bot enabled with {len(config.get('symbols', []))} symbols")
