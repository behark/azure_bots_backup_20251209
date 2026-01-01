#!/usr/bin/env python3
"""
Unified Signal Manager - Centralized Signal Tracking System

Replaces duplicated SignalTracker classes across all bots with a single,
robust implementation featuring:

- Signal lifecycle states (PENDING → ACTIVE → PARTIAL → CLOSED)
- Partial position tracking (TP1 hit = 50% closed, 50% running)
- Trailing stop management
- Breakeven automation
- Time-based expiry
- Performance analytics
- Portfolio Manager integration

Usage:
    from core.signal_manager import get_signal_manager, Signal, SignalState
    
    manager = get_signal_manager("strat_bot")
    
    # Create new signal
    signal = Signal(
        symbol="BTC/USDT",
        direction="LONG",
        entry=42000.0,
        stop_loss=41500.0,
        take_profit_1=42500.0,
        take_profit_2=43000.0,
    )
    manager.add_signal(signal)
    
    # Check signals for TP/SL hits
    results = manager.check_signals(price_fetcher)
    
    # Get performance stats
    stats = manager.get_performance_stats()
"""

import json
import logging
import secrets
import string
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Default price tolerance for TP/SL hit detection (0.5%)
DEFAULT_PRICE_TOLERANCE = 0.005

# Default max signal age before expiry (hours)
DEFAULT_MAX_SIGNAL_AGE_HOURS = 48


class SignalState(str, Enum):
    """Signal lifecycle states."""
    PENDING = "pending"          # Signal created, waiting for entry
    ACTIVE = "active"            # Entry hit, position open
    PARTIAL_TP1 = "partial_tp1"  # TP1 hit, partial position closed
    PARTIAL_TP2 = "partial_tp2"  # TP2 hit, smaller position remains
    BREAKEVEN = "breakeven"      # SL moved to entry
    TRAILING = "trailing"        # Trailing stop active
    CLOSED = "closed"            # Position fully closed


class CloseReason(str, Enum):
    """Reasons for signal closure."""
    TP1 = "TP1"
    TP2 = "TP2"
    TP3 = "TP3"
    SL = "SL"
    TRAILING_SL = "TRAILING_SL"
    BREAKEVEN_SL = "BREAKEVEN_SL"
    EXPIRED = "EXPIRED"
    MANUAL = "MANUAL"
    REVERSAL = "REVERSAL"
    EMERGENCY = "EMERGENCY"


@dataclass
class Signal:
    """Represents a trading signal with full lifecycle tracking."""
    # Core identifiers
    signal_id: str = ""
    bot_name: str = ""
    symbol: str = ""
    direction: str = ""  # LONG/SHORT or BULLISH/BEARISH
    
    # Price levels
    entry: float = 0.0
    stop_loss: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    take_profit_3: float = 0.0
    
    # Current tracking
    current_stop_loss: float = 0.0  # May be modified by trailing/breakeven
    current_price: float = 0.0
    
    # State
    state: str = SignalState.PENDING.value
    
    # Partial position tracking (1.0 = 100% position remaining)
    position_remaining: float = 1.0
    
    # TP hit tracking
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    
    # Timestamps
    created_at: str = ""
    entry_hit_at: str = ""
    closed_at: str = ""
    last_updated: str = ""
    
    # Excursion tracking (for analytics)
    max_favorable_excursion: float = 0.0  # Best unrealized P&L
    max_adverse_excursion: float = 0.0    # Worst unrealized P&L
    
    # Result data
    close_reason: str = ""
    exit_price: float = 0.0
    realized_pnl_pct: float = 0.0
    
    # Metadata
    exchange: str = "MEXC"
    timeframe: str = "5m"
    confidence: float = 0.0
    pattern_name: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.signal_id:
            self.signal_id = self._generate_id()
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.last_updated:
            self.last_updated = self.created_at
        if self.current_stop_loss == 0.0:
            self.current_stop_loss = self.stop_loss
        # Normalize direction
        self.direction = self._normalize_direction(self.direction)
    
    def _generate_id(self) -> str:
        """Generate unique signal ID."""
        symbol_short = self.symbol.replace('/USDT:USDT', '').replace('/USDT', '').replace(':USDT', '')[:6]
        dir_short = 'L' if self.direction in ('LONG', 'BULLISH', 'BUY') else 'S'
        random_part = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(4))
        timestamp = datetime.now().strftime("%H%M")
        return f"{symbol_short}-{dir_short}-{timestamp}-{random_part}"
    
    @staticmethod
    def _normalize_direction(direction: str) -> str:
        """Normalize direction to LONG/SHORT."""
        direction = direction.upper()
        if direction in ('BULLISH', 'BUY', 'LONG'):
            return 'LONG'
        return 'SHORT'
    
    @property
    def is_long(self) -> bool:
        return self.direction == 'LONG'
    
    @property
    def is_open(self) -> bool:
        return self.state not in (SignalState.CLOSED.value,)
    
    @property
    def age_hours(self) -> float:
        """Get signal age in hours."""
        try:
            created = datetime.fromisoformat(self.created_at.replace('Z', '+00:00'))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return (now - created).total_seconds() / 3600
        except (ValueError, AttributeError):
            return 0.0
    
    @property
    def duration_str(self) -> str:
        """Get human-readable duration."""
        hours = self.age_hours
        if hours < 1:
            return f"{int(hours * 60)}m"
        elif hours < 24:
            return f"{hours:.1f}h"
        else:
            return f"{hours / 24:.1f}d"
    
    def calculate_pnl(self, exit_price: float) -> float:
        """Calculate P&L percentage."""
        if self.entry == 0:
            return 0.0
        if self.is_long:
            return ((exit_price - self.entry) / self.entry) * 100
        return ((self.entry - exit_price) / self.entry) * 100
    
    def update_excursions(self, current_price: float) -> None:
        """Update max favorable/adverse excursion."""
        if self.entry == 0:
            return
        
        current_pnl = self.calculate_pnl(current_price)
        self.current_price = current_price
        
        if current_pnl > self.max_favorable_excursion:
            self.max_favorable_excursion = current_pnl
        if current_pnl < self.max_adverse_excursion:
            self.max_adverse_excursion = current_pnl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Signal":
        """Create Signal from dictionary."""
        # Handle extra field specially
        extra = data.pop('extra', {}) if 'extra' in data else {}
        # Filter to only valid fields
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        signal = cls(**valid_fields)
        signal.extra = extra
        return signal


@dataclass
class SignalResult:
    """Result of a closed signal for notifications."""
    signal: Signal
    close_reason: CloseReason
    exit_price: float
    pnl_pct: float
    duration_str: str
    mfe: float  # Max Favorable Excursion
    mae: float  # Max Adverse Excursion


class SignalManager:
    """
    Unified signal management for a bot.
    
    Features:
    - Lifecycle state tracking
    - Partial position management
    - Trailing stop support
    - Breakeven automation
    - Time-based expiry
    - Excursion tracking
    """
    
    def __init__(
        self,
        bot_name: str,
        state_file: Optional[Path] = None,
        price_tolerance: float = DEFAULT_PRICE_TOLERANCE,
        max_signal_age_hours: float = DEFAULT_MAX_SIGNAL_AGE_HOURS,
        enable_breakeven: bool = True,
        breakeven_trigger_pnl: float = 1.0,  # Move to breakeven at 1% profit
        partial_tp1_size: float = 0.5,  # Close 50% at TP1
        partial_tp2_size: float = 0.3,  # Close 30% more at TP2
    ):
        self.bot_name = bot_name
        self.state_file = state_file or BASE_DIR / f"{bot_name}" / f"{bot_name}_signals.json"
        self.price_tolerance = price_tolerance
        self.max_signal_age_hours = max_signal_age_hours
        self.enable_breakeven = enable_breakeven
        self.breakeven_trigger_pnl = breakeven_trigger_pnl
        self.partial_tp1_size = partial_tp1_size
        self.partial_tp2_size = partial_tp2_size
        
        self.signals: Dict[str, Signal] = {}
        self.history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        
        # Load existing state
        self._load_state()
        
        logger.info("SignalManager initialized for %s with %d open signals", 
                   bot_name, len(self.signals))
    
    def _load_state(self) -> None:
        """Load state from file."""
        if not self.state_file.exists():
            return
        
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
            
            # Load open signals
            for sig_id, sig_data in data.get("signals", {}).items():
                try:
                    self.signals[sig_id] = Signal.from_dict(sig_data)
                except (TypeError, KeyError) as e:
                    logger.warning("Failed to load signal %s: %s", sig_id, e)
            
            # Load history (keep last 500)
            self.history = data.get("history", [])[-500:]
            
        except (json.JSONDecodeError, IOError) as e:
            logger.error("Failed to load signal state: %s", e)
    
    def _save_state(self) -> None:
        """Save state to file (atomic write)."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self.state_file.with_suffix('.tmp')
        
        try:
            data = {
                "signals": {k: v.to_dict() for k, v in self.signals.items()},
                "history": self.history[-500:],  # Keep last 500
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            temp_file.replace(self.state_file)
            
        except Exception as e:
            logger.error("Failed to save signal state: %s", e)
            if temp_file.exists():
                temp_file.unlink()
    
    def add_signal(self, signal: Signal) -> Signal:
        """Add a new signal."""
        with self._lock:
            signal.bot_name = self.bot_name
            signal.state = SignalState.ACTIVE.value
            signal.entry_hit_at = datetime.now(timezone.utc).isoformat()
            signal.last_updated = signal.entry_hit_at
            
            self.signals[signal.signal_id] = signal
            self._save_state()
            
            logger.info("Signal added: %s %s %s", 
                       signal.symbol, signal.direction, signal.signal_id)
            
            return signal
    
    def get_signal(self, signal_id: str) -> Optional[Signal]:
        """Get a signal by ID."""
        return self.signals.get(signal_id)
    
    def get_open_signals(self) -> Dict[str, Signal]:
        """Get all open signals."""
        return {k: v for k, v in self.signals.items() if v.is_open}
    
    def has_open_signal(self, symbol: str, direction: Optional[str] = None) -> bool:
        """Check if there's already an open signal for symbol/direction."""
        for signal in self.signals.values():
            if signal.symbol == symbol and signal.is_open:
                if direction is None or signal.direction == Signal._normalize_direction(direction):
                    return True
        return False
    
    def update_trailing_stop(self, signal_id: str, new_stop: float) -> bool:
        """Update trailing stop for a signal."""
        with self._lock:
            if signal_id not in self.signals:
                return False
            
            signal = self.signals[signal_id]
            old_stop = signal.current_stop_loss
            
            # Only update if new stop is better (higher for LONG, lower for SHORT)
            if signal.is_long:
                if new_stop <= old_stop:
                    return False
            else:
                if new_stop >= old_stop:
                    return False
            
            signal.current_stop_loss = new_stop
            signal.state = SignalState.TRAILING.value
            signal.last_updated = datetime.now(timezone.utc).isoformat()
            
            self._save_state()
            
            logger.info("Trailing stop updated for %s: %.6f → %.6f", 
                       signal_id, old_stop, new_stop)
            return True
    
    def move_to_breakeven(self, signal_id: str) -> bool:
        """Move stop loss to entry price (breakeven)."""
        with self._lock:
            if signal_id not in self.signals:
                return False
            
            signal = self.signals[signal_id]
            
            if signal.state == SignalState.BREAKEVEN.value:
                return False  # Already at breakeven
            
            old_stop = signal.current_stop_loss
            signal.current_stop_loss = signal.entry
            signal.state = SignalState.BREAKEVEN.value
            signal.last_updated = datetime.now(timezone.utc).isoformat()
            
            self._save_state()
            
            logger.info("Moved to breakeven for %s: SL %.6f → %.6f (entry)", 
                       signal_id, old_stop, signal.entry)
            return True
    
    def close_signal(
        self, 
        signal_id: str, 
        exit_price: float, 
        reason: CloseReason,
        partial: bool = False,
        partial_size: float = 0.0,
    ) -> Optional[SignalResult]:
        """
        Close a signal (fully or partially).
        
        Args:
            signal_id: Signal ID to close
            exit_price: Exit price
            reason: Close reason (TP1, TP2, SL, etc.)
            partial: If True, only close part of position
            partial_size: Size of position to close (0.0 to 1.0)
            
        Returns:
            SignalResult with details, or None if signal not found
        """
        with self._lock:
            if signal_id not in self.signals:
                return None
            
            signal = self.signals[signal_id]
            now = datetime.now(timezone.utc).isoformat()
            
            # Calculate P&L
            pnl_pct = signal.calculate_pnl(exit_price)
            
            # Update signal
            signal.exit_price = exit_price
            signal.realized_pnl_pct = pnl_pct
            signal.close_reason = reason.value
            signal.last_updated = now
            
            if partial and partial_size > 0:
                # Partial close
                signal.position_remaining -= partial_size
                signal.position_remaining = max(0.0, signal.position_remaining)
                
                # Update state based on which TP hit
                if reason == CloseReason.TP1:
                    signal.tp1_hit = True
                    signal.state = SignalState.PARTIAL_TP1.value
                elif reason == CloseReason.TP2:
                    signal.tp2_hit = True
                    signal.state = SignalState.PARTIAL_TP2.value
                
                logger.info("Partial close %s: %s at %.6f (%.1f%% remaining)", 
                           signal_id, reason.value, exit_price, signal.position_remaining * 100)
            else:
                # Full close
                signal.position_remaining = 0.0
                signal.state = SignalState.CLOSED.value
                signal.closed_at = now
                
                # Add to history
                self.history.append(signal.to_dict())
                
                # Remove from active signals
                del self.signals[signal_id]
                
                logger.info("Signal closed %s: %s at %.6f, P&L: %.2f%%", 
                           signal_id, reason.value, exit_price, pnl_pct)
            
            self._save_state()
            
            return SignalResult(
                signal=signal,
                close_reason=reason,
                exit_price=exit_price,
                pnl_pct=pnl_pct,
                duration_str=signal.duration_str,
                mfe=signal.max_favorable_excursion,
                mae=signal.max_adverse_excursion,
            )
    
    def check_signals(
        self,
        price_fetcher: Callable[[str], Optional[float]],
        notifier: Optional[Any] = None,
    ) -> List[SignalResult]:
        """
        Check all open signals for TP/SL hits.
        
        Args:
            price_fetcher: Function that takes symbol and returns current price
            notifier: Optional notifier for sending alerts
            
        Returns:
            List of SignalResults for any closed signals
        """
        results: List[SignalResult] = []
        
        with self._lock:
            signals_to_check = list(self.signals.items())
        
        for signal_id, signal in signals_to_check:
            if not signal.is_open:
                continue
            
            # Get current price
            try:
                price = price_fetcher(signal.symbol)
                if price is None:
                    continue
            except Exception as e:
                logger.debug("Error fetching price for %s: %s", signal.symbol, e)
                continue
            
            # Update excursions
            signal.update_excursions(price)
            
            # Check for expiry
            if signal.age_hours >= self.max_signal_age_hours:
                result = self.close_signal(signal_id, price, CloseReason.EXPIRED)
                if result:
                    results.append(result)
                continue
            
            # Check for breakeven trigger
            if self.enable_breakeven and signal.state == SignalState.ACTIVE.value:
                current_pnl = signal.calculate_pnl(price)
                if current_pnl >= self.breakeven_trigger_pnl:
                    self.move_to_breakeven(signal_id)
            
            # Check TP/SL hits
            result = self._check_tp_sl_hit(signal, price)
            if result:
                results.append(result)
        
        return results
    
    def _check_tp_sl_hit(self, signal: Signal, price: float) -> Optional[SignalResult]:
        """Check if price hit any TP or SL level."""
        tol = self.price_tolerance
        
        tp1 = signal.take_profit_1
        tp2 = signal.take_profit_2
        tp3 = signal.take_profit_3
        sl = signal.current_stop_loss  # Use current (may be trailing/breakeven)
        
        if signal.is_long:
            # LONG: TPs above entry, SL below
            # Check TP3 first (highest)
            if tp3 > 0 and not signal.tp3_hit and price >= (tp3 * (1 - tol)):
                return self.close_signal(signal.signal_id, price, CloseReason.TP3)
            
            # Check TP2
            if tp2 > 0 and not signal.tp2_hit and price >= (tp2 * (1 - tol)):
                if signal.position_remaining > self.partial_tp2_size:
                    # Partial close at TP2
                    self.close_signal(signal.signal_id, price, CloseReason.TP2, 
                                     partial=True, partial_size=self.partial_tp2_size)
                    signal.tp2_hit = True
                    signal.tp1_hit = True  # Implied
                    return None  # Don't return result for partial
                else:
                    # Full close at TP2
                    return self.close_signal(signal.signal_id, price, CloseReason.TP2)
            
            # Check TP1
            if tp1 > 0 and not signal.tp1_hit and price >= (tp1 * (1 - tol)):
                if signal.position_remaining > self.partial_tp1_size:
                    # Partial close at TP1
                    self.close_signal(signal.signal_id, price, CloseReason.TP1,
                                     partial=True, partial_size=self.partial_tp1_size)
                    signal.tp1_hit = True
                    return None  # Don't return result for partial
                else:
                    return self.close_signal(signal.signal_id, price, CloseReason.TP1)
            
            # Check SL
            if sl > 0 and price <= (sl * (1 + tol)):
                reason = CloseReason.SL
                if signal.state == SignalState.BREAKEVEN.value:
                    reason = CloseReason.BREAKEVEN_SL
                elif signal.state == SignalState.TRAILING.value:
                    reason = CloseReason.TRAILING_SL
                return self.close_signal(signal.signal_id, price, reason)
        
        else:
            # SHORT: TPs below entry, SL above
            # Check TP3 first (lowest)
            if tp3 > 0 and not signal.tp3_hit and price <= (tp3 * (1 + tol)):
                return self.close_signal(signal.signal_id, price, CloseReason.TP3)
            
            # Check TP2
            if tp2 > 0 and not signal.tp2_hit and price <= (tp2 * (1 + tol)):
                if signal.position_remaining > self.partial_tp2_size:
                    self.close_signal(signal.signal_id, price, CloseReason.TP2,
                                     partial=True, partial_size=self.partial_tp2_size)
                    signal.tp2_hit = True
                    signal.tp1_hit = True
                    return None
                else:
                    return self.close_signal(signal.signal_id, price, CloseReason.TP2)
            
            # Check TP1
            if tp1 > 0 and not signal.tp1_hit and price <= (tp1 * (1 + tol)):
                if signal.position_remaining > self.partial_tp1_size:
                    self.close_signal(signal.signal_id, price, CloseReason.TP1,
                                     partial=True, partial_size=self.partial_tp1_size)
                    signal.tp1_hit = True
                    return None
                else:
                    return self.close_signal(signal.signal_id, price, CloseReason.TP1)
            
            # Check SL
            if sl > 0 and price >= (sl * (1 - tol)):
                reason = CloseReason.SL
                if signal.state == SignalState.BREAKEVEN.value:
                    reason = CloseReason.BREAKEVEN_SL
                elif signal.state == SignalState.TRAILING.value:
                    reason = CloseReason.TRAILING_SL
                return self.close_signal(signal.signal_id, price, reason)
        
        return None
    
    def get_performance_stats(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Args:
            symbol: Optional filter by symbol
            
        Returns:
            Dict with performance metrics
        """
        history = self.history
        if symbol:
            history = [h for h in history if h.get("symbol") == symbol]
        
        if not history:
            return {
                "total": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "avg_winner": 0.0,
                "avg_loser": 0.0,
                "profit_factor": 0.0,
                "avg_duration_hours": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "tp1_count": 0,
                "tp2_count": 0,
                "tp3_count": 0,
                "sl_count": 0,
                "current_streak": 0,
            }
        
        # Calculate stats
        total = len(history)
        pnls = [h.get("realized_pnl_pct", 0) for h in history]
        
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]
        
        wins = len(winners)
        losses = len(losers)
        
        total_pnl = sum(pnls)
        avg_pnl = total_pnl / total if total > 0 else 0
        
        avg_winner = sum(winners) / wins if wins > 0 else 0
        avg_loser = sum(losers) / losses if losses > 0 else 0
        
        gross_profit = sum(winners)
        gross_loss = abs(sum(losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
        
        # Count by result
        tp1_count = sum(1 for h in history if h.get("close_reason") == "TP1")
        tp2_count = sum(1 for h in history if h.get("close_reason") == "TP2")
        tp3_count = sum(1 for h in history if h.get("close_reason") == "TP3")
        sl_count = sum(1 for h in history if "SL" in h.get("close_reason", ""))
        
        # Average duration
        durations = []
        for h in history:
            try:
                created = datetime.fromisoformat(h.get("created_at", "").replace('Z', '+00:00'))
                closed = datetime.fromisoformat(h.get("closed_at", "").replace('Z', '+00:00'))
                duration_hours = (closed - created).total_seconds() / 3600
                durations.append(duration_hours)
            except (ValueError, TypeError):
                pass
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Current streak
        streak = 0
        for h in reversed(history):
            pnl = h.get("realized_pnl_pct", 0)
            if pnl > 0:
                if streak >= 0:
                    streak += 1
                else:
                    break
            elif pnl < 0:
                if streak <= 0:
                    streak -= 1
                else:
                    break
        
        return {
            "total": total,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / total * 100) if total > 0 else 0,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "avg_winner": avg_winner,
            "avg_loser": avg_loser,
            "profit_factor": profit_factor,
            "avg_duration_hours": avg_duration,
            "best_trade": max(pnls) if pnls else 0,
            "worst_trade": min(pnls) if pnls else 0,
            "tp1_count": tp1_count,
            "tp2_count": tp2_count,
            "tp3_count": tp3_count,
            "sl_count": sl_count,
            "current_streak": streak,
        }
    
    def get_symbol_stats(self, symbol: str) -> Dict[str, int]:
        """Get TP/SL counts for a specific symbol."""
        stats = {"TP1": 0, "TP2": 0, "TP3": 0, "SL": 0}
        
        for h in self.history:
            if h.get("symbol") != symbol:
                continue
            reason = h.get("close_reason", "")
            if reason in stats:
                stats[reason] += 1
            elif "SL" in reason:
                stats["SL"] += 1
        
        return stats
    
    def get_last_result(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the last closed signal result.
        
        Args:
            symbol: Optional filter by symbol
            
        Returns:
            Dict with last result info or None
        """
        if not self.history:
            return None
        
        # Search from most recent
        for h in reversed(self.history):
            if symbol is None or h.get("symbol") == symbol:
                return {
                    "symbol": h.get("symbol", ""),
                    "direction": h.get("direction", ""),
                    "result": h.get("close_reason", ""),
                    "pnl_pct": h.get("realized_pnl_pct", 0),
                    "entry": h.get("entry", 0),
                    "exit_price": h.get("exit_price", 0),
                    "closed_at": h.get("closed_at", ""),
                    "duration": h.get("duration_str", ""),
                }
        
        return None
    
    def get_last_results(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the last N closed signal results.
        
        Args:
            limit: Number of results to return
            
        Returns:
            List of result dicts (most recent first)
        """
        results = []
        for h in reversed(self.history[-limit:]):
            results.append({
                "symbol": h.get("symbol", ""),
                "direction": h.get("direction", ""),
                "result": h.get("close_reason", ""),
                "pnl_pct": h.get("realized_pnl_pct", 0),
                "is_win": h.get("close_reason", "").startswith("TP"),
            })
        return results


# Global instances per bot
_instances: Dict[str, SignalManager] = {}
_instance_lock = threading.Lock()


def get_signal_manager(bot_name: str, **kwargs) -> SignalManager:
    """Get or create SignalManager for a bot."""
    with _instance_lock:
        if bot_name not in _instances:
            _instances[bot_name] = SignalManager(bot_name, **kwargs)
        return _instances[bot_name]


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    manager = get_signal_manager("test_bot")
    
    # Create a signal
    signal = Signal(
        symbol="BTC/USDT",
        direction="LONG",
        entry=42000.0,
        stop_loss=41500.0,
        take_profit_1=42500.0,
        take_profit_2=43000.0,
        take_profit_3=44000.0,
    )
    
    manager.add_signal(signal)
    
    print(f"Created signal: {signal.signal_id}")
    print(f"Open signals: {len(manager.get_open_signals())}")
    
    # Get stats
    stats = manager.get_performance_stats()
    print(f"Performance: {stats}")

