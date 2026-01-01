#!/usr/bin/env python3
"""
Portfolio Risk Manager - Cross-Bot Risk Coordination

Provides centralized risk management across all trading bots:
- Track total open positions across all bots
- Enforce maximum portfolio exposure limits
- Detect correlated positions (prevent multiple same-direction trades)
- Emergency shutdown capability
- Real-time portfolio status
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Base directory for state files
BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_STATE_FILE = BASE_DIR / "portfolio_state.json"


@dataclass
class PortfolioConfig:
    """Portfolio-level risk configuration."""
    # Maximum concurrent open positions across ALL bots
    max_total_positions: int = 10
    # Maximum positions per single asset (e.g., max 2 BTC trades)
    max_positions_per_asset: int = 2
    # Maximum positions in same direction (e.g., max 5 longs at once)
    max_same_direction: int = 6
    # Maximum daily drawdown before emergency stop (%)
    max_daily_drawdown_pct: float = 10.0
    # Maximum weekly drawdown before emergency stop (%)
    max_weekly_drawdown_pct: float = 20.0
    # Correlated assets that should count together
    correlation_groups: Dict[str, List[str]] = field(default_factory=lambda: {
        "BTC": ["BTC", "WBTC"],
        "ETH": ["ETH", "WETH", "STETH"],
        "STABLE": ["USDT", "USDC", "DAI", "BUSD"],
    })
    # Emergency stop enabled
    emergency_stop_enabled: bool = True
    # Alert on position count threshold
    position_alert_threshold: int = 8


@dataclass
class OpenPosition:
    """Represents an open position from any bot."""
    signal_id: str
    bot_name: str
    symbol: str
    direction: str  # LONG or SHORT
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    size_usdt: float
    opened_at: str
    unrealized_pnl_pct: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenPosition":
        return cls(**data)


@dataclass
class PortfolioStatus:
    """Current portfolio status."""
    total_positions: int
    long_positions: int
    short_positions: int
    positions_by_bot: Dict[str, int]
    positions_by_asset: Dict[str, int]
    total_exposure_usdt: float
    unrealized_pnl_pct: float
    daily_pnl_pct: float
    weekly_pnl_pct: float
    is_emergency_stopped: bool
    last_updated: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PortfolioRiskManager:
    """
    Centralized portfolio risk manager.
    
    Tracks all open positions across bots and enforces portfolio-level limits.
    Should be used by all bots before opening new positions.
    
    Usage:
        manager = PortfolioRiskManager()
        
        # Check if new position is allowed
        if manager.can_open_position("volume_bot", "BTC/USDT", "LONG"):
            # Open the position
            manager.register_position(position)
        
        # On position close
        manager.close_position(signal_id)
    """
    
    _instance: Optional["PortfolioRiskManager"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "PortfolioRiskManager":
        """Singleton pattern - only one instance across all bots."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Optional[PortfolioConfig] = None):
        if self._initialized:
            return
        
        self.config = config or PortfolioConfig()
        self.positions: Dict[str, OpenPosition] = {}
        self.state_lock = threading.RLock()
        self.emergency_stopped = False
        self.daily_pnl_history: List[Dict[str, Any]] = []
        self.notifier: Optional[Any] = None
        
        # Load existing state
        self._load_state()
        self._initialized = True
        
        logger.info("PortfolioRiskManager initialized with max %d positions", 
                   self.config.max_total_positions)
    
    def set_notifier(self, notifier: Any) -> None:
        """Set the Telegram notifier for alerts."""
        self.notifier = notifier
    
    def _load_state(self) -> None:
        """Load portfolio state from file."""
        if not PORTFOLIO_STATE_FILE.exists():
            return
        
        try:
            with open(PORTFOLIO_STATE_FILE, 'r') as f:
                data = json.load(f)
            
            # Load positions
            positions_data = data.get("positions", {})
            for sig_id, pos_data in positions_data.items():
                try:
                    self.positions[sig_id] = OpenPosition.from_dict(pos_data)
                except (TypeError, KeyError) as e:
                    logger.warning("Failed to load position %s: %s", sig_id, e)
            
            self.emergency_stopped = data.get("emergency_stopped", False)
            self.daily_pnl_history = data.get("daily_pnl_history", [])
            
            logger.info("Loaded %d positions from state file", len(self.positions))
            
        except (json.JSONDecodeError, IOError) as e:
            logger.error("Failed to load portfolio state: %s", e)
    
    def _save_state(self) -> None:
        """Save portfolio state to file (atomic write)."""
        temp_file = PORTFOLIO_STATE_FILE.with_suffix('.tmp')
        
        try:
            data = {
                "positions": {k: v.to_dict() for k, v in self.positions.items()},
                "emergency_stopped": self.emergency_stopped,
                "daily_pnl_history": self.daily_pnl_history[-30:],  # Keep last 30 days
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            temp_file.replace(PORTFOLIO_STATE_FILE)
            
        except Exception as e:
            logger.error("Failed to save portfolio state: %s", e)
            if temp_file.exists():
                temp_file.unlink()
    
    def _get_base_asset(self, symbol: str) -> str:
        """Extract base asset from symbol (e.g., BTC/USDT -> BTC)."""
        return symbol.upper().split("/")[0].replace("USDT", "").replace(":USDT", "")
    
    def _get_correlation_group(self, asset: str) -> str:
        """Get the correlation group for an asset."""
        for group_name, assets in self.config.correlation_groups.items():
            if asset.upper() in [a.upper() for a in assets]:
                return group_name
        return asset  # Return asset itself if not in any group
    
    def can_open_position(
        self, 
        bot_name: str, 
        symbol: str, 
        direction: str,
        check_only: bool = True
    ) -> Tuple[bool, str]:
        """
        Check if a new position can be opened.
        
        Args:
            bot_name: Name of the bot requesting position
            symbol: Trading symbol (e.g., BTC/USDT)
            direction: LONG or SHORT
            check_only: If True, only check without modifying state
            
        Returns:
            Tuple of (can_open: bool, reason: str)
        """
        with self.state_lock:
            # Check emergency stop
            if self.emergency_stopped:
                return False, "Portfolio emergency stop is active"
            
            # Normalize direction
            direction = direction.upper()
            if direction in ["BULLISH", "BUY"]:
                direction = "LONG"
            elif direction in ["BEARISH", "SELL"]:
                direction = "SHORT"
            
            base_asset = self._get_base_asset(symbol)
            correlation_group = self._get_correlation_group(base_asset)
            
            # Count current positions
            total = len(self.positions)
            longs = sum(1 for p in self.positions.values() if p.direction == "LONG")
            shorts = sum(1 for p in self.positions.values() if p.direction == "SHORT")
            
            # Count positions for this asset (including correlated)
            asset_count = 0
            for pos in self.positions.values():
                pos_asset = self._get_base_asset(pos.symbol)
                pos_group = self._get_correlation_group(pos_asset)
                if pos_group == correlation_group:
                    asset_count += 1
            
            # Check total positions
            if total >= self.config.max_total_positions:
                return False, f"Max total positions reached ({total}/{self.config.max_total_positions})"
            
            # Check same direction
            same_dir_count = longs if direction == "LONG" else shorts
            if same_dir_count >= self.config.max_same_direction:
                return False, f"Max {direction} positions reached ({same_dir_count}/{self.config.max_same_direction})"
            
            # Check per-asset limit
            if asset_count >= self.config.max_positions_per_asset:
                return False, f"Max positions for {base_asset} reached ({asset_count}/{self.config.max_positions_per_asset})"
            
            # Alert if approaching limit
            if total >= self.config.position_alert_threshold and self.notifier:
                self._send_alert(
                    f"⚠️ Portfolio nearing limit: {total}/{self.config.max_total_positions} positions",
                    priority="medium"
                )
            
            return True, "OK"
    
    def register_position(self, position: OpenPosition) -> bool:
        """
        Register a new open position.
        
        Args:
            position: The position to register
            
        Returns:
            True if registered successfully, False if rejected
        """
        with self.state_lock:
            # Double-check we can open
            can_open, reason = self.can_open_position(
                position.bot_name, 
                position.symbol, 
                position.direction,
                check_only=False
            )
            
            if not can_open:
                logger.warning("Position rejected: %s - %s", position.signal_id, reason)
                return False
            
            self.positions[position.signal_id] = position
            self._save_state()
            
            logger.info("Position registered: %s %s %s from %s", 
                       position.symbol, position.direction, 
                       position.signal_id, position.bot_name)
            
            return True
    
    def close_position(self, signal_id: str, exit_price: float = 0.0, result: str = "") -> bool:
        """
        Close/remove a position from tracking.
        
        Args:
            signal_id: ID of the signal/position to close
            exit_price: Exit price for P&L calculation
            result: Result of the trade (TP1, TP2, SL, etc.)
            
        Returns:
            True if position was found and removed
        """
        with self.state_lock:
            if signal_id not in self.positions:
                return False
            
            position = self.positions.pop(signal_id)
            
            # Calculate P&L if exit price provided
            if exit_price > 0 and position.entry_price > 0:
                if position.direction == "LONG":
                    pnl_pct = ((exit_price - position.entry_price) / position.entry_price) * 100
                else:
                    pnl_pct = ((position.entry_price - exit_price) / position.entry_price) * 100
                
                logger.info("Position closed: %s %s | %s | P&L: %.2f%%", 
                           position.symbol, position.direction, result, pnl_pct)
            
            self._save_state()
            return True
    
    def update_position_price(self, signal_id: str, current_price: float) -> None:
        """Update current price for a position."""
        with self.state_lock:
            if signal_id in self.positions:
                pos = self.positions[signal_id]
                pos.current_price = current_price
                
                # Calculate unrealized P&L
                if pos.direction == "LONG":
                    pos.unrealized_pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
                else:
                    pos.unrealized_pnl_pct = ((pos.entry_price - current_price) / pos.entry_price) * 100
    
    def get_status(self) -> PortfolioStatus:
        """Get current portfolio status."""
        with self.state_lock:
            positions_by_bot: Dict[str, int] = {}
            positions_by_asset: Dict[str, int] = {}
            total_exposure = 0.0
            total_unrealized_pnl = 0.0
            longs = 0
            shorts = 0
            
            for pos in self.positions.values():
                # Count by bot
                positions_by_bot[pos.bot_name] = positions_by_bot.get(pos.bot_name, 0) + 1
                
                # Count by asset
                base_asset = self._get_base_asset(pos.symbol)
                positions_by_asset[base_asset] = positions_by_asset.get(base_asset, 0) + 1
                
                # Sum exposure
                total_exposure += pos.size_usdt
                total_unrealized_pnl += pos.unrealized_pnl_pct
                
                # Count direction
                if pos.direction == "LONG":
                    longs += 1
                else:
                    shorts += 1
            
            avg_unrealized_pnl = total_unrealized_pnl / len(self.positions) if self.positions else 0.0
            
            return PortfolioStatus(
                total_positions=len(self.positions),
                long_positions=longs,
                short_positions=shorts,
                positions_by_bot=positions_by_bot,
                positions_by_asset=positions_by_asset,
                total_exposure_usdt=total_exposure,
                unrealized_pnl_pct=avg_unrealized_pnl,
                daily_pnl_pct=self._get_daily_pnl(),
                weekly_pnl_pct=self._get_weekly_pnl(),
                is_emergency_stopped=self.emergency_stopped,
                last_updated=datetime.now(timezone.utc).isoformat(),
            )
    
    def _get_daily_pnl(self) -> float:
        """Get today's realized P&L percentage."""
        today = datetime.now(timezone.utc).date().isoformat()
        for entry in self.daily_pnl_history:
            if entry.get("date") == today:
                return entry.get("pnl_pct", 0.0)
        return 0.0
    
    def _get_weekly_pnl(self) -> float:
        """Get this week's realized P&L percentage."""
        week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).date().isoformat()
        total_pnl = 0.0
        for entry in self.daily_pnl_history:
            if entry.get("date", "") >= week_ago:
                total_pnl += entry.get("pnl_pct", 0.0)
        return total_pnl
    
    def record_daily_pnl(self, pnl_pct: float) -> None:
        """Record today's P&L for tracking."""
        today = datetime.now(timezone.utc).date().isoformat()
        
        with self.state_lock:
            # Update or add today's entry
            for entry in self.daily_pnl_history:
                if entry.get("date") == today:
                    entry["pnl_pct"] = pnl_pct
                    self._save_state()
                    return
            
            self.daily_pnl_history.append({
                "date": today,
                "pnl_pct": pnl_pct,
            })
            self._save_state()
    
    def trigger_emergency_stop(self, reason: str) -> None:
        """Trigger emergency stop for all bots."""
        with self.state_lock:
            self.emergency_stopped = True
            self._save_state()
        
        logger.critical("🚨 EMERGENCY STOP TRIGGERED: %s", reason)
        
        if self.notifier:
            self._send_alert(
                f"🚨 EMERGENCY STOP 🚨\n\n"
                f"Reason: {reason}\n\n"
                f"All new positions blocked.\n"
                f"Manual intervention required.",
                priority="critical"
            )
    
    def reset_emergency_stop(self) -> None:
        """Reset emergency stop (manual action required)."""
        with self.state_lock:
            self.emergency_stopped = False
            self._save_state()
        
        logger.info("Emergency stop reset")
    
    def check_drawdown_limits(self) -> None:
        """Check if drawdown limits are breached and trigger emergency stop if needed."""
        if not self.config.emergency_stop_enabled:
            return
        
        daily_pnl = self._get_daily_pnl()
        weekly_pnl = self._get_weekly_pnl()
        
        if daily_pnl <= -self.config.max_daily_drawdown_pct:
            self.trigger_emergency_stop(
                f"Daily drawdown limit breached: {daily_pnl:.2f}% (limit: -{self.config.max_daily_drawdown_pct}%)"
            )
        elif weekly_pnl <= -self.config.max_weekly_drawdown_pct:
            self.trigger_emergency_stop(
                f"Weekly drawdown limit breached: {weekly_pnl:.2f}% (limit: -{self.config.max_weekly_drawdown_pct}%)"
            )
    
    def _send_alert(self, message: str, priority: str = "normal") -> None:
        """Send alert via notifier."""
        if self.notifier:
            try:
                self.notifier.send_message(message, parse_mode="HTML")
            except Exception as e:
                logger.error("Failed to send alert: %s", e)
    
    def get_positions_summary(self) -> str:
        """Get a formatted summary of all open positions."""
        with self.state_lock:
            if not self.positions:
                return "📊 No open positions"
            
            lines = ["📊 <b>Open Positions</b>\n"]
            
            # Group by bot
            by_bot: Dict[str, List[OpenPosition]] = {}
            for pos in self.positions.values():
                if pos.bot_name not in by_bot:
                    by_bot[pos.bot_name] = []
                by_bot[pos.bot_name].append(pos)
            
            for bot_name, positions in sorted(by_bot.items()):
                lines.append(f"\n<b>{bot_name}</b> ({len(positions)})")
                for pos in positions:
                    emoji = "🟢" if pos.direction == "LONG" else "🔴"
                    pnl_emoji = "📈" if pos.unrealized_pnl_pct > 0 else "📉"
                    lines.append(
                        f"  {emoji} {pos.symbol} | {pnl_emoji} {pos.unrealized_pnl_pct:+.2f}%"
                    )
            
            status = self.get_status()
            lines.append(f"\n<b>Total:</b> {status.total_positions} positions")
            lines.append(f"<b>Long/Short:</b> {status.long_positions}/{status.short_positions}")
            lines.append(f"<b>Unrealized P&L:</b> {status.unrealized_pnl_pct:+.2f}%")
            
            return "\n".join(lines)
    
    def sync_from_bot_states(self, bot_state_files: Dict[str, Path]) -> int:
        """
        Sync positions from individual bot state files.
        
        Args:
            bot_state_files: Dict of {bot_name: state_file_path}
            
        Returns:
            Number of positions synced
        """
        synced = 0
        
        for bot_name, state_file in bot_state_files.items():
            if not state_file.exists():
                continue
            
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                open_signals = state.get("open_signals", {})
                for signal_id, signal_data in open_signals.items():
                    if signal_id in self.positions:
                        continue  # Already tracked
                    
                    # Create position from signal data
                    position = OpenPosition(
                        signal_id=signal_id,
                        bot_name=bot_name,
                        symbol=signal_data.get("symbol", "UNKNOWN"),
                        direction=signal_data.get("direction", "LONG"),
                        entry_price=float(signal_data.get("entry", 0)),
                        current_price=float(signal_data.get("entry", 0)),
                        stop_loss=float(signal_data.get("stop_loss", 0)),
                        take_profit_1=float(signal_data.get("take_profit_1", 0)),
                        take_profit_2=float(signal_data.get("take_profit_2", 0)),
                        size_usdt=0.0,  # Not tracked in signals
                        opened_at=signal_data.get("created_at", datetime.now(timezone.utc).isoformat()),
                    )
                    
                    self.positions[signal_id] = position
                    synced += 1
                    
            except (json.JSONDecodeError, IOError, KeyError) as e:
                logger.warning("Failed to sync from %s: %s", state_file, e)
        
        if synced > 0:
            self._save_state()
            logger.info("Synced %d positions from bot states", synced)
        
        return synced


# Convenience function to get the singleton instance
def get_portfolio_manager() -> PortfolioRiskManager:
    """Get the global PortfolioRiskManager instance."""
    return PortfolioRiskManager()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    manager = get_portfolio_manager()
    
    # Check if we can open a position
    can_open, reason = manager.can_open_position("test_bot", "BTC/USDT", "LONG")
    print(f"Can open BTC LONG: {can_open} - {reason}")
    
    # Get status
    status = manager.get_status()
    print(f"Total positions: {status.total_positions}")
    print(f"Emergency stopped: {status.is_emergency_stopped}")

