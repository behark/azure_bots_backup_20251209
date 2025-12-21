#!/usr/bin/env python3
"""
Unified Trade Configuration System
Centralized configuration for TP/SL calculations, risk management, and position sizing.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "global_config.json"


@dataclass
class RiskConfig:
    """Risk management configuration."""
    # ATR multipliers for stop loss
    sl_atr_multiplier: float = 1.5
    # ATR multipliers for take profits
    tp1_atr_multiplier: float = 2.0
    tp2_atr_multiplier: float = 3.0
    tp3_atr_multiplier: float = 4.5
    # Minimum risk-reward ratio (if below this, skip signal)
    min_risk_reward: float = 1.3
    # Maximum risk per trade (% of account)
    max_risk_percent: float = 2.0
    # Use trailing stop
    use_trailing_stop: bool = False
    trailing_stop_activation: float = 1.0  # ATR units of profit before trailing activates
    trailing_stop_distance: float = 0.5    # ATR units for trailing distance
    # Buffer for stop loss (% added to SL to avoid wicks)
    sl_buffer_percent: float = 0.75


@dataclass
class SymbolConfig:
    """Per-symbol configuration."""
    symbol: str
    enabled: bool = True
    period: str = "15m"
    cooldown_minutes: int = 30
    # Override global risk settings per symbol
    sl_atr_multiplier: Optional[float] = None
    tp1_atr_multiplier: Optional[float] = None
    tp2_atr_multiplier: Optional[float] = None
    min_risk_reward: Optional[float] = None
    # Symbol-specific settings
    min_volume_24h: float = 0  # Minimum 24h volume in USDT
    max_spread_percent: float = 1.0  # Maximum spread to trade
    # Direction filter (None = both, "LONG" or "SHORT")
    direction_filter: Optional[str] = None


@dataclass
class BotConfig:
    """Configuration for a trading bot."""
    bot_name: str
    enabled: bool = True
    # Global risk settings
    risk: RiskConfig = field(default_factory=RiskConfig)
    # Symbol configurations
    symbols: List[SymbolConfig] = field(default_factory=list)
    # Bot-specific settings
    max_open_signals: int = 7
    interval_seconds: int = 300
    default_cooldown_minutes: int = 30


class TradeConfigManager:
    """Manages trade configurations for all bots."""
    
    def __init__(self, config_file: Path = CONFIG_FILE):
        self.config_file = config_file
        self.configs: Dict[str, BotConfig] = {}
        self.global_risk = RiskConfig()
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file."""
        if not self.config_file.exists():
            logger.info("No config file found, using defaults")
            return
        
        try:
            data = json.loads(self.config_file.read_text())
            
            # Load global risk settings
            if "global_risk" in data:
                self.global_risk = RiskConfig(**data["global_risk"])
            
            # Load bot configs
            for bot_name, bot_data in data.get("bots", {}).items():
                risk_data = bot_data.pop("risk", {})
                symbols_data = bot_data.pop("symbols", [])
                
                risk = RiskConfig(**{**asdict(self.global_risk), **risk_data})
                symbols = [SymbolConfig(**s) for s in symbols_data]
                
                self.configs[bot_name] = BotConfig(
                    bot_name=bot_name,
                    risk=risk,
                    symbols=symbols,
                    **bot_data
                )
                
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to load config: {e}")
    
    def save_config(self) -> None:
        """Save configuration to file."""
        data = {
            "global_risk": asdict(self.global_risk),
            "bots": {}
        }
        
        for bot_name, config in self.configs.items():
            bot_data = {
                "enabled": config.enabled,
                "max_open_signals": config.max_open_signals,
                "interval_seconds": config.interval_seconds,
                "default_cooldown_minutes": config.default_cooldown_minutes,
                "risk": asdict(config.risk),
                "symbols": [asdict(s) for s in config.symbols],
            }
            data["bots"][bot_name] = bot_data
        
        self.config_file.write_text(json.dumps(data, indent=2))
        logger.info(f"Config saved to {self.config_file}")
    
    def get_bot_config(self, bot_name: str) -> BotConfig:
        """Get configuration for a specific bot."""
        if bot_name not in self.configs:
            self.configs[bot_name] = BotConfig(
                bot_name=bot_name,
                risk=RiskConfig(**asdict(self.global_risk))
            )
        return self.configs[bot_name]
    
    def get_symbol_config(self, bot_name: str, symbol: str) -> SymbolConfig:
        """Get configuration for a specific symbol in a bot."""
        bot_config = self.get_bot_config(bot_name)
        
        for sym_config in bot_config.symbols:
            if sym_config.symbol.upper() == symbol.upper():
                return sym_config
        
        # Return default config if not found
        return SymbolConfig(symbol=symbol)
    
    def get_effective_risk(self, bot_name: str, symbol: str) -> RiskConfig:
        """Get effective risk config merging global, bot, and symbol settings."""
        bot_config = self.get_bot_config(bot_name)
        sym_config = self.get_symbol_config(bot_name, symbol)
        
        # Start with bot's risk config
        effective = RiskConfig(**asdict(bot_config.risk))
        
        # Override with symbol-specific settings if present
        if sym_config.sl_atr_multiplier is not None:
            effective.sl_atr_multiplier = sym_config.sl_atr_multiplier
        if sym_config.tp1_atr_multiplier is not None:
            effective.tp1_atr_multiplier = sym_config.tp1_atr_multiplier
        if sym_config.tp2_atr_multiplier is not None:
            effective.tp2_atr_multiplier = sym_config.tp2_atr_multiplier
        if sym_config.min_risk_reward is not None:
            effective.min_risk_reward = sym_config.min_risk_reward
        
        return effective


# Global instance
_config_manager: Optional[TradeConfigManager] = None


def get_config_manager() -> TradeConfigManager:
    """Get the global config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = TradeConfigManager()
    return _config_manager


def create_default_config() -> None:
    """Create a default configuration file."""
    default_symbols = [
        "POWER", "BLUAI", "IRYS", "VVV", "ON", "CLO", 
        "APR", "KITE", "LAB", "MINA", "RLS"
    ]
    
    config = {
        "global_risk": {
            "sl_atr_multiplier": 1.5,
            "tp1_atr_multiplier": 2.0,
            "tp2_atr_multiplier": 3.5,
            "tp3_atr_multiplier": 5.0,
            "min_risk_reward": 1.5,
            "max_risk_percent": 2.0,
            "use_trailing_stop": False,
            "trailing_stop_activation": 1.0,
            "trailing_stop_distance": 0.5,
            "sl_buffer_percent": 0.5
        },
        "bots": {
            "funding_bot": {
                "enabled": True,
                "max_open_signals": 7,
                "interval_seconds": 300,
                "default_cooldown_minutes": 30,
                "risk": {
                    "sl_atr_multiplier": 1.5,
                    "tp1_atr_multiplier": 2.5,
                    "tp2_atr_multiplier": 4.0
                },
                "symbols": [{"symbol": s, "period": "15m", "cooldown_minutes": 30} for s in default_symbols]
            },
            "liquidation_bot": {
                "enabled": True,
                "max_open_signals": 7,
                "interval_seconds": 300,
                "default_cooldown_minutes": 30,
                "risk": {
                    "sl_atr_multiplier": 1.8,
                    "tp1_atr_multiplier": 3.0,
                    "tp2_atr_multiplier": 5.0
                },
                "symbols": [{"symbol": s, "period": "5m", "cooldown_minutes": 30} for s in default_symbols]
            },
            "harmonic_bot": {
                "enabled": True,
                "max_open_signals": 5,
                "interval_seconds": 300,
                "default_cooldown_minutes": 30,
                "risk": {
                    "sl_atr_multiplier": 2.0,
                    "tp1_atr_multiplier": 2.5,
                    "tp2_atr_multiplier": 4.0,
                    "min_risk_reward": 2.0
                },
                "symbols": [{"symbol": s, "period": "15m", "cooldown_minutes": 30} for s in default_symbols]
            },
            "candlestick_bot": {
                "enabled": True,
                "max_open_signals": 5,
                "interval_seconds": 300,
                "default_cooldown_minutes": 30,
                "risk": {
                    "sl_atr_multiplier": 1.5,
                    "tp1_atr_multiplier": 1.5,
                    "tp2_atr_multiplier": 2.5
                },
                "symbols": [{"symbol": s, "period": "15m", "cooldown_minutes": 30} for s in default_symbols]
            },
            "volume_bot": {
                "enabled": True,
                "max_open_signals": 7,
                "interval_seconds": 300,
                "default_cooldown_minutes": 30,
                "risk": {
                    "sl_atr_multiplier": 1.5,
                    "tp1_atr_multiplier": 2.0,
                    "tp2_atr_multiplier": 3.0
                },
                "symbols": [{"symbol": s, "period": "15m", "cooldown_minutes": 30} for s in default_symbols]
            },
            "psar_bot": {
                "enabled": True,
                "max_open_signals": 5,
                "interval_seconds": 300,
                "default_cooldown_minutes": 30,
                "risk": {
                    "sl_atr_multiplier": 1.0,
                    "tp1_atr_multiplier": 2.0,
                    "tp2_atr_multiplier": 3.5,
                    "use_trailing_stop": True
                },
                "symbols": [{"symbol": s, "period": "15m", "cooldown_minutes": 30} for s in default_symbols]
            },
            "most_bot": {
                "enabled": True,
                "max_open_signals": 5,
                "interval_seconds": 300,
                "default_cooldown_minutes": 30,
                "risk": {
                    "sl_atr_multiplier": 1.0,
                    "tp1_atr_multiplier": 2.0,
                    "tp2_atr_multiplier": 3.5,
                    "use_trailing_stop": True
                },
                "symbols": [{"symbol": s, "period": "15m", "cooldown_minutes": 30} for s in default_symbols]
            },
            "mtf_bot": {
                "enabled": True,
                "max_open_signals": 5,
                "interval_seconds": 300,
                "default_cooldown_minutes": 30,
                "risk": {
                    "sl_atr_multiplier": 1.5,
                    "tp1_atr_multiplier": 2.5,
                    "tp2_atr_multiplier": 4.0,
                    "min_risk_reward": 1.8
                },
                "symbols": [{"symbol": s, "period": "15m", "cooldown_minutes": 30} for s in default_symbols]
            },
            "diy_bot": {
                "enabled": True,
                "max_open_signals": 5,
                "interval_seconds": 300,
                "default_cooldown_minutes": 30,
                "risk": {
                    "sl_atr_multiplier": 1.5,
                    "tp1_atr_multiplier": 2.5,
                    "tp2_atr_multiplier": 4.0
                },
                "symbols": [{"symbol": s, "period": "15m", "cooldown_minutes": 30} for s in default_symbols]
            },
            "strat_bot": {
                "enabled": True,
                "max_open_signals": 5,
                "interval_seconds": 300,
                "default_cooldown_minutes": 30,
                "risk": {
                    "sl_atr_multiplier": 1.5,
                    "tp1_atr_multiplier": 2.0,
                    "tp2_atr_multiplier": 3.0
                },
                "symbols": [{"symbol": s, "period": "15m", "cooldown_minutes": 30} for s in default_symbols]
            },
            "fib_reversal_bot": {
                "enabled": True,
                "max_open_signals": 5,
                "interval_seconds": 300,
                "default_cooldown_minutes": 30,
                "risk": {
                    "sl_atr_multiplier": 1.5,
                    "tp1_atr_multiplier": 2.0,
                    "tp2_atr_multiplier": 3.5,
                    "min_risk_reward": 1.5
                },
                "symbols": [{"symbol": s, "period": "15m", "cooldown_minutes": 30} for s in default_symbols]
            },
            "fib_swing_bot": {
                "enabled": True,
                "max_open_signals": 5,
                "interval_seconds": 300,
                "default_cooldown_minutes": 30,
                "risk": {
                    "sl_atr_multiplier": 1.5,
                    "tp1_atr_multiplier": 2.5,
                    "tp2_atr_multiplier": 4.0,
                    "min_risk_reward": 2.0
                },
                "symbols": [{"symbol": s, "period": "15m", "cooldown_minutes": 30} for s in default_symbols]
            }
        }
    }
    
    CONFIG_FILE.write_text(json.dumps(config, indent=2))
    print(f"Default config created: {CONFIG_FILE}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Trade Configuration Manager")
    parser.add_argument("--create-default", action="store_true", help="Create default config file")
    parser.add_argument("--show", action="store_true", help="Show current config")
    args = parser.parse_args()
    
    if args.create_default:
        create_default_config()
    elif args.show:
        manager = get_config_manager()
        print(json.dumps({
            "global_risk": asdict(manager.global_risk),
            "bots": {k: v.bot_name for k, v in manager.configs.items()}
        }, indent=2))
    else:
        parser.print_help()
