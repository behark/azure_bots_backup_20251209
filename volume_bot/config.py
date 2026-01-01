#!/usr/bin/env python3
"""Configuration module for Volume VN Bot with environment variable management."""

import os
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("volume_vn.config")


@dataclass
class AnalysisConfig:
    """Analysis parameters."""
    candle_limit: int = 200
    request_timeout_seconds: int = 30
    volume_spike_threshold: float = 1.5
    buying_pressure_threshold: float = 1.2
    buying_pressure_sample_size: int = 3
    rsi_periods: int = 14


@dataclass
class RiskManagementConfig:
    """Risk management parameters."""
    max_open_signals: int = 50
    default_stop_loss_pct: float = 1.5  # 1.5% stop loss
    min_risk_pct: float = 1.0  # Minimum 1% risk per trade
    tp1_multiplier: float = 2.0
    tp2_multiplier: float = 3.0
    min_risk_reward: float = 1.5
    max_position_size_usd: float = 10000
    max_daily_loss_usd: float = 500
    max_total_risk_pct: float = 5.0  # Max 5% of account at risk


@dataclass
class SignalManagementConfig:
    """Signal lifecycle management."""
    cooldown_minutes: int = 5
    max_signal_age_hours: int = 24
    allow_simultaneous_directions: bool = False
    check_exchange_for_duplicates: bool = True
    check_timeframe_for_duplicates: bool = False
    result_notification_cooldown_minutes: float = 15.0
    enable_result_notifications: bool = True
    min_confidence_score: float = 2.0


@dataclass
class RateLimitingConfig:
    """Rate limiting configuration."""
    base_delay_seconds: float = 0.5
    max_retries: int = 5
    backoff_multiplier: float = 2.0
    calls_per_minute: int = 60
    rate_limit_backoff_base: float = 60.0  # seconds
    rate_limit_backoff_max: float = 300.0  # seconds


@dataclass
class ExecutionConfig:
    """Bot execution parameters."""
    cycle_interval_seconds: int = 60
    symbol_delay_seconds: int = 5
    health_check_interval_seconds: int = 3600
    enable_startup_notification: bool = True
    enable_shutdown_notification: bool = True
    enable_detailed_logging: bool = False
    log_level: str = "INFO"


@dataclass
class FilteringConfig:
    """Signal filtering parameters."""
    min_factors_long: int = 4
    min_factors_short: int = 4
    require_volume_spike: bool = False
    require_near_hvn: bool = False
    min_rsi_long: int = 45
    max_rsi_long: int = 65
    min_rsi_short: int = 35
    max_rsi_short: int = 55


@dataclass
class ExchangeCredentials:
    """Exchange API credentials."""
    api_key: Optional[str] = None
    secret: Optional[str] = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.secret)


@dataclass
class TelegramConfig:
    """Telegram notification configuration."""
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate Telegram configuration."""
        if not self.bot_token:
            return False, "TELEGRAM_BOT_TOKEN is missing"
        if not self.chat_id:
            return False, "TELEGRAM_CHAT_ID is missing"
        # Basic format validation
        if not self.bot_token.count(':') >= 1:
            return False, "Invalid Telegram bot token format"
        if not (self.chat_id.startswith('-') or self.chat_id.lstrip('-').isdigit()):
            return False, "Invalid Telegram chat ID format"
        return True, None


class VolumeConfig:
    """Centralized configuration for Volume VN Bot."""

    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file
        self.analysis = AnalysisConfig()
        self.risk = RiskManagementConfig()
        self.signal = SignalManagementConfig()
        self.rate_limit = RateLimitingConfig()
        self.execution = ExecutionConfig()
        self.filtering = FilteringConfig()

        # Load from environment first
        self._load_from_env()

        # Override with config file if provided
        if config_file and config_file.exists():
            self._load_from_file(config_file)

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Analysis
        self.analysis.candle_limit = int(os.getenv('VOLUME_BOT_CANDLE_LIMIT', self.analysis.candle_limit))
        self.analysis.request_timeout_seconds = int(os.getenv('VOLUME_BOT_REQUEST_TIMEOUT', self.analysis.request_timeout_seconds))
        self.analysis.volume_spike_threshold = float(os.getenv('VOLUME_BOT_SPIKE_THRESHOLD', self.analysis.volume_spike_threshold))
        self.analysis.buying_pressure_threshold = float(os.getenv('VOLUME_BOT_PRESSURE_THRESHOLD', self.analysis.buying_pressure_threshold))

        # Risk Management
        self.risk.max_open_signals = int(os.getenv('VOLUME_BOT_MAX_OPEN_SIGNALS', self.risk.max_open_signals))
        self.risk.default_stop_loss_pct = float(os.getenv('VOLUME_BOT_STOP_LOSS_PCT', self.risk.default_stop_loss_pct))
        self.risk.min_risk_pct = float(os.getenv('VOLUME_BOT_MIN_RISK_PCT', self.risk.min_risk_pct))
        self.risk.tp1_multiplier = float(os.getenv('VOLUME_BOT_TP1_MULTIPLIER', self.risk.tp1_multiplier))
        self.risk.tp2_multiplier = float(os.getenv('VOLUME_BOT_TP2_MULTIPLIER', self.risk.tp2_multiplier))
        self.risk.min_risk_reward = float(os.getenv('VOLUME_BOT_MIN_RISK_REWARD', self.risk.min_risk_reward))
        self.risk.max_position_size_usd = float(os.getenv('VOLUME_BOT_MAX_POSITION_USD', self.risk.max_position_size_usd))

        # Signal Management
        self.signal.cooldown_minutes = int(os.getenv('VOLUME_BOT_COOLDOWN_MINUTES', self.signal.cooldown_minutes))
        self.signal.max_signal_age_hours = int(os.getenv('VOLUME_BOT_MAX_SIGNAL_AGE_HOURS', self.signal.max_signal_age_hours))
        
        # Rate Limiting
        self.rate_limit.base_delay_seconds = float(os.getenv('VOLUME_BOT_BASE_DELAY', self.rate_limit.base_delay_seconds))
        self.rate_limit.max_retries = int(os.getenv('VOLUME_BOT_MAX_RETRIES', self.rate_limit.max_retries))
        self.rate_limit.calls_per_minute = int(os.getenv('VOLUME_BOT_CALLS_PER_MINUTE', self.rate_limit.calls_per_minute))

        # Execution
        self.execution.cycle_interval_seconds = int(os.getenv('VOLUME_BOT_CYCLE_INTERVAL', self.execution.cycle_interval_seconds))
        self.execution.symbol_delay_seconds = int(os.getenv('VOLUME_BOT_SYMBOL_DELAY', self.execution.symbol_delay_seconds))

    def _load_from_file(self, config_file: Path) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            volume_config = data.get('volume_bot', {})
            
            # Analysis
            if 'analysis' in volume_config:
                for key, value in volume_config['analysis'].items():
                    if hasattr(self.analysis, key):
                        setattr(self.analysis, key, value)
            
            # Risk Management
            if 'risk_management' in volume_config:
                for key, value in volume_config['risk_management'].items():
                    if hasattr(self.risk, key):
                        setattr(self.risk, key, value)
            
            # Signal Management
            if 'signal_management' in volume_config:
                for key, value in volume_config['signal_management'].items():
                    if hasattr(self.signal, key):
                        setattr(self.signal, key, value)
            
            # Rate Limiting
            if 'rate_limiting' in volume_config:
                for key, value in volume_config['rate_limiting'].items():
                    if hasattr(self.rate_limit, key):
                        setattr(self.rate_limit, key, value)
            
            # Execution
            if 'execution' in volume_config:
                for key, value in volume_config['execution'].items():
                    if hasattr(self.execution, key):
                        setattr(self.execution, key, value)

            # Filtering
            if 'filtering' in volume_config:
                for key, value in volume_config['filtering'].items():
                    if hasattr(self.filtering, key):
                        setattr(self.filtering, key, value)

            logger.info("Loaded configuration from %s", config_file)
        except Exception as e:
            logger.warning("Failed to load config file %s: %s", config_file, e)

    def get_exchange_credentials(self, exchange: str) -> ExchangeCredentials:
        """Get API credentials for exchange."""
        exchange_upper = exchange.upper()
        api_key = os.getenv(f'{exchange_upper}_API_KEY')
        secret = os.getenv(f'{exchange_upper}_SECRET')
        return ExchangeCredentials(api_key=api_key, secret=secret)

    def get_telegram_config(self) -> TelegramConfig:
        """Get Telegram configuration."""
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN_VOLUME') or os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        return TelegramConfig(bot_token=bot_token, chat_id=chat_id)

    def validate_environment(self) -> tuple[bool, list[str]]:
        """Validate all required environment variables are set."""
        errors = []
        
        # Check Telegram (required for notifications)
        telegram = self.get_telegram_config()
        is_valid, error = telegram.validate()
        if not is_valid and error:
            errors.append(f"Telegram: {error}")
        
        # Check at least one exchange is configured
        has_exchange = False
        for exchange in ['binanceusdm', 'mexc', 'bybit']:
            creds = self.get_exchange_credentials(exchange)
            if creds.is_configured:
                has_exchange = True
                break
        
        if not has_exchange:
            errors.append("No exchange credentials configured (need at least one: BINANCEUSDM_API_KEY/SECRET, MEXC_API_KEY/SECRET, or BYBIT_API_KEY/SECRET)")
        
        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            'analysis': {
                'candle_limit': self.analysis.candle_limit,
                'request_timeout_seconds': self.analysis.request_timeout_seconds,
                'volume_spike_threshold': self.analysis.volume_spike_threshold,
                'buying_pressure_threshold': self.analysis.buying_pressure_threshold,
                'buying_pressure_sample_size': self.analysis.buying_pressure_sample_size,
            },
            'risk_management': {
                'max_open_signals': self.risk.max_open_signals,
                'default_stop_loss_pct': self.risk.default_stop_loss_pct,
                'min_risk_pct': self.risk.min_risk_pct,
                'tp1_multiplier': self.risk.tp1_multiplier,
                'tp2_multiplier': self.risk.tp2_multiplier,
                'min_risk_reward': self.risk.min_risk_reward,
                'max_position_size_usd': self.risk.max_position_size_usd,
                'max_daily_loss_usd': self.risk.max_daily_loss_usd,
            },
            'signal_management': {
                'cooldown_minutes': self.signal.cooldown_minutes,
                'max_signal_age_hours': self.signal.max_signal_age_hours,
                'allow_simultaneous_directions': self.signal.allow_simultaneous_directions,
            },
            'rate_limiting': {
                'base_delay_seconds': self.rate_limit.base_delay_seconds,
                'max_retries': self.rate_limit.max_retries,
                'calls_per_minute': self.rate_limit.calls_per_minute,
            },
            'execution': {
                'cycle_interval_seconds': self.execution.cycle_interval_seconds,
                'symbol_delay_seconds': self.execution.symbol_delay_seconds,
                'health_check_interval_seconds': self.execution.health_check_interval_seconds,
                'enable_detailed_logging': self.execution.enable_detailed_logging,
                'log_level': self.execution.log_level,
            },
            'filtering': {
                'min_factors_long': self.filtering.min_factors_long,
                'min_factors_short': self.filtering.min_factors_short,
                'require_volume_spike': self.filtering.require_volume_spike,
                'require_near_hvn': self.filtering.require_near_hvn,
                'min_rsi_long': self.filtering.min_rsi_long,
                'max_rsi_long': self.filtering.max_rsi_long,
                'min_rsi_short': self.filtering.min_rsi_short,
                'max_rsi_short': self.filtering.max_rsi_short,
            },
        }


def load_config(config_file: Optional[Path] = None) -> VolumeConfig:
    """Load configuration from environment and optional file."""
    return VolumeConfig(config_file=config_file)
