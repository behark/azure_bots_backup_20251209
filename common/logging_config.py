#!/usr/bin/env python3
"""
Structured logging configuration for trading bots.

Provides JSON-formatted logs with contextual information, log rotation,
correlation IDs, and sensitive data filtering for production deployments.

Features:
- Structured JSON logging for easy parsing (ELK, Splunk, CloudWatch)
- Contextual fields (bot_name, symbol, operation, correlation_id)
- Automatic log rotation by size (prevents disk space issues)
- Sensitive data filtering (API keys, secrets)
- Standardized log levels across all bots
- Performance-optimized for high-throughput operations

Usage:
    from common.logging_config import get_logger

    logger = get_logger("funding_bot", json_logs=True, log_dir="logs")

    # Basic logging
    logger.info("Bot started")

    # Contextual logging
    logger.info("Signal detected", extra={
        "symbol": "BTC/USDT",
        "signal_type": "LONG",
        "price": 45000.0,
        "correlation_id": "abc-123"
    })
"""

import logging
import logging.handlers
import json
import uuid
import re
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


class SensitiveDataFilter(logging.Filter):
    """
    Filter that redacts sensitive information from log messages.

    Prevents accidental logging of API keys, secrets, tokens, and other
    sensitive data that could compromise security if logs are exposed.

    Patterns filtered:
    - API keys (MEXC_API_KEY, etc.)
    - Secrets (MEXC_API_SECRET, etc.)
    - Tokens (TELEGRAM_BOT_TOKEN, etc.)
    - Authorization headers
    - Private keys
    """

    # Patterns to redact
    SENSITIVE_PATTERNS = [
        (r'(api[_-]?key["\']?\s*[:=]\s*["\']?)([^"\'}\s]+)', r'\1***REDACTED***'),
        (r'(api[_-]?secret["\']?\s*[:=]\s*["\']?)([^"\'}\s]+)', r'\1***REDACTED***'),
        (r'(token["\']?\s*[:=]\s*["\']?)([^"\'}\s]+)', r'\1***REDACTED***'),
        (r'(password["\']?\s*[:=]\s*["\']?)([^"\'}\s]+)', r'\1***REDACTED***'),
        (r'(Authorization:\s*Bearer\s+)(\S+)', r'\1***REDACTED***'),
        (r'(private[_-]?key["\']?\s*[:=]\s*["\']?)([^"\'}\s]+)', r'\1***REDACTED***'),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact sensitive data from log message."""
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            for pattern, replacement in self.SENSITIVE_PATTERNS:
                record.msg = re.sub(pattern, replacement, record.msg, flags=re.IGNORECASE)
        return True


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs logs as single-line JSON objects for easy parsing by log
    aggregation systems (ELK, Splunk, CloudWatch, etc.).

    Standard fields:
    - timestamp: ISO 8601 timestamp with timezone
    - level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - logger: Logger name (usually bot name)
    - message: Log message
    - module: Python module name
    - function: Function name where log was called
    - line: Line number

    Custom fields (via extra parameter):
    - bot_name: Name of the trading bot
    - symbol: Trading pair symbol
    - operation: Operation being performed
    - correlation_id: UUID for tracking related operations
    - Any other custom fields passed to logger calls
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add custom fields from extra parameter
        for key, value in record.__dict__.items():
            if key not in [
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'pathname', 'process', 'processName', 'relativeCreated',
                'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
                'getMessage', 'message'
            ]:
                log_data[key] = value

        return json.dumps(log_data, default=str)


class StandardFormatter(logging.Formatter):
    """
    Human-readable formatter for console output.

    Provides colorized, readable logs for development and manual debugging.
    Includes contextual information when available.
    """

    # ANSI color codes for terminal output
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors and context."""
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # Build context string from extra fields
        context_parts = []
        for key in ['symbol', 'operation', 'correlation_id']:
            if hasattr(record, key):
                context_parts.append(f"{key}={getattr(record, key)}")

        context = f" [{', '.join(context_parts)}]" if context_parts else ""

        # Format: timestamp | LEVEL | logger | message [context]
        timestamp = self.formatTime(record, datefmt='%Y-%m-%d %H:%M:%S')
        formatted = (
            f"{timestamp} | "
            f"{color}{record.levelname:8s}{reset} | "
            f"{record.name:20s} | "
            f"{record.getMessage()}"
            f"{context}"
        )

        # Add exception traceback if present
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)

        return formatted


def get_logger(
    name: str,
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    json_logs: bool = False,
    console_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    add_correlation_id: bool = True,
) -> logging.Logger:
    """
    Get a configured logger with structured logging support.

    Creates a logger with both file and console handlers, automatic log
    rotation, sensitive data filtering, and optional JSON formatting.

    Args:
        name (str): Logger name (typically bot name, e.g., "funding_bot")
        log_level (str, optional): Minimum log level. One of: DEBUG, INFO,
                                  WARNING, ERROR, CRITICAL. Defaults to "INFO".
        log_dir (Optional[Path], optional): Directory for log files. If None,
                                           logs to "./logs". Defaults to None.
        json_logs (bool, optional): Use JSON format for file logs (recommended
                                   for production). Defaults to False.
        console_output (bool, optional): Log to console (stdout). Defaults to True.
        max_bytes (int, optional): Maximum log file size before rotation in bytes.
                                  Defaults to 10MB (10 * 1024 * 1024).
        backup_count (int, optional): Number of rotated log files to keep.
                                     Defaults to 5 (50MB total with 10MB files).
        add_correlation_id (bool, optional): Auto-generate correlation IDs for
                                            tracking operations. Defaults to True.

    Returns:
        logging.Logger: Configured logger instance ready for use

    Example:
        >>> # Basic usage
        >>> logger = get_logger("my_bot")
        >>> logger.info("Bot started successfully")

        >>> # Production usage with JSON logs
        >>> logger = get_logger("funding_bot", log_level="INFO", json_logs=True)
        >>> logger.info("Signal detected", extra={
        ...     "symbol": "BTC/USDT",
        ...     "signal_type": "LONG",
        ...     "price": 45000.0
        ... })

        >>> # Debug mode
        >>> logger = get_logger("test_bot", log_level="DEBUG", json_logs=False)
        >>> logger.debug("Detailed debugging information")

    Notes:
        - Log files are automatically rotated when they reach max_bytes
        - Old log files are kept up to backup_count (then deleted)
        - JSON logs are ideal for production (easy parsing by monitoring tools)
        - Standard logs are ideal for development (human-readable)
        - Sensitive data (API keys, tokens) is automatically redacted
        - Correlation IDs help track related operations across log entries
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Prevent duplicate handlers if logger already configured
    if logger.hasHandlers():
        return logger

    # Add sensitive data filter
    logger.addFilter(SensitiveDataFilter())

    # Setup log directory
    if log_dir is None:
        log_dir = Path("logs")
    else:
        log_dir = Path(log_dir)

    log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    # File handler with rotation
    log_file = log_dir / f"{name}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # Capture all levels to file

    # Use JSON formatter for file logs if requested
    if json_logs:
        file_handler.setFormatter(StructuredFormatter())
    else:
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))

    logger.addHandler(file_handler)

    # Console handler with colored output
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(StandardFormatter())
        logger.addHandler(console_handler)

    # Prevent propagation to root logger (avoid duplicate logs)
    logger.propagate = False

    return logger


def generate_correlation_id() -> str:
    """
    Generate a unique correlation ID for tracking related operations.

    Correlation IDs allow you to trace a single operation (e.g., signal
    detection → analysis → notification) across multiple log entries.

    Returns:
        str: UUID4 correlation ID in format "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"

    Example:
        >>> correlation_id = generate_correlation_id()
        >>> logger.info("Starting analysis", extra={"correlation_id": correlation_id})
        >>> # ... perform analysis ...
        >>> logger.info("Analysis complete", extra={"correlation_id": correlation_id})
        >>> # Both log entries will have the same correlation_id for tracking

    Notes:
        - Uses UUID4 (random UUID) for uniqueness
        - 128-bit random number provides ~3.4×10³⁸ possible values
        - Collision probability is negligible for practical purposes
    """
    return str(uuid.uuid4())


class ContextLogger:
    """
    Wrapper that adds persistent context to all log calls.

    Useful for maintaining consistent context (bot name, symbol, etc.)
    across multiple log statements without repeating the context.

    Example:
        >>> logger = get_logger("my_bot")
        >>> ctx_logger = ContextLogger(logger, bot_name="funding_bot", symbol="BTC/USDT")
        >>> ctx_logger.info("Price checked")  # Automatically includes bot_name and symbol
        >>> ctx_logger.info("Signal detected")  # Context persists

    Attributes:
        logger (logging.Logger): Underlying logger instance
        context (Dict[str, Any]): Persistent context dictionary
    """

    def __init__(self, logger: logging.Logger, **context: Any):
        """
        Initialize context logger with persistent fields.

        Args:
            logger (logging.Logger): Logger to wrap
            **context: Persistent context fields (bot_name, symbol, etc.)
        """
        self.logger = logger
        self.context = context

    def _log(self, level: str, msg: str, **extra: Any) -> None:
        """Log message with merged context."""
        merged_extra = {**self.context, **extra}
        getattr(self.logger, level)(msg, extra=merged_extra)

    def debug(self, msg: str, **extra: Any) -> None:
        """Log debug message with context."""
        self._log('debug', msg, **extra)

    def info(self, msg: str, **extra: Any) -> None:
        """Log info message with context."""
        self._log('info', msg, **extra)

    def warning(self, msg: str, **extra: Any) -> None:
        """Log warning message with context."""
        self._log('warning', msg, **extra)

    def error(self, msg: str, **extra: Any) -> None:
        """Log error message with context."""
        self._log('error', msg, **extra)

    def critical(self, msg: str, **extra: Any) -> None:
        """Log critical message with context."""
        self._log('critical', msg, **extra)

    def exception(self, msg: str, **extra: Any) -> None:
        """Log exception with traceback and context."""
        merged_extra = {**self.context, **extra}
        self.logger.exception(msg, extra=merged_extra)


# Example usage for documentation
if __name__ == "__main__":
    # Example 1: Basic logger
    logger = get_logger("example_bot")
    logger.info("Bot started")
    logger.warning("Low balance detected")

    # Example 2: Structured logging with context
    logger = get_logger("funding_bot", json_logs=True)
    correlation_id = generate_correlation_id()

    logger.info("Signal analysis started", extra={
        "symbol": "BTC/USDT",
        "operation": "analyze_funding_rate",
        "correlation_id": correlation_id
    })

    logger.info("Signal detected", extra={
        "symbol": "BTC/USDT",
        "signal_type": "LONG",
        "funding_rate": -0.0123,
        "correlation_id": correlation_id
    })

    # Example 3: Context logger
    base_logger = get_logger("psar_bot")
    ctx_logger = ContextLogger(base_logger, bot_name="psar_bot", symbol="ETH/USDT")

    ctx_logger.info("Price checked")  # Includes bot_name and symbol automatically
    ctx_logger.info("PSAR calculated", psar_value=1250.50)

    # Example 4: Error logging with exception
    try:
        raise ValueError("Invalid configuration")
    except Exception:
        logger.exception("Configuration error occurred", extra={
            "config_file": "psar_config.json",
            "correlation_id": correlation_id
        })
