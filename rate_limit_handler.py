"""
Rate Limit Handler for Trading Bots

This module provides centralized rate limiting and error handling for exchange API calls.
It implements exponential backoff for rate limit errors (HTTP 510) and automatic retry logic.

Usage:
    from rate_limit_handler import RateLimitHandler
    
    handler = RateLimitHandler(base_delay=0.5, max_retries=5)
    
    # Wrap API calls
    result = handler.execute(exchange.fetch_ticker, symbol)
"""

import time
import logging
from typing import Any, Callable, Optional, TypeVar
from functools import wraps

F = TypeVar('F', bound=Callable[..., Any])

logger = logging.getLogger(__name__)


class RateLimitHandler:
    """Handles API rate limiting with exponential backoff and automatic retries."""
    
    def __init__(
        self, 
        base_delay: float = 0.5,
        max_retries: int = 5,
        backoff_factor: float = 2.0,
        max_backoff: float = 30.0
    ):
        """
        Initialize the rate limit handler.
        
        Args:
            base_delay: Minimum delay between API calls (seconds)
            max_retries: Maximum number of retry attempts
            backoff_factor: Multiplier for exponential backoff
            max_backoff: Maximum delay between retries (seconds)
        """
        self.base_delay = base_delay
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff
        self.last_call_time: float = 0.0

    def _apply_base_delay(self) -> None:
        """Apply the base delay between API calls."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.base_delay:
            sleep_time = self.base_delay - time_since_last_call
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = self.base_delay * (self.backoff_factor ** attempt)
        return min(delay, self.max_backoff)
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if the error is a rate limit error (HTTP 510)."""
        error_str = str(error).lower()
        return (
            '510' in error_str or
            'requests are too frequent' in error_str or
            'rate limit' in error_str or
            'too many requests' in error_str
        )
    
    def execute(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """
        Execute a function with rate limiting and automatic retry.
        
        Args:
            func: The function to call (e.g., exchange.fetch_ticker)
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function call
            
        Raises:
            Exception: If all retry attempts fail
        """
        # Apply base delay before making the call
        self._apply_base_delay()
        
        last_error: Exception = RuntimeError(f"API call failed after {self.max_retries} attempts")
        
        for attempt in range(self.max_retries):
            try:
                result = func(*args, **kwargs)
                
                # Reset last call time on success
                self.last_call_time = time.time()
                
                if attempt > 0:
                    logger.info(f"API call succeeded after {attempt} retries")
                
                return result
                
            except Exception as e:
                last_error = e
                
                # Check if it's a rate limit error
                if self._is_rate_limit_error(e):
                    if attempt < self.max_retries - 1:
                        backoff_delay = self._calculate_backoff(attempt)
                        logger.warning(
                            f"Rate limit hit (attempt {attempt + 1}/{self.max_retries}). "
                            f"Retrying in {backoff_delay:.2f}s... Error: {str(e)[:100]}"
                        )
                        time.sleep(backoff_delay)
                        continue
                    else:
                        logger.error(
                            f"Rate limit error after {self.max_retries} attempts: {str(e)}"
                        )
                else:
                    # Not a rate limit error, raise immediately
                    logger.error(f"API call failed with non-rate-limit error: {str(e)}")
                    raise
        
        # All retries exhausted
        logger.error(
            f"API call failed after {self.max_retries} attempts. "
            f"Last error: {str(last_error)}"
        )
        raise last_error


class RateLimitedExchange:
    """Wrapper for exchange objects that adds rate limiting to all API calls."""

    def __init__(self, exchange: Any, handler: Optional[RateLimitHandler] = None) -> None:
        """
        Initialize rate-limited exchange wrapper.
        
        Args:
            exchange: The exchange object (e.g., ccxt exchange)
            handler: Optional custom RateLimitHandler instance
        """
        self._exchange = exchange
        self._handler = handler or RateLimitHandler()
        
    def __getattr__(self, name: str) -> Any:
        """
        Intercept method calls and wrap API methods with rate limiting.
        """
        attr = getattr(self._exchange, name)

        # If it's a callable method that looks like an API call, wrap it
        if callable(attr) and (
            name.startswith('fetch_') or
            name.startswith('create_') or
            name.startswith('cancel_') or
            name in ['load_markets', 'fetch_balance', 'fetch_ohlcv']
        ):
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                return self._handler.execute(attr, *args, **kwargs)
            return wrapped

        return attr


def rate_limited(
    base_delay: float = 0.5,
    max_retries: int = 5,
    backoff_factor: float = 2.0,
    max_backoff: float = 30.0
) -> Callable[[F], F]:
    """
    Decorator to add rate limiting to individual functions.

    Usage:
        @rate_limited(base_delay=1.0, max_retries=3)
        def fetch_data():
            return exchange.fetch_ticker('BTC/USDT')
    """
    handler = RateLimitHandler(base_delay, max_retries, backoff_factor, max_backoff)

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return handler.execute(func, *args, **kwargs)
        return wrapper  # type: ignore[return-value]
    return decorator


# Global instance for simple use cases
global_handler = RateLimitHandler()


def safe_api_call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Simple function to make a rate-limited API call using the global handler.
    
    Usage:
        ticker = safe_api_call(exchange.fetch_ticker, 'BTC/USDT')
    """
    return global_handler.execute(func, *args, **kwargs)
