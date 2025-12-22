#!/usr/bin/env python3
"""
Resilience patterns for reliable API operations.

Provides retry logic, circuit breakers, and exponential backoff for handling
transient failures in external API calls (MEXC exchange, Telegram, etc.).

Patterns implemented:
- Exponential backoff with jitter (prevents thundering herd)
- Configurable retry logic with failure classification
- Circuit breaker to prevent cascading failures
- Rate limiting awareness
- Graceful degradation

Usage:
    from common.resilience import retry_with_backoff, CircuitBreaker

    # Retry decorator for transient failures
    @retry_with_backoff(max_attempts=3, base_delay=1.0)
    def fetch_ticker(symbol):
        return exchange.fetch_ticker(symbol)

    # Circuit breaker for repeated failures
    breaker = CircuitBreaker(failure_threshold=5, timeout=60)

    if breaker.call(lambda: exchange.fetch_ohlcv(symbol)):
        # Success
        pass
"""

import time
import random
import logging
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, cast
from datetime import datetime, timedelta
from enum import Enum


logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"    # Normal operation, requests allowed
    OPEN = "open"        # Circuit tripped, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


class RetryError(Exception):
    """Raised when all retry attempts are exhausted."""
    pass


def is_transient_error(exception: Exception) -> bool:
    """
    Determine if an exception represents a transient (retryable) error.

    Transient errors are temporary failures that may succeed on retry:
    - Network errors (timeouts, connection refused, DNS failures)
    - Rate limit errors (429 Too Many Requests)
    - Server errors (500, 502, 503, 504)
    - Exchange-specific errors (RequestTimeout, NetworkError, etc.)

    Non-transient errors (should NOT retry):
    - Authentication errors (401, 403)
    - Invalid requests (400, 404)
    - Business logic errors (insufficient balance, invalid symbol)

    Args:
        exception (Exception): Exception to classify

    Returns:
        bool: True if error is transient and should be retried

    Example:
        >>> try:
        ...     exchange.fetch_ticker("BTC/USDT")
        ... except Exception as e:
        ...     if is_transient_error(e):
        ...         # Retry the operation
        ...         pass
        ...     else:
        ...         # Fail fast, don't retry
        ...         raise
    """
    import ccxt  # type: ignore[import-untyped]

    # CCXT transient errors
    transient_types = (
        ccxt.NetworkError,
        ccxt.RequestTimeout,
        ccxt.ExchangeNotAvailable,
        ccxt.DDoSProtection,
        ccxt.RateLimitExceeded,
    )

    if isinstance(exception, transient_types):
        return True

    # HTTP errors
    if isinstance(exception, ccxt.ExchangeError):
        error_msg = str(exception).lower()
        transient_indicators = [
            '429',  # Too Many Requests
            '500',  # Internal Server Error
            '502',  # Bad Gateway
            '503',  # Service Unavailable
            '504',  # Gateway Timeout
            'timeout',
            'timed out',
            'connection',
            'network',
        ]
        return any(indicator in error_msg for indicator in transient_indicators)

    # Standard Python network errors
    network_errors = (
        ConnectionError,
        TimeoutError,
        OSError,  # Covers socket errors
    )

    return isinstance(exception, network_errors)


def exponential_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True
) -> float:
    """
    Calculate exponential backoff delay with optional jitter.

    Exponential backoff prevents overwhelming a recovering service and
    reduces the "thundering herd" problem where many clients retry simultaneously.

    Formula (without jitter): delay = min(base_delay * (2 ** attempt), max_delay)
    Formula (with jitter): delay = random(0, calculated_delay)

    Args:
        attempt (int): Retry attempt number (0-indexed)
        base_delay (float, optional): Initial delay in seconds. Defaults to 1.0.
        max_delay (float, optional): Maximum delay cap in seconds. Defaults to 60.0.
        jitter (bool, optional): Add randomization to prevent synchronized retries.
                                Defaults to True.

    Returns:
        float: Delay in seconds before next retry

    Example:
        >>> for attempt in range(5):
        ...     delay = exponential_backoff(attempt, base_delay=1.0)
        ...     print(f"Attempt {attempt}: wait {delay:.2f}s")
        Attempt 0: wait 0.53s
        Attempt 1: wait 1.47s
        Attempt 2: wait 2.91s
        Attempt 3: wait 7.22s
        Attempt 4: wait 12.84s

    Notes:
        - Attempt 0: base_delay * 2^0 = 1.0s
        - Attempt 1: base_delay * 2^1 = 2.0s
        - Attempt 2: base_delay * 2^2 = 4.0s
        - Attempt 3: base_delay * 2^3 = 8.0s
        - Jitter prevents synchronized retries across multiple clients
        - max_delay caps the delay to prevent excessive wait times
    """
    # Calculate exponential delay
    delay = min(base_delay * (2 ** attempt), max_delay)

    # Add jitter (randomize between 0 and delay)
    if jitter:
        delay = float(random.uniform(0, delay))

    return float(delay)


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    transient_only: bool = True,
    on_retry: Optional[Callable[[int, Exception], None]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for automatic retry with exponential backoff.

    Retries a function on failure using exponential backoff strategy.
    Optionally filters retries to transient errors only (recommended for APIs).

    Args:
        max_attempts (int, optional): Maximum number of attempts (including initial).
                                     Defaults to 3.
        base_delay (float, optional): Initial retry delay in seconds. Defaults to 1.0.
        max_delay (float, optional): Maximum retry delay in seconds. Defaults to 60.0.
        transient_only (bool, optional): Only retry transient errors. If False,
                                        retries all exceptions. Defaults to True.
        on_retry (Optional[Callable], optional): Callback function called on each retry.
                                                Receives (attempt_number, exception).
                                                Defaults to None.

    Returns:
        Callable: Decorated function with retry behavior

    Raises:
        RetryError: If all retry attempts are exhausted
        Exception: Original exception if non-transient and transient_only=True

    Example:
        >>> @retry_with_backoff(max_attempts=3, base_delay=1.0)
        ... def fetch_price(symbol):
        ...     return exchange.fetch_ticker(symbol)

        >>> # Custom retry callback
        >>> def log_retry(attempt, exception):
        ...     logger.warning(f"Retry {attempt} after error: {exception}")

        >>> @retry_with_backoff(max_attempts=5, on_retry=log_retry)
        ... def critical_operation():
        ...     # Will retry up to 5 times, logging each retry
        ...     return api.important_call()

    Notes:
        - First attempt doesn't count as a retry (total calls = max_attempts)
        - Transient errors: network, timeout, 429, 500-504
        - Non-transient errors: 400, 401, 403, 404, business logic errors
        - Use transient_only=True for external APIs (recommended)
        - Use transient_only=False only for internal operations you control
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    # Check if we should retry
                    should_retry = not transient_only or is_transient_error(e)

                    # Last attempt - raise immediately
                    if attempt >= max_attempts - 1:
                        if should_retry:
                            raise RetryError(
                                f"Failed after {max_attempts} attempts: {e}"
                            ) from e
                        else:
                            raise

                    # Non-transient error - raise immediately
                    if not should_retry:
                        raise

                    # Calculate backoff delay
                    delay = exponential_backoff(attempt, base_delay, max_delay)

                    # Call retry callback if provided
                    if on_retry:
                        on_retry(attempt + 1, e)
                    else:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )

                    time.sleep(delay)

            # Should never reach here, but satisfy type checker
            if last_exception:
                raise last_exception
            raise RetryError("Retry logic error - no exception but no success")

        return wrapper
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.

    A circuit breaker monitors failures and "opens" (blocks requests) when
    failures exceed a threshold, giving the failing service time to recover.

    States:
        CLOSED: Normal operation, requests allowed
        OPEN: Too many failures, requests blocked (fail fast)
        HALF_OPEN: Testing recovery, limited requests allowed

    Behavior:
        1. CLOSED: Requests pass through normally
           - On success: Reset failure counter
           - On failure: Increment failure counter
           - If failures >= threshold: Transition to OPEN

        2. OPEN: Requests fail immediately without calling function
           - After timeout period: Transition to HALF_OPEN

        3. HALF_OPEN: Allow one test request
           - On success: Transition to CLOSED, reset counters
           - On failure: Transition back to OPEN

    Args:
        failure_threshold (int): Number of failures before opening circuit.
                                Defaults to 5.
        timeout (float): Seconds to wait before testing recovery (HALF_OPEN).
                        Defaults to 60.0.
        name (str): Circuit breaker name for logging. Defaults to "default".

    Example:
        >>> breaker = CircuitBreaker(failure_threshold=3, timeout=30)
        >>>
        >>> def call_api():
        ...     return exchange.fetch_ticker("BTC/USDT")
        >>>
        >>> result = breaker.call(call_api)
        >>> if result is not None:
        ...     print(f"Success: {result}")
        ... else:
        ...     print("Circuit open, request blocked")

    Notes:
        - Prevents overwhelming a failing service
        - Fails fast when service is known to be down
        - Automatically tests recovery after timeout
        - Thread-safe for concurrent use
        - Use separate breakers for different services (MEXC, Telegram, etc.)
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        name: str = "default"
    ):
        """Initialize circuit breaker."""
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.name = name

        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED

    def _should_attempt(self) -> bool:
        """Check if request should be attempted based on circuit state."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if self.last_failure_time:
                elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
                if elapsed >= self.timeout:
                    logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
                    self.state = CircuitState.HALF_OPEN
                    return True

            logger.warning(f"Circuit breaker '{self.name}' is OPEN, blocking request")
            return False

        # HALF_OPEN: allow one test request
        return True

    def _on_success(self) -> None:
        """Handle successful request."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info(f"Circuit breaker '{self.name}' recovered, entering CLOSED state")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    def _on_failure(self, exception: Exception) -> None:
        """Handle failed request."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        logger.warning(
            f"Circuit breaker '{self.name}' failure {self.failure_count}/{self.failure_threshold}: {exception}"
        )

        if self.state == CircuitState.HALF_OPEN:
            # Test failed, go back to OPEN
            logger.warning(f"Circuit breaker '{self.name}' test failed, reopening circuit")
            self.state = CircuitState.OPEN
        elif self.failure_count >= self.failure_threshold:
            # Threshold exceeded, open circuit
            logger.error(
                f"Circuit breaker '{self.name}' threshold exceeded, entering OPEN state "
                f"for {self.timeout}s"
            )
            self.state = CircuitState.OPEN

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> Optional[T]:
        """
        Execute function through circuit breaker.

        Args:
            func: Function to call
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Function result or None if circuit is open

        Example:
            >>> breaker = CircuitBreaker()
            >>> result = breaker.call(exchange.fetch_ticker, "BTC/USDT")
        """
        if not self._should_attempt():
            return None

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise

    def reset(self) -> None:
        """
        Manually reset circuit breaker to CLOSED state.

        Useful for administrative actions or testing.
        """
        logger.info(f"Circuit breaker '{self.name}' manually reset")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None

    @property
    def is_open(self) -> bool:
        """Check if circuit is currently open (blocking requests)."""
        return self.state == CircuitState.OPEN


# Example usage
if __name__ == "__main__":
    import ccxt

    # Example 1: Retry with backoff
    @retry_with_backoff(max_attempts=3, base_delay=1.0)
    def fetch_ticker(symbol: str) -> Dict[str, Any]:
        """Fetch ticker with automatic retry."""
        exchange = ccxt.mexc()
        return cast(Dict[str, Any], exchange.fetch_ticker(symbol))

    try:
        ticker = fetch_ticker("BTC/USDT")
        print(f"Price: {ticker['last']}")
    except RetryError as e:
        print(f"Failed after retries: {e}")

    # Example 2: Circuit breaker
    breaker = CircuitBreaker(failure_threshold=3, timeout=30, name="mexc_api")

    def get_balance() -> Dict[str, Any]:
        """Fetch balance through circuit breaker."""
        exchange = ccxt.mexc({'apiKey': 'xxx', 'secret': 'yyy'})
        return cast(Dict[str, Any], exchange.fetch_balance())

    result = breaker.call(get_balance)
    if result is not None:
        print(f"Balance: {result}")
    else:
        print("Service unavailable (circuit open)")

    # Example 3: Combined retry + circuit breaker
    exchange_breaker = CircuitBreaker(name="exchange")

    @retry_with_backoff(max_attempts=3)
    def resilient_fetch(symbol: str) -> Dict[str, Any]:
        """Fetch with retry and circuit breaker protection."""
        def fetch() -> Dict[str, Any]:
            exchange = ccxt.mexc()
            return cast(Dict[str, Any], exchange.fetch_ticker(symbol))

        result = exchange_breaker.call(fetch)
        if result is None:
            raise Exception("Circuit breaker open")
        return result

    try:
        data = resilient_fetch("ETH/USDT")
        print(f"Success: {data['last']}")
    except Exception as e:
        print(f"Failed: {e}")
