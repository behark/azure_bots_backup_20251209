"""
Unit tests for Resilience Module (common/resilience.py).

Tests retry logic, circuit breakers, exponential backoff, and error classification.
Target: 80%+ code coverage

Run tests:
    python3 -m pytest tests/test_resilience.py -v
    python3 -m pytest tests/test_resilience.py --cov=common.resilience --cov-report=term-missing
"""

import pytest  # type: ignore[import-not-found]
import sys
import time
from pathlib import Path
from typing import Any, List, Tuple
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.resilience import (
    CircuitState,
    RetryError,
    is_transient_error,
    exponential_backoff,
    retry_with_backoff,
    CircuitBreaker,
)


class TestCircuitState:
    """Test CircuitState enum."""

    def test_circuit_states_exist(self) -> None:
        """Test that all circuit states are defined."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"

    def test_circuit_state_count(self) -> None:
        """Test that there are exactly 3 states."""
        assert len(CircuitState) == 3


class TestRetryError:
    """Test RetryError exception."""

    def test_retry_error_is_exception(self) -> None:
        """Test that RetryError is an Exception."""
        assert issubclass(RetryError, Exception)

    def test_retry_error_message(self) -> None:
        """Test RetryError with message."""
        error = RetryError("All retries exhausted")
        assert str(error) == "All retries exhausted"

    def test_retry_error_can_be_raised(self) -> None:
        """Test that RetryError can be raised and caught."""
        with pytest.raises(RetryError) as exc_info:
            raise RetryError("Test error")
        assert "Test error" in str(exc_info.value)


class TestIsTransientError:
    """Test is_transient_error function."""

    def test_network_error_is_transient(self) -> None:
        """Test that ccxt NetworkError is transient."""
        with patch.dict('sys.modules', {'ccxt': MagicMock()}):
            import ccxt  # type: ignore[import-not-found]

            # Create mock exception that is instance of NetworkError
            mock_error = MagicMock(spec=Exception)
            mock_error.__class__ = type('NetworkError', (Exception,), {})

            # Mock the isinstance check by making ccxt.NetworkError a tuple
            ccxt.NetworkError = type('NetworkError', (Exception,), {})
            ccxt.RequestTimeout = type('RequestTimeout', (Exception,), {})
            ccxt.ExchangeNotAvailable = type('ExchangeNotAvailable', (Exception,), {})
            ccxt.DDoSProtection = type('DDoSProtection', (Exception,), {})
            ccxt.RateLimitExceeded = type('RateLimitExceeded', (Exception,), {})
            ccxt.ExchangeError = type('ExchangeError', (Exception,), {})

            # Test actual NetworkError
            network_error = ccxt.NetworkError("Connection failed")
            result = is_transient_error(network_error)
            assert result is True

    def test_timeout_error_is_transient(self) -> None:
        """Test that ccxt RequestTimeout is transient."""
        with patch.dict('sys.modules', {'ccxt': MagicMock()}):
            import ccxt

            ccxt.NetworkError = type('NetworkError', (Exception,), {})
            ccxt.RequestTimeout = type('RequestTimeout', (Exception,), {})
            ccxt.ExchangeNotAvailable = type('ExchangeNotAvailable', (Exception,), {})
            ccxt.DDoSProtection = type('DDoSProtection', (Exception,), {})
            ccxt.RateLimitExceeded = type('RateLimitExceeded', (Exception,), {})
            ccxt.ExchangeError = type('ExchangeError', (Exception,), {})

            timeout_error = ccxt.RequestTimeout("Request timed out")
            result = is_transient_error(timeout_error)
            assert result is True

    def test_rate_limit_error_is_transient(self) -> None:
        """Test that ccxt RateLimitExceeded is transient."""
        with patch.dict('sys.modules', {'ccxt': MagicMock()}):
            import ccxt

            ccxt.NetworkError = type('NetworkError', (Exception,), {})
            ccxt.RequestTimeout = type('RequestTimeout', (Exception,), {})
            ccxt.ExchangeNotAvailable = type('ExchangeNotAvailable', (Exception,), {})
            ccxt.DDoSProtection = type('DDoSProtection', (Exception,), {})
            ccxt.RateLimitExceeded = type('RateLimitExceeded', (Exception,), {})
            ccxt.ExchangeError = type('ExchangeError', (Exception,), {})

            rate_limit_error = ccxt.RateLimitExceeded("429 Too Many Requests")
            result = is_transient_error(rate_limit_error)
            assert result is True

    def test_connection_error_is_transient(self) -> None:
        """Test that Python ConnectionError is transient."""
        error = ConnectionError("Connection refused")
        result = is_transient_error(error)
        assert result is True

    def test_timeout_error_python_is_transient(self) -> None:
        """Test that Python TimeoutError is transient."""
        error = TimeoutError("Operation timed out")
        result = is_transient_error(error)
        assert result is True

    def test_os_error_is_transient(self) -> None:
        """Test that OSError (socket errors) is transient."""
        error = OSError("Socket error")
        result = is_transient_error(error)
        assert result is True

    def test_value_error_is_not_transient(self) -> None:
        """Test that ValueError is not transient."""
        error = ValueError("Invalid value")
        result = is_transient_error(error)
        assert result is False

    def test_key_error_is_not_transient(self) -> None:
        """Test that KeyError is not transient."""
        error = KeyError("missing_key")
        result = is_transient_error(error)
        assert result is False

    def test_exchange_error_with_429_is_transient(self) -> None:
        """Test that ExchangeError with 429 status is transient."""
        with patch.dict('sys.modules', {'ccxt': MagicMock()}):
            import ccxt

            ccxt.NetworkError = type('NetworkError', (Exception,), {})
            ccxt.RequestTimeout = type('RequestTimeout', (Exception,), {})
            ccxt.ExchangeNotAvailable = type('ExchangeNotAvailable', (Exception,), {})
            ccxt.DDoSProtection = type('DDoSProtection', (Exception,), {})
            ccxt.RateLimitExceeded = type('RateLimitExceeded', (Exception,), {})
            ccxt.ExchangeError = type('ExchangeError', (Exception,), {})

            error = ccxt.ExchangeError("429 Too Many Requests")
            result = is_transient_error(error)
            assert result is True

    def test_exchange_error_with_500_is_transient(self) -> None:
        """Test that ExchangeError with 500 status is transient."""
        with patch.dict('sys.modules', {'ccxt': MagicMock()}):
            import ccxt

            ccxt.NetworkError = type('NetworkError', (Exception,), {})
            ccxt.RequestTimeout = type('RequestTimeout', (Exception,), {})
            ccxt.ExchangeNotAvailable = type('ExchangeNotAvailable', (Exception,), {})
            ccxt.DDoSProtection = type('DDoSProtection', (Exception,), {})
            ccxt.RateLimitExceeded = type('RateLimitExceeded', (Exception,), {})
            ccxt.ExchangeError = type('ExchangeError', (Exception,), {})

            error = ccxt.ExchangeError("500 Internal Server Error")
            result = is_transient_error(error)
            assert result is True

    def test_exchange_error_with_503_is_transient(self) -> None:
        """Test that ExchangeError with 503 status is transient."""
        with patch.dict('sys.modules', {'ccxt': MagicMock()}):
            import ccxt

            ccxt.NetworkError = type('NetworkError', (Exception,), {})
            ccxt.RequestTimeout = type('RequestTimeout', (Exception,), {})
            ccxt.ExchangeNotAvailable = type('ExchangeNotAvailable', (Exception,), {})
            ccxt.DDoSProtection = type('DDoSProtection', (Exception,), {})
            ccxt.RateLimitExceeded = type('RateLimitExceeded', (Exception,), {})
            ccxt.ExchangeError = type('ExchangeError', (Exception,), {})

            error = ccxt.ExchangeError("503 Service Unavailable")
            result = is_transient_error(error)
            assert result is True

    def test_exchange_error_with_timeout_message_is_transient(self) -> None:
        """Test that ExchangeError with timeout in message is transient."""
        with patch.dict('sys.modules', {'ccxt': MagicMock()}):
            import ccxt

            ccxt.NetworkError = type('NetworkError', (Exception,), {})
            ccxt.RequestTimeout = type('RequestTimeout', (Exception,), {})
            ccxt.ExchangeNotAvailable = type('ExchangeNotAvailable', (Exception,), {})
            ccxt.DDoSProtection = type('DDoSProtection', (Exception,), {})
            ccxt.RateLimitExceeded = type('RateLimitExceeded', (Exception,), {})
            ccxt.ExchangeError = type('ExchangeError', (Exception,), {})

            error = ccxt.ExchangeError("Connection timeout occurred")
            result = is_transient_error(error)
            assert result is True


class TestExponentialBackoff:
    """Test exponential_backoff function."""

    def test_first_attempt_base_delay(self) -> None:
        """Test that first attempt uses base delay."""
        delay = exponential_backoff(0, base_delay=1.0, jitter=False)
        assert delay == 1.0

    def test_second_attempt_doubles(self) -> None:
        """Test that second attempt doubles the delay."""
        delay = exponential_backoff(1, base_delay=1.0, jitter=False)
        assert delay == 2.0

    def test_third_attempt_quadruples(self) -> None:
        """Test that third attempt quadruples the base delay."""
        delay = exponential_backoff(2, base_delay=1.0, jitter=False)
        assert delay == 4.0

    def test_exponential_growth(self) -> None:
        """Test exponential growth pattern."""
        delays = [exponential_backoff(i, base_delay=1.0, jitter=False) for i in range(5)]
        assert delays == [1.0, 2.0, 4.0, 8.0, 16.0]

    def test_max_delay_cap(self) -> None:
        """Test that delay is capped at max_delay."""
        delay = exponential_backoff(10, base_delay=1.0, max_delay=60.0, jitter=False)
        assert delay == 60.0

    def test_custom_base_delay(self) -> None:
        """Test custom base delay."""
        delay = exponential_backoff(0, base_delay=2.0, jitter=False)
        assert delay == 2.0

        delay = exponential_backoff(1, base_delay=2.0, jitter=False)
        assert delay == 4.0

    def test_jitter_adds_randomness(self) -> None:
        """Test that jitter adds randomness to delay."""
        delays = [exponential_backoff(2, base_delay=1.0, jitter=True) for _ in range(10)]

        # All delays should be between 0 and 4.0 (base * 2^2)
        for delay in delays:
            assert 0 <= delay <= 4.0

        # With jitter, not all values should be identical
        unique_delays = set(delays)
        assert len(unique_delays) > 1  # Should have some variation

    def test_jitter_respects_max_delay(self) -> None:
        """Test that jitter respects max_delay."""
        delays = [exponential_backoff(10, base_delay=1.0, max_delay=10.0, jitter=True) for _ in range(10)]

        for delay in delays:
            assert 0 <= delay <= 10.0

    def test_returns_float(self) -> None:
        """Test that function returns a float."""
        delay = exponential_backoff(0, base_delay=1.0)
        assert isinstance(delay, float)


class TestRetryWithBackoff:
    """Test retry_with_backoff decorator."""

    def test_successful_function_returns_immediately(self) -> None:
        """Test that successful function returns without retry."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        def successful_func() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()
        assert result == "success"
        assert call_count == 1

    def test_retries_on_transient_error(self) -> None:
        """Test that function retries on transient errors."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.01, transient_only=False)
        def flaky_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count == 3

    def test_raises_retry_error_after_max_attempts(self) -> None:
        """Test that RetryError is raised after max attempts."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.01, transient_only=False)
        def always_fails() -> str:
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")

        with pytest.raises(RetryError) as exc_info:
            always_fails()

        assert call_count == 3
        assert "Failed after 3 attempts" in str(exc_info.value)

    def test_non_transient_error_not_retried(self) -> None:
        """Test that non-transient errors are not retried."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.01, transient_only=True)
        def raises_value_error() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid input")

        with pytest.raises(ValueError):
            raises_value_error()

        assert call_count == 1  # Should not retry

    def test_on_retry_callback_called(self) -> None:
        """Test that on_retry callback is called on each retry."""
        retry_calls: List[tuple[int, Exception]] = []

        def on_retry(attempt: int, exception: Exception) -> None:
            retry_calls.append((attempt, exception))

        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.01, transient_only=False, on_retry=on_retry)
        def flaky_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert len(retry_calls) == 2  # Two retries before success
        assert retry_calls[0][0] == 1  # First retry
        assert retry_calls[1][0] == 2  # Second retry

    def test_transient_only_false_retries_all(self) -> None:
        """Test that transient_only=False retries all exceptions."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.01, transient_only=False)
        def raises_value_error() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Will be retried")
            return "success"

        result = raises_value_error()
        assert result == "success"
        assert call_count == 3

    def test_preserves_function_name(self) -> None:
        """Test that decorator preserves function name."""
        @retry_with_backoff(max_attempts=3)
        def my_function() -> str:
            return "test"

        assert my_function.__name__ == "my_function"

    def test_passes_arguments(self) -> None:
        """Test that arguments are passed through correctly."""
        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        def add(a: int, b: int) -> int:
            return a + b

        result = add(2, 3)
        assert result == 5

    def test_passes_kwargs(self) -> None:
        """Test that keyword arguments are passed through correctly."""
        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        result = greet("World", greeting="Hi")
        assert result == "Hi, World!"

    def test_last_attempt_non_transient_raises_original(self) -> None:
        """Test that non-transient error on last attempt raises original exception."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.01, transient_only=True)
        def mixed_errors() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient error")
            # Last attempt raises non-transient error
            raise ValueError("Non-transient on last attempt")

        with pytest.raises(ValueError) as exc_info:
            mixed_errors()

        assert call_count == 3
        assert "Non-transient on last attempt" in str(exc_info.value)


class TestCircuitBreaker:
    """Test CircuitBreaker class."""

    def test_initial_state_is_closed(self) -> None:
        """Test that circuit starts in CLOSED state."""
        breaker = CircuitBreaker()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.is_open is False

    def test_custom_parameters(self) -> None:
        """Test circuit breaker with custom parameters."""
        breaker = CircuitBreaker(failure_threshold=10, timeout=120.0, name="custom")
        assert breaker.failure_threshold == 10
        assert breaker.timeout == 120.0
        assert breaker.name == "custom"

    def test_successful_call_returns_result(self) -> None:
        """Test that successful call returns the result."""
        breaker = CircuitBreaker()

        def success_func() -> str:
            return "success"

        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.failure_count == 0

    def test_successful_call_resets_failure_count(self) -> None:
        """Test that successful call resets failure count."""
        breaker = CircuitBreaker(failure_threshold=5)

        # Simulate some failures
        breaker.failure_count = 3

        def success_func() -> str:
            return "success"

        breaker.call(success_func)
        assert breaker.failure_count == 0

    def test_failure_increments_count(self) -> None:
        """Test that failure increments failure count."""
        breaker = CircuitBreaker(failure_threshold=5)

        def fail_func() -> str:
            raise ValueError("Error")

        with pytest.raises(ValueError):
            breaker.call(fail_func)

        assert breaker.failure_count == 1

    def test_circuit_opens_at_threshold(self) -> None:
        """Test that circuit opens when threshold is reached."""
        breaker = CircuitBreaker(failure_threshold=3)

        def fail_func() -> str:
            raise ValueError("Error")

        # Trigger failures until threshold
        for _ in range(3):
            with pytest.raises(ValueError):
                breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open is True

    def test_open_circuit_blocks_requests(self) -> None:
        """Test that open circuit blocks requests."""
        breaker = CircuitBreaker(failure_threshold=1, timeout=60.0)

        # Force open state
        breaker.state = CircuitState.OPEN
        breaker.last_failure_time = datetime.utcnow()

        def success_func() -> str:
            return "should not be called"

        result = breaker.call(success_func)
        assert result is None

    def test_circuit_transitions_to_half_open_after_timeout(self) -> None:
        """Test that circuit transitions to HALF_OPEN after timeout."""
        breaker = CircuitBreaker(failure_threshold=1, timeout=0.1)

        # Force open state with old failure time
        breaker.state = CircuitState.OPEN
        breaker.last_failure_time = datetime.utcnow() - timedelta(seconds=1)

        def success_func() -> str:
            return "success"

        # Should transition to HALF_OPEN and allow request
        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED

    def test_half_open_success_closes_circuit(self) -> None:
        """Test that success in HALF_OPEN state closes circuit."""
        breaker = CircuitBreaker(failure_threshold=3)
        breaker.state = CircuitState.HALF_OPEN

        def success_func() -> str:
            return "success"

        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_half_open_failure_reopens_circuit(self) -> None:
        """Test that failure in HALF_OPEN state reopens circuit."""
        breaker = CircuitBreaker(failure_threshold=3)
        breaker.state = CircuitState.HALF_OPEN

        def fail_func() -> str:
            raise ValueError("Error")

        with pytest.raises(ValueError):
            breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN

    def test_reset_closes_circuit(self) -> None:
        """Test that reset() closes the circuit."""
        breaker = CircuitBreaker(failure_threshold=3)
        breaker.state = CircuitState.OPEN
        breaker.failure_count = 5
        breaker.last_failure_time = datetime.utcnow()

        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.last_failure_time is None

    def test_call_with_args_and_kwargs(self) -> None:
        """Test that call passes arguments correctly."""
        breaker = CircuitBreaker()

        def add(a: int, b: int, multiplier: int = 1) -> int:
            return (a + b) * multiplier

        result = breaker.call(add, 2, 3, multiplier=2)
        assert result == 10

    def test_is_open_property(self) -> None:
        """Test is_open property."""
        breaker = CircuitBreaker()

        assert breaker.is_open is False

        breaker.state = CircuitState.OPEN
        assert breaker.is_open is True

        breaker.state = CircuitState.HALF_OPEN
        assert breaker.is_open is False

    def test_multiple_breakers_independent(self) -> None:
        """Test that multiple breakers are independent."""
        breaker1 = CircuitBreaker(name="breaker1")
        breaker2 = CircuitBreaker(name="breaker2")

        # Fail breaker1
        breaker1.state = CircuitState.OPEN

        # breaker2 should still be closed
        assert breaker1.is_open is True
        assert breaker2.is_open is False


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker."""

    def test_full_lifecycle(self) -> None:
        """Test complete circuit breaker lifecycle."""
        breaker = CircuitBreaker(failure_threshold=2, timeout=0.1, name="test")
        call_count = 0

        def unreliable_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("Service unavailable")
            return "success"

        # First two calls fail
        with pytest.raises(ConnectionError):
            breaker.call(unreliable_func)
        # First failure doesn't open circuit yet
        assert breaker.state.value == "closed"

        with pytest.raises(ConnectionError):
            breaker.call(unreliable_func)
        # Second failure opens circuit (threshold=2)
        assert breaker.state.value == "open"

        # Third call is blocked
        result = breaker.call(unreliable_func)
        assert result is None
        assert call_count == 2  # Function wasn't called

        # Wait for timeout
        time.sleep(0.15)

        # Fourth call succeeds (HALF_OPEN -> CLOSED)
        result = breaker.call(unreliable_func)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED


class TestRetryWithCircuitBreaker:
    """Test combining retry with circuit breaker."""

    def test_retry_with_circuit_breaker(self) -> None:
        """Test combining retry decorator with circuit breaker."""
        breaker = CircuitBreaker(failure_threshold=5, name="combined")
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.01, transient_only=False)
        def resilient_call() -> str:
            nonlocal call_count
            result = breaker.call(lambda: "success")
            if result is None:
                raise Exception("Circuit open")
            call_count += 1
            return result

        result = resilient_call()
        assert result == "success"
        assert call_count == 1


# Fixtures
@pytest.fixture  # type: ignore[untyped-decorator]
def fresh_breaker() -> CircuitBreaker:
    """Fixture providing a fresh circuit breaker."""
    return CircuitBreaker(failure_threshold=3, timeout=1.0, name="test")


class TestWithFixtures:
    """Tests using fixtures."""

    def test_breaker_from_fixture(self, fresh_breaker: CircuitBreaker) -> None:
        """Test using circuit breaker fixture."""
        assert fresh_breaker.state == CircuitState.CLOSED
        assert fresh_breaker.name == "test"
        assert fresh_breaker.failure_threshold == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
