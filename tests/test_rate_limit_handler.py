"""
Unit tests for Rate Limit Handler (rate_limit_handler.py).

Tests rate limiting, exponential backoff, retry logic, and exchange wrapping.
Target: 80%+ code coverage

Run tests:
    python3 -m pytest tests/test_rate_limit_handler.py -v
    python3 -m pytest tests/test_rate_limit_handler.py --cov=rate_limit_handler --cov-report=term-missing
"""

import pytest  # type: ignore[import-not-found]
import sys
import time
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rate_limit_handler import (
    RateLimitHandler,
    RateLimitedExchange,
    rate_limited,
    safe_api_call,
    global_handler,
)


class TestRateLimitHandlerInit:
    """Test RateLimitHandler initialization."""

    def test_default_parameters(self) -> None:
        """Test initialization with default parameters."""
        handler = RateLimitHandler()

        assert handler.base_delay == 0.5
        assert handler.max_retries == 5
        assert handler.backoff_factor == 2.0
        assert handler.max_backoff == 30.0
        assert handler.last_call_time == 0.0

    def test_custom_parameters(self) -> None:
        """Test initialization with custom parameters."""
        handler = RateLimitHandler(
            base_delay=1.0,
            max_retries=3,
            backoff_factor=3.0,
            max_backoff=60.0
        )

        assert handler.base_delay == 1.0
        assert handler.max_retries == 3
        assert handler.backoff_factor == 3.0
        assert handler.max_backoff == 60.0


class TestCalculateBackoff:
    """Test backoff calculation."""

    def test_first_attempt_backoff(self) -> None:
        """Test backoff for first attempt."""
        handler = RateLimitHandler(base_delay=1.0, backoff_factor=2.0)

        backoff = handler._calculate_backoff(0)
        assert backoff == 1.0  # 1.0 * 2^0 = 1.0

    def test_second_attempt_backoff(self) -> None:
        """Test backoff for second attempt."""
        handler = RateLimitHandler(base_delay=1.0, backoff_factor=2.0)

        backoff = handler._calculate_backoff(1)
        assert backoff == 2.0  # 1.0 * 2^1 = 2.0

    def test_third_attempt_backoff(self) -> None:
        """Test backoff for third attempt."""
        handler = RateLimitHandler(base_delay=1.0, backoff_factor=2.0)

        backoff = handler._calculate_backoff(2)
        assert backoff == 4.0  # 1.0 * 2^2 = 4.0

    def test_backoff_capped_at_max(self) -> None:
        """Test that backoff is capped at max_backoff."""
        handler = RateLimitHandler(base_delay=1.0, backoff_factor=2.0, max_backoff=5.0)

        # Attempt 10 would be 1.0 * 2^10 = 1024, but capped at 5.0
        backoff = handler._calculate_backoff(10)
        assert backoff == 5.0

    def test_custom_backoff_factor(self) -> None:
        """Test with custom backoff factor."""
        handler = RateLimitHandler(base_delay=1.0, backoff_factor=3.0)

        backoff = handler._calculate_backoff(2)
        assert backoff == 9.0  # 1.0 * 3^2 = 9.0


class TestIsRateLimitError:
    """Test rate limit error detection."""

    def test_detects_510_error(self) -> None:
        """Test detection of HTTP 510 error."""
        handler = RateLimitHandler()

        error = Exception("HTTP 510 - Too many requests")
        assert handler._is_rate_limit_error(error) is True

    def test_detects_requests_too_frequent(self) -> None:
        """Test detection of 'requests are too frequent' message."""
        handler = RateLimitHandler()

        error = Exception("Requests are too frequent. Please slow down.")
        assert handler._is_rate_limit_error(error) is True

    def test_detects_rate_limit_message(self) -> None:
        """Test detection of 'rate limit' message."""
        handler = RateLimitHandler()

        error = Exception("Rate limit exceeded")
        assert handler._is_rate_limit_error(error) is True

    def test_detects_too_many_requests(self) -> None:
        """Test detection of 'too many requests' message."""
        handler = RateLimitHandler()

        error = Exception("429 Too Many Requests")
        assert handler._is_rate_limit_error(error) is True

    def test_case_insensitive(self) -> None:
        """Test that detection is case-insensitive."""
        handler = RateLimitHandler()

        error = Exception("RATE LIMIT EXCEEDED")
        assert handler._is_rate_limit_error(error) is True

    def test_non_rate_limit_error(self) -> None:
        """Test that non-rate-limit errors are not detected."""
        handler = RateLimitHandler()

        error = Exception("Invalid symbol")
        assert handler._is_rate_limit_error(error) is False

    def test_network_error_not_rate_limit(self) -> None:
        """Test that network errors are not rate limit errors."""
        handler = RateLimitHandler()

        error = Exception("Connection refused")
        assert handler._is_rate_limit_error(error) is False


class TestApplyBaseDelay:
    """Test base delay application."""

    def test_applies_delay_when_needed(self) -> None:
        """Test that delay is applied when calls are too fast."""
        handler = RateLimitHandler(base_delay=0.1)

        # Set last call time to now
        handler.last_call_time = time.time()

        start = time.time()
        handler._apply_base_delay()
        elapsed = time.time() - start

        # Should have waited approximately base_delay
        assert elapsed >= 0.05  # Allow some tolerance

    def test_no_delay_when_enough_time_passed(self) -> None:
        """Test that no delay when enough time has passed."""
        handler = RateLimitHandler(base_delay=0.1)

        # Set last call time to long ago
        handler.last_call_time = time.time() - 10.0

        start = time.time()
        handler._apply_base_delay()
        elapsed = time.time() - start

        # Should not have waited
        assert elapsed < 0.05

    def test_updates_last_call_time(self) -> None:
        """Test that last_call_time is updated after delay."""
        handler = RateLimitHandler(base_delay=0.01)
        handler.last_call_time = 0.0

        before = time.time()
        handler._apply_base_delay()
        after = time.time()

        assert handler.last_call_time >= before
        assert handler.last_call_time <= after


class TestExecute:
    """Test the execute method."""

    def test_successful_call(self) -> None:
        """Test successful API call returns result."""
        handler = RateLimitHandler(base_delay=0.01)

        def success_func() -> str:
            return "success"

        result = handler.execute(success_func)
        assert result == "success"

    def test_passes_arguments(self) -> None:
        """Test that arguments are passed to function."""
        handler = RateLimitHandler(base_delay=0.01)

        def add(a: int, b: int) -> int:
            return a + b

        result = handler.execute(add, 2, 3)
        assert result == 5

    def test_passes_kwargs(self) -> None:
        """Test that keyword arguments are passed to function."""
        handler = RateLimitHandler(base_delay=0.01)

        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        result = handler.execute(greet, "World", greeting="Hi")
        assert result == "Hi, World!"

    def test_retries_on_rate_limit_error(self) -> None:
        """Test that rate limit errors trigger retries."""
        handler = RateLimitHandler(base_delay=0.01, max_retries=3)
        call_count = 0

        def flaky_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Rate limit exceeded")
            return "success"

        result = handler.execute(flaky_func)
        assert result == "success"
        assert call_count == 3

    def test_raises_after_max_retries(self) -> None:
        """Test that exception is raised after max retries."""
        handler = RateLimitHandler(base_delay=0.01, max_retries=3)
        call_count = 0

        def always_rate_limited() -> str:
            nonlocal call_count
            call_count += 1
            raise Exception("510 Too Many Requests")

        with pytest.raises(Exception) as exc_info:
            handler.execute(always_rate_limited)

        assert call_count == 3
        assert "510" in str(exc_info.value)

    def test_non_rate_limit_error_not_retried(self) -> None:
        """Test that non-rate-limit errors are raised immediately."""
        handler = RateLimitHandler(base_delay=0.01, max_retries=3)
        call_count = 0

        def raises_value_error() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid input")

        with pytest.raises(ValueError):
            handler.execute(raises_value_error)

        assert call_count == 1  # Should not retry

    def test_updates_last_call_time_on_success(self) -> None:
        """Test that last_call_time is updated on success."""
        handler = RateLimitHandler(base_delay=0.01)
        handler.last_call_time = 0.0

        def success_func() -> str:
            return "success"

        before = time.time()
        handler.execute(success_func)

        assert handler.last_call_time >= before

    def test_logs_retry_success(self) -> None:
        """Test that successful retry is logged."""
        handler = RateLimitHandler(base_delay=0.01, max_retries=3)
        call_count = 0

        def succeeds_after_retry() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Rate limit hit")
            return "success"

        with patch('rate_limit_handler.logger') as mock_logger:
            result = handler.execute(succeeds_after_retry)

            assert result == "success"
            # Check that info was called for retry success
            mock_logger.info.assert_called()


class TestRateLimitedExchange:
    """Test RateLimitedExchange wrapper."""

    def test_wraps_fetch_methods(self) -> None:
        """Test that fetch_* methods are wrapped."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_ticker.return_value = {"last": 50000}

        wrapped = RateLimitedExchange(mock_exchange)
        result = wrapped.fetch_ticker("BTC/USDT")

        assert result == {"last": 50000}
        mock_exchange.fetch_ticker.assert_called_once_with("BTC/USDT")

    def test_wraps_create_methods(self) -> None:
        """Test that create_* methods are wrapped."""
        mock_exchange = MagicMock()
        mock_exchange.create_order.return_value = {"id": "123"}

        wrapped = RateLimitedExchange(mock_exchange)
        result = wrapped.create_order("BTC/USDT", "limit", "buy", 1.0)

        assert result == {"id": "123"}

    def test_wraps_cancel_methods(self) -> None:
        """Test that cancel_* methods are wrapped."""
        mock_exchange = MagicMock()
        mock_exchange.cancel_order.return_value = {"status": "canceled"}

        wrapped = RateLimitedExchange(mock_exchange)
        result = wrapped.cancel_order("123")

        assert result == {"status": "canceled"}

    def test_wraps_load_markets(self) -> None:
        """Test that load_markets is wrapped."""
        mock_exchange = MagicMock()
        mock_exchange.load_markets.return_value = {"BTC/USDT": {}}

        wrapped = RateLimitedExchange(mock_exchange)
        result = wrapped.load_markets()

        assert result == {"BTC/USDT": {}}

    def test_wraps_fetch_balance(self) -> None:
        """Test that fetch_balance is wrapped."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_balance.return_value = {"USDT": 1000}

        wrapped = RateLimitedExchange(mock_exchange)
        result = wrapped.fetch_balance()

        assert result == {"USDT": 1000}

    def test_wraps_fetch_ohlcv(self) -> None:
        """Test that fetch_ohlcv is wrapped."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = [[1, 2, 3, 4, 5]]

        wrapped = RateLimitedExchange(mock_exchange)
        result = wrapped.fetch_ohlcv("BTC/USDT", "1h")

        assert result == [[1, 2, 3, 4, 5]]

    def test_non_api_methods_not_wrapped(self) -> None:
        """Test that non-API methods pass through directly."""
        mock_exchange = MagicMock()
        mock_exchange.name = "TestExchange"
        mock_exchange.some_property = "value"

        wrapped = RateLimitedExchange(mock_exchange)

        assert wrapped.name == "TestExchange"
        assert wrapped.some_property == "value"

    def test_custom_handler(self) -> None:
        """Test using a custom handler."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_ticker.return_value = {"last": 50000}

        custom_handler = RateLimitHandler(base_delay=0.01, max_retries=2)
        wrapped = RateLimitedExchange(mock_exchange, handler=custom_handler)

        result = wrapped.fetch_ticker("BTC/USDT")
        assert result == {"last": 50000}

    def test_retries_rate_limit_errors(self) -> None:
        """Test that wrapped methods retry on rate limit errors."""
        mock_exchange = MagicMock()
        call_count = 0

        def side_effect(*args: Any, **kwargs: Any) -> dict[str, int]:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Rate limit exceeded")
            return {"last": 50000}

        mock_exchange.fetch_ticker.side_effect = side_effect

        handler = RateLimitHandler(base_delay=0.01, max_retries=3)
        wrapped = RateLimitedExchange(mock_exchange, handler=handler)

        result = wrapped.fetch_ticker("BTC/USDT")
        assert result == {"last": 50000}
        assert call_count == 2


class TestRateLimitedDecorator:
    """Test the rate_limited decorator."""

    def test_decorator_wraps_function(self) -> None:
        """Test that decorator wraps function."""
        @rate_limited(base_delay=0.01)
        def test_func() -> str:
            return "success"

        result = test_func()
        assert result == "success"

    def test_decorator_preserves_name(self) -> None:
        """Test that decorator preserves function name."""
        @rate_limited(base_delay=0.01)
        def my_function() -> str:
            return "test"

        assert my_function.__name__ == "my_function"

    def test_decorator_with_arguments(self) -> None:
        """Test decorator with function arguments."""
        @rate_limited(base_delay=0.01)
        def add(a: int, b: int) -> int:
            return a + b

        result = add(2, 3)
        assert result == 5

    def test_decorator_retries_rate_limit(self) -> None:
        """Test that decorated function retries on rate limit."""
        call_count = 0

        @rate_limited(base_delay=0.01, max_retries=3)
        def flaky_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Rate limit")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count == 2

    def test_decorator_custom_parameters(self) -> None:
        """Test decorator with custom parameters."""
        @rate_limited(base_delay=0.02, max_retries=2, backoff_factor=3.0, max_backoff=10.0)
        def test_func() -> str:
            return "success"

        result = test_func()
        assert result == "success"


class TestSafeApiCall:
    """Test the safe_api_call function."""

    def test_basic_call(self) -> None:
        """Test basic API call."""
        def success_func() -> str:
            return "success"

        result = safe_api_call(success_func)
        assert result == "success"

    def test_with_arguments(self) -> None:
        """Test API call with arguments."""
        def add(a: int, b: int) -> int:
            return a + b

        result = safe_api_call(add, 2, 3)
        assert result == 5

    def test_with_kwargs(self) -> None:
        """Test API call with keyword arguments."""
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        result = safe_api_call(greet, "World", greeting="Hi")
        assert result == "Hi, World!"


class TestGlobalHandler:
    """Test the global handler instance."""

    def test_global_handler_exists(self) -> None:
        """Test that global handler is instantiated."""
        assert global_handler is not None
        assert isinstance(global_handler, RateLimitHandler)

    def test_global_handler_default_params(self) -> None:
        """Test global handler has default parameters."""
        assert global_handler.base_delay == 0.5
        assert global_handler.max_retries == 5


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_retries(self) -> None:
        """Test handler with zero max retries."""
        handler = RateLimitHandler(base_delay=0.01, max_retries=1)

        def always_fails() -> str:
            raise Exception("Rate limit")

        with pytest.raises(Exception):
            handler.execute(always_fails)

    def test_very_small_base_delay(self) -> None:
        """Test handler with very small base delay."""
        handler = RateLimitHandler(base_delay=0.001)

        def success_func() -> str:
            return "success"

        result = handler.execute(success_func)
        assert result == "success"

    def test_large_backoff_factor(self) -> None:
        """Test handler with large backoff factor."""
        handler = RateLimitHandler(
            base_delay=0.01,
            backoff_factor=10.0,
            max_backoff=0.1
        )

        # Even with large factor, should be capped
        backoff = handler._calculate_backoff(5)
        assert backoff == 0.1  # Capped at max_backoff

    def test_exception_preserves_original_message(self) -> None:
        """Test that original exception message is preserved."""
        handler = RateLimitHandler(base_delay=0.01, max_retries=1)

        def raises_with_message() -> str:
            raise Exception("Specific error message for 510")

        with pytest.raises(Exception) as exc_info:
            handler.execute(raises_with_message)

        assert "Specific error message" in str(exc_info.value)


# Fixtures
@pytest.fixture  # type: ignore[untyped-decorator]
def handler() -> RateLimitHandler:
    """Fixture providing a configured handler."""
    return RateLimitHandler(base_delay=0.01, max_retries=3)


@pytest.fixture  # type: ignore[untyped-decorator]
def mock_exchange() -> MagicMock:
    """Fixture providing a mock exchange."""
    exchange = MagicMock()
    exchange.fetch_ticker.return_value = {"last": 50000}
    exchange.fetch_balance.return_value = {"USDT": 1000}
    exchange.load_markets.return_value = {"BTC/USDT": {}}
    return exchange


class TestWithFixtures:
    """Tests using fixtures."""

    def test_handler_from_fixture(self, handler: RateLimitHandler) -> None:
        """Test using handler fixture."""
        assert handler.base_delay == 0.01
        assert handler.max_retries == 3

    def test_exchange_from_fixture(self, mock_exchange: MagicMock) -> None:
        """Test using mock exchange fixture."""
        wrapped = RateLimitedExchange(mock_exchange)

        result = wrapped.fetch_ticker("BTC/USDT")
        assert result == {"last": 50000}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
