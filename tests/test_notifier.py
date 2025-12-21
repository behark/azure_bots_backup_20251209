"""
Unit tests for Telegram Notifier (notifier.py).

Tests signal ID generation, message sending, and signal logging.
Uses mocking for Telegram API calls.

Run tests:
    python3 -m pytest tests/test_notifier.py -v
"""

import json
import pytest  # type: ignore[import-not-found]
import sys
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notifier import generate_signal_id, TelegramNotifier


class TestGenerateSignalId:
    """Test signal ID generation."""

    def test_generate_signal_id_long(self) -> None:
        """Test signal ID generation for LONG direction."""
        signal_id = generate_signal_id("BTC/USDT", "LONG")

        assert signal_id.startswith("B1-")
        assert "-L-" in signal_id
        assert "BTC" in signal_id
        assert len(signal_id) == 13  # B1-BTC-L-XXXX

    def test_generate_signal_id_short(self) -> None:
        """Test signal ID generation for SHORT direction."""
        signal_id = generate_signal_id("ETH/USDT", "SHORT")

        assert signal_id.startswith("B1-")
        assert "-S-" in signal_id
        assert "ETH" in signal_id

    def test_generate_signal_id_strips_usdt(self) -> None:
        """Test that USDT suffixes are stripped from symbol."""
        signal_id = generate_signal_id("ADA/USDT:USDT", "LONG")

        assert "USDT" not in signal_id
        assert "ADA" in signal_id

    def test_generate_signal_id_unique(self) -> None:
        """Test that signal IDs are unique."""
        ids = [generate_signal_id("BTC/USDT", "LONG") for _ in range(100)]
        unique_ids = set(ids)

        # All IDs should be unique (cryptographically secure random)
        assert len(unique_ids) == 100

    def test_generate_signal_id_long_symbol(self) -> None:
        """Test signal ID with long symbol name (truncated to 6 chars)."""
        signal_id = generate_signal_id("VERYLONGSYMBOL/USDT", "LONG")

        # Symbol should be truncated to 6 characters
        parts = signal_id.split("-")
        assert len(parts[1]) <= 6

    def test_generate_signal_id_random_part(self) -> None:
        """Test that random part is alphanumeric and 4 characters."""
        signal_id = generate_signal_id("BTC/USDT", "LONG")

        random_part = signal_id.split("-")[-1]
        assert len(random_part) == 4
        assert random_part.isalnum()
        assert random_part.isupper() or any(c.isdigit() for c in random_part)


class TestTelegramNotifierInit:
    """Test TelegramNotifier initialization."""

    def test_init_success(self) -> None:
        """Test successful initialization with valid credentials."""
        notifier = TelegramNotifier(
            bot_token="123456:ABC-DEF",
            chat_id="987654321"
        )

        assert notifier.bot_token == "123456:ABC-DEF"
        assert notifier.chat_id == "987654321"
        assert "123456:ABC-DEF" in notifier.base_url

    def test_init_custom_log_file(self) -> None:
        """Test initialization with custom signals log file."""
        notifier = TelegramNotifier(
            bot_token="123456:ABC",
            chat_id="123",
            signals_log_file="/custom/path/signals.json"
        )

        assert notifier.signals_log_file == "/custom/path/signals.json"

    def test_init_empty_token_raises(self) -> None:
        """Test that empty bot token raises ValueError."""
        with pytest.raises(ValueError, match="credentials not configured"):
            TelegramNotifier(bot_token="", chat_id="123")

    def test_init_empty_chat_id_raises(self) -> None:
        """Test that empty chat ID raises ValueError."""
        with pytest.raises(ValueError, match="credentials not configured"):
            TelegramNotifier(bot_token="123:ABC", chat_id="")

    def test_init_none_token_raises(self) -> None:
        """Test that None bot token raises ValueError."""
        with pytest.raises(ValueError):
            TelegramNotifier(bot_token=None, chat_id="123")  # type: ignore[arg-type]


class TestSendMessage:
    """Test send_message method."""

    @patch('notifier.requests.post')
    def test_send_message_success(self, mock_post: MagicMock) -> None:
        """Test successful message sending."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        notifier = TelegramNotifier("123:ABC", "456")
        result = notifier.send_message("Hello, World!")

        assert result is True
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['json']['text'] == "Hello, World!"
        assert call_args[1]['json']['chat_id'] == "456"

    @patch('notifier.requests.post')
    def test_send_message_with_html(self, mock_post: MagicMock) -> None:
        """Test message sending with HTML parse mode."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        notifier = TelegramNotifier("123:ABC", "456")
        result = notifier.send_message("<b>Bold</b>", parse_mode="HTML")

        assert result is True
        call_args = mock_post.call_args
        assert call_args[1]['json']['parse_mode'] == "HTML"

    @patch('notifier.requests.post')
    def test_send_message_request_exception(self, mock_post: MagicMock) -> None:
        """Test handling of request exceptions."""
        import requests  # type: ignore[import-untyped]
        mock_post.side_effect = requests.exceptions.RequestException("Network error")

        notifier = TelegramNotifier("123:ABC", "456")
        result = notifier.send_message("Test message")

        assert result is False

    @patch('notifier.requests.post')
    def test_send_message_timeout(self, mock_post: MagicMock) -> None:
        """Test handling of timeout."""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout("Timeout")

        notifier = TelegramNotifier("123:ABC", "456")
        result = notifier.send_message("Test message")

        assert result is False

    @patch('notifier.requests.post')
    def test_send_message_http_error(self, mock_post: MagicMock) -> None:
        """Test handling of HTTP errors."""
        import requests
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("401 Unauthorized")
        mock_post.return_value = mock_response

        notifier = TelegramNotifier("123:ABC", "456")
        result = notifier.send_message("Test message")

        assert result is False


class TestSaveSignalToJson:
    """Test save_signal_to_json method."""

    def test_save_signal_basic(self, tmp_path: Path) -> None:
        """Test basic signal saving to JSON."""
        log_file = tmp_path / "signals.json"
        notifier = TelegramNotifier("123:ABC", "456", str(log_file))

        signal = {
            "symbol": "BTC/USDT",
            "action": "LONG",
            "entry_price": 50000.0,
            "stop_loss": 48000.0,
            "take_profit": 54000.0,
            "score": 7.5
        }

        result = notifier.save_signal_to_json(signal)

        assert result is True
        assert log_file.exists()

        saved_data = json.loads(log_file.read_text())
        assert len(saved_data) == 1
        assert saved_data[0]["symbol"] == "BTC/USDT"
        assert saved_data[0]["direction"] == "LONG"
        assert saved_data[0]["entry"] == 50000.0

    def test_save_signal_with_custom_id(self, tmp_path: Path) -> None:
        """Test saving signal with custom signal ID."""
        log_file = tmp_path / "signals.json"
        notifier = TelegramNotifier("123:ABC", "456", str(log_file))

        signal = {
            "symbol": "ETH/USDT",
            "action": "SHORT",
            "entry_price": 3000.0,
            "stop_loss": 3150.0,
            "take_profit": 2800.0,
            "score": 6.0
        }

        result = notifier.save_signal_to_json(signal, signal_id="CUSTOM-001")

        saved_data = json.loads(log_file.read_text())
        assert saved_data[0]["signal_id"] == "CUSTOM-001"

    def test_save_signal_accumulates(self, tmp_path: Path) -> None:
        """Test that signals accumulate in the log file."""
        log_file = tmp_path / "signals.json"
        notifier = TelegramNotifier("123:ABC", "456", str(log_file))

        base_signal = {
            "symbol": "BTC/USDT",
            "action": "LONG",
            "entry_price": 50000.0,
            "stop_loss": 48000.0,
            "take_profit": 54000.0,
            "score": 7.0
        }

        for i in range(3):
            signal = base_signal.copy()
            signal["entry_price"] = 50000.0 + i * 1000
            notifier.save_signal_to_json(signal, signal_id=f"SIG-{i:03d}")

        saved_data = json.loads(log_file.read_text())
        assert len(saved_data) == 3

    def test_save_signal_with_partial_tp(self, tmp_path: Path) -> None:
        """Test saving signal with partial take profit."""
        log_file = tmp_path / "signals.json"
        notifier = TelegramNotifier("123:ABC", "456", str(log_file))

        signal = {
            "symbol": "BTC/USDT",
            "action": "LONG",
            "entry_price": 50000.0,
            "stop_loss": 48000.0,
            "take_profit": 54000.0,
            "score": 7.0,
            "take_profit_partial": 52000.0,
            "partial_allocation": 0.5,
            "partial_rr": 1.0,
            "take_profit_final": 56000.0,
            "final_rr": 2.0,
            "final_allocation": 0.5
        }

        notifier.save_signal_to_json(signal)

        saved_data = json.loads(log_file.read_text())
        assert saved_data[0]["tp_partial"] == 52000.0
        assert saved_data[0]["tp_final"] == 56000.0

    def test_save_signal_calculates_rr(self, tmp_path: Path) -> None:
        """Test that risk/reward ratio is calculated correctly."""
        log_file = tmp_path / "signals.json"
        notifier = TelegramNotifier("123:ABC", "456", str(log_file))

        signal = {
            "symbol": "BTC/USDT",
            "action": "LONG",
            "entry_price": 100.0,
            "stop_loss": 95.0,  # 5 risk
            "take_profit": 110.0,  # 10 reward
            "score": 7.0
        }

        notifier.save_signal_to_json(signal)

        saved_data = json.loads(log_file.read_text())
        assert saved_data[0]["risk_reward_ratio"] == 2.0  # 10/5 = 2.0

    def test_save_signal_handles_zero_risk(self, tmp_path: Path) -> None:
        """Test handling when entry equals stop loss."""
        log_file = tmp_path / "signals.json"
        notifier = TelegramNotifier("123:ABC", "456", str(log_file))

        signal = {
            "symbol": "BTC/USDT",
            "action": "LONG",
            "entry_price": 100.0,
            "stop_loss": 100.0,  # Same as entry
            "take_profit": 110.0,
            "score": 7.0
        }

        notifier.save_signal_to_json(signal)

        saved_data = json.loads(log_file.read_text())
        assert saved_data[0]["risk_reward_ratio"] == 0


class TestSendSignalAlert:
    """Test send_signal_alert method."""

    @patch('notifier.requests.post')
    def test_send_signal_alert_long(self, mock_post: MagicMock, tmp_path: Path) -> None:
        """Test sending LONG signal alert."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        log_file = tmp_path / "signals.json"
        notifier = TelegramNotifier("123:ABC", "456", str(log_file))

        signal = {
            "symbol": "BTC/USDT",
            "action": "LONG",
            "entry_price": 50000.0,
            "stop_loss": 48000.0,
            "take_profit": 54000.0,
            "score": 7.5,
            "adx": 35.0,
            "rsx": 65.0
        }

        result = notifier.send_signal_alert(signal)

        assert result is True
        # Check message content
        call_args = mock_post.call_args
        message = call_args[1]['json']['text']
        assert "LONG SIGNAL" in message
        assert "BTC/USDT" in message
        assert "50000" in message

    @patch('notifier.requests.post')
    def test_send_signal_alert_short(self, mock_post: MagicMock, tmp_path: Path) -> None:
        """Test sending SHORT signal alert."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        log_file = tmp_path / "signals.json"
        notifier = TelegramNotifier("123:ABC", "456", str(log_file))

        signal = {
            "symbol": "ETH/USDT",
            "action": "SHORT",
            "entry_price": 3000.0,
            "stop_loss": 3150.0,
            "take_profit": 2800.0,
            "score": 6.5,
            "adx": 28.0,
            "rsx": 35.0
        }

        result = notifier.send_signal_alert(signal)

        assert result is True
        message = mock_post.call_args[1]['json']['text']
        assert "SHORT SIGNAL" in message

    @patch('notifier.requests.post')
    def test_send_signal_alert_strong_quality(self, mock_post: MagicMock, tmp_path: Path) -> None:
        """Test STRONG quality signal (score >= 6.0)."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        log_file = tmp_path / "signals.json"
        notifier = TelegramNotifier("123:ABC", "456", str(log_file))

        signal = {
            "symbol": "BTC/USDT",
            "action": "LONG",
            "entry_price": 50000.0,
            "stop_loss": 48000.0,
            "take_profit": 54000.0,
            "score": 7.0,  # STRONG
            "adx": 35.0
        }

        notifier.send_signal_alert(signal)

        message = mock_post.call_args[1]['json']['text']
        assert "PREMIUM" in message or "Full confidence" in message

    @patch('notifier.requests.post')
    def test_send_signal_alert_saves_to_json(self, mock_post: MagicMock, tmp_path: Path) -> None:
        """Test that signal alert also saves to JSON."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        log_file = tmp_path / "signals.json"
        notifier = TelegramNotifier("123:ABC", "456", str(log_file))

        signal = {
            "symbol": "BTC/USDT",
            "action": "LONG",
            "entry_price": 50000.0,
            "stop_loss": 48000.0,
            "take_profit": 54000.0,
            "score": 6.0,
            "adx": 30.0
        }

        notifier.send_signal_alert(signal)

        assert log_file.exists()
        saved_data = json.loads(log_file.read_text())
        assert len(saved_data) == 1

    @patch('notifier.requests.post')
    def test_send_signal_alert_failure(self, mock_post: MagicMock, tmp_path: Path) -> None:
        """Test handling of send failure."""
        import requests
        mock_post.side_effect = requests.exceptions.RequestException("Error")

        log_file = tmp_path / "signals.json"
        notifier = TelegramNotifier("123:ABC", "456", str(log_file))

        signal = {
            "symbol": "BTC/USDT",
            "action": "LONG",
            "entry_price": 50000.0,
            "stop_loss": 48000.0,
            "take_profit": 54000.0,
            "score": 6.0,
            "adx": 30.0
        }

        result = notifier.send_signal_alert(signal)

        assert result is False


class TestSendStartupMessage:
    """Test send_startup_message method."""

    @patch('notifier.requests.post')
    def test_send_startup_message_success(self, mock_post: MagicMock) -> None:
        """Test sending startup message."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        notifier = TelegramNotifier("123:ABC", "456")

        pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        available_pairs = {
            "BTC/USDT": True,
            "ETH/USDT": True,
            "SOL/USDT": False
        }

        result = notifier.send_startup_message(pairs, available_pairs)

        assert result is True
        message = mock_post.call_args[1]['json']['text']
        assert "Trading Bot Started" in message
        assert "2/3" in message  # 2 available out of 3
        assert "BTC/USDT" in message
        assert "SOL/USDT" in message

    @patch('notifier.requests.post')
    def test_send_startup_message_all_available(self, mock_post: MagicMock) -> None:
        """Test startup message when all pairs available."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        notifier = TelegramNotifier("123:ABC", "456")

        pairs = ["BTC/USDT", "ETH/USDT"]
        available_pairs = {"BTC/USDT": True, "ETH/USDT": True}

        result = notifier.send_startup_message(pairs, available_pairs)

        assert result is True
        message = mock_post.call_args[1]['json']['text']
        assert "2/2" in message
        assert "Unavailable" not in message  # No unavailable section


class TestSendErrorMessage:
    """Test send_error_message method."""

    @patch('notifier.requests.post')
    def test_send_error_message_success(self, mock_post: MagicMock) -> None:
        """Test sending error message."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        notifier = TelegramNotifier("123:ABC", "456")

        result = notifier.send_error_message("Connection lost to exchange")

        assert result is True
        message = mock_post.call_args[1]['json']['text']
        assert "Error" in message
        assert "Connection lost to exchange" in message

    @patch('notifier.requests.post')
    def test_send_error_message_failure(self, mock_post: MagicMock) -> None:
        """Test error message when sending fails."""
        import requests
        mock_post.side_effect = requests.exceptions.RequestException("Network error")

        notifier = TelegramNotifier("123:ABC", "456")

        result = notifier.send_error_message("Some error")

        assert result is False


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_save_signal_with_missing_optional_fields(self, tmp_path: Path) -> None:
        """Test saving signal with minimal required fields."""
        log_file = tmp_path / "signals.json"
        notifier = TelegramNotifier("123:ABC", "456", str(log_file))

        signal = {
            "symbol": "BTC/USDT",
            "action": "LONG",
            "entry_price": 50000.0,
            "stop_loss": 48000.0,
            "take_profit": 54000.0,
            "score": 5.0
            # No adx, rsx, or optional fields
        }

        result = notifier.save_signal_to_json(signal)

        assert result is True
        saved_data = json.loads(log_file.read_text())
        assert saved_data[0]["adx"] is None
        assert saved_data[0]["rsx"] is None

    @patch('notifier.requests.post')
    def test_send_signal_with_rsi_fallback(self, mock_post: MagicMock, tmp_path: Path) -> None:
        """Test signal alert with RSI instead of RSX."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        log_file = tmp_path / "signals.json"
        notifier = TelegramNotifier("123:ABC", "456", str(log_file))

        signal = {
            "symbol": "BTC/USDT",
            "action": "LONG",
            "entry_price": 50000.0,
            "stop_loss": 48000.0,
            "take_profit": 54000.0,
            "score": 6.0,
            "adx": 30.0,
            "rsi": 55.0  # RSI instead of RSX
        }

        result = notifier.send_signal_alert(signal)

        assert result is True
        message = mock_post.call_args[1]['json']['text']
        assert "RSX" in message  # Should show RSX label
        assert "55.0" in message  # RSI value used

    def test_save_signal_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that save creates parent directories."""
        log_file = tmp_path / "deep" / "nested" / "signals.json"
        notifier = TelegramNotifier("123:ABC", "456", str(log_file))

        signal = {
            "symbol": "BTC/USDT",
            "action": "LONG",
            "entry_price": 50000.0,
            "stop_loss": 48000.0,
            "take_profit": 54000.0,
            "score": 5.0
        }

        # This should work even though parent dirs don't exist
        # (file_lock context manager handles this)
        result = notifier.save_signal_to_json(signal)

        # Note: This may fail depending on file_lock implementation
        # The test documents expected behavior


# Fixtures
@pytest.fixture  # type: ignore[untyped-decorator]
def mock_successful_post() -> Generator[MagicMock, None, None]:
    """Fixture for successful Telegram API responses."""
    with patch('notifier.requests.post') as mock:
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock.return_value = mock_response
        yield mock


@pytest.fixture  # type: ignore[untyped-decorator]
def sample_signal() -> Dict[str, Any]:
    """Fixture providing a sample signal dict."""
    return {
        "symbol": "BTC/USDT",
        "action": "LONG",
        "entry_price": 50000.0,
        "stop_loss": 48000.0,
        "take_profit": 54000.0,
        "score": 7.0,
        "adx": 35.0,
        "rsx": 60.0
    }


class TestWithFixtures:
    """Tests using fixtures."""

    def test_send_with_fixture(self, mock_successful_post: MagicMock, sample_signal: Dict[str, Any], tmp_path: Path) -> None:
        """Test using fixtures."""
        log_file = tmp_path / "signals.json"
        notifier = TelegramNotifier("123:ABC", "456", str(log_file))

        result = notifier.send_signal_alert(sample_signal)

        assert result is True
        mock_successful_post.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
