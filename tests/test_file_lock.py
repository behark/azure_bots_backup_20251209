"""
Unit tests for File Lock Utility (file_lock.py).

Tests file locking, JSON read/write operations, and state management.
Target: 80%+ code coverage

Run tests:
    python3 -m pytest tests/test_file_lock.py -v
    python3 -m pytest tests/test_file_lock.py --cov=file_lock --cov-report=term-missing
"""

import pytest  # type: ignore[import-not-found]
import sys
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from file_lock import (
    FileLockError,
    file_lock,
    safe_read_json,
    safe_write_json,
    SafeStateManager,
)


class TestFileLockError:
    """Test FileLockError exception."""

    def test_is_exception(self) -> None:
        """Test that FileLockError is an Exception."""
        assert issubclass(FileLockError, Exception)

    def test_error_message(self) -> None:
        """Test FileLockError with message."""
        error = FileLockError("Lock failed")
        assert str(error) == "Lock failed"

    def test_can_be_raised(self) -> None:
        """Test that FileLockError can be raised and caught."""
        with pytest.raises(FileLockError) as exc_info:
            raise FileLockError("Test error")
        assert "Test error" in str(exc_info.value)


class TestFileLock:
    """Test file_lock context manager."""

    def test_basic_lock_and_unlock(self, tmp_path: Path) -> None:
        """Test basic lock acquisition and release."""
        test_file = tmp_path / "test.json"
        test_file.write_text("{}")

        with file_lock(test_file):
            # Lock is held, we can perform operations
            assert test_file.exists()

        # After context, lock should be released
        lock_file = test_file.with_suffix(".json.lock")
        # Lock file might still exist but should be unlocked

    def test_creates_lock_file(self, tmp_path: Path) -> None:
        """Test that lock file is created."""
        test_file = tmp_path / "test.json"
        test_file.write_text("{}")

        with file_lock(test_file):
            lock_file = test_file.with_suffix(".json.lock")
            assert lock_file.exists()

    def test_lock_file_permissions(self, tmp_path: Path) -> None:
        """Test that lock file has secure permissions."""
        test_file = tmp_path / "test.json"
        test_file.write_text("{}")

        with file_lock(test_file):
            lock_file = test_file.with_suffix(".json.lock")
            # Check permissions are 0o600 (owner read/write only)
            mode = os.stat(lock_file).st_mode & 0o777
            assert mode == 0o600

    def test_lock_timeout(self, tmp_path: Path) -> None:
        """Test that lock times out when held by another process."""
        test_file = tmp_path / "test.json"
        test_file.write_text("{}")
        lock_file = test_file.with_suffix(".json.lock")

        # Manually hold the lock
        import fcntl
        held_lock = open(lock_file, 'w')
        fcntl.flock(held_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        try:
            # Try to acquire lock with short timeout
            with pytest.raises(FileLockError) as exc_info:
                with file_lock(test_file, timeout=0.1):
                    pass  # Should never reach here

            assert "Could not acquire lock" in str(exc_info.value)
        finally:
            fcntl.flock(held_lock.fileno(), fcntl.LOCK_UN)
            held_lock.close()

    def test_lock_released_on_exception(self, tmp_path: Path) -> None:
        """Test that lock is released even if exception occurs."""
        test_file = tmp_path / "test.json"
        test_file.write_text("{}")

        try:
            with file_lock(test_file):
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Lock should be released, we should be able to acquire it again
        with file_lock(test_file, timeout=0.5):
            assert True  # Lock acquired successfully

    def test_concurrent_access_prevented(self, tmp_path: Path) -> None:
        """Test that concurrent access is serialized."""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"counter": 0}')

        results: List[int] = []
        errors: List[Exception] = []

        def increment() -> None:
            try:
                with file_lock(test_file, timeout=5.0):
                    data = json.loads(test_file.read_text())
                    data["counter"] += 1
                    time.sleep(0.05)  # Simulate some work
                    test_file.write_text(json.dumps(data))
                    results.append(data["counter"])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=increment) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should complete without errors
        assert len(errors) == 0
        # Counter should be exactly 5
        final_data = json.loads(test_file.read_text())
        assert final_data["counter"] == 5


class TestSafeReadJson:
    """Test safe_read_json function."""

    def test_read_existing_file(self, tmp_path: Path) -> None:
        """Test reading an existing JSON file."""
        test_file = tmp_path / "test.json"
        test_data = {"key": "value", "number": 42}
        test_file.write_text(json.dumps(test_data))

        result = safe_read_json(test_file)
        assert result == test_data

    def test_read_nonexistent_file_returns_default(self, tmp_path: Path) -> None:
        """Test that nonexistent file returns default."""
        test_file = tmp_path / "nonexistent.json"

        result = safe_read_json(test_file)
        assert result == {}

    def test_read_with_custom_default(self, tmp_path: Path) -> None:
        """Test reading with custom default value."""
        test_file = tmp_path / "nonexistent.json"
        default = {"default_key": "default_value"}

        result = safe_read_json(test_file, default=default)
        assert result == default

    def test_read_invalid_json_returns_default(self, tmp_path: Path) -> None:
        """Test that invalid JSON returns default."""
        test_file = tmp_path / "invalid.json"
        test_file.write_text("not valid json {{{")

        result = safe_read_json(test_file)
        assert result == {}

    def test_read_complex_json(self, tmp_path: Path) -> None:
        """Test reading complex nested JSON."""
        test_file = tmp_path / "complex.json"
        test_data = {
            "nested": {"a": 1, "b": [1, 2, 3]},
            "list": [{"x": 1}, {"y": 2}],
            "boolean": True,
            "null": None
        }
        test_file.write_text(json.dumps(test_data))

        result = safe_read_json(test_file)
        assert result == test_data

    def test_read_with_none_default(self, tmp_path: Path) -> None:
        """Test that None default becomes empty dict."""
        test_file = tmp_path / "nonexistent.json"

        result = safe_read_json(test_file, default=None)
        assert result == {}


class TestSafeWriteJson:
    """Test safe_write_json function."""

    def test_write_basic_json(self, tmp_path: Path) -> None:
        """Test writing basic JSON data."""
        test_file = tmp_path / "test.json"
        test_data = {"key": "value", "number": 42}

        result = safe_write_json(test_file, test_data)

        assert result is True
        assert test_file.exists()
        saved_data = json.loads(test_file.read_text())
        assert saved_data == test_data

    def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that write creates parent directories."""
        test_file = tmp_path / "subdir" / "nested" / "test.json"
        test_data = {"key": "value"}

        result = safe_write_json(test_file, test_data)

        assert result is True
        assert test_file.exists()
        assert test_file.parent.exists()

    def test_write_with_custom_indent(self, tmp_path: Path) -> None:
        """Test writing with custom indentation."""
        test_file = tmp_path / "test.json"
        test_data = {"key": "value"}

        safe_write_json(test_file, test_data, indent=4)

        content = test_file.read_text()
        # With indent=4, content should have 4-space indentation
        assert "    " in content

    def test_write_file_permissions(self, tmp_path: Path) -> None:
        """Test that written file has secure permissions."""
        test_file = tmp_path / "test.json"
        test_data = {"key": "value"}

        safe_write_json(test_file, test_data)

        mode = os.stat(test_file).st_mode & 0o777
        assert mode == 0o600

    def test_write_overwrites_existing(self, tmp_path: Path) -> None:
        """Test that write overwrites existing file."""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"old": "data"}')

        new_data = {"new": "data"}
        safe_write_json(test_file, new_data)

        saved_data = json.loads(test_file.read_text())
        assert saved_data == new_data

    def test_write_complex_json(self, tmp_path: Path) -> None:
        """Test writing complex nested JSON."""
        test_file = tmp_path / "complex.json"
        test_data = {
            "nested": {"a": 1, "b": [1, 2, 3]},
            "list": [{"x": 1}, {"y": 2}],
            "boolean": True,
            "null": None
        }

        result = safe_write_json(test_file, test_data)

        assert result is True
        saved_data = json.loads(test_file.read_text())
        assert saved_data == test_data

    def test_write_io_error_returns_false(self, tmp_path: Path) -> None:
        """Test that IO error returns False."""
        test_file = tmp_path / "test.json"

        with patch('file_lock.file_lock') as mock_lock:
            mock_lock.side_effect = IOError("Disk full")

            result = safe_write_json(test_file, {"key": "value"})

            assert result is False


class TestSafeStateManager:
    """Test SafeStateManager class."""

    def test_init(self, tmp_path: Path) -> None:
        """Test SafeStateManager initialization."""
        test_file = tmp_path / "state.json"
        manager = SafeStateManager(test_file)

        assert manager.file_path == test_file
        assert manager._cache is None

    def test_load_creates_default(self, tmp_path: Path) -> None:
        """Test that load creates default state for new file."""
        test_file = tmp_path / "state.json"
        manager = SafeStateManager(test_file)

        state = manager.load()

        assert state == {"last_alert": {}, "open_signals": {}}
        assert manager._cache == state

    def test_load_existing_file(self, tmp_path: Path) -> None:
        """Test loading existing state file."""
        test_file = tmp_path / "state.json"
        existing_data = {"key": "value", "count": 10}
        test_file.write_text(json.dumps(existing_data))

        manager = SafeStateManager(test_file)
        state = manager.load()

        assert state == existing_data
        assert manager._cache == existing_data

    def test_save_data(self, tmp_path: Path) -> None:
        """Test saving state data."""
        test_file = tmp_path / "state.json"
        manager = SafeStateManager(test_file)
        test_data = {"key": "value", "count": 5}

        result = manager.save(test_data)

        assert result is True
        assert manager._cache == test_data
        saved_data = json.loads(test_file.read_text())
        assert saved_data == test_data

    def test_save_uses_cache_when_no_data(self, tmp_path: Path) -> None:
        """Test that save uses cache when no data provided."""
        test_file = tmp_path / "state.json"
        manager = SafeStateManager(test_file)
        manager._cache = {"cached": "data"}

        result = manager.save()

        assert result is True
        saved_data = json.loads(test_file.read_text())
        assert saved_data == {"cached": "data"}

    def test_save_returns_false_when_no_cache(self, tmp_path: Path) -> None:
        """Test that save returns False when no cache."""
        test_file = tmp_path / "state.json"
        manager = SafeStateManager(test_file)

        result = manager.save()

        assert result is False

    def test_update_key(self, tmp_path: Path) -> None:
        """Test updating a specific key."""
        test_file = tmp_path / "state.json"
        test_file.write_text('{"existing": "value"}')
        manager = SafeStateManager(test_file)

        result = manager.update("new_key", "new_value")

        assert result is True
        saved_data = json.loads(test_file.read_text())
        assert saved_data["new_key"] == "new_value"
        assert saved_data["existing"] == "value"

    def test_update_loads_if_no_cache(self, tmp_path: Path) -> None:
        """Test that update loads file if cache is empty."""
        test_file = tmp_path / "state.json"
        test_file.write_text('{"existing": "value"}')
        manager = SafeStateManager(test_file)

        # Cache should be None initially
        assert manager._cache is None

        result = manager.update("new_key", "new_value")

        assert result is True
        assert manager._cache is not None
        assert manager._cache["existing"] == "value"
        assert manager._cache["new_key"] == "new_value"

    def test_get_existing_key(self, tmp_path: Path) -> None:
        """Test getting an existing key."""
        test_file = tmp_path / "state.json"
        test_file.write_text('{"key": "value"}')
        manager = SafeStateManager(test_file)

        result = manager.get("key")

        assert result == "value"

    def test_get_nonexistent_key_returns_default(self, tmp_path: Path) -> None:
        """Test getting nonexistent key returns default."""
        test_file = tmp_path / "state.json"
        test_file.write_text('{"key": "value"}')
        manager = SafeStateManager(test_file)

        result = manager.get("nonexistent", "default_value")

        assert result == "default_value"

    def test_get_loads_if_no_cache(self, tmp_path: Path) -> None:
        """Test that get loads file if cache is empty."""
        test_file = tmp_path / "state.json"
        test_file.write_text('{"key": "value"}')
        manager = SafeStateManager(test_file)

        # Cache should be None initially
        assert manager._cache is None

        result = manager.get("key")

        assert result == "value"
        assert manager._cache is not None

    def test_get_with_no_file_returns_default(self, tmp_path: Path) -> None:
        """Test get with no file returns default."""
        test_file = tmp_path / "nonexistent.json"
        manager = SafeStateManager(test_file)

        result = manager.get("key", "default")

        assert result == "default"

    def test_full_workflow(self, tmp_path: Path) -> None:
        """Test complete workflow: load, update, save, get."""
        test_file = tmp_path / "state.json"
        manager = SafeStateManager(test_file)

        # Initial load (creates defaults)
        state = manager.load()
        assert "last_alert" in state

        # Update values
        manager.update("counter", 1)
        manager.update("symbols", ["BTC", "ETH"])

        # Get values
        assert manager.get("counter") == 1
        assert manager.get("symbols") == ["BTC", "ETH"]

        # Create new manager and verify persistence
        manager2 = SafeStateManager(test_file)
        state2 = manager2.load()
        assert state2["counter"] == 1
        assert state2["symbols"] == ["BTC", "ETH"]


class TestConcurrentAccess:
    """Test concurrent access patterns."""

    def test_concurrent_writes(self, tmp_path: Path) -> None:
        """Test that concurrent writes are serialized."""
        test_file = tmp_path / "state.json"
        test_file.write_text('{"counter": 0}')

        errors: List[Exception] = []

        def increment() -> None:
            try:
                manager = SafeStateManager(test_file)
                state = manager.load()
                state["counter"] += 1
                time.sleep(0.02)
                manager.save(state)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=increment) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors
        assert len(errors) == 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_json_file(self, tmp_path: Path) -> None:
        """Test reading empty file."""
        test_file = tmp_path / "empty.json"
        test_file.write_text("")

        result = safe_read_json(test_file)
        assert result == {}  # Returns default on error

    def test_json_array_file(self, tmp_path: Path) -> None:
        """Test reading JSON array (not object)."""
        test_file = tmp_path / "array.json"
        test_file.write_text("[1, 2, 3]")

        result = safe_read_json(test_file)
        # The function casts to Dict, behavior depends on implementation
        # This tests current behavior

    def test_unicode_content(self, tmp_path: Path) -> None:
        """Test handling unicode content."""
        test_file = tmp_path / "unicode.json"
        test_data = {"message": "Hello 世界 🌍", "emoji": "🚀"}

        safe_write_json(test_file, test_data)
        result = safe_read_json(test_file)

        assert result == test_data

    def test_very_large_data(self, tmp_path: Path) -> None:
        """Test handling large data."""
        test_file = tmp_path / "large.json"
        test_data = {"items": list(range(10000))}

        result = safe_write_json(test_file, test_data)
        assert result is True

        loaded = safe_read_json(test_file)
        assert loaded == test_data

    def test_deeply_nested_data(self, tmp_path: Path) -> None:
        """Test deeply nested data structures."""
        test_file = tmp_path / "nested.json"

        # Create deeply nested structure
        data: Dict[str, Any] = {"level": 0}
        current = data
        for i in range(1, 50):
            current["nested"] = {"level": i}
            current = current["nested"]

        result = safe_write_json(test_file, data)
        assert result is True

        loaded = safe_read_json(test_file)
        assert loaded["level"] == 0
        assert loaded["nested"]["level"] == 1


# Fixtures
@pytest.fixture  # type: ignore[untyped-decorator]
def state_manager(tmp_path: Path) -> SafeStateManager:
    """Fixture providing a SafeStateManager instance."""
    return SafeStateManager(tmp_path / "state.json")


@pytest.fixture  # type: ignore[untyped-decorator]
def populated_state_file(tmp_path: Path) -> Path:
    """Fixture providing a pre-populated state file."""
    file_path = tmp_path / "populated.json"
    data = {
        "last_alert": {"BTC/USDT": 1234567890},
        "open_signals": {"SIG-001": {"symbol": "BTC/USDT"}},
        "counter": 42
    }
    file_path.write_text(json.dumps(data))
    return file_path


class TestWithFixtures:
    """Tests using fixtures."""

    def test_state_manager_fixture(self, state_manager: SafeStateManager) -> None:
        """Test using state manager fixture."""
        state = state_manager.load()
        assert "last_alert" in state

    def test_populated_file_fixture(self, populated_state_file: Path) -> None:
        """Test using populated file fixture."""
        manager = SafeStateManager(populated_state_file)
        state = manager.load()

        assert state["counter"] == 42
        assert "BTC/USDT" in state["last_alert"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
