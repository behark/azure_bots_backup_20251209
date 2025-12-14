#!/usr/bin/env python3
"""File locking utility for safe JSON state file operations."""

import fcntl
import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class FileLockError(Exception):
    """Raised when file locking fails."""
    pass


@contextmanager
def file_lock(file_path: Path, timeout: float = 5.0):
    """
    Context manager for file locking using fcntl.

    Usage:
        with file_lock(Path("state.json")):
            # read/write operations

    Sets secure file permissions (0o600) on lock files for security.
    """
    lock_path = file_path.with_suffix(file_path.suffix + ".lock")
    lock_file = None

    try:
        # Create lock file with secure permissions (owner read/write only)
        lock_file = open(lock_path, 'w')
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()


def safe_read_json(file_path: Path, default: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Safely read JSON file with file locking.
    
    Args:
        file_path: Path to JSON file
        default: Default value if file doesn't exist or is invalid
        
    Returns:
        Parsed JSON data or default value
    """
    if default is None:
        default = {}
    
    if not file_path.exists():
        return default
    
    try:
        with file_lock(file_path):
            data = json.loads(file_path.read_text())
            return data
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("Failed to read %s: %s", file_path, e)
        return default


def safe_write_json(file_path: Path, data: Dict[str, Any], indent: int = 2) -> bool:
    """
    Safely write JSON file with file locking.

    Args:
        file_path: Path to JSON file
        data: Data to write
        indent: JSON indentation level

    Returns:
        True if successful, False otherwise

    Sets secure permissions on directories (0o700) and files (0o600).
    """
    try:
        # Create parent directory with secure permissions (owner only)
        file_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        with file_lock(file_path):
            file_path.write_text(json.dumps(data, indent=indent))
            # Set secure file permissions (owner read/write only)
            os.chmod(file_path, 0o600)
        return True
    except (IOError, OSError) as e:
        logger.error("Failed to write %s: %s", file_path, e)
        return False


class SafeStateManager:
    """
    Thread-safe state manager for bot state files.
    Uses file locking to prevent race conditions.
    """
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._cache: Optional[Dict[str, Any]] = None
    
    def load(self) -> Dict[str, Any]:
        """Load state from file."""
        data = safe_read_json(self.file_path, {"last_alert": {}, "open_signals": {}})
        self._cache = data
        return data
    
    def save(self, data: Optional[Dict[str, Any]] = None) -> bool:
        """Save state to file."""
        if data is not None:
            self._cache = data
        if self._cache is None:
            return False
        return safe_write_json(self.file_path, self._cache)
    
    def update(self, key: str, value: Any) -> bool:
        """Update a specific key in the state."""
        if self._cache is None:
            self.load()
        self._cache[key] = value
        return self.save()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a specific key from the state."""
        if self._cache is None:
            self.load()
        return self._cache.get(key, default)
