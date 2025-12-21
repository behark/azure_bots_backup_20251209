"""
Unit tests for Signal Statistics Tracker (signal_stats.py).

Tests signal recording, P&L calculations, statistics, and message formatting.
Target: 80%+ code coverage

Run tests:
    python3 -m pytest tests/test_signal_stats.py -v
    python3 -m pytest tests/test_signal_stats.py --cov=signal_stats --cov-report=html
"""

import json
import pytest  # type: ignore[import-not-found]
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, cast

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_stats import SignalStats, SignalRecord


class TestSignalRecord:
    """Test SignalRecord data class."""

    def test_signal_record_creation(self) -> None:
        """Test creating a SignalRecord object."""
        record = SignalRecord(
            signal_id="TEST-001",
            symbol="BTC/USDT",
            direction="LONG",
            entry=50000.0,
            exit=52000.0,
            result="TP1",
            pnl_pct=4.0,
            created_at="2025-01-01T00:00:00+00:00",
            closed_at="2025-01-01T01:00:00+00:00",
            extra={"timeframe": "1h"}
        )

        assert record.signal_id == "TEST-001"
        assert record.symbol == "BTC/USDT"
        assert record.direction == "LONG"
        assert record.entry == 50000.0
        assert record.exit == 52000.0
        assert record.result == "TP1"
        assert record.pnl_pct == 4.0
        assert record.extra == {"timeframe": "1h"}

    def test_signal_record_empty_extra(self) -> None:
        """Test SignalRecord with empty extra dict."""
        record = SignalRecord(
            signal_id="TEST-002",
            symbol="ETH/USDT",
            direction="SHORT",
            entry=3000.0,
            exit=2900.0,
            result="TP2",
            pnl_pct=3.33,
            created_at="2025-01-01T00:00:00+00:00",
            closed_at="2025-01-01T02:00:00+00:00",
            extra={}
        )

        assert record.extra == {}


class TestSignalStatsInitialization:
    """Test SignalStats initialization."""

    def test_init_creates_file(self, tmp_path: Path) -> None:
        """Test that initialization creates parent directory."""
        stats_file = tmp_path / "subdir" / "stats.json"
        stats = SignalStats("Test Bot", stats_file)

        assert stats.bot_name == "Test Bot"
        assert stats.file_path == stats_file
        assert stats_file.parent.exists()

    def test_init_loads_existing_data(self, tmp_path: Path) -> None:
        """Test loading existing stats file."""
        stats_file = tmp_path / "stats.json"
        existing_data = {
            "open": {"SIG-001": {"symbol": "BTC/USDT", "direction": "LONG", "entry": 50000}},
            "history": [{"id": "OLD-001", "result": "TP1", "pnl_pct": 2.5}]
        }
        stats_file.write_text(json.dumps(existing_data))

        stats = SignalStats("Test Bot", stats_file)

        open_data = cast(Dict[str, Any], stats.data.get("open", {}))
        history_data = cast(List[Any], stats.data.get("history", []))
        assert "SIG-001" in open_data
        assert len(history_data) == 1

    def test_init_handles_corrupted_file(self, tmp_path: Path) -> None:
        """Test handling of corrupted JSON file."""
        stats_file = tmp_path / "stats.json"
        stats_file.write_text("not valid json {{{")

        stats = SignalStats("Test Bot", stats_file)

        # Should return default empty structure
        assert stats.data == {"open": {}, "history": []}


class TestRecordOpen:
    """Test recording open positions."""

    def test_record_open_basic(self, tmp_path: Path) -> None:
        """Test basic open position recording."""
        stats = SignalStats("Test Bot", tmp_path / "stats.json")

        stats.record_open(
            signal_id="SIG-001",
            symbol="BTC/USDT",
            direction="LONG",
            entry=50000.0,
            created_at="2025-01-01T00:00:00+00:00"
        )

        open_positions = cast(Dict[str, Any], stats.data.get("open", {}))
        assert "SIG-001" in open_positions
        position = cast(Dict[str, Any], open_positions["SIG-001"])
        assert position["symbol"] == "BTC/USDT"
        assert position["direction"] == "LONG"
        assert position["entry"] == 50000.0

    def test_record_open_with_extra(self, tmp_path: Path) -> None:
        """Test open position with extra data."""
        stats = SignalStats("Test Bot", tmp_path / "stats.json")

        stats.record_open(
            signal_id="SIG-002",
            symbol="ETH/USDT",
            direction="SHORT",
            entry=3000.0,
            created_at="2025-01-01T00:00:00+00:00",
            extra={"timeframe": "4h", "exchange": "binance"}
        )

        open_data = cast(Dict[str, Any], stats.data["open"])
        record = cast(Dict[str, Any], open_data["SIG-002"])
        extra = cast(Dict[str, Any], record["extra"])
        assert extra["timeframe"] == "4h"
        assert extra["exchange"] == "binance"

    def test_record_open_persists_to_file(self, tmp_path: Path) -> None:
        """Test that open position is saved to file."""
        stats_file = tmp_path / "stats.json"
        stats = SignalStats("Test Bot", stats_file)

        stats.record_open(
            signal_id="SIG-003",
            symbol="SOL/USDT",
            direction="LONG",
            entry=100.0,
            created_at="2025-01-01T00:00:00+00:00"
        )

        # Read file directly to verify persistence
        saved_data = json.loads(stats_file.read_text())
        assert "SIG-003" in saved_data["open"]

    def test_record_multiple_open_positions(self, tmp_path: Path) -> None:
        """Test recording multiple open positions."""
        stats = SignalStats("Test Bot", tmp_path / "stats.json")

        for i in range(5):
            stats.record_open(
                signal_id=f"SIG-{i:03d}",
                symbol=f"TOKEN{i}/USDT",
                direction="LONG" if i % 2 == 0 else "SHORT",
                entry=100.0 * (i + 1),
                created_at="2025-01-01T00:00:00+00:00"
            )

        open_data = cast(Dict[str, Any], stats.data["open"])
        assert len(open_data) == 5


class TestRecordClose:
    """Test closing positions."""

    def test_record_close_tp1(self, tmp_path: Path) -> None:
        """Test closing position with TP1 result."""
        stats = SignalStats("Test Bot", tmp_path / "stats.json")

        # Open position first
        stats.record_open(
            signal_id="SIG-001",
            symbol="BTC/USDT",
            direction="LONG",
            entry=50000.0,
            created_at="2025-01-01T00:00:00+00:00"
        )

        # Close with TP1
        record = stats.record_close(
            signal_id="SIG-001",
            exit_price=52000.0,
            result="TP1",
            closed_at="2025-01-01T01:00:00+00:00"
        )

        assert record is not None
        assert record.result == "TP1"
        assert record.pnl_pct == pytest.approx(4.0, rel=0.01)
        open_data = cast(Dict[str, Any], stats.data["open"])
        history_data = cast(List[Any], stats.data["history"])
        assert "SIG-001" not in open_data
        assert len(history_data) == 1

    def test_record_close_sl(self, tmp_path: Path) -> None:
        """Test closing position with SL result."""
        stats = SignalStats("Test Bot", tmp_path / "stats.json")

        stats.record_open(
            signal_id="SIG-002",
            symbol="ETH/USDT",
            direction="LONG",
            entry=3000.0,
            created_at="2025-01-01T00:00:00+00:00"
        )

        record = stats.record_close(
            signal_id="SIG-002",
            exit_price=2850.0,
            result="SL"
        )

        assert record is not None
        assert record.result == "SL"
        assert record.pnl_pct == pytest.approx(-5.0, rel=0.01)

    def test_record_close_nonexistent_signal(self, tmp_path: Path) -> None:
        """Test closing a signal that doesn't exist."""
        stats = SignalStats("Test Bot", tmp_path / "stats.json")

        record = stats.record_close(
            signal_id="NONEXISTENT",
            exit_price=100.0,
            result="TP1"
        )

        assert record is None

    def test_record_close_short_position(self, tmp_path: Path) -> None:
        """Test closing a SHORT position."""
        stats = SignalStats("Test Bot", tmp_path / "stats.json")

        stats.record_open(
            signal_id="SIG-003",
            symbol="BTC/USDT",
            direction="SHORT",
            entry=50000.0,
            created_at="2025-01-01T00:00:00+00:00"
        )

        record = stats.record_close(
            signal_id="SIG-003",
            exit_price=48000.0,
            result="TP1"
        )

        # SHORT: profit when price goes down
        assert record is not None
        assert record.pnl_pct == pytest.approx(4.0, rel=0.01)


class TestDiscard:
    """Test discarding positions."""

    def test_discard_existing_signal(self, tmp_path: Path) -> None:
        """Test discarding an existing open signal."""
        stats = SignalStats("Test Bot", tmp_path / "stats.json")

        stats.record_open(
            signal_id="SIG-001",
            symbol="BTC/USDT",
            direction="LONG",
            entry=50000.0,
            created_at="2025-01-01T00:00:00+00:00"
        )

        stats.discard("SIG-001")

        open_data = cast(Dict[str, Any], stats.data["open"])
        history_data = cast(List[Any], stats.data["history"])
        assert "SIG-001" not in open_data
        # Should NOT be in history (discarded, not closed)
        assert len(history_data) == 0

    def test_discard_nonexistent_signal(self, tmp_path: Path) -> None:
        """Test discarding a signal that doesn't exist."""
        stats = SignalStats("Test Bot", tmp_path / "stats.json")

        # Should not raise error
        stats.discard("NONEXISTENT")


class TestCalculatePnL:
    """Test P&L calculations."""

    def test_pnl_long_profit(self) -> None:
        """Test P&L for profitable LONG trade."""
        pnl = SignalStats._calculate_pnl("LONG", 100.0, 110.0)
        assert pnl == pytest.approx(10.0, rel=0.01)

    def test_pnl_long_loss(self) -> None:
        """Test P&L for losing LONG trade."""
        pnl = SignalStats._calculate_pnl("LONG", 100.0, 95.0)
        assert pnl == pytest.approx(-5.0, rel=0.01)

    def test_pnl_short_profit(self) -> None:
        """Test P&L for profitable SHORT trade."""
        pnl = SignalStats._calculate_pnl("SHORT", 100.0, 90.0)
        assert pnl == pytest.approx(10.0, rel=0.01)

    def test_pnl_short_loss(self) -> None:
        """Test P&L for losing SHORT trade."""
        pnl = SignalStats._calculate_pnl("SHORT", 100.0, 105.0)
        assert pnl == pytest.approx(-5.0, rel=0.01)

    def test_pnl_bullish_direction(self) -> None:
        """Test P&L with BULLISH direction (should work like LONG)."""
        pnl = SignalStats._calculate_pnl("BULLISH", 100.0, 110.0)
        assert pnl == pytest.approx(10.0, rel=0.01)

    def test_pnl_bearish_direction(self) -> None:
        """Test P&L with BEARISH direction (should work like SHORT)."""
        pnl = SignalStats._calculate_pnl("BEARISH", 100.0, 90.0)
        assert pnl == pytest.approx(10.0, rel=0.01)

    def test_pnl_buy_direction(self) -> None:
        """Test P&L with BUY direction (should work like LONG)."""
        pnl = SignalStats._calculate_pnl("BUY", 100.0, 110.0)
        assert pnl == pytest.approx(10.0, rel=0.01)

    def test_pnl_zero_entry(self) -> None:
        """Test P&L with zero entry (edge case)."""
        pnl = SignalStats._calculate_pnl("LONG", 0.0, 100.0)
        assert pnl == 0.0

    def test_pnl_case_insensitive(self) -> None:
        """Test P&L with lowercase direction."""
        pnl = SignalStats._calculate_pnl("long", 100.0, 110.0)
        assert pnl == pytest.approx(10.0, rel=0.01)


class TestSymbolTPSLCounts:
    """Test TP/SL counting by symbol."""

    def test_counts_empty_history(self, tmp_path: Path) -> None:
        """Test counts with empty history."""
        stats = SignalStats("Test Bot", tmp_path / "stats.json")

        counts = stats.symbol_tp_sl_counts("BTC/USDT")

        assert counts == {"TP1": 0, "TP2": 0, "SL": 0}

    def test_counts_with_history(self, tmp_path: Path) -> None:
        """Test counts with trades in history."""
        stats = SignalStats("Test Bot", tmp_path / "stats.json")

        # Add some trades to history
        stats.data["history"] = [
            {"symbol": "BTC/USDT", "result": "TP1"},
            {"symbol": "BTC/USDT", "result": "TP1"},
            {"symbol": "BTC/USDT", "result": "TP2"},
            {"symbol": "BTC/USDT", "result": "SL"},
            {"symbol": "ETH/USDT", "result": "TP1"},  # Different symbol
        ]

        counts = stats.symbol_tp_sl_counts("BTC/USDT")

        assert counts["TP1"] == 2
        assert counts["TP2"] == 1
        assert counts["SL"] == 1

    def test_counts_filters_by_symbol(self, tmp_path: Path) -> None:
        """Test that counts are filtered by symbol."""
        stats = SignalStats("Test Bot", tmp_path / "stats.json")

        stats.data["history"] = [
            {"symbol": "BTC/USDT", "result": "TP1"},
            {"symbol": "ETH/USDT", "result": "TP1"},
            {"symbol": "SOL/USDT", "result": "TP2"},
        ]

        btc_counts = stats.symbol_tp_sl_counts("BTC/USDT")
        eth_counts = stats.symbol_tp_sl_counts("ETH/USDT")

        assert btc_counts["TP1"] == 1
        assert eth_counts["TP1"] == 1


class TestGetSummary:
    """Test summary statistics."""

    def test_summary_empty_history(self, tmp_path: Path) -> None:
        """Test summary with empty history."""
        stats = SignalStats("Test Bot", tmp_path / "stats.json")

        summary = stats.get_summary()

        assert summary["win_rate"] == 0.0
        assert summary["tp_hits"] == 0
        assert summary["sl_hits"] == 0
        assert summary["total_pnl"] == 0.0

    def test_summary_with_wins(self, tmp_path: Path) -> None:
        """Test summary with winning trades."""
        stats = SignalStats("Test Bot", tmp_path / "stats.json")

        stats.data["history"] = [
            {"result": "TP1", "pnl_pct": 3.0},
            {"result": "TP2", "pnl_pct": 5.0},
            {"result": "TP1", "pnl_pct": 2.0},
            {"result": "SL", "pnl_pct": -2.0},
        ]

        summary = stats.get_summary()

        assert summary["tp_hits"] == 3
        assert summary["sl_hits"] == 1
        assert summary["win_rate"] == pytest.approx(75.0, rel=0.01)
        assert summary["total_pnl"] == pytest.approx(8.0, rel=0.01)

    def test_summary_all_losses(self, tmp_path: Path) -> None:
        """Test summary with all losing trades."""
        stats = SignalStats("Test Bot", tmp_path / "stats.json")

        stats.data["history"] = [
            {"result": "SL", "pnl_pct": -2.0},
            {"result": "SL", "pnl_pct": -3.0},
        ]

        summary = stats.get_summary()

        assert summary["win_rate"] == 0.0
        assert summary["tp_hits"] == 0
        assert summary["sl_hits"] == 2
        assert summary["total_pnl"] == pytest.approx(-5.0, rel=0.01)


class TestBuildSummaryMessage:
    """Test summary message building."""

    def test_build_summary_message_win(self, tmp_path: Path) -> None:
        """Test building summary message for a win."""
        stats = SignalStats("Test Bot", tmp_path / "stats.json")
        stats.data["history"] = [{"result": "TP1", "pnl_pct": 3.0}]

        record = SignalRecord(
            signal_id="SIG-001",
            symbol="BTC/USDT",
            direction="LONG",
            entry=50000.0,
            exit=52000.0,
            result="TP1",
            pnl_pct=4.0,
            created_at="2025-01-01T00:00:00+00:00",
            closed_at="2025-01-01T01:00:00+00:00",
            extra={}
        )

        message = stats.build_summary_message(record)

        assert "TAKE PROFIT HIT" in message
        assert "BTC/USDT" in message
        assert "LONG" in message
        assert "50000" in message
        assert "52000" in message
        assert "+4.00%" in message

    def test_build_summary_message_loss(self, tmp_path: Path) -> None:
        """Test building summary message for a loss."""
        stats = SignalStats("Test Bot", tmp_path / "stats.json")

        record = SignalRecord(
            signal_id="SIG-002",
            symbol="ETH/USDT",
            direction="SHORT",
            entry=3000.0,
            exit=3150.0,
            result="SL",
            pnl_pct=-5.0,
            created_at="2025-01-01T00:00:00+00:00",
            closed_at="2025-01-01T01:00:00+00:00",
            extra={}
        )

        message = stats.build_summary_message(record)

        assert "STOP LOSS HIT" in message
        assert "ETH/USDT" in message
        assert "-5.00%" in message


class TestBuildInitialAlert:
    """Test initial alert message building."""

    def test_build_initial_alert_long(self, tmp_path: Path) -> None:
        """Test building initial alert for LONG signal."""
        stats = SignalStats("Test Bot", tmp_path / "stats.json")

        message = stats.build_initial_alert(
            symbol="BTC/USDT",
            direction="LONG",
            entry=50000.0,
            tp1=52000.0,
            tp2=54000.0,
            sl=48000.0
        )

        assert "LONG SIGNAL" in message
        assert "BTC/USDT" in message
        assert "50000" in message
        assert "52000" in message
        assert "54000" in message
        assert "48000" in message

    def test_build_initial_alert_short(self, tmp_path: Path) -> None:
        """Test building initial alert for SHORT signal."""
        stats = SignalStats("Test Bot", tmp_path / "stats.json")

        message = stats.build_initial_alert(
            symbol="ETH/USDT",
            direction="SHORT",
            entry=3000.0,
            tp1=2800.0,
            sl=3150.0
        )

        assert "SHORT SIGNAL" in message
        assert "ETH/USDT" in message

    def test_build_initial_alert_with_history(self, tmp_path: Path) -> None:
        """Test initial alert includes historical stats."""
        stats = SignalStats("Test Bot", tmp_path / "stats.json")

        # Add some history
        stats.data["history"] = [
            {"symbol": "BTC/USDT", "result": "TP1"},
            {"symbol": "BTC/USDT", "result": "TP2"},
            {"symbol": "BTC/USDT", "result": "SL"},
        ]

        message = stats.build_initial_alert(
            symbol="BTC/USDT",
            direction="LONG",
            entry=50000.0
        )

        assert "History" in message
        assert "TP1 1" in message or "TP1: 1" in message
        assert "Win rate" in message or "win rate" in message.lower()


class TestFilePersistence:
    """Test file persistence and data integrity."""

    def test_data_survives_reload(self, tmp_path: Path) -> None:
        """Test that data survives instance recreation."""
        stats_file = tmp_path / "stats.json"

        # Create and populate
        stats1 = SignalStats("Test Bot", stats_file)
        stats1.record_open(
            signal_id="SIG-001",
            symbol="BTC/USDT",
            direction="LONG",
            entry=50000.0,
            created_at="2025-01-01T00:00:00+00:00"
        )

        # Recreate from file
        stats2 = SignalStats("Test Bot", stats_file)

        open_data = cast(Dict[str, Any], stats2.data["open"])
        assert "SIG-001" in open_data
        position = cast(Dict[str, Any], open_data["SIG-001"])
        assert position["symbol"] == "BTC/USDT"

    def test_history_accumulates(self, tmp_path: Path) -> None:
        """Test that history accumulates across sessions."""
        stats_file = tmp_path / "stats.json"

        # Session 1
        stats1 = SignalStats("Test Bot", stats_file)
        stats1.record_open("SIG-001", "BTC/USDT", "LONG", 50000.0, "2025-01-01T00:00:00")
        stats1.record_close("SIG-001", 52000.0, "TP1")

        # Session 2
        stats2 = SignalStats("Test Bot", stats_file)
        stats2.record_open("SIG-002", "ETH/USDT", "SHORT", 3000.0, "2025-01-02T00:00:00")
        stats2.record_close("SIG-002", 2900.0, "TP1")

        # Session 3 - verify all history
        stats3 = SignalStats("Test Bot", stats_file)
        history_data = cast(List[Any], stats3.data["history"])
        assert len(history_data) == 2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_extra_on_open(self, tmp_path: Path) -> None:
        """Test recording with None extra."""
        stats = SignalStats("Test Bot", tmp_path / "stats.json")

        stats.record_open(
            signal_id="SIG-001",
            symbol="BTC/USDT",
            direction="LONG",
            entry=50000.0,
            created_at="2025-01-01T00:00:00+00:00",
            extra=None
        )

        open_data = cast(Dict[str, Any], stats.data["open"])
        position = cast(Dict[str, Any], open_data["SIG-001"])
        assert position["extra"] == {}

    def test_very_small_pnl(self, tmp_path: Path) -> None:
        """Test P&L with very small price difference."""
        stats = SignalStats("Test Bot", tmp_path / "stats.json")

        stats.record_open("SIG-001", "BTC/USDT", "LONG", 50000.0, "2025-01-01T00:00:00")
        record = stats.record_close("SIG-001", 50000.01, "TP1")

        assert record is not None
        assert record.pnl_pct == pytest.approx(0.00002, rel=0.1)

    def test_large_pnl(self, tmp_path: Path) -> None:
        """Test P&L with large price movement."""
        stats = SignalStats("Test Bot", tmp_path / "stats.json")

        stats.record_open("SIG-001", "BTC/USDT", "LONG", 10000.0, "2025-01-01T00:00:00")
        record = stats.record_close("SIG-001", 20000.0, "TP2")

        assert record is not None
        assert record.pnl_pct == pytest.approx(100.0, rel=0.01)


# Fixtures
@pytest.fixture  # type: ignore[untyped-decorator]
def stats_with_history(tmp_path: Path) -> SignalStats:
    """Fixture providing SignalStats with pre-populated history."""
    stats = SignalStats("Test Bot", tmp_path / "stats.json")
    stats.data["history"] = [
        {"id": "H1", "symbol": "BTC/USDT", "result": "TP1", "pnl_pct": 3.0},
        {"id": "H2", "symbol": "BTC/USDT", "result": "TP2", "pnl_pct": 5.0},
        {"id": "H3", "symbol": "BTC/USDT", "result": "SL", "pnl_pct": -2.0},
        {"id": "H4", "symbol": "ETH/USDT", "result": "TP1", "pnl_pct": 2.5},
    ]
    return stats


class TestWithFixture:
    """Tests using fixtures."""

    def test_summary_from_fixture(self, stats_with_history: SignalStats) -> None:
        """Test summary using fixture with pre-populated data."""
        summary = stats_with_history.get_summary()

        assert summary["tp_hits"] == 3
        assert summary["sl_hits"] == 1
        assert summary["total_pnl"] == pytest.approx(8.5, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
