"""
Unit tests for TP/SL Calculator (tp_sl_calculator.py).

Tests all calculation methods, validation logic, edge cases, and error handling.
Target: 80%+ code coverage

Run tests:
    python3 -m pytest tests/test_tp_sl_calculator.py -v
    python3 -m pytest tests/test_tp_sl_calculator.py --cov=tp_sl_calculator --cov-report=html
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tp_sl_calculator import (
    TPSLCalculator,
    TradeLevels,
    CalculationMethod,
    calculate_atr,
    quick_calculate,
)


class TestTradeLevels:
    """Test TradeLevels data class."""

    def test_trade_levels_creation(self):
        """Test creating a valid TradeLevels object."""
        levels = TradeLevels(
            entry=100.0,
            stop_loss=95.0,
            take_profit_1=110.0,
            take_profit_2=120.0,
            take_profit_3=None,
            risk_reward_1=2.0,
            risk_reward_2=4.0,
            risk_amount=5.0,
            is_valid=True,
            rejection_reason=None
        )

        assert levels.entry == 100.0
        assert levels.stop_loss == 95.0
        assert levels.take_profit_1 == 110.0
        assert levels.is_valid is True

    def test_trade_levels_to_dict(self):
        """Test converting TradeLevels to dictionary."""
        levels = TradeLevels(
            entry=100.0,
            stop_loss=95.0,
            take_profit_1=110.0,
            take_profit_2=120.0,
            risk_reward_1=2.0,
            risk_reward_2=4.0,
            risk_amount=5.0,
            is_valid=True
        )

        result = levels.to_dict()

        assert isinstance(result, dict)
        assert result['entry'] == 100.0
        assert result['stop_loss'] == 95.0
        assert result['is_valid'] is True


class TestTPSLCalculator:
    """Test TPSLCalculator main class."""

    def test_calculator_initialization(self):
        """Test calculator initialization with default parameters."""
        calc = TPSLCalculator()

        assert calc.min_risk_reward == 1.8
        assert calc.max_sl_percent == 5.0
        assert calc.min_sl_percent == 0.2
        assert calc.sl_buffer_percent == 0.75

    def test_calculator_custom_parameters(self):
        """Test calculator initialization with custom parameters."""
        calc = TPSLCalculator(
            min_risk_reward=2.5,
            max_sl_percent=3.0,
            min_sl_percent=0.5,
            sl_buffer_percent=1.0
        )

        assert calc.min_risk_reward == 2.5
        assert calc.max_sl_percent == 3.0
        assert calc.min_sl_percent == 0.5
        assert calc.sl_buffer_percent == 1.0


class TestATRCalculation:
    """Test ATR-based TP/SL calculations."""

    def test_atr_long_trade(self):
        """Test ATR calculation for LONG trade."""
        calc = TPSLCalculator(min_risk_reward=1.5)

        levels = calc.calculate(
            entry=100.0,
            direction="LONG",
            atr=2.5,
            sl_multiplier=1.5,
            tp1_multiplier=2.0,
            tp2_multiplier=3.5,
            method=CalculationMethod.ATR
        )

        assert levels.is_valid
        assert levels.stop_loss < levels.entry
        assert levels.take_profit_1 > levels.entry
        assert levels.take_profit_2 > levels.take_profit_1
        assert levels.risk_reward_1 > 0

    def test_atr_short_trade(self):
        """Test ATR calculation for SHORT trade."""
        calc = TPSLCalculator(min_risk_reward=1.5)

        levels = calc.calculate(
            entry=100.0,
            direction="SHORT",
            atr=2.5,
            sl_multiplier=1.5,
            tp1_multiplier=2.0,
            tp2_multiplier=3.5,
            method=CalculationMethod.ATR
        )

        assert levels.is_valid
        assert levels.stop_loss > levels.entry
        assert levels.take_profit_1 < levels.entry
        assert levels.take_profit_2 < levels.take_profit_1

    def test_atr_with_custom_sl(self):
        """Test ATR calculation with custom stop loss."""
        calc = TPSLCalculator()

        custom_sl = 96.0
        levels = calc.calculate(
            entry=100.0,
            direction="LONG",
            atr=2.5,
            sl_multiplier=1.5,
            tp1_multiplier=2.0,
            tp2_multiplier=3.5,
            custom_sl=custom_sl
        )

        assert levels.stop_loss == custom_sl

    def test_atr_fallback_when_none(self):
        """Test ATR fallback to 1% when ATR is None."""
        calc = TPSLCalculator()

        levels = calc.calculate(
            entry=100.0,
            direction="LONG",
            atr=None,  # No ATR provided
            sl_multiplier=1.5,
            tp1_multiplier=2.0,
            tp2_multiplier=3.5
        )

        # Should still produce valid levels using 1% fallback
        assert levels.stop_loss < levels.entry
        assert levels.take_profit_1 > levels.entry

    def test_atr_with_buffer(self):
        """Test that buffer is applied to stop loss."""
        calc = TPSLCalculator(sl_buffer_percent=0.75)

        levels = calc.calculate(
            entry=100.0,
            direction="LONG",
            atr=2.0,
            sl_multiplier=1.0,  # 1x ATR
            tp1_multiplier=2.0,
            tp2_multiplier=3.0
        )

        # SL should be: 100 - (2.0 * 1.0075 * 1.0) ≈ 97.985
        expected_sl = 100.0 - (2.0 * 1.0075)
        assert abs(levels.stop_loss - expected_sl) < 0.01


class TestStructureBasedCalculation:
    """Test structure-based TP/SL calculations."""

    def test_structure_long_trade(self):
        """Test structure calculation for LONG trade."""
        calc = TPSLCalculator(min_risk_reward=1.5)

        levels = calc.calculate(
            entry=100.0,
            direction="LONG",
            swing_low=95.0,
            swing_high=None,
            tp1_multiplier=2.0,
            tp2_multiplier=3.0,
            method=CalculationMethod.STRUCTURE
        )

        assert levels.is_valid
        assert levels.stop_loss < 95.0  # Below swing low with buffer
        assert levels.take_profit_1 > levels.entry

    def test_structure_short_trade(self):
        """Test structure calculation for SHORT trade."""
        calc = TPSLCalculator(min_risk_reward=1.5)

        levels = calc.calculate(
            entry=100.0,
            direction="SHORT",
            swing_high=105.0,
            swing_low=None,
            tp1_multiplier=2.0,
            tp2_multiplier=3.0,
            method=CalculationMethod.STRUCTURE
        )

        assert levels.is_valid
        assert levels.stop_loss > 105.0  # Above swing high with buffer
        assert levels.take_profit_1 < levels.entry

    def test_structure_missing_swing_low(self):
        """Test structure calculation fails without required swing low."""
        calc = TPSLCalculator()

        levels = calc.calculate(
            entry=100.0,
            direction="LONG",
            swing_low=None,  # Missing required swing low
            swing_high=None,
            tp1_multiplier=2.0,
            tp2_multiplier=3.0,
            method=CalculationMethod.STRUCTURE
        )

        assert not levels.is_valid
        assert "swing low" in levels.rejection_reason.lower()

    def test_structure_missing_swing_high(self):
        """Test structure calculation fails without required swing high."""
        calc = TPSLCalculator()

        levels = calc.calculate(
            entry=100.0,
            direction="SHORT",
            swing_high=None,  # Missing required swing high
            swing_low=None,
            tp1_multiplier=2.0,
            tp2_multiplier=3.0,
            method=CalculationMethod.STRUCTURE
        )

        assert not levels.is_valid
        assert "swing high" in levels.rejection_reason.lower()


class TestFibonacciCalculation:
    """Test Fibonacci-based TP/SL calculations."""

    def test_fibonacci_long_trade(self):
        """Test Fibonacci calculation for LONG trade."""
        calc = TPSLCalculator(min_risk_reward=1.5)

        levels = calc.calculate(
            entry=100.0,
            direction="LONG",
            swing_high=110.0,
            swing_low=90.0,
            method=CalculationMethod.FIBONACCI
        )

        assert levels.is_valid
        # TP1 should be at 0.618 * 20 = 12.36 above entry
        expected_tp1 = 100.0 + (20.0 * 0.618)
        assert abs(levels.take_profit_1 - expected_tp1) < 0.1

        # TP2 should be at 1.0 * 20 = 20 above entry
        expected_tp2 = 100.0 + 20.0
        assert abs(levels.take_profit_2 - expected_tp2) < 0.1

    def test_fibonacci_short_trade(self):
        """Test Fibonacci calculation for SHORT trade."""
        calc = TPSLCalculator(min_risk_reward=1.5)

        levels = calc.calculate(
            entry=100.0,
            direction="SHORT",
            swing_high=110.0,
            swing_low=90.0,
            method=CalculationMethod.FIBONACCI
        )

        assert levels.is_valid
        # TP1 should be 0.618 * 20 = 12.36 below entry
        expected_tp1 = 100.0 - (20.0 * 0.618)
        assert abs(levels.take_profit_1 - expected_tp1) < 0.1

    def test_fibonacci_missing_swing_points(self):
        """Test Fibonacci calculation fails without both swing points."""
        calc = TPSLCalculator()

        levels = calc.calculate(
            entry=100.0,
            direction="LONG",
            swing_high=None,
            swing_low=90.0,
            method=CalculationMethod.FIBONACCI
        )

        assert not levels.is_valid
        assert "fibonacci" in levels.rejection_reason.lower()


class TestValidation:
    """Test validation logic."""

    def test_stop_loss_too_wide(self):
        """Test rejection when stop loss is too wide."""
        calc = TPSLCalculator(max_sl_percent=3.0)

        levels = calc.calculate(
            entry=100.0,
            direction="LONG",
            atr=5.0,  # Very wide ATR
            sl_multiplier=3.0,  # Will create >3% SL
            tp1_multiplier=2.0,
            tp2_multiplier=3.0
        )

        assert not levels.is_valid
        assert "too far" in levels.rejection_reason.lower()

    def test_stop_loss_too_tight(self):
        """Test rejection when stop loss is too tight."""
        calc = TPSLCalculator(min_sl_percent=0.5)

        levels = calc.calculate(
            entry=100.0,
            direction="LONG",
            atr=0.1,  # Very small ATR
            sl_multiplier=0.5,
            tp1_multiplier=2.0,
            tp2_multiplier=3.0
        )

        assert not levels.is_valid
        assert "too tight" in levels.rejection_reason.lower()

    def test_risk_reward_too_low(self):
        """Test rejection when risk:reward ratio is too low."""
        calc = TPSLCalculator(min_risk_reward=2.5)

        levels = calc.calculate(
            entry=100.0,
            direction="LONG",
            atr=2.5,
            sl_multiplier=2.0,
            tp1_multiplier=2.0,  # Only 1R, below 2.5 minimum
            tp2_multiplier=3.0
        )

        assert not levels.is_valid
        assert "r:r too low" in levels.rejection_reason.lower()

    def test_long_inverted_levels(self):
        """Test rejection of inverted LONG trade levels."""
        calc = TPSLCalculator()

        # Manually create inverted levels (this shouldn't happen in practice)
        levels = calc._validate_and_build(
            entry=100.0,
            sl=105.0,  # SL above entry (invalid for LONG)
            tp1=110.0,
            tp2=115.0,
            tp3=None,
            direction="LONG"
        )

        assert not levels.is_valid
        assert "invalid long" in levels.rejection_reason.lower()

    def test_short_inverted_levels(self):
        """Test rejection of inverted SHORT trade levels."""
        calc = TPSLCalculator()

        # Manually create inverted levels
        levels = calc._validate_and_build(
            entry=100.0,
            sl=95.0,  # SL below entry (invalid for SHORT)
            tp1=90.0,
            tp2=85.0,
            tp3=None,
            direction="SHORT"
        )

        assert not levels.is_valid
        assert "invalid short" in levels.rejection_reason.lower()


class TestDirectionNormalization:
    """Test direction string normalization."""

    def test_bullish_normalization(self):
        """Test that BULLISH, BUY, LONG all normalize to LONG."""
        calc = TPSLCalculator()

        for direction in ["BULLISH", "BUY", "LONG", "bullish", "buy", "long"]:
            levels = calc.calculate(
                entry=100.0,
                direction=direction,
                atr=2.0,
                sl_multiplier=1.5,
                tp1_multiplier=2.0,
                tp2_multiplier=3.0
            )
            # All should produce SL below entry (LONG behavior)
            assert levels.stop_loss < levels.entry

    def test_bearish_normalization(self):
        """Test that BEARISH, SELL, SHORT normalize to SHORT."""
        calc = TPSLCalculator()

        for direction in ["BEARISH", "SELL", "SHORT", "bearish", "sell", "short"]:
            levels = calc.calculate(
                entry=100.0,
                direction=direction,
                atr=2.0,
                sl_multiplier=1.5,
                tp1_multiplier=2.0,
                tp2_multiplier=3.0
            )
            # All should produce SL above entry (SHORT behavior)
            assert levels.stop_loss > levels.entry


class TestPositionSizing:
    """Test position sizing calculations."""

    def test_position_sizing_basic(self):
        """Test basic position sizing calculation."""
        calc = TPSLCalculator()

        sizing = calc.calculate_position_size(
            account_balance=1000.0,
            risk_percent=2.0,
            entry=100.0,
            stop_loss=95.0,
            leverage=1.0
        )

        assert sizing['risk_amount_usdt'] == 20.0  # 2% of 1000
        assert sizing['risk_percent'] == 2.0
        assert sizing['price_risk_percent'] == 5.0  # 5% price risk
        assert abs(sizing['position_size_usdt'] - 400.0) < 0.1  # 20 / 0.05 = 400

    def test_position_sizing_with_leverage(self):
        """Test position sizing with leverage."""
        calc = TPSLCalculator()

        sizing = calc.calculate_position_size(
            account_balance=1000.0,
            risk_percent=2.0,
            entry=100.0,
            stop_loss=95.0,
            leverage=10.0
        )

        # Margin required should be position_size / leverage
        expected_margin = sizing['position_size_usdt'] / 10.0
        assert abs(sizing['margin_required'] - expected_margin) < 0.1


class TestTrailingStop:
    """Test trailing stop calculations."""

    def test_trailing_stop_long_not_activated(self):
        """Test trailing stop not activated when profit insufficient."""
        calc = TPSLCalculator()

        new_sl = calc.calculate_trailing_stop(
            entry=100.0,
            current_price=101.0,  # Only 1 ATR profit
            direction="LONG",
            atr=2.0,
            activation_atr=1.5,  # Needs 1.5 ATR profit
            trail_distance_atr=0.5,
            current_sl=95.0
        )

        # Should return current SL unchanged
        assert new_sl == 95.0

    def test_trailing_stop_long_activated(self):
        """Test trailing stop activation for LONG trade."""
        calc = TPSLCalculator()

        new_sl = calc.calculate_trailing_stop(
            entry=100.0,
            current_price=104.0,  # 2 ATR profit
            direction="LONG",
            atr=2.0,
            activation_atr=1.0,  # Activated at 1 ATR
            trail_distance_atr=0.5,  # Trail 0.5 ATR below price
            current_sl=95.0
        )

        # New SL should be 104 - (2.0 * 0.5) = 103.0
        expected_sl = 104.0 - 1.0
        assert abs(new_sl - expected_sl) < 0.01

    def test_trailing_stop_short_activated(self):
        """Test trailing stop activation for SHORT trade."""
        calc = TPSLCalculator()

        new_sl = calc.calculate_trailing_stop(
            entry=100.0,
            current_price=96.0,  # 2 ATR profit
            direction="SHORT",
            atr=2.0,
            activation_atr=1.0,
            trail_distance_atr=0.5,
            current_sl=105.0
        )

        # New SL should be 96 + (2.0 * 0.5) = 97.0
        expected_sl = 96.0 + 1.0
        assert abs(new_sl - expected_sl) < 0.01

    def test_trailing_stop_only_moves_favorably(self):
        """Test that trailing stop only moves in favorable direction."""
        calc = TPSLCalculator()

        # First move price up
        new_sl1 = calc.calculate_trailing_stop(
            entry=100.0,
            current_price=105.0,
            direction="LONG",
            atr=2.0,
            activation_atr=1.0,
            trail_distance_atr=0.5,
            current_sl=95.0
        )

        assert new_sl1 > 95.0  # SL should move up

        # Now price pulls back - SL should not move back down
        new_sl2 = calc.calculate_trailing_stop(
            entry=100.0,
            current_price=103.0,  # Price lower than before
            direction="LONG",
            atr=2.0,
            activation_atr=1.0,
            trail_distance_atr=0.5,
            current_sl=new_sl1  # Use previous SL
        )

        # SL should stay at new_sl1, not move down
        assert new_sl2 == new_sl1


class TestATRFunction:
    """Test standalone ATR calculation function."""

    def test_atr_with_dict_candles(self):
        """Test ATR calculation with dictionary candles."""
        candles = [
            {'high': 102, 'low': 98, 'close': 100},
            {'high': 105, 'low': 99, 'close': 103},
            {'high': 107, 'low': 102, 'close': 105},
            {'high': 106, 'low': 103, 'close': 104},
            {'high': 108, 'low': 104, 'close': 107},
        ]

        atr = calculate_atr(candles, period=3)

        assert atr is not None
        assert atr > 0

    def test_atr_with_list_candles(self):
        """Test ATR calculation with list/OHLCV candles."""
        candles = [
            [1000, 100, 102, 98, 100, 1000],  # timestamp, O, H, L, C, V
            [1001, 100, 105, 99, 103, 1000],
            [1002, 103, 107, 102, 105, 1000],
            [1003, 105, 106, 103, 104, 1000],
            [1004, 104, 108, 104, 107, 1000],
        ]

        atr = calculate_atr(candles, period=3)

        assert atr is not None
        assert atr > 0

    def test_atr_insufficient_data(self):
        """Test ATR returns None with insufficient data."""
        candles = [
            {'high': 102, 'low': 98, 'close': 100},
        ]

        atr = calculate_atr(candles, period=14)

        assert atr is None

    def test_atr_exact_period(self):
        """Test ATR with exactly required candles."""
        candles = [{'high': 100 + i, 'low': 95 + i, 'close': 98 + i} for i in range(15)]

        atr = calculate_atr(candles, period=14)

        assert atr is not None
        assert atr > 0


class TestQuickCalculate:
    """Test quick_calculate convenience function."""

    def test_quick_calculate_long(self):
        """Test quick_calculate with LONG trade."""
        levels = quick_calculate(
            entry=100.0,
            direction="LONG",
            atr=2.5
        )

        assert levels.is_valid
        assert levels.stop_loss < levels.entry
        assert levels.take_profit_1 > levels.entry

    def test_quick_calculate_short(self):
        """Test quick_calculate with SHORT trade."""
        levels = quick_calculate(
            entry=100.0,
            direction="SHORT",
            atr=2.5
        )

        assert levels.is_valid
        assert levels.stop_loss > levels.entry
        assert levels.take_profit_1 < levels.entry

    def test_quick_calculate_custom_multipliers(self):
        """Test quick_calculate with custom multipliers."""
        levels = quick_calculate(
            entry=100.0,
            direction="LONG",
            atr=2.0,
            sl_mult=1.0,
            tp1_mult=1.5,
            tp2_mult=2.5
        )

        # Should respect custom multipliers
        assert levels.is_valid
        expected_sl = 100.0 - (2.0 * 1.0075)  # ATR * buffer
        assert abs(levels.stop_loss - expected_sl) < 0.1


# Test fixtures
@pytest.fixture
def default_calculator():
    """Fixture providing a default calculator instance."""
    return TPSLCalculator()


@pytest.fixture
def strict_calculator():
    """Fixture providing a strict calculator instance."""
    return TPSLCalculator(
        min_risk_reward=2.5,
        max_sl_percent=2.0,
        min_sl_percent=0.5
    )


# Integration tests
class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_trade_workflow(self, default_calculator):
        """Test complete workflow from calculation to position sizing."""
        # Calculate levels
        levels = default_calculator.calculate(
            entry=45000.0,
            direction="LONG",
            atr=500.0,
            sl_multiplier=1.5,
            tp1_multiplier=2.0,
            tp2_multiplier=3.5
        )

        assert levels.is_valid

        # Calculate position size
        sizing = default_calculator.calculate_position_size(
            account_balance=10000.0,
            risk_percent=2.0,
            entry=levels.entry,
            stop_loss=levels.stop_loss,
            leverage=10.0
        )

        assert sizing['risk_amount_usdt'] == 200.0
        assert sizing['margin_required'] < sizing['position_size_usdt']

    def test_multiple_methods_comparison(self):
        """Test comparing results from different calculation methods."""
        calc = TPSLCalculator(min_risk_reward=1.5)

        # ATR method
        atr_levels = calc.calculate(
            entry=100.0,
            direction="LONG",
            atr=2.0,
            sl_multiplier=1.5,
            tp1_multiplier=2.0,
            tp2_multiplier=3.0,
            method=CalculationMethod.ATR
        )

        # Structure method
        structure_levels = calc.calculate(
            entry=100.0,
            direction="LONG",
            swing_low=95.0,
            tp1_multiplier=2.0,
            tp2_multiplier=3.0,
            method=CalculationMethod.STRUCTURE
        )

        # Both should be valid but may have different SL/TP values
        assert atr_levels.is_valid
        assert structure_levels.is_valid
        # Structure-based should have SL below swing low
        assert structure_levels.stop_loss < 95.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
