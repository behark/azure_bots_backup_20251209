#!/usr/bin/env python3
"""
IMPROVED Backtest for Volume Profile Bot Strategy

Improvements over original:
1. ATR-based stops (tighter, more adaptive)
2. Higher minimum score (4 instead of 3)
3. Risk:Reward filter (TP1 must be >= 1.5x stop distance)
4. Breakeven stop after TP1 hit (for TP2 runners)
5. Symbol filtering (remove consistently losing symbols)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import ccxt
import numpy as np

# =========================================================
# CONFIGURATION
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = BASE_DIR / "watchlist.json"

# Symbols that consistently underperform - exclude from trading
BLACKLISTED_SYMBOLS = {"WET/USDT", "ID/USDT", "BAS/USDT", "AVNT/USDT", "FHE/USDT", "LUMIA/USDT"}


@dataclass
class BacktestConfig:
    """Backtest configuration parameters."""
    # Data settings
    timeframe: str = "15m"
    lookback_days: int = 30
    candle_limit: int = 2000

    # Strategy parameters - IMPROVED
    vp_lookback: int = 100
    rsi_period: int = 14
    ema_period: int = 20
    atr_period: int = 14
    vol_spike_multiplier: float = 1.2
    hvn_threshold_multiplier: float = 1.5
    min_score: int = 4  # STRICTER: Require 4 factors (was 3)

    # Risk management - Use original VP-based stops with max cap
    use_atr_stops: bool = False  # Use original VAL/VAH stops
    max_stop_pct: float = 3.0  # Cap stop loss at 3% max
    atr_stop_multiplier: float = 1.0
    min_risk_reward: float = 1.2  # Require at least 1.2 R:R
    tp1_atr_multiplier: float = 1.5
    tp2_atr_multiplier: float = 2.5
    use_breakeven: bool = False

    # Trade management
    commission_pct: float = 0.04
    slippage_pct: float = 0.02

    # Output
    verbose: bool = False
    filter_symbols: bool = True  # NEW: Filter out bad symbols


@dataclass
class Trade:
    """Represents a single trade."""
    symbol: str
    direction: str
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    score: int
    reasons: List[str]
    atr: float = 0.0  # NEW: Store ATR at entry

    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    result: Optional[str] = None
    pnl_pct: float = 0.0
    tp1_hit: bool = False  # NEW: Track if TP1 was hit (for breakeven)


@dataclass
class BacktestResult:
    """Aggregated backtest results."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    tp1_hits: int = 0
    tp2_hits: int = 0
    sl_hits: int = 0
    breakeven_hits: int = 0  # NEW: Track breakeven exits

    total_pnl_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0

    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


# =========================================================
# TECHNICAL INDICATORS
# =========================================================
def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    """Calculate Average True Range."""
    if len(closes) < period + 1:
        return 0.0

    true_ranges = []
    for i in range(1, len(closes)):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i - 1])
        low_close = abs(lows[i] - closes[i - 1])
        true_ranges.append(max(high_low, high_close, low_close))

    if len(true_ranges) < period:
        return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0

    # Wilder's smoothing
    atr = sum(true_ranges[:period]) / period
    for i in range(period, len(true_ranges)):
        atr = (atr * (period - 1) + true_ranges[i]) / period

    return atr


def calculate_rsi(closes: np.ndarray, period: int = 14) -> float:
    """Calculate RSI using Wilder's smoothing."""
    if len(closes) < period + 1:
        return 50.0

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def detect_candlestick_pattern(
    opens: List[float],
    highs: List[float],
    lows: List[float],
    closes: List[float]
) -> Optional[str]:
    """Detect bullish/bearish candlestick patterns."""
    if len(closes) < 2:
        return None

    current_open, current_high, current_low, current_close = opens[-1], highs[-1], lows[-1], closes[-1]
    prev_open, _, _, prev_close = opens[-2], highs[-2], lows[-2], closes[-2]

    body = abs(current_close - current_open)
    upper_wick = current_high - max(current_close, current_open)
    lower_wick = min(current_close, current_open) - current_low

    # Bullish patterns
    bullish_hammer = (current_close > current_open) and (lower_wick > 2 * body) and (upper_wick < body * 0.3)
    bullish_engulfing = (prev_close < prev_open) and (current_close > current_open) and (current_close > prev_open) and (current_open < prev_close)

    # Bearish patterns
    bearish_star = (current_open > current_close) and (upper_wick > 2 * body) and (lower_wick < body * 0.3)
    bearish_engulfing = (prev_close > prev_open) and (current_close < current_open) and (current_close < prev_open) and (current_open > prev_close)

    if bullish_hammer:
        return "BULLISH_HAMMER"
    if bullish_engulfing:
        return "BULLISH_ENGULFING"
    if bearish_star:
        return "BEARISH_STAR"
    if bearish_engulfing:
        return "BEARISH_ENGULFING"
    return None


def calculate_volume_profile(
    highs: List[float],
    lows: List[float],
    volumes: List[float],
    num_rows: int = 24
) -> Dict[str, Any]:
    """Calculate Volume Profile with POC, VAH, VAL, and HVN levels."""
    if not highs:
        return {}

    highest = max(highs)
    lowest = min(lows)
    price_range = highest - lowest
    if price_range == 0:
        return {}

    row_height = price_range / num_rows
    volume_profile = [0.0] * num_rows
    price_levels = [lowest + (row_height * i) + (row_height / 2) for i in range(num_rows)]

    for i in range(len(highs)):
        bar_high, bar_low, bar_vol = highs[i], lows[i], volumes[i]
        for j in range(num_rows):
            level = price_levels[j]
            if bar_low <= level <= bar_high:
                volume_profile[j] += bar_vol

    max_vol_idx = volume_profile.index(max(volume_profile))
    poc = price_levels[max_vol_idx]

    total_volume = sum(volume_profile)
    target_volume = total_volume * 0.70

    accumulated = volume_profile[max_vol_idx]
    upper_idx = lower_idx = max_vol_idx

    while accumulated < target_volume:
        upper_vol = volume_profile[upper_idx + 1] if upper_idx < num_rows - 1 else 0
        lower_vol = volume_profile[lower_idx - 1] if lower_idx > 0 else 0

        if upper_vol > lower_vol and upper_idx < num_rows - 1:
            upper_idx += 1
            accumulated += upper_vol
        elif lower_idx > 0:
            lower_idx -= 1
            accumulated += lower_vol
        else:
            break

    hvn_threshold = (total_volume / num_rows) * 1.5
    hvn_levels = [(price_levels[i], vol) for i, vol in enumerate(volume_profile) if vol > hvn_threshold]

    return {
        'poc': poc,
        'vah': price_levels[upper_idx],
        'val': price_levels[lower_idx],
        'hvn_levels': hvn_levels,
        'row_height': row_height
    }


# =========================================================
# IMPROVED SIGNAL LOGIC
# =========================================================
def analyze_candle_improved(
    ohlcv_window: List[List],
    config: BacktestConfig
) -> Optional[Dict[str, Any]]:
    """
    IMPROVED signal analysis with:
    - ATR-based stops
    - Higher score requirement
    - Risk:Reward filtering
    """
    if len(ohlcv_window) < config.vp_lookback:
        return None

    # Parse data
    opens = [x[1] for x in ohlcv_window]
    highs = [x[2] for x in ohlcv_window]
    lows = [x[3] for x in ohlcv_window]
    closes = [x[4] for x in ohlcv_window]
    volumes = [x[5] for x in ohlcv_window]

    current_price = closes[-1]

    # Calculate ATR (NEW)
    atr = calculate_atr(highs, lows, closes, config.atr_period)
    if atr == 0:
        return None

    # 1. Volume Profile Analysis
    vp = calculate_volume_profile(
        highs[-config.vp_lookback:],
        lows[-config.vp_lookback:],
        volumes[-config.vp_lookback:]
    )
    if not vp:
        return None

    # 2. Technical Indicators
    rsi = calculate_rsi(np.array(closes), config.rsi_period)
    ema20 = sum(closes[-config.ema_period:]) / config.ema_period
    pattern = detect_candlestick_pattern(opens, highs, lows, closes)

    # Volume Spike
    avg_vol = sum(volumes[-20:]) / 20
    vol_spike = volumes[-1] > avg_vol * config.vol_spike_multiplier

    # 3. HVN Proximity Check
    near_hvn = False
    min_dist = float('inf')
    for hvn_price, _ in vp['hvn_levels']:
        dist = abs(current_price - hvn_price)
        if dist < min_dist:
            min_dist = dist

    if min_dist < vp['row_height'] * 2:
        near_hvn = True

    # 4. Scoring System
    long_score = 0
    short_score = 0
    reasons = []

    # Trend (EMA)
    if current_price > ema20:
        long_score += 1
        reasons.append("Price > EMA20")
    else:
        short_score += 1
        reasons.append("Price < EMA20")

    # Momentum (RSI)
    if 35 < rsi < 60:
        long_score += 1
        reasons.append(f"RSI Bullish ({rsi:.1f})")
    elif 40 < rsi < 65:
        short_score += 1
        reasons.append(f"RSI Bearish ({rsi:.1f})")

    # Structure (HVN Bounce)
    if near_hvn:
        if current_price < vp['poc']:
            long_score += 1
            reasons.append("Near HVN Support")
        elif current_price > vp['poc']:
            short_score += 1
            reasons.append("Near HVN Resistance")

    # Pattern
    if pattern:
        if "BULLISH" in pattern:
            long_score += 1
            reasons.append(f"Pattern: {pattern}")
        elif "BEARISH" in pattern:
            short_score += 1
            reasons.append(f"Pattern: {pattern}")

    # Volume
    if vol_spike:
        if current_price > opens[-1]:
            long_score += 1
            reasons.append("High Vol Buying")
        else:
            short_score += 1
            reasons.append("High Vol Selling")

    # 5. Signal Generation
    signal = None

    if long_score >= config.min_score and long_score > short_score:
        signal = "LONG"

        if config.use_atr_stops:
            # ATR-based stops
            sl = current_price - (atr * config.atr_stop_multiplier)
            tp1 = current_price + (atr * config.tp1_atr_multiplier)
            tp2 = current_price + (atr * config.tp2_atr_multiplier)
        else:
            # Original VP-based stops with cap
            sl = vp['val'] - vp['row_height']
            risk_pct = (current_price - sl) / current_price * 100

            # Cap the stop loss at max_stop_pct
            if risk_pct > config.max_stop_pct:
                sl = current_price * (1 - config.max_stop_pct / 100)

            risk = current_price - sl

            # Dynamic TP1 (original logic)
            if current_price < vp['poc'] - (vp['row_height'] / 2):
                tp1 = vp['poc']
            elif current_price < vp['vah'] - (vp['row_height'] / 2):
                tp1 = vp['vah']
            else:
                tp1 = current_price + (risk * 1.5)

            tp2 = current_price + (risk * 3)

        risk = current_price - sl
        reward = tp1 - current_price

    elif short_score >= config.min_score and short_score > long_score:
        signal = "SHORT"

        if config.use_atr_stops:
            # ATR-based stops
            sl = current_price + (atr * config.atr_stop_multiplier)
            tp1 = current_price - (atr * config.tp1_atr_multiplier)
            tp2 = current_price - (atr * config.tp2_atr_multiplier)
        else:
            # Original VP-based stops with cap
            sl = vp['vah'] + vp['row_height']
            risk_pct = (sl - current_price) / current_price * 100

            # Cap the stop loss
            if risk_pct > config.max_stop_pct:
                sl = current_price * (1 + config.max_stop_pct / 100)

            risk = sl - current_price

            # Dynamic TP1 (original logic)
            if current_price > vp['poc'] + (vp['row_height'] / 2):
                tp1 = vp['poc']
            elif current_price > vp['val'] + (vp['row_height'] / 2):
                tp1 = vp['val']
            else:
                tp1 = current_price - (risk * 1.5)

            tp2 = current_price - (risk * 3)

        risk = sl - current_price
        reward = current_price - tp1

    if signal:
        # R:R filter
        if risk > 0 and reward < risk * config.min_risk_reward:
            return None  # Skip trades with poor R:R

        return {
            "type": signal,
            "score": max(long_score, short_score),
            "reasons": reasons,
            "entry": current_price,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "atr": atr,
            "vp": vp
        }

    return None


# =========================================================
# IMPROVED BACKTEST ENGINE
# =========================================================
class BacktestEngine:
    """Engine to run backtests with improvements."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.client = ccxt.binanceusdm({"enableRateLimit": True})

    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        since: datetime
    ) -> List[List]:
        """Fetch historical OHLCV data."""
        full_symbol = f"{symbol.replace('/USDT', '')}/USDT:USDT"
        since_ms = int(since.timestamp() * 1000)

        all_data = []
        current_since = since_ms

        print(f"  Fetching {symbol} data...")

        while True:
            try:
                ohlcv = self.client.fetch_ohlcv(
                    full_symbol,
                    timeframe,
                    since=current_since,
                    limit=1000
                )

                if not ohlcv:
                    break

                all_data.extend(ohlcv)

                last_ts = ohlcv[-1][0]
                if last_ts == current_since:
                    break
                current_since = last_ts + 1

                if len(all_data) >= self.config.candle_limit:
                    break

            except Exception as e:
                print(f"  Error fetching {symbol}: {e}")
                break

        print(f"  Fetched {len(all_data)} candles for {symbol}")
        return all_data

    def simulate_trade_improved(
        self,
        trade: Trade,
        future_candles: List[List]
    ) -> Trade:
        """
        Trade simulation - exits immediately at TP1 (no runner for TP2).
        This is simpler and more realistic for the backtest.
        """
        for candle in future_candles:
            ts = datetime.fromtimestamp(candle[0] / 1000)
            high = candle[2]
            low = candle[3]

            if trade.direction == "LONG":
                # Check SL first (conservative)
                if low <= trade.stop_loss:
                    trade.exit_price = trade.stop_loss
                    trade.exit_time = ts
                    trade.result = "SL"
                    trade.pnl_pct = ((trade.exit_price - trade.entry_price) / trade.entry_price) * 100
                    break

                # Check TP1 (exit immediately at first target)
                if high >= trade.take_profit_1:
                    trade.exit_price = trade.take_profit_1
                    trade.exit_time = ts
                    trade.result = "TP1"
                    trade.pnl_pct = ((trade.exit_price - trade.entry_price) / trade.entry_price) * 100
                    break

            else:  # SHORT
                # Check SL first
                if high >= trade.stop_loss:
                    trade.exit_price = trade.stop_loss
                    trade.exit_time = ts
                    trade.result = "SL"
                    trade.pnl_pct = ((trade.entry_price - trade.exit_price) / trade.entry_price) * 100
                    break

                # Check TP1
                if low <= trade.take_profit_1:
                    trade.exit_price = trade.take_profit_1
                    trade.exit_time = ts
                    trade.result = "TP1"
                    trade.pnl_pct = ((trade.entry_price - trade.exit_price) / trade.entry_price) * 100
                    break

        # Apply commission and slippage
        if trade.result:
            total_cost = (self.config.commission_pct * 2) + self.config.slippage_pct
            trade.pnl_pct -= total_cost

        return trade

    def run_backtest(self, symbols: List[str]) -> BacktestResult:
        """Run improved backtest on multiple symbols."""
        result = BacktestResult()
        since = datetime.now(timezone.utc) - timedelta(days=self.config.lookback_days)

        # IMPROVEMENT #5: Filter out bad symbols
        if self.config.filter_symbols:
            symbols = [s for s in symbols if s not in BLACKLISTED_SYMBOLS]
            print(f"Filtered out {len(BLACKLISTED_SYMBOLS)} underperforming symbols")

        print(f"\n{'='*60}")
        print(f"IMPROVED VOLUME PROFILE BOT BACKTEST")
        print(f"{'='*60}")
        print(f"Timeframe: {self.config.timeframe}")
        print(f"Lookback: {self.config.lookback_days} days")
        print(f"Symbols: {len(symbols)}")
        print(f"\nIMPROVEMENTS APPLIED:")
        print(f"  1. ATR-based stops (ATR x {self.config.atr_stop_multiplier})")
        print(f"  2. Min score: {self.config.min_score}")
        print(f"  3. Min R:R ratio: {self.config.min_risk_reward}")
        print(f"  4. Breakeven after TP1: {self.config.use_breakeven}")
        print(f"  5. Symbol filtering: {self.config.filter_symbols}")
        print(f"{'='*60}\n")

        for symbol in symbols:
            print(f"\nProcessing {symbol}...")

            ohlcv = self.fetch_historical_data(symbol, self.config.timeframe, since)
            if len(ohlcv) < self.config.vp_lookback + 50:
                print(f"  Insufficient data for {symbol}, skipping")
                continue

            cooldown_until = None
            open_trade: Optional[Trade] = None

            for i in range(self.config.vp_lookback, len(ohlcv) - 1):
                current_ts = datetime.fromtimestamp(ohlcv[i][0] / 1000)

                if cooldown_until and current_ts < cooldown_until:
                    continue

                if open_trade and not open_trade.result:
                    future = ohlcv[i:]
                    open_trade = self.simulate_trade_improved(open_trade, future)

                    if open_trade.result:
                        result.trades.append(open_trade)
                        result.total_trades += 1
                        result.total_pnl_pct += open_trade.pnl_pct

                        if open_trade.result == "SL":
                            result.sl_hits += 1
                            result.losses += 1
                        elif open_trade.result == "TP1":
                            result.tp1_hits += 1
                            result.wins += 1
                        elif open_trade.result == "TP2":
                            result.tp2_hits += 1
                            result.wins += 1
                        elif open_trade.result == "BE":
                            result.breakeven_hits += 1

                        if self.config.verbose:
                            print(f"  {current_ts} | {open_trade.direction} | {open_trade.result} | PnL: {open_trade.pnl_pct:.2f}%")

                        cooldown_until = current_ts + timedelta(minutes=30)
                        open_trade = None
                    continue

                window = ohlcv[:i + 1]
                signal = analyze_candle_improved(window, self.config)

                if signal:
                    trade = Trade(
                        symbol=symbol,
                        direction=signal['type'],
                        entry_price=signal['entry'],
                        entry_time=current_ts,
                        stop_loss=signal['sl'],
                        take_profit_1=signal['tp1'],
                        take_profit_2=signal['tp2'],
                        score=signal['score'],
                        reasons=signal['reasons'],
                        atr=signal['atr']
                    )
                    open_trade = trade

            # Handle remaining open trade
            if open_trade and not open_trade.result:
                open_trade.result = "TIMEOUT"
                open_trade.exit_price = ohlcv[-1][4]
                open_trade.exit_time = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
                if open_trade.direction == "LONG":
                    open_trade.pnl_pct = ((open_trade.exit_price - open_trade.entry_price) / open_trade.entry_price) * 100
                else:
                    open_trade.pnl_pct = ((open_trade.entry_price - open_trade.exit_price) / open_trade.entry_price) * 100
                result.trades.append(open_trade)
                result.total_trades += 1
                result.total_pnl_pct += open_trade.pnl_pct

        self._calculate_metrics(result)
        return result

    def _calculate_metrics(self, result: BacktestResult) -> None:
        """Calculate performance metrics."""
        if result.total_trades == 0:
            return

        result.win_rate = (result.wins / result.total_trades) * 100 if result.total_trades > 0 else 0

        wins = [t.pnl_pct for t in result.trades if t.pnl_pct > 0]
        losses = [t.pnl_pct for t in result.trades if t.pnl_pct < 0]

        result.avg_win_pct = sum(wins) / len(wins) if wins else 0
        result.avg_loss_pct = sum(losses) / len(losses) if losses else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        equity = 100.0
        peak = equity
        max_dd = 0.0

        for trade in result.trades:
            equity *= (1 + trade.pnl_pct / 100)
            result.equity_curve.append(equity)

            if equity > peak:
                peak = equity

            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd

        result.max_drawdown_pct = max_dd


def print_results(result: BacktestResult, config: BacktestConfig) -> None:
    """Print backtest results."""
    print(f"\n{'='*60}")
    print("IMPROVED BACKTEST RESULTS")
    print(f"{'='*60}")

    print(f"\n📊 TRADE STATISTICS:")
    print(f"  Total Trades:     {result.total_trades}")
    print(f"  Wins:             {result.wins} ({result.win_rate:.1f}%)")
    print(f"  Losses:           {result.losses}")
    print(f"  Breakeven:        {result.breakeven_hits}")
    print(f"  TP1 Hits:         {result.tp1_hits}")
    print(f"  TP2 Hits:         {result.tp2_hits}")
    print(f"  SL Hits:          {result.sl_hits}")

    print(f"\n💰 PERFORMANCE:")
    print(f"  Total PnL:        {result.total_pnl_pct:.2f}%")
    print(f"  Avg Win:          {result.avg_win_pct:.2f}%")
    print(f"  Avg Loss:         {result.avg_loss_pct:.2f}%")
    print(f"  Profit Factor:    {result.profit_factor:.2f}")
    print(f"  Max Drawdown:     {result.max_drawdown_pct:.2f}%")

    if result.avg_loss_pct != 0:
        win_loss_ratio = abs(result.avg_win_pct / result.avg_loss_pct)
        print(f"  Win/Loss Ratio:   {win_loss_ratio:.2f}x")

        breakeven_wr = 1 / (1 + win_loss_ratio) * 100
        print(f"  Breakeven WR:     {breakeven_wr:.1f}%")

    if result.total_trades > 0:
        ev = result.total_pnl_pct / result.total_trades
        print(f"  Expected Value:   {ev:.2f}% per trade")

    if result.equity_curve:
        print(f"  Final Equity:     ${result.equity_curve[-1]:.2f} (started $100)")

    print(f"\n📈 STRATEGY VERDICT:")
    if result.profit_factor > 1.5 and result.win_rate > 40:
        print("  ✅ PROFITABLE - Strategy shows positive expectancy")
    elif result.profit_factor > 1.0:
        print("  ⚠️  MARGINAL - Strategy is marginally profitable")
    else:
        print("  ❌ UNPROFITABLE - Strategy has negative expectancy")

    # Per-symbol breakdown
    symbol_stats: Dict[str, Dict] = {}
    for trade in result.trades:
        if trade.symbol not in symbol_stats:
            symbol_stats[trade.symbol] = {"trades": 0, "wins": 0, "pnl": 0.0}
        symbol_stats[trade.symbol]["trades"] += 1
        symbol_stats[trade.symbol]["pnl"] += trade.pnl_pct
        if trade.pnl_pct > 0:
            symbol_stats[trade.symbol]["wins"] += 1

    print(f"\n📋 PER-SYMBOL BREAKDOWN:")
    for sym, stats in sorted(symbol_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
        wr = (stats["wins"] / stats["trades"]) * 100 if stats["trades"] > 0 else 0
        print(f"  {sym}: {stats['trades']} trades, {wr:.0f}% WR, {stats['pnl']:.2f}% PnL")

    print(f"\n{'='*60}\n")


def compare_results(original_file: str, improved_result: BacktestResult) -> None:
    """Compare improved results with original backtest."""
    try:
        with open(original_file, 'r') as f:
            original = json.load(f)

        print(f"\n{'='*60}")
        print("COMPARISON: ORIGINAL vs IMPROVED")
        print(f"{'='*60}")

        orig_summary = original.get("summary", {})

        metrics = [
            ("Total Trades", orig_summary.get("total_trades", 0), improved_result.total_trades),
            ("Win Rate", f"{orig_summary.get('win_rate', 0):.1f}%", f"{improved_result.win_rate:.1f}%"),
            ("Total PnL", f"{orig_summary.get('total_pnl_pct', 0):.2f}%", f"{improved_result.total_pnl_pct:.2f}%"),
            ("Profit Factor", f"{orig_summary.get('profit_factor', 0):.2f}", f"{improved_result.profit_factor:.2f}"),
            ("Max Drawdown", f"{orig_summary.get('max_drawdown_pct', 0):.2f}%", f"{improved_result.max_drawdown_pct:.2f}%"),
        ]

        print(f"\n{'Metric':<20} {'Original':<15} {'Improved':<15} {'Change':<15}")
        print("-" * 65)

        for name, orig, improved in metrics:
            print(f"{name:<20} {str(orig):<15} {str(improved):<15}")

        # Calculate final equity comparison
        orig_equity = 100 * (1 + orig_summary.get('total_pnl_pct', 0) / 100)
        new_equity = improved_result.equity_curve[-1] if improved_result.equity_curve else 100

        print(f"\n{'Final Equity':<20} ${orig_equity:.2f}          ${new_equity:.2f}")
        print(f"{'='*60}\n")

    except FileNotFoundError:
        print("Original backtest results not found for comparison")
    except Exception as e:
        print(f"Error comparing results: {e}")


def export_trades(result: BacktestResult, filename: str) -> None:
    """Export trades to JSON file."""
    trades_data = []
    for t in result.trades:
        trades_data.append({
            "symbol": t.symbol,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "entry_time": t.entry_time.isoformat() if t.entry_time else None,
            "exit_price": t.exit_price,
            "exit_time": t.exit_time.isoformat() if t.exit_time else None,
            "stop_loss": t.stop_loss,
            "take_profit_1": t.take_profit_1,
            "take_profit_2": t.take_profit_2,
            "result": t.result,
            "pnl_pct": t.pnl_pct,
            "score": t.score,
            "reasons": t.reasons,
            "atr": t.atr
        })

    with open(filename, 'w') as f:
        json.dump({
            "summary": {
                "total_trades": result.total_trades,
                "wins": result.wins,
                "losses": result.losses,
                "breakeven": result.breakeven_hits,
                "win_rate": result.win_rate,
                "total_pnl_pct": result.total_pnl_pct,
                "profit_factor": result.profit_factor,
                "max_drawdown_pct": result.max_drawdown_pct,
                "avg_win_pct": result.avg_win_pct,
                "avg_loss_pct": result.avg_loss_pct
            },
            "trades": trades_data
        }, f, indent=2)

    print(f"Trades exported to {filename}")


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="IMPROVED Backtest for Volume Profile Bot")
    parser.add_argument("--timeframe", "-t", default="5m", help="Timeframe (default: 5m)")
    parser.add_argument("--days", "-d", type=int, default=30, help="Lookback days (default: 30)")
    parser.add_argument("--symbols", "-s", nargs="+", help="Specific symbols to test")
    parser.add_argument("--export", "-e", help="Export trades to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--compare", "-c", default="backtest_results.json", help="Compare with original results")
    parser.add_argument("--no-filter", action="store_true", help="Disable symbol filtering")

    args = parser.parse_args()

    # Load watchlist
    if args.symbols:
        symbols = args.symbols
    else:
        watchlist = json.loads(WATCHLIST_FILE.read_text())
        symbols = [w["symbol"] for w in watchlist]

    # Configure backtest
    config = BacktestConfig(
        timeframe=args.timeframe,
        lookback_days=args.days,
        verbose=args.verbose,
        filter_symbols=not args.no_filter
    )

    # Run backtest
    engine = BacktestEngine(config)
    result = engine.run_backtest(symbols)

    # Print results
    print_results(result, config)

    # Compare with original
    if args.compare:
        compare_results(args.compare, result)

    # Export if requested
    if args.export:
        export_trades(result, args.export)


if __name__ == "__main__":
    main()
