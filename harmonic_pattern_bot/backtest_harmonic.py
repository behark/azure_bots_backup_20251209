#!/usr/bin/env python3
"""
Backtest for Harmonic Pattern Bot

Tests the non-repainting harmonic pattern strategy on historical data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ccxt
import numpy as np

# =========================================================
# CONFIGURATION
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = BASE_DIR / "harmonic_watchlist.json"


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    timeframe: str = "1h"
    lookback_days: int = 90
    candle_limit: int = 2000

    # ZigZag settings
    zigzag_depth: int = 12
    zigzag_deviation: float = 5.0

    # Pattern settings
    fib_tolerance: float = 0.05
    min_pattern_score: float = 70.0

    # Risk management
    min_risk_reward: float = 1.5
    tp1_rr: float = 1.5
    tp2_rr: float = 2.5
    max_stop_pct: float = 3.0

    # Trade costs
    commission_pct: float = 0.04
    slippage_pct: float = 0.02

    verbose: bool = False


@dataclass
class SwingPoint:
    """Represents a confirmed swing high or low."""
    index: int
    price: float
    timestamp: datetime
    swing_type: str  # "high" or "low"


@dataclass
class Trade:
    """Represents a single trade."""
    symbol: str
    pattern: str
    direction: str
    entry_price: float
    entry_time: datetime
    entry_index: int
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    score: float

    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    result: Optional[str] = None
    pnl_pct: float = 0.0


@dataclass
class BacktestResult:
    """Aggregated backtest results."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    tp1_hits: int = 0
    tp2_hits: int = 0
    sl_hits: int = 0

    total_pnl_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0

    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    pattern_stats: Dict[str, Dict] = field(default_factory=dict)


# =========================================================
# PATTERN RATIOS
# =========================================================
PATTERN_RATIOS = {
    "gartley": {
        "xab": (0.618, 0.618),
        "abc": (0.382, 0.886),
        "bcd": (1.272, 1.618),
        "xad": (0.786, 0.786),
    },
    "bat": {
        "xab": (0.382, 0.500),
        "abc": (0.382, 0.886),
        "bcd": (1.618, 2.618),
        "xad": (0.886, 0.886),
    },
    "butterfly": {
        "xab": (0.786, 0.786),
        "abc": (0.382, 0.886),
        "bcd": (1.618, 2.618),
        "xad": (1.272, 1.618),
    },
    "crab": {
        "xab": (0.382, 0.618),
        "abc": (0.382, 0.886),
        "bcd": (2.240, 3.618),
        "xad": (1.618, 1.618),
    },
    "shark": {
        "xab": (0.446, 0.618),
        "abc": (1.130, 1.618),
        "bcd": (1.618, 2.240),
        "xad": (0.886, 1.130),
    },
    "abcd": {
        "abc": (0.382, 0.886),
        "bcd": (1.130, 2.618),
    },
}


# =========================================================
# SWING DETECTION (Non-Repainting)
# =========================================================
def find_swing_points(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    timestamps: List[datetime],
    depth: int = 12,
    deviation_pct: float = 5.0
) -> List[SwingPoint]:
    """Find confirmed swing points (non-repainting)."""
    swings = []
    n = len(highs)

    if n < depth * 2:
        return swings

    # Find swing highs
    for i in range(depth, n - depth):
        is_swing_high = True
        for j in range(i - depth, i + depth + 1):
            if j != i and highs[j] >= highs[i]:
                is_swing_high = False
                break

        if is_swing_high:
            left_low = min(lows[i-depth:i])
            right_low = min(lows[i+1:i+depth+1])
            deviation = (highs[i] - min(left_low, right_low)) / highs[i] * 100

            if deviation >= deviation_pct:
                swings.append(SwingPoint(
                    index=i,
                    price=highs[i],
                    timestamp=timestamps[i],
                    swing_type="high"
                ))

    # Find swing lows
    for i in range(depth, n - depth):
        is_swing_low = True
        for j in range(i - depth, i + depth + 1):
            if j != i and lows[j] <= lows[i]:
                is_swing_low = False
                break

        if is_swing_low:
            left_high = max(highs[i-depth:i])
            right_high = max(highs[i+1:i+depth+1])
            deviation = (max(left_high, right_high) - lows[i]) / lows[i] * 100

            if deviation >= deviation_pct:
                swings.append(SwingPoint(
                    index=i,
                    price=lows[i],
                    timestamp=timestamps[i],
                    swing_type="low"
                ))

    # Sort by index
    swings.sort(key=lambda x: x.index)

    # Keep only alternating high-low sequence
    filtered = []
    for swing in swings:
        if not filtered:
            filtered.append(swing)
        elif swing.swing_type != filtered[-1].swing_type:
            filtered.append(swing)
        elif swing.swing_type == "high" and swing.price > filtered[-1].price:
            filtered[-1] = swing
        elif swing.swing_type == "low" and swing.price < filtered[-1].price:
            filtered[-1] = swing

    return filtered


# =========================================================
# PATTERN DETECTION
# =========================================================
def calculate_ratio(p1: float, p2: float, p3: float) -> float:
    """Calculate Fibonacci ratio between three points."""
    if abs(p1 - p2) == 0:
        return 0
    return abs(p2 - p3) / abs(p1 - p2)


def score_ratio(actual: float, ideal_range: Tuple[float, float], tolerance: float = 0.05) -> float:
    """Score how close a ratio is to the ideal range."""
    ideal_min, ideal_max = ideal_range
    ideal_mid = (ideal_min + ideal_max) / 2

    if ideal_min <= actual <= ideal_max:
        distance = abs(actual - ideal_mid) / (ideal_max - ideal_min + 0.001)
        return 100 - (distance * 20)

    extended_min = ideal_min * (1 - tolerance)
    extended_max = ideal_max * (1 + tolerance)

    if extended_min <= actual <= extended_max:
        if actual < ideal_min:
            distance = (ideal_min - actual) / (ideal_min * tolerance)
        else:
            distance = (actual - ideal_max) / (ideal_max * tolerance)
        return 80 - (distance * 30)

    return 0


def detect_pattern_at_index(
    swings: List[SwingPoint],
    swing_index: int,
    config: BacktestConfig
) -> Optional[Dict[str, Any]]:
    """Detect harmonic pattern ending at the given swing index."""
    if swing_index < 4:
        return None

    # Get 5 swings ending at swing_index
    pattern_swings = swings[swing_index-4:swing_index+1]
    if len(pattern_swings) < 5:
        return None

    x, a, b, c, d = pattern_swings

    # Determine direction
    if d.swing_type == "low":
        direction = "bullish"
        if not (x.swing_type == "high" and a.swing_type == "low" and
                b.swing_type == "high" and c.swing_type == "low"):
            return None
    else:
        direction = "bearish"
        if not (x.swing_type == "low" and a.swing_type == "high" and
                b.swing_type == "low" and c.swing_type == "high"):
            return None

    # Calculate ratios
    xab = calculate_ratio(x.price, a.price, b.price)
    abc = calculate_ratio(a.price, b.price, c.price)
    bcd = calculate_ratio(b.price, c.price, d.price)
    xad = calculate_ratio(x.price, a.price, d.price)

    ratios = {"xab": xab, "abc": abc, "bcd": bcd, "xad": xad}

    # Score each pattern
    best_pattern = None
    best_score = 0

    for pattern_name, ideal_ratios in PATTERN_RATIOS.items():
        scores = []
        for ratio_name, ideal_range in ideal_ratios.items():
            if ratio_name in ratios:
                score = score_ratio(ratios[ratio_name], ideal_range, config.fib_tolerance)
                scores.append(score)

        if scores and all(s > 0 for s in scores):
            avg_score = sum(scores) / len(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_pattern = pattern_name

    if best_pattern is None or best_score < config.min_pattern_score:
        return None

    # Calculate entry, SL, TPs
    cd_range = abs(c.price - d.price)
    entry = d.price

    if direction == "bullish":
        stop_loss = d.price - (cd_range * 0.236)
        risk = entry - stop_loss

        # Cap stop loss
        if risk / entry * 100 > config.max_stop_pct:
            stop_loss = entry * (1 - config.max_stop_pct / 100)
            risk = entry - stop_loss

        tp1 = entry + (risk * config.tp1_rr)
        tp2 = entry + (risk * config.tp2_rr)
    else:
        stop_loss = d.price + (cd_range * 0.236)
        risk = stop_loss - entry

        if risk / entry * 100 > config.max_stop_pct:
            stop_loss = entry * (1 + config.max_stop_pct / 100)
            risk = stop_loss - entry

        tp1 = entry - (risk * config.tp1_rr)
        tp2 = entry - (risk * config.tp2_rr)

    return {
        "pattern": best_pattern,
        "direction": direction,
        "score": best_score,
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit_1": tp1,
        "take_profit_2": tp2,
        "d_index": d.index,
        "ratios": ratios,
    }


# =========================================================
# BACKTEST ENGINE
# =========================================================
class BacktestEngine:
    """Engine to run backtests on historical data."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.client = ccxt.binanceusdm({"enableRateLimit": True})

    def fetch_historical_data(self, symbol: str, timeframe: str, since: datetime) -> List[List]:
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

    def simulate_trade(self, trade: Trade, ohlcv: List[List]) -> Trade:
        """Simulate a trade using future candles."""
        for i in range(trade.entry_index + 1, len(ohlcv)):
            candle = ohlcv[i]
            ts = datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc)
            high = candle[2]
            low = candle[3]

            if trade.direction == "bullish":
                # Check SL first
                if low <= trade.stop_loss:
                    trade.exit_price = trade.stop_loss
                    trade.exit_time = ts
                    trade.result = "SL"
                    trade.pnl_pct = ((trade.exit_price - trade.entry_price) / trade.entry_price) * 100
                    break

                # Check TP1
                if high >= trade.take_profit_1:
                    trade.exit_price = trade.take_profit_1
                    trade.exit_time = ts
                    trade.result = "TP1"
                    trade.pnl_pct = ((trade.exit_price - trade.entry_price) / trade.entry_price) * 100
                    break

            else:  # bearish
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

        # Apply costs
        if trade.result:
            total_cost = (self.config.commission_pct * 2) + self.config.slippage_pct
            trade.pnl_pct -= total_cost

        return trade

    def run_backtest(self, symbols: List[str]) -> BacktestResult:
        """Run backtest on multiple symbols."""
        result = BacktestResult()
        since = datetime.now(timezone.utc) - timedelta(days=self.config.lookback_days)

        print(f"\n{'='*60}")
        print("HARMONIC PATTERN BOT BACKTEST")
        print(f"{'='*60}")
        print(f"Timeframe: {self.config.timeframe}")
        print(f"Lookback: {self.config.lookback_days} days")
        print(f"Symbols: {len(symbols)}")
        print(f"Min Pattern Score: {self.config.min_pattern_score}")
        print(f"Min R:R: {self.config.min_risk_reward}")
        print(f"TP1: {self.config.tp1_rr}R | TP2: {self.config.tp2_rr}R")
        print(f"{'='*60}\n")

        for symbol in symbols:
            print(f"\nProcessing {symbol}...")

            ohlcv = self.fetch_historical_data(symbol, self.config.timeframe, since)
            if len(ohlcv) < 200:
                print(f"  Insufficient data for {symbol}, skipping")
                continue

            # Parse data
            timestamps = [datetime.fromtimestamp(x[0] / 1000, tz=timezone.utc) for x in ohlcv]
            highs = [x[2] for x in ohlcv]
            lows = [x[3] for x in ohlcv]
            closes = [x[4] for x in ohlcv]

            # Find all swing points
            swings = find_swing_points(
                highs, lows, closes, timestamps,
                depth=self.config.zigzag_depth,
                deviation_pct=self.config.zigzag_deviation
            )

            if len(swings) < 5:
                print(f"  Not enough swings for {symbol}")
                continue

            print(f"  Found {len(swings)} swing points")

            # Track last trade index to avoid overlapping trades
            last_trade_end_index = 0
            patterns_found = 0

            # Scan for patterns at each swing
            for swing_idx in range(4, len(swings)):
                d_swing = swings[swing_idx]

                # Skip if too close to last trade
                if d_swing.index < last_trade_end_index + self.config.zigzag_depth:
                    continue

                # Detect pattern
                signal = detect_pattern_at_index(swings, swing_idx, self.config)

                if signal:
                    patterns_found += 1

                    trade = Trade(
                        symbol=symbol,
                        pattern=signal["pattern"],
                        direction=signal["direction"],
                        entry_price=signal["entry"],
                        entry_time=d_swing.timestamp,
                        entry_index=d_swing.index,
                        stop_loss=signal["stop_loss"],
                        take_profit_1=signal["take_profit_1"],
                        take_profit_2=signal["take_profit_2"],
                        score=signal["score"],
                    )

                    # Simulate trade
                    trade = self.simulate_trade(trade, ohlcv)

                    if trade.result:
                        result.trades.append(trade)
                        result.total_trades += 1
                        result.total_pnl_pct += trade.pnl_pct

                        # Update pattern stats
                        if trade.pattern not in result.pattern_stats:
                            result.pattern_stats[trade.pattern] = {
                                "trades": 0, "wins": 0, "pnl": 0.0
                            }
                        result.pattern_stats[trade.pattern]["trades"] += 1
                        result.pattern_stats[trade.pattern]["pnl"] += trade.pnl_pct

                        if trade.result == "SL":
                            result.sl_hits += 1
                            result.losses += 1
                        elif trade.result == "TP1":
                            result.tp1_hits += 1
                            result.wins += 1
                            result.pattern_stats[trade.pattern]["wins"] += 1
                        elif trade.result == "TP2":
                            result.tp2_hits += 1
                            result.wins += 1
                            result.pattern_stats[trade.pattern]["wins"] += 1

                        if self.config.verbose:
                            print(f"    {trade.entry_time} | {trade.pattern.upper()} {trade.direction} | {trade.result} | {trade.pnl_pct:.2f}%")

                        # Update last trade end
                        if trade.exit_time:
                            # Find index of exit
                            for idx, ts in enumerate(timestamps):
                                if ts >= trade.exit_time:
                                    last_trade_end_index = idx
                                    break

            print(f"  Patterns found: {patterns_found}")

        # Calculate final metrics
        self._calculate_metrics(result)
        return result

    def _calculate_metrics(self, result: BacktestResult) -> None:
        """Calculate performance metrics."""
        if result.total_trades == 0:
            return

        result.win_rate = (result.wins / result.total_trades) * 100

        wins = [t.pnl_pct for t in result.trades if t.pnl_pct > 0]
        losses = [t.pnl_pct for t in result.trades if t.pnl_pct < 0]

        result.avg_win_pct = sum(wins) / len(wins) if wins else 0
        result.avg_loss_pct = sum(losses) / len(losses) if losses else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Equity curve
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
    print("BACKTEST RESULTS")
    print(f"{'='*60}")

    print(f"\n📊 TRADE STATISTICS:")
    print(f"  Total Trades:     {result.total_trades}")
    print(f"  Wins:             {result.wins} ({result.win_rate:.1f}%)")
    print(f"  Losses:           {result.losses}")
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
    if result.profit_factor > 1.5 and result.win_rate > 35:
        print("  ✅ PROFITABLE - Strategy shows positive expectancy")
    elif result.profit_factor > 1.0:
        print("  ⚠️  MARGINAL - Strategy is marginally profitable")
    else:
        print("  ❌ UNPROFITABLE - Strategy has negative expectancy")

    # Pattern breakdown
    if result.pattern_stats:
        print(f"\n📋 PATTERN BREAKDOWN:")
        for pattern, stats in sorted(result.pattern_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
            wr = (stats["wins"] / stats["trades"]) * 100 if stats["trades"] > 0 else 0
            print(f"  {pattern.upper()}: {stats['trades']} trades, {wr:.0f}% WR, {stats['pnl']:.2f}% PnL")

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


def export_results(result: BacktestResult, filename: str) -> None:
    """Export results to JSON."""
    trades_data = []
    for t in result.trades:
        trades_data.append({
            "symbol": t.symbol,
            "pattern": t.pattern,
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
        })

    with open(filename, 'w') as f:
        json.dump({
            "summary": {
                "total_trades": result.total_trades,
                "wins": result.wins,
                "losses": result.losses,
                "win_rate": result.win_rate,
                "total_pnl_pct": result.total_pnl_pct,
                "profit_factor": result.profit_factor,
                "max_drawdown_pct": result.max_drawdown_pct,
                "avg_win_pct": result.avg_win_pct,
                "avg_loss_pct": result.avg_loss_pct,
            },
            "pattern_stats": result.pattern_stats,
            "trades": trades_data
        }, f, indent=2)

    print(f"Results exported to {filename}")


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Backtest Harmonic Pattern Strategy")
    parser.add_argument("--timeframe", "-t", default="1h", help="Timeframe (default: 1h)")
    parser.add_argument("--days", "-d", type=int, default=90, help="Lookback days (default: 90)")
    parser.add_argument("--symbols", "-s", nargs="+", help="Specific symbols to test")
    parser.add_argument("--export", "-e", help="Export results to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--min-score", type=float, default=70.0, help="Min pattern score (default: 70)")
    parser.add_argument("--min-rr", type=float, default=1.5, help="Min risk:reward (default: 1.5)")

    args = parser.parse_args()

    # Load watchlist
    if args.symbols:
        symbols = args.symbols
    else:
        watchlist = json.loads(WATCHLIST_FILE.read_text())
        symbols = [w["symbol"] for w in watchlist]

    # Configure
    config = BacktestConfig(
        timeframe=args.timeframe,
        lookback_days=args.days,
        verbose=args.verbose,
        min_pattern_score=args.min_score,
        min_risk_reward=args.min_rr,
        tp1_rr=args.min_rr,
        tp2_rr=args.min_rr + 1.0,
    )

    # Run backtest
    engine = BacktestEngine(config)
    result = engine.run_backtest(symbols)

    # Print results
    print_results(result, config)

    # Export if requested
    if args.export:
        export_results(result, args.export)


if __name__ == "__main__":
    main()
