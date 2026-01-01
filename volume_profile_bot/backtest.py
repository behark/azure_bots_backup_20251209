#!/usr/bin/env python3
"""
Backtest for Volume Profile Bot Strategy

This script backtests the exact same logic from volume_profile_bot.py
to validate the strategy before live trading.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ccxt
import numpy as np

# =========================================================
# CONFIGURATION
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = BASE_DIR / "watchlist.json"


@dataclass
class BacktestConfig:
    """Backtest configuration parameters."""
    # Data settings
    timeframe: str = "15m"
    lookback_days: int = 30
    candle_limit: int = 2000  # Max candles to fetch

    # Strategy parameters (matching live bot)
    vp_lookback: int = 100  # Candles for volume profile
    rsi_period: int = 14
    ema_period: int = 20
    vol_spike_multiplier: float = 1.2
    hvn_threshold_multiplier: float = 1.5
    min_score: int = 3

    # Trade management
    commission_pct: float = 0.04  # 0.04% per trade (taker fee)
    slippage_pct: float = 0.02    # 0.02% slippage

    # Output
    verbose: bool = False


@dataclass
class Trade:
    """Represents a single trade."""
    symbol: str
    direction: str  # LONG or SHORT
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    score: int
    reasons: List[str]

    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    result: Optional[str] = None  # SL, TP1, TP2
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


# =========================================================
# TECHNICAL INDICATORS (Same as live bot)
# =========================================================
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

    if bullish_hammer: return "BULLISH_HAMMER"
    if bullish_engulfing: return "BULLISH_ENGULFING"
    if bearish_star: return "BEARISH_STAR"
    if bearish_engulfing: return "BEARISH_ENGULFING"
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
# SIGNAL LOGIC (Same as live bot)
# =========================================================
def analyze_candle(
    ohlcv_window: List[List],
    config: BacktestConfig
) -> Optional[Dict[str, Any]]:
    """
    Analyze a window of OHLCV data and generate signals.
    This mirrors the live bot's analyze_symbol() method.
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

    # 1. Volume Profile Analysis (Last 100 candles)
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
    else:
        short_score += 1

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
            reasons.append("Bounce off Volume Support")
        elif current_price > vp['poc']:
            short_score += 1
            reasons.append("Reject off Volume Resistance")

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
        sl = vp['val'] - vp['row_height']
        risk = current_price - sl
        if risk <= 0:
            risk = current_price * 0.005

        # Dynamic TP1
        if current_price < vp['poc'] - (vp['row_height'] / 2):
            tp1 = vp['poc']
        elif current_price < vp['vah'] - (vp['row_height'] / 2):
            tp1 = vp['vah']
        else:
            tp1 = current_price + (risk * 1.5)

        tp2 = current_price + (risk * 3)

    elif short_score >= config.min_score and short_score > long_score:
        signal = "SHORT"
        sl = vp['vah'] + vp['row_height']
        risk = sl - current_price
        if risk <= 0:
            risk = current_price * 0.005

        # Dynamic TP1
        if current_price > vp['poc'] + (vp['row_height'] / 2):
            tp1 = vp['poc']
        elif current_price > vp['val'] + (vp['row_height'] / 2):
            tp1 = vp['val']
        else:
            tp1 = current_price - (risk * 1.5)

        tp2 = current_price - (risk * 3)

    if signal:
        return {
            "type": signal,
            "score": max(long_score, short_score),
            "reasons": reasons,
            "entry": current_price,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "vp": vp
        }
    return None


# =========================================================
# BACKTEST ENGINE
# =========================================================
class BacktestEngine:
    """Engine to run backtests on historical data."""

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

                # Move to next batch
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

    def simulate_trade(
        self,
        trade: Trade,
        future_candles: List[List]
    ) -> Trade:
        """
        Simulate a trade using future candles to check for TP/SL hits.
        Returns the trade with exit information filled in.
        """
        for candle in future_candles:
            ts = datetime.fromtimestamp(candle[0] / 1000)
            high = candle[2]
            low = candle[3]

            if trade.direction == "LONG":
                # Check SL first (more conservative)
                if low <= trade.stop_loss:
                    trade.exit_price = trade.stop_loss
                    trade.exit_time = ts
                    trade.result = "SL"
                    trade.pnl_pct = ((trade.exit_price - trade.entry_price) / trade.entry_price) * 100
                    break
                # Check TP2 before TP1 (if price gaps through both)
                elif high >= trade.take_profit_2:
                    trade.exit_price = trade.take_profit_2
                    trade.exit_time = ts
                    trade.result = "TP2"
                    trade.pnl_pct = ((trade.exit_price - trade.entry_price) / trade.entry_price) * 100
                    break
                elif high >= trade.take_profit_1:
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
                # Check TP2 before TP1
                elif low <= trade.take_profit_2:
                    trade.exit_price = trade.take_profit_2
                    trade.exit_time = ts
                    trade.result = "TP2"
                    trade.pnl_pct = ((trade.entry_price - trade.exit_price) / trade.entry_price) * 100
                    break
                elif low <= trade.take_profit_1:
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
        """Run backtest on multiple symbols."""
        result = BacktestResult()
        since = datetime.utcnow() - timedelta(days=self.config.lookback_days)

        print(f"\n{'='*60}")
        print(f"VOLUME PROFILE BOT BACKTEST")
        print(f"{'='*60}")
        print(f"Timeframe: {self.config.timeframe}")
        print(f"Lookback: {self.config.lookback_days} days")
        print(f"Symbols: {len(symbols)}")
        print(f"{'='*60}\n")

        for symbol in symbols:
            print(f"\nProcessing {symbol}...")

            # Fetch data
            ohlcv = self.fetch_historical_data(symbol, self.config.timeframe, since)
            if len(ohlcv) < self.config.vp_lookback + 50:
                print(f"  Insufficient data for {symbol}, skipping")
                continue

            # Walk through candles
            cooldown_until = None
            open_trade: Optional[Trade] = None

            for i in range(self.config.vp_lookback, len(ohlcv) - 1):
                current_ts = datetime.fromtimestamp(ohlcv[i][0] / 1000)

                # Check cooldown
                if cooldown_until and current_ts < cooldown_until:
                    continue

                # Check if we have an open trade
                if open_trade and not open_trade.result:
                    # Simulate trade with remaining candles
                    future = ohlcv[i:]
                    open_trade = self.simulate_trade(open_trade, future)

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

                        if self.config.verbose:
                            print(f"  {current_ts} | {open_trade.direction} | {open_trade.result} | PnL: {open_trade.pnl_pct:.2f}%")

                        # Set cooldown (30 minutes as in live bot)
                        cooldown_until = current_ts + timedelta(minutes=30)
                        open_trade = None
                    continue

                # Analyze for new signal
                window = ohlcv[:i+1]
                signal = analyze_candle(window, self.config)

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
                        reasons=signal['reasons']
                    )
                    open_trade = trade

            # Handle any remaining open trade
            if open_trade and not open_trade.result:
                # Mark as timeout if not closed
                open_trade.result = "TIMEOUT"
                open_trade.exit_price = ohlcv[-1][4]  # Close at last price
                open_trade.exit_time = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
                if open_trade.direction == "LONG":
                    open_trade.pnl_pct = ((open_trade.exit_price - open_trade.entry_price) / open_trade.entry_price) * 100
                else:
                    open_trade.pnl_pct = ((open_trade.entry_price - open_trade.exit_price) / open_trade.entry_price) * 100
                result.trades.append(open_trade)
                result.total_trades += 1
                result.total_pnl_pct += open_trade.pnl_pct

        # Calculate final metrics
        self._calculate_metrics(result)
        return result

    def _calculate_metrics(self, result: BacktestResult) -> None:
        """Calculate performance metrics."""
        if result.total_trades == 0:
            return

        # Win rate
        result.win_rate = (result.wins / result.total_trades) * 100 if result.total_trades > 0 else 0

        # Average win/loss
        wins = [t.pnl_pct for t in result.trades if t.pnl_pct > 0]
        losses = [t.pnl_pct for t in result.trades if t.pnl_pct < 0]

        result.avg_win_pct = sum(wins) / len(wins) if wins else 0
        result.avg_loss_pct = sum(losses) / len(losses) if losses else 0

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Equity curve and max drawdown
        equity = 100.0  # Start with $100
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
    """Print backtest results in a formatted way."""
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

        # Calculate breakeven win rate
        breakeven_wr = 1 / (1 + win_loss_ratio) * 100
        print(f"  Breakeven WR:     {breakeven_wr:.1f}%")

    # Expected value per trade
    if result.total_trades > 0:
        ev = result.total_pnl_pct / result.total_trades
        print(f"  Expected Value:   {ev:.2f}% per trade")

    # Final equity
    if result.equity_curve:
        print(f"  Final Equity:     ${result.equity_curve[-1]:.2f} (started $100)")

    print(f"\n📈 STRATEGY VERDICT:")
    if result.profit_factor > 1.5 and result.win_rate > 40:
        print("  ✅ PROFITABLE - Strategy shows positive expectancy")
    elif result.profit_factor > 1.0:
        print("  ⚠️  MARGINAL - Strategy is marginally profitable")
    else:
        print("  ❌ UNPROFITABLE - Strategy has negative expectancy")

    # Trade breakdown by symbol
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
            "reasons": t.reasons
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
                "max_drawdown_pct": result.max_drawdown_pct
            },
            "trades": trades_data
        }, f, indent=2)

    print(f"Trades exported to {filename}")


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Backtest Volume Profile Bot Strategy")
    parser.add_argument("--timeframe", "-t", default="15m", help="Timeframe (default: 15m)")
    parser.add_argument("--days", "-d", type=int, default=30, help="Lookback days (default: 30)")
    parser.add_argument("--symbols", "-s", nargs="+", help="Specific symbols to test")
    parser.add_argument("--export", "-e", help="Export trades to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--min-score", type=int, default=3, help="Minimum score for signals (default: 3)")

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
        min_score=args.min_score
    )

    # Run backtest
    engine = BacktestEngine(config)
    result = engine.run_backtest(symbols)

    # Print results
    print_results(result, config)

    # Export if requested
    if args.export:
        export_trades(result, args.export)


if __name__ == "__main__":
    main()
