#!/usr/bin/env python3
"""Backtest script for Volume VN Bot.

Simulates trading signals over historical data to evaluate strategy performance.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ccxt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
WATCHLIST_FILE = BASE_DIR / "volume_watchlist.json"

sys.path.insert(0, str(ROOT_DIR))

# Import volume profile calculation
import volume_profile as vp


def calculate_rsi(closes: np.ndarray, period: int = 14) -> float:
    """Calculate RSI indicator."""
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


def detect_candlestick_pattern(opens: List[float], highs: List[float],
                                lows: List[float], closes: List[float]) -> Optional[str]:
    """Detect candlestick patterns."""
    if len(closes) < 2:
        return None
    current_open, current_high, current_low, current_close = opens[-1], highs[-1], lows[-1], closes[-1]
    prev_open, _, _, prev_close = opens[-2], highs[-2], lows[-2], closes[-2]
    body = abs(current_close - current_open)
    upper_wick = current_high - max(current_close, current_open)
    lower_wick = min(current_close, current_open) - current_low

    bullish_hammer = (current_close > current_open) and (lower_wick > 2 * body) and (upper_wick < body * 0.3)
    bullish_engulfing = (prev_close < prev_open) and (current_close > current_open) and (current_close > prev_open) and (current_open < prev_close)
    bearish_star = (current_open > current_close) and (upper_wick > 2 * body) and (lower_wick < body * 0.3)
    bearish_engulfing = (prev_close > prev_open) and (current_close < current_open) and (current_close < prev_open) and (current_open > prev_close)

    if bullish_hammer: return "BULLISH_HAMMER"
    if bullish_engulfing: return "BULLISH_ENGULFING"
    if bearish_star: return "BEARISH_STAR"
    if bearish_engulfing: return "BEARISH_ENGULFING"
    return None


def build_factors(current_price: float, ema20: float, rsi: float, pattern: Optional[str],
                  volume_spike: bool, near_hvn: bool, poc: float,
                  recent_candles: List[Tuple[float, float, float]]) -> Tuple[List[str], List[str]]:
    """Build long and short factors list."""
    long_factors: List[str] = []
    short_factors: List[str] = []

    if current_price > ema20:
        long_factors.append("Price > EMA20")
    else:
        short_factors.append("Price < EMA20")

    if 45 < rsi < 80:
        long_factors.append(f"RSI favorable ({rsi:.1f})")
    elif 20 < rsi < 55:
        short_factors.append(f"RSI favorable ({rsi:.1f})")

    if rsi > 50 and current_price > ema20:
        long_factors.append("Bullish Momentum (RSI Trend)")

    if near_hvn:
        if current_price < poc:
            long_factors.append("Near HVN below POC (support)")
        else:
            short_factors.append("Near HVN above POC (resistance)")

    if pattern:
        if "BULLISH" in pattern:
            long_factors.append(pattern)
        else:
            short_factors.append(pattern)

    if volume_spike:
        long_factors.append("Volume spike")

    if len(recent_candles) >= 6:
        greens = [v for o, c, v in reversed(recent_candles) if c > o][:3]
        reds = [v for o, c, v in reversed(recent_candles) if c < o][:3]
        if len(greens) >= 3 and len(reds) >= 3:
            if sum(greens) / 3 > (sum(reds) / 3) * 1.2:
                long_factors.append("Buying Pressure Dominant")

    return long_factors, short_factors


def generate_signal(current_price: float, vp_result: Dict[str, Any],
                    long_factors: List[str], short_factors: List[str]) -> Tuple[str, Optional[Dict[str, float]]]:
    """Generate trading signal based on factors."""
    val = float(vp_result.get("val", current_price))
    vah = float(vp_result.get("vah", current_price))
    row_height = float(vp_result.get("row_height", 0.0))

    if len(long_factors) >= 3 and len(long_factors) > len(short_factors):
        raw_sl = val - row_height
        custom_sl = min(raw_sl, current_price * 0.985)  # 1.5% stop loss
        risk = max(current_price - custom_sl, current_price * 0.01)
        tp1 = current_price + risk * 2.0
        tp2 = current_price + risk * 3.0
        return "LONG", {"entry": current_price, "sl": custom_sl, "tp1": tp1, "tp2": tp2}

    if len(short_factors) >= 3 and len(short_factors) > len(long_factors):
        raw_sl = vah + row_height
        custom_sl = max(raw_sl, current_price * 1.015)  # 1.5% stop loss
        risk = max(custom_sl - current_price, current_price * 0.01)
        tp1 = current_price - risk * 2.0
        tp2 = current_price - risk * 3.0
        return "SHORT", {"entry": current_price, "sl": custom_sl, "tp1": tp1, "tp2": tp2}

    return "NEUTRAL", None


def check_trade_result(direction: str, entry: float, sl: float, tp1: float, tp2: float,
                       future_candles: List[List[float]]) -> Tuple[str, float, int]:
    """
    Check trade result using future price data.
    Returns: (result, exit_price, bars_held)
    """
    for i, candle in enumerate(future_candles):
        high = candle[2]
        low = candle[3]

        if direction == "LONG":
            # Check SL first (conservative)
            if low <= sl:
                return "SL", sl, i + 1
            if high >= tp2:
                return "TP2", tp2, i + 1
            if high >= tp1:
                return "TP1", tp1, i + 1
        else:  # SHORT
            if high >= sl:
                return "SL", sl, i + 1
            if low <= tp2:
                return "TP2", tp2, i + 1
            if low <= tp1:
                return "TP1", tp1, i + 1

    # If no exit, return last close as exit
    if future_candles:
        return "OPEN", future_candles[-1][4], len(future_candles)
    return "OPEN", entry, 0


def backtest_symbol(client: ccxt.Exchange, symbol: str, timeframe: str,
                    lookback: int = 500, min_candles: int = 200) -> Dict[str, Any]:
    """Backtest a single symbol."""
    results = {
        "symbol": symbol,
        "timeframe": timeframe,
        "trades": [],
        "tp1_count": 0,
        "tp2_count": 0,
        "sl_count": 0,
        "open_count": 0,
        "total_pnl_pct": 0.0,
    }

    try:
        market_symbol = f"{symbol.replace('/USDT', '')}/USDT:USDT"
        ohlcv = client.fetch_ohlcv(market_symbol, timeframe, limit=lookback)
        if not ohlcv or len(ohlcv) < min_candles:
            results["error"] = f"Insufficient data: {len(ohlcv) if ohlcv else 0} candles"
            return results
    except Exception as e:
        results["error"] = str(e)
        return results

    opens = [x[1] for x in ohlcv]
    highs = [x[2] for x in ohlcv]
    lows = [x[3] for x in ohlcv]
    closes = [x[4] for x in ohlcv]
    volumes = [x[5] for x in ohlcv]

    # Iterate through candles, checking for signals
    cooldown = 0
    for i in range(100, len(ohlcv) - 50):  # Need 100 for VP, 50 for forward testing
        if cooldown > 0:
            cooldown -= 1
            continue

        # Use data up to candle i (exclusive of current incomplete candle)
        hist_opens = opens[:i]
        hist_highs = highs[:i]
        hist_lows = lows[:i]
        hist_closes = closes[:i]
        hist_volumes = volumes[:i]

        current_price = hist_closes[-1]
        ema20 = sum(hist_closes[-20:]) / 20

        # Volume profile on last 100 candles
        vp_result = vp.calculate_volume_profile(
            hist_highs[-100:], hist_lows[-100:], hist_closes[-100:], hist_volumes[-100:]
        )
        if not vp_result:
            continue

        rsi = calculate_rsi(np.array(hist_closes))
        pattern = detect_candlestick_pattern(hist_opens[:-1], hist_highs[:-1], hist_lows[:-1], hist_closes[:-1])

        # Volume spike detection
        if len(hist_volumes) >= 21:
            avg_volume = sum(hist_volumes[-21:-1]) / 20
            last_closed_volume = hist_volumes[-2]
            volume_spike = last_closed_volume > avg_volume * 1.5
        else:
            volume_spike = False

        poc = float(vp_result.get("poc", current_price))
        row_height = float(vp_result.get("row_height", 0.0))

        # Check if near HVN
        near_hvn = False
        hvn_levels = vp_result.get("hvn_levels", [])
        for entry in hvn_levels:
            if isinstance(entry, (list, tuple)) and len(entry) >= 1:
                hvn_price = entry[0]
                if abs(current_price - hvn_price) < row_height * 2:
                    near_hvn = True
                    break

        recent_candles = list(zip(hist_opens[-20:], hist_closes[-20:], hist_volumes[-20:]))
        long_factors, short_factors = build_factors(
            current_price, ema20, rsi, pattern, volume_spike, near_hvn, poc, recent_candles
        )

        signal, setup = generate_signal(current_price, vp_result, long_factors, short_factors)

        if signal != "NEUTRAL" and setup:
            # Check result using future candles
            future_candles = ohlcv[i:i+50]  # Next 50 candles
            result, exit_price, bars = check_trade_result(
                signal, setup["entry"], setup["sl"], setup["tp1"], setup["tp2"], future_candles
            )

            # Calculate PnL
            if signal == "LONG":
                pnl_pct = ((exit_price - setup["entry"]) / setup["entry"]) * 100
            else:
                pnl_pct = ((setup["entry"] - exit_price) / setup["entry"]) * 100

            trade = {
                "direction": signal,
                "entry": setup["entry"],
                "sl": setup["sl"],
                "tp1": setup["tp1"],
                "tp2": setup["tp2"],
                "result": result,
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
                "bars_held": bars,
                "factors": long_factors if signal == "LONG" else short_factors,
            }
            results["trades"].append(trade)
            results["total_pnl_pct"] += pnl_pct

            if result == "TP1":
                results["tp1_count"] += 1
            elif result == "TP2":
                results["tp2_count"] += 1
            elif result == "SL":
                results["sl_count"] += 1
            else:
                results["open_count"] += 1

            # Set cooldown (don't take new trades for 10 bars)
            cooldown = 10

    return results


def run_backtest():
    """Run backtest on all watchlist symbols."""
    print("=" * 60)
    print("VOLUME VN BOT - BACKTEST")
    print("=" * 60)
    print()

    # Load watchlist
    watchlist = json.loads(WATCHLIST_FILE.read_text())
    print(f"Loaded {len(watchlist)} symbols from watchlist")
    print()

    # Initialize exchange client
    client = ccxt.binanceusdm({"enableRateLimit": True})

    all_results = []
    total_trades = 0
    total_tp1 = 0
    total_tp2 = 0
    total_sl = 0
    total_pnl = 0.0

    for item in watchlist:
        symbol = item.get("symbol", "")
        timeframe = item.get("timeframe", "5m")

        print(f"Backtesting {symbol} ({timeframe})...", end=" ", flush=True)

        result = backtest_symbol(client, symbol, timeframe)
        all_results.append(result)

        if "error" in result:
            print(f"ERROR: {result['error']}")
            continue

        trades = len(result["trades"])
        if trades == 0:
            print("No signals generated")
            continue

        wins = result["tp1_count"] + result["tp2_count"]
        win_rate = (wins / trades) * 100 if trades > 0 else 0

        print(f"{trades} trades | Win Rate: {win_rate:.1f}% | PnL: {result['total_pnl_pct']:+.2f}%")

        total_trades += trades
        total_tp1 += result["tp1_count"]
        total_tp2 += result["tp2_count"]
        total_sl += result["sl_count"]
        total_pnl += result["total_pnl_pct"]

    # Summary
    print()
    print("=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Total Trades: {total_trades}")
    print(f"TP1 Hits: {total_tp1}")
    print(f"TP2 Hits: {total_tp2}")
    print(f"SL Hits: {total_sl}")

    wins = total_tp1 + total_tp2
    if total_trades > 0:
        win_rate = (wins / total_trades) * 100
        avg_pnl = total_pnl / total_trades
        print(f"Win Rate: {win_rate:.1f}% ({wins}/{total_trades})")
        print(f"Total PnL: {total_pnl:+.2f}%")
        print(f"Average PnL per Trade: {avg_pnl:+.2f}%")
    else:
        print("No trades generated")

    print("=" * 60)

    return all_results


if __name__ == "__main__":
    run_backtest()
