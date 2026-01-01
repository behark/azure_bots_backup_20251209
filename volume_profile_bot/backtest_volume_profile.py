#!/usr/bin/env python3
"""Backtest script for Volume Profile Bot (Smart Confluence).

Simulates trading signals over historical data to evaluate strategy performance.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ccxt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = BASE_DIR / "watchlist.json"


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


def calculate_volume_profile(highs: List[float], lows: List[float],
                              volumes: List[float], num_rows: int = 24) -> Dict[str, Any]:
    """Calculate volume profile metrics."""
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


def analyze_candle(opens: List[float], highs: List[float], lows: List[float],
                   closes: List[float], volumes: List[float]) -> Optional[Dict[str, Any]]:
    """Analyze a candle and generate signal if conditions are met."""
    current_price = closes[-1]

    # Volume Profile Analysis (Last 100 candles)
    vp = calculate_volume_profile(highs[-100:], lows[-100:], volumes[-100:])
    if not vp:
        return None

    # Technical Indicators
    rsi = calculate_rsi(np.array(closes))
    ema20 = sum(closes[-20:]) / 20
    pattern = detect_candlestick_pattern(opens, highs, lows, closes)

    # Volume Spike (1.2x average of last 20)
    avg_vol = sum(volumes[-20:]) / 20
    vol_spike = volumes[-1] > avg_vol * 1.2

    # Context Analysis - Near HVN
    near_hvn = False
    min_dist = float('inf')
    for hvn_price, _ in vp['hvn_levels']:
        dist = abs(current_price - hvn_price)
        if dist < min_dist:
            min_dist = dist

    if min_dist < vp['row_height'] * 2:
        near_hvn = True

    # Scoring System
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
        if current_price < vp['poc']:  # Below POC + Near HVN = Support
            long_score += 1
            reasons.append("Bounce off Volume Support")
        elif current_price > vp['poc']:  # Above POC + Near HVN = Resistance
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

    # Signal Generation
    signal = None
    if long_score >= 3 and long_score > short_score:
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

    elif short_score >= 3 and short_score > long_score:
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
                    lookback: int = 500, min_candles: int = 150) -> Dict[str, Any]:
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

        # Use data up to candle i
        hist_opens = opens[:i]
        hist_highs = highs[:i]
        hist_lows = lows[:i]
        hist_closes = closes[:i]
        hist_volumes = volumes[:i]

        signal = analyze_candle(hist_opens, hist_highs, hist_lows, hist_closes, hist_volumes)

        if signal:
            # Check result using future candles
            future_candles = ohlcv[i:i+50]  # Next 50 candles
            result, exit_price, bars = check_trade_result(
                signal["type"], signal["entry"], signal["sl"], signal["tp1"], signal["tp2"], future_candles
            )

            # Calculate PnL
            if signal["type"] == "LONG":
                pnl_pct = ((exit_price - signal["entry"]) / signal["entry"]) * 100
            else:
                pnl_pct = ((signal["entry"] - exit_price) / signal["entry"]) * 100

            trade = {
                "direction": signal["type"],
                "score": signal["score"],
                "entry": signal["entry"],
                "sl": signal["sl"],
                "tp1": signal["tp1"],
                "tp2": signal["tp2"],
                "result": result,
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
                "bars_held": bars,
                "reasons": signal["reasons"],
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
    print("VOLUME PROFILE BOT (SMART CONFLUENCE) - BACKTEST")
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
        timeframe = item.get("timeframe", item.get("period", "5m"))

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
