#!/usr/bin/env python3
"""Backtest for Volume Profile Bot V2."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ccxt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = BASE_DIR / "watchlist.json"

# V2 Configuration
CONFIG = {
    "min_score": 4,
    "rsi_long_min": 40,
    "rsi_long_max": 55,
    "rsi_short_min": 45,
    "rsi_short_max": 60,
    "volume_spike_threshold": 1.5,
    "min_risk_reward": 1.5,
    "max_sl_percent": 2.0,
    "min_sl_percent": 0.5,
    "tp1_rr": 1.5,
    "tp2_rr": 2.5,
}


def calculate_rsi(closes: np.ndarray, period: int = 14) -> float:
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


def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 0.0
    tr_list = []
    for i in range(1, len(closes)):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i-1])
        low_close = abs(lows[i] - closes[i-1])
        tr_list.append(max(high_low, high_close, low_close))
    if len(tr_list) < period:
        return sum(tr_list) / len(tr_list) if tr_list else 0.0
    return sum(tr_list[-period:]) / period


def detect_candlestick_pattern(opens: List[float], highs: List[float],
                                lows: List[float], closes: List[float]) -> Optional[str]:
    if len(closes) < 3:
        return None
    idx = -1
    current_open, current_high, current_low, current_close = opens[idx], highs[idx], lows[idx], closes[idx]
    prev_open, prev_high, prev_low, prev_close = opens[idx-1], highs[idx-1], lows[idx-1], closes[idx-1]
    body = abs(current_close - current_open)
    if body == 0:
        body = 0.0001
    upper_wick = current_high - max(current_close, current_open)
    lower_wick = min(current_close, current_open) - current_low

    bullish_hammer = (current_close > current_open) and (lower_wick > 2 * body) and (upper_wick < body * 0.3)
    bullish_engulfing = (prev_close < prev_open) and (current_close > current_open) and \
                        (current_close > prev_open) and (current_open < prev_close) and \
                        (body > abs(prev_close - prev_open) * 1.2)
    bearish_star = (current_open > current_close) and (upper_wick > 2 * body) and (lower_wick < body * 0.3)
    bearish_engulfing = (prev_close > prev_open) and (current_close < current_open) and \
                        (current_close < prev_open) and (current_open > prev_close) and \
                        (body > abs(prev_close - prev_open) * 1.2)

    if bullish_hammer: return "BULLISH_HAMMER"
    if bullish_engulfing: return "BULLISH_ENGULFING"
    if bearish_star: return "BEARISH_STAR"
    if bearish_engulfing: return "BEARISH_ENGULFING"
    return None


def calculate_volume_profile(highs: List[float], lows: List[float],
                              volumes: List[float], num_rows: int = 24) -> Dict[str, Any]:
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


def analyze_v2(opens: List[float], highs: List[float], lows: List[float],
               closes: List[float], volumes: List[float]) -> Optional[Dict[str, Any]]:
    """V2 analysis with improved logic."""
    current_price = closes[-1]

    vp = calculate_volume_profile(highs[-100:], lows[-100:], volumes[-100:])
    if not vp:
        return None

    rsi = calculate_rsi(np.array(closes))
    ema20 = sum(closes[-20:]) / 20
    ema50 = sum(closes[-50:]) / 50
    atr = calculate_atr(highs, lows, closes)
    pattern = detect_candlestick_pattern(opens, highs, lows, closes)

    avg_vol = sum(volumes[-20:]) / 20
    vol_spike = volumes[-1] > avg_vol * CONFIG["volume_spike_threshold"]

    near_hvn = False
    min_dist = float('inf')
    for hvn_price, _ in vp['hvn_levels']:
        dist = abs(current_price - hvn_price)
        if dist < min_dist:
            min_dist = dist
    if min_dist < vp['row_height'] * 1.5:
        near_hvn = True

    long_score = 0
    short_score = 0
    reasons = []

    # Trend Filter
    bullish_trend = ema20 > ema50 and current_price > ema20
    bearish_trend = ema20 < ema50 and current_price < ema20

    if bullish_trend:
        long_score += 1
        reasons.append("Bullish Trend")
    elif bearish_trend:
        short_score += 1
        reasons.append("Bearish Trend")

    # RSI - No overlap
    if CONFIG["rsi_long_min"] < rsi < CONFIG["rsi_long_max"]:
        long_score += 1
        reasons.append(f"RSI Bullish ({rsi:.1f})")
    elif CONFIG["rsi_short_min"] < rsi < CONFIG["rsi_short_max"]:
        short_score += 1
        reasons.append(f"RSI Bearish ({rsi:.1f})")

    # HVN
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
        if closes[-1] > opens[-1]:
            long_score += 1
            reasons.append("Volume Spike (Buying)")
        else:
            short_score += 1
            reasons.append("Volume Spike (Selling)")

    # Value Area
    if current_price < vp['val']:
        long_score += 1
        reasons.append("Below Value Area")
    elif current_price > vp['vah']:
        short_score += 1
        reasons.append("Above Value Area")

    signal = None
    min_score = CONFIG["min_score"]

    if long_score >= min_score and long_score > short_score and bullish_trend:
        signal = "LONG"
        atr_sl = current_price - (atr * 1.5)
        vp_sl = vp['val'] - (vp['row_height'] * 0.5)
        sl = max(atr_sl, vp_sl)
        sl = max(sl, current_price * (1 - CONFIG["max_sl_percent"] / 100))
        sl = min(sl, current_price * (1 - CONFIG["min_sl_percent"] / 100))
        risk = current_price - sl
        tp1 = current_price + (risk * CONFIG["tp1_rr"])
        tp2 = current_price + (risk * CONFIG["tp2_rr"])
        rr_ratio = (tp1 - current_price) / risk if risk > 0 else 0
        if rr_ratio < CONFIG["min_risk_reward"]:
            return None

    elif short_score >= min_score and short_score > long_score and bearish_trend:
        signal = "SHORT"
        atr_sl = current_price + (atr * 1.5)
        vp_sl = vp['vah'] + (vp['row_height'] * 0.5)
        sl = min(atr_sl, vp_sl)
        sl = min(sl, current_price * (1 + CONFIG["max_sl_percent"] / 100))
        sl = max(sl, current_price * (1 + CONFIG["min_sl_percent"] / 100))
        risk = sl - current_price
        tp1 = current_price - (risk * CONFIG["tp1_rr"])
        tp2 = current_price - (risk * CONFIG["tp2_rr"])
        rr_ratio = (current_price - tp1) / risk if risk > 0 else 0
        if rr_ratio < CONFIG["min_risk_reward"]:
            return None

    if signal:
        return {
            "type": signal,
            "score": max(long_score, short_score),
            "reasons": reasons,
            "entry": current_price,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "rr_ratio": rr_ratio,
        }
    return None


def check_trade_result(direction: str, entry: float, sl: float, tp1: float, tp2: float,
                       future_candles: List[List[float]]) -> Tuple[str, float, int]:
    for i, candle in enumerate(future_candles):
        high = candle[2]
        low = candle[3]
        if direction == "LONG":
            if low <= sl:
                return "SL", sl, i + 1
            if high >= tp2:
                return "TP2", tp2, i + 1
            if high >= tp1:
                return "TP1", tp1, i + 1
        else:
            if high >= sl:
                return "SL", sl, i + 1
            if low <= tp2:
                return "TP2", tp2, i + 1
            if low <= tp1:
                return "TP1", tp1, i + 1
    if future_candles:
        return "OPEN", future_candles[-1][4], len(future_candles)
    return "OPEN", entry, 0


def backtest_symbol(client: ccxt.Exchange, symbol: str, timeframe: str,
                    lookback: int = 500) -> Dict[str, Any]:
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
        if not ohlcv or len(ohlcv) < 150:
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

    cooldown = 0
    for i in range(100, len(ohlcv) - 50):
        if cooldown > 0:
            cooldown -= 1
            continue

        hist_opens = opens[:i]
        hist_highs = highs[:i]
        hist_lows = lows[:i]
        hist_closes = closes[:i]
        hist_volumes = volumes[:i]

        signal = analyze_v2(hist_opens, hist_highs, hist_lows, hist_closes, hist_volumes)

        if signal:
            future_candles = ohlcv[i:i+50]
            result, exit_price, bars = check_trade_result(
                signal["type"], signal["entry"], signal["sl"], signal["tp1"], signal["tp2"], future_candles
            )

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

            cooldown = 15  # Longer cooldown

    return results


def run_backtest():
    print("=" * 60)
    print("VOLUME PROFILE BOT V2 - BACKTEST")
    print("=" * 60)
    print(f"Config: min_score={CONFIG['min_score']}, min_rr={CONFIG['min_risk_reward']}")
    print()

    watchlist = json.loads(WATCHLIST_FILE.read_text())
    print(f"Loaded {len(watchlist)} symbols")
    print()

    client = ccxt.binanceusdm({"enableRateLimit": True})

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

        if "error" in result:
            print(f"ERROR: {result['error']}")
            continue

        trades = len(result["trades"])
        if trades == 0:
            print("No signals")
            continue

        wins = result["tp1_count"] + result["tp2_count"]
        win_rate = (wins / trades) * 100 if trades > 0 else 0

        print(f"{trades} trades | WR: {win_rate:.1f}% | PnL: {result['total_pnl_pct']:+.2f}%")

        total_trades += trades
        total_tp1 += result["tp1_count"]
        total_tp2 += result["tp2_count"]
        total_sl += result["sl_count"]
        total_pnl += result["total_pnl_pct"]

    print()
    print("=" * 60)
    print("V2 BACKTEST SUMMARY")
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
        print(f"Avg PnL/Trade: {avg_pnl:+.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    run_backtest()
