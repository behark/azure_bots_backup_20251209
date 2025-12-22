#!/usr/bin/env python3
"""
Volume Profile Analyzer - Binance Futures
Finds POC (Point of Control), Value Area, and HVN (High Volume Nodes)

Usage: python3 volume_profile.py <SYMBOL> [TIMEFRAME]
Example: python3 volume_profile.py APR 15m
"""

import sys
import ccxt  # type: ignore[import-untyped]
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy.typing as npt


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol to always have /USDT suffix (without duplication).

    Handles cases where symbol may already contain /USDT to avoid
    creating malformed symbols like 'NIGHT/USDT/USDT'.
    """
    base = symbol.replace("/USDT", "").replace("_USDT", "")
    return f"{base}/USDT"


def calculate_volume_profile(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: List[float],
    num_rows: int = 24
) -> Dict[str, Any]:
    """
    Calculate volume profile analysis including Point of Control, Value Area, and High Volume Nodes.

    This function divides the price range into horizontal levels (rows) and distributes
    traded volume across these levels. It identifies key support/resistance zones where
    significant trading activity occurred.

    Algorithm:
        1. Divide price range (highest - lowest) into num_rows equal levels
        2. Distribute each candle's volume across price levels it touched
        3. Find POC as the level with maximum volume
        4. Calculate Value Area containing 70% of total volume around POC
        5. Identify High Volume Nodes (HVNs) exceeding 1.5x average volume

    Args:
        highs (list[float]): High prices for each candle
        lows (list[float]): Low prices for each candle
        closes (list[float]): Close prices for each candle (not currently used)
        volumes (list[float]): Trading volumes for each candle
        num_rows (int, optional): Number of price levels to divide range into.
                                   More rows = finer granularity. Defaults to 24.

    Returns:
        dict: Volume profile analysis containing:
            - poc (float): Point of Control - price level with highest volume
            - vah (float): Value Area High - upper bound of 70% volume area
            - val (float): Value Area Low - lower bound of 70% volume area
            - hvn_levels (list[tuple]): High Volume Nodes as (price, volume) tuples
            - volume_profile (list[tuple]): Full profile as (price, volume) tuples
            - row_height (float): Height of each price level in the profile
            - avg_vol (float): Average volume per price level
            - total_volume (float): Total volume across all levels

    Example:
        >>> highs = [100.5, 101.0, 100.8]
        >>> lows = [99.5, 100.0, 99.8]
        >>> closes = [100.0, 100.5, 100.3]
        >>> volumes = [1000, 1200, 900]
        >>> vp = calculate_volume_profile(highs, lows, closes, volumes, num_rows=12)
        >>> print(f"POC: ${vp['poc']:.2f}")
        >>> print(f"Value Area: ${vp['val']:.2f} - ${vp['vah']:.2f}")

    Notes:
        - POC acts as a price magnet where price tends to return
        - Value Area represents fair value range (70% of volume)
        - HVNs are strong support/resistance levels
        - Price above POC suggests bullish control, below suggests bearish
    """
    highest = max(highs)
    lowest = min(lows)
    price_range = highest - lowest
    row_height = price_range / num_rows

    # Initialize volume at each price level
    volume_profile = [0.0] * num_rows
    price_levels = [lowest + (row_height * i) + (row_height / 2) for i in range(num_rows)]

    # Distribute volume across price levels
    for i in range(len(highs)):
        bar_high = highs[i]
        bar_low = lows[i]
        bar_volume = volumes[i]

        for j in range(num_rows):
            price_level = price_levels[j]
            if bar_low <= price_level <= bar_high:
                volume_profile[j] += bar_volume

    # Find POC (highest volume level)
    max_vol_idx = volume_profile.index(max(volume_profile))
    poc = price_levels[max_vol_idx]

    # Calculate Value Area (70% of volume)
    total_volume = sum(volume_profile)
    target_volume = total_volume * 0.70
    avg_vol = total_volume / num_rows

    accumulated = volume_profile[max_vol_idx]
    upper_idx = max_vol_idx
    lower_idx = max_vol_idx

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

    vah = price_levels[upper_idx]  # Value Area High
    val = price_levels[lower_idx]  # Value Area Low

    # Find HVN (High Volume Nodes) - levels with > 1.5x average volume
    hvn_threshold = avg_vol * 1.5
    hvn_levels = []
    for i, vol in enumerate(volume_profile):
        if vol > hvn_threshold:
            hvn_levels.append((price_levels[i], vol))

    return {
        'poc': poc,
        'vah': vah,
        'val': val,
        'hvn_levels': hvn_levels,
        'volume_profile': list(zip(price_levels, volume_profile)),
        'row_height': row_height,
        'avg_vol': avg_vol,
        'total_volume': total_volume
    }


def calculate_rsi(
    closes: Union[List[float], "npt.NDArray[np.floating[Any]]"],
    period: int = 14
) -> float:
    """
    Calculate Relative Strength Index (RSI) momentum oscillator.

    RSI measures the speed and magnitude of price changes to identify overbought (>70)
    or oversold (<30) conditions. Uses exponential moving average of gains vs losses.

    Algorithm:
        1. Calculate price deltas between consecutive closes
        2. Separate into gains (positive deltas) and losses (negative deltas)
        3. Calculate initial average gain and loss over the period
        4. For remaining periods, use exponential smoothing:
           avg_gain = (avg_gain × (period-1) + current_gain) / period
        5. Calculate RS = avg_gain / avg_loss
        6. Convert to RSI = 100 - (100 / (1 + RS))

    Args:
        closes (list[float] or np.ndarray): Close prices in chronological order
        period (int, optional): Lookback period for RSI calculation. Defaults to 14.

    Returns:
        float: RSI value between 0 and 100
            - RSI > 70: Overbought (potential sell signal)
            - RSI < 30: Oversold (potential buy signal)
            - RSI 40-60: Neutral zone

    Example:
        >>> closes = [100, 102, 101, 103, 105, 104, 106, 108]
        >>> rsi = calculate_rsi(closes, period=5)
        >>> print(f"RSI: {rsi:.1f}")

    Notes:
        - Returns 100 if there are no losses (all gains)
        - Standard period is 14, but shorter periods (7, 9) are more reactive
        - Works best in ranging markets, can give false signals in strong trends
        - Often used with divergence analysis for reversal signals
    """
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def detect_candlestick_pattern(
    opens: List[float],
    highs: List[float],
    lows: List[float],
    closes: List[float]
) -> Optional[str]:
    """
    Detect common bullish and bearish candlestick reversal patterns.

    Analyzes the last two candles to identify four classic reversal patterns:
    - Bullish Hammer: Long lower wick signals rejection of lower prices
    - Bullish Engulfing: Larger bullish candle engulfs previous bearish candle
    - Bearish Star: Long upper wick signals rejection of higher prices
    - Bearish Engulfing: Larger bearish candle engulfs previous bullish candle

    Pattern Requirements:
        Bullish Hammer:
            - Current candle is bullish (close > open)
            - Lower wick > 2× body size
            - Upper wick < 0.3× body size
            - Indicates buying pressure after dip

        Bullish Engulfing:
            - Previous candle is bearish
            - Current candle is bullish
            - Current close > previous open
            - Current open < previous close
            - Indicates trend reversal to upside

        Bearish Star:
            - Current candle is bearish (close < open)
            - Upper wick > 2× body size
            - Lower wick < 0.3× body size
            - Indicates selling pressure after rally

        Bearish Engulfing:
            - Previous candle is bullish
            - Current candle is bearish
            - Current close < previous open
            - Current open > previous close
            - Indicates trend reversal to downside

    Args:
        opens (list[float]): Open prices (needs at least last 2 candles)
        highs (list[float]): High prices (needs at least last 2 candles)
        lows (list[float]): Low prices (needs at least last 2 candles)
        closes (list[float]): Close prices (needs at least last 2 candles)

    Returns:
        str or None: Pattern name if detected, None otherwise
            - "BULLISH_HAMMER"
            - "BULLISH_ENGULFING"
            - "BEARISH_STAR"
            - "BEARISH_ENGULFING"
            - None (no pattern detected)

    Example:
        >>> opens = [100, 102, 101]
        >>> highs = [101, 103, 105]  # Large upper wick
        >>> lows = [99, 101, 100]
        >>> closes = [100.5, 102.5, 100.5]  # Bearish close
        >>> pattern = detect_candlestick_pattern(opens, highs, lows, closes)
        >>> if pattern:
        ...     print(f"Pattern detected: {pattern}")

    Notes:
        - Works best at support/resistance levels
        - Requires volume confirmation for higher accuracy
        - More reliable on higher timeframes (4h, 1d)
        - Should be combined with trend analysis and other indicators
        - Single pattern alone is not sufficient for trading decision
    """
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
    elif bullish_engulfing:
        return "BULLISH_ENGULFING"
    elif bearish_star:
        return "BEARISH_STAR"
    elif bearish_engulfing:
        return "BEARISH_ENGULFING"
    return None


def analyze_volume_profile(symbol: str, timeframe: str = "15m") -> None:
    """
    Perform comprehensive volume profile analysis and generate trading signals for a cryptocurrency.

    This is the main analysis orchestrator that fetches market data, calculates volume profile,
    evaluates technical indicators (RSI, EMA, candlestick patterns), and generates actionable
    trading signals (LONG/SHORT/NEUTRAL) with entry, stop-loss, and take-profit levels.

    Analysis Components:
        1. **Data Fetching**: Retrieves 200 OHLCV candles from Binance Futures
        2. **Volume Profile**: Calculates POC, VAH/VAL, and HVNs from last 100 candles
        3. **Technical Indicators**:
           - RSI(14): Momentum oscillator for overbought/oversold conditions
           - EMA(20): Trend direction indicator
           - Candlestick Patterns: Reversal pattern detection
           - Volume Analysis: Identifies volume spikes (>1.5x average)
        4. **Price Context**: Determines price position relative to VP levels
        5. **Signal Generation**: Multi-factor scoring system for trade signals
        6. **Trade Setup**: Calculates risk-reward based entries and exits

    Signal Logic (LONG):
        Requires ≥3 bullish factors AND more bullish than bearish factors:
        - Price > EMA20
        - RSI between 35-60 (not overbought)
        - Price near HVN below POC (support bounce)
        - Bullish candlestick pattern detected
        - Volume spike with bullish close

        Trade Setup:
        - Entry: Current price
        - Stop Loss: VAL - row_height (below value area)
        - TP1: POC (point of control as magnet)
        - TP2: 2R (2× risk distance)

    Signal Logic (SHORT):
        Requires ≥3 bearish factors AND more bearish than bullish factors:
        - Price < EMA20
        - RSI between 40-65 (not oversold)
        - Price near HVN above POC (resistance rejection)
        - Bearish candlestick pattern detected
        - Volume spike with bearish close

        Trade Setup:
        - Entry: Current price
        - Stop Loss: VAH + row_height (above value area)
        - TP1: POC (point of control as magnet)
        - TP2: 2R (2× risk distance)

    Args:
        symbol (str): Cryptocurrency symbol (e.g., "BTC", "APR", "MINA").
                     Automatically converted to uppercase and appended with "/USDT:USDT"
                     for Binance Futures format.
        timeframe (str, optional): Candlestick timeframe for analysis.
                                  Valid values: "1m", "5m", "15m", "1h", "4h", "1d".
                                  Defaults to "15m".

    Returns:
        None: Prints comprehensive analysis to stdout including:
            - Market overview (price, EMA, RSI, volume)
            - Volume profile levels (POC, VAH, VAL)
            - Price position context
            - High volume nodes (HVNs)
            - Bullish/bearish factor scoring
            - Trading signal (LONG/SHORT/NEUTRAL)
            - Trade setup with entry, SL, TP levels
            - ASCII volume profile histogram
            - Risk disclaimer

    Example:
        >>> analyze_volume_profile("APR", "15m")
        ======================================================================
        VOLUME PROFILE ANALYSIS: APR/USDT
        Timeframe: 15m
        ======================================================================

        ✓ Loaded 200 candles from Binance Futures

        ======================================================================
        📊 MARKET OVERVIEW
        ======================================================================
        Current Price: $1.234567
        EMA(20):       $1.220000
        RSI(14):       45.3
        Volume:        12,345

        [... full analysis output ...]

    Raises:
        Exception: If Binance API request fails (e.g., invalid symbol, network error).
                  Error is caught and printed to user with helpful message.

    Notes:
        - Uses last 100 candles for volume profile (sufficient for intraday analysis)
        - Full 200 candles used for RSI calculation (needs warmup period)
        - HVNs identified as levels with >1.5× average volume
        - Value Area represents 70% of total volume distribution
        - POC acts as a "price magnet" where price tends to gravitate
        - Signals are educational; not financial advice
        - Lower timeframes (1m, 5m) generate more signals but higher noise
        - Higher timeframes (1h, 4h, 1d) generate fewer but more reliable signals
    """
    symbol = symbol.upper().replace("/USDT", "").replace("_USDT", "")
    full_symbol = f"{symbol}/USDT:USDT"

    print("=" * 70)
    print(f"VOLUME PROFILE ANALYSIS: {symbol}/USDT")
    print(f"Timeframe: {timeframe}")
    print("=" * 70)

    # Fetch data
    try:
        exchange = ccxt.binanceusdm()
        ohlcv = exchange.fetch_ohlcv(full_symbol, timeframe, limit=200)
        print(f"\n✓ Loaded {len(ohlcv)} candles from Binance Futures")
    except Exception as e:
        print(f"\n❌ Error fetching data: {e}")
        return

    # Extract OHLCV
    opens = [x[1] for x in ohlcv]
    highs = [x[2] for x in ohlcv]
    lows = [x[3] for x in ohlcv]
    closes = [x[4] for x in ohlcv]
    volumes = [x[5] for x in ohlcv]

    current_price = closes[-1]
    ema20 = sum(closes[-20:]) / 20

    # Calculate Volume Profile
    vp = calculate_volume_profile(highs[-100:], lows[-100:], closes[-100:], volumes[-100:])

    # Calculate RSI
    rsi = calculate_rsi(np.array(closes))

    # Detect candlestick pattern
    pattern = detect_candlestick_pattern(opens, highs, lows, closes)

    # Check volume spike
    avg_volume = sum(volumes[-20:]) / 20
    current_volume = volumes[-1]
    volume_spike = current_volume > avg_volume * 1.5

    # Price position relative to VP levels
    price_vs_poc = "ABOVE" if current_price > vp['poc'] else "BELOW"
    in_value_area = vp['val'] <= current_price <= vp['vah']

    # Find nearest HVN
    nearest_hvn = None
    min_dist = float('inf')
    for hvn_price, hvn_vol in vp['hvn_levels']:
        dist = abs(current_price - hvn_price)
        if dist < min_dist:
            min_dist = dist
            nearest_hvn = hvn_price

    near_hvn = min_dist < vp['row_height'] * 2 if nearest_hvn else False

    print(f"\n{'=' * 70}")
    print("📊 MARKET OVERVIEW")
    print("=" * 70)
    print(f"Current Price: ${current_price:.6f}")
    print(f"EMA(20):       ${ema20:.6f}")
    print(f"RSI(14):       {rsi:.1f}")
    print(f"Volume:        {current_volume:,.0f} {'📈 SPIKE!' if volume_spike else ''}")

    print(f"\n{'=' * 70}")
    print("📊 VOLUME PROFILE LEVELS")
    print("=" * 70)
    print(f"POC (Point of Control): ${vp['poc']:.6f}")
    print(f"VAH (Value Area High):  ${vp['vah']:.6f}")
    print(f"VAL (Value Area Low):   ${vp['val']:.6f}")

    print(f"\n🎯 Price is {price_vs_poc} POC")
    print(f"📍 {'INSIDE' if in_value_area else 'OUTSIDE'} Value Area")

    if vp['hvn_levels']:
        print(f"\n{'=' * 70}")
        print(f"🔵 HIGH VOLUME NODES (HVN) - {len(vp['hvn_levels'])} found")
        print("=" * 70)
        for hvn_price, hvn_vol in sorted(vp['hvn_levels'], key=lambda x: x[1], reverse=True)[:5]:
            dist_pct = ((hvn_price - current_price) / current_price) * 100
            marker = " ← NEAREST" if hvn_price == nearest_hvn else ""
            print(f"  ${hvn_price:.6f} ({dist_pct:+.2f}% from price){marker}")

    # Generate trading signal
    print(f"\n{'=' * 70}")
    print("🔍 SIGNAL ANALYSIS")
    print("=" * 70)

    # Conditions
    long_conditions = []
    short_conditions = []

    # EMA
    if current_price > ema20:
        long_conditions.append("Price > EMA20")
    else:
        short_conditions.append("Price < EMA20")

    # RSI
    if 35 < rsi < 60:
        long_conditions.append(f"RSI favorable ({rsi:.1f})")
    elif 40 < rsi < 65:
        short_conditions.append(f"RSI favorable ({rsi:.1f})")

    # Near HVN
    if near_hvn:
        if current_price < vp['poc']:
            long_conditions.append("Near HVN below POC (support)")
        else:
            short_conditions.append("Near HVN above POC (resistance)")

    # Candlestick pattern
    if pattern:
        if "BULLISH" in pattern:
            long_conditions.append(f"Pattern: {pattern}")
        else:
            short_conditions.append(f"Pattern: {pattern}")

    # Volume spike
    if volume_spike:
        long_conditions.append("Volume spike") if current_price > opens[-1] else short_conditions.append("Volume spike")

    print(f"\n🟢 BULLISH factors ({len(long_conditions)}):")
    for cond in long_conditions:
        print(f"   ✓ {cond}")

    print(f"\n🔴 BEARISH factors ({len(short_conditions)}):")
    for cond in short_conditions:
        print(f"   ✓ {cond}")

    # Determine signal
    print(f"\n{'=' * 70}")

    if len(long_conditions) >= 3 and len(long_conditions) > len(short_conditions):
        sl = vp['val'] - vp['row_height']
        tp1 = vp['poc']
        risk = current_price - sl
        tp2 = current_price + (risk * 2)

        print("🟢 SIGNAL: LONG")
        print("=" * 70)
        print("\n🎯 TRADE SETUP:")
        print(f"   Entry:     ${current_price:.6f}")
        print(f"   Stop Loss: ${sl:.6f} ({((sl - current_price) / current_price) * 100:+.2f}%)")
        print(f"   TP1 (POC): ${tp1:.6f} ({((tp1 - current_price) / current_price) * 100:+.2f}%)")
        print(f"   TP2 (2R):  ${tp2:.6f} ({((tp2 - current_price) / current_price) * 100:+.2f}%)")

    elif len(short_conditions) >= 3 and len(short_conditions) > len(long_conditions):
        sl = vp['vah'] + vp['row_height']
        tp1 = vp['poc']
        risk = sl - current_price
        tp2 = current_price - (risk * 2)

        print("🔴 SIGNAL: SHORT")
        print("=" * 70)
        print("\n🎯 TRADE SETUP:")
        print(f"   Entry:     ${current_price:.6f}")
        print(f"   Stop Loss: ${sl:.6f} ({((sl - current_price) / current_price) * 100:+.2f}%)")
        print(f"   TP1 (POC): ${tp1:.6f} ({((tp1 - current_price) / current_price) * 100:+.2f}%)")
        print(f"   TP2 (2R):  ${tp2:.6f} ({((tp2 - current_price) / current_price) * 100:+.2f}%)")

    else:
        print("⚪ SIGNAL: NEUTRAL - Wait for better setup")
        print("=" * 70)
        print("\nKey levels to watch:")
        print(f"   Support (VAL): ${vp['val']:.6f}")
        print(f"   Magnet (POC):  ${vp['poc']:.6f}")
        print(f"   Resistance (VAH): ${vp['vah']:.6f}")

    # Volume Profile Visualization (ASCII)
    print(f"\n{'=' * 70}")
    print("📊 VOLUME PROFILE HISTOGRAM")
    print("=" * 70)

    max_vol = max(v for _, v in vp['volume_profile'])
    for price, vol in reversed(vp['volume_profile']):
        bar_len = int((vol / max_vol) * 40)
        bar = "█" * bar_len

        # Markers
        marker = ""
        if abs(price - vp['poc']) < vp['row_height'] / 2:
            marker = " ← POC"
        elif abs(price - vp['vah']) < vp['row_height'] / 2:
            marker = " ← VAH"
        elif abs(price - vp['val']) < vp['row_height'] / 2:
            marker = " ← VAL"

        # Color based on price position
        if abs(price - current_price) < vp['row_height']:
            marker += " ★ PRICE"

        print(f"${price:.6f} |{bar}{marker}")

    print(f"\n{'=' * 70}")
    print("⚠️  This is not financial advice. Trade at your own risk.")
    print("=" * 70)


def main() -> None:
    """
    Command-line interface entry point for volume profile analysis.

    Parses command-line arguments to extract symbol and optional timeframe,
    then invokes the main analysis function. Provides usage help when
    arguments are missing or incorrect.

    Usage:
        python3 volume_profile.py <SYMBOL> [TIMEFRAME]

    Arguments:
        SYMBOL (required): Cryptocurrency symbol (case-insensitive)
                          Examples: apr, BTC, mina, eth
        TIMEFRAME (optional): Candlestick timeframe
                             Valid: 1m, 5m, 15m, 1h, 4h, 1d
                             Default: 15m

    Examples:
        $ python3 volume_profile.py apr
        # Analyzes APR/USDT on 15m timeframe (default)

        $ python3 volume_profile.py btc 1h
        # Analyzes BTC/USDT on 1h timeframe

        $ python3 volume_profile.py mina 5m
        # Analyzes MINA/USDT on 5m timeframe

    Shell Alias (recommended):
        Add to ~/.bashrc or ~/.zshrc:
        alias vol='python3 /path/to/volume_profile.py'

        Then use:
        $ vol apr        # Quick analysis
        $ vol btc 1h     # With timeframe

    Returns:
        None: Prints usage help or delegates to analyze_volume_profile()

    Notes:
        - Symbol is automatically converted to uppercase
        - Binance Futures format (SYMBOL/USDT:USDT) is auto-applied
        - Requires active internet connection for Binance API
        - No API key needed (public market data endpoint)
    """
    if len(sys.argv) < 2:
        print("=" * 70)
        print("VOLUME PROFILE ANALYZER")
        print("=" * 70)
        print("\nUsage: vol <symbol> [timeframe]")
        print("\nFinds: POC, Value Area (VAH/VAL), High Volume Nodes")
        print("\nExamples:")
        print("  vol apr        → Analyze APR/USDT on 15m")
        print("  vol btc 1h     → Analyze BTC/USDT on 1h")
        print("  vol mina 5m    → Analyze MINA/USDT on 5m")
        print("\nTimeframes: 1m, 5m, 15m, 1h, 4h, 1d")
        return

    symbol = sys.argv[1].upper()
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "15m"
    analyze_volume_profile(symbol, timeframe)


if __name__ == "__main__":
    main()
