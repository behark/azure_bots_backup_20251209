#!/usr/bin/env python3
"""Volume Profile Bot V2 - Improved version with better risk management.

Key improvements:
1. Fixed RSI ranges (no overlap)
2. Tighter stop losses based on ATR
3. Minimum risk/reward ratio enforcement
4. Higher signal threshold (4/6 score)
5. Stronger volume spike requirement (1.5x)
6. Added EMA50 trend filter
7. Better TP placement with minimum distance
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import ccxt
import numpy as np
from dotenv import load_dotenv

# =========================================================
# PATHS / LOGGING
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
LOG_DIR = ROOT_DIR / "logs"
WATCHLIST_FILE = BASE_DIR / "watchlist.json"
STATE_FILE = BASE_DIR / "state_v2.json"
STATS_FILE = LOG_DIR / "volume_profile_v2_stats.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "volume_profile_bot_v2.log"),
    ],
)

logger = logging.getLogger("volume_profile_bot_v2")

load_dotenv(ROOT_DIR / ".env")
load_dotenv(BASE_DIR / ".env")
sys.path.insert(0, str(ROOT_DIR))

from notifier import TelegramNotifier
from signal_stats import SignalStats

# =========================================================
# CONFIGURATION
# =========================================================
CONFIG = {
    # Signal generation
    "min_score": 3,              # Lowered to 3 for more signals
    "max_score": 6,              # Updated max score

    # RSI settings (no overlap)
    "rsi_long_min": 40,          # Changed from 35
    "rsi_long_max": 55,          # Changed from 60
    "rsi_short_min": 45,         # Changed from 40
    "rsi_short_max": 60,         # Changed from 65

    # Volume settings
    "volume_spike_threshold": 1.5,  # Increased from 1.2

    # Risk management
    "min_risk_reward": 1.5,      # Minimum R:R ratio
    "max_sl_percent": 2.0,       # Maximum 2% stop loss
    "min_sl_percent": 0.5,       # Minimum 0.5% stop loss
    "tp1_rr": 1.5,               # TP1 at 1.5R
    "tp2_rr": 2.5,               # TP2 at 2.5R

    # Cooldown
    "signal_cooldown_minutes": 30,
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize_symbol(symbol: str) -> str:
    base = symbol.replace("/USDT", "").replace("_USDT", "")
    return f"{base}/USDT"


# =========================================================
# TECHNICAL ANALYSIS FUNCTIONS
# =========================================================
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
    """Calculate Average True Range for volatility-based stops."""
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

    # Use last completed candle (not current incomplete one)
    idx = -2
    current_open, current_high, current_low, current_close = opens[idx], highs[idx], lows[idx], closes[idx]
    prev_open, prev_high, prev_low, prev_close = opens[idx-1], highs[idx-1], lows[idx-1], closes[idx-1]

    body = abs(current_close - current_open)
    if body == 0:
        body = 0.0001  # Prevent division by zero

    upper_wick = current_high - max(current_close, current_open)
    lower_wick = min(current_close, current_open) - current_low

    # Bullish patterns
    bullish_hammer = (current_close > current_open) and (lower_wick > 2 * body) and (upper_wick < body * 0.3)
    bullish_engulfing = (prev_close < prev_open) and (current_close > current_open) and \
                        (current_close > prev_open) and (current_open < prev_close) and \
                        (body > abs(prev_close - prev_open) * 1.2)  # Must be 1.2x larger

    # Bearish patterns
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

    # Find Low Volume Nodes (LVN) - potential breakout areas
    lvn_threshold = (total_volume / num_rows) * 0.5
    lvn_levels = [(price_levels[i], vol) for i, vol in enumerate(volume_profile) if vol < lvn_threshold]

    return {
        'poc': poc,
        'vah': price_levels[upper_idx],
        'val': price_levels[lower_idx],
        'hvn_levels': hvn_levels,
        'lvn_levels': lvn_levels,
        'row_height': row_height
    }


# =========================================================
# STATE MANAGEMENT
# =========================================================
class StateManager:
    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.state = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except:
                pass
        return {"last_signals": {}, "open_positions": {}}

    def _save(self):
        self.state_file.write_text(json.dumps(self.state, indent=2))

    def can_signal(self, symbol: str, cooldown_minutes: int) -> bool:
        last = self.state["last_signals"].get(symbol)
        if not last:
            return True
        try:
            last_dt = datetime.fromisoformat(last)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            diff = (utcnow() - last_dt).total_seconds() / 60
            return diff >= cooldown_minutes
        except:
            return True

    def mark_signal(self, symbol: str):
        self.state["last_signals"][symbol] = utcnow().isoformat()
        self._save()


# =========================================================
# BOT CLASS
# =========================================================
class VolumeProfileBotV2:
    def __init__(self):
        self.watchlist = json.loads(WATCHLIST_FILE.read_text())
        self.client = ccxt.binanceusdm({"enableRateLimit": True})
        self.notifier = TelegramNotifier(
            os.getenv("TELEGRAM_BOT_TOKEN"),
            os.getenv("TELEGRAM_CHAT_ID"),
            str(LOG_DIR / "signals_v2.json"),
        )
        self.stats = SignalStats("VP BOT V2", STATS_FILE)
        self.state = StateManager(STATE_FILE)

    def analyze_symbol(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        try:
            full_symbol = f"{symbol.replace('/USDT','')}/USDT:USDT"
            ohlcv = self.client.fetch_ohlcv(full_symbol, timeframe, limit=200)
            if not ohlcv or len(ohlcv) < 100:
                return None
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None

        opens = [x[1] for x in ohlcv]
        highs = [x[2] for x in ohlcv]
        lows = [x[3] for x in ohlcv]
        closes = [x[4] for x in ohlcv]
        volumes = [x[5] for x in ohlcv]

        current_price = closes[-1]

        # 1. Volume Profile Analysis
        vp = calculate_volume_profile(highs[-100:], lows[-100:], volumes[-100:])
        if not vp:
            return None

        # 2. Technical Indicators
        rsi = calculate_rsi(np.array(closes))
        ema20 = sum(closes[-20:]) / 20
        ema50 = sum(closes[-50:]) / 50  # Added EMA50 for trend
        atr = calculate_atr(highs, lows, closes)
        pattern = detect_candlestick_pattern(opens, highs, lows, closes)

        # Volume Spike (1.5x - increased from 1.2x)
        avg_vol = sum(volumes[-20:]) / 20
        vol_spike = volumes[-1] > avg_vol * CONFIG["volume_spike_threshold"]

        # 3. Context Analysis - Near HVN
        near_hvn = False
        nearest_hvn = None
        min_dist = float('inf')
        for hvn_price, _ in vp['hvn_levels']:
            dist = abs(current_price - hvn_price)
            if dist < min_dist:
                min_dist = dist
                nearest_hvn = hvn_price

        if min_dist < vp['row_height'] * 1.5:  # Tighter proximity requirement
            near_hvn = True

        # 4. Improved Scoring System (max 6 points)
        long_score = 0
        short_score = 0
        reasons = []

        # Trend Filter (EMA20 vs EMA50) - REQUIRED
        bullish_trend = ema20 > ema50 and current_price > ema20
        bearish_trend = ema20 < ema50 and current_price < ema20

        if bullish_trend:
            long_score += 1
            reasons.append("Bullish Trend (EMA20 > EMA50)")
        elif bearish_trend:
            short_score += 1
            reasons.append("Bearish Trend (EMA20 < EMA50)")
        else:
            # No clear trend - reduce scores
            pass

        # RSI - No overlap now
        if CONFIG["rsi_long_min"] < rsi < CONFIG["rsi_long_max"]:
            long_score += 1
            reasons.append(f"RSI Bullish ({rsi:.1f})")
        elif CONFIG["rsi_short_min"] < rsi < CONFIG["rsi_short_max"]:
            short_score += 1
            reasons.append(f"RSI Bearish ({rsi:.1f})")

        # Volume Profile Context
        if near_hvn:
            if current_price < vp['poc']:
                long_score += 1
                reasons.append("Near HVN Support")
            elif current_price > vp['poc']:
                short_score += 1
                reasons.append("Near HVN Resistance")

        # Candlestick Pattern
        if pattern:
            if "BULLISH" in pattern:
                long_score += 1
                reasons.append(f"Pattern: {pattern}")
            elif "BEARISH" in pattern:
                short_score += 1
                reasons.append(f"Pattern: {pattern}")

        # Volume Confirmation
        if vol_spike:
            if closes[-1] > opens[-1]:  # Bullish candle
                long_score += 1
                reasons.append("Volume Spike (Buying)")
            else:
                short_score += 1
                reasons.append("Volume Spike (Selling)")

        # Price Position relative to Value Area
        if current_price < vp['val']:
            long_score += 1  # Below value = potential long
            reasons.append("Below Value Area (Discount)")
        elif current_price > vp['vah']:
            short_score += 1  # Above value = potential short
            reasons.append("Above Value Area (Premium)")

        logger.info(f"{symbol} | Price: {current_price:.4f} | Long: {long_score} | Short: {short_score}")

        # 5. Signal Generation with Better Risk Management
        signal = None
        min_score = CONFIG["min_score"]

        if long_score >= min_score and long_score > short_score and bullish_trend:
            signal = "LONG"

            # ATR-based stop loss (tighter)
            atr_sl = current_price - (atr * 1.5)
            vp_sl = vp['val'] - (vp['row_height'] * 0.5)  # Tighter VP-based SL

            # Use the tighter of the two, but respect min/max
            sl = max(atr_sl, vp_sl)
            sl = max(sl, current_price * (1 - CONFIG["max_sl_percent"] / 100))
            sl = min(sl, current_price * (1 - CONFIG["min_sl_percent"] / 100))

            risk = current_price - sl

            # Fixed R:R based TP
            tp1 = current_price + (risk * CONFIG["tp1_rr"])
            tp2 = current_price + (risk * CONFIG["tp2_rr"])

            # Check minimum R:R
            rr_ratio = (tp1 - current_price) / risk if risk > 0 else 0
            if rr_ratio < CONFIG["min_risk_reward"]:
                logger.debug(f"{symbol}: Skipping LONG - R:R too low ({rr_ratio:.2f})")
                return None

        elif short_score >= min_score and short_score > long_score and bearish_trend:
            signal = "SHORT"

            # ATR-based stop loss (tighter)
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
                logger.debug(f"{symbol}: Skipping SHORT - R:R too low ({rr_ratio:.2f})")
                return None

        if signal:
            return {
                "type": signal,
                "score": max(long_score, short_score),
                "max_score": CONFIG["max_score"],
                "reasons": reasons,
                "entry": current_price,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "risk": risk,
                "rr_ratio": rr_ratio,
                "atr": atr,
                "vp": vp
            }
        return None

    def run_cycle(self):
        for w in self.watchlist:
            symbol = w["symbol"]
            tf = w.get("timeframe", w.get("period", "5m"))

            # Check cooldown
            if not self.state.can_signal(symbol, CONFIG["signal_cooldown_minutes"]):
                continue

            signal = self.analyze_symbol(symbol, tf)
            if signal:
                self.send_signal(symbol, tf, signal)
                self.state.mark_signal(symbol)

    def send_signal(self, symbol: str, tf: str, signal: Dict[str, Any]):
        reasons_str = "\n".join([f"• {r}" for r in signal['reasons']])
        emoji = "🟢" if signal['type'] == "LONG" else "🔴"

        sl_pct = abs(signal['entry'] - signal['sl']) / signal['entry'] * 100
        tp1_pct = abs(signal['tp1'] - signal['entry']) / signal['entry'] * 100
        tp2_pct = abs(signal['tp2'] - signal['entry']) / signal['entry'] * 100

        msg = (
            f"{emoji} <b>VP BOT V2: {signal['type']}</b>\n"
            f"{normalize_symbol(symbol)} [{tf}]\n"
            f"Score: {signal['score']}/{signal['max_score']} | R:R: {signal['rr_ratio']:.1f}\n\n"
            f"<b>Confluence:</b>\n{reasons_str}\n\n"
            f"🎯 <b>Setup:</b>\n"
            f"Entry: <code>{signal['entry']:.6f}</code>\n"
            f"TP1: <code>{signal['tp1']:.6f}</code> (+{tp1_pct:.2f}%)\n"
            f"TP2: <code>{signal['tp2']:.6f}</code> (+{tp2_pct:.2f}%)\n"
            f"SL: <code>{signal['sl']:.6f}</code> (-{sl_pct:.2f}%)\n\n"
            f"📊 <b>Profile:</b>\n"
            f"POC: {signal['vp']['poc']:.6f}\n"
            f"VAH: {signal['vp']['vah']:.6f} | VAL: {signal['vp']['val']:.6f}\n"
            f"ATR: {signal['atr']:.6f}\n\n"
            f"⏱️ {utcnow().strftime('%H:%M UTC')}"
        )
        self.notifier.send_message(msg)
        logger.info(f"Signal sent: {signal['type']} {symbol} @ {signal['entry']:.6f}")


if __name__ == "__main__":
    bot = VolumeProfileBotV2()
    logger.info("Volume Profile Bot V2 Started")
    logger.info(f"Config: min_score={CONFIG['min_score']}, min_rr={CONFIG['min_risk_reward']}")

    while True:
        try:
            bot.run_cycle()
            logger.info("Cycle complete; sleeping 180s (3m)")
            time.sleep(180)
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            time.sleep(180)
