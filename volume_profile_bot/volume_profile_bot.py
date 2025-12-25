#!/usr/bin/env python3
from __future__ import annotations

import argparse
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    # Windows doesn't have fcntl - use a no-op fallback
    HAS_FCNTL = False
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import ccxt
import numpy as np
import numpy.typing as npt
from dotenv import load_dotenv

# =========================================================
# PATHS / LOGGING
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
LOG_DIR = ROOT_DIR / "logs"
WATCHLIST_FILE = BASE_DIR / "watchlist.json"
STATE_FILE = BASE_DIR / "vp_state.json"
STATS_FILE = LOG_DIR / "volume_profile_stats.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "volume_profile_bot.log"),
    ],
)

logger = logging.getLogger("volume_profile_bot")

load_dotenv(ROOT_DIR / ".env")
load_dotenv(BASE_DIR / ".env")
sys.path.insert(0, str(ROOT_DIR))

# =========================================================
# REQUIRED MODULES
# =========================================================
from notifier import TelegramNotifier
from signal_stats import SignalStats

# =========================================================
# STRATEGY CONFIGURATION (Optimized via backtesting)
# =========================================================
# Symbols that consistently underperform - skip these
BLACKLISTED_SYMBOLS = {"WET/USDT", "ID/USDT", "BAS/USDT", "AVNT/USDT", "FHE/USDT", "LUMIA/USDT"}

# Signal quality filters
MIN_SCORE = 4  # Require 4 confluence factors (was 3)
MAX_STOP_LOSS_PCT = 3.0  # Cap stop loss at 3% to prevent large losses
MIN_RISK_REWARD = 1.2  # Minimum risk:reward ratio

# =========================================================
# UTILS
# =========================================================
def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def normalize_symbol(symbol: str) -> str:
    base = symbol.replace("/USDT", "").replace("_USDT", "")
    return f"{base}/USDT"

# =========================================================
# TECHNICAL ANALYSIS FUNCTIONS (Ported from CLI)
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

def detect_candlestick_pattern(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Optional[str]:
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

def calculate_volume_profile(highs: List[float], lows: List[float], volumes: List[float], num_rows: int = 24) -> Dict[str, Any]:
    if not highs: return {}
    
    highest = max(highs)
    lowest = min(lows)
    price_range = highest - lowest
    if price_range == 0: return {}
    
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
# STATE MANAGEMENT
# =========================================================
class StateManager:
    """Thread-safe state management for signal tracking."""

    def __init__(self) -> None:
        self.state_lock = threading.Lock()
        self.state = self._load_state()

    def _empty_state(self) -> Dict[str, Any]:
        return {"open_signals": {}, "closed_signals": {}, "last_alerts": {}}

    def _load_state(self) -> Dict[str, Any]:
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, 'r') as f:
                    data = json.load(f)
                    # Ensure all required keys exist
                    for key in self._empty_state():
                        if key not in data:
                            data[key] = {}
                    return cast(Dict[str, Any], data)
            except Exception:
                return self._empty_state()
        return self._empty_state()

    def _save_state(self) -> None:
        with self.state_lock:
            temp_file = STATE_FILE.with_suffix('.tmp')
            try:
                with open(temp_file, 'w') as f:
                    if HAS_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    json.dump(self.state, f, indent=2)
                    if HAS_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                temp_file.replace(STATE_FILE)
            except Exception as e:
                logger.error(f"Failed to save state: {e}")

    def should_alert(self, symbol: str, timeframe: str, cooldown_mins: int = 30) -> bool:
        """Check cooldown to prevent duplicate alerts."""
        key = f"{symbol}-{timeframe}"
        last_ts = self.state.get("last_alerts", {}).get(key)
        if not last_ts:
            return True
        try:
            last = datetime.fromisoformat(last_ts)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - last) >= timedelta(minutes=cooldown_mins)
        except Exception:
            return True

    def add_signal(self, sig_id: str, signal_data: Dict[str, Any], symbol: str, timeframe: str) -> None:
        """Add a new signal to open_signals."""
        signals = self.state.setdefault("open_signals", {})
        signal_data["created_at"] = datetime.now(timezone.utc).isoformat()
        signals[sig_id] = signal_data

        last_alerts = self.state.setdefault("last_alerts", {})
        last_alerts[f"{symbol}-{timeframe}"] = datetime.now(timezone.utc).isoformat()

        self._save_state()
        logger.info(f"Signal added: {sig_id}")

    def close_signal(self, sig_id: str, exit_price: float, result: str) -> None:
        """Move signal from open to closed."""
        signals = self.state.get("open_signals", {})
        if sig_id not in signals:
            return

        closed = self.state.setdefault("closed_signals", {})
        closed_signal = signals[sig_id].copy()
        closed_signal["closed_at"] = datetime.now(timezone.utc).isoformat()
        closed_signal["exit_price"] = exit_price
        closed_signal["result"] = result

        # Calculate PnL
        entry = closed_signal.get("entry", 0)
        direction = closed_signal.get("type", "LONG")
        if entry > 0:
            if direction == "LONG":
                pnl = (exit_price - entry) / entry * 100
            else:
                pnl = (entry - exit_price) / entry * 100
            closed_signal["pnl_percent"] = pnl

        closed[sig_id] = closed_signal
        del signals[sig_id]
        self._save_state()
        logger.info(f"Signal closed: {sig_id} -> {result}")

    def get_open_signals(self) -> Dict[str, Any]:
        return self.state.get("open_signals", {})


# =========================================================
# BOT CLASS
# =========================================================
class VolumeProfileBot:
    def __init__(self):
        self.watchlist = json.loads(WATCHLIST_FILE.read_text())
        self.client = ccxt.binanceusdm({"enableRateLimit": True})
        self.notifier = TelegramNotifier(
            os.getenv("TELEGRAM_BOT_TOKEN"),
            os.getenv("TELEGRAM_CHAT_ID"),
            str(LOG_DIR / "signals.json"),
        )
        self.stats = SignalStats("VP BOT", STATS_FILE)
        self.state = StateManager()

    def analyze_symbol(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        # Fetch data
        try:
            full_symbol = f"{symbol.replace('/USDT','')}/USDT:USDT"
            ohlcv = self.client.fetch_ohlcv(full_symbol, timeframe, limit=200)
            if not ohlcv or len(ohlcv) < 50: return None
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None

        # Parse data
        opens = [x[1] for x in ohlcv]
        highs = [x[2] for x in ohlcv]
        lows = [x[3] for x in ohlcv]
        closes = [x[4] for x in ohlcv]
        volumes = [x[5] for x in ohlcv]

        current_price = closes[-1]
        
        # 1. Volume Profile Analysis (Last 100 candles)
        vp = calculate_volume_profile(highs[-100:], lows[-100:], volumes[-100:])
        if not vp: return None

        # 2. Technical Indicators
        rsi = calculate_rsi(np.array(closes))
        ema20 = sum(closes[-20:]) / 20
        pattern = detect_candlestick_pattern(opens, highs, lows, closes)
        
        # Volume Spike (1.2x average of last 20)
        avg_vol = sum(volumes[-20:]) / 20
        vol_spike = volumes[-1] > avg_vol * 1.2

        # 3. Context Analysis
        near_hvn = False
        min_dist = float('inf')
        for hvn_price, _ in vp['hvn_levels']:
            dist = abs(current_price - hvn_price)
            if dist < min_dist: min_dist = dist
        
        if min_dist < vp['row_height'] * 2:
            near_hvn = True

        # 4. Scoring System
        long_score = 0
        short_score = 0
        reasons = []

        # Trend (EMA)
        if current_price > ema20:
            long_score += 1
            # reasons.append("Price > EMA20")
        else:
            short_score += 1
            # reasons.append("Price < EMA20")

        # Momentum (RSI)
        if 35 < rsi < 60:
            long_score += 1
            reasons.append(f"RSI Bullish ({rsi:.1f})")
        elif 40 < rsi < 65:
            short_score += 1
            reasons.append(f"RSI Bearish ({rsi:.1f})")

        # Structure (HVN Bounce)
        if near_hvn:
            if current_price < vp['poc']: # Below POC + Near HVN = Support
                long_score += 1
                reasons.append("Bounce off Volume Support")
            elif current_price > vp['poc']: # Above POC + Near HVN = Resistance
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

        # Log status
        logger.info(f"{symbol} | Price: {current_price:.4f} | Long Score: {long_score} | Short Score: {short_score}")

        # 5. Signal Generation (Optimized via backtesting)
        signal = None
        sl = tp1 = tp2 = risk = reward = 0.0

        # IMPROVEMENT: Require MIN_SCORE (4) instead of 3
        if long_score >= MIN_SCORE and long_score > short_score:
            signal = "LONG"
            sl = vp['val'] - vp['row_height']

            # IMPROVEMENT: Cap stop loss at MAX_STOP_LOSS_PCT (3%)
            risk_pct = (current_price - sl) / current_price * 100
            if risk_pct > MAX_STOP_LOSS_PCT:
                sl = current_price * (1 - MAX_STOP_LOSS_PCT / 100)

            risk = current_price - sl
            if risk <= 0:
                risk = current_price * 0.005  # Fallback

            # Dynamic TP1
            if current_price < vp['poc'] - (vp['row_height'] / 2):
                tp1 = vp['poc']
            elif current_price < vp['vah'] - (vp['row_height'] / 2):
                tp1 = vp['vah']
            else:
                tp1 = current_price + (risk * 1.5)

            tp2 = current_price + (risk * 3)
            reward = tp1 - current_price

        elif short_score >= MIN_SCORE and short_score > long_score:
            signal = "SHORT"
            sl = vp['vah'] + vp['row_height']

            # IMPROVEMENT: Cap stop loss at MAX_STOP_LOSS_PCT (3%)
            risk_pct = (sl - current_price) / current_price * 100
            if risk_pct > MAX_STOP_LOSS_PCT:
                sl = current_price * (1 + MAX_STOP_LOSS_PCT / 100)

            risk = sl - current_price
            if risk <= 0:
                risk = current_price * 0.005  # Fallback

            # Dynamic TP1
            if current_price > vp['poc'] + (vp['row_height'] / 2):
                tp1 = vp['poc']
            elif current_price > vp['val'] + (vp['row_height'] / 2):
                tp1 = vp['val']
            else:
                tp1 = current_price - (risk * 1.5)

            tp2 = current_price - (risk * 3)
            reward = current_price - tp1

        if signal:
            # IMPROVEMENT: R:R filter - skip trades with poor risk:reward
            if risk > 0 and reward < risk * MIN_RISK_REWARD:
                logger.debug(f"{symbol} | Skipped: R:R {reward/risk:.2f} < {MIN_RISK_REWARD}")
                return None

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

    def run_cycle(self):
        # First check open signals for TP/SL
        self.check_open_signals()

        # Then scan for new signals
        for w in self.watchlist:
            symbol = w["symbol"]
            tf = w.get("period", "1m")  # Default to 1m

            # IMPROVEMENT: Skip blacklisted symbols (underperformers from backtesting)
            if symbol in BLACKLISTED_SYMBOLS:
                continue

            # Check cooldown before analyzing
            if not self.state.should_alert(symbol, tf, cooldown_mins=30):
                continue

            signal = self.analyze_symbol(symbol, tf)
            if signal:
                self.send_signal(symbol, tf, signal)

    def check_open_signals(self) -> None:
        """Check open signals for TP/SL hits."""
        signals = self.state.get_open_signals()
        if not signals:
            return

        for sig_id, payload in list(signals.items()):
            symbol = payload.get("symbol")
            direction = payload.get("type")

            try:
                full_symbol = f"{symbol.replace('/USDT','')}/USDT:USDT"
                ticker = self.client.fetch_ticker(full_symbol)
                price = ticker.get("last")
            except Exception as e:
                logger.debug(f"Error fetching {symbol}: {e}")
                continue

            if not price:
                continue

            tp1, tp2 = payload.get("tp1"), payload.get("tp2")
            sl = payload.get("sl")

            res = None
            if direction == "LONG":
                if price >= tp2:
                    res = "TP2"
                elif price >= tp1:
                    res = "TP1"
                elif price <= sl:
                    res = "SL"
            else:  # SHORT
                if price <= tp2:
                    res = "TP2"
                elif price <= tp1:
                    res = "TP1"
                elif price >= sl:
                    res = "SL"

            if res:
                entry = payload.get("entry", 0)
                if direction == "LONG":
                    pnl = (price - entry) / entry * 100
                else:
                    pnl = (entry - price) / entry * 100

                msg = f"🎯 {normalize_symbol(symbol)} VP {res} HIT!\n💰 PnL: {pnl:.2f}%"
                self.notifier.send_message(msg)
                self.state.close_signal(sig_id, price, res)

    def send_signal(self, symbol: str, tf: str, signal: Dict[str, Any]) -> None:
        reasons_str = "\n".join([f"• {r}" for r in signal['reasons']])
        emoji = "🟢" if signal['type'] == "LONG" else "🔴"

        # Create signal ID
        sig_id = f"{symbol}-{tf}-{utcnow().isoformat()}"

        msg = (
            f"{emoji} <b>SMART CONFLUENCE: {signal['type']}</b>\n"
            f"{normalize_symbol(symbol)} [{tf}] (Score: {signal['score']}/5)\n\n"
            f"<b>Why?</b>\n{reasons_str}\n\n"
            f"🎯 <b>Setup:</b>\n"
            f"Entry: {signal['entry']:.4f}\n"
            f"TP1 (POC): {signal['tp1']:.4f}\n"
            f"TP2 (2R): {signal['tp2']:.4f}\n"
            f"SL: {signal['sl']:.4f}\n\n"
            f"📊 <b>Profile:</b>\n"
            f"VAH: {signal['vp']['vah']:.4f} | VAL: {signal['vp']['val']:.4f}\n"
            f"⏱️ {utcnow().strftime('%H:%M UTC')}"
        )
        self.notifier.send_message(msg)

        # Save signal to state (excluding non-serializable vp data)
        signal_data = {
            "symbol": symbol,
            "timeframe": tf,
            "type": signal['type'],
            "score": signal['score'],
            "reasons": signal['reasons'],
            "entry": signal['entry'],
            "tp1": signal['tp1'],
            "tp2": signal['tp2'],
            "sl": signal['sl'],
            "vah": signal['vp']['vah'],
            "val": signal['vp']['val'],
            "poc": signal['vp']['poc'],
        }
        self.state.add_signal(sig_id, signal_data, symbol, tf)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    bot = VolumeProfileBot()
    logger.info("Smart Confluence Bot Started")
    
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
            time.sleep(10)