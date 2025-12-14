#!/usr/bin/env python3
"""
Fibonacci Swing Trading Bot
Based on Fibonacci retracement + Swing confirmation strategy

Detects:
- Swing High/Low pivot points
- Fibonacci retracement levels (38.2%, 50%, 61.8%)
- Swing Low Confirmation (held for X candles)
- Entry signals: Premium, Confirmed, Standard

Entry Types:
- PREMIUM:   Fibonacci level + Swing confirmation (BEST!)
- CONFIRMED: Swing low confirmed + near bottom
- STANDARD:  At Fibonacci level in uptrend
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypedDict, TYPE_CHECKING, cast

import ccxt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

if TYPE_CHECKING:
    from notifier import TelegramNotifier
    from signal_stats import SignalStats
    from health_monitor import HealthMonitor, RateLimiter
    from tp_sl_calculator import TPSLCalculator
    from trade_config import get_config_manager
    from rate_limit_handler import RateLimitHandler
else:
    try:
        from notifier import TelegramNotifier
    except Exception:
        TelegramNotifier = None  # type: ignore[misc]
    try:
        from signal_stats import SignalStats
    except Exception:
        SignalStats = None  # type: ignore[misc]
    try:
        from health_monitor import HealthMonitor, RateLimiter
    except Exception:
        HealthMonitor = None  # type: ignore[misc]
        RateLimiter = None  # type: ignore[misc]
    try:
        from tp_sl_calculator import TPSLCalculator
    except Exception:
        TPSLCalculator = None  # type: ignore[misc]
    try:
        from trade_config import get_config_manager
    except Exception:
        get_config_manager = None  # type: ignore[misc]
    try:
        from rate_limit_handler import RateLimitHandler
    except Exception:
        RateLimitHandler = None  # type: ignore[misc]


class WatchItem(TypedDict, total=False):
    symbol: str
    timeframe: str
    cooldown_minutes: int

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).parent / "logs" / "fib_swing_bot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

STATS_FILE = Path(__file__).parent / "logs" / "fib_stats.json"
STATE_FILE = Path(__file__).parent / "logs" / "fib_state.json"


@dataclass
class FibSignal:
    """Fibonacci swing signal"""
    signal_id: str
    symbol: str
    timeframe: str
    direction: str
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    quality: str  # PREMIUM, CONFIRMED, STANDARD
    
    swing_high: float
    swing_low: float
    fib_level: str  # 38.2%, 50.0%, 61.8%
    fib_price: float
    
    swing_confirmed: bool
    bars_since_swing: int
    uptrend: bool
    high_volume: bool
    
    created_at: str
    exchange: str = "mexc"
    
    def as_dict(self) -> Dict:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "direction": self.direction,
            "entry": float(self.entry),
            "stop_loss": float(self.stop_loss),
            "tp1": float(self.tp1),
            "tp2": float(self.tp2),
            "tp3": float(self.tp3),
            "quality": self.quality,
            "swing_high": float(self.swing_high),
            "swing_low": float(self.swing_low),
            "fib_level": self.fib_level,
            "fib_price": float(self.fib_price),
            "swing_confirmed": bool(self.swing_confirmed),
            "bars_since_swing": int(self.bars_since_swing),
            "uptrend": bool(self.uptrend),
            "high_volume": bool(self.high_volume),
            "created_at": self.created_at,
            "exchange": self.exchange,
        }


class FibSwingDetector:
    """Detects swing points and calculates Fibonacci levels"""
    
    def __init__(self, lookback: int = 10):
        self.lookback = lookback
    
    @staticmethod
    def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        ema = np.zeros(len(prices))
        multiplier = 2 / (period + 1)
        ema[period-1] = np.mean(prices[:period])
        for i in range(period, len(prices)):
            ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        return ema
    
    def find_swing_high(self, highs: np.ndarray, index: int) -> bool:
        """Check if index is a swing high"""
        if index < self.lookback or index >= len(highs) - self.lookback:
            return False
        window = highs[index - self.lookback : index + self.lookback + 1]
        return highs[index] == max(window)
    
    def find_swing_low(self, lows: np.ndarray, index: int) -> bool:
        """Check if index is a swing low"""
        if index < self.lookback or index >= len(lows) - self.lookback:
            return False
        window = lows[index - self.lookback : index + self.lookback + 1]
        return lows[index] == min(window)
    
    def detect_swing_points(
        self, highs: np.ndarray, lows: np.ndarray
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """Detect all swing highs and lows"""
        swing_highs = []
        swing_lows = []
        
        for i in range(self.lookback, len(highs) - self.lookback):
            if self.find_swing_high(highs, i):
                swing_highs.append((i, highs[i]))
            if self.find_swing_low(lows, i):
                swing_lows.append((i, lows[i]))
        
        return swing_highs, swing_lows
    
    @staticmethod
    def check_swing_confirmation(
        lows: np.ndarray, 
        swing_low_idx: int, 
        swing_low_price: float, 
        confirmation_candles: int = 5
    ) -> Tuple[bool, int]:
        """Check if swing low has been confirmed (held for X candles)"""
        if swing_low_idx + confirmation_candles >= len(lows):
            bars_since = len(lows) - 1 - swing_low_idx
            return False, bars_since
        
        bars_since = len(lows) - 1 - swing_low_idx
        
        # Check if all candles stayed above swing low
        for i in range(1, min(confirmation_candles + 1, bars_since + 1)):
            if swing_low_idx + i < len(lows):
                if lows[swing_low_idx + i] < swing_low_price:
                    return False, bars_since
        
        return bars_since >= confirmation_candles, bars_since
    
    @staticmethod
    def calculate_fibonacci_levels(swing_high: float, swing_low: float, bullish: bool) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels for bullish or bearish swings"""
        diff = swing_high - swing_low
        levels: Dict[str, float] = {}
        ratios = {
            '0.0%': 0.0,
            '23.6%': 0.236,
            '38.2%': 0.382,
            '50.0%': 0.5,
            '61.8%': 0.618,
            '65.0%': 0.65,
            '78.6%': 0.786,
            '100%': 1.0,
        }
        for name, ratio in ratios.items():
            if bullish:
                price = swing_high - diff * ratio
            else:
                price = swing_low + diff * ratio
            levels[name] = price
        return levels


class FibSignalEvaluator:
    """Evaluates Fibonacci signals and determines entry conditions"""
    
    def __init__(self, confirmation_candles: int = 5):
        self.confirmation_candles = confirmation_candles
        self.detector = FibSwingDetector()
    
    def analyze(
        self, 
        symbol: str, 
        timeframe: str, 
        ohlcv: List
    ) -> Optional[FibSignal]:
        """Analyze market data and generate Fibonacci signal if conditions met"""
        
        if len(ohlcv) < 100:
            return None
        
        # Extract OHLCV
        closes = np.array([x[4] for x in ohlcv])
        highs = np.array([x[2] for x in ohlcv])
        lows = np.array([x[3] for x in ohlcv])
        volumes = np.array([x[5] for x in ohlcv])
        
        current_price = closes[-1]
        
        # Calculate EMAs
        ema_fast = self.detector.calculate_ema(closes, 20)
        ema_slow = self.detector.calculate_ema(closes, 50)
        
        uptrend = ema_fast[-1] > ema_slow[-1]
        
        # Volume analysis
        volume_ma = np.mean(volumes[-20:])
        high_volume = volumes[-1] > volume_ma
        
        # Detect swing points
        swing_highs, swing_lows = self.detector.detect_swing_points(highs, lows)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None

        signal_direction = None
        swing_high = swing_low = 0.0
        recent_idx = prev_idx = 0

        if uptrend:
            # Use previous swing low as anchor, recent swing high as reference
            swing_low_idx, swing_low_price = swing_lows[-2]
            swing_high_idx, swing_high_price = swing_highs[-1]
            if swing_high_idx <= swing_low_idx:
                return None
            swing_low = swing_low_price
            swing_high = swing_high_price
            signal_direction = "LONG"
        else:
            # Downtrend: previous swing high anchor, recent swing low reference
            swing_high_idx, swing_high_price = swing_highs[-2]
            swing_low_idx, swing_low_price = swing_lows[-1]
            if swing_low_idx <= swing_high_idx:
                return None
            swing_high = swing_high_price
            swing_low = swing_low_price
            signal_direction = "SHORT"

        fib_levels = self.detector.calculate_fibonacci_levels(swing_high, swing_low, bullish=uptrend)
        diff = abs(swing_high - swing_low)
        tolerance = diff * 0.02

        fib_names = {
            '38.2%': fib_levels['38.2%'],
            '50.0%': fib_levels['50.0%'],
            '61.8%': fib_levels['61.8%'],
            '65.0%': fib_levels['65.0%'],
        }

        fib_level_name = None
        fib_level_price = None
        for name in ['65.0%', '61.8%', '50.0%', '38.2%']:
            target = fib_names.get(name)
            if target is not None and abs(current_price - target) <= tolerance:
                fib_level_name = name
                fib_level_price = target
                break

        if fib_level_name is None:
            return None

        if uptrend:
            confirmation_idx, confirmation_price = swing_low_idx, swing_low
            is_confirmed, bars_since = self.detector.check_swing_confirmation(
                lows, confirmation_idx, confirmation_price, self.confirmation_candles
            )
            in_zone = current_price >= swing_low and current_price <= swing_high
        else:
            confirmation_idx, confirmation_price = swing_high_idx, swing_high
            is_confirmed, bars_since = self.detector.check_swing_confirmation(
                highs, confirmation_idx, confirmation_price, self.confirmation_candles
            )
            in_zone = current_price <= swing_high and current_price >= swing_low

        if not in_zone:
            return None

        if not uptrend and signal_direction == "LONG":
            return None
        if uptrend and signal_direction == "SHORT":
            return None

        quality = self._determine_quality(fib_level_name)
        if quality == "WEAK" and not high_volume:
            return None
        
        # Calculate trade setup
        direction = signal_direction
        if TPSLCalculator is not None:
            from tp_sl_calculator import CalculationMethod
            if get_config_manager is not None:
                config_mgr = get_config_manager()
                risk_config = config_mgr.get_effective_risk("fib_swing_bot", symbol)
                min_rr = risk_config.min_risk_reward
            else:
                min_rr = 2.0

            calculator = TPSLCalculator(min_risk_reward=min_rr)
            levels = calculator.calculate(
                entry=current_price,
                direction=direction,
                swing_high=swing_high if direction == "SHORT" else None,
                swing_low=swing_low if direction == "LONG" else None,
                method=CalculationMethod.STRUCTURE,
            )

            if levels.is_valid:
                stop_loss = levels.stop_loss
                tp1 = levels.take_profit_1
                tp2 = levels.take_profit_2
                tp3 = levels.take_profit_3 or (
                    current_price + (levels.take_profit_2 - current_price) * (1 if direction == "LONG" else -1)
                )
            else:
                stop_loss, tp1, tp2, tp3 = self._fallback_levels(direction, current_price, swing_high, swing_low)
        else:
            stop_loss, tp1, tp2, tp3 = self._fallback_levels(direction, current_price, swing_high, swing_low)
        
        # Create signal
        signal_id = f"{symbol}-{timeframe}-{datetime.utcnow().isoformat()}"
        
        signal = FibSignal(
            signal_id=signal_id,
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            entry=current_price,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            quality=quality,
            swing_high=swing_high,
            swing_low=swing_low,
            fib_level=fib_level_name or "N/A",
            fib_price=fib_level_price or 0.0,
            swing_confirmed=is_confirmed,
            bars_since_swing=bars_since,
            uptrend=uptrend,
            high_volume=high_volume,
            created_at=datetime.utcnow().isoformat(),
        )
        
        return signal

    def _determine_quality(self, fib_level: Optional[str]) -> str:
        if fib_level in ("61.8%", "65.0%"):
            return "PREMIUM"
        if fib_level == "50.0%":
            return "CONFIRMED"
        if fib_level == "38.2%":
            return "WEAK"
        return "STANDARD"

    def _fallback_levels(self, direction: str, entry: float, swing_high: float, swing_low: float):
        diff = abs(swing_high - swing_low)
        if direction == "LONG":
            stop_loss = swing_low - diff * 0.02
            risk = entry - stop_loss
            tp1 = entry + risk
            tp2 = entry + risk * 2
            tp3 = entry + risk * 3
        else:
            stop_loss = swing_high + diff * 0.02
            risk = stop_loss - entry
            tp1 = entry - risk
            tp2 = entry - risk * 2
            tp3 = entry - risk * 3
        return stop_loss, tp1, tp2, tp3


class SignalTracker:
    """Tracks open signals and monitors TP/SL hits"""
    
    def __init__(self, stats: Optional[SignalStats] = None):
        self.state_file = STATE_FILE
        self.state: Dict[str, Any] = self._load_state()
        self.stats = stats
    
    def _load_state(self) -> Dict[str, Any]:
        """Load state from file"""
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except json.JSONDecodeError:
                return {"open_signals": {}, "last_alerts": {}}
        return {"open_signals": {}, "last_alerts": {}}
    
    def _save_state(self) -> None:
        """Save state to file"""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self.state, indent=2))
    
    def can_alert(self, symbol: str, timeframe: str, cooldown_minutes: int) -> bool:
        """Check if enough time has passed since last alert"""
        key = f"{symbol}-{timeframe}"
        last_alerts = self.state.get("last_alerts", {})
        last_ts = last_alerts.get(key)
        
        if not last_ts:
            return True
        
        last_dt = datetime.fromisoformat(last_ts)
        return datetime.utcnow() - last_dt >= timedelta(minutes=cooldown_minutes)
    
    def mark_alert(self, symbol: str, timeframe: str) -> None:
        """Mark that an alert was sent"""
        key = f"{symbol}-{timeframe}"
        self.state.setdefault("last_alerts", {})[key] = datetime.utcnow().isoformat()
        self._save_state()
    
    def add_signal(self, signal: FibSignal) -> None:
        """Add signal to tracking"""
        signals = self.state.setdefault("open_signals", {})
        signals[signal.signal_id] = signal.as_dict()
        self._save_state()
        
        # Record in stats
        if self.stats:
            try:
                self.stats.record_open(
                    signal_id=signal.signal_id,
                    symbol=f"{signal.symbol}/USDT",
                    direction=signal.direction,
                    entry=signal.entry,
                    created_at=signal.created_at,
                    extra={
                        "timeframe": signal.timeframe,
                        "quality": signal.quality,
                        "fib_level": signal.fib_level,
                        "swing_confirmed": signal.swing_confirmed,
                    },
                )
            except Exception:
                pass
    
    def check_open_signals(self, notifier: Optional[TelegramNotifier]) -> None:
        """Check all open signals for TP/SL hits"""
        signals = self.state.get("open_signals", {})
        if not signals:
            return
        
        exchange = ccxt.mexc({"enableRateLimit": True, "options": {"defaultType": "swap"}})
        updated = False
        
        for signal_id, signal_data in list(signals.items()):
            if not isinstance(signal_data, dict):
                signals.pop(signal_id, None)
                updated = True
                continue
            symbol = signal_data.get("symbol")
            if not isinstance(symbol, str):
                signals.pop(signal_id, None)
                updated = True
                continue
            try:
                ticker = exchange.fetch_ticker(f"{symbol}/USDT:USDT")
                current_price = ticker.get("last") if isinstance(ticker, dict) else None
            except Exception as exc:
                logger.warning("Ticker fetch failed for %s: %s", symbol, exc)
                continue
            if not isinstance(current_price, (int, float)):
                continue
            entry_raw = signal_data.get("entry")
            tp1_raw = signal_data.get("tp1")
            tp2_raw = signal_data.get("tp2")
            tp3_raw = signal_data.get("tp3")
            sl_raw = signal_data.get("stop_loss")
            if not all(isinstance(v, (int, float, str)) for v in (entry_raw, tp1_raw, tp2_raw, tp3_raw, sl_raw)):
                signals.pop(signal_id, None)
                updated = True
                continue
            try:
                entry = float(entry_raw)  # type: ignore[arg-type]
                tp1 = float(tp1_raw)      # type: ignore[arg-type]
                tp2 = float(tp2_raw)      # type: ignore[arg-type]
                tp3 = float(tp3_raw)      # type: ignore[arg-type]
                sl = float(sl_raw)        # type: ignore[arg-type]
            except Exception:
                signals.pop(signal_id, None)
                updated = True
                continue
            
            # Check TP/SL hits (LONG only)
            hit_tp3 = current_price >= tp3
            hit_tp2 = current_price >= tp2
            hit_tp1 = current_price >= tp1
            hit_sl = current_price <= sl
            
            result = None
            if hit_tp3:
                result = "TP3"
            elif hit_tp2:
                result = "TP2"
            elif hit_tp1:
                result = "TP1"
            elif hit_sl:
                result = "SL"
            
            if result:
                # Record close
                summary_message = None
                if self.stats:
                    stats_record = self.stats.record_close(
                        signal_id,
                        exit_price=current_price,
                        result=result,
                    )
                    if stats_record:
                        summary_message = self.stats.build_summary_message(stats_record)
                    else:
                        self.stats.discard(signal_id)
                
                # Send message
                if summary_message:
                    message = summary_message
                else:
                    timeframe_val = signal_data.get("timeframe", "")
                    pnl = ((current_price - entry) / entry) * 100 if entry else 0.0
                    message = (
                        f"🎯 {symbol}/USDT {timeframe_val} {result} hit!\n"
                        f"Entry {entry:.6f} | Exit {current_price:.6f}\n"
                        f"P&L: {pnl:+.2f}%"
                    )
                
                if notifier:
                    notifier.send_message(message)
                
                logger.info("Signal %s closed with %s", signal_id, result)
                signals.pop(signal_id)
                updated = True
        
        if updated:
            self._save_state()


class FibSwingBot:
    """Main Fibonacci Swing Bot"""
    
    def __init__(self, watchlist_file: Path, interval: int = 300):
        self.watchlist_file = watchlist_file
        self.interval = interval
        self.watchlist: List[WatchItem] = self._load_watchlist()
        self.notifier = self._build_notifier()
        self.stats = SignalStats("Fib Swing Bot", STATS_FILE) if SignalStats else None
        self.tracker = SignalTracker(self.stats)
        self.evaluator = FibSignalEvaluator(confirmation_candles=5)
        self.health_monitor = self._build_health_monitor()
        self.rate_limiter = RateLimiter(calls_per_minute=30) if RateLimiter else None
        self.exchange = ccxt.mexc({"enableRateLimit": True, "options": {"defaultType": "swap"}})
        self.rate_limiter_handler = RateLimitHandler(base_delay=0.5, max_retries=5) if RateLimitHandler else None
        
        logger.info("Fib Swing Bot initialized with %d symbols", len(self.watchlist))
    
    def _load_watchlist(self) -> List[WatchItem]:
        """Load watchlist from JSON file"""
        if not self.watchlist_file.exists():
            logger.error("Watchlist file not found: %s", self.watchlist_file)
            return []
        
        try:
            data = json.loads(self.watchlist_file.read_text())
        except json.JSONDecodeError as e:
            logger.error("Failed to parse watchlist: %s", e)
            return []
        result: List[WatchItem] = []
        for row in data if isinstance(data, list) else []:
            if not isinstance(row, dict):
                continue
            symbol_val = row.get("symbol")
            if not isinstance(symbol_val, str):
                continue
            timeframe_val = row.get("timeframe", "5m")
            timeframe = timeframe_val if isinstance(timeframe_val, str) else "5m"
            cooldown_raw = row.get("cooldown_minutes", 30)
            try:
                cooldown = int(cooldown_raw)
            except Exception:
                cooldown = 30
            result.append({"symbol": symbol_val.upper(), "timeframe": timeframe, "cooldown_minutes": cooldown})
        return result
    
    def _build_notifier(self) -> Optional[TelegramNotifier]:
        """Build Telegram notifier"""
        from dotenv import load_dotenv
        load_dotenv()
        
        token = os.getenv("TELEGRAM_BOT_TOKEN_FIB")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if token and chat_id:
            return TelegramNotifier(token, chat_id)
        
        logger.warning("Telegram credentials not found for Fib bot")
        return None
    
    def _build_health_monitor(self) -> Optional[HealthMonitor]:
        """Build health monitor"""
        if self.notifier and HealthMonitor:
            return HealthMonitor(
                bot_name="Fib Swing Bot",
                notifier=self.notifier,
                heartbeat_interval=3600,  # 1 hour in seconds
            )
        return None
    
    def _format_signal_message(self, signal: FibSignal) -> str:
        """Format Fibonacci signal for Telegram"""
        quality_emoji = {
            "PREMIUM": "🟡⭐⭐⭐",
            "CONFIRMED": "🟢⭐⭐",
            "STANDARD": "🔵⭐"
        }
        
        emoji = quality_emoji.get(signal.quality, "🔵")
        
        risk_pct = abs((signal.stop_loss - signal.entry) / signal.entry) * 100
        tp1_pct = ((signal.tp1 - signal.entry) / signal.entry) * 100
        tp2_pct = ((signal.tp2 - signal.entry) / signal.entry) * 100
        tp3_pct = ((signal.tp3 - signal.entry) / signal.entry) * 100
        
        lines = [
            f"{emoji} <b>{signal.quality} FIBONACCI ENTRY</b> {emoji}",
            f"<b>{signal.symbol}/USDT</b> | {signal.timeframe}",
            "",
            "📐 <b>FIBONACCI SETUP:</b>",
            f"Swing High: <code>${signal.swing_high:.6f}</code>",
            f"Swing Low:  <code>${signal.swing_low:.6f}</code> {'✅' if signal.swing_confirmed else '⏳'} {'CONFIRMED' if signal.swing_confirmed else f'{signal.bars_since_swing} bars'}",
            "",
            "🎯 <b>ENTRY ZONE:</b>",
            f"Current Price: <code>${signal.entry:.6f}</code>",
        ]
        
        if signal.fib_level != "N/A":
            lines.append(f"Fib Level: <b>{signal.fib_level}</b> (${signal.fib_price:.6f}) 💎")
        
        lines.extend([
            "",
            "📈 <b>TRADE SETUP:</b>",
            f"Entry:     <code>${signal.entry:.6f}</code>",
            f"Stop Loss: <code>${signal.stop_loss:.6f}</code> ({-risk_pct:.2f}%)",
            f"TP1 (1R):  <code>${signal.tp1:.6f}</code> (+{tp1_pct:.2f}%)",
            f"TP2 (2R):  <code>${signal.tp2:.6f}</code> (+{tp2_pct:.2f}%)",
            f"TP3 (3R):  <code>${signal.tp3:.6f}</code> (+{tp3_pct:.2f}%)",
            "",
            "✅ <b>CONDITIONS MET:</b>",
        ])
        
        if signal.uptrend:
            lines.append("✅ Uptrend (EMA20 > EMA50)")
        if signal.fib_level != "N/A":
            lines.append(f"✅ At Fibonacci {signal.fib_level} level")
        if signal.swing_confirmed:
            lines.append(f"✅ Swing low confirmed ({signal.bars_since_swing} bars)")
        if signal.high_volume:
            lines.append("✅ High volume")
        
        lines.extend([
            "",
            f"Risk:Reward: 1:2 (TP2) | Quality: <b>{signal.quality}</b>",
            f"⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        ])

        # Historical TP/SL stats per symbol
        if self.stats is not None:
            symbol_key = f"{signal.symbol}/USDT"
            counts = self.stats.symbol_tp_sl_counts(symbol_key)
            tp1 = counts.get("TP1", 0)
            tp2 = counts.get("TP2", 0)
            sl = counts.get("SL", 0)
            total = tp1 + tp2 + sl
            if total > 0:
                win_rate = (tp1 + tp2) / total * 100.0
                lines.append("")
                lines.append(
                    f"📈 <b>History:</b> TP1 {tp1} | TP2 {tp2} | SL {sl} "
                    f"(Win rate: {win_rate:.1f}%)"
                )
        
        return "\n".join(lines)
    
    def run(self, run_once: bool = False) -> None:
        """Main bot loop"""
        logger.info("Starting Fib Swing Bot for %d symbols", len(self.watchlist))
        
        # Send startup message
        if self.health_monitor:
            self.health_monitor.send_startup_message()
        
        try:
            while True:
                try:
                    # Check open signals
                    self.tracker.check_open_signals(self.notifier)
                    
                    # Scan watchlist
                    for item in self.watchlist:
                        symbol_val = item.get("symbol") if isinstance(item, dict) else None
                        if not isinstance(symbol_val, str):
                            continue
                        symbol = symbol_val
                        timeframe_val = item.get("timeframe", "5m") if isinstance(item, dict) else "5m"
                        timeframe = timeframe_val if isinstance(timeframe_val, str) else "5m"
                        cooldown_raw = item.get("cooldown_minutes", 30) if isinstance(item, dict) else 30
                        try:
                            cooldown = int(cooldown_raw)
                        except Exception:
                            cooldown = 30
                        
                        # Check cooldown
                        if not self.tracker.can_alert(symbol, timeframe, cooldown):
                            logger.debug("Cooldown active for %s %s", symbol, timeframe)
                            continue
                        
                        # Rate limiting
                        if self.rate_limiter:
                            self.rate_limiter.wait_if_needed()
                        
                        # Fetch data
                        try:
                            if self.rate_limiter_handler:
                                ohlcv = self.rate_limiter_handler.execute(
                                    self.exchange.fetch_ohlcv,
                                    f"{symbol}/USDT:USDT",
                                    timeframe=timeframe,
                                    limit=200
                                )
                            else:
                                ohlcv = self.exchange.fetch_ohlcv(
                                    f"{symbol}/USDT:USDT", 
                                    timeframe, 
                                    limit=200
                                )
                        except Exception as exc:
                            logger.warning("Failed to fetch %s %s: %s", symbol, timeframe, exc)
                            continue
                        
                        # Analyze
                        signal = self.evaluator.analyze(symbol, timeframe, ohlcv)
                        
                        if signal:
                            # MAX OPEN SIGNALS LIMIT
                            MAX_OPEN_SIGNALS = 50
                            current_open = len(self.tracker.state.get("open_signals", {}))
                            if current_open >= MAX_OPEN_SIGNALS:
                                logger.info(
                                    "Max open signals limit reached (%d/%d). Skipping %s",
                                    current_open, MAX_OPEN_SIGNALS, symbol
                                )
                                continue
                            
                            # Send alert
                            message = self._format_signal_message(signal)
                            if self.notifier:
                                self.notifier.send_message(message)
                            
                            logger.info(
                                "%s %s signal: %s at %.6f (Fib %s)", 
                                signal.symbol, signal.quality, signal.direction, 
                                signal.entry, signal.fib_level
                            )
                            
                            # Track signal
                            self.tracker.add_signal(signal)
                            self.tracker.mark_alert(symbol, timeframe)
                        
                        time.sleep(1)  # Small delay between symbols
                    
                    # Record successful cycle
                    if self.health_monitor:
                        self.health_monitor.record_cycle()
                    
                    if run_once:
                        break
                    
                    logger.info("Cycle complete; sleeping %d seconds", self.interval)
                    time.sleep(self.interval)
                
                except Exception as exc:
                    logger.error("Error in cycle: %s", exc, exc_info=True)
                    if self.health_monitor:
                        self.health_monitor.record_error(str(exc))
                    if run_once:
                        raise
                    time.sleep(10)
        
        finally:
            # Send shutdown message
            if self.health_monitor:
                self.health_monitor.send_shutdown_message()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fibonacci Swing Bot")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds")
    args = parser.parse_args()
    
    watchlist_file = Path(__file__).parent / "fib_watchlist.json"
    bot = FibSwingBot(watchlist_file, interval=args.interval)
    bot.run(run_once=not args.loop)


if __name__ == "__main__":
    main()
