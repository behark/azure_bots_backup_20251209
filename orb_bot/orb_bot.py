#!/usr/bin/env python3
"""
Opening Range Breakout (ORB) Bot - Multi-Stage ORB Detection
Detects breakouts from Opening Range (5min, 15min, 30min, 60min)
Based on the LuxyBig Dynamic ORB strategy
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict, cast

import ccxt

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
STATE_FILE = BASE_DIR / "orb_state.json"
WATCHLIST_FILE = BASE_DIR / "orb_watchlist.json"
STATS_FILE = LOG_DIR / "orb_stats.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "orb_bot.log"),
    ],
)

logger = logging.getLogger("orb_bot")

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
    load_dotenv(BASE_DIR.parent / ".env")
except ImportError:
    pass

sys.path.append(str(BASE_DIR.parent))
try:
    from notifier import TelegramNotifier
    from signal_stats import SignalStats
    from health_monitor import HealthMonitor, RateLimiter
    from tp_sl_calculator import TPSLCalculator, CalculationMethod
    from trade_config import get_config_manager
    from rate_limit_handler import RateLimitHandler
except ImportError:
    TelegramNotifier = None
    SignalStats = None
    HealthMonitor = None
    RateLimiter = None
    TPSLCalculator = None
    CalculationMethod = None
    get_config_manager = None
    RateLimitHandler = None


class WatchItem(TypedDict):
    symbol: str
    period: str
    cooldown_minutes: int
    orb_stages: List[int]  # [5, 15, 30, 60] minutes


def load_watchlist() -> List[WatchItem]:
    """Load watchlist from JSON file."""
    if not WATCHLIST_FILE.exists():
        logger.error("Watchlist file missing: %s", WATCHLIST_FILE)
        return []
    
    try:
        data = json.loads(WATCHLIST_FILE.read_text())
    except json.JSONDecodeError as exc:
        logger.error("Invalid watchlist JSON: %s", exc)
        return []
    
    normalized: List[WatchItem] = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        symbol_val = item.get("symbol")
        if not isinstance(symbol_val, str):
            continue
        period_val = item.get("period", "5m")
        cooldown_raw = item.get("cooldown_minutes", 30)
        orb_stages = item.get("orb_stages", [5, 15, 30, 60])
        try:
            cooldown = int(cooldown_raw)
        except Exception:
            cooldown = 30
        normalized.append(
            {
                "symbol": symbol_val.upper(),
                "period": period_val if isinstance(period_val, str) else "5m",
                "cooldown_minutes": cooldown,
                "orb_stages": orb_stages if isinstance(orb_stages, list) else [5, 15, 30, 60],
            }
        )
    return normalized


def human_ts() -> str:
    """Return human-readable UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class ORBLevel:
    """Opening Range Breakout level for a specific stage."""
    stage_minutes: int  # 5, 15, 30, 60
    high: float
    low: float
    mid: float
    range_pct: float
    is_complete: bool = False
    completion_time: Optional[datetime] = None
    breakout_up: bool = False
    breakout_down: bool = False
    breakout_time: Optional[datetime] = None
    cycles_up: int = 0
    cycles_down: int = 0
    last_breakout_price: Optional[float] = None
    last_retest_time: Optional[datetime] = None
    bars_outside: int = 0
    
    @property
    def stage_name(self) -> str:
        return f"ORB{self.stage_minutes}"
    
    @property
    def range_dollars(self) -> float:
        return self.high - self.low
    
    def is_above_range(self, price: float, buffer_pct: float = 0.2) -> bool:
        """Check if price is above ORB high with buffer."""
        threshold = self.high * (1 + buffer_pct / 100)
        return price > threshold
    
    def is_below_range(self, price: float, buffer_pct: float = 0.2) -> bool:
        """Check if price is below ORB low with buffer."""
        threshold = self.low * (1 - buffer_pct / 100)
        return price < threshold
    
    def is_inside_range(self, price: float) -> bool:
        """Check if price is inside ORB range."""
        return self.low <= price <= self.high
    
    def reset_breakout(self) -> None:
        """Reset breakout tracking."""
        self.breakout_up = False
        self.breakout_down = False
        self.breakout_time = None
        self.last_breakout_price = None
        self.bars_outside = 0


@dataclass
class ORBSession:
    """Tracks opening range for a symbol in current session."""
    symbol: str
    session_start: datetime
    candles_collected: List[Dict[str, Any]] = field(default_factory=list)
    orb5: Optional[ORBLevel] = None
    orb15: Optional[ORBLevel] = None
    orb30: Optional[ORBLevel] = None
    orb60: Optional[ORBLevel] = None
    current_price: float = 0.0
    atr: float = 0.0
    volume_ma: float = 0.0
    last_signal_time: Optional[datetime] = None
    
    def calculate_orb_level(self, stage_minutes: int) -> Optional[ORBLevel]:
        """Calculate ORB level for given stage."""
        if len(self.candles_collected) < stage_minutes:
            return None
        
        # Take first N minutes of candles (assuming 1-minute candles)
        stage_candles = self.candles_collected[:stage_minutes]
        if not stage_candles:
            return None
        
        high = max(c['high'] for c in stage_candles)
        low = min(c['low'] for c in stage_candles)
        mid = (high + low) / 2
        range_pct = ((high - low) / low) * 100 if low > 0 else 0
        
        return ORBLevel(
            stage_minutes=stage_minutes,
            high=high,
            low=low,
            mid=mid,
            range_pct=range_pct,
            is_complete=True,
            completion_time=datetime.now(timezone.utc)
        )
    
    def update_orb_levels(self) -> None:
        """Update all ORB levels based on collected candles."""
        if self.orb5 is None and len(self.candles_collected) >= 5:
            self.orb5 = self.calculate_orb_level(5)
            if self.orb5:
                logger.info(f"{self.symbol} | ORB5 Complete: ${self.orb5.low:.2f}-${self.orb5.high:.2f} ({self.orb5.range_pct:.2f}%)")
        
        if self.orb15 is None and len(self.candles_collected) >= 15:
            self.orb15 = self.calculate_orb_level(15)
            if self.orb15:
                logger.info(f"{self.symbol} | ORB15 Complete: ${self.orb15.low:.2f}-${self.orb15.high:.2f} ({self.orb15.range_pct:.2f}%)")
        
        if self.orb30 is None and len(self.candles_collected) >= 30:
            self.orb30 = self.calculate_orb_level(30)
            if self.orb30:
                logger.info(f"{self.symbol} | ORB30 Complete: ${self.orb30.low:.2f}-${self.orb30.high:.2f} ({self.orb30.range_pct:.2f}%)")
        
        if self.orb60 is None and len(self.candles_collected) >= 60:
            self.orb60 = self.calculate_orb_level(60)
            if self.orb60:
                logger.info(f"{self.symbol} | ORB60 Complete: ${self.orb60.low:.2f}-${self.orb60.high:.2f} ({self.orb60.range_pct:.2f}%)")
    
    def get_active_orb_levels(self) -> List[ORBLevel]:
        """Get list of completed ORB levels."""
        levels = []
        if self.orb5 and self.orb5.is_complete:
            levels.append(self.orb5)
        if self.orb15 and self.orb15.is_complete:
            levels.append(self.orb15)
        if self.orb30 and self.orb30.is_complete:
            levels.append(self.orb30)
        if self.orb60 and self.orb60.is_complete:
            levels.append(self.orb60)
        return levels


class ORBDetector:
    """Opening Range Breakout detector with multi-stage support."""
    
    def __init__(
        self,
        exchange: ccxt.Exchange,
        notifier: Optional[Any] = None,
        stats: Optional[Any] = None,
        tpsl_calc: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.exchange = exchange
        self.notifier = notifier
        self.stats = stats
        self.tpsl_calc = tpsl_calc
        self.config = config
        
        # ORB Configuration - use defaults if config not available
        self.breakout_buffer_pct = 0.2
        self.retest_buffer_pct = 0.2
        self.min_retest_distance_pct = 0.5
        self.min_bars_outside = 2
        self.max_cycles = 6
        self.enable_volume_filter = False
        self.volume_multiplier = 1.5
        self.enable_trend_filter = False
        
        # Track sessions per symbol
        self.sessions: Dict[str, ORBSession] = {}
        
        logger.info("ORB Detector initialized with multi-stage support")
    
    def is_market_open(self) -> bool:
        """Check if market is in trading hours (9:30 AM - 4:00 PM EST)."""
        now = datetime.now(timezone.utc)
        # Convert to EST (UTC-5)
        est_offset = timedelta(hours=-5)
        est_time = now + est_offset
        
        hour = est_time.hour
        minute = est_time.minute
        weekday = est_time.weekday()
        
        # Monday=0 to Friday=4
        if weekday > 4:  # Weekend
            return False
        
        # 9:30 AM = 9*60+30 = 570 minutes
        # 4:00 PM = 16*60 = 960 minutes
        time_minutes = hour * 60 + minute
        return 570 <= time_minutes < 960
    
    def get_session_start(self) -> datetime:
        """Get current session's start time (00:00 UTC for crypto)."""
        now = datetime.now(timezone.utc)
        
        # For crypto: Use 00:00 UTC as daily session start (24/7 market)
        session_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # If we're before the session start, use previous day
        if now < session_start:
            session_start -= timedelta(days=1)
        
        return session_start
    
    def initialize_session(self, symbol: str) -> ORBSession:
        """Initialize a new ORB session for symbol."""
        session_start = self.get_session_start()
        session = ORBSession(symbol=symbol, session_start=session_start)
        self.sessions[symbol] = session
        logger.info(f"{symbol} | New ORB session initialized at {session_start.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        return session
    
    def fetch_candles(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch recent candles for symbol."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            candles = []
            for bar in ohlcv:
                candles.append({
                    'timestamp': bar[0],
                    'open': bar[1],
                    'high': bar[2],
                    'low': bar[3],
                    'close': bar[4],
                    'volume': bar[5],
                })
            return candles
        except Exception as e:
            logger.error(f"{symbol} | Error fetching candles: {e}")
            return []
    
    def calculate_atr(self, candles: List[Dict[str, Any]], period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(candles) < period + 1:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(candles)):
            high = candles[i]['high']
            low = candles[i]['low']
            prev_close = candles[i-1]['close']
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        if len(true_ranges) >= period:
            return sum(true_ranges[-period:]) / period
        return 0.0
    
    def calculate_volume_ma(self, candles: List[Dict[str, Any]], period: int = 20) -> float:
        """Calculate volume moving average."""
        if len(candles) < period:
            return 0.0
        recent_volumes = [c['volume'] for c in candles[-period:]]
        return sum(recent_volumes) / len(recent_volumes)
    
    def detect_breakout(
        self,
        symbol: str,
        orb_level: ORBLevel,
        current_price: float,
        current_volume: float,
        session: ORBSession
    ) -> Optional[Dict[str, Any]]:
        """Detect ORB breakout."""
        
        # Skip if max cycles reached
        if orb_level.cycles_up >= self.max_cycles and orb_level.cycles_down >= self.max_cycles:
            return None
        
        # Check if price is outside range
        is_above = orb_level.is_above_range(current_price, self.breakout_buffer_pct)
        is_below = orb_level.is_below_range(current_price, self.breakout_buffer_pct)
        
        # Track bars outside range
        if is_above or is_below:
            orb_level.bars_outside += 1
        else:
            # Back inside range - check for retest
            if orb_level.breakout_up and orb_level.bars_outside >= self.min_bars_outside:
                # Retest after upward breakout
                if orb_level.cycles_up < self.max_cycles:
                    orb_level.cycles_up += 1
                    orb_level.last_retest_time = datetime.now(timezone.utc)
                    logger.info(f"{symbol} | {orb_level.stage_name} RETEST UP (Cycle #{orb_level.cycles_up}) at ${current_price:.2f}")
                    return {
                        'type': 'retest_up',
                        'stage': orb_level.stage_name,
                        'price': current_price,
                        'orb_high': orb_level.high,
                        'orb_low': orb_level.low,
                        'cycle': orb_level.cycles_up,
                    }
            
            elif orb_level.breakout_down and orb_level.bars_outside >= self.min_bars_outside:
                # Retest after downward breakout
                if orb_level.cycles_down < self.max_cycles:
                    orb_level.cycles_down += 1
                    orb_level.last_retest_time = datetime.now(timezone.utc)
                    logger.info(f"{symbol} | {orb_level.stage_name} RETEST DOWN (Cycle #{orb_level.cycles_down}) at ${current_price:.2f}")
                    return {
                        'type': 'retest_down',
                        'stage': orb_level.stage_name,
                        'price': current_price,
                        'orb_high': orb_level.high,
                        'orb_low': orb_level.low,
                        'cycle': orb_level.cycles_down,
                    }
            
            orb_level.bars_outside = 0
            orb_level.reset_breakout()
        
        # Detect new breakout UP
        if is_above and not orb_level.breakout_up and orb_level.cycles_up < self.max_cycles:
            if orb_level.bars_outside >= self.min_bars_outside:
                # Volume filter
                if self.enable_volume_filter and session.volume_ma > 0:
                    if current_volume < session.volume_ma * self.volume_multiplier:
                        logger.debug(f"{symbol} | {orb_level.stage_name} breakout UP rejected: low volume")
                        return None
                
                orb_level.breakout_up = True
                orb_level.breakout_down = False
                orb_level.breakout_time = datetime.now(timezone.utc)
                orb_level.last_breakout_price = current_price
                orb_level.cycles_up += 1
                
                logger.info(f"{symbol} | {orb_level.stage_name} BREAKOUT UP (Cycle #{orb_level.cycles_up}) at ${current_price:.2f}")
                
                return {
                    'type': 'breakout_up',
                    'stage': orb_level.stage_name,
                    'price': current_price,
                    'orb_high': orb_level.high,
                    'orb_low': orb_level.low,
                    'range_pct': orb_level.range_pct,
                    'cycle': orb_level.cycles_up,
                    'atr': session.atr,
                }
        
        # Detect new breakout DOWN
        elif is_below and not orb_level.breakout_down and orb_level.cycles_down < self.max_cycles:
            if orb_level.bars_outside >= self.min_bars_outside:
                # Volume filter
                if self.enable_volume_filter and session.volume_ma > 0:
                    if current_volume < session.volume_ma * self.volume_multiplier:
                        logger.debug(f"{symbol} | {orb_level.stage_name} breakout DOWN rejected: low volume")
                        return None
                
                orb_level.breakout_down = True
                orb_level.breakout_up = False
                orb_level.breakout_time = datetime.now(timezone.utc)
                orb_level.last_breakout_price = current_price
                orb_level.cycles_down += 1
                
                logger.info(f"{symbol} | {orb_level.stage_name} BREAKOUT DOWN (Cycle #{orb_level.cycles_down}) at ${current_price:.2f}")
                
                return {
                    'type': 'breakout_down',
                    'stage': orb_level.stage_name,
                    'price': current_price,
                    'orb_high': orb_level.high,
                    'orb_low': orb_level.low,
                    'range_pct': orb_level.range_pct,
                    'cycle': orb_level.cycles_down,
                    'atr': session.atr,
                }
        
        return None
    
    def scan_symbol(self, symbol: str, period: str = '5m') -> Optional[Dict[str, Any]]:
        """Scan symbol for ORB signals."""
        
        # Check if market is open (disabled for crypto - trades 24/7)
        # For crypto, we track ORB from 00:00 UTC daily
        # if not self.is_market_open():
        #     return None
        
        # Get or create session
        if symbol not in self.sessions:
            session = self.initialize_session(symbol)
        else:
            session = self.sessions[symbol]
        
        # Check if need to reset session (new day)
        now = datetime.now(timezone.utc)
        session_age = now - session.session_start
        if session_age.total_seconds() > 24 * 3600:  # More than 24 hours
            session = self.initialize_session(symbol)
        
        # Fetch 1-minute candles since session start
        candles = self.fetch_candles(symbol, '1m', 100)
        if not candles:
            return None
        
        # Update session with candles from session start
        session_start_ms = int(session.session_start.timestamp() * 1000)
        session_candles = [c for c in candles if c['timestamp'] >= session_start_ms]
        
        if session_candles:
            session.candles_collected = session_candles
            session.current_price = session_candles[-1]['close']
            
            # Calculate indicators
            session.atr = self.calculate_atr(candles)
            session.volume_ma = self.calculate_volume_ma(candles)
            
            # Update ORB levels
            session.update_orb_levels()
            
            # Check for breakouts in all active ORB levels
            current_volume = session_candles[-1]['volume']
            
            for orb_level in session.get_active_orb_levels():
                signal = self.detect_breakout(symbol, orb_level, session.current_price, current_volume, session)
                if signal:
                    # Add session info
                    signal['symbol'] = symbol
                    signal['timestamp'] = human_ts()
                    signal['volume_ma'] = session.volume_ma
                    signal['current_volume'] = current_volume
                    
                    # Calculate TP/SL if available
                    if self.tpsl_calc:
                        try:
                            direction = 'LONG' if 'up' in signal['type'] else 'SHORT'
                            tp_sl = self.tpsl_calc.calculate(
                                entry=signal['price'],
                                direction=direction,
                                atr=session.atr,
                                method=CalculationMethod.ATR,
                                swing_high=orb_level.high,
                                swing_low=orb_level.low,
                            )
                            signal['tp'] = tp_sl.take_profit_1
                            signal['sl'] = tp_sl.stop_loss
                            signal['rr_ratio'] = tp_sl.risk_reward_1
                        except Exception as e:
                            logger.warning(f"{symbol} | TP/SL calculation failed: {e}")
                    
                    return signal
        
        return None


class ORBBot:
    """Main ORB Bot orchestrator."""
    
    def __init__(self, loop: bool = False, test_mode: bool = False):
        self.loop = loop
        self.test_mode = test_mode
        
        # Initialize exchange
        api_key = os.getenv("MEXC_API_KEY", "")
        api_secret = os.getenv("MEXC_API_SECRET", "")
        
        if not api_key or not api_secret:
            logger.warning("API credentials not found - running in read-only mode")
        
        self.exchange = ccxt.mexc({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        # Initialize components
        self.notifier = self._init_notifier()
        self.stats = SignalStats("orb_bot", STATS_FILE) if SignalStats else None
        self.health = HealthMonitor("orb_bot") if HealthMonitor else None
        self.tpsl_calc = TPSLCalculator() if TPSLCalculator else None
        
        # Load configuration
        config_manager = get_config_manager() if get_config_manager else None
        self.config = config_manager.get_bot_config("orb_bot") if config_manager else None
        
        # Initialize detector
        self.detector = ORBDetector(
            exchange=self.exchange,
            notifier=self.notifier,
            stats=self.stats,
            tpsl_calc=self.tpsl_calc,
            config=self.config.__dict__ if self.config else None
        )
        
        # Load watchlist
        self.watchlist = load_watchlist()
        logger.info(f"Loaded {len(self.watchlist)} symbols to watch")
        
        # State tracking
        self.state = self.load_state()
        
        logger.info("ORB Bot initialized successfully")
    
    def _init_notifier(self):
        """Initialize Telegram notifier with ORB bot token."""
        if TelegramNotifier is None:
            logger.warning("Telegram notifier unavailable")
            return None
        token = os.getenv("TELEGRAM_BOT_TOKEN_ORB") or os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            logger.warning("Telegram credentials missing")
            return None
        return TelegramNotifier(
            bot_token=token,
            chat_id=chat_id,
            signals_log_file=str(LOG_DIR / "orb_signals.json")
        )
    
    def load_state(self) -> Dict[str, Any]:
        """Load bot state from file."""
        if STATE_FILE.exists():
            try:
                return json.loads(STATE_FILE.read_text())
            except Exception as e:
                logger.error(f"Error loading state: {e}")
        return {"last_signals": {}, "session_count": 0}
    
    def save_state(self) -> None:
        """Save bot state to file."""
        try:
            STATE_FILE.write_text(json.dumps(self.state, indent=2))
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def process_signal(self, signal: Dict[str, Any]) -> None:
        """Process detected signal."""
        symbol = signal['symbol']
        signal_type = signal['type']
        stage = signal['stage']
        price = signal['price']
        
        # Create notification message with rich formatting
        if signal_type == 'breakout_up':
            direction = "BULLISH"
            entry = signal['price']
            tp = signal.get('tp')
            sl = signal.get('sl')
            rr = signal.get('rr_ratio', 'N/A')
            
            extra_data = {
                "orb_stage": stage,
                "orb_range": f"${signal['orb_low']:.4f} - ${signal['orb_high']:.4f}",
                "range_pct": f"{signal['range_pct']:.2f}%",
                "cycle": f"#{signal['cycle']}",
                "breakout_type": "Upward Breakout 🔼",
                "rr_ratio": str(rr) if rr != 'N/A' else None,
            }
            
            if self.stats and tp is not None and sl is not None:
                message = self.stats.build_initial_alert(
                    symbol=symbol,
                    direction=direction,
                    entry=entry,
                    tp1=tp,
                    tp2=None,
                    sl=sl,
                    extra_data=extra_data,
                )
            else:
                message = f"🔼 {stage} BREAKOUT UP\nSymbol: {symbol}\nPrice: ${price:.4f}\nORB Range: ${signal['orb_low']:.4f} - ${signal['orb_high']:.4f}\nRange: {signal['range_pct']:.2f}%\nCycle: #{signal['cycle']}"
        
        elif signal_type == 'breakout_down':
            direction = "BEARISH"
            entry = signal['price']
            tp = signal.get('tp')
            sl = signal.get('sl')
            rr = signal.get('rr_ratio', 'N/A')
            
            extra_data = {
                "orb_stage": stage,
                "orb_range": f"${signal['orb_low']:.4f} - ${signal['orb_high']:.4f}",
                "range_pct": f"{signal['range_pct']:.2f}%",
                "cycle": f"#{signal['cycle']}",
                "breakout_type": "Downward Breakout 🔽",
                "rr_ratio": str(rr) if rr != 'N/A' else None,
            }
            
            if self.stats and tp is not None and sl is not None:
                message = self.stats.build_initial_alert(
                    symbol=symbol,
                    direction=direction,
                    entry=entry,
                    tp1=tp,
                    tp2=None,
                    sl=sl,
                    extra_data=extra_data,
                )
            else:
                message = f"🔽 {stage} BREAKOUT DOWN\nSymbol: {symbol}\nPrice: ${price:.4f}\nORB Range: ${signal['orb_low']:.4f} - ${signal['orb_high']:.4f}\nRange: {signal['range_pct']:.2f}%\nCycle: #{signal['cycle']}"
        
        elif signal_type == 'retest_up':
            message = f"🔄 {stage} RETEST (Up Cycle #{signal['cycle']})\n"
            message += f"Symbol: {symbol}\n"
            message += f"Price: ${price:.4f}\n"
            message += f"Back to ORB High: ${signal['orb_high']:.4f}"
        
        elif signal_type == 'retest_down':
            message = f"🔄 {stage} RETEST (Down Cycle #{signal['cycle']})\n"
            message += f"Symbol: {symbol}\n"
            message += f"Price: ${price:.4f}\n"
            message += f"Back to ORB Low: ${signal['orb_low']:.4f}"
        
        else:
            message = f"ORB Signal: {signal_type}\n{symbol} @ ${price:.4f}"
        
        logger.info(f"Signal: {message}")
        
        # Send notification
        if self.notifier and not self.test_mode:
            try:
                self.notifier.send_message(message)
                logger.info(f"Telegram notification sent for {symbol}")
            except Exception as e:
                logger.error(f"Notification failed: {e}")
        
        # Update stats
        if self.stats:
            try:
                signal_id = f"ORB-{symbol.replace('/USDT', '')}-{signal_type[:4].upper()}-{int(time.time())}"
                direction = "LONG" if "up" in signal_type else "SHORT"
                
                # Get TP/SL from signal if available
                tp = signal.get('tp')
                sl = signal.get('sl')
                
                extra_data = {
                    'stage': stage,
                    'type': signal_type,
                    'tp': tp,
                    'sl': sl
                }
                
                self.stats.record_open(
                    signal_id=signal_id,
                    symbol=symbol,
                    direction=direction,
                    entry=price,
                    created_at=signal['timestamp'],
                    extra=extra_data
                )
            except Exception as e:
                logger.error(f"Stats update failed: {e}")
        
        # Update state
        self.state['last_signals'][symbol] = {
            'type': signal_type,
            'stage': stage,
            'price': price,
            'timestamp': signal['timestamp']
        }
        self.save_state()
    
    def monitor_open_signals(self) -> None:
        """Monitor open signals for TP/SL hits."""
        if not self.stats:
            return
        
        open_signals = self.stats.data.get("open", {})
        if not isinstance(open_signals, dict):
            return
        
        for signal_id, signal_data in list(open_signals.items()):
            if not isinstance(signal_data, dict):
                continue
            
            symbol = signal_data.get("symbol")
            if not isinstance(symbol, str):
                continue
            
            try:
                # Fetch current price
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker.get("last") if isinstance(ticker, dict) else None
                if current_price is None and isinstance(ticker, dict):
                    current_price = ticker.get("close")
                if not isinstance(current_price, (int, float)):
                    continue
                
                current_price = float(current_price)
                
                # Get signal details
                extra = signal_data.get("extra", {})
                if not isinstance(extra, dict):
                    extra = {}
                
                entry = float(signal_data.get("entry", 0))
                direction = str(signal_data.get("direction", ""))
                
                # Get TP/SL from extra
                tp = extra.get("tp")
                sl = extra.get("sl")
                
                if tp is None or sl is None:
                    continue
                
                tp = float(tp)
                sl = float(sl)
                
                # Check if TP or SL hit
                result = None
                if direction == "LONG":
                    if current_price >= tp:
                        result = "TP1"
                    elif current_price <= sl:
                        result = "SL"
                else:  # SHORT
                    if current_price <= tp:
                        result = "TP1"
                    elif current_price >= sl:
                        result = "SL"
                
                if result:
                    record = self.stats.record_close(
                        signal_id=signal_id,
                        exit_price=current_price,
                        result=result
                    )
                    
                    if record and self.notifier:
                        try:
                            summary_message = self.stats.build_summary_message(record)
                            self.notifier.send_message(summary_message)
                            logger.info(f"ORB signal {signal_id} closed with {result}")
                        except Exception as e:
                            logger.error(f"Failed to send close alert: {e}")
                    
            except Exception as e:
                logger.debug(f"Error monitoring {signal_id}: {e}")
                continue
    
    def run_scan(self) -> None:
        """Run single scan cycle."""
        logger.info(f"Scanning {len(self.watchlist)} symbols...")
        
        # Monitor existing signals
        self.monitor_open_signals()
        
        signals_found = 0
        for watch_item in self.watchlist:
            symbol = watch_item['symbol']
            period = watch_item.get('period', '5m')
            
            try:
                signal = self.detector.scan_symbol(symbol, period)
                if signal:
                    self.process_signal(signal)
                    signals_found += 1
                    
                    # Small delay between signals
                    time.sleep(1)
                
            except Exception as e:
                logger.error(f"{symbol} | Scan error: {e}")
            
            # Rate limiting
            time.sleep(0.5)
        
        if signals_found > 0:
            logger.info(f"Scan complete: {signals_found} signals found")
        else:
            logger.debug("Scan complete: no signals")
        
        # Update health
        if self.health:
            try:
                self.health.check_heartbeat()
            except Exception:
                pass
    
    def run(self) -> None:
        """Main run loop."""
        logger.info("ORB Bot starting...")
        
        # Send startup notification
        if self.notifier and not self.test_mode:
            try:
                startup_msg = f"🚀 ORB Bot Started\n\n"
                startup_msg += f"📊 Watching {len(self.watchlist)} symbols\n"
                startup_msg += f"⏱️ Scan interval: 1 minute\n"
                startup_msg += f"📈 ORB Stages: 5/15/30/60 min\n"
                startup_msg += f"✅ Multi-cycle tracking enabled\n\n"
                startup_msg += f"Bot is now monitoring for opening range breakouts!"
                self.notifier.send_message(startup_msg)
                logger.info("Startup notification sent")
            except Exception as e:
                logger.warning(f"Failed to send startup notification: {e}")
        
        if self.test_mode:
            logger.info("Running in TEST MODE - no notifications sent")
        
        try:
            if self.loop:
                logger.info("Entering continuous loop (1-minute intervals)")
                while True:
                    self.run_scan()
                    logger.info("Waiting 1 minute...")
                    time.sleep(60)  # 1 minute
            else:
                logger.info("Running single scan")
                self.run_scan()
        
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            if self.notifier:
                try:
                    self.notifier.send_message(f"⚠️ ORB Bot crashed: {e}")
                except Exception:
                    pass
            raise
        finally:
            self.save_state()
            logger.info("ORB Bot shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Opening Range Breakout (ORB) Bot")
    parser.add_argument('--loop', action='store_true', help='Run continuously')
    parser.add_argument('--test', action='store_true', help='Test mode (no notifications)')
    args = parser.parse_args()
    
    bot = ORBBot(loop=args.loop, test_mode=args.test)
    bot.run()


if __name__ == "__main__":
    main()
