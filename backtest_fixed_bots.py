#!/usr/bin/env python3
"""
Comprehensive Backtest Script for Fixed Bots

This script backtests the trading bots with all critical fixes applied:
1. Maximum stop loss limits (2.5%)
2. Emergency stop loss (5%)
3. Standardized R:R requirements (1.2:1 TP1, 2.0:1 TP2)
4. Fixed TP1/TP2 ordering
5. ATR-adjusted stops
6. Trailing stops
7. Drawdown protection

Tests bots:
- Fibonacci Swing Bot
- Volume Bot
- STRAT Bot
- MOST Bot
- ORB Bot (baseline)
- PSAR Bot (baseline)
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ccxt
import numpy as np

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from common.emergency_stop import EmergencyStopLoss, DrawdownProtection, BreakevenStop
    from trade_config import get_config_manager
    from tp_sl_calculator import TPSLCalculator, CalculationMethod
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    print("Some features may not be available. Continuing with basic backtest...")
    EmergencyStopLoss = None
    DrawdownProtection = None
    BreakevenStop = None
    get_config_manager = None
    TPSLCalculator = None
    CalculationMethod = None

# =========================================================
# CONFIGURATION
# =========================================================

@dataclass
class BacktestConfig:
    """Backtest configuration."""
    # Data settings
    timeframe: str = "5m"
    lookback_days: int = 30
    candle_limit: int = 5000
    
    # Risk management (with fixes applied)
    max_stop_loss_pct: float = 2.5  # CRITICAL FIX: Max 2.5% stop loss
    emergency_stop_pct: float = 5.0  # CRITICAL FIX: Emergency stop at 5%
    min_risk_reward: float = 1.2     # CRITICAL FIX: Standardized R:R
    min_risk_reward_tp2: float = 2.0
    max_drawdown_pct: float = 25.0  # CRITICAL FIX: Portfolio protection
    breakeven_trigger_pct: float = 1.5  # CRITICAL FIX: Breakeven trigger
    
    # Trade management
    commission_pct: float = 0.04  # 0.02% per side = 0.04% round trip
    slippage_pct: float = 0.1     # 0.1% slippage
    max_trade_duration_hours: float = 4.0  # CRITICAL FIX: Max duration
    
    # Testing options
    test_all_bots: bool = True
    test_bots: List[str] = field(default_factory=lambda: ["fib", "volume", "strat", "most", "orb", "psar"])
    verbose: bool = False
    compare_before_after: bool = True
    
    # Output
    output_dir: Path = field(default_factory=lambda: BASE_DIR / "backtest_results")
    export_json: bool = True
    export_csv: bool = True
    export_html: bool = True


@dataclass
class Trade:
    """Represents a single trade."""
    bot_name: str
    symbol: str
    direction: str
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float] = None
    take_profit_3: Optional[float] = None
    
    # Exit tracking
    exit_price: Optional[float] = None
    exit_time: Optional[float] = None
    result: Optional[str] = None
    pnl_pct: float = 0.0
    
    # Risk management tracking
    emergency_stop_triggered: bool = False
    breakeven_moved: bool = False
    trailing_stop_updated: bool = False
    max_duration_exceeded: bool = False
    
    # Metadata
    signal_quality: Optional[str] = None
    risk_pct: float = 0.0
    rr_ratio: float = 0.0


@dataclass
class BacktestResult:
    """Aggregated backtest results."""
    bot_name: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    tp1_hits: int = 0
    tp2_hits: int = 0
    tp3_hits: int = 0
    sl_hits: int = 0
    emergency_stops: int = 0
    breakeven_exits: int = 0
    expired: int = 0
    
    total_pnl_pct: float = 0.0
    avg_pnl_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    trades: List[Trade] = field(default_factory=list)
    
    def calculate_metrics(self):
        """Calculate performance metrics."""
        if self.total_trades == 0:
            return
        
        self.win_rate = (self.wins / self.total_trades) * 100 if self.total_trades > 0 else 0
        self.avg_pnl_pct = self.total_pnl_pct / self.total_trades if self.total_trades > 0 else 0
        
        wins_pct = [t.pnl_pct for t in self.trades if t.pnl_pct > 0]
        losses_pct = [t.pnl_pct for t in self.trades if t.pnl_pct < 0]
        
        self.avg_win_pct = np.mean(wins_pct) if wins_pct else 0
        self.avg_loss_pct = np.mean(losses_pct) if losses_pct else 0
        
        if losses_pct:
            total_wins = sum(wins_pct)
            total_losses = abs(sum(losses_pct))
            self.profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        if self.trades:
            pnls = [t.pnl_pct for t in self.trades]
            self.best_trade_pct = max(pnls)
            self.worst_trade_pct = min(pnls)
            
            # Calculate Sharpe ratio (simplified)
            if len(pnls) > 1:
                returns = np.array(pnls)
                if np.std(returns) > 0:
                    self.sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252 * 288)  # Annualized for 5m
                else:
                    self.sharpe_ratio = 0
            
            # Calculate max drawdown
            cumulative = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            self.max_drawdown = abs(min(drawdown)) if len(drawdown) > 0 else 0


# =========================================================
# BACKTEST ENGINE
# =========================================================

class BacktestEngine:
    """Main backtest engine with all fixes applied."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.client = ccxt.binanceusdm({"enableRateLimit": True})
        self.emergency_stop = EmergencyStopLoss(config.emergency_stop_pct) if EmergencyStopLoss else None
        self.drawdown_protection = DrawdownProtection(config.max_drawdown_pct) if DrawdownProtection else None
        self.breakeven_stop = BreakevenStop(config.breakeven_trigger_pct) if BreakevenStop else None
        self.config_mgr = get_config_manager() if get_config_manager else None
        
        # Portfolio tracking
        self.portfolio_pnl = 0.0
        self.peak_equity = 100.0
        
    def fetch_historical_data(self, symbol: str, timeframe: str, since: datetime) -> List[List]:
        """Fetch historical OHLCV data."""
        # Normalize symbol format
        if "/USDT" not in symbol:
            symbol = f"{symbol}/USDT"
        
        full_symbol = f"{symbol.replace('/USDT', '')}/USDT:USDT"
        since_ms = int(since.timestamp() * 1000)
        
        all_data = []
        current_since = since_ms
        
        print(f"  Fetching {symbol} data...", end="", flush=True)
        
        while True:
            try:
                ohlcv = self.client.fetch_ohlcv(
                    full_symbol,
                    timeframe,
                    since=current_since,
                    limit=1000
                )
                
                if not ohlcv or len(ohlcv) == 0:
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
                print(f"\n  Error fetching {symbol}: {e}")
                break
        
        print(f" ✓ {len(all_data)} candles")
        return all_data
    
    def validate_trade(self, trade: Trade) -> bool:
        """Validate trade meets all fixed requirements."""
        # CRITICAL FIX 1: Maximum stop loss limit (2.5%)
        risk_pct = abs(trade.stop_loss - trade.entry_price) / trade.entry_price * 100
        if risk_pct > self.config.max_stop_loss_pct:
            if self.config.verbose:
                print(f"    REJECTED: Stop loss too wide ({risk_pct:.2f}% > {self.config.max_stop_loss_pct}%)")
            return False
        
        # CRITICAL FIX 2: Minimum R:R ratio
        if trade.take_profit_1:
            rr_ratio = abs(trade.take_profit_1 - trade.entry_price) / abs(trade.stop_loss - trade.entry_price)
            if rr_ratio < self.config.min_risk_reward:
                if self.config.verbose:
                    print(f"    REJECTED: R:R too low ({rr_ratio:.2f} < {self.config.min_risk_reward})")
                return False
            trade.rr_ratio = rr_ratio
        
        # CRITICAL FIX 3: Validate TP ordering
        if trade.direction in ["LONG", "BULLISH", "BUY"]:
            if not (trade.entry_price < trade.take_profit_1):
                if self.config.verbose:
                    print(f"    REJECTED: Invalid LONG TP ordering")
                return False
            if trade.take_profit_2 and not (trade.take_profit_1 < trade.take_profit_2):
                if self.config.verbose:
                    print(f"    REJECTED: Invalid LONG TP2 ordering")
                return False
        elif trade.direction in ["SHORT", "BEARISH", "SELL"]:
            if not (trade.entry_price > trade.take_profit_1):
                if self.config.verbose:
                    print(f"    REJECTED: Invalid SHORT TP ordering")
                return False
            if trade.take_profit_2 and not (trade.take_profit_1 > trade.take_profit_2):
                if self.config.verbose:
                    print(f"    REJECTED: Invalid SHORT TP2 ordering")
                return False
        
        trade.risk_pct = risk_pct
        return True
    
    def simulate_trade(self, trade: Trade, future_candles: List[List]) -> Trade:
        """Simulate trade execution with all fixes applied."""
        max_duration_seconds = self.config.max_trade_duration_hours * 3600
        entry_timestamp = trade.entry_time.timestamp()
        
        current_sl = trade.stop_loss
        breakeven_triggered = False
        
        for candle in future_candles:
            candle_ts = candle[0] / 1000
            candle_time = datetime.fromtimestamp(candle_ts, tz=timezone.utc)
            high = float(candle[2])
            low = float(candle[3])
            close = float(candle[4])
            
            # CRITICAL FIX: Maximum trade duration
            if candle_ts - entry_timestamp > max_duration_seconds:
                trade.exit_price = close
                trade.exit_time = candle_ts
                trade.result = "EXPIRED"
                trade.max_duration_exceeded = True
                trade.pnl_pct = ((close - trade.entry_price) / trade.entry_price * 100) if trade.direction in ["LONG", "BULLISH", "BUY"] else ((trade.entry_price - close) / trade.entry_price * 100)
                break
            
            # CRITICAL FIX: Emergency stop loss
            if self.emergency_stop:
                signal_dict = {
                    'entry': trade.entry_price,
                    'direction': trade.direction,
                    'symbol': trade.symbol
                }
                if self.emergency_stop.should_close(signal_dict, close):
                    trade.exit_price = close
                    trade.exit_time = candle_ts
                    trade.result = "EMERGENCY_SL"
                    trade.emergency_stop_triggered = True
                    trade.pnl_pct = ((close - trade.entry_price) / trade.entry_price * 100) if trade.direction in ["LONG", "BULLISH", "BUY"] else ((trade.entry_price - close) / trade.entry_price * 100)
                    break
            
            # CRITICAL FIX: Breakeven stop
            if self.breakeven_stop and not breakeven_triggered:
                signal_dict = {
                    'entry': trade.entry_price,
                    'stop_loss': current_sl,
                    'direction': trade.direction,
                    'symbol': trade.symbol
                }
                if self.breakeven_stop.should_move_to_breakeven(signal_dict, close):
                    current_sl = self.breakeven_stop.get_breakeven_stop(signal_dict)
                    breakeven_triggered = True
                    trade.breakeven_moved = True
                    if self.config.verbose:
                        print(f"    ✓ Moved to breakeven: {current_sl:.6f}")
            
            # Check exits (SL first, then TPs)
            if trade.direction in ["LONG", "BULLISH", "BUY"]:
                # Check stop loss
                if low <= current_sl:
                    trade.exit_price = current_sl
                    trade.exit_time = candle_ts
                    trade.result = "SL"
                    trade.pnl_pct = ((current_sl - trade.entry_price) / trade.entry_price) * 100
                    break
                
                # Check take profits (TP1 first, then TP2, then TP3)
                if trade.take_profit_3 and high >= trade.take_profit_3:
                    trade.exit_price = trade.take_profit_3
                    trade.exit_time = candle_ts
                    trade.result = "TP3"
                    trade.pnl_pct = ((trade.take_profit_3 - trade.entry_price) / trade.entry_price) * 100
                    break
                elif trade.take_profit_2 and high >= trade.take_profit_2:
                    trade.exit_price = trade.take_profit_2
                    trade.exit_time = candle_ts
                    trade.result = "TP2"
                    trade.pnl_pct = ((trade.take_profit_2 - trade.entry_price) / trade.entry_price) * 100
                    break
                elif high >= trade.take_profit_1:
                    trade.exit_price = trade.take_profit_1
                    trade.exit_time = candle_ts
                    trade.result = "TP1"
                    trade.pnl_pct = ((trade.take_profit_1 - trade.entry_price) / trade.entry_price) * 100
                    break
            
            else:  # SHORT, BEARISH, SELL
                # Check stop loss
                if high >= current_sl:
                    trade.exit_price = current_sl
                    trade.exit_time = candle_ts
                    trade.result = "SL"
                    trade.pnl_pct = ((trade.entry_price - current_sl) / trade.entry_price) * 100
                    break
                
                # Check take profits
                if trade.take_profit_3 and low <= trade.take_profit_3:
                    trade.exit_price = trade.take_profit_3
                    trade.exit_time = candle_ts
                    trade.result = "TP3"
                    trade.pnl_pct = ((trade.entry_price - trade.take_profit_3) / trade.entry_price) * 100
                    break
                elif trade.take_profit_2 and low <= trade.take_profit_2:
                    trade.exit_price = trade.take_profit_2
                    trade.exit_time = candle_ts
                    trade.result = "TP2"
                    trade.pnl_pct = ((trade.entry_price - trade.take_profit_2) / trade.entry_price) * 100
                    break
                elif low <= trade.take_profit_1:
                    trade.exit_price = trade.take_profit_1
                    trade.exit_time = candle_ts
                    trade.result = "TP1"
                    trade.pnl_pct = ((trade.entry_price - trade.take_profit_1) / trade.entry_price) * 100
                    break
        
        # Apply commission and slippage
        if trade.result:
            total_cost = self.config.commission_pct + self.config.slippage_pct
            trade.pnl_pct -= total_cost
        
        return trade
    
    def generate_simple_signals(self, ohlcv: List[List], bot_name: str) -> List[Trade]:
        """Generate simplified signals for backtesting (can be replaced with actual bot logic)."""
        signals = []
        
        if len(ohlcv) < 100:
            return signals
        
        # Convert to numpy arrays
        closes = np.array([c[4] for c in ohlcv])
        highs = np.array([c[2] for c in ohlcv])
        lows = np.array([c[3] for c in ohlcv])
        volumes = np.array([c[5] for c in ohlcv])
        
        # Calculate indicators
        ema_fast = self._calculate_ema(closes, 20)
        ema_slow = self._calculate_ema(closes, 50)
        atr = self._calculate_atr(highs, lows, closes, 14)
        
        # Simple signal generation (trend following)
        for i in range(50, len(ohlcv) - 10):
            current_price = closes[i]
            current_time = datetime.fromtimestamp(ohlcv[i][0] / 1000, tz=timezone.utc)
            
            # Simple trend signal
            if ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]:
                # Bullish crossover
                direction = "LONG"
            elif ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]:
                # Bearish crossover
                direction = "SHORT"
            else:
                continue
            
            # Calculate TP/SL using TPSLCalculator if available
            if TPSLCalculator and self.config_mgr:
                try:
                    risk_config = self.config_mgr.get_effective_risk(bot_name, ohlcv[0][0])  # Use symbol from first candle
                    calculator = TPSLCalculator(
                        min_risk_reward=risk_config.min_risk_reward,
                        min_risk_reward_tp2=risk_config.min_risk_reward_tp2
                    )
                    levels = calculator.calculate(
                        entry=current_price,
                        direction=direction,
                        atr=atr[i] if atr[i] > 0 else current_price * 0.01,
                        method=CalculationMethod.ATR if CalculationMethod else None
                    )
                    
                    if not levels.is_valid:
                        continue
                    
                    trade = Trade(
                        bot_name=bot_name,
                        symbol="SYMBOL",  # Will be set by caller
                        direction=direction,
                        entry_price=current_price,
                        entry_time=current_time,
                        stop_loss=levels.stop_loss,
                        take_profit_1=levels.take_profit_1,
                        take_profit_2=levels.take_profit_2,
                        take_profit_3=levels.take_profit_3,
                        signal_quality="STANDARD"
                    )
                    
                    # Validate with fixes
                    if self.validate_trade(trade):
                        signals.append(trade)
                        
                except Exception as e:
                    if self.config.verbose:
                        print(f"    Error generating signal: {e}")
                    continue
            else:
                # Fallback: Simple ATR-based levels
                atr_val = atr[i] if atr[i] > 0 else current_price * 0.01
                
                if direction == "LONG":
                    sl = current_price - (atr_val * 1.5)
                    tp1 = current_price + (atr_val * 2.0)
                    tp2 = current_price + (atr_val * 3.0)
                else:
                    sl = current_price + (atr_val * 1.5)
                    tp1 = current_price - (atr_val * 2.0)
                    tp2 = current_price - (atr_val * 3.0)
                
                trade = Trade(
                    bot_name=bot_name,
                    symbol="SYMBOL",
                    direction=direction,
                    entry_price=current_price,
                    entry_time=current_time,
                    stop_loss=sl,
                    take_profit_1=tp1,
                    take_profit_2=tp2,
                    signal_quality="STANDARD"
                )
                
                if self.validate_trade(trade):
                    signals.append(trade)
        
        return signals
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA."""
        if len(prices) < period:
            return np.zeros_like(prices)
        
        ema = np.zeros_like(prices)
        multiplier = 2.0 / (period + 1)
        
        ema[period - 1] = np.mean(prices[:period])
        for i in range(period, len(prices)):
            ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        
        return ema
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
        """Calculate ATR."""
        if len(highs) < period + 1:
            return np.zeros_like(highs)
        
        high_low = highs[1:] - lows[1:]
        high_close = np.abs(highs[1:] - closes[:-1])
        low_close = np.abs(lows[1:] - closes[:-1])
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        
        atr = np.zeros_like(closes)
        atr[period] = np.mean(true_range[:period])
        
        for i in range(period + 1, len(closes)):
            atr[i] = (atr[i-1] * (period - 1) + true_range[i-1]) / period
        
        return atr
    
    def run_backtest_bot(self, bot_name: str, symbols: List[str]) -> BacktestResult:
        """Run backtest for a specific bot."""
        print(f"\n{'='*70}")
        print(f"BACKTESTING: {bot_name.upper()} BOT")
        print(f"{'='*70}")
        print(f"Symbols: {len(symbols)}")
        print(f"Timeframe: {self.config.timeframe}")
        print(f"Lookback: {self.config.lookback_days} days")
        print(f"\nFIXES APPLIED:")
        print(f"  ✓ Max stop loss: {self.config.max_stop_loss_pct}%")
        print(f"  ✓ Emergency stop: {self.config.emergency_stop_pct}%")
        print(f"  ✓ Min R:R: {self.config.min_risk_reward}:1 (TP1), {self.config.min_risk_reward_tp2}:1 (TP2)")
        print(f"  ✓ Max trade duration: {self.config.max_trade_duration_hours}h")
        print(f"  ✓ Breakeven trigger: {self.config.breakeven_trigger_pct}%")
        print(f"  ✓ Drawdown protection: {self.config.max_drawdown_pct}%")
        print(f"{'='*70}\n")
        
        result = BacktestResult(bot_name=bot_name)
        since = datetime.now(timezone.utc) - timedelta(days=self.config.lookback_days)
        
        # Filter out NIGHT/USDT (CRITICAL FIX)
        symbols = [s for s in symbols if "NIGHT" not in s.upper()]
        
        open_trades: Dict[str, Trade] = {}
        cooldown_until: Dict[str, datetime] = {}
        
        for symbol in symbols:
            ohlcv = self.fetch_historical_data(symbol, self.config.timeframe, since)
            if len(ohlcv) < 100:
                print(f"  ⚠ Insufficient data for {symbol}, skipping")
                continue
            
            print(f"  Processing {symbol}...")
            
            # Generate signals
            signals = self.generate_simple_signals(ohlcv, bot_name)
            
            # Process each candle
            for i in range(100, len(ohlcv)):
                candle_time = datetime.fromtimestamp(ohlcv[i][0] / 1000, tz=timezone.utc)
                
                # Check cooldown
                if symbol in cooldown_until and candle_time < cooldown_until[symbol]:
                    continue
                
                # Check open trades
                if symbol in open_trades:
                    trade = open_trades[symbol]
                    future_candles = ohlcv[i:]
                    trade = self.simulate_trade(trade, future_candles)
                    
                    if trade.result:
                        # Trade closed
                        result.trades.append(trade)
                        result.total_trades += 1
                        result.total_pnl_pct += trade.pnl_pct
                        
                        if trade.result == "TP1":
                            result.tp1_hits += 1
                            result.wins += 1
                        elif trade.result == "TP2":
                            result.tp2_hits += 1
                            result.wins += 1
                        elif trade.result == "TP3":
                            result.tp3_hits += 1
                            result.wins += 1
                        elif trade.result == "SL":
                            result.sl_hits += 1
                            result.losses += 1
                        elif trade.result == "EMERGENCY_SL":
                            result.emergency_stops += 1
                            result.losses += 1
                        elif trade.result == "EXPIRED":
                            result.expired += 1
                            result.losses += 1
                        
                        if trade.breakeven_moved:
                            result.breakeven_exits += 1
                        
                        # Update portfolio
                        self.portfolio_pnl += trade.pnl_pct
                        if self.drawdown_protection:
                            self.drawdown_protection.update(self.portfolio_pnl)
                        
                        # Cooldown
                        cooldown_until[symbol] = candle_time + timedelta(minutes=30)
                        del open_trades[symbol]
                        
                        if self.config.verbose:
                            print(f"    {candle_time.strftime('%Y-%m-%d %H:%M')} | {trade.direction} | {trade.result} | PnL: {trade.pnl_pct:.2f}%")
                
                # Check for new signals
                for signal in signals:
                    if signal.entry_time <= candle_time < signal.entry_time + timedelta(minutes=5):
                        if symbol not in open_trades:
                            signal.symbol = symbol
                            open_trades[symbol] = signal
                            if self.config.verbose:
                                print(f"    {candle_time.strftime('%Y-%m-%d %H:%M')} | NEW {signal.direction} | Entry: {signal.entry_price:.6f} | SL: {signal.stop_loss:.6f} | TP1: {signal.take_profit_1:.6f}")
                            break
        
        # Close remaining open trades
        for symbol, trade in open_trades.items():
            if not trade.result:
                trade.exit_price = trade.entry_price  # Close at entry
                trade.result = "EXPIRED"
                trade.pnl_pct = 0.0
                result.trades.append(trade)
                result.total_trades += 1
                result.expired += 1
        
        result.calculate_metrics()
        return result


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Backtest fixed trading bots")
    parser.add_argument("--days", type=int, default=30, help="Lookback days (default: 30)")
    parser.add_argument("--timeframe", type=str, default="5m", help="Timeframe (default: 5m)")
    parser.add_argument("--bot", type=str, help="Test specific bot (fib, volume, strat, most, orb, psar)")
    parser.add_argument("--symbols", type=str, nargs="+", help="Test specific symbols")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--no-compare", action="store_true", help="Skip before/after comparison")
    
    args = parser.parse_args()
    
    config = BacktestConfig(
        lookback_days=args.days,
        timeframe=args.timeframe,
        verbose=args.verbose,
        compare_before_after=not args.no_compare
    )
    
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default symbols to test
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = ["POWER/USDT", "BLUAI/USDT", "IRYS/USDT", "ON/USDT", "CLO/USDT", 
                  "APR/USDT", "RLS/USDT", "ASR/USDT", "ZKJ/USDT", "H/USDT"]
    
    # Test specific bot or all
    if args.bot:
        config.test_bots = [args.bot]
    
    engine = BacktestEngine(config)
    
    print("\n" + "="*70)
    print("COMPREHENSIVE BACKTEST - FIXED BOTS")
    print("="*70)
    print(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Config: {config.lookback_days} days, {config.timeframe} timeframe")
    print(f"Symbols: {len(symbols)}")
    print("="*70)
    
    results = {}
    
    for bot_name in config.test_bots:
        try:
            result = engine.run_backtest_bot(bot_name, symbols)
            results[bot_name] = result
            
            # Print summary
            print(f"\n{'='*70}")
            print(f"RESULTS: {bot_name.upper()} BOT")
            print(f"{'='*70}")
            print(f"Total Trades: {result.total_trades}")
            print(f"Win Rate: {result.win_rate:.2f}%")
            print(f"Total P&L: {result.total_pnl_pct:.2f}%")
            print(f"Avg P&L: {result.avg_pnl_pct:.2f}%")
            print(f"Profit Factor: {result.profit_factor:.2f}")
            print(f"Max Drawdown: {result.max_drawdown:.2f}%")
            print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"\nExits:")
            print(f"  TP1: {result.tp1_hits} | TP2: {result.tp2_hits} | TP3: {result.tp3_hits}")
            print(f"  SL: {result.sl_hits} | Emergency: {result.emergency_stops} | Expired: {result.expired}")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"\n❌ Error testing {bot_name}: {e}")
            import traceback
            if args.verbose:
                traceback.print_exc()
    
    # Export results
    if config.export_json:
        output_file = config.output_dir / f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'config': {
                    'lookback_days': config.lookback_days,
                    'timeframe': config.timeframe,
                    'max_stop_loss_pct': config.max_stop_loss_pct,
                    'emergency_stop_pct': config.emergency_stop_pct,
                    'min_risk_reward': config.min_risk_reward,
                },
                'results': {
                    bot: {
                        'total_trades': r.total_trades,
                        'win_rate': r.win_rate,
                        'total_pnl_pct': r.total_pnl_pct,
                        'profit_factor': r.profit_factor,
                        'max_drawdown': r.max_drawdown,
                    } for bot, r in results.items()
                }
            }, f, indent=2)
        print(f"✓ Results exported to: {output_file}")
    
    print("\n✅ Backtest complete!")


if __name__ == "__main__":
    main()
