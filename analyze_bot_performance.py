#!/usr/bin/env python3
"""
Comprehensive Bot Performance Analyzer

This script provides detailed analysis for each trading bot including:
- Win rate, profit factor, and P&L statistics
- Symbol-level performance breakdown
- Time-based analysis (daily, weekly, monthly)
- Risk-adjusted metrics (Sharpe ratio, max drawdown)
- Trade duration analysis
- Detailed trade history with charts

Usage:
    python analyze_bot_performance.py                    # Full analysis all bots
    python analyze_bot_performance.py --bot diy          # Specific bot
    python analyze_bot_performance.py --export html      # Export to HTML report
    python analyze_bot_performance.py --export json      # Export to JSON
    python analyze_bot_performance.py --export csv       # Export to CSV
    python analyze_bot_performance.py --detailed         # Include detailed trade list
    python analyze_bot_performance.py --summary          # Summary only
"""

import argparse
import json
import csv
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import statistics

BASE_DIR = Path(__file__).resolve().parent

# Bot stats file mappings
BOT_CONFIGS = {
    "diy_bot": {
        "name": "DIY Bot",
        "stats_file": BASE_DIR / "diy_bot" / "logs" / "diy_stats.json",
        "description": "Multi-indicator confluence strategy (30+ indicators)",
    },
    "fib_swing_bot": {
        "name": "Fibonacci Swing Bot",
        "stats_file": BASE_DIR / "fib_swing_bot" / "logs" / "fib_stats.json",
        "description": "Fibonacci retracement levels with swing confirmation",
    },
    "funding_bot": {
        "name": "Funding Rate Bot",
        "stats_file": BASE_DIR / "funding_bot" / "logs" / "funding_stats.json",
        "description": "Funding rate arbitrage strategy",
    },
    "harmonic_bot": {
        "name": "Harmonic Bot",
        "stats_file": BASE_DIR / "harmonic_bot" / "logs" / "harmonic_stats.json",
        "description": "Harmonic pattern detection (Gartley, Butterfly, Bat, Crab)",
    },
    "harmonic_pattern_bot": {
        "name": "Harmonic Pattern Bot",
        "stats_file": BASE_DIR / "harmonic_pattern_bot" / "logs" / "harmonic_pattern_stats.json",
        "description": "Advanced harmonic pattern recognition",
    },
    "liquidation_bot": {
        "name": "Liquidation Bot",
        "stats_file": BASE_DIR / "liquidation_bot" / "logs" / "liquidation_stats.json",
        "description": "Liquidation cluster analysis",
    },
    "most_bot": {
        "name": "MOST Bot",
        "stats_file": BASE_DIR / "most_bot" / "logs" / "most_stats.json",
        "description": "Moving Stop Loss (MOST) indicator with EMA",
    },
    "mtf_bot": {
        "name": "MTF Bot",
        "stats_file": BASE_DIR / "mtf_bot" / "logs" / "mtf_stats.json",
        "description": "Multi-timeframe analysis strategy",
    },
    "orb_bot": {
        "name": "ORB Bot",
        "stats_file": BASE_DIR / "orb_bot" / "logs" / "orb_stats.json",
        "description": "Opening Range Breakout strategy",
    },
    "psar_bot": {
        "name": "PSAR Bot",
        "stats_file": BASE_DIR / "psar_bot" / "logs" / "psar_stats.json",
        "description": "Parabolic SAR with trailing stop",
    },
    "strat_bot": {
        "name": "STRAT Bot",
        "stats_file": BASE_DIR / "strat_bot" / "logs" / "strat_stats.json",
        "description": "Rob Smith's STRAT strategy",
    },
    "volume_bot": {
        "name": "Volume Bot",
        "stats_file": BASE_DIR / "volume_bot" / "logs" / "volume_stats.json",
        "description": "Volume profile and volume node analysis",
    },
    "volume_profile_bot": {
        "name": "Volume Profile Bot",
        "stats_file": BASE_DIR / "volume_profile_bot" / "logs" / "volume_profile_stats.json",
        "description": "Volume profile POC and value area",
    },
}


@dataclass
class Trade:
    """Individual trade record."""
    signal_id: str
    symbol: str
    direction: str
    entry: float
    exit: float
    result: str
    pnl_pct: float
    created_at: str
    closed_at: str
    duration_hours: Optional[float] = None
    extra: Dict[str, Any] = None


@dataclass
class SymbolStats:
    """Statistics for a single symbol."""
    symbol: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    best_trade: float
    worst_trade: float
    tp1_hits: int
    tp2_hits: int
    tp3_hits: int
    sl_hits: int


@dataclass
class BotPerformance:
    """Comprehensive performance metrics for a bot."""
    bot_name: str
    description: str
    
    # Trade counts
    total_trades: int
    open_trades: int
    closed_trades: int
    wins: int
    losses: int
    
    # Win rates
    win_rate: float
    tp1_hit_rate: float
    tp2_hit_rate: float
    tp3_hit_rate: float
    
    # P&L metrics
    total_pnl: float
    avg_pnl: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    
    # Risk metrics
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_duration: Optional[int]
    
    # Trade analysis
    avg_trade_duration_hours: float
    median_trade_duration_hours: float
    
    # Result breakdown
    tp1_hits: int
    tp2_hits: int
    tp3_hits: int
    sl_hits: int
    expired: int
    
    # Symbol analysis
    symbols_traded: int
    most_traded_symbol: str
    best_symbol: Tuple[str, float]
    worst_symbol: Tuple[str, float]
    symbol_stats: List[SymbolStats]
    
    # Time analysis
    daily_pnl: Dict[str, float]
    weekly_pnl: Dict[str, float]
    monthly_pnl: Dict[str, float]
    
    # Detailed trades
    recent_trades: List[Trade]
    best_trades: List[Trade]
    worst_trades: List[Trade]


def load_json_safe(path: Path) -> Dict[str, Any]:
    """Safely load JSON file."""
    if not path.exists():
        return {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}", file=sys.stderr)
        return {}


def normalize_direction(direction: Optional[str]) -> str:
    """Normalize direction to LONG/SHORT."""
    if not direction:
        return "UNKNOWN"
    d = str(direction).upper()
    if d in ["LONG", "BULLISH", "BUY"]:
        return "LONG"
    elif d in ["SHORT", "BEARISH", "SELL"]:
        return "SHORT"
    return d


def parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """Parse datetime string safely."""
    if not dt_str:
        return None
    try:
        # Try ISO format first
        return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    except:
        try:
            # Try common formats
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
                try:
                    return datetime.strptime(dt_str, fmt)
                except:
                    continue
        except:
            pass
    return None


def calculate_trade_duration(created_at: str, closed_at: str) -> Optional[float]:
    """Calculate trade duration in hours."""
    start = parse_datetime(created_at)
    end = parse_datetime(closed_at)
    if start and end:
        return (end - start).total_seconds() / 3600
    return None


def calculate_sharpe_ratio(pnls: List[float], risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio."""
    if len(pnls) < 2:
        return 0.0
    
    avg_return = statistics.mean(pnls)
    std_return = statistics.stdev(pnls)
    
    if std_return == 0:
        return 0.0
    
    return (avg_return - risk_free_rate) / std_return


def calculate_max_drawdown(pnls: List[float]) -> Tuple[float, Optional[int]]:
    """Calculate maximum drawdown and its duration."""
    if not pnls:
        return 0.0, None
    
    cumulative = [0]
    for pnl in pnls:
        cumulative.append(cumulative[-1] + pnl)
    
    max_dd = 0.0
    max_dd_duration = 0
    peak = cumulative[0]
    peak_idx = 0
    
    for i, value in enumerate(cumulative):
        if value > peak:
            peak = value
            peak_idx = i
        else:
            dd = peak - value
            if dd > max_dd:
                max_dd = dd
                max_dd_duration = i - peak_idx
    
    return max_dd, max_dd_duration if max_dd_duration > 0 else None


def extract_trades_from_stats(stats_data: Dict[str, Any]) -> List[Trade]:
    """Extract trade records from stats file."""
    trades = []
    
    # Check "history" key (most common)
    history = stats_data.get("history", [])
    if isinstance(history, list):
        for item in history:
            if not isinstance(item, dict):
                continue
            
            signal_id = item.get("id", "unknown")
            created_at = item.get("created_at", "")
            closed_at = item.get("closed_at", "")
            
            duration = calculate_trade_duration(created_at, closed_at)
            
            trades.append(Trade(
                signal_id=signal_id,
                symbol=item.get("symbol", "UNKNOWN"),
                direction=normalize_direction(item.get("direction")),
                entry=float(item.get("entry", 0) or 0),
                exit=float(item.get("exit", 0) or 0),
                result=item.get("result", "UNKNOWN"),
                pnl_pct=float(item.get("pnl_pct", 0) or 0),
                created_at=created_at,
                closed_at=closed_at,
                duration_hours=duration,
                extra=item.get("extra", {}),
            ))
    
    return trades


def calculate_symbol_stats(trades: List[Trade], symbol: str) -> SymbolStats:
    """Calculate statistics for a specific symbol."""
    symbol_trades = [t for t in trades if t.symbol == symbol]
    
    if not symbol_trades:
        return SymbolStats(
            symbol=symbol,
            total_trades=0,
            wins=0,
            losses=0,
            win_rate=0.0,
            total_pnl=0.0,
            avg_pnl=0.0,
            best_trade=0.0,
            worst_trade=0.0,
            tp1_hits=0,
            tp2_hits=0,
            tp3_hits=0,
            sl_hits=0,
        )
    
    pnls = [t.pnl_pct for t in symbol_trades]
    tp1_hits = len([t for t in symbol_trades if "TP1" in t.result.upper()])
    tp2_hits = len([t for t in symbol_trades if "TP2" in t.result.upper()])
    tp3_hits = len([t for t in symbol_trades if "TP3" in t.result.upper()])
    sl_hits = len([t for t in symbol_trades if "SL" in t.result.upper()])
    
    wins = tp1_hits + tp2_hits + tp3_hits
    losses = sl_hits
    
    return SymbolStats(
        symbol=symbol,
        total_trades=len(symbol_trades),
        wins=wins,
        losses=losses,
        win_rate=(wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0,
        total_pnl=sum(pnls),
        avg_pnl=statistics.mean(pnls) if pnls else 0.0,
        best_trade=max(pnls) if pnls else 0.0,
        worst_trade=min(pnls) if pnls else 0.0,
        tp1_hits=tp1_hits,
        tp2_hits=tp2_hits,
        tp3_hits=tp3_hits,
        sl_hits=sl_hits,
    )


def analyze_bot_performance(bot_key: str, config: Dict[str, Any]) -> Optional[BotPerformance]:
    """Analyze performance for a single bot."""
    stats_file = config["stats_file"]
    
    if not stats_file.exists():
        print(f"Warning: Stats file not found for {config['name']}: {stats_file}", file=sys.stderr)
        return None
    
    stats_data = load_json_safe(stats_file)
    
    # Extract trades
    closed_trades = extract_trades_from_stats(stats_data)
    
    # Count open trades
    open_data = stats_data.get("open", {})
    open_count = len(open_data) if isinstance(open_data, dict) else 0
    
    if not closed_trades:
        print(f"Info: No closed trades found for {config['name']}", file=sys.stderr)
        # Return minimal stats
        return BotPerformance(
            bot_name=config["name"],
            description=config["description"],
            total_trades=open_count,
            open_trades=open_count,
            closed_trades=0,
            wins=0,
            losses=0,
            win_rate=0.0,
            tp1_hit_rate=0.0,
            tp2_hit_rate=0.0,
            tp3_hit_rate=0.0,
            total_pnl=0.0,
            avg_pnl=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            best_trade=0.0,
            worst_trade=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=None,
            avg_trade_duration_hours=0.0,
            median_trade_duration_hours=0.0,
            tp1_hits=0,
            tp2_hits=0,
            tp3_hits=0,
            sl_hits=0,
            expired=0,
            symbols_traded=0,
            most_traded_symbol="N/A",
            best_symbol=("N/A", 0.0),
            worst_symbol=("N/A", 0.0),
            symbol_stats=[],
            daily_pnl={},
            weekly_pnl={},
            monthly_pnl={},
            recent_trades=[],
            best_trades=[],
            worst_trades=[],
        )
    
    # Calculate basic metrics
    pnls = [t.pnl_pct for t in closed_trades]
    win_pnls = [p for p in pnls if p > 0]
    loss_pnls = [p for p in pnls if p < 0]
    
    # Count results
    tp1_hits = len([t for t in closed_trades if "TP1" in t.result.upper()])
    tp2_hits = len([t for t in closed_trades if "TP2" in t.result.upper()])
    tp3_hits = len([t for t in closed_trades if "TP3" in t.result.upper()])
    sl_hits = len([t for t in closed_trades if "SL" in t.result.upper()])
    expired = len([t for t in closed_trades if "EXPIRED" in t.result.upper() or "TIMEOUT" in t.result.upper()])
    
    wins = tp1_hits + tp2_hits + tp3_hits
    losses = sl_hits
    
    # P&L metrics
    total_pnl = sum(pnls)
    avg_pnl = statistics.mean(pnls) if pnls else 0.0
    avg_win = statistics.mean(win_pnls) if win_pnls else 0.0
    avg_loss = statistics.mean(loss_pnls) if loss_pnls else 0.0
    best_trade = max(pnls) if pnls else 0.0
    worst_trade = min(pnls) if pnls else 0.0
    
    # Profit factor
    gross_profit = sum(win_pnls) if win_pnls else 0.0
    gross_loss = abs(sum(loss_pnls)) if loss_pnls else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)
    
    # Risk metrics
    sharpe = calculate_sharpe_ratio(pnls)
    max_dd, max_dd_duration = calculate_max_drawdown(pnls)
    
    # Trade duration
    durations = [t.duration_hours for t in closed_trades if t.duration_hours is not None]
    avg_duration = statistics.mean(durations) if durations else 0.0
    median_duration = statistics.median(durations) if durations else 0.0
    
    # Symbol analysis
    symbols = list(set(t.symbol for t in closed_trades))
    symbol_count = defaultdict(int)
    for t in closed_trades:
        symbol_count[t.symbol] += 1
    
    most_traded = max(symbol_count.items(), key=lambda x: x[1])[0] if symbol_count else "N/A"
    
    symbol_stats_list = [calculate_symbol_stats(closed_trades, sym) for sym in symbols]
    symbol_stats_list.sort(key=lambda x: x.total_pnl, reverse=True)
    
    best_symbol = (symbol_stats_list[0].symbol, symbol_stats_list[0].total_pnl) if symbol_stats_list else ("N/A", 0.0)
    worst_symbol = (symbol_stats_list[-1].symbol, symbol_stats_list[-1].total_pnl) if symbol_stats_list else ("N/A", 0.0)
    
    # Time-based analysis
    daily_pnl = defaultdict(float)
    weekly_pnl = defaultdict(float)
    monthly_pnl = defaultdict(float)
    
    for trade in closed_trades:
        dt = parse_datetime(trade.closed_at)
        if dt:
            daily_pnl[dt.strftime("%Y-%m-%d")] += trade.pnl_pct
            weekly_pnl[dt.strftime("%Y-W%U")] += trade.pnl_pct
            monthly_pnl[dt.strftime("%Y-%m")] += trade.pnl_pct
    
    # Sort trades for reporting
    recent_trades = sorted(closed_trades, key=lambda x: x.closed_at, reverse=True)[:20]
    best_trades = sorted(closed_trades, key=lambda x: x.pnl_pct, reverse=True)[:10]
    worst_trades = sorted(closed_trades, key=lambda x: x.pnl_pct)[:10]
    
    # Win rates
    total_closed = len(closed_trades)
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0
    tp1_rate = (tp1_hits / total_closed * 100) if total_closed > 0 else 0.0
    tp2_rate = (tp2_hits / total_closed * 100) if total_closed > 0 else 0.0
    tp3_rate = (tp3_hits / total_closed * 100) if total_closed > 0 else 0.0
    
    return BotPerformance(
        bot_name=config["name"],
        description=config["description"],
        total_trades=total_closed + open_count,
        open_trades=open_count,
        closed_trades=total_closed,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        tp1_hit_rate=tp1_rate,
        tp2_hit_rate=tp2_rate,
        tp3_hit_rate=tp3_rate,
        total_pnl=total_pnl,
        avg_pnl=avg_pnl,
        avg_win=avg_win,
        avg_loss=avg_loss,
        best_trade=best_trade,
        worst_trade=worst_trade,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_duration,
        avg_trade_duration_hours=avg_duration,
        median_trade_duration_hours=median_duration,
        tp1_hits=tp1_hits,
        tp2_hits=tp2_hits,
        tp3_hits=tp3_hits,
        sl_hits=sl_hits,
        expired=expired,
        symbols_traded=len(symbols),
        most_traded_symbol=most_traded,
        best_symbol=best_symbol,
        worst_symbol=worst_symbol,
        symbol_stats=symbol_stats_list,
        daily_pnl=dict(daily_pnl),
        weekly_pnl=dict(weekly_pnl),
        monthly_pnl=dict(monthly_pnl),
        recent_trades=recent_trades,
        best_trades=best_trades,
        worst_trades=worst_trades,
    )


def print_bot_performance(perf: BotPerformance, detailed: bool = False) -> None:
    """Print formatted performance report for a bot."""
    print(f"\n{'='*80}")
    print(f"  {perf.bot_name.upper()}")
    print(f"  {perf.description}")
    print(f"{'='*80}")
    
    # Overview
    print(f"\n{'  OVERVIEW'}")
    print(f"  {'-'*76}")
    print(f"  Total Trades:           {perf.total_trades:>8}")
    print(f"  Open Trades:            {perf.open_trades:>8}")
    print(f"  Closed Trades:          {perf.closed_trades:>8}")
    
    if perf.closed_trades == 0:
        print(f"\n  No closed trades yet.")
        return
    
    # Results breakdown
    print(f"\n{'  RESULTS BREAKDOWN'}")
    print(f"  {'-'*76}")
    print(f"  Wins:                   {perf.wins:>8} ({perf.win_rate:>6.1f}%)")
    print(f"  Losses:                 {perf.losses:>8}")
    print(f"  TP1 Hits:               {perf.tp1_hits:>8} ({perf.tp1_hit_rate:>6.1f}%)")
    print(f"  TP2 Hits:               {perf.tp2_hits:>8} ({perf.tp2_hit_rate:>6.1f}%)")
    if perf.tp3_hits > 0:
        print(f"  TP3 Hits:               {perf.tp3_hits:>8} ({perf.tp3_hit_rate:>6.1f}%)")
    print(f"  SL Hits:                {perf.sl_hits:>8}")
    if perf.expired > 0:
        print(f"  Expired:                {perf.expired:>8}")
    
    # P&L Performance
    print(f"\n{'  P&L PERFORMANCE'}")
    print(f"  {'-'*76}")
    print(f"  Total P&L:              {perf.total_pnl:>+8.2f}%")
    print(f"  Average P&L:            {perf.avg_pnl:>+8.2f}%")
    print(f"  Average Win:            {perf.avg_win:>+8.2f}%")
    print(f"  Average Loss:           {perf.avg_loss:>+8.2f}%")
    print(f"  Best Trade:             {perf.best_trade:>+8.2f}%")
    print(f"  Worst Trade:            {perf.worst_trade:>+8.2f}%")
    
    # Risk Metrics
    print(f"\n{'  RISK METRICS'}")
    print(f"  {'-'*76}")
    pf_str = f"{perf.profit_factor:.2f}" if perf.profit_factor != float('inf') else "∞"
    print(f"  Profit Factor:          {pf_str:>8}")
    print(f"  Sharpe Ratio:           {perf.sharpe_ratio:>8.2f}")
    print(f"  Max Drawdown:           {perf.max_drawdown:>8.2f}%")
    if perf.max_drawdown_duration:
        print(f"  Max DD Duration:        {perf.max_drawdown_duration:>8} trades")
    
    # Trade Duration
    print(f"\n{'  TRADE DURATION'}")
    print(f"  {'-'*76}")
    print(f"  Average Duration:       {perf.avg_trade_duration_hours:>8.1f} hours")
    print(f"  Median Duration:        {perf.median_trade_duration_hours:>8.1f} hours")
    
    # Symbol Analysis
    print(f"\n{'  SYMBOL ANALYSIS'}")
    print(f"  {'-'*76}")
    print(f"  Symbols Traded:         {perf.symbols_traded:>8}")
    print(f"  Most Traded:            {perf.most_traded_symbol:>12}")
    print(f"  Best Symbol:            {perf.best_symbol[0]:>12} ({perf.best_symbol[1]:>+7.2f}%)")
    print(f"  Worst Symbol:           {perf.worst_symbol[0]:>12} ({perf.worst_symbol[1]:>+7.2f}%)")
    
    # Top symbols by P&L
    if perf.symbol_stats:
        print(f"\n{'  TOP SYMBOLS BY P&L'}")
        print(f"  {'-'*76}")
        print(f"  {'Symbol':<12} {'Trades':>7} {'Win%':>7} {'Total P&L':>10} {'Avg P&L':>10}")
        print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*10} {'-'*10}")
        for sym_stat in perf.symbol_stats[:10]:
            print(f"  {sym_stat.symbol:<12} {sym_stat.total_trades:>7} "
                  f"{sym_stat.win_rate:>6.1f}% {sym_stat.total_pnl:>+9.2f}% "
                  f"{sym_stat.avg_pnl:>+9.2f}%")
    
    # Time-based performance
    if perf.monthly_pnl:
        print(f"\n{'  MONTHLY P&L'}")
        print(f"  {'-'*76}")
        sorted_months = sorted(perf.monthly_pnl.items(), reverse=True)[:6]
        for month, pnl in sorted_months:
            print(f"  {month}:              {pnl:>+8.2f}%")
    
    # Recent trades
    if detailed and perf.recent_trades:
        print(f"\n{'  RECENT TRADES (Last 10)'}")
        print(f"  {'-'*76}")
        print(f"  {'Symbol':<12} {'Dir':<6} {'Result':<8} {'P&L':>8} {'Duration':>10}")
        print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*8} {'-'*10}")
        for trade in perf.recent_trades[:10]:
            dur_str = f"{trade.duration_hours:.1f}h" if trade.duration_hours else "N/A"
            print(f"  {trade.symbol:<12} {trade.direction:<6} {trade.result:<8} "
                  f"{trade.pnl_pct:>+7.2f}% {dur_str:>10}")
    
    # Best trades
    if detailed and perf.best_trades:
        print(f"\n{'  BEST TRADES (Top 5)'}")
        print(f"  {'-'*76}")
        print(f"  {'Symbol':<12} {'Dir':<6} {'Result':<8} {'P&L':>8} {'Date':<12}")
        print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*8} {'-'*12}")
        for trade in perf.best_trades[:5]:
            dt = parse_datetime(trade.closed_at)
            date_str = dt.strftime("%Y-%m-%d") if dt else "N/A"
            print(f"  {trade.symbol:<12} {trade.direction:<6} {trade.result:<8} "
                  f"{trade.pnl_pct:>+7.2f}% {date_str:<12}")
    
    # Worst trades
    if detailed and perf.worst_trades:
        print(f"\n{'  WORST TRADES (Bottom 5)'}")
        print(f"  {'-'*76}")
        print(f"  {'Symbol':<12} {'Dir':<6} {'Result':<8} {'P&L':>8} {'Date':<12}")
        print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*8} {'-'*12}")
        for trade in perf.worst_trades[:5]:
            dt = parse_datetime(trade.closed_at)
            date_str = dt.strftime("%Y-%m-%d") if dt else "N/A"
            print(f"  {trade.symbol:<12} {trade.direction:<6} {trade.result:<8} "
                  f"{trade.pnl_pct:>+7.2f}% {date_str:<12}")


def print_overall_summary(all_performance: List[BotPerformance]) -> None:
    """Print overall summary across all bots."""
    print(f"\n{'='*80}")
    print(f"  OVERALL PORTFOLIO SUMMARY")
    print(f"{'='*80}")
    
    # Filter bots with closed trades
    active_bots = [p for p in all_performance if p.closed_trades > 0]
    
    if not active_bots:
        print(f"\n  No closed trades across any bots yet.")
        return
    
    # Aggregate metrics
    total_trades = sum(p.total_trades for p in all_performance)
    total_open = sum(p.open_trades for p in all_performance)
    total_closed = sum(p.closed_trades for p in all_performance)
    total_wins = sum(p.wins for p in active_bots)
    total_losses = sum(p.losses for p in active_bots)
    total_pnl = sum(p.total_pnl for p in active_bots)
    
    overall_win_rate = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0.0
    
    print(f"\n{'  AGGREGATE STATISTICS'}")
    print(f"  {'-'*76}")
    print(f"  Active Bots:            {len(active_bots):>8} / {len(all_performance)}")
    print(f"  Total Trades:           {total_trades:>8}")
    print(f"  Open Trades:            {total_open:>8}")
    print(f"  Closed Trades:          {total_closed:>8}")
    print(f"  Total Wins:             {total_wins:>8}")
    print(f"  Total Losses:           {total_losses:>8}")
    print(f"  Overall Win Rate:       {overall_win_rate:>7.1f}%")
    print(f"  Total Portfolio P&L:    {total_pnl:>+8.2f}%")
    
    # Bot rankings
    print(f"\n{'  BOT RANKINGS BY TOTAL P&L'}")
    print(f"  {'-'*76}")
    print(f"  {'Rank':<6} {'Bot':<25} {'Trades':>8} {'Win%':>7} {'Total P&L':>10}")
    print(f"  {'-'*6} {'-'*25} {'-'*8} {'-'*7} {'-'*10}")
    
    ranked = sorted(active_bots, key=lambda x: x.total_pnl, reverse=True)
    for i, bot in enumerate(ranked, 1):
        print(f"  {i:<6} {bot.bot_name:<25} {bot.closed_trades:>8} "
              f"{bot.win_rate:>6.1f}% {bot.total_pnl:>+9.2f}%")
    
    # Best metrics
    if active_bots:
        best_win_rate = max(active_bots, key=lambda x: x.win_rate)
        best_pf = max(active_bots, key=lambda x: x.profit_factor if x.profit_factor != float('inf') else 0)
        best_sharpe = max(active_bots, key=lambda x: x.sharpe_ratio)
        
        print(f"\n{'  BEST PERFORMERS'}")
        print(f"  {'-'*76}")
        print(f"  Best Win Rate:          {best_win_rate.bot_name} ({best_win_rate.win_rate:.1f}%)")
        pf_str = f"{best_pf.profit_factor:.2f}" if best_pf.profit_factor != float('inf') else "∞"
        print(f"  Best Profit Factor:     {best_pf.bot_name} ({pf_str})")
        print(f"  Best Sharpe Ratio:      {best_sharpe.bot_name} ({best_sharpe.sharpe_ratio:.2f})")


def export_to_json(all_performance: List[BotPerformance], output_path: Path) -> None:
    """Export analysis to JSON."""
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bots": []
    }
    
    for perf in all_performance:
        bot_data = asdict(perf)
        # Convert tuples to lists for JSON serialization
        bot_data["best_symbol"] = list(bot_data["best_symbol"])
        bot_data["worst_symbol"] = list(bot_data["worst_symbol"])
        output["bots"].append(bot_data)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✓ Exported analysis to {output_path}")


def export_to_csv(all_performance: List[BotPerformance], output_path: Path) -> None:
    """Export summary to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Bot', 'Description', 'Total Trades', 'Open', 'Closed', 'Wins', 'Losses',
            'Win Rate %', 'Total P&L %', 'Avg P&L %', 'Profit Factor', 'Sharpe Ratio',
            'Max Drawdown %', 'Avg Duration (h)', 'Symbols Traded', 'Best Symbol', 'Best Symbol P&L %'
        ])
        
        for perf in all_performance:
            pf_str = f"{perf.profit_factor:.2f}" if perf.profit_factor != float('inf') else "INF"
            writer.writerow([
                perf.bot_name,
                perf.description,
                perf.total_trades,
                perf.open_trades,
                perf.closed_trades,
                perf.wins,
                perf.losses,
                f"{perf.win_rate:.1f}",
                f"{perf.total_pnl:.2f}",
                f"{perf.avg_pnl:.2f}",
                pf_str,
                f"{perf.sharpe_ratio:.2f}",
                f"{perf.max_drawdown:.2f}",
                f"{perf.avg_trade_duration_hours:.1f}",
                perf.symbols_traded,
                perf.best_symbol[0],
                f"{perf.best_symbol[1]:.2f}",
            ])
    
    print(f"\n✓ Exported summary to {output_path}")


def export_to_html(all_performance: List[BotPerformance], output_path: Path) -> None:
    """Export analysis to HTML report."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Bot Performance Analysis</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; }}
        h3 {{ color: #7f8c8d; margin-top: 20px; }}
        .bot-section {{ margin: 30px 0; padding: 20px; background: #fafafa; border-left: 4px solid #3498db; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric-label {{ font-weight: bold; color: #7f8c8d; }}
        .metric-value {{ color: #2c3e50; font-size: 1.1em; }}
        .positive {{ color: #27ae60; font-weight: bold; }}
        .negative {{ color: #e74c3c; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #34495e; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ecf0f1; }}
        tr:hover {{ background: #f8f9fa; }}
        .summary {{ background: #e8f5e9; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .timestamp {{ color: #95a5a6; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Bot Performance Analysis Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
"""
    
    # Overall summary
    active_bots = [p for p in all_performance if p.closed_trades > 0]
    if active_bots:
        total_pnl = sum(p.total_pnl for p in active_bots)
        total_trades = sum(p.closed_trades for p in active_bots)
        total_wins = sum(p.wins for p in active_bots)
        total_losses = sum(p.losses for p in active_bots)
        overall_wr = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0.0
        
        pnl_class = "positive" if total_pnl > 0 else "negative"
        
        html += f"""
        <div class="summary">
            <h2>📈 Portfolio Summary</h2>
            <div class="metric">
                <span class="metric-label">Active Bots:</span>
                <span class="metric-value">{len(active_bots)}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Total Trades:</span>
                <span class="metric-value">{total_trades}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Win Rate:</span>
                <span class="metric-value">{overall_wr:.1f}%</span>
            </div>
            <div class="metric">
                <span class="metric-label">Total P&L:</span>
                <span class="metric-value {pnl_class}">{total_pnl:+.2f}%</span>
            </div>
        </div>
"""
    
    # Individual bot sections
    for perf in sorted(all_performance, key=lambda x: x.total_pnl, reverse=True):
        if perf.closed_trades == 0:
            continue
        
        pnl_class = "positive" if perf.total_pnl > 0 else "negative"
        pf_str = f"{perf.profit_factor:.2f}" if perf.profit_factor != float('inf') else "∞"
        
        html += f"""
        <div class="bot-section">
            <h2>{perf.bot_name}</h2>
            <p style="color: #7f8c8d; font-style: italic;">{perf.description}</p>
            
            <h3>Performance Metrics</h3>
            <div class="metric">
                <span class="metric-label">Total Trades:</span>
                <span class="metric-value">{perf.closed_trades}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Win Rate:</span>
                <span class="metric-value">{perf.win_rate:.1f}%</span>
            </div>
            <div class="metric">
                <span class="metric-label">Total P&L:</span>
                <span class="metric-value {pnl_class}">{perf.total_pnl:+.2f}%</span>
            </div>
            <div class="metric">
                <span class="metric-label">Avg P&L:</span>
                <span class="metric-value">{perf.avg_pnl:+.2f}%</span>
            </div>
            <div class="metric">
                <span class="metric-label">Profit Factor:</span>
                <span class="metric-value">{pf_str}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Sharpe Ratio:</span>
                <span class="metric-value">{perf.sharpe_ratio:.2f}</span>
            </div>
            
            <h3>Top Symbols</h3>
            <table>
                <tr>
                    <th>Symbol</th>
                    <th>Trades</th>
                    <th>Win Rate</th>
                    <th>Total P&L</th>
                    <th>Avg P&L</th>
                </tr>
"""
        
        for sym_stat in perf.symbol_stats[:5]:
            pnl_class = "positive" if sym_stat.total_pnl > 0 else "negative"
            html += f"""
                <tr>
                    <td>{sym_stat.symbol}</td>
                    <td>{sym_stat.total_trades}</td>
                    <td>{sym_stat.win_rate:.1f}%</td>
                    <td class="{pnl_class}">{sym_stat.total_pnl:+.2f}%</td>
                    <td>{sym_stat.avg_pnl:+.2f}%</td>
                </tr>
"""
        
        html += """
            </table>
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"\n✓ Exported HTML report to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive bot performance analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_bot_performance.py                    # Full analysis
  python analyze_bot_performance.py --bot diy          # Specific bot
  python analyze_bot_performance.py --detailed         # Include trade details
  python analyze_bot_performance.py --export html      # HTML report
  python analyze_bot_performance.py --export json      # JSON export
  python analyze_bot_performance.py --export csv       # CSV export
        """
    )
    
    parser.add_argument("--bot", type=str, help="Analyze specific bot only")
    parser.add_argument("--detailed", action="store_true", help="Include detailed trade lists")
    parser.add_argument("--summary", action="store_true", help="Show summary only (no individual bots)")
    parser.add_argument("--export", choices=["html", "json", "csv"], help="Export format")
    parser.add_argument("--output", type=str, help="Output file path (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Determine which bots to analyze
    bots_to_analyze = BOT_CONFIGS.keys()
    if args.bot:
        matching = [b for b in bots_to_analyze if args.bot.lower() in b.lower()]
        if not matching:
            print(f"Error: No bot matching '{args.bot}' found", file=sys.stderr)
            print(f"Available bots: {', '.join(BOT_CONFIGS.keys())}", file=sys.stderr)
            sys.exit(1)
        bots_to_analyze = matching
    
    # Analyze all bots
    all_performance: List[BotPerformance] = []
    
    print(f"\n{'#'*80}")
    print(f"#  COMPREHENSIVE BOT PERFORMANCE ANALYSIS")
    print(f"#  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}")
    
    for bot_key in bots_to_analyze:
        config = BOT_CONFIGS[bot_key]
        perf = analyze_bot_performance(bot_key, config)
        if perf:
            all_performance.append(perf)
    
    if not all_performance:
        print("\nNo performance data found for any bots.", file=sys.stderr)
        sys.exit(1)
    
    # Export if requested
    if args.export:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if args.output:
            output_path = Path(args.output)
        else:
            if args.export == "html":
                output_path = BASE_DIR / f"bot_performance_report_{timestamp}.html"
            elif args.export == "json":
                output_path = BASE_DIR / f"bot_performance_data_{timestamp}.json"
            else:  # csv
                output_path = BASE_DIR / f"bot_performance_summary_{timestamp}.csv"
        
        if args.export == "html":
            export_to_html(all_performance, output_path)
        elif args.export == "json":
            export_to_json(all_performance, output_path)
        else:  # csv
            export_to_csv(all_performance, output_path)
    
    # Print reports
    if not args.summary:
        for perf in sorted(all_performance, key=lambda x: x.total_pnl, reverse=True):
            print_bot_performance(perf, detailed=args.detailed)
    
    # Always print overall summary if multiple bots
    if len(all_performance) > 1:
        print_overall_summary(all_performance)
    
    print(f"\n{'='*80}")
    print(f"  Analysis complete!")
    print(f"  Use --export html/json/csv to save results")
    print(f"  Use --detailed for full trade lists")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
