#!/usr/bin/env python3
"""
Performance Statistics Helper
Provides consistent performance stats format for all bots.
"""

from typing import Any, Dict, Optional


def get_performance_stats(
    stats: Optional[Any] = None,
    symbol: str = "",
    tracker: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Get standardized performance statistics for a symbol.
    
    This function provides a consistent format for all bots to use when
    displaying performance history in signal alerts.
    
    Args:
        stats: SignalStats instance (from signal_stats.py)
        symbol: Trading pair symbol (e.g., "BTC/USDT")
        tracker: Alternative tracker object with stats attribute
    
    Returns:
        Dictionary with standardized performance stats:
        {
            "total": int,      # Total closed trades
            "wins": int,        # Total wins (TP1 + TP2)
            "tp1": int,         # TP1 hits
            "tp2": int,         # TP2 hits
            "tp3": int,         # TP3 hits (if available)
            "sl": int,          # Stop loss hits
            "avg_pnl": float,   # Average P&L percentage (optional)
        }
    
    Example:
        >>> from signal_stats import SignalStats
        >>> stats = SignalStats("my_bot", Path("stats.json"))
        >>> perf = get_performance_stats(stats, "BTC/USDT")
        >>> print(perf)
        {'total': 10, 'wins': 7, 'tp1': 5, 'tp2': 2, 'tp3': 0, 'sl': 3, 'avg_pnl': 0.0}
    """
    # Initialize default values
    tp1_count = 0
    tp2_count = 0
    tp3_count = 0
    sl_count = 0
    avg_pnl = 0.0
    
    # Try to get stats from various sources
    stats_obj = None
    
    # Method 1: Direct stats object
    if stats is not None:
        stats_obj = stats
    
    # Method 2: Tracker with stats attribute
    elif tracker is not None and hasattr(tracker, 'stats'):
        stats_obj = tracker.stats
    
    # Method 3: Tracker itself is stats
    elif tracker is not None and hasattr(tracker, 'symbol_tp_sl_counts'):
        stats_obj = tracker
    
    # Get counts if stats object is available
    if stats_obj is not None and symbol:
        try:
            # Try symbol_tp_sl_counts method (SignalStats)
            if hasattr(stats_obj, 'symbol_tp_sl_counts'):
                counts = stats_obj.symbol_tp_sl_counts(symbol)
                tp1_count = counts.get("TP1", 0)
                tp2_count = counts.get("TP2", 0)
                sl_count = counts.get("SL", 0)
            
            # Try get_avg_pnl method if available
            if hasattr(stats_obj, 'get_avg_pnl'):
                try:
                    avg_pnl = stats_obj.get_avg_pnl(symbol) or 0.0
                except Exception:
                    avg_pnl = 0.0
        except Exception:
            # If anything fails, use defaults (0 values)
            pass
    
    # Calculate totals
    total = tp1_count + tp2_count + tp3_count + sl_count
    wins = tp1_count + tp2_count + tp3_count
    
    # Return standardized format
    return {
        "total": total,
        "wins": wins,
        "tp1": tp1_count,
        "tp2": tp2_count,
        "tp3": tp3_count,
        "sl": sl_count,
        "avg_pnl": avg_pnl,
    }


def format_performance_line(perf_stats: Dict[str, Any]) -> str:
    """
    Format performance stats as a single line for signal alerts.
    
    Args:
        perf_stats: Performance stats dictionary from get_performance_stats()
    
    Returns:
        Formatted string like: "📈 History: 70% Win (7/10) | TP:7 SL:3 | PnL:+1.2%"
    """
    total = perf_stats.get("total", 0)
    wins = perf_stats.get("wins", 0)
    tp1 = perf_stats.get("tp1", 0)
    tp2 = perf_stats.get("tp2", 0)
    tp3 = perf_stats.get("tp3", 0)
    sl = perf_stats.get("sl", 0)
    avg_pnl = perf_stats.get("avg_pnl", 0.0)
    
    if total == 0:
        return "📈 History: 0% Win (0/0) | TP:0 SL:0"
    
    win_rate = (wins / total * 100) if total > 0 else 0
    tp_total = tp1 + tp2 + tp3
    
    # Build base line
    line = f"📈 History: {win_rate:.0f}% Win ({wins}/{total}) | TP:{tp_total} SL:{sl}"
    
    # Add P&L if available and non-zero
    if avg_pnl != 0:
        line += f" | PnL:{avg_pnl:+.1f}%"
    
    return line
