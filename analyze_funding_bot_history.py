#!/usr/bin/env python3
"""Analyze historical Funding Bot signals from log files."""

import re
from collections import defaultdict
from typing import Any, Dict, Optional
from pathlib import Path

def parse_log_file(log_path: Path) -> list[Dict[str, Any]]:
    """Parse log file and extract signal closure information."""
    signals = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Match lines like: "ADA/USDT BULLISH TP1 hit!"
            match = re.search(r'🎯 ([A-Z]+)/USDT (BULLISH|BEARISH) (TP1|TP2|SL) hit', line)
            if match:
                symbol = match.group(1)
                direction = match.group(2)
                result = match.group(3)
                
                # Extract timestamp from log line
                time_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                timestamp = time_match.group(1) if time_match else None
                
                signals.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'direction': direction,
                    'result': result
                })
    
    return signals

def analyze_signals(signals: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze signal performance. Always returns a stats dict."""
    total = len(signals)
    if total == 0:
        return {
            'total': 0,
            'tp1': 0,
            'tp2': 0,
            'sl': 0,
            'wins': 0,
            'win_rate': 0.0,
            'symbol_stats': {},
            'direction_stats': {},
            'daily_stats': {},
        }
        
    tp1_count = sum(1 for s in signals if s['result'] == 'TP1')
    tp2_count = sum(1 for s in signals if s['result'] == 'TP2')
    sl_count = sum(1 for s in signals if s['result'] == 'SL')
    
    wins = tp1_count + tp2_count
    win_rate = (wins / total * 100) if total > 0 else 0
    
    # Symbol performance
    symbol_stats = defaultdict(lambda: {'TP1': 0, 'TP2': 0, 'SL': 0})
    for signal in signals:
        symbol_stats[signal['symbol']][signal['result']] += 1
    
    # Direction performance
    direction_stats = defaultdict(lambda: {'TP1': 0, 'TP2': 0, 'SL': 0})
    for signal in signals:
        direction_stats[signal['direction']][signal['result']] += 1
    
    # Daily performance
    daily_stats = defaultdict(lambda: {'TP1': 0, 'TP2': 0, 'SL': 0, 'total': 0})
    for signal in signals:
        if signal['timestamp']:
            date = signal['timestamp'].split()[0]
            daily_stats[date][signal['result']] += 1
            daily_stats[date]['total'] += 1
    
    return {
        'total': total,
        'tp1': tp1_count,
        'tp2': tp2_count,
        'sl': sl_count,
        'wins': wins,
        'win_rate': win_rate,
        'symbol_stats': dict(symbol_stats),
        'direction_stats': dict(direction_stats),
        'daily_stats': dict(daily_stats)
    }

def print_report(stats: Dict[str, Any], bot_name: str) -> None:
    """Print formatted analysis report."""
    print("=" * 70)
    print(" " * 10 + f"{bot_name.upper()} HISTORICAL PERFORMANCE")
    print("=" * 70)
    print()
    
    print("📊 OVERALL STATISTICS")
    print("-" * 70)
    print(f"Total Signals Closed:  {stats['total']}")
    print(f"Take Profit 1 (TP1):   {stats['tp1']} signals ({stats['tp1']/stats['total']*100:.1f}%)")
    print(f"Take Profit 2 (TP2):   {stats['tp2']} signals ({stats['tp2']/stats['total']*100:.1f}%)")
    print(f"Stop Loss (SL):        {stats['sl']} signals ({stats['sl']/stats['total']*100:.1f}%)")
    print()
    print(f"✅ Total Wins (TP1+TP2): {stats['wins']}")
    print(f"❌ Total Losses (SL):    {stats['sl']}")
    print(f"🎯 WIN RATE:             {stats['win_rate']:.2f}%")
    print()
    
    # Risk/Reward approximation
    estimated_r = (stats['tp1'] * 1) + (stats['tp2'] * 2) - (stats['sl'] * 1)
    print(f"💰 Estimated R-Multiple: {estimated_r:+.1f}R")
    print("   (Assuming TP1=1R, TP2=2R, SL=-1R)")
    print()
    
    # Direction performance
    print("📈 DIRECTION PERFORMANCE")
    print("-" * 70)
    print(f"{'Direction':<15} {'Total':<8} {'TP1':<6} {'TP2':<6} {'SL':<6} {'Win%':<8}")
    print("-" * 70)
    for direction, counts in stats['direction_stats'].items():
        total_dir = sum(counts.values())
        wins_dir = counts['TP1'] + counts['TP2']
        win_rate_dir = (wins_dir / total_dir * 100) if total_dir > 0 else 0
        print(f"{direction:<15} {total_dir:<8} {counts['TP1']:<6} {counts['TP2']:<6} "
              f"{counts['SL']:<6} {win_rate_dir:>6.1f}%")
    print()
    
    # Symbol performance (top 10 by volume)
    print("📈 TOP PERFORMING SYMBOLS (by total signals)")
    print("-" * 70)
    sorted_symbols = sorted(stats['symbol_stats'].items(), 
                           key=lambda x: sum(x[1].values()), reverse=True)[:10]
    
    print(f"{'Symbol':<15} {'Total':<8} {'TP1':<6} {'TP2':<6} {'SL':<6} {'Win%':<8}")
    print("-" * 70)
    for symbol, counts in sorted_symbols:
        total_sym = sum(counts.values())
        wins_sym = counts['TP1'] + counts['TP2']
        win_rate_sym = (wins_sym / total_sym * 100) if total_sym > 0 else 0
        print(f"{symbol:<15} {total_sym:<8} {counts['TP1']:<6} {counts['TP2']:<6} "
              f"{counts['SL']:<6} {win_rate_sym:>6.1f}%")
    print()
    
    # Daily performance
    print("📅 DAILY PERFORMANCE")
    print("-" * 70)
    sorted_days = sorted(stats['daily_stats'].items(), reverse=True)[:7]
    
    print(f"{'Date':<15} {'Total':<8} {'TP1':<6} {'TP2':<6} {'SL':<6} {'Win%':<8}")
    print("-" * 70)
    for date, counts in sorted_days:
        wins_day = counts['TP1'] + counts['TP2']
        win_rate_day = (wins_day / counts['total'] * 100) if counts['total'] > 0 else 0
        print(f"{date:<15} {counts['total']:<8} {counts['TP1']:<6} {counts['TP2']:<6} "
              f"{counts['SL']:<6} {win_rate_day:>6.1f}%")
    print()
    
    # Best symbols (by win rate, min 3 signals)
    print("🏆 BEST SYMBOLS (Win Rate, min 3 signals)")
    print("-" * 70)
    best_symbols = [(sym, counts) for sym, counts in stats['symbol_stats'].items()
                    if sum(counts.values()) >= 3]
    if best_symbols:
        best_symbols.sort(key=lambda x: (x[1]['TP1'] + x[1]['TP2']) / sum(x[1].values()), 
                         reverse=True)[:5]
        
        print(f"{'Symbol':<15} {'Total':<8} {'Wins':<6} {'Loss':<6} {'Win%':<8}")
        print("-" * 70)
        for symbol, counts in best_symbols:
            total_sym = sum(counts.values())
            wins_sym = counts['TP1'] + counts['TP2']
            win_rate_sym = (wins_sym / total_sym * 100)
            print(f"{symbol:<15} {total_sym:<8} {wins_sym:<6} {counts['SL']:<6} "
                  f"{win_rate_sym:>6.1f}%")
    else:
        print("No symbols with enough signals found.")
    print()
    
    print("=" * 70)

def main():
    """Main analysis function."""
    # Find log file
    base_dir = Path(__file__).parent / "funding_bot" / "logs"
    log_file = base_dir / "funding_bot.log"
    
    if not log_file.exists():
        print(f"Error: Log file not found at {log_file}")
        return
    
    print(f"Analyzing log file: {log_file}")
    print("This may take a moment...")
    print()
    
    # Parse and analyze
    signals = parse_log_file(log_file)
    
    if not signals:
        print("No historical signals found in log file.")
        return
    
    stats = analyze_signals(signals)
    if stats:
        print_report(stats, "Funding Bot")
        print(f"\n✅ Analysis complete! Found {stats['total']} closed signals.")
    else:
        print("No data to analyze.")

if __name__ == "__main__":
    main()
