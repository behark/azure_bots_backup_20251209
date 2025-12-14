#!/usr/bin/env python3
"""Analyze historical performance for all three bots."""

import json
from pathlib import Path
from collections import defaultdict
import fnmatch

def analyze_stats_json(json_path, bot_name):
    """Analyze bot performance from stats JSON file."""
    if not json_path.exists():
        print(f"⚠️  No stats file found for {bot_name}")
        return None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    history = data.get('history', [])
    if not history:
        print(f"⚠️  No historical data found for {bot_name}")
        return None
    
    total = len(history)
    tp1_count = sum(1 for s in history if s['result'] == 'TP1')
    tp2_count = sum(1 for s in history if s['result'] == 'TP2')
    sl_count = sum(1 for s in history if s['result'] == 'SL')
    
    wins = tp1_count + tp2_count
    win_rate = (wins / total * 100) if total > 0 else 0
    
    # Calculate actual P&L
    total_pnl = sum(s.get('pnl_pct', 0) for s in history)
    avg_win = sum(s.get('pnl_pct', 0) for s in history if s['result'] in ['TP1', 'TP2']) / wins if wins > 0 else 0
    avg_loss = sum(s.get('pnl_pct', 0) for s in history if s['result'] == 'SL') / sl_count if sl_count > 0 else 0
    
    # Symbol performance
    symbol_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0})
    for signal in history:
        symbol = signal['symbol'].replace('/USDT', '')
        if signal['result'] in ['TP1', 'TP2']:
            symbol_stats[symbol]['wins'] += 1
        else:
            symbol_stats[symbol]['losses'] += 1
        symbol_stats[symbol]['pnl'] += signal.get('pnl_pct', 0)
    
    # Direction performance
    direction_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0})
    for signal in history:
        direction = signal.get('direction', 'UNKNOWN')
        if signal['result'] in ['TP1', 'TP2']:
            direction_stats[direction]['wins'] += 1
        else:
            direction_stats[direction]['losses'] += 1
        direction_stats[direction]['pnl'] += signal.get('pnl_pct', 0)
    
    return {
        'bot_name': bot_name,
        'total': total,
        'tp1': tp1_count,
        'tp2': tp2_count,
        'sl': sl_count,
        'wins': wins,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'symbol_stats': dict(symbol_stats),
        'direction_stats': dict(direction_stats)
    }

def print_bot_report(stats):
    """Print individual bot report."""
    print("=" * 70)
    print(" " * 15 + f"{stats['bot_name'].upper()}")
    print("=" * 70)
    print()
    
    print("📊 OVERALL STATISTICS")
    print("-" * 70)
    print(f"Total Signals Closed:  {stats['total']}")
    print(f"Take Profit 1 (TP1):   {stats['tp1']} ({stats['tp1']/stats['total']*100:.1f}%)")
    print(f"Take Profit 2 (TP2):   {stats['tp2']} ({stats['tp2']/stats['total']*100:.1f}%)")
    print(f"Stop Loss (SL):        {stats['sl']} ({stats['sl']/stats['total']*100:.1f}%)")
    print()
    print(f"✅ Total Wins:         {stats['wins']}")
    print(f"❌ Total Losses:       {stats['sl']}")
    print(f"🎯 WIN RATE:           {stats['win_rate']:.2f}%")
    print()
    print(f"💰 Total P&L:          {stats['total_pnl']:+.2f}%")
    print(f"📈 Avg Win:            {stats['avg_win']:+.2f}%")
    print(f"📉 Avg Loss:           {stats['avg_loss']:+.2f}%")
    if stats['avg_loss'] != 0:
        risk_reward = abs(stats['avg_win'] / stats['avg_loss'])
        print(f"⚖️  Risk/Reward Ratio:  1:{risk_reward:.2f}")
    print()
    
    # Direction performance
    if stats['direction_stats']:
        print("📈 DIRECTION PERFORMANCE")
        print("-" * 70)
        print(f"{'Direction':<15} {'Total':<8} {'Wins':<6} {'Loss':<6} {'Win%':<8} {'P&L%':<8}")
        print("-" * 70)
        for direction, counts in stats['direction_stats'].items():
            total_dir = counts['wins'] + counts['losses']
            win_rate_dir = (counts['wins'] / total_dir * 100) if total_dir > 0 else 0
            print(f"{direction:<15} {total_dir:<8} {counts['wins']:<6} {counts['losses']:<6} "
                  f"{win_rate_dir:>6.1f}% {counts['pnl']:>+7.1f}%")
        print()
    
    # Top symbols
    if stats['symbol_stats']:
        print("🏆 TOP 10 SYMBOLS (by P&L)")
        print("-" * 70)
        sorted_symbols = sorted(stats['symbol_stats'].items(), 
                               key=lambda x: x[1]['pnl'], reverse=True)[:10]
        
        print(f"{'Symbol':<15} {'Total':<8} {'Wins':<6} {'Loss':<6} {'Win%':<8} {'P&L%':<8}")
        print("-" * 70)
        for symbol, counts in sorted_symbols:
            total_sym = counts['wins'] + counts['losses']
            win_rate_sym = (counts['wins'] / total_sym * 100) if total_sym > 0 else 0
            print(f"{symbol:<15} {total_sym:<8} {counts['wins']:<6} {counts['losses']:<6} "
                  f"{win_rate_sym:>6.1f}% {counts['pnl']:>+7.1f}%")
        print()
        
        # Worst symbols
        print("⚠️  WORST 5 SYMBOLS (by P&L)")
        print("-" * 70)
        worst_symbols = sorted(stats['symbol_stats'].items(), 
                              key=lambda x: x[1]['pnl'])[:5]
        
        print(f"{'Symbol':<15} {'Total':<8} {'Wins':<6} {'Loss':<6} {'Win%':<8} {'P&L%':<8}")
        print("-" * 70)
        for symbol, counts in worst_symbols:
            total_sym = counts['wins'] + counts['losses']
            win_rate_sym = (counts['wins'] / total_sym * 100) if total_sym > 0 else 0
            print(f"{symbol:<15} {total_sym:<8} {counts['wins']:<6} {counts['losses']:<6} "
                  f"{win_rate_sym:>6.1f}% {counts['pnl']:>+7.1f}%")
        print()
    
    print("=" * 70)
    print()

def print_comparison(all_stats):
    """Print comparison of all bots."""
    print("\n")
    print("=" * 90)
    print(" " * 25 + "🏆 ALL BOTS COMPARISON 🏆")
    print("=" * 90)
    print()
    
    print(f"{'Bot':<20} {'Signals':<10} {'Win Rate':<12} {'Total P&L':<12} {'Avg Win':<10} {'Avg Loss':<10}")
    print("-" * 90)
    
    for stats in all_stats:
        print(f"{stats['bot_name']:<20} {stats['total']:<10} {stats['win_rate']:>6.1f}% "
              f"{stats['total_pnl']:>+10.2f}% {stats['avg_win']:>+8.2f}% {stats['avg_loss']:>+9.2f}%")
    
    print()
    print("=" * 90)
    print()
    
    # Overall summary
    total_signals = sum(s['total'] for s in all_stats)
    total_wins = sum(s['wins'] for s in all_stats)
    total_pnl = sum(s['total_pnl'] for s in all_stats)
    overall_win_rate = (total_wins / total_signals * 100) if total_signals > 0 else 0
    
    print("📊 COMBINED STATISTICS")
    print("-" * 90)
    print(f"Total Signals Across All Bots:  {total_signals}")
    print(f"Total Wins:                      {total_wins}")
    print(f"Overall Win Rate:                {overall_win_rate:.2f}%")
    print(f"Combined P&L:                    {total_pnl:+.2f}%")
    print()
    
    # Best and worst bot
    best_bot = max(all_stats, key=lambda x: x['total_pnl'])
    worst_bot = min(all_stats, key=lambda x: x['total_pnl'])
    
    print(f"🥇 Best Performing Bot:  {best_bot['bot_name']} ({best_bot['total_pnl']:+.2f}% P&L)")
    print(f"🏴 Worst Performing Bot: {worst_bot['bot_name']} ({worst_bot['total_pnl']:+.2f}% P&L)")
    print()
    
    print("=" * 90)

def main():
    """Main analysis function."""
    base_dir = Path(__file__).parent

    bots = []
    for bot_dir in base_dir.iterdir():
        if not bot_dir.is_dir() or not bot_dir.name.endswith('_bot'):
            continue
        logs_dir = bot_dir / 'logs'
        if not logs_dir.exists() or not logs_dir.is_dir():
            continue
        stats_file = None
        for candidate in logs_dir.iterdir():
            if candidate.is_file() and fnmatch.fnmatch(candidate.name, '*stats.json'):
                stats_file = candidate
                break
        if stats_file:
            bot_name = bot_dir.name.replace('_', ' ').title()
            bots.append((bot_name, stats_file))
    
    all_stats = []
    
    print("🔍 Analyzing all trading bots...")
    print()
    
    for bot_name, json_path in bots:
        stats = analyze_stats_json(json_path, bot_name)
        if stats:
            all_stats.append(stats)
            print_bot_report(stats)
    
    if len(all_stats) > 1:
        print_comparison(all_stats)
    
    print("\n✅ Analysis complete!")
    print(f"📄 Analyzed {len(all_stats)} bot(s)")

if __name__ == "__main__":
    main()
