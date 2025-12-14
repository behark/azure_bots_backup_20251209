#!/usr/bin/env python3
"""Generate comprehensive bot performance report."""

import json
from pathlib import Path
import fnmatch

bots_data = []

base_dir = Path(__file__).parent
bot_dirs = [p for p in base_dir.iterdir() if p.is_dir() and p.name.endswith('_bot')]

for bot_dir in bot_dirs:
    bot_name = bot_dir.name.replace('_', ' ').title()
    stats_file = None
    logs_dir = bot_dir / 'logs'
    if logs_dir.exists():
        for f in logs_dir.iterdir():
            if f.is_file() and fnmatch.fnmatch(f.name, '*stats.json'):
                stats_file = f
                break
    if stats_file is None:
        continue
    
    if stats_file.exists():
        try:
            with open(stats_file) as f:
                data = json.load(f)
            
            total_closed = len(data.get('closed', {}))
            total_open = len(data.get('open', {}))
            
            # Count wins/losses
            wins = 0
            losses = 0
            for sig in data.get('closed', {}).values():
                result = sig.get('result', sig.get('exit_reason', ''))
                if 'TP' in str(result) or 'PROFIT' in str(result).upper():
                    wins += 1
                elif 'SL' in str(result) or 'STOP' in str(result).upper() or 'LOSS' in str(result).upper():
                    losses += 1
            
            win_rate = (wins / total_closed * 100) if total_closed > 0 else 0
            
            bots_data.append({
                'name': bot_name,
                'open': total_open,
                'closed': total_closed,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'active': total_open > 0 or total_closed > 0
            })
        except Exception as e:
            print(f"Error reading {bot_name}: {e}")
            bots_data.append({
                'name': bot_name,
                'open': 0,
                'closed': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'active': False
            })

# Print results
print("\n" + "="*80)
print("                    🏆 COMPLETE BOT PERFORMANCE REPORT 🏆")
print("="*80 + "\n")

# Active bots
active_bots = [b for b in bots_data if b['active']]
inactive_bots = [b for b in bots_data if not b['active']]

print("📊 ACTIVE BOTS WITH SIGNALS:\n")
print(f"{'Bot Name':<25} {'Open':<8} {'Closed':<8} {'Wins':<8} {'Losses':<8} {'Win Rate':<10}")
print("-" * 80)

for bot in sorted(active_bots, key=lambda x: x['closed'], reverse=True):
    print(f"{bot['name']:<25} {bot['open']:<8} {bot['closed']:<8} {bot['wins']:<8} {bot['losses']:<8} {bot['win_rate']:>6.1f}%")

print("\n" + "="*80)
print("\n⏸️  BOTS WITHOUT SIGNALS YET:\n")
for bot in inactive_bots:
    print(f"  • {bot['name']}")

# Summary
total_open = sum(b['open'] for b in active_bots)
total_closed = sum(b['closed'] for b in active_bots)
total_wins = sum(b['wins'] for b in active_bots)
total_losses = sum(b['losses'] for b in active_bots)
overall_win_rate = (total_wins / total_closed * 100) if total_closed > 0 else 0

print("\n" + "="*80)
print("📈 OVERALL STATISTICS:")
print(f"  Total Open Signals:    {total_open}")
print(f"  Total Closed Signals:  {total_closed}")
print(f"  Total Wins:            {total_wins}")
print(f"  Total Losses:          {total_losses}")
print(f"  Overall Win Rate:      {overall_win_rate:.2f}%")
print(f"  Active Bots:           {len(active_bots)}/{len(bots_data)}")
print("="*80 + "\n")
