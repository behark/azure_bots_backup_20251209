#!/usr/bin/env python3
"""
Detailed per-symbol performance analysis across all bots.
Shows which symbols work best for each bot.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Any, DefaultDict, Dict
import fnmatch

BASE_DIR = Path(__file__).resolve().parent

def discover_stats_files() -> Dict[str, Path]:
    stats_map: Dict[str, Path] = {}
    for path in BASE_DIR.iterdir():
        if not path.is_dir():
            continue
        if not path.name.endswith("_bot"):
            continue
        logs_dir = path / "logs"
        if not logs_dir.exists() or not logs_dir.is_dir():
            continue
        for stats_file in logs_dir.iterdir():
            if stats_file.is_file() and fnmatch.fnmatch(stats_file.name, "*stats.json"):
                bot_name = path.name.replace("_bot", "")
                stats_map[bot_name] = stats_file
                break
    return stats_map

def load_stats(bot_name: str, stats_files: Dict[str, Path]) -> Dict:
    """Load stats for a bot."""
    stats_file = stats_files.get(bot_name)
    if not stats_file or not stats_file.exists():
        return {}
    
    try:
        with open(stats_file, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def analyze_symbol_performance(bot_name: str, stats: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Analyze performance by symbol for a bot."""
    symbol_data: DefaultDict[str, Dict[str, Any]] = defaultdict(lambda: {
        "total": 0,
        "wins": 0,
        "losses": 0,
        "tp1": 0,
        "tp2": 0,
        "sl": 0,
        "pnl": 0.0,
        "directions": defaultdict(int),
        "win_pnl": 0.0,
        "loss_pnl": 0.0,
    })
    
    closed_signals = stats.get("history", [])
    
    for signal_data in closed_signals:
        symbol = signal_data.get("symbol", "").replace("/USDT", "").replace(":USDT", "")
        if not symbol:
            continue
        
        result = signal_data.get("result", "")
        direction = signal_data.get("direction", "")
        pnl_pct = signal_data.get("pnl_pct", 0)
        
        data = symbol_data[symbol]
        data["total"] += 1
        data["directions"][direction] += 1
        
        if result in ["TP1", "TP2"]:
            data["wins"] += 1
            data["win_pnl"] += pnl_pct
            if result == "TP1":
                data["tp1"] += 1
            else:
                data["tp2"] += 1
        elif result == "SL":
            data["losses"] += 1
            data["loss_pnl"] += pnl_pct
            data["sl"] += 1
        
        data["pnl"] += pnl_pct
    
    return dict(symbol_data)

def generate_bot_report(bot_name: str, stats_files: Dict[str, Path]) -> str:
    """Generate detailed report for a bot."""
    stats = load_stats(bot_name, stats_files)
    if not stats:
        return f"\n## {bot_name.upper()} BOT - NO DATA AVAILABLE\n"
    
    symbol_perf = analyze_symbol_performance(bot_name, stats)
    if not symbol_perf:
        return f"\n## {bot_name.upper()} BOT - NO CLOSED SIGNALS\n"
    
    # Sort by P&L
    sorted_symbols = sorted(symbol_perf.items(), key=lambda x: x[1]["pnl"], reverse=True)
    
    report = f"\n{'='*80}\n"
    report += f"## {bot_name.upper()} BOT - DETAILED SYMBOL PERFORMANCE\n"
    report += f"{'='*80}\n\n"
    
    # Overall stats
    total_signals = sum(d["total"] for d in symbol_perf.values())
    total_wins = sum(d["wins"] for d in symbol_perf.values())
    total_pnl = sum(d["pnl"] for d in symbol_perf.values())
    overall_win_rate = (total_wins / total_signals * 100) if total_signals > 0 else 0
    
    report += "**OVERALL PERFORMANCE:**\n"
    report += f"- Total Signals: {total_signals}\n"
    report += f"- Win Rate: {overall_win_rate:.2f}%\n"
    report += f"- Total P&L: {total_pnl:+.2f}%\n\n"
    
    # Top performers
    report += "### 🏆 TOP PERFORMERS (Best 5)\n\n"
    report += f"{'Rank':<6} {'Symbol':<12} {'Signals':<10} {'Win%':<10} {'P&L':<12} {'TP1/TP2/SL':<15} {'Rating':<10}\n"
    report += f"{'-'*80}\n"
    
    for i, (symbol, data) in enumerate(sorted_symbols[:5], 1):
        win_rate = (data["wins"] / data["total"] * 100) if data["total"] > 0 else 0
        
        # Rating based on P&L and win rate
        if data["pnl"] > 50 and win_rate > 60:
            rating = "⭐⭐⭐⭐⭐"
        elif data["pnl"] > 30 and win_rate > 50:
            rating = "⭐⭐⭐⭐"
        elif data["pnl"] > 10:
            rating = "⭐⭐⭐"
        elif data["pnl"] > 0:
            rating = "⭐⭐"
        else:
            rating = "⭐"
        
        tp_string = f"{data['tp1']}/{data['tp2']}/{data['sl']}"
        
        report += f"#{i:<5} {symbol:<12} {data['total']:<10} {win_rate:<9.1f}% {data['pnl']:+11.2f}% {tp_string:<15} {rating:<10}\n"
    
    report += "\n"
    
    # Detailed breakdown for ALL symbols
    report += "### 📊 COMPLETE SYMBOL BREAKDOWN\n\n"
    
    for symbol, data in sorted_symbols:
        win_rate = (data["wins"] / data["total"] * 100) if data["total"] > 0 else 0
        avg_win = data["win_pnl"] / data["wins"] if data["wins"] > 0 else 0
        avg_loss = data["loss_pnl"] / data["losses"] if data["losses"] > 0 else 0
        
        # Determine performance status
        if data["pnl"] > 20:
            status = "🔥 EXCELLENT"
        elif data["pnl"] > 10:
            status = "✅ GOOD"
        elif data["pnl"] > 0:
            status = "🟢 POSITIVE"
        elif data["pnl"] > -10:
            status = "⚠️ SLIGHT LOSS"
        else:
            status = "❌ POOR"
        
        report += f"\n**{symbol}** - {status}\n"
        report += "```\n"
        report += f"Total Signals:    {data['total']}\n"
        report += f"Win Rate:         {win_rate:.1f}% ({data['wins']}W / {data['losses']}L)\n"
        report += f"Total P&L:        {data['pnl']:+.2f}%\n"
        report += f"TP1 / TP2 / SL:   {data['tp1']} / {data['tp2']} / {data['sl']}\n"
        report += f"Avg Win:          {avg_win:+.2f}%\n"
        report += f"Avg Loss:         {avg_loss:+.2f}%\n"
        
        # Direction breakdown
        if data["directions"]:
            report += "Directions:       "
            dir_parts = [f"{d}: {c}" for d, c in data["directions"].items()]
            report += ", ".join(dir_parts) + "\n"
        
        report += "```\n"
    
    # Bottom performers
    report += "\n### ⚠️ WORST PERFORMERS (Bottom 5)\n\n"
    report += f"{'Rank':<6} {'Symbol':<12} {'Signals':<10} {'Win%':<10} {'P&L':<12} {'Status':<20}\n"
    report += f"{'-'*80}\n"
    
    worst_symbols = sorted_symbols[-5:]
    worst_symbols.reverse()
    
    for i, (symbol, data) in enumerate(worst_symbols, 1):
        win_rate = (data["wins"] / data["total"] * 100) if data["total"] > 0 else 0
        
        if data["pnl"] < -20:
            status = "❌ REMOVE ASAP"
        elif data["pnl"] < -10:
            status = "⚠️ CONSIDER REMOVING"
        elif data["pnl"] < 0:
            status = "⚠️ MONITOR"
        else:
            status = "🟢 ACCEPTABLE"
        
        report += f"#{i:<5} {symbol:<12} {data['total']:<10} {win_rate:<9.1f}% {data['pnl']:+11.2f}% {status:<20}\n"
    
    return report

def generate_cross_bot_comparison(stats_files: Dict[str, Path]) -> str:
    """Generate comparison of symbols across all bots."""
    report = f"\n{'='*80}\n"
    report += "## CROSS-BOT SYMBOL COMPARISON\n"
    report += f"{'='*80}\n\n"
    report += "Shows how each symbol performs across different bots.\n\n"
    
    # Load all data
    all_symbols = set()
    bot_symbol_data = {}
    
    discovered_bots = sorted(stats_files.keys())
    for bot_name in discovered_bots:
        stats = load_stats(bot_name, stats_files)
        symbol_perf = analyze_symbol_performance(bot_name, stats)
        bot_symbol_data[bot_name] = symbol_perf
        all_symbols.update(symbol_perf.keys())
    
    # Sort symbols by total signals across all bots
    symbol_totals = []
    for symbol in all_symbols:
        total_signals = sum(
            bot_symbol_data.get(bot, {}).get(symbol, {}).get("total", 0)
            for bot in discovered_bots
        )
        symbol_totals.append((symbol, total_signals))
    
    sorted_all_symbols = [s[0] for s in sorted(symbol_totals, key=lambda x: x[1], reverse=True)]
    
    report += "### 📊 SYMBOL PERFORMANCE MATRIX\n\n"
    
    for symbol in sorted_all_symbols:
        report += f"\n**{symbol}**\n"
        report += "```\n"
        
        any_data = False
        for bot_name in discovered_bots:
            data = bot_symbol_data.get(bot_name, {}).get(symbol)
            
            if data:
                any_data = True
                win_rate = (data["wins"] / data["total"] * 100) if data["total"] > 0 else 0
                
                # Performance indicator
                if data["pnl"] > 50:
                    indicator = "🔥🔥🔥"
                elif data["pnl"] > 20:
                    indicator = "🔥"
                elif data["pnl"] > 0:
                    indicator = "✅"
                elif data["pnl"] > -10:
                    indicator = "⚠️"
                else:
                    indicator = "❌"
                
                report += f"{bot_name.upper():<12} {indicator} {data['total']:>3} signals | {win_rate:>5.1f}% win | {data['pnl']:+7.2f}% P&L\n"
            else:
                report += f"{bot_name.upper():<12} ⚪ Not monitored by this bot\n"
        
        if not any_data:
            report += "No data available\n"
        
        report += "```\n"
    
    return report

def generate_recommendations(stats_files: Dict[str, Path]) -> str:
    """Generate actionable recommendations."""
    report = f"\n{'='*80}\n"
    report += "## 🎯 ACTIONABLE RECOMMENDATIONS\n"
    report += f"{'='*80}\n\n"
    
    # Load all data
    bot_symbol_data = {}
    discovered_bots = sorted(stats_files.keys())
    for bot_name in discovered_bots:
        stats = load_stats(bot_name, stats_files)
        symbol_perf = analyze_symbol_performance(bot_name, stats)
        bot_symbol_data[bot_name] = symbol_perf
    
    # Find symbols to remove
    report += "### ❌ SYMBOLS TO CONSIDER REMOVING\n\n"
    
    for bot_name in discovered_bots:
        symbol_perf = bot_symbol_data.get(bot_name, {})
        bad_symbols = [(s, d) for s, d in symbol_perf.items() if d["pnl"] < -10 or (d["total"] >= 5 and d["wins"] / d["total"] < 0.30)]
        
        if bad_symbols:
            report += f"**{bot_name.upper()} Bot:**\n"
            for symbol, data in sorted(bad_symbols, key=lambda x: x[1]["pnl"]):
                win_rate = (data["wins"] / data["total"] * 100) if data["total"] > 0 else 0
                report += f"- {symbol}: {data['pnl']:+.1f}% P&L, {win_rate:.1f}% win rate ({data['total']} signals)\n"
            report += "\n"
    
    # Find best performers to potentially increase position size
    report += "### 🌟 BEST PERFORMERS (Consider Increasing Position Size)\n\n"
    
    for bot_name in discovered_bots:
        symbol_perf = bot_symbol_data.get(bot_name, {})
        great_symbols = [(s, d) for s, d in symbol_perf.items() if d["pnl"] > 50 and d["total"] >= 5]
        
        if great_symbols:
            report += f"**{bot_name.upper()} Bot:**\n"
            for symbol, data in sorted(great_symbols, key=lambda x: x[1]["pnl"], reverse=True):
                win_rate = (data["wins"] / data["total"] * 100) if data["total"] > 0 else 0
                report += f"- {symbol}: {data['pnl']:+.1f}% P&L, {win_rate:.1f}% win rate ({data['total']} signals) ⭐\n"
            report += "\n"
    
    # Cross-bot insights
    report += "### 💡 CROSS-BOT INSIGHTS\n\n"
    
    # Find symbols that work in one bot but not another
    all_symbols = set()
    for data in bot_symbol_data.values():
        all_symbols.update(data.keys())
    
    for symbol in all_symbols:
        performances = {}
        for bot_name in discovered_bots:
            data = bot_symbol_data.get(bot_name, {}).get(symbol)
            if data and data["total"] >= 3:
                performances[bot_name] = data["pnl"]
        
        if len(performances) >= 2:
            best_bot = max(performances.items(), key=lambda x: x[1])
            worst_bot = min(performances.items(), key=lambda x: x[1])
            
            # If there's a big difference
            if best_bot[1] > 20 and worst_bot[1] < -10:
                report += f"- **{symbol}**: Works GREAT in {best_bot[0].upper()} ({best_bot[1]:+.1f}%) but POOR in {worst_bot[0].upper()} ({worst_bot[1]:+.1f}%)\n"
                report += f"  → Consider removing from {worst_bot[0]} bot only!\n\n"
    
    return report

def main():
    """Generate complete analysis."""
    print("Generating detailed symbol analysis...")
    stats_files = discover_stats_files()
    if not stats_files:
        print("No bot stats files discovered. Exiting.")
        return
    
    # Generate full report
    full_report = "# 📊 DETAILED SYMBOL PERFORMANCE ANALYSIS\n\n"
    full_report += "**Generated:** " + str(Path(__file__).resolve().parent) + "\n"
    full_report += "**Purpose:** Understand which symbols perform best for each bot\n\n"
    
    # Individual bot reports
    for bot_name in sorted(stats_files.keys()):
        full_report += generate_bot_report(bot_name, stats_files)
        full_report += "\n\n"
    
    # Cross-bot comparison
    full_report += generate_cross_bot_comparison(stats_files)
    full_report += "\n\n"
    
    # Recommendations
    full_report += generate_recommendations(stats_files)
    
    # Save report
    output_file = BASE_DIR / "DETAILED_SYMBOL_ANALYSIS.md"
    output_file.write_text(full_report)
    
    print(f"\n✅ Analysis complete! Report saved to: {output_file}")
    print("\nQuick Summary:")
    
    # Print quick summary to console
    for bot_name in sorted(stats_files.keys()):
        stats = load_stats(bot_name, stats_files)
        symbol_perf = analyze_symbol_performance(bot_name, stats)
        
        if symbol_perf:
            sorted_symbols = sorted(symbol_perf.items(), key=lambda x: x[1]["pnl"], reverse=True)
            
            print(f"\n{bot_name.upper()} Bot:")
            print(f"  Best: {sorted_symbols[0][0]} ({sorted_symbols[0][1]['pnl']:+.1f}%)")
            print(f"  Worst: {sorted_symbols[-1][0]} ({sorted_symbols[-1][1]['pnl']:+.1f}%)")

if __name__ == "__main__":
    main()
