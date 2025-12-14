#!/usr/bin/env python3
"""
Performance Dashboard - Aggregates statistics from all trading bots
Displays win rates, P&L, and active signals across all bots.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import fnmatch

BASE_DIR = Path(__file__).resolve().parent

def discover_bots() -> Dict[str, Path]:
    bots: Dict[str, Path] = {}
    for bot_dir in BASE_DIR.iterdir():
        if not bot_dir.is_dir() or not bot_dir.name.endswith("_bot"):
            continue
        logs_dir = bot_dir / "logs"
        if not logs_dir.exists() or not logs_dir.is_dir():
            continue
        for candidate in logs_dir.iterdir():
            if candidate.is_file() and fnmatch.fnmatch(candidate.name, "*stats.json"):
                formatted = bot_dir.name.replace("_", " ").title()
                bots[formatted] = candidate
                break
    return bots


def load_bot_stats(file_path: Path) -> Optional[Dict]:
    """Load statistics from a bot's stats file."""
    if not file_path.exists():
        return None
    try:
        return json.loads(file_path.read_text())
    except (json.JSONDecodeError, IOError):
        return None


def calculate_summary(stats: Dict) -> Dict:
    """Calculate summary statistics from bot data."""
    history = stats.get("history", [])
    open_signals = stats.get("open", {})
    
    if not history:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "open_signals": len(open_signals),
        }
    
    total = len(history)
    wins = sum(1 for t in history if str(t.get("result", "")).startswith("TP"))
    losses = sum(1 for t in history if str(t.get("result", "")) == "SL")
    
    total_pnl = sum(float(t.get("pnl_pct", 0)) for t in history)
    avg_pnl = total_pnl / total if total > 0 else 0
    win_rate = (wins / total) * 100 if total > 0 else 0
    
    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "open_signals": len(open_signals),
    }


def get_recent_trades(stats: Dict, limit: int = 5) -> List[Dict]:
    """Get most recent trades from history."""
    history = stats.get("history", [])
    return history[-limit:] if history else []


def print_dashboard():
    """Print the performance dashboard."""
    print("\n" + "=" * 80)
    print(" " * 20 + "TRADING BOTS PERFORMANCE DASHBOARD")
    print(" " * 20 + f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 80)
    
    all_summaries = {}
    total_trades = 0
    total_wins = 0
    total_losses = 0
    total_pnl = 0.0
    total_open = 0
    
    bots = discover_bots()
    if not bots:
        print("No bots discovered. Exiting dashboard.")
        return

    # Collect stats from all bots
    for bot_name, stats_file in bots.items():
        stats = load_bot_stats(stats_file)
        if stats:
            summary = calculate_summary(stats)
            all_summaries[bot_name] = summary
            total_trades += summary["total_trades"]
            total_wins += summary["wins"]
            total_losses += summary["losses"]
            total_pnl += summary["total_pnl"]
            total_open += summary["open_signals"]
        else:
            all_summaries[bot_name] = None
    
    # Print overall summary
    print("\n" + "-" * 80)
    print(" OVERALL SUMMARY")
    print("-" * 80)
    overall_win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    print(f" Total Trades:    {total_trades:>6}")
    print(f" Winning Trades:  {total_wins:>6}")
    print(f" Losing Trades:   {total_losses:>6}")
    print(f" Win Rate:        {overall_win_rate:>5.1f}%")
    print(f" Total P&L:       {total_pnl:>+6.2f}%")
    print(f" Open Signals:    {total_open:>6}")
    
    # Print individual bot stats
    print("\n" + "-" * 80)
    print(" BOT PERFORMANCE")
    print("-" * 80)
    print(f"{'Bot Name':<20} {'Trades':>8} {'Wins':>6} {'Losses':>8} {'Win%':>7} {'P&L%':>10} {'Open':>6}")
    print("-" * 80)
    
    for bot_name, summary in all_summaries.items():
        if summary is None:
            print(f"{bot_name:<20} {'No data':<60}")
        else:
            print(
                f"{bot_name:<20} "
                f"{summary['total_trades']:>8} "
                f"{summary['wins']:>6} "
                f"{summary['losses']:>8} "
                f"{summary['win_rate']:>6.1f}% "
                f"{summary['total_pnl']:>+9.2f}% "
                f"{summary['open_signals']:>6}"
            )
    
    # Print top performers
    print("\n" + "-" * 80)
    print(" TOP PERFORMERS (by P&L)")
    print("-" * 80)
    
    sorted_bots = sorted(
        [(name, s) for name, s in all_summaries.items() if s and s["total_trades"] > 0],
        key=lambda x: x[1]["total_pnl"],
        reverse=True
    )
    
    for i, (name, summary) in enumerate(sorted_bots[:5], 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f" {emoji} {i}. {name:<18} P&L: {summary['total_pnl']:>+7.2f}% | Win Rate: {summary['win_rate']:.1f}%")
    
    # Print worst performers
    if len(sorted_bots) > 5:
        print("\n" + "-" * 80)
        print(" NEEDS ATTENTION (lowest P&L)")
        print("-" * 80)
        
        for name, summary in sorted_bots[-3:]:
            if summary["total_pnl"] < 0:
                print(f" ⚠️  {name:<18} P&L: {summary['total_pnl']:>+7.2f}% | Win Rate: {summary['win_rate']:.1f}%")
    
    # Print active signals
    print("\n" + "-" * 80)
    print(" ACTIVE SIGNALS")
    print("-" * 80)
    
    active_signals = []
    for bot_name, stats_file in bots.items():
        stats = load_bot_stats(stats_file)
        if stats:
            for signal_id, signal_data in stats.get("open", {}).items():
                active_signals.append({
                    "bot": bot_name,
                    "symbol": signal_data.get("symbol", "Unknown"),
                    "direction": signal_data.get("direction", "Unknown"),
                    "entry": signal_data.get("entry", 0),
                    "created": signal_data.get("created_at", "Unknown"),
                })
    
    if active_signals:
        print(f"{'Bot':<18} {'Symbol':<12} {'Direction':<8} {'Entry':>12} {'Created':<20}")
        print("-" * 80)
        for sig in active_signals[:10]:  # Show max 10 active signals
            print(
                f"{sig['bot']:<18} "
                f"{sig['symbol']:<12} "
                f"{sig['direction']:<8} "
                f"{sig['entry']:>12.6f} "
                f"{sig['created']:<20}"
            )
        if len(active_signals) > 10:
            print(f"... and {len(active_signals) - 10} more active signals")
    else:
        print(" No active signals")
    
    print("\n" + "=" * 80)


def generate_html_report(output_file: Path = BASE_DIR / "dashboard.html"):
    """Generate an HTML dashboard report."""
    bots = discover_bots()
    if not bots:
        print("No bots discovered. Unable to build HTML report.")
        return

    all_summaries = {}
    
    for bot_name, stats_file in bots.items():
        stats = load_bot_stats(stats_file)
        if stats:
            all_summaries[bot_name] = calculate_summary(stats)
    
    # Calculate totals
    total_trades = sum(s["total_trades"] for s in all_summaries.values() if s)
    total_wins = sum(s["wins"] for s in all_summaries.values() if s)
    total_pnl = sum(s["total_pnl"] for s in all_summaries.values() if s)
    overall_win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Trading Bots Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
        h1 {{ color: #00d4ff; text-align: center; }}
        .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .card {{ background: #16213e; padding: 20px; border-radius: 10px; text-align: center; min-width: 150px; }}
        .card h3 {{ color: #00d4ff; margin: 0; font-size: 24px; }}
        .card p {{ color: #888; margin: 5px 0 0 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #333; }}
        th {{ background: #16213e; color: #00d4ff; }}
        tr:hover {{ background: #1f2b47; }}
        .positive {{ color: #00ff88; }}
        .negative {{ color: #ff4444; }}
        .timestamp {{ text-align: center; color: #666; margin-top: 20px; }}
    </style>
</head>
<body>
    <h1>Trading Bots Performance Dashboard</h1>
    
    <div class="summary">
        <div class="card">
            <h3>{total_trades}</h3>
            <p>Total Trades</p>
        </div>
        <div class="card">
            <h3>{overall_win_rate:.1f}%</h3>
            <p>Win Rate</p>
        </div>
        <div class="card">
            <h3 class="{'positive' if total_pnl >= 0 else 'negative'}">{total_pnl:+.2f}%</h3>
            <p>Total P&L</p>
        </div>
        <div class="card">
            <h3>{len([s for s in all_summaries.values() if s])}</h3>
            <p>Active Bots</p>
        </div>
    </div>
    
    <table>
        <tr>
            <th>Bot Name</th>
            <th>Trades</th>
            <th>Wins</th>
            <th>Losses</th>
            <th>Win Rate</th>
            <th>P&L</th>
            <th>Open Signals</th>
        </tr>
"""
    
    for bot_name, summary in all_summaries.items():
        if summary:
            pnl_class = "positive" if summary["total_pnl"] >= 0 else "negative"
            html += f"""
        <tr>
            <td>{bot_name}</td>
            <td>{summary['total_trades']}</td>
            <td>{summary['wins']}</td>
            <td>{summary['losses']}</td>
            <td>{summary['win_rate']:.1f}%</td>
            <td class="{pnl_class}">{summary['total_pnl']:+.2f}%</td>
            <td>{summary['open_signals']}</td>
        </tr>
"""
    
    html += f"""
    </table>
    
    <p class="timestamp">Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
</body>
</html>
"""
    
    output_file.write_text(html)
    print(f"HTML report saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Trading Bots Performance Dashboard")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("--output", type=str, default="dashboard.html", help="HTML output file")
    args = parser.parse_args()
    
    if args.html:
        generate_html_report(Path(args.output))
    else:
        print_dashboard()
