#!/usr/bin/env python3
"""
Comprehensive Bot Performance Analysis Script
Analyzes all bots' performance, signal history, win/loss ratios, and provides actionable recommendations.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional
import sys


class BotPerformanceAnalyzer:
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.bots = self._discover_bots()
        self.global_config = self._load_global_config()

    def _discover_bots(self) -> List[str]:
        """Discover all bot directories."""
        bot_dirs = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and item.name.endswith('_bot'):
                bot_dirs.append(item.name)
        return sorted(bot_dirs)

    def _load_global_config(self) -> Dict:
        """Load global configuration."""
        config_file = self.base_dir / "global_config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        return {}

    def _load_stats_file(self, bot_name: str) -> Optional[Dict]:
        """Load stats file for a bot."""
        # Try different possible locations
        possible_paths = [
            self.base_dir / bot_name / "logs" / f"{bot_name.replace('_bot', '')}_stats.json",
            self.base_dir / bot_name / "logs" / f"{bot_name}_stats.json",
        ]

        for stats_path in possible_paths:
            if stats_path.exists():
                try:
                    with open(stats_path, 'r') as f:
                        return json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse {stats_path}")
                    return None
        return None

    def _calculate_signal_metrics(self, closed_signals: List[Dict]) -> Dict[str, Any]:
        """Calculate metrics from closed signals."""
        if not closed_signals:
            return {
                "total_signals": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "avg_profit": 0.0,
                "avg_loss": 0.0,
                "total_pnl": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "tp1_hits": 0,
                "tp2_hits": 0,
                "tp3_hits": 0,
                "sl_hits": 0,
                "avg_pnl": 0.0,
            }

        wins = [s for s in closed_signals if s.get('pnl_pct', 0) > 0]
        losses = [s for s in closed_signals if s.get('pnl_pct', 0) <= 0]

        # Count TP and SL hits
        tp1_hits = sum(1 for s in closed_signals if s.get('result') == 'TP1')
        tp2_hits = sum(1 for s in closed_signals if s.get('result') == 'TP2')
        tp3_hits = sum(1 for s in closed_signals if s.get('result') == 'TP3')
        sl_hits = sum(1 for s in closed_signals if s.get('result') == 'SL')

        pnls = [s.get('pnl_pct', 0) for s in closed_signals]

        return {
            "total_signals": len(closed_signals),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": (len(wins) / len(closed_signals) * 100) if closed_signals else 0,
            "avg_profit": sum(s.get('pnl_pct', 0) for s in wins) / len(wins) if wins else 0,
            "avg_loss": sum(s.get('pnl_pct', 0) for s in losses) / len(losses) if losses else 0,
            "total_pnl": sum(pnls),
            "best_trade": max(pnls) if pnls else 0,
            "worst_trade": min(pnls) if pnls else 0,
            "tp1_hits": tp1_hits,
            "tp2_hits": tp2_hits,
            "tp3_hits": tp3_hits,
            "sl_hits": sl_hits,
            "avg_pnl": sum(pnls) / len(pnls) if pnls else 0,
        }

    def _analyze_by_symbol(self, closed_signals: List[Dict]) -> Dict[str, Dict]:
        """Analyze performance by symbol/pair."""
        by_symbol = defaultdict(list)

        for signal in closed_signals:
            symbol = signal.get('symbol', 'UNKNOWN')
            by_symbol[symbol].append(signal)

        results = {}
        for symbol, signals in by_symbol.items():
            results[symbol] = self._calculate_signal_metrics(signals)

        return results

    def _analyze_by_direction(self, closed_signals: List[Dict]) -> Dict[str, Dict]:
        """Analyze performance by direction (LONG/SHORT or BULLISH/BEARISH)."""
        by_direction = defaultdict(list)

        for signal in closed_signals:
            direction = signal.get('direction', 'UNKNOWN')
            # Normalize direction names
            if direction in ['BULLISH', 'LONG']:
                direction = 'LONG'
            elif direction in ['BEARISH', 'SHORT']:
                direction = 'SHORT'
            by_direction[direction].append(signal)

        results = {}
        for direction, signals in by_direction.items():
            results[direction] = self._calculate_signal_metrics(signals)

        return results

    def _get_recommendations(self, bot_name: str, overall_metrics: Dict,
                           by_symbol: Dict, by_direction: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Overall performance
        if overall_metrics['total_signals'] == 0:
            recommendations.append("⚠️  No closed signals found - bot may not be trading or needs more time")
            return recommendations

        win_rate = overall_metrics['win_rate']
        total_pnl = overall_metrics['total_pnl']

        if win_rate < 40:
            recommendations.append(f"❌ Low win rate ({win_rate:.1f}%) - Consider disabling or reviewing strategy")
        elif win_rate > 60:
            recommendations.append(f"✅ Strong win rate ({win_rate:.1f}%) - Bot performing well")

        if total_pnl < -10:
            recommendations.append(f"❌ Significant losses ({total_pnl:.2f}%) - DISABLE THIS BOT")
        elif total_pnl > 10:
            recommendations.append(f"✅ Profitable overall ({total_pnl:.2f}%) - Keep running")

        # Symbol-specific recommendations
        worst_symbols = sorted(
            [(sym, data['total_pnl']) for sym, data in by_symbol.items()],
            key=lambda x: x[1]
        )[:3]

        if worst_symbols and worst_symbols[0][1] < -5:
            recommendations.append(
                f"🔴 Remove worst performing pairs: {', '.join([s[0] for s in worst_symbols if s[1] < -5])}"
            )

        best_symbols = sorted(
            [(sym, data['total_pnl']) for sym, data in by_symbol.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]

        if best_symbols and best_symbols[0][1] > 5:
            recommendations.append(
                f"✅ Best performing pairs: {', '.join([s[0] for s in best_symbols if s[1] > 5])}"
            )

        # Direction analysis
        for direction, data in by_direction.items():
            if data['total_signals'] >= 5:
                if data['win_rate'] < 35:
                    recommendations.append(
                        f"⚠️  {direction} signals underperforming ({data['win_rate']:.1f}% win rate)"
                    )

        # Risk-reward analysis
        if overall_metrics['avg_loss'] != 0:
            avg_win_loss_ratio = abs(overall_metrics['avg_profit'] / overall_metrics['avg_loss'])
            if avg_win_loss_ratio < 1:
                recommendations.append(
                    f"⚠️  Poor risk-reward ratio ({avg_win_loss_ratio:.2f}) - Losses bigger than wins"
                )

        # TP/SL hit analysis
        total_closed = overall_metrics['total_signals']
        sl_rate = (overall_metrics['sl_hits'] / total_closed * 100) if total_closed else 0

        if sl_rate > 70:
            recommendations.append(
                f"⚠️  High stop loss hit rate ({sl_rate:.1f}%) - Consider adjusting SL parameters"
            )

        return recommendations

    def analyze_bot(self, bot_name: str) -> Dict[str, Any]:
        """Analyze a single bot's performance."""
        print(f"\n{'='*80}")
        print(f"Analyzing: {bot_name}")
        print(f"{'='*80}")

        stats = self._load_stats_file(bot_name)

        if not stats:
            print(f"⚠️  No stats file found for {bot_name}")
            return {
                "bot_name": bot_name,
                "status": "NO_DATA",
                "error": "Stats file not found"
            }

        # Try both 'closed' and 'history' keys for closed signals
        closed_signals = stats.get('closed', stats.get('history', []))
        open_signals = stats.get('open', {})

        # Convert open signals dict to list if needed
        if isinstance(open_signals, dict):
            open_signals = list(open_signals.values())

        overall_metrics = self._calculate_signal_metrics(closed_signals)
        by_symbol = self._analyze_by_symbol(closed_signals)
        by_direction = self._analyze_by_direction(closed_signals)
        recommendations = self._get_recommendations(bot_name, overall_metrics, by_symbol, by_direction)

        # Get bot config
        bot_config = self.global_config.get('bots', {}).get(bot_name.replace('_bot', ''), {})

        result = {
            "bot_name": bot_name,
            "status": "ACTIVE" if bot_config.get('enabled', False) else "DISABLED",
            "config": {
                "enabled": bot_config.get('enabled', False),
                "max_open_signals": bot_config.get('max_open_signals', 'N/A'),
                "symbols_count": len(bot_config.get('symbols', [])),
                "symbols": [s.get('symbol') for s in bot_config.get('symbols', [])],
            },
            "open_positions": len(open_signals),
            "overall_metrics": overall_metrics,
            "by_symbol": by_symbol,
            "by_direction": by_direction,
            "recommendations": recommendations,
        }

        # Print summary
        self._print_bot_summary(result)

        return result

    def _print_bot_summary(self, result: Dict):
        """Print formatted bot summary."""
        print(f"\nStatus: {result['status']}")
        print(f"Symbols: {result['config']['symbols_count']} pairs")
        print(f"Open Positions: {result['open_positions']}")

        metrics = result['overall_metrics']
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"  Total Signals: {metrics['total_signals']}")
        print(f"  Win Rate: {metrics['win_rate']:.2f}% ({metrics['wins']}W / {metrics['losses']}L)")
        print(f"  Total PnL: {metrics['total_pnl']:.2f}%")
        print(f"  Average PnL: {metrics['avg_pnl']:.2f}%")
        print(f"  Average Win: {metrics['avg_profit']:.2f}%")
        print(f"  Average Loss: {metrics['avg_loss']:.2f}%")
        print(f"  Best Trade: {metrics['best_trade']:.2f}%")
        print(f"  Worst Trade: {metrics['worst_trade']:.2f}%")

        print(f"\n🎯 EXIT ANALYSIS:")
        print(f"  TP1 Hits: {metrics['tp1_hits']}")
        print(f"  TP2 Hits: {metrics['tp2_hits']}")
        print(f"  TP3 Hits: {metrics['tp3_hits']}")
        print(f"  SL Hits: {metrics['sl_hits']}")

        if result['by_direction']:
            print(f"\n📈 BY DIRECTION:")
            for direction, data in result['by_direction'].items():
                print(f"  {direction}: {data['win_rate']:.1f}% WR, {data['total_pnl']:.2f}% PnL ({data['total_signals']} signals)")

        if result['by_symbol']:
            print(f"\n💱 TOP 5 BEST PERFORMING PAIRS:")
            sorted_symbols = sorted(
                result['by_symbol'].items(),
                key=lambda x: x[1]['total_pnl'],
                reverse=True
            )[:5]
            for symbol, data in sorted_symbols:
                if data['total_signals'] > 0:
                    print(f"  {symbol}: {data['win_rate']:.1f}% WR, {data['total_pnl']:.2f}% PnL ({data['total_signals']} signals)")

            print(f"\n💱 TOP 5 WORST PERFORMING PAIRS:")
            sorted_symbols = sorted(
                result['by_symbol'].items(),
                key=lambda x: x[1]['total_pnl']
            )[:5]
            for symbol, data in sorted_symbols:
                if data['total_signals'] > 0:
                    print(f"  {symbol}: {data['win_rate']:.1f}% WR, {data['total_pnl']:.2f}% PnL ({data['total_signals']} signals)")

        if result['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in result['recommendations']:
                print(f"  {rec}")

    def analyze_all_bots(self) -> Dict[str, Any]:
        """Analyze all discovered bots."""
        print(f"\n{'#'*80}")
        print(f"# BOT PERFORMANCE ANALYSIS REPORT")
        print(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"# Total Bots: {len(self.bots)}")
        print(f"{'#'*80}")

        results = {}
        for bot_name in self.bots:
            results[bot_name] = self.analyze_bot(bot_name)

        # Overall summary
        self._print_overall_summary(results)

        # Save detailed report
        self._save_report(results)

        return results

    def _print_overall_summary(self, results: Dict[str, Any]):
        """Print overall summary across all bots."""
        print(f"\n{'='*80}")
        print(f"OVERALL SUMMARY - ALL BOTS")
        print(f"{'='*80}")

        total_signals = 0
        total_wins = 0
        total_losses = 0
        total_pnl = 0
        bots_with_data = 0

        bot_rankings = []

        for bot_name, data in results.items():
            if data.get('status') == 'NO_DATA':
                continue

            metrics = data['overall_metrics']
            if metrics['total_signals'] > 0:
                bots_with_data += 1
                total_signals += metrics['total_signals']
                total_wins += metrics['wins']
                total_losses += metrics['losses']
                total_pnl += metrics['total_pnl']

                bot_rankings.append({
                    'name': bot_name,
                    'win_rate': metrics['win_rate'],
                    'total_pnl': metrics['total_pnl'],
                    'signals': metrics['total_signals'],
                })

        print(f"\nBots with Data: {bots_with_data}/{len(results)}")
        print(f"Total Signals: {total_signals}")
        print(f"Overall Win Rate: {(total_wins/total_signals*100) if total_signals else 0:.2f}%")
        print(f"Total PnL: {total_pnl:.2f}%")

        print(f"\n🏆 BOT RANKINGS BY PnL:")
        sorted_bots = sorted(bot_rankings, key=lambda x: x['total_pnl'], reverse=True)
        for i, bot in enumerate(sorted_bots, 1):
            status_icon = "✅" if bot['total_pnl'] > 0 else "❌"
            print(f"  {i}. {status_icon} {bot['name']}: {bot['total_pnl']:.2f}% PnL, "
                  f"{bot['win_rate']:.1f}% WR ({bot['signals']} signals)")

        print(f"\n🎯 BOT RANKINGS BY WIN RATE:")
        sorted_bots = sorted(bot_rankings, key=lambda x: x['win_rate'], reverse=True)
        for i, bot in enumerate(sorted_bots, 1):
            status_icon = "✅" if bot['win_rate'] > 50 else "❌"
            print(f"  {i}. {status_icon} {bot['name']}: {bot['win_rate']:.1f}% WR, "
                  f"{bot['total_pnl']:.2f}% PnL ({bot['signals']} signals)")

        # Action items
        print(f"\n{'='*80}")
        print(f"🚨 RECOMMENDED ACTIONS:")
        print(f"{'='*80}")

        for bot in sorted_bots:
            if bot['total_pnl'] < -10:
                print(f"  ❌ DISABLE: {bot['name']} (Loss: {bot['total_pnl']:.2f}%)")
            elif bot['win_rate'] < 35 and bot['signals'] >= 10:
                print(f"  ⚠️  REVIEW: {bot['name']} (Low WR: {bot['win_rate']:.1f}%)")

        best_bots = [b for b in sorted_bots if b['total_pnl'] > 10 and b['win_rate'] > 55]
        if best_bots:
            print(f"\n  ✅ KEEP RUNNING:")
            for bot in best_bots:
                print(f"     - {bot['name']} ({bot['total_pnl']:.2f}% PnL, {bot['win_rate']:.1f}% WR)")

    def _save_report(self, results: Dict[str, Any]):
        """Save detailed report to JSON file."""
        report_file = self.base_dir / f"bot_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n📄 Detailed report saved to: {report_file}")


def main():
    """Main entry point."""
    # Get base directory from command line or use current directory
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    analyzer = BotPerformanceAnalyzer(base_dir)
    analyzer.analyze_all_bots()


if __name__ == "__main__":
    main()
