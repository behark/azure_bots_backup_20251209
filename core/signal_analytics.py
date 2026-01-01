#!/usr/bin/env python3
"""
Signal Analytics - Deep Performance Insights

Provides advanced analytics for trading signals:
- Pattern/setup performance comparison
- Time-based performance (by hour, day of week)
- Symbol performance rankings
- Win rate by market conditions
- Drawdown analysis
- Expected value calculations

Usage:
    from core.signal_analytics import SignalAnalytics
    
    analytics = SignalAnalytics(history)
    
    # Get best performing setups
    top_setups = analytics.get_top_setups(limit=5)
    
    # Performance by time
    hourly = analytics.get_hourly_performance()
    
    # Symbol rankings
    rankings = analytics.get_symbol_rankings()
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SetupStats:
    """Statistics for a specific setup/pattern."""
    name: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    profit_factor: float
    avg_winner: float
    avg_loser: float
    best_trade: float
    worst_trade: float
    avg_duration_hours: float
    expectancy: float  # Expected value per trade


@dataclass
class TimeSlotStats:
    """Statistics for a time slot (hour or day)."""
    slot: str  # "09" for hour, "Monday" for day
    total_trades: int
    win_rate: float
    avg_pnl: float
    total_pnl: float


@dataclass
class SymbolRanking:
    """Ranking data for a symbol."""
    symbol: str
    total_trades: int
    wins: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    profit_factor: float
    last_trade_date: str


class SignalAnalytics:
    """
    Advanced analytics for trading signal history.
    
    Analyzes patterns, timing, symbols, and provides actionable insights.
    """
    
    def __init__(self, history: List[Dict[str, Any]]):
        """
        Initialize with trade history.
        
        Args:
            history: List of trade records with fields:
                - symbol, direction, result, pnl_pct
                - created_at, closed_at
                - pattern_name (optional)
                - extra (optional dict with metadata)
        """
        self.history = history
        self._cache: Dict[str, Any] = {}
    
    @classmethod
    def from_file(cls, file_path: Path) -> "SignalAnalytics":
        """Load analytics from a JSON file."""
        if not file_path.exists():
            return cls([])
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            history = data.get("history", [])
            return cls(history)
        except (json.JSONDecodeError, IOError) as e:
            logger.error("Failed to load analytics data: %s", e)
            return cls([])
    
    @classmethod
    def from_multiple_bots(cls, bot_names: List[str], base_dir: Path) -> "SignalAnalytics":
        """Aggregate history from multiple bots."""
        all_history = []
        
        for bot_name in bot_names:
            file_path = base_dir / bot_name / f"{bot_name}_signals.json"
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    for trade in data.get("history", []):
                        trade["bot_name"] = bot_name
                        all_history.append(trade)
                except (json.JSONDecodeError, IOError):
                    pass
        
        return cls(all_history)
    
    def _parse_datetime(self, dt_str: str) -> Optional[datetime]:
        """Parse datetime string."""
        if not dt_str:
            return None
        try:
            dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, AttributeError):
            return None
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall performance statistics."""
        if not self.history:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "profit_factor": 0,
                "avg_pnl": 0,
                "expectancy": 0,
            }
        
        total = len(self.history)
        pnls = [h.get("realized_pnl_pct", h.get("pnl_pct", 0)) for h in self.history]
        
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]
        
        wins = len(winners)
        win_rate = (wins / total) * 100 if total > 0 else 0
        
        gross_profit = sum(winners)
        gross_loss = abs(sum(losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
        
        total_pnl = sum(pnls)
        avg_pnl = total_pnl / total if total > 0 else 0
        
        # Expectancy = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)
        avg_win = sum(winners) / wins if wins > 0 else 0
        avg_loss = abs(sum(losers) / len(losers)) if losers else 0
        loss_rate = (len(losers) / total) if total > 0 else 0
        expectancy = (win_rate / 100 * avg_win) - (loss_rate * avg_loss)
        
        return {
            "total_trades": total,
            "wins": wins,
            "losses": len(losers),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "profit_factor": profit_factor,
            "avg_pnl": avg_pnl,
            "avg_winner": avg_win,
            "avg_loser": avg_loss,
            "best_trade": max(pnls) if pnls else 0,
            "worst_trade": min(pnls) if pnls else 0,
            "expectancy": expectancy,
        }
    
    def get_top_setups(self, limit: int = 10, min_trades: int = 5) -> List[SetupStats]:
        """
        Get top performing setups/patterns.
        
        Args:
            limit: Maximum number of setups to return
            min_trades: Minimum trades required for inclusion
            
        Returns:
            List of SetupStats sorted by expectancy
        """
        # Group by pattern/setup
        by_setup: Dict[str, List[Dict]] = defaultdict(list)
        
        for trade in self.history:
            # Try to get pattern name from various fields
            pattern = (
                trade.get("pattern_name") or
                trade.get("extra", {}).get("pattern_name") or
                trade.get("extra", {}).get("strategy") or
                trade.get("bot_name", "Unknown")
            )
            by_setup[pattern].append(trade)
        
        # Calculate stats for each setup
        results = []
        for name, trades in by_setup.items():
            if len(trades) < min_trades:
                continue
            
            stats = self._calculate_setup_stats(name, trades)
            results.append(stats)
        
        # Sort by expectancy (expected value per trade)
        results.sort(key=lambda x: x.expectancy, reverse=True)
        
        return results[:limit]
    
    def _calculate_setup_stats(self, name: str, trades: List[Dict]) -> SetupStats:
        """Calculate statistics for a setup."""
        total = len(trades)
        pnls = [t.get("realized_pnl_pct", t.get("pnl_pct", 0)) for t in trades]
        
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]
        
        wins = len(winners)
        losses = len(losers)
        win_rate = (wins / total) * 100 if total > 0 else 0
        
        total_pnl = sum(pnls)
        avg_pnl = total_pnl / total if total > 0 else 0
        
        avg_winner = sum(winners) / wins if wins > 0 else 0
        avg_loser = abs(sum(losers) / losses) if losses > 0 else 0
        
        gross_profit = sum(winners)
        gross_loss = abs(sum(losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
        
        # Expectancy
        loss_rate = losses / total if total > 0 else 0
        expectancy = (win_rate / 100 * avg_winner) - (loss_rate * avg_loser)
        
        # Average duration
        durations = []
        for t in trades:
            created = self._parse_datetime(t.get("created_at", ""))
            closed = self._parse_datetime(t.get("closed_at", ""))
            if created and closed:
                duration_hours = (closed - created).total_seconds() / 3600
                durations.append(duration_hours)
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return SetupStats(
            name=name,
            total_trades=total,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_pnl=avg_pnl,
            profit_factor=profit_factor,
            avg_winner=avg_winner,
            avg_loser=avg_loser,
            best_trade=max(pnls) if pnls else 0,
            worst_trade=min(pnls) if pnls else 0,
            avg_duration_hours=avg_duration,
            expectancy=expectancy,
        )
    
    def get_hourly_performance(self) -> List[TimeSlotStats]:
        """
        Get performance broken down by hour of day (UTC).
        
        Useful for identifying best/worst trading hours.
        """
        by_hour: Dict[int, List[Dict]] = defaultdict(list)
        
        for trade in self.history:
            created = self._parse_datetime(trade.get("created_at", ""))
            if created:
                by_hour[created.hour].append(trade)
        
        results = []
        for hour in range(24):
            trades = by_hour.get(hour, [])
            if not trades:
                continue
            
            total = len(trades)
            pnls = [t.get("realized_pnl_pct", t.get("pnl_pct", 0)) for t in trades]
            wins = sum(1 for p in pnls if p > 0)
            
            results.append(TimeSlotStats(
                slot=f"{hour:02d}:00",
                total_trades=total,
                win_rate=(wins / total) * 100 if total > 0 else 0,
                avg_pnl=sum(pnls) / total if total > 0 else 0,
                total_pnl=sum(pnls),
            ))
        
        return results
    
    def get_daily_performance(self) -> List[TimeSlotStats]:
        """
        Get performance broken down by day of week.
        
        Useful for identifying best/worst trading days.
        """
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        by_day: Dict[int, List[Dict]] = defaultdict(list)
        
        for trade in self.history:
            created = self._parse_datetime(trade.get("created_at", ""))
            if created:
                by_day[created.weekday()].append(trade)
        
        results = []
        for day_idx, day_name in enumerate(days):
            trades = by_day.get(day_idx, [])
            if not trades:
                continue
            
            total = len(trades)
            pnls = [t.get("realized_pnl_pct", t.get("pnl_pct", 0)) for t in trades]
            wins = sum(1 for p in pnls if p > 0)
            
            results.append(TimeSlotStats(
                slot=day_name,
                total_trades=total,
                win_rate=(wins / total) * 100 if total > 0 else 0,
                avg_pnl=sum(pnls) / total if total > 0 else 0,
                total_pnl=sum(pnls),
            ))
        
        return results
    
    def get_symbol_rankings(self, min_trades: int = 3) -> List[SymbolRanking]:
        """
        Get symbols ranked by profitability.
        
        Args:
            min_trades: Minimum trades required for inclusion
            
        Returns:
            List of SymbolRanking sorted by profit factor
        """
        by_symbol: Dict[str, List[Dict]] = defaultdict(list)
        
        for trade in self.history:
            symbol = trade.get("symbol", "UNKNOWN")
            by_symbol[symbol].append(trade)
        
        results = []
        for symbol, trades in by_symbol.items():
            if len(trades) < min_trades:
                continue
            
            total = len(trades)
            pnls = [t.get("realized_pnl_pct", t.get("pnl_pct", 0)) for t in trades]
            
            winners = [p for p in pnls if p > 0]
            losers = [p for p in pnls if p < 0]
            
            wins = len(winners)
            gross_profit = sum(winners)
            gross_loss = abs(sum(losers))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
            
            # Get last trade date
            last_date = ""
            for t in reversed(trades):
                closed = t.get("closed_at", "")
                if closed:
                    last_date = closed[:10]  # Just the date part
                    break
            
            results.append(SymbolRanking(
                symbol=symbol,
                total_trades=total,
                wins=wins,
                win_rate=(wins / total) * 100 if total > 0 else 0,
                total_pnl=sum(pnls),
                avg_pnl=sum(pnls) / total if total > 0 else 0,
                profit_factor=profit_factor,
                last_trade_date=last_date,
            ))
        
        # Sort by profit factor
        results.sort(key=lambda x: x.profit_factor, reverse=True)
        
        return results
    
    def get_direction_comparison(self) -> Dict[str, Dict[str, float]]:
        """Compare LONG vs SHORT performance."""
        by_direction: Dict[str, List[Dict]] = {"LONG": [], "SHORT": []}
        
        for trade in self.history:
            direction = trade.get("direction", "").upper()
            if direction in ("LONG", "BULLISH", "BUY"):
                by_direction["LONG"].append(trade)
            elif direction in ("SHORT", "BEARISH", "SELL"):
                by_direction["SHORT"].append(trade)
        
        results = {}
        for direction, trades in by_direction.items():
            if not trades:
                results[direction] = {"total": 0, "win_rate": 0, "avg_pnl": 0}
                continue
            
            total = len(trades)
            pnls = [t.get("realized_pnl_pct", t.get("pnl_pct", 0)) for t in trades]
            wins = sum(1 for p in pnls if p > 0)
            
            results[direction] = {
                "total": total,
                "wins": wins,
                "win_rate": (wins / total) * 100 if total > 0 else 0,
                "total_pnl": sum(pnls),
                "avg_pnl": sum(pnls) / total if total > 0 else 0,
            }
        
        return results
    
    def get_drawdown_analysis(self) -> Dict[str, Any]:
        """
        Analyze drawdown characteristics.
        
        Returns max drawdown, average drawdown, and recovery stats.
        """
        if not self.history:
            return {"max_drawdown": 0, "avg_drawdown": 0, "current_drawdown": 0}
        
        # Sort by close time
        sorted_trades = sorted(
            self.history,
            key=lambda x: x.get("closed_at", "")
        )
        
        # Calculate equity curve
        equity = 100.0  # Start at 100
        peak = 100.0
        drawdowns = []
        current_dd = 0
        
        for trade in sorted_trades:
            pnl = trade.get("realized_pnl_pct", trade.get("pnl_pct", 0))
            equity *= (1 + pnl / 100)
            
            if equity > peak:
                peak = equity
                if current_dd > 0:
                    drawdowns.append(current_dd)
                current_dd = 0
            else:
                current_dd = (peak - equity) / peak * 100
        
        # Add final drawdown if any
        if current_dd > 0:
            drawdowns.append(current_dd)
        
        return {
            "max_drawdown": max(drawdowns) if drawdowns else 0,
            "avg_drawdown": sum(drawdowns) / len(drawdowns) if drawdowns else 0,
            "current_drawdown": current_dd,
            "drawdown_count": len(drawdowns),
            "final_equity": equity,
        }
    
    def get_recent_performance(self, days: int = 7) -> Dict[str, Any]:
        """Get performance for the last N days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        recent = []
        for trade in self.history:
            closed = self._parse_datetime(trade.get("closed_at", ""))
            if closed and closed >= cutoff:
                recent.append(trade)
        
        if not recent:
            return {
                "period": f"Last {days} days",
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
            }
        
        total = len(recent)
        pnls = [t.get("realized_pnl_pct", t.get("pnl_pct", 0)) for t in recent]
        wins = sum(1 for p in pnls if p > 0)
        
        return {
            "period": f"Last {days} days",
            "total_trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": (wins / total) * 100 if total > 0 else 0,
            "total_pnl": sum(pnls),
            "avg_pnl": sum(pnls) / total if total > 0 else 0,
        }
    
    def get_best_worst_trades(self, limit: int = 5) -> Dict[str, List[Dict]]:
        """Get the best and worst trades."""
        sorted_by_pnl = sorted(
            self.history,
            key=lambda x: x.get("realized_pnl_pct", x.get("pnl_pct", 0)),
            reverse=True
        )
        
        return {
            "best": sorted_by_pnl[:limit],
            "worst": sorted_by_pnl[-limit:] if len(sorted_by_pnl) >= limit else sorted_by_pnl[::-1][:limit],
        }
    
    def get_similar_setup_performance(
        self, 
        symbol: str, 
        direction: str, 
        pattern_name: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Get performance of similar setups for context in new signals.
        
        Args:
            symbol: Symbol to match
            direction: Direction to match
            pattern_name: Optional pattern to match
            limit: Max number of historical trades to consider
            
        Returns:
            Dict with win rate, avg P&L, and last N results
        """
        direction = direction.upper()
        if direction in ("BULLISH", "BUY"):
            direction = "LONG"
        elif direction in ("BEARISH", "SELL"):
            direction = "SHORT"
        
        matches = []
        for trade in self.history:
            trade_dir = trade.get("direction", "").upper()
            if trade_dir in ("BULLISH", "BUY"):
                trade_dir = "LONG"
            elif trade_dir in ("BEARISH", "SELL"):
                trade_dir = "SHORT"
            
            # Match symbol and direction
            if trade.get("symbol") == symbol and trade_dir == direction:
                # Optionally match pattern
                if pattern_name:
                    trade_pattern = (
                        trade.get("pattern_name") or
                        trade.get("extra", {}).get("pattern_name")
                    )
                    if trade_pattern != pattern_name:
                        continue
                
                matches.append(trade)
        
        # Sort by time (most recent first)
        matches.sort(key=lambda x: x.get("closed_at", ""), reverse=True)
        recent = matches[:limit]
        
        if not recent:
            return {
                "matches": 0,
                "win_rate": 0,
                "avg_pnl": 0,
                "last_results": [],
            }
        
        pnls = [t.get("realized_pnl_pct", t.get("pnl_pct", 0)) for t in recent]
        wins = sum(1 for p in pnls if p > 0)
        
        # Get simple result list like ["TP1", "SL", "TP2", ...]
        last_results = []
        for t in recent[:5]:
            result = t.get("close_reason", t.get("result", ""))
            last_results.append(result)
        
        return {
            "matches": len(recent),
            "win_rate": (wins / len(recent)) * 100,
            "avg_pnl": sum(pnls) / len(recent),
            "total_pnl": sum(pnls),
            "last_results": last_results,
        }
    
    def generate_insights_report(self) -> str:
        """Generate a human-readable insights report."""
        lines = ["📊 <b>Trading Performance Insights</b>", ""]
        
        # Overall stats
        overall = self.get_overall_stats()
        lines.append("<b>Overall Performance:</b>")
        lines.append(f"  Total Trades: {overall['total_trades']}")
        lines.append(f"  Win Rate: {overall['win_rate']:.1f}%")
        lines.append(f"  Total P&L: {overall['total_pnl']:+.2f}%")
        lines.append(f"  Profit Factor: {overall['profit_factor']:.2f}")
        lines.append(f"  Expectancy: {overall['expectancy']:+.2f}% per trade")
        lines.append("")
        
        # Top setups
        top_setups = self.get_top_setups(limit=3, min_trades=3)
        if top_setups:
            lines.append("<b>Top Performing Setups:</b>")
            for i, setup in enumerate(top_setups, 1):
                lines.append(f"  {i}. {setup.name}: {setup.win_rate:.0f}% WR, {setup.avg_pnl:+.2f}% avg")
            lines.append("")
        
        # Direction comparison
        direction = self.get_direction_comparison()
        lines.append("<b>Long vs Short:</b>")
        for dir_name, stats in direction.items():
            if stats["total"] > 0:
                lines.append(f"  {dir_name}: {stats['win_rate']:.0f}% WR ({stats['total']} trades)")
        lines.append("")
        
        # Best trading times
        hourly = self.get_hourly_performance()
        if hourly:
            best_hour = max(hourly, key=lambda x: x.avg_pnl) if hourly else None
            worst_hour = min(hourly, key=lambda x: x.avg_pnl) if hourly else None
            if best_hour and worst_hour:
                lines.append("<b>Best Trading Times:</b>")
                lines.append(f"  Best: {best_hour.slot} UTC ({best_hour.avg_pnl:+.2f}% avg)")
                lines.append(f"  Worst: {worst_hour.slot} UTC ({worst_hour.avg_pnl:+.2f}% avg)")
                lines.append("")
        
        # Drawdown
        dd = self.get_drawdown_analysis()
        lines.append("<b>Risk Metrics:</b>")
        lines.append(f"  Max Drawdown: {dd['max_drawdown']:.2f}%")
        lines.append(f"  Current Drawdown: {dd['current_drawdown']:.2f}%")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Load from a bot's signals file
    base_dir = Path(__file__).parent.parent
    
    analytics = SignalAnalytics.from_multiple_bots(
        ["strat_bot", "volume_bot", "harmonic_bot"],
        base_dir
    )
    
    print(analytics.generate_insights_report().replace("<b>", "").replace("</b>", ""))

