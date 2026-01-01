#!/usr/bin/env python3
"""
P&L Reporter - Daily/Weekly Performance Summaries

Provides automated performance reporting:
- Daily P&L summary with breakdown by bot
- Weekly performance reports
- Win rate and trade statistics
- Scheduled Telegram notifications
"""

import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PNL_STATE_FILE = BASE_DIR / "pnl_state.json"

# Bot stats files locations
BOT_STATS_FILES = {
    "strat_bot": BASE_DIR / "strat_bot" / "logs" / "strat_stats.json",
    "volume_bot": BASE_DIR / "volume_bot" / "logs" / "volume_stats.json",
    "harmonic_bot": BASE_DIR / "harmonic_bot" / "logs" / "harmonic_stats.json",
    "fib_swing_bot": BASE_DIR / "fib_swing_bot" / "logs" / "fib_stats.json",
    "most_bot": BASE_DIR / "most_bot" / "logs" / "most_stats.json",
    "psar_bot": BASE_DIR / "psar_bot" / "logs" / "psar_stats.json",
    "orb_bot": BASE_DIR / "orb_bot" / "logs" / "orb_stats.json",
    "mtf_bot": BASE_DIR / "mtf_bot" / "logs" / "mtf_stats.json",
    "liquidation_bot": BASE_DIR / "liquidation_bot" / "logs" / "liquidation_stats.json",
    "funding_bot": BASE_DIR / "funding_bot" / "logs" / "funding_stats.json",
    "diy_bot": BASE_DIR / "diy_bot" / "logs" / "diy_stats.json",
    "volume_profile_bot": BASE_DIR / "logs" / "volume_profile_stats.json",
}


@dataclass
class TradeRecord:
    """Individual trade record."""
    signal_id: str
    bot_name: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    result: str  # TP1, TP2, TP3, SL, EXPIRED, EMERGENCY_SL
    pnl_pct: float
    opened_at: str
    closed_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeRecord":
        return cls(**data)


@dataclass
class DailySummary:
    """Daily trading summary."""
    date: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl_pct: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    trades_by_bot: Dict[str, int] = field(default_factory=dict)
    pnl_by_bot: Dict[str, float] = field(default_factory=dict)
    trades_by_result: Dict[str, int] = field(default_factory=dict)
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DailySummary":
        return cls(**data)


@dataclass
class PnLConfig:
    """P&L Reporter configuration."""
    # When to send daily summary (UTC hour)
    daily_summary_hour: int = 0  # Midnight UTC
    # When to send weekly summary (day of week, 0=Monday)
    weekly_summary_day: int = 0  # Monday
    # Enable automatic daily summaries
    enable_daily_summary: bool = True
    # Enable automatic weekly summaries
    enable_weekly_summary: bool = True
    # Minimum trades for summary
    min_trades_for_summary: int = 1
    # Alert on significant loss
    loss_alert_threshold_pct: float = -5.0
    # Alert on significant win
    win_alert_threshold_pct: float = 10.0


class PnLReporter:
    """
    Automated P&L reporting and summaries.
    
    Features:
    - Aggregate trades from all bot stats files
    - Calculate daily/weekly/monthly performance
    - Send scheduled Telegram summaries
    - Track performance trends
    
    Usage:
        reporter = PnLReporter()
        reporter.set_notifier(telegram_notifier)
        reporter.start_scheduler()
    """
    
    def __init__(self, config: Optional[PnLConfig] = None):
        self.config = config or PnLConfig()
        self.notifier: Optional[Any] = None
        self.trades: List[TradeRecord] = []
        self.daily_summaries: Dict[str, DailySummary] = {}
        self.state_lock = threading.RLock()
        self.scheduler_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.last_daily_summary_date: Optional[str] = None
        self.last_weekly_summary_date: Optional[str] = None
        
        # Load state
        self._load_state()
        
        # Initial sync from bot stats
        self.sync_from_bot_stats()
        
        logger.info("PnLReporter initialized with %d trades", len(self.trades))
    
    def set_notifier(self, notifier: Any) -> None:
        """Set the Telegram notifier."""
        self.notifier = notifier
    
    def _load_state(self) -> None:
        """Load P&L state from file."""
        if not PNL_STATE_FILE.exists():
            return
        
        try:
            with open(PNL_STATE_FILE, 'r') as f:
                data = json.load(f)
            
            # Load trades
            for trade_data in data.get("trades", []):
                try:
                    self.trades.append(TradeRecord.from_dict(trade_data))
                except (TypeError, KeyError):
                    pass
            
            # Load daily summaries
            for date_str, summary_data in data.get("daily_summaries", {}).items():
                try:
                    self.daily_summaries[date_str] = DailySummary.from_dict(summary_data)
                except (TypeError, KeyError):
                    pass
            
            self.last_daily_summary_date = data.get("last_daily_summary_date")
            self.last_weekly_summary_date = data.get("last_weekly_summary_date")
            
        except (json.JSONDecodeError, IOError) as e:
            logger.error("Failed to load P&L state: %s", e)
    
    def _save_state(self) -> None:
        """Save P&L state to file."""
        temp_file = PNL_STATE_FILE.with_suffix('.tmp')
        
        try:
            # Keep only last 1000 trades to prevent unbounded growth
            recent_trades = self.trades[-1000:]
            
            # Keep only last 90 days of summaries
            cutoff = (datetime.now(timezone.utc) - timedelta(days=90)).date().isoformat()
            recent_summaries = {k: v.to_dict() for k, v in self.daily_summaries.items() if k >= cutoff}
            
            data = {
                "trades": [t.to_dict() for t in recent_trades],
                "daily_summaries": recent_summaries,
                "last_daily_summary_date": self.last_daily_summary_date,
                "last_weekly_summary_date": self.last_weekly_summary_date,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            temp_file.replace(PNL_STATE_FILE)
            
        except Exception as e:
            logger.error("Failed to save P&L state: %s", e)
            if temp_file.exists():
                temp_file.unlink()
    
    def sync_from_bot_stats(self) -> int:
        """
        Sync trades from bot stats files.
        
        Returns:
            Number of new trades synced
        """
        synced = 0
        existing_ids = {t.signal_id for t in self.trades}
        
        for bot_name, stats_file in BOT_STATS_FILES.items():
            if not stats_file.exists():
                continue
            
            try:
                with open(stats_file, 'r') as f:
                    data = json.load(f)
                
                history = data.get("history", [])
                for entry in history:
                    signal_id = entry.get("signal_id", "")
                    if signal_id in existing_ids:
                        continue
                    
                    # Extract trade data
                    try:
                        trade = TradeRecord(
                            signal_id=signal_id,
                            bot_name=bot_name,
                            symbol=entry.get("symbol", "UNKNOWN"),
                            direction=entry.get("direction", "LONG"),
                            entry_price=float(entry.get("entry", 0)),
                            exit_price=float(entry.get("exit_price", 0)),
                            result=entry.get("result", "UNKNOWN"),
                            pnl_pct=float(entry.get("pnl_pct", 0)),
                            opened_at=entry.get("opened_at", ""),
                            closed_at=entry.get("closed_at", ""),
                        )
                        
                        self.trades.append(trade)
                        existing_ids.add(signal_id)
                        synced += 1
                        
                    except (TypeError, ValueError, KeyError) as e:
                        logger.debug("Failed to parse trade from %s: %s", bot_name, e)
                        
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Failed to read stats from %s: %s", stats_file, e)
        
        if synced > 0:
            self._save_state()
            logger.info("Synced %d new trades from bot stats", synced)
        
        return synced
    
    def record_trade(self, trade: TradeRecord) -> None:
        """Record a new trade."""
        with self.state_lock:
            self.trades.append(trade)
            self._update_daily_summary(trade)
            self._save_state()
            
            # Check for alerts
            self._check_trade_alerts(trade)
    
    def _update_daily_summary(self, trade: TradeRecord) -> None:
        """Update daily summary with new trade."""
        # Get trade date
        try:
            closed_dt = datetime.fromisoformat(trade.closed_at.replace('Z', '+00:00'))
            trade_date = closed_dt.date().isoformat()
        except (ValueError, AttributeError):
            trade_date = datetime.now(timezone.utc).date().isoformat()
        
        if trade_date not in self.daily_summaries:
            self.daily_summaries[trade_date] = DailySummary(date=trade_date)
        
        summary = self.daily_summaries[trade_date]
        summary.total_trades += 1
        summary.total_pnl_pct += trade.pnl_pct
        
        is_win = trade.result in ["TP1", "TP2", "TP3"]
        if is_win:
            summary.winning_trades += 1
        else:
            summary.losing_trades += 1
        
        summary.best_trade_pnl = max(summary.best_trade_pnl, trade.pnl_pct)
        summary.worst_trade_pnl = min(summary.worst_trade_pnl, trade.pnl_pct)
        
        # Update by bot
        summary.trades_by_bot[trade.bot_name] = summary.trades_by_bot.get(trade.bot_name, 0) + 1
        summary.pnl_by_bot[trade.bot_name] = summary.pnl_by_bot.get(trade.bot_name, 0) + trade.pnl_pct
        
        # Update by result
        summary.trades_by_result[trade.result] = summary.trades_by_result.get(trade.result, 0) + 1
    
    def _check_trade_alerts(self, trade: TradeRecord) -> None:
        """Check if trade triggers any alerts."""
        if not self.notifier:
            return
        
        if trade.pnl_pct >= self.config.win_alert_threshold_pct:
            self._send_alert(
                f"🎉 <b>Big Win Alert!</b>\n\n"
                f"<b>Bot:</b> {trade.bot_name}\n"
                f"<b>Symbol:</b> {trade.symbol}\n"
                f"<b>Result:</b> {trade.result}\n"
                f"<b>P&L:</b> +{trade.pnl_pct:.2f}%"
            )
        elif trade.pnl_pct <= self.config.loss_alert_threshold_pct:
            self._send_alert(
                f"⚠️ <b>Significant Loss Alert</b>\n\n"
                f"<b>Bot:</b> {trade.bot_name}\n"
                f"<b>Symbol:</b> {trade.symbol}\n"
                f"<b>Result:</b> {trade.result}\n"
                f"<b>P&L:</b> {trade.pnl_pct:.2f}%"
            )
    
    def get_daily_summary(self, date_str: Optional[str] = None) -> DailySummary:
        """Get summary for a specific date (default: today)."""
        if date_str is None:
            date_str = datetime.now(timezone.utc).date().isoformat()
        
        with self.state_lock:
            if date_str in self.daily_summaries:
                return self.daily_summaries[date_str]
            
            # Calculate from trades if not cached
            summary = DailySummary(date=date_str)
            
            for trade in self.trades:
                try:
                    closed_dt = datetime.fromisoformat(trade.closed_at.replace('Z', '+00:00'))
                    if closed_dt.date().isoformat() == date_str:
                        summary.total_trades += 1
                        summary.total_pnl_pct += trade.pnl_pct
                        
                        if trade.result in ["TP1", "TP2", "TP3"]:
                            summary.winning_trades += 1
                        else:
                            summary.losing_trades += 1
                        
                        summary.best_trade_pnl = max(summary.best_trade_pnl, trade.pnl_pct)
                        summary.worst_trade_pnl = min(summary.worst_trade_pnl, trade.pnl_pct)
                        
                        summary.trades_by_bot[trade.bot_name] = summary.trades_by_bot.get(trade.bot_name, 0) + 1
                        summary.pnl_by_bot[trade.bot_name] = summary.pnl_by_bot.get(trade.bot_name, 0) + trade.pnl_pct
                        summary.trades_by_result[trade.result] = summary.trades_by_result.get(trade.result, 0) + 1
                        
                except (ValueError, AttributeError):
                    pass
            
            return summary
    
    def get_weekly_summary(self) -> Dict[str, Any]:
        """Get summary for the current week."""
        with self.state_lock:
            now = datetime.now(timezone.utc)
            week_start = now - timedelta(days=now.weekday())
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
            
            total_trades = 0
            winning_trades = 0
            total_pnl = 0.0
            trades_by_bot: Dict[str, int] = {}
            pnl_by_bot: Dict[str, float] = {}
            trades_by_result: Dict[str, int] = {}
            best_trade = 0.0
            worst_trade = 0.0
            
            for trade in self.trades:
                try:
                    closed_dt = datetime.fromisoformat(trade.closed_at.replace('Z', '+00:00'))
                    if closed_dt.tzinfo is None:
                        closed_dt = closed_dt.replace(tzinfo=timezone.utc)
                    
                    if closed_dt >= week_start:
                        total_trades += 1
                        total_pnl += trade.pnl_pct
                        
                        if trade.result in ["TP1", "TP2", "TP3"]:
                            winning_trades += 1
                        
                        best_trade = max(best_trade, trade.pnl_pct)
                        worst_trade = min(worst_trade, trade.pnl_pct)
                        
                        trades_by_bot[trade.bot_name] = trades_by_bot.get(trade.bot_name, 0) + 1
                        pnl_by_bot[trade.bot_name] = pnl_by_bot.get(trade.bot_name, 0) + trade.pnl_pct
                        trades_by_result[trade.result] = trades_by_result.get(trade.result, 0) + 1
                        
                except (ValueError, AttributeError):
                    pass
            
            return {
                "week_start": week_start.date().isoformat(),
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": total_trades - winning_trades,
                "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                "total_pnl_pct": total_pnl,
                "best_trade_pnl": best_trade,
                "worst_trade_pnl": worst_trade,
                "trades_by_bot": trades_by_bot,
                "pnl_by_bot": pnl_by_bot,
                "trades_by_result": trades_by_result,
            }
    
    def format_daily_summary_message(self, summary: Optional[DailySummary] = None) -> str:
        """Format daily summary for Telegram."""
        if summary is None:
            summary = self.get_daily_summary()
        
        if summary.total_trades == 0:
            return f"📊 <b>Daily Summary - {summary.date}</b>\n\nNo trades today."
        
        pnl_emoji = "📈" if summary.total_pnl_pct >= 0 else "📉"
        
        lines = [
            f"📊 <b>Daily Summary - {summary.date}</b>",
            "",
            f"<b>Overall:</b>",
            f"  Trades: {summary.total_trades}",
            f"  Win Rate: {summary.win_rate:.1f}% ({summary.winning_trades}W / {summary.losing_trades}L)",
            f"  {pnl_emoji} P&L: {summary.total_pnl_pct:+.2f}%",
            f"  Best: {summary.best_trade_pnl:+.2f}%",
            f"  Worst: {summary.worst_trade_pnl:+.2f}%",
        ]
        
        if summary.pnl_by_bot:
            lines.append("")
            lines.append("<b>By Bot:</b>")
            for bot, pnl in sorted(summary.pnl_by_bot.items(), key=lambda x: x[1], reverse=True):
                trades = summary.trades_by_bot.get(bot, 0)
                emoji = "🟢" if pnl >= 0 else "🔴"
                lines.append(f"  {emoji} {bot}: {pnl:+.2f}% ({trades} trades)")
        
        if summary.trades_by_result:
            lines.append("")
            lines.append("<b>Results:</b>")
            result_order = ["TP1", "TP2", "TP3", "SL", "EXPIRED", "EMERGENCY_SL"]
            for result in result_order:
                if result in summary.trades_by_result:
                    count = summary.trades_by_result[result]
                    emoji = "✅" if result.startswith("TP") else "❌"
                    lines.append(f"  {emoji} {result}: {count}")
        
        return "\n".join(lines)
    
    def format_weekly_summary_message(self) -> str:
        """Format weekly summary for Telegram."""
        summary = self.get_weekly_summary()
        
        if summary["total_trades"] == 0:
            return f"📊 <b>Weekly Summary</b>\n\nNo trades this week."
        
        pnl_emoji = "📈" if summary["total_pnl_pct"] >= 0 else "📉"
        
        lines = [
            f"📊 <b>Weekly Summary</b>",
            f"Week starting: {summary['week_start']}",
            "",
            f"<b>Overall Performance:</b>",
            f"  Total Trades: {summary['total_trades']}",
            f"  Win Rate: {summary['win_rate']:.1f}%",
            f"  {pnl_emoji} Total P&L: {summary['total_pnl_pct']:+.2f}%",
            f"  Best Trade: {summary['best_trade_pnl']:+.2f}%",
            f"  Worst Trade: {summary['worst_trade_pnl']:+.2f}%",
        ]
        
        if summary["pnl_by_bot"]:
            lines.append("")
            lines.append("<b>Bot Performance:</b>")
            for bot, pnl in sorted(summary["pnl_by_bot"].items(), key=lambda x: x[1], reverse=True):
                trades = summary["trades_by_bot"].get(bot, 0)
                emoji = "🏆" if pnl == max(summary["pnl_by_bot"].values()) else ("🟢" if pnl >= 0 else "🔴")
                lines.append(f"  {emoji} {bot}: {pnl:+.2f}% ({trades} trades)")
        
        return "\n".join(lines)
    
    def send_daily_summary(self) -> bool:
        """Send daily summary notification."""
        if not self.notifier:
            logger.warning("No notifier set, cannot send daily summary")
            return False
        
        # Sync latest trades
        self.sync_from_bot_stats()
        
        # Get yesterday's summary (most useful at start of new day)
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
        summary = self.get_daily_summary(yesterday)
        
        if summary.total_trades < self.config.min_trades_for_summary:
            logger.info("Not enough trades for daily summary (%d < %d)",
                       summary.total_trades, self.config.min_trades_for_summary)
            return False
        
        message = self.format_daily_summary_message(summary)
        
        try:
            self.notifier.send_message(message, parse_mode="HTML")
            self.last_daily_summary_date = yesterday
            self._save_state()
            logger.info("Sent daily summary for %s", yesterday)
            return True
        except Exception as e:
            logger.error("Failed to send daily summary: %s", e)
            return False
    
    def send_weekly_summary(self) -> bool:
        """Send weekly summary notification."""
        if not self.notifier:
            return False
        
        # Sync latest trades
        self.sync_from_bot_stats()
        
        message = self.format_weekly_summary_message()
        
        try:
            self.notifier.send_message(message, parse_mode="HTML")
            self.last_weekly_summary_date = datetime.now(timezone.utc).date().isoformat()
            self._save_state()
            logger.info("Sent weekly summary")
            return True
        except Exception as e:
            logger.error("Failed to send weekly summary: %s", e)
            return False
    
    def _scheduler_loop(self) -> None:
        """Background thread for scheduled reports."""
        while self.scheduler_running:
            try:
                now = datetime.now(timezone.utc)
                
                # Check for daily summary
                if self.config.enable_daily_summary:
                    if now.hour == self.config.daily_summary_hour:
                        today = now.date().isoformat()
                        if self.last_daily_summary_date != today:
                            self.send_daily_summary()
                
                # Check for weekly summary
                if self.config.enable_weekly_summary:
                    if now.weekday() == self.config.weekly_summary_day and now.hour == self.config.daily_summary_hour:
                        today = now.date().isoformat()
                        if self.last_weekly_summary_date != today:
                            self.send_weekly_summary()
                
            except Exception as e:
                logger.error("Scheduler error: %s", e)
            
            # Sleep for 1 minute
            for _ in range(60):
                if not self.scheduler_running:
                    break
                time.sleep(1)
    
    def start_scheduler(self) -> None:
        """Start the scheduled report background thread."""
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            return
        
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("P&L report scheduler started")
    
    def stop_scheduler(self) -> None:
        """Stop the scheduler thread."""
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=10)
        logger.info("P&L report scheduler stopped")
    
    def _send_alert(self, message: str) -> None:
        """Send alert via notifier."""
        if self.notifier:
            try:
                self.notifier.send_message(message, parse_mode="HTML")
            except Exception as e:
                logger.error("Failed to send alert: %s", e)


def get_pnl_reporter() -> PnLReporter:
    """Get or create the global PnLReporter instance."""
    if not hasattr(get_pnl_reporter, '_instance'):
        get_pnl_reporter._instance = PnLReporter()
    return get_pnl_reporter._instance


if __name__ == "__main__":
    # Example usage / CLI
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    parser = argparse.ArgumentParser(description="P&L Reporter")
    parser.add_argument("command", choices=["daily", "weekly", "sync", "status"])
    args = parser.parse_args()
    
    reporter = get_pnl_reporter()
    
    if args.command == "daily":
        message = reporter.format_daily_summary_message()
        print(message.replace("<b>", "").replace("</b>", ""))
    
    elif args.command == "weekly":
        message = reporter.format_weekly_summary_message()
        print(message.replace("<b>", "").replace("</b>", ""))
    
    elif args.command == "sync":
        count = reporter.sync_from_bot_stats()
        print(f"Synced {count} new trades")
    
    elif args.command == "status":
        print(f"Total trades tracked: {len(reporter.trades)}")
        print(f"Daily summaries cached: {len(reporter.daily_summaries)}")
        print(f"Last daily summary: {reporter.last_daily_summary_date}")
        print(f"Last weekly summary: {reporter.last_weekly_summary_date}")

