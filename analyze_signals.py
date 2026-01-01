#!/usr/bin/env python3
"""
Signal Performance Analyzer - Analyzes all bot signals across state files.

Usage:
    python analyze_signals.py              # Full analysis
    python analyze_signals.py --bot diy    # Analyze specific bot
    python analyze_signals.py --json       # Output as JSON
    python analyze_signals.py --csv        # Export to CSV
"""

import argparse
import json
import csv
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

BASE_DIR = Path(__file__).resolve().parent

# Bot state file mappings
BOT_STATE_FILES = {
    "diy_bot": BASE_DIR / "diy_bot" / "diy_state.json",
    "fib_swing_bot": BASE_DIR / "fib_swing_bot" / "fib_state.json",
    "funding_bot": BASE_DIR / "funding_bot" / "funding_state.json",
    "harmonic_bot": BASE_DIR / "harmonic_bot" / "harmonic_state.json",
    "liquidation_bot": BASE_DIR / "liquidation_bot" / "liquidation_state.json",
    "most_bot": BASE_DIR / "most_bot" / "most_state.json",
    "mtf_bot": BASE_DIR / "mtf_bot" / "mtf_state.json",
    "orb_bot": BASE_DIR / "orb_bot" / "orb_state.json",
    "psar_bot": BASE_DIR / "psar_bot" / "psar_state.json",
    "strat_bot": BASE_DIR / "strat_bot" / "strat_state.json",
    "volume_bot": BASE_DIR / "volume_bot" / "volume_vn_state.json",
    "volume_profile_bot": BASE_DIR / "volume_profile_bot" / "vp_state.json",
}

# Also check stats files for historical data
BOT_STATS_FILES = {
    "diy_bot": BASE_DIR / "diy_bot" / "logs" / "diy_stats.json",
    "fib_swing_bot": BASE_DIR / "fib_swing_bot" / "logs" / "fib_swing_stats.json",
    "funding_bot": BASE_DIR / "funding_bot" / "logs" / "funding_stats.json",
    "harmonic_bot": BASE_DIR / "harmonic_bot" / "logs" / "harmonic_stats.json",
    "liquidation_bot": BASE_DIR / "liquidation_bot" / "logs" / "liquidation_stats.json",
    "most_bot": BASE_DIR / "most_bot" / "logs" / "most_stats.json",
    "mtf_bot": BASE_DIR / "mtf_bot" / "logs" / "mtf_stats.json",
    "orb_bot": BASE_DIR / "orb_bot" / "logs" / "orb_stats.json",
    "psar_bot": BASE_DIR / "psar_bot" / "logs" / "psar_stats.json",
    "strat_bot": BASE_DIR / "strat_bot" / "logs" / "strat_stats.json",
    "volume_bot": BASE_DIR / "volume_bot" / "logs" / "volume_stats.json",
    "volume_profile_bot": BASE_DIR / "logs" / "volume_profile_stats.json",
}


@dataclass
class SignalRecord:
    """Normalized signal record for analysis."""
    bot: str
    signal_id: str
    symbol: str
    direction: str
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: Optional[float]
    result: Optional[str]
    exit_price: Optional[float]
    pnl_percent: Optional[float]
    created_at: Optional[str]
    closed_at: Optional[str]
    is_open: bool
    exchange: Optional[str]
    timeframe: Optional[str]


@dataclass
class BotStats:
    """Statistics for a single bot."""
    bot_name: str
    total_signals: int
    open_signals: int
    closed_signals: int
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
    expired: int
    avg_win: float
    avg_loss: float
    profit_factor: float
    symbols_traded: int
    most_traded_symbol: str
    best_symbol: Tuple[str, float]
    worst_symbol: Tuple[str, float]


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


def extract_signals_from_state(bot_name: str, state: Dict[str, Any]) -> List[SignalRecord]:
    """Extract signals from a bot's state file."""
    signals = []

    # Process open signals
    open_signals = state.get("open_signals", {})
    if isinstance(open_signals, dict):
        for sig_id, data in open_signals.items():
            if not isinstance(data, dict):
                continue
            signals.append(SignalRecord(
                bot=bot_name,
                signal_id=sig_id,
                symbol=data.get("symbol", "UNKNOWN"),
                direction=normalize_direction(data.get("direction") or data.get("type")),
                entry=float(data.get("entry", 0) or 0),
                stop_loss=float(data.get("stop_loss") or data.get("sl") or 0),
                tp1=float(data.get("take_profit_1") or data.get("tp1") or 0),
                tp2=float(data.get("take_profit_2") or data.get("tp2") or 0),
                tp3=float(data.get("take_profit_3") or data.get("tp3") or 0) if data.get("take_profit_3") or data.get("tp3") else None,
                result=None,
                exit_price=None,
                pnl_percent=None,
                created_at=data.get("created_at") or data.get("timestamp"),
                closed_at=None,
                is_open=True,
                exchange=data.get("exchange"),
                timeframe=data.get("timeframe"),
            ))

    # Process closed signals
    closed_signals = state.get("closed_signals", {})
    if isinstance(closed_signals, dict):
        for sig_id, data in closed_signals.items():
            if not isinstance(data, dict):
                continue
            signals.append(SignalRecord(
                bot=bot_name,
                signal_id=sig_id,
                symbol=data.get("symbol", "UNKNOWN"),
                direction=normalize_direction(data.get("direction") or data.get("type")),
                entry=float(data.get("entry", 0) or 0),
                stop_loss=float(data.get("stop_loss") or data.get("sl") or 0),
                tp1=float(data.get("take_profit_1") or data.get("tp1") or 0),
                tp2=float(data.get("take_profit_2") or data.get("tp2") or 0),
                tp3=float(data.get("take_profit_3") or data.get("tp3") or 0) if data.get("take_profit_3") or data.get("tp3") else None,
                result=data.get("result") or data.get("closed_reason"),
                exit_price=float(data.get("exit_price", 0) or 0),
                pnl_percent=float(data.get("pnl_percent") or data.get("pnl_pct") or 0),
                created_at=data.get("created_at") or data.get("timestamp"),
                closed_at=data.get("closed_at"),
                is_open=False,
                exchange=data.get("exchange"),
                timeframe=data.get("timeframe"),
            ))

    return signals


def extract_signals_from_stats(bot_name: str, stats: Dict[str, Any]) -> List[SignalRecord]:
    """Extract signals from a bot's stats file."""
    signals = []

    # Stats files typically have a "signals" or "history" key
    history = stats.get("signals", {}) or stats.get("history", {}) or stats.get("closed", {})
    if isinstance(history, dict):
        for sig_id, data in history.items():
            if not isinstance(data, dict):
                continue
            # Skip if already processed (check for result)
            if not data.get("result") and not data.get("exit_price"):
                continue
            signals.append(SignalRecord(
                bot=bot_name,
                signal_id=sig_id,
                symbol=data.get("symbol", "UNKNOWN"),
                direction=normalize_direction(data.get("direction") or data.get("type")),
                entry=float(data.get("entry", 0) or 0),
                stop_loss=float(data.get("stop_loss") or data.get("sl") or 0),
                tp1=float(data.get("tp1") or 0),
                tp2=float(data.get("tp2") or 0),
                tp3=float(data.get("tp3") or 0) if data.get("tp3") else None,
                result=data.get("result"),
                exit_price=float(data.get("exit_price", 0) or 0),
                pnl_percent=float(data.get("pnl_pct") or data.get("pnl_percent") or 0),
                created_at=data.get("opened_at") or data.get("timestamp"),
                closed_at=data.get("closed_at"),
                is_open=False,
                exchange=data.get("exchange"),
                timeframe=data.get("timeframe"),
            ))

    return signals


def calculate_bot_stats(bot_name: str, signals: List[SignalRecord]) -> BotStats:
    """Calculate statistics for a bot."""
    closed = [s for s in signals if not s.is_open and s.result]
    open_sigs = [s for s in signals if s.is_open]

    # Count results
    tp1_hits = len([s for s in closed if s.result and "TP1" in s.result.upper()])
    tp2_hits = len([s for s in closed if s.result and "TP2" in s.result.upper()])
    tp3_hits = len([s for s in closed if s.result and "TP3" in s.result.upper()])
    sl_hits = len([s for s in closed if s.result and "SL" in s.result.upper()])
    expired = len([s for s in closed if s.result and ("EXPIRED" in s.result.upper() or "TIMEOUT" in s.result.upper())])

    wins = tp1_hits + tp2_hits + tp3_hits
    losses = sl_hits

    # PnL calculations
    pnls = [s.pnl_percent for s in closed if s.pnl_percent is not None]
    win_pnls = [p for p in pnls if p > 0]
    loss_pnls = [p for p in pnls if p < 0]

    total_pnl = sum(pnls) if pnls else 0
    avg_pnl = total_pnl / len(pnls) if pnls else 0
    best_trade = max(pnls) if pnls else 0
    worst_trade = min(pnls) if pnls else 0
    avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0
    avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0

    # Profit factor
    gross_profit = sum(win_pnls) if win_pnls else 0
    gross_loss = abs(sum(loss_pnls)) if loss_pnls else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

    # Symbol analysis
    symbol_pnl: Dict[str, List[float]] = defaultdict(list)
    symbol_count: Dict[str, int] = defaultdict(int)
    for s in closed:
        symbol_count[s.symbol] += 1
        if s.pnl_percent is not None:
            symbol_pnl[s.symbol].append(s.pnl_percent)

    symbols_traded = len(symbol_count)
    most_traded = max(symbol_count.items(), key=lambda x: x[1])[0] if symbol_count else "N/A"

    # Best/worst symbols
    symbol_avg_pnl = {sym: sum(pnls)/len(pnls) for sym, pnls in symbol_pnl.items() if pnls}
    best_symbol = max(symbol_avg_pnl.items(), key=lambda x: x[1]) if symbol_avg_pnl else ("N/A", 0)
    worst_symbol = min(symbol_avg_pnl.items(), key=lambda x: x[1]) if symbol_avg_pnl else ("N/A", 0)

    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

    return BotStats(
        bot_name=bot_name,
        total_signals=len(signals),
        open_signals=len(open_sigs),
        closed_signals=len(closed),
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        total_pnl=total_pnl,
        avg_pnl=avg_pnl,
        best_trade=best_trade,
        worst_trade=worst_trade,
        tp1_hits=tp1_hits,
        tp2_hits=tp2_hits,
        tp3_hits=tp3_hits,
        sl_hits=sl_hits,
        expired=expired,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        symbols_traded=symbols_traded,
        most_traded_symbol=most_traded,
        best_symbol=best_symbol,
        worst_symbol=worst_symbol,
    )


def print_bot_stats(stats: BotStats) -> None:
    """Print formatted stats for a bot."""
    print(f"\n{'='*60}")
    print(f"  {stats.bot_name.upper()}")
    print(f"{'='*60}")

    print(f"\n  SIGNAL COUNTS")
    print(f"  {'─'*40}")
    print(f"  Total Signals:    {stats.total_signals:>8}")
    print(f"  Open Signals:     {stats.open_signals:>8}")
    print(f"  Closed Signals:   {stats.closed_signals:>8}")

    if stats.closed_signals > 0:
        print(f"\n  RESULTS")
        print(f"  {'─'*40}")
        print(f"  TP1 Hits:         {stats.tp1_hits:>8}")
        print(f"  TP2 Hits:         {stats.tp2_hits:>8}")
        print(f"  TP3 Hits:         {stats.tp3_hits:>8}")
        print(f"  SL Hits:          {stats.sl_hits:>8}")
        print(f"  Expired:          {stats.expired:>8}")

        print(f"\n  PERFORMANCE")
        print(f"  {'─'*40}")
        print(f"  Win Rate:         {stats.win_rate:>7.1f}%")
        print(f"  Total PnL:        {stats.total_pnl:>+7.2f}%")
        print(f"  Avg PnL:          {stats.avg_pnl:>+7.2f}%")
        print(f"  Best Trade:       {stats.best_trade:>+7.2f}%")
        print(f"  Worst Trade:      {stats.worst_trade:>+7.2f}%")
        print(f"  Avg Win:          {stats.avg_win:>+7.2f}%")
        print(f"  Avg Loss:         {stats.avg_loss:>+7.2f}%")
        pf_str = f"{stats.profit_factor:.2f}" if stats.profit_factor != float('inf') else "∞"
        print(f"  Profit Factor:    {pf_str:>8}")

        print(f"\n  SYMBOLS")
        print(f"  {'─'*40}")
        print(f"  Symbols Traded:   {stats.symbols_traded:>8}")
        print(f"  Most Traded:      {stats.most_traded_symbol:>8}")
        print(f"  Best Symbol:      {stats.best_symbol[0]:>8} ({stats.best_symbol[1]:+.2f}%)")
        print(f"  Worst Symbol:     {stats.worst_symbol[0]:>8} ({stats.worst_symbol[1]:+.2f}%)")


def print_overall_summary(all_stats: List[BotStats], all_signals: List[SignalRecord]) -> None:
    """Print overall summary across all bots."""
    print(f"\n{'='*60}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*60}")

    total_signals = sum(s.total_signals for s in all_stats)
    total_open = sum(s.open_signals for s in all_stats)
    total_closed = sum(s.closed_signals for s in all_stats)
    total_wins = sum(s.wins for s in all_stats)
    total_losses = sum(s.losses for s in all_stats)
    total_pnl = sum(s.total_pnl for s in all_stats)

    print(f"\n  TOTALS")
    print(f"  {'─'*40}")
    print(f"  Total Bots:       {len(all_stats):>8}")
    print(f"  Total Signals:    {total_signals:>8}")
    print(f"  Open Signals:     {total_open:>8}")
    print(f"  Closed Signals:   {total_closed:>8}")
    print(f"  Total Wins:       {total_wins:>8}")
    print(f"  Total Losses:     {total_losses:>8}")

    overall_wr = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0
    print(f"\n  PERFORMANCE")
    print(f"  {'─'*40}")
    print(f"  Overall Win Rate: {overall_wr:>7.1f}%")
    print(f"  Total PnL:        {total_pnl:>+7.2f}%")

    # Best/worst bots
    bots_with_closed = [s for s in all_stats if s.closed_signals > 0]
    if bots_with_closed:
        best_bot = max(bots_with_closed, key=lambda x: x.total_pnl)
        worst_bot = min(bots_with_closed, key=lambda x: x.total_pnl)
        best_wr_bot = max(bots_with_closed, key=lambda x: x.win_rate)

        print(f"\n  BOT RANKINGS")
        print(f"  {'─'*40}")
        print(f"  Best PnL:         {best_bot.bot_name} ({best_bot.total_pnl:+.2f}%)")
        print(f"  Worst PnL:        {worst_bot.bot_name} ({worst_bot.total_pnl:+.2f}%)")
        print(f"  Best Win Rate:    {best_wr_bot.bot_name} ({best_wr_bot.win_rate:.1f}%)")

    # Symbol analysis across all bots
    symbol_pnl: Dict[str, List[float]] = defaultdict(list)
    for s in all_signals:
        if not s.is_open and s.pnl_percent is not None:
            symbol_pnl[s.symbol].append(s.pnl_percent)

    if symbol_pnl:
        symbol_total = {sym: sum(pnls) for sym, pnls in symbol_pnl.items()}
        best_symbol = max(symbol_total.items(), key=lambda x: x[1])
        worst_symbol = min(symbol_total.items(), key=lambda x: x[1])
        most_traded = max(symbol_pnl.items(), key=lambda x: len(x[1]))

        print(f"\n  TOP SYMBOLS (ACROSS ALL BOTS)")
        print(f"  {'─'*40}")
        print(f"  Most Traded:      {most_traded[0]} ({len(most_traded[1])} signals)")
        print(f"  Best Total PnL:   {best_symbol[0]} ({best_symbol[1]:+.2f}%)")
        print(f"  Worst Total PnL:  {worst_symbol[0]} ({worst_symbol[1]:+.2f}%)")


def print_recent_signals(signals: List[SignalRecord], limit: int = 20) -> None:
    """Print recent closed signals."""
    closed = [s for s in signals if not s.is_open and s.closed_at]
    closed.sort(key=lambda x: x.closed_at or "", reverse=True)

    if not closed:
        return

    print(f"\n{'='*60}")
    print(f"  RECENT CLOSED SIGNALS (Last {limit})")
    print(f"{'='*60}\n")

    print(f"  {'Bot':<15} {'Symbol':<12} {'Dir':<6} {'Result':<8} {'PnL':>8}")
    print(f"  {'-'*15} {'-'*12} {'-'*6} {'-'*8} {'-'*8}")

    for s in closed[:limit]:
        pnl_str = f"{s.pnl_percent:+.2f}%" if s.pnl_percent else "N/A"
        result = s.result or "N/A"
        print(f"  {s.bot:<15} {s.symbol:<12} {s.direction:<6} {result:<8} {pnl_str:>8}")


def export_to_csv(signals: List[SignalRecord], output_path: Path) -> None:
    """Export all signals to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'bot', 'signal_id', 'symbol', 'direction', 'entry', 'stop_loss',
            'tp1', 'tp2', 'tp3', 'result', 'exit_price', 'pnl_percent',
            'created_at', 'closed_at', 'is_open', 'exchange', 'timeframe'
        ])
        for s in signals:
            writer.writerow([
                s.bot, s.signal_id, s.symbol, s.direction, s.entry, s.stop_loss,
                s.tp1, s.tp2, s.tp3 or '', s.result or '', s.exit_price or '',
                s.pnl_percent or '', s.created_at or '', s.closed_at or '',
                s.is_open, s.exchange or '', s.timeframe or ''
            ])
    print(f"\nExported {len(signals)} signals to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze signal performance across all bots")
    parser.add_argument("--bot", type=str, help="Analyze specific bot only")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--csv", type=str, help="Export to CSV file")
    parser.add_argument("--recent", type=int, default=20, help="Number of recent signals to show")
    parser.add_argument("--open-only", action="store_true", help="Show only open signals")
    parser.add_argument("--closed-only", action="store_true", help="Show only closed signals")
    args = parser.parse_args()

    all_signals: List[SignalRecord] = []
    all_stats: List[BotStats] = []

    # Determine which bots to analyze
    bots_to_analyze = BOT_STATE_FILES.keys()
    if args.bot:
        matching = [b for b in bots_to_analyze if args.bot.lower() in b.lower()]
        if not matching:
            print(f"Error: No bot matching '{args.bot}' found", file=sys.stderr)
            print(f"Available bots: {', '.join(BOT_STATE_FILES.keys())}", file=sys.stderr)
            sys.exit(1)
        bots_to_analyze = matching

    # Load and process signals
    for bot_name in bots_to_analyze:
        signals = []

        # Load from state file
        state_file = BOT_STATE_FILES.get(bot_name)
        if state_file:
            state = load_json_safe(state_file)
            signals.extend(extract_signals_from_state(bot_name, state))

        # Load from stats file (for historical data)
        stats_file = BOT_STATS_FILES.get(bot_name)
        if stats_file:
            stats = load_json_safe(stats_file)
            stats_signals = extract_signals_from_stats(bot_name, stats)
            # Deduplicate by signal_id
            existing_ids = {s.signal_id for s in signals}
            for s in stats_signals:
                if s.signal_id not in existing_ids:
                    signals.append(s)

        if signals:
            bot_stats = calculate_bot_stats(bot_name, signals)
            all_stats.append(bot_stats)
            all_signals.extend(signals)

    # Filter if needed
    if args.open_only:
        all_signals = [s for s in all_signals if s.is_open]
    elif args.closed_only:
        all_signals = [s for s in all_signals if not s.is_open]

    # Output
    if args.json:
        output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_bots": len(all_stats),
                "total_signals": len(all_signals),
                "total_open": len([s for s in all_signals if s.is_open]),
                "total_closed": len([s for s in all_signals if not s.is_open]),
            },
            "bots": [asdict(s) for s in all_stats],
            "signals": [asdict(s) for s in all_signals],
        }
        print(json.dumps(output, indent=2, default=str))
    elif args.csv:
        export_to_csv(all_signals, Path(args.csv))
    else:
        # Print formatted output
        print(f"\n{'#'*60}")
        print(f"#  SIGNAL PERFORMANCE ANALYSIS")
        print(f"#  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*60}")

        for bot_stats in sorted(all_stats, key=lambda x: x.total_pnl, reverse=True):
            print_bot_stats(bot_stats)

        if len(all_stats) > 1:
            print_overall_summary(all_stats, all_signals)

        print_recent_signals(all_signals, args.recent)

        print(f"\n{'='*60}")
        print(f"  Use --csv signals.csv to export all data")
        print(f"  Use --json for programmatic access")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
