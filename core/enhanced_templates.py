#!/usr/bin/env python3
"""
Enhanced Message Templates - Rich Context Signal Notifications

Provides beautiful, informative Telegram messages with:
- Historical performance for symbol/pattern
- ATR-based volatility indicator
- Similar setup comparisons
- Win/loss streaks
- Duration and excursion data
- Animated progress indicators

Usage:
    from core.enhanced_templates import SignalMessageBuilder, ResultMessageBuilder
    
    # New signal
    message = SignalMessageBuilder(signal, stats).build()
    
    # Result
    message = ResultMessageBuilder(result, stats).build()
"""

import html
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Try to import Signal classes, fall back gracefully
try:
    from core.signal_manager import Signal, SignalResult, SignalState
except ImportError:
    Signal = None
    SignalResult = None
    SignalState = None


# =============================================================================
# EMOJI CONFIGURATION
# =============================================================================

EMOJIS = {
    # Direction
    "long": "🟢",
    "short": "🔴",
    
    # Confidence levels
    "extreme": "🔥🔥🔥",
    "very_high": "🔥🔥",
    "high": "🔥",
    "moderate": "⚡",
    "low": "💨",
    
    # Results
    "tp1": "🎯",
    "tp2": "🚀",
    "tp3": "💎",
    "sl": "❌",
    "breakeven": "🟰",
    "trailing": "📐",
    "expired": "⏰",
    
    # Stats
    "win_streak": "🔥",
    "loss_streak": "❄️",
    "stats": "📊",
    "history": "📈",
    "pnl_up": "📈",
    "pnl_down": "📉",
    
    # Meta
    "entry": "💰",
    "target": "🎯",
    "stop": "🛑",
    "rr": "⚖️",
    "time": "⏱️",
    "exchange": "🏦",
    "pattern": "🔷",
    "atr": "📏",
    "volatility_high": "🌊",
    "volatility_low": "🏖️",
    
    # Performance
    "mfe": "📈",
    "mae": "📉",
    "duration": "⏳",
}

# Confidence thresholds
CONFIDENCE_THRESHOLDS = [
    (80, "extreme"),
    (70, "very_high"),
    (60, "high"),
    (40, "moderate"),
    (0, "low"),
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_confidence_emoji(confidence: float) -> str:
    """Get emoji based on confidence level."""
    for threshold, level in CONFIDENCE_THRESHOLDS:
        if confidence >= threshold:
            return EMOJIS.get(level, "")
    return ""


def get_streak_display(streak: int) -> str:
    """Get streak display string."""
    if streak > 0:
        return f"{EMOJIS['win_streak']} {streak}W streak"
    elif streak < 0:
        return f"{EMOJIS['loss_streak']} {abs(streak)}L streak"
    return ""


def format_price(price: float, precision: int = 6) -> str:
    """Format price with appropriate precision."""
    if price == 0:
        return "0"
    
    # Auto-adjust precision based on price magnitude
    if price >= 1000:
        precision = 2
    elif price >= 100:
        precision = 3
    elif price >= 1:
        precision = 4
    elif price >= 0.01:
        precision = 5
    
    return f"{price:.{precision}f}"


def format_pnl(pnl: float) -> str:
    """Format P&L with color indicator."""
    if pnl > 0:
        return f"+{pnl:.2f}%"
    return f"{pnl:.2f}%"


def format_duration(hours: float) -> str:
    """Format duration nicely."""
    if hours < 1:
        minutes = int(hours * 60)
        return f"{minutes}m"
    elif hours < 24:
        return f"{hours:.1f}h"
    else:
        days = hours / 24
        return f"{days:.1f}d"


def calculate_rr(entry: float, target: float, stop: float) -> float:
    """Calculate risk:reward ratio."""
    risk = abs(entry - stop)
    reward = abs(target - entry)
    return reward / risk if risk > 0 else 0


# =============================================================================
# NEW SIGNAL MESSAGE BUILDER
# =============================================================================

@dataclass
class SignalMessageBuilder:
    """
    Builds rich signal messages with historical context.
    
    Example output:
    
    🟢 LONG STRAT | BTC 🔥🔥
    
    💰 Entry: 42,000.00
    🛑 SL: 41,500.00 (-1.2%)
    🎯 TP1: 42,500.00 (+1.2%) | 1:1.0
    🚀 TP2: 43,000.00 (+2.4%) | 1:2.0
    
    📊 History: 68% Win (17/25)
    🔥 3W streak | Avg PnL: +1.2%
    
    📏 ATR: 450 (1.1%) 🌊 High volatility
    ⏱️ Est. time to TP1: ~2-4h
    
    🏦 MEXC 5m | STRAT 2-1-2
    """
    
    # Required data
    bot_name: str
    symbol: str
    direction: str
    entry: float
    stop_loss: float
    take_profit_1: float
    
    # Optional price levels
    take_profit_2: Optional[float] = None
    take_profit_3: Optional[float] = None
    
    # Performance stats
    stats: Optional[Dict[str, Any]] = None
    symbol_stats: Optional[Dict[str, int]] = None
    last_result: Optional[Dict[str, Any]] = None  # Last signal result for this symbol
    
    # Signal metadata
    confidence: float = 0.0
    pattern_name: Optional[str] = None
    signal_id: Optional[str] = None
    exchange: str = "MEXC"
    timeframe: str = "5m"
    
    # Technical data
    atr: Optional[float] = None
    atr_percent: Optional[float] = None
    rsi: Optional[float] = None
    adx: Optional[float] = None
    
    # Extra info
    reasons: Optional[List[str]] = None
    extra_info: Optional[str] = None
    
    def build(self) -> str:
        """Build the complete message."""
        lines = []
        
        # Header
        lines.append(self._build_header())
        lines.append("")
        
        # Price levels
        lines.extend(self._build_price_levels())
        lines.append("")
        
        # Performance history
        lines.extend(self._build_history())
        
        # Volatility info
        vol_line = self._build_volatility()
        if vol_line:
            lines.append("")
            lines.append(vol_line)
        
        # Time estimate
        time_est = self._build_time_estimate()
        if time_est:
            lines.append(time_est)
        
        # Footer
        lines.append("")
        lines.append(self._build_footer())
        
        # Reasons
        if self.reasons and len(self.reasons) > 0:
            reasons_short = self.reasons[:3]
            lines.append(f"💡 {', '.join(reasons_short)}")
        
        return "\n".join(lines)
    
    def _build_header(self) -> str:
        """Build header line."""
        # Clean symbol
        symbol_clean = self.symbol.replace("/USDT:USDT", "").replace(":USDT", "").replace("/USDT", "")
        symbol_clean = html.escape(symbol_clean)
        
        # Direction
        is_long = self.direction.upper() in ("LONG", "BULLISH", "BUY")
        dir_emoji = EMOJIS["long"] if is_long else EMOJIS["short"]
        dir_text = "LONG" if is_long else "SHORT"
        
        # Confidence emoji
        conf_emoji = get_confidence_emoji(self.confidence) if self.confidence else ""
        
        # Pattern or bot name
        if self.pattern_name:
            name = html.escape(self.pattern_name)
        else:
            name = self.bot_name.upper()
        
        header = f"{dir_emoji} <b>{dir_text} {name}</b> | {symbol_clean}"
        if conf_emoji:
            header += f" {conf_emoji}"
        
        return header
    
    def _build_price_levels(self) -> List[str]:
        """Build price level lines."""
        lines = []
        
        # Entry
        lines.append(f"{EMOJIS['entry']} Entry: <code>{format_price(self.entry)}</code>")
        
        # Stop Loss with percentage
        sl_pct = ((self.stop_loss - self.entry) / self.entry) * 100
        lines.append(f"{EMOJIS['stop']} SL: <code>{format_price(self.stop_loss)}</code> ({sl_pct:+.1f}%)")
        
        # TP1 with RR
        tp1_pct = ((self.take_profit_1 - self.entry) / self.entry) * 100
        rr1 = calculate_rr(self.entry, self.take_profit_1, self.stop_loss)
        tp_line = f"{EMOJIS['tp1']} TP1: <code>{format_price(self.take_profit_1)}</code> ({tp1_pct:+.1f}%) | 1:{rr1:.1f}"
        
        # TP2 on same line if exists
        if self.take_profit_2:
            tp2_pct = ((self.take_profit_2 - self.entry) / self.entry) * 100
            rr2 = calculate_rr(self.entry, self.take_profit_2, self.stop_loss)
            lines.append(tp_line)
            lines.append(f"{EMOJIS['tp2']} TP2: <code>{format_price(self.take_profit_2)}</code> ({tp2_pct:+.1f}%) | 1:{rr2:.1f}")
        else:
            lines.append(tp_line)
        
        # TP3 if exists
        if self.take_profit_3:
            tp3_pct = ((self.take_profit_3 - self.entry) / self.entry) * 100
            rr3 = calculate_rr(self.entry, self.take_profit_3, self.stop_loss)
            lines.append(f"{EMOJIS['tp3']} TP3: <code>{format_price(self.take_profit_3)}</code> ({tp3_pct:+.1f}%) | 1:{rr3:.1f}")
        
        return lines
    
    def _build_history(self) -> List[str]:
        """Build performance history section."""
        lines = []
        
        stats = self.stats or {}
        total = stats.get("total", 0)
        wins = stats.get("wins", 0)
        win_rate = stats.get("win_rate", 0)
        streak = stats.get("current_streak", 0)
        avg_pnl = stats.get("avg_pnl", 0)
        
        # Build history line with last result
        if total > 0:
            history_line = f"{EMOJIS['stats']} History: {win_rate:.0f}% Win ({wins}/{total})"
            
            # Add last result inline if available
            if self.last_result:
                last_res = self.last_result.get("result", "")
                last_pnl = self.last_result.get("pnl_pct", 0)
                is_win = last_res.startswith("TP") or last_pnl > 0
                result_emoji = "✅" if is_win else "❌"
                history_line += f" | Last: {last_res} {format_pnl(last_pnl)} {result_emoji}"
            
            lines.append(history_line)
            
            # Streak and avg PnL on same line
            extras = []
            streak_str = get_streak_display(streak)
            if streak_str:
                extras.append(streak_str)
            if avg_pnl != 0:
                extras.append(f"Avg: {format_pnl(avg_pnl)}")
            
            if extras:
                lines.append(" | ".join(extras))
        else:
            lines.append(f"{EMOJIS['stats']} History: First trade on this setup!")
        
        # Symbol-specific stats if available
        if self.symbol_stats:
            tp1 = self.symbol_stats.get("TP1", 0)
            tp2 = self.symbol_stats.get("TP2", 0)
            tp3 = self.symbol_stats.get("TP3", 0)
            sl = self.symbol_stats.get("SL", 0)
            sym_total = tp1 + tp2 + tp3 + sl
            if sym_total > 0:
                symbol_clean = self.symbol.replace("/USDT:USDT", "").replace("/USDT", "")
                sym_wr = ((tp1 + tp2 + tp3) / sym_total) * 100
                lines.append(f"{EMOJIS['history']} {symbol_clean}: {sym_wr:.0f}% (TP:{tp1+tp2+tp3} SL:{sl})")
        
        return lines
    
    def _build_volatility(self) -> Optional[str]:
        """Build volatility indicator."""
        if self.atr is None and self.atr_percent is None:
            return None
        
        atr_pct = self.atr_percent or (self.atr / self.entry * 100 if self.atr and self.entry else 0)
        
        # Volatility classification
        if atr_pct >= 2.0:
            vol_emoji = EMOJIS["volatility_high"]
            vol_text = "High volatility"
        elif atr_pct >= 1.0:
            vol_emoji = ""
            vol_text = "Normal"
        else:
            vol_emoji = EMOJIS["volatility_low"]
            vol_text = "Low volatility"
        
        atr_display = f"{self.atr:.2f}" if self.atr else f"{atr_pct:.2f}%"
        return f"{EMOJIS['atr']} ATR: {atr_display} ({atr_pct:.1f}%) {vol_emoji} {vol_text}"
    
    def _build_time_estimate(self) -> Optional[str]:
        """Build estimated time to TP1."""
        if self.atr is None or self.entry == 0:
            return None
        
        # Distance to TP1 in price
        distance = abs(self.take_profit_1 - self.entry)
        
        # Estimate bars needed (ATR is typically per bar)
        bars_estimate = distance / self.atr if self.atr > 0 else 0
        
        if bars_estimate <= 0:
            return None
        
        # Convert to time based on timeframe
        tf_minutes = {
            "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "4h": 240, "1d": 1440,
        }
        minutes_per_bar = tf_minutes.get(self.timeframe, 5)
        
        min_hours = (bars_estimate * minutes_per_bar) / 60
        max_hours = min_hours * 2  # Upper estimate
        
        return f"{EMOJIS['time']} Est. time to TP1: ~{format_duration(min_hours)}-{format_duration(max_hours)}"
    
    def _build_footer(self) -> str:
        """Build footer line."""
        parts = [f"{EMOJIS['exchange']} {self.exchange} {self.timeframe}"]
        
        if self.pattern_name:
            parts.append(f"{EMOJIS['pattern']} {self.pattern_name}")
        
        if self.signal_id:
            parts.append(f"ID: <code>{self.signal_id}</code>")
        
        return " | ".join(parts)


# =============================================================================
# RESULT MESSAGE BUILDER
# =============================================================================

@dataclass
class ResultMessageBuilder:
    """
    Builds rich result messages with performance context.
    
    Example output:
    
    🎯✅ TP1 HIT! | BTC LONG
    
    💰 Entry: 42,000.00
    🏁 Exit: 42,500.00
    📈 P&L: +1.19%
    
    ⏳ Duration: 2.5h
    📈 MFE: +1.5% (touched +1.5% before exit)
    📉 MAE: -0.3% (worst drawdown)
    
    📊 Updated Stats: 68% Win (18/26)
    🔥 4W streak! | Total PnL: +15.2%
    
    ⏱️ 2024-01-15 14:30 UTC
    """
    
    # Core result data
    symbol: str
    direction: str
    result: str  # TP1, TP2, SL, etc.
    entry: float
    exit_price: float
    pnl_pct: float
    
    # Performance context
    duration_str: Optional[str] = None
    mfe: float = 0.0  # Max Favorable Excursion
    mae: float = 0.0  # Max Adverse Excursion
    
    # Stats
    stats: Optional[Dict[str, Any]] = None
    
    # Optional data
    stop_loss: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    signal_id: Optional[str] = None
    bot_name: str = ""
    
    def build(self) -> str:
        """Build the complete result message."""
        lines = []
        
        # Header
        lines.append(self._build_header())
        lines.append("")
        
        # Price info
        lines.extend(self._build_price_info())
        lines.append("")
        
        # Duration and excursions
        lines.extend(self._build_trade_analysis())
        lines.append("")
        
        # Updated stats
        lines.extend(self._build_stats())
        
        # Timestamp
        lines.append("")
        lines.append(f"{EMOJIS['time']} {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        
        return "\n".join(lines)
    
    def _build_header(self) -> str:
        """Build result header."""
        symbol_clean = self.symbol.replace("/USDT:USDT", "").replace(":USDT", "").replace("/USDT", "")
        symbol_clean = html.escape(symbol_clean)
        
        is_long = self.direction.upper() in ("LONG", "BULLISH", "BUY")
        dir_text = "LONG" if is_long else "SHORT"
        
        # Result emoji and text
        result_upper = self.result.upper()
        if result_upper == "TP1":
            emoji = f"{EMOJIS['tp1']}✅"
            result_text = "TP1 HIT!"
        elif result_upper == "TP2":
            emoji = f"{EMOJIS['tp2']}✅"
            result_text = "TP2 HIT! 🎉"
        elif result_upper == "TP3":
            emoji = f"{EMOJIS['tp3']}✅"
            result_text = "TP3 HIT! 💎🎉"
        elif "BREAKEVEN" in result_upper:
            emoji = f"{EMOJIS['breakeven']}"
            result_text = "Breakeven Exit"
        elif "TRAILING" in result_upper:
            emoji = f"{EMOJIS['trailing']}✅"
            result_text = "Trailing Stop Hit"
        elif result_upper == "SL":
            emoji = EMOJIS["sl"]
            result_text = "STOP LOSS HIT"
        elif result_upper == "EXPIRED":
            emoji = EMOJIS["expired"]
            result_text = "EXPIRED"
        else:
            emoji = "📌"
            result_text = self.result.upper()
        
        return f"{emoji} <b>{result_text}</b> | {symbol_clean} {dir_text}"
    
    def _build_price_info(self) -> List[str]:
        """Build price information section."""
        lines = []
        
        lines.append(f"{EMOJIS['entry']} Entry: <code>{format_price(self.entry)}</code>")
        lines.append(f"🏁 Exit: <code>{format_price(self.exit_price)}</code>")
        
        # P&L with emoji
        pnl_emoji = EMOJIS["pnl_up"] if self.pnl_pct >= 0 else EMOJIS["pnl_down"]
        lines.append(f"{pnl_emoji} <b>P&L: {format_pnl(self.pnl_pct)}</b>")
        
        return lines
    
    def _build_trade_analysis(self) -> List[str]:
        """Build trade analysis section."""
        lines = []
        
        # Duration
        if self.duration_str:
            lines.append(f"{EMOJIS['duration']} Duration: {self.duration_str}")
        
        # Max Favorable Excursion
        if self.mfe != 0:
            lines.append(f"{EMOJIS['mfe']} MFE: {format_pnl(self.mfe)} (best unrealized)")
        
        # Max Adverse Excursion
        if self.mae != 0:
            lines.append(f"{EMOJIS['mae']} MAE: {format_pnl(self.mae)} (worst drawdown)")
        
        # If no excursion data, show original levels
        if not lines:
            if self.stop_loss > 0:
                lines.append(f"{EMOJIS['stop']} SL was: <code>{format_price(self.stop_loss)}</code>")
            if self.take_profit_1 > 0:
                lines.append(f"{EMOJIS['tp1']} TP1 was: <code>{format_price(self.take_profit_1)}</code>")
        
        return lines
    
    def _build_stats(self) -> List[str]:
        """Build updated stats section."""
        lines = []
        
        stats = self.stats or {}
        total = stats.get("total", 0)
        wins = stats.get("wins", 0)
        win_rate = stats.get("win_rate", 0)
        total_pnl = stats.get("total_pnl", 0)
        streak = stats.get("current_streak", 0)
        
        if total > 0:
            lines.append(f"{EMOJIS['stats']} <b>Updated Stats:</b> {win_rate:.0f}% Win ({wins}/{total})")
            
            # Streak and total PnL
            extras = []
            streak_str = get_streak_display(streak)
            if streak_str:
                # Add celebration for long streaks
                if streak >= 5:
                    streak_str += " 🎉"
                extras.append(streak_str)
            
            if total_pnl != 0:
                extras.append(f"Total PnL: {format_pnl(total_pnl)}")
            
            if extras:
                lines.append(" | ".join(extras))
        
        return lines


# =============================================================================
# PARTIAL EXIT MESSAGE BUILDER
# =============================================================================

@dataclass
class PartialExitMessageBuilder:
    """
    Builds messages for partial position exits.
    
    Example:
    
    🎯 TP1 HIT (50% closed) | BTC LONG
    
    📈 Partial P&L: +1.2%
    📊 50% position remaining
    🛑 SL moved to breakeven
    """
    
    symbol: str
    direction: str
    result: str
    exit_price: float
    pnl_pct: float
    position_remaining: float
    moved_to_breakeven: bool = False
    new_stop_loss: Optional[float] = None
    
    def build(self) -> str:
        """Build partial exit message."""
        lines = []
        
        symbol_clean = self.symbol.replace("/USDT:USDT", "").replace("/USDT", "")
        is_long = self.direction.upper() in ("LONG", "BULLISH", "BUY")
        dir_text = "LONG" if is_long else "SHORT"
        
        closed_pct = int((1 - self.position_remaining) * 100)
        
        # Header
        emoji = EMOJIS["tp1"] if "TP1" in self.result else EMOJIS["tp2"]
        lines.append(f"{emoji} <b>{self.result} HIT ({closed_pct}% closed)</b> | {symbol_clean} {dir_text}")
        lines.append("")
        
        # P&L
        pnl_emoji = EMOJIS["pnl_up"] if self.pnl_pct >= 0 else EMOJIS["pnl_down"]
        lines.append(f"{pnl_emoji} Partial P&L: {format_pnl(self.pnl_pct)}")
        
        # Remaining position
        remaining_pct = int(self.position_remaining * 100)
        lines.append(f"{EMOJIS['stats']} {remaining_pct}% position remaining")
        
        # Breakeven info
        if self.moved_to_breakeven:
            lines.append(f"{EMOJIS['breakeven']} SL moved to breakeven")
        elif self.new_stop_loss:
            lines.append(f"{EMOJIS['stop']} New SL: <code>{format_price(self.new_stop_loss)}</code>")
        
        return "\n".join(lines)


# =============================================================================
# STREAK ALERT MESSAGE
# =============================================================================

def build_streak_alert(bot_name: str, streak: int, total_pnl: float) -> str:
    """Build a streak milestone alert."""
    if streak >= 10:
        emoji = "🏆🔥"
        title = "INCREDIBLE STREAK!"
    elif streak >= 7:
        emoji = "🔥🔥🔥"
        title = "AMAZING STREAK!"
    elif streak >= 5:
        emoji = "🔥🔥"
        title = "HOT STREAK!"
    else:
        emoji = "🔥"
        title = "Win Streak!"
    
    lines = [
        f"{emoji} <b>{bot_name} - {title}</b>",
        "",
        f"🎯 {streak} consecutive wins!",
        f"{EMOJIS['pnl_up']} Streak P&L: {format_pnl(total_pnl)}",
        "",
        "Keep it going! 💪",
    ]
    
    return "\n".join(lines)


# =============================================================================
# DAILY SUMMARY MESSAGE
# =============================================================================

def build_daily_summary(
    date: str,
    total_trades: int,
    wins: int,
    losses: int,
    total_pnl: float,
    best_trade: float,
    worst_trade: float,
    by_bot: Dict[str, Dict[str, Any]],
) -> str:
    """Build daily summary message."""
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    pnl_emoji = EMOJIS["pnl_up"] if total_pnl >= 0 else EMOJIS["pnl_down"]
    
    lines = [
        f"📊 <b>Daily Summary - {date}</b>",
        "",
        f"<b>Overall:</b>",
        f"  Trades: {total_trades}",
        f"  Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)",
        f"  {pnl_emoji} P&L: {format_pnl(total_pnl)}",
        f"  Best: {format_pnl(best_trade)}",
        f"  Worst: {format_pnl(worst_trade)}",
    ]
    
    if by_bot:
        lines.append("")
        lines.append("<b>By Bot:</b>")
        for bot, stats in sorted(by_bot.items(), key=lambda x: x[1].get("pnl", 0), reverse=True):
            pnl = stats.get("pnl", 0)
            trades = stats.get("trades", 0)
            emoji = "🟢" if pnl >= 0 else "🔴"
            lines.append(f"  {emoji} {bot}: {format_pnl(pnl)} ({trades} trades)")
    
    return "\n".join(lines)


# =============================================================================
# EMERGENCY ALERT MESSAGE
# =============================================================================

def build_emergency_alert(reason: str, positions_count: int) -> str:
    """Build emergency stop alert."""
    lines = [
        "🚨🚨🚨 <b>EMERGENCY STOP ACTIVATED</b> 🚨🚨🚨",
        "",
        f"<b>Reason:</b> {html.escape(reason)}",
        "",
        f"📊 Open positions: {positions_count}",
        "",
        "⛔ All new signals BLOCKED",
        "⚠️ Existing positions continue to monitor",
        "",
        "Use /reset_emergency to resume trading",
    ]
    
    return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS (Backward Compatible)
# =============================================================================

def format_signal_message(
    bot_name: str,
    symbol: str,
    direction: str,
    entry: float,
    stop_loss: float,
    tp1: float,
    tp2: Optional[float] = None,
    tp3: Optional[float] = None,
    confidence: Optional[float] = None,
    pattern_name: Optional[str] = None,
    stats: Optional[Dict[str, Any]] = None,
    symbol_stats: Optional[Dict[str, int]] = None,
    last_result: Optional[Dict[str, Any]] = None,
    exchange: str = "MEXC",
    timeframe: str = "5m",
    atr: Optional[float] = None,
    signal_id: Optional[str] = None,
    reasons: Optional[List[str]] = None,
    **kwargs
) -> str:
    """
    Convenience function for backward compatibility.
    Wraps SignalMessageBuilder.
    """
    builder = SignalMessageBuilder(
        bot_name=bot_name,
        symbol=symbol,
        direction=direction,
        entry=entry,
        stop_loss=stop_loss,
        take_profit_1=tp1,
        take_profit_2=tp2,
        take_profit_3=tp3,
        confidence=confidence or 0,
        pattern_name=pattern_name,
        stats=stats,
        symbol_stats=symbol_stats,
        last_result=last_result,
        exchange=exchange,
        timeframe=timeframe,
        atr=atr,
        signal_id=signal_id,
        reasons=reasons,
    )
    return builder.build()


def format_result_message(
    symbol: str,
    direction: str,
    result: str,
    entry: float,
    exit_price: float,
    pnl_pct: Optional[float] = None,
    stop_loss: float = 0,
    tp1: float = 0,
    tp2: Optional[float] = None,
    stats: Optional[Dict[str, Any]] = None,
    duration_str: Optional[str] = None,
    mfe: float = 0,
    mae: float = 0,
    **kwargs
) -> str:
    """
    Convenience function for backward compatibility.
    Wraps ResultMessageBuilder.
    """
    # Calculate P&L if not provided
    if pnl_pct is None:
        is_long = direction.upper() in ("LONG", "BULLISH", "BUY")
        if is_long:
            pnl_pct = ((exit_price - entry) / entry) * 100 if entry else 0
        else:
            pnl_pct = ((entry - exit_price) / entry) * 100 if entry else 0
    
    builder = ResultMessageBuilder(
        symbol=symbol,
        direction=direction,
        result=result,
        entry=entry,
        exit_price=exit_price,
        pnl_pct=pnl_pct,
        stop_loss=stop_loss,
        take_profit_1=tp1,
        take_profit_2=tp2 or 0,
        stats=stats,
        duration_str=duration_str,
        mfe=mfe,
        mae=mae,
    )
    return builder.build()

