"""
Centralized Message Templates for All Trading Bots
===================================================
This module provides standardized Telegram message formatting for all bots.
Edit the templates here to change alerts across ALL bots at once.

Usage in bots:
    from message_templates import format_signal_message, format_result_message

    message = format_signal_message(
        bot_name="PSAR",
        symbol="POWER/USDT",
        direction="BULLISH",
        entry=0.001234,
        stop_loss=0.001200,
        tp1=0.001300,
        tp2=0.001400,
        ...
    )
    notifier.send_message(message)
"""

import html
from datetime import datetime
from typing import Any, Dict, List, Optional


# =============================================================================
# CONFIGURATION - Edit these to customize all alerts
# =============================================================================

# Direction emojis
EMOJI_BULLISH = "🟢"
EMOJI_BEARISH = "🔴"

# Confidence level emojis and thresholds
CONFIDENCE_LEVELS = {
    "EXTREME": {"min": 80, "emoji": "🔥🔥🔥"},
    "VERY_HIGH": {"min": 70, "emoji": "🔥🔥"},
    "HIGH": {"min": 60, "emoji": "🔥"},
    "MODERATE": {"min": 0, "emoji": "⚡"},
}

# Result emojis
EMOJI_TP_HIT = "✅"
EMOJI_SL_HIT = "❌"
EMOJI_EXPIRED = "⏰"

# Exchange and timeframe defaults
DEFAULT_EXCHANGE = "MEXC"
DEFAULT_TIMEFRAME = "15m"


# =============================================================================
# SIGNAL MESSAGE TEMPLATE
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
    strength: Optional[float] = None,
    indicator_value: Optional[float] = None,
    indicator_name: Optional[str] = None,
    adx: Optional[float] = None,
    rsi: Optional[float] = None,
    funding_rate: Optional[float] = None,
    pattern_name: Optional[str] = None,
    reasons: Optional[List[str]] = None,
    exchange: str = DEFAULT_EXCHANGE,
    timeframe: str = DEFAULT_TIMEFRAME,
    current_price: Optional[float] = None,
    signal_id: Optional[str] = None,
    performance_stats: Optional[Dict[str, Any]] = None,
    extra_info: Optional[str] = None,
) -> str:
    """
    Format a standardized signal message for Telegram.

    Args:
        bot_name: Name of the bot (e.g., "PSAR", "HARMONIC", "FUNDING")
        symbol: Trading pair (e.g., "POWER/USDT")
        direction: "BULLISH" or "BEARISH"
        entry: Entry price
        stop_loss: Stop loss price
        tp1: Take profit 1 price
        tp2: Take profit 2 price (optional)
        tp3: Take profit 3 price (optional)
        confidence: Confidence score 0-100 (optional)
        strength: Trend strength 0-1 (optional)
        indicator_value: Main indicator value (e.g., PSAR, MOST level)
        indicator_name: Name of the indicator (e.g., "PSAR", "MOST")
        adx: ADX value (optional)
        rsi: RSI value (optional)
        funding_rate: Funding rate for funding bot (optional)
        pattern_name: Pattern name for harmonic bot (optional)
        reasons: List of signal reasons (optional)
        exchange: Exchange name
        timeframe: Timeframe string
        current_price: Current market price (optional)
        signal_id: Unique signal ID (optional)
        performance_stats: Dict with tp1, tp2, sl, wins, total, avg_pnl (optional)
        extra_info: Additional info to append (optional)

    Returns:
        Formatted HTML message string
    """
    # Clean symbol
    safe_symbol = html.escape(symbol.replace("/USDT:USDT", "").replace(":USDT", ""))
    if not safe_symbol.endswith("/USDT"):
        safe_symbol = f"{safe_symbol}/USDT"

    # Direction emoji
    emoji = EMOJI_BULLISH if direction == "BULLISH" else EMOJI_BEARISH

    # Confidence emoji
    conf_emoji = ""
    conf_text = ""
    if confidence is not None:
        for level, config in CONFIDENCE_LEVELS.items():
            min_val = config["min"]
            if isinstance(min_val, (int, float)) and confidence >= min_val:
                emoji_val = config["emoji"]
                conf_emoji = str(emoji_val) if emoji_val else ""
                conf_text = level.replace("_", " ")
                break
    elif strength is not None:
        strength_pct = strength * 100
        for level, config in CONFIDENCE_LEVELS.items():
            min_val = config["min"]
            if isinstance(min_val, (int, float)) and strength_pct >= min_val:
                emoji_val = config["emoji"]
                conf_emoji = str(emoji_val) if emoji_val else ""
                break

    # Build header
    if pattern_name:
        header = f"{emoji} <b>{direction} {html.escape(pattern_name)} - {safe_symbol}</b>"
    else:
        header = f"{emoji} <b>{direction} {bot_name} - {safe_symbol}</b>"

    if conf_emoji:
        header += f" {conf_emoji}"

    lines = [header, ""]

    # Signal ID if provided
    if signal_id:
        lines.append(f"🆔 <code>{signal_id}</code>")
        lines.append("")

    # Strategy info
    lines.append(f"📊 <b>Strategy:</b> {bot_name}")

    # Current price
    if current_price is not None:
        lines.append(f"💵 <b>Current:</b> <code>{current_price:.6f}</code>")

    # Indicator value (PSAR, MOST, etc.)
    if indicator_value is not None and indicator_name:
        lines.append(f"📍 <b>{indicator_name}:</b> <code>{indicator_value:.6f}</code>")

    # Funding rate (for funding bot)
    if funding_rate is not None:
        lines.append(f"📈 <b>Funding Rate:</b> {funding_rate*100:.4f}%")

    # Technical indicators
    if adx is not None:
        lines.append(f"💪 <b>ADX:</b> {adx:.1f}")
    if rsi is not None:
        lines.append(f"📉 <b>RSI:</b> {rsi:.1f}")

    # Confidence/Strength
    if confidence is not None:
        lines.append(f"⭐ <b>Confidence:</b> {confidence:.0f}% ({conf_text})")
    elif strength is not None:
        lines.append(f"⚡ <b>Strength:</b> {strength*100:.0f}%")

    lines.append("")

    # Trade levels
    lines.append("<b>🎯 TRADE LEVELS:</b>")
    lines.append(f"💰 Entry: <code>{entry:.6f}</code>")
    lines.append(f"🛑 Stop Loss: <code>{stop_loss:.6f}</code>")
    lines.append(f"🎯 TP1: <code>{tp1:.6f}</code>")
    if tp2 is not None:
        lines.append(f"🚀 TP2: <code>{tp2:.6f}</code>")
    if tp3 is not None:
        lines.append(f"🎆 TP3: <code>{tp3:.6f}</code>")

    # Risk/Reward calculation
    risk = abs(entry - stop_loss)
    if risk > 0:
        rr1 = abs(tp1 - entry) / risk
        lines.append("")
        rr_text = f"⚖️ <b>R:R:</b> 1:{rr1:.2f}"
        if tp2 is not None:
            rr2 = abs(tp2 - entry) / risk
            rr_text += f" | 1:{rr2:.2f}"
        lines.append(rr_text)

    # Reasons (for funding, liquidation bots)
    if reasons and len(reasons) > 0:
        lines.append("")
        lines.append(f"📝 <b>Reasons:</b> {', '.join(reasons)}")

    # Performance history
    if performance_stats and performance_stats.get("total", 0) > 0:
        stats = performance_stats
        win_rate = (stats.get("wins", 0) / stats["total"]) * 100
        lines.append("")
        lines.append(f"📈 <b>History:</b> TP1:{stats.get('tp1', 0)} | TP2:{stats.get('tp2', 0)} | SL:{stats.get('sl', 0)}")
        lines.append(f"🏆 <b>Win Rate:</b> {win_rate:.1f}% ({stats.get('wins', 0)}/{stats['total']})")
        if stats.get("avg_pnl") is not None:
            lines.append(f"💵 <b>Avg PnL:</b> {stats['avg_pnl']:+.2f}%")

    # Footer with exchange and timeframe
    lines.append("")
    lines.append(f"🏦 {exchange} | ⏰ {timeframe}")

    # Extra info
    if extra_info:
        lines.append("")
        lines.append(f"💡 {extra_info}")

    return "\n".join(lines)


# =============================================================================
# RESULT MESSAGE TEMPLATE (TP/SL Hit)
# =============================================================================

def format_result_message(
    symbol: str,
    direction: str,
    result: str,
    entry: float,
    exit_price: float,
    stop_loss: float,
    tp1: float,
    tp2: Optional[float] = None,
    signal_id: Optional[str] = None,
    performance_stats: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Format a result message when TP or SL is hit.

    Args:
        symbol: Trading pair
        direction: "BULLISH" or "BEARISH"
        result: "TP1", "TP2", "TP3", "SL", or "EXPIRED"
        entry: Entry price
        exit_price: Exit price
        stop_loss: Stop loss price
        tp1: Take profit 1 price
        tp2: Take profit 2 price (optional)
        signal_id: Signal ID (optional)
        performance_stats: Performance stats dict (optional)

    Returns:
        Formatted HTML message string
    """
    safe_symbol = html.escape(symbol.replace("/USDT:USDT", "").replace(":USDT", ""))

    # Calculate P&L
    if direction == "BULLISH":
        pnl_pct = ((exit_price - entry) / entry) * 100
    else:
        pnl_pct = ((entry - exit_price) / entry) * 100

    # Result emoji
    if result.startswith("TP"):
        emoji = EMOJI_TP_HIT
        result_text = f"{result} HIT! 🎉"
    elif result == "SL":
        emoji = EMOJI_SL_HIT
        result_text = "STOP LOSS HIT"
    else:
        emoji = EMOJI_EXPIRED
        result_text = "EXPIRED"

    lines = [
        f"{emoji} <b>{direction} {safe_symbol} - {result_text}</b>",
        "",
    ]

    if signal_id:
        lines.append(f"🆔 <code>{signal_id}</code>")
        lines.append("")

    lines.extend([
        f"💰 Entry: <code>{entry:.6f}</code>",
        f"🎯 Exit: <code>{exit_price:.6f}</code>",
        f"📊 <b>PnL: {pnl_pct:+.2f}%</b>",
        "",
        f"🛑 SL was: <code>{stop_loss:.6f}</code>",
        f"🎯 TP1 was: <code>{tp1:.6f}</code>",
    ])

    if tp2 is not None:
        lines.append(f"🚀 TP2 was: <code>{tp2:.6f}</code>")

    # Performance summary
    if performance_stats and performance_stats.get("total", 0) > 0:
        stats = performance_stats
        win_rate = (stats.get("wins", 0) / stats["total"]) * 100
        lines.append("")
        lines.append(f"📈 <b>Updated Stats:</b>")
        lines.append(f"   Win Rate: {win_rate:.1f}% ({stats.get('wins', 0)}/{stats['total']})")
        if stats.get("total_pnl") is not None:
            lines.append(f"   Total PnL: {stats['total_pnl']:+.2f}%")

    return "\n".join(lines)


# =============================================================================
# STARTUP MESSAGE TEMPLATE
# =============================================================================

def format_startup_message(
    bot_name: str,
    symbols: List[str],
    exchange: str = DEFAULT_EXCHANGE,
    timeframe: str = DEFAULT_TIMEFRAME,
    strategy_description: Optional[str] = None,
) -> str:
    """
    Format a startup message.

    Args:
        bot_name: Name of the bot
        symbols: List of symbols being monitored
        exchange: Exchange name
        timeframe: Timeframe
        strategy_description: Optional strategy description

    Returns:
        Formatted HTML message string
    """
    lines = [
        f"🤖 <b>{bot_name} Bot Started</b>",
        "",
        f"📊 <b>Monitoring {len(symbols)} pairs on {exchange} ({timeframe})</b>",
        "",
    ]

    # List symbols
    if len(symbols) <= 10:
        for sym in symbols:
            lines.append(f"  ✅ {sym}")
    else:
        for sym in symbols[:7]:
            lines.append(f"  ✅ {sym}")
        lines.append(f"  ... and {len(symbols) - 7} more")

    if strategy_description:
        lines.append("")
        lines.append(f"⚙️ <b>Strategy:</b> {strategy_description}")

    lines.extend([
        "",
        "✨ Ready to send trading signals!"
    ])

    return "\n".join(lines)


# =============================================================================
# ERROR/SHUTDOWN MESSAGE TEMPLATES
# =============================================================================

def format_error_message(bot_name: str, error: str) -> str:
    """Format an error message."""
    return f"⚠️ <b>{bot_name} Bot Error</b>\n\n{html.escape(error)}"


def format_shutdown_message(bot_name: str, reason: str = "Manual shutdown") -> str:
    """Format a shutdown message."""
    return f"🛑 <b>{bot_name} Bot Stopped</b>\n\nReason: {reason}"


# =============================================================================
# HEALTH CHECK MESSAGE
# =============================================================================

def format_health_message(
    bot_name: str,
    uptime_hours: float,
    signals_sent: int,
    errors: int,
    last_signal_ago: Optional[str] = None,
) -> str:
    """Format a health check message."""
    lines = [
        f"💓 <b>{bot_name} Health Check</b>",
        "",
        f"⏱️ Uptime: {uptime_hours:.1f} hours",
        f"📨 Signals sent: {signals_sent}",
        f"❌ Errors: {errors}",
    ]

    if last_signal_ago:
        lines.append(f"🕐 Last signal: {last_signal_ago}")

    lines.append("")
    lines.append("✅ Bot is running normally")

    return "\n".join(lines)
