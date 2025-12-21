#!/usr/bin/env python3
"""Shared signal statistics tracker for VM bots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from file_lock import file_lock


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SignalRecord:
    signal_id: str
    symbol: str
    direction: str
    entry: float
    exit: float
    result: str
    pnl_pct: float
    created_at: str
    closed_at: str
    extra: Dict[str, object]


class SignalStats:
    def __init__(self, bot_name: str, file_path: Path):
        self.bot_name = bot_name
        self.file_path = file_path
        self.data = self._load()

    def symbol_tp_sl_counts(self, symbol: str) -> dict:
        """Return TP1/TP2/SL counts for a given symbol from history.

        Symbol must match the stored `symbol` field exactly (e.g. "ON/USDT").
        """
        counts = {"TP1": 0, "TP2": 0, "SL": 0}
        history_data = self.data.get("history", [])
        history: List[Dict[str, object]] = history_data if isinstance(history_data, list) else []
        for trade in history:
            if trade.get("symbol") != symbol:
                continue
            res = str(trade.get("result", ""))
            if res in counts:
                counts[res] += 1
        return counts

    def _load(self) -> Dict[str, object]:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if self.file_path.exists():
            try:
                with file_lock(self.file_path):
                    return json.loads(self.file_path.read_text())
            except json.JSONDecodeError:
                return {"open": {}, "history": []}
        return {"open": {}, "history": []}

    def _save(self) -> None:
        with file_lock(self.file_path):
            self.file_path.write_text(json.dumps(self.data, indent=2))

    def record_open(
        self,
        signal_id: str,
        symbol: str,
        direction: str,
        entry: float,
        created_at: str,
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        open_positions_data = self.data.setdefault("open", {})
        open_positions: Dict[str, object] = open_positions_data if isinstance(open_positions_data, dict) else {}
        open_positions[signal_id] = {
            "symbol": symbol,
            "direction": direction,
            "entry": entry,
            "created_at": created_at,
            "extra": extra or {},
        }
        self._save()

    def record_close(
        self,
        signal_id: str,
        exit_price: float,
        result: str,
        closed_at: Optional[str] = None,
    ) -> Optional[SignalRecord]:
        open_positions_data = self.data.setdefault("open", {})
        open_positions: Dict[str, object] = open_positions_data if isinstance(open_positions_data, dict) else {}
        record_data = open_positions.pop(signal_id, None)
        if record_data is None:
            return None
        
        # Cast to dict for type safety
        record: Dict[str, object] = record_data if isinstance(record_data, dict) else {}
        if not record:
            return None

        entry_value = record.get("entry", 0)
        entry_price = float(entry_value) if isinstance(entry_value, (int, float, str)) else 0.0
        direction = str(record.get("direction", ""))
        pnl_pct = self._calculate_pnl(direction, entry_price, exit_price)
        closed_at = closed_at or _utc_now()

        history_data = self.data.setdefault("history", [])
        history: List[Dict[str, object]] = history_data if isinstance(history_data, list) else []
        
        extra_data = record.get("extra", {})
        extra: Dict[str, object] = extra_data if isinstance(extra_data, dict) else {}
        
        history_entry = {
            "id": signal_id,
            "symbol": record.get("symbol", ""),
            "direction": direction,
            "entry": entry_price,
            "exit": exit_price,
            "result": result,
            "pnl_pct": pnl_pct,
            "created_at": record.get("created_at", ""),
            "closed_at": closed_at,
            "extra": extra,
        }
        history.append(history_entry)
        self._save()

        return SignalRecord(
            signal_id=signal_id,
            symbol=str(history_entry.get("symbol", "")),
            direction=direction,
            entry=entry_price,
            exit=exit_price,
            result=result,
            pnl_pct=pnl_pct,
            created_at=str(history_entry.get("created_at", "")),
            closed_at=closed_at,
            extra=extra,
        )

    def discard(self, signal_id: str) -> None:
        open_positions_data = self.data.setdefault("open", {})
        open_positions: Dict[str, object] = open_positions_data if isinstance(open_positions_data, dict) else {}
        if signal_id in open_positions:
            open_positions.pop(signal_id)
            self._save()

    @staticmethod
    def _calculate_pnl(direction: str, entry: float, exit_price: float) -> float:
        if entry == 0:
            return 0.0
        # Normalize direction to handle BULLISH/BEARISH as well as LONG/SHORT
        direction = direction.upper()
        if direction in ("LONG", "BULLISH", "BUY"):
            return ((exit_price - entry) / entry) * 100
        return ((entry - exit_price) / entry) * 100

    def get_summary(self) -> Dict[str, float | int]:
        history_data = self.data.get("history", [])
        history: List[Dict[str, object]] = history_data if isinstance(history_data, list) else []
        total = len(history)
        if total == 0:
            return {"win_rate": 0.0, "tp_hits": 0, "sl_hits": 0, "total_pnl": 0.0}

        tp_hits = sum(1 for item in history if str(item.get("result", "")).startswith("TP"))
        sl_hits = sum(1 for item in history if str(item.get("result", "")) == "SL")
        wins = tp_hits
        win_rate = (wins / total) * 100
        total_pnl = sum(
            float(pnl) if isinstance(pnl := item.get("pnl_pct", 0), (int, float, str)) else 0.0
            for item in history
        )
        return {
            "win_rate": win_rate,
            "tp_hits": tp_hits,
            "sl_hits": sl_hits,
            "total_pnl": total_pnl,
        }

    def build_summary_message(self, record: SignalRecord) -> str:
        summary = self.get_summary()
        is_win = record.result.startswith("TP")
        title_icon = "🎯✅" if is_win else "⛔️❌"
        result_label = "TAKE PROFIT HIT" if is_win else "STOP LOSS HIT"
        symbol = record.extra.get("display_symbol", record.symbol)
        timeframe = record.extra.get("timeframe")
        exchange = record.extra.get("exchange")

        header = f"{title_icon} {self.bot_name} - {result_label} {title_icon}"
        identifier = f"{record.created_at}_{symbol}_{record.direction.upper()}"

        lines = [
            header,
            "",
            f"🆔 {identifier}",
            "",
            f"📊 Symbol: {symbol}",
            f"📍 Direction: {record.direction.upper()}",
            f"💰 Entry: <code>{record.entry:.6f}</code>",
            f"🏁 Exit: <code>{record.exit:.6f}</code>",
            f"📈 P&L: <code>{record.pnl_pct:+.2f}%</code>",
        ]

        if timeframe or exchange:
            time_str = f"🕒 Timeframe: {timeframe}" if timeframe else ""
            exch_str = f"🏦 Exchange: {exchange}" if exchange else ""
            meta_line = " | ".join(filter(None, [time_str, exch_str]))
            if meta_line:
                lines.append(meta_line)

        lines.extend([
            "",
            "📊 Performance Stats:",
            f"Win Rate: {summary['win_rate']:.1f}%",
            f"TP Hits: {summary['tp_hits']} | SL Hits: {summary['sl_hits']}",
            f"Total P&L: {summary['total_pnl']:+.2f}%",
            "",
            f"⏰ {record.closed_at}",
        ])

        return "\n".join(lines)

    def build_initial_alert(
        self,
        symbol: str,
        direction: str,
        entry: float,
        tp1: Optional[float] = None,
        tp2: Optional[float] = None,
        sl: Optional[float] = None,
        extra_data: Optional[Dict[str, object]] = None,
    ) -> str:
        """Build initial alert message with historical stats for new signals."""
        extra = extra_data or {}
        emoji = "🟢" if direction.upper() in ("BULLISH", "LONG", "BUY") else "🔴"
        
        lines = [f"{emoji} <b>{direction.upper()} SIGNAL - {symbol}</b>", ""]
        
        # Historical TP/SL stats
        counts = self.symbol_tp_sl_counts(symbol)
        tp1_count = counts.get("TP1", 0)
        tp2_count = counts.get("TP2", 0)
        sl_count = counts.get("SL", 0)
        total = tp1_count + tp2_count + sl_count
        
        if total > 0:
            win_rate = ((tp1_count + tp2_count) / total) * 100.0
            lines.append(
                f"📈 <b>History:</b> TP1 {tp1_count} | TP2 {tp2_count} | SL {sl_count} "
                f"(Win rate: {win_rate:.1f}%)"
            )
            lines.append("")
        
        # Entry price
        lines.append(f"💰 Entry: <code>{entry:.6f}</code>")
        
        # Add extra data fields
        for key, value in extra.items():
            if key in ("display_symbol", "timeframe", "exchange", "signal_id"):
                continue
            if isinstance(value, (int, float)):
                lines.append(f"📊 {key.replace('_', ' ').title()}: <code>{value:.6f}</code>")
            elif isinstance(value, str):
                lines.append(f"📝 {key.replace('_', ' ').title()}: {value}")
        
        # Targets
        if tp1 or tp2 or sl:
            lines.append("")
            if tp1 and tp2:
                lines.append(
                    f"🎯 Targets: TP1 <code>{tp1:.6f}</code> | TP2 <code>{tp2:.6f}</code>"
                )
            elif tp1:
                lines.append(f"🎯 Target: <code>{tp1:.6f}</code>")
            if sl:
                lines.append(f"🛑 Stop: <code>{sl:.6f}</code>")
        
        # Timestamp
        lines.append("")
        lines.append(f"⏱️ {_utc_now()}")
        
        return "\n".join(lines)
