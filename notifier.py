import logging
import requests
import json
import secrets
import string
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path

from file_lock import file_lock

logger = logging.getLogger(__name__)

def generate_signal_id(symbol: str, direction: str) -> str:
    """Generate a unique signal ID for tracking.
    Format: B1-{SYMBOL}-{DIRECTION}-{RANDOM4}
    Example: B1-ADA-L-X7K2

    Uses cryptographically secure random generation for security.
    """
    symbol_short = symbol.replace('/USDT:USDT', '').replace('/USDT', '').replace(':USDT', '')[:6]
    dir_short = 'L' if direction == 'LONG' else 'S'
    random_part = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(4))
    return f"B1-{symbol_short}-{dir_short}-{random_part}"

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str, signals_log_file: str = "signals_log.json"):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.signals_log_file = signals_log_file
        
        if not bot_token or not chat_id:
            logger.error("Telegram bot token or chat ID not configured")
            raise ValueError("Telegram credentials not configured")
        
        logger.info("Telegram notifier initialized")
    
    def save_signal_to_json(self, signal: Dict, signals_log_file: Optional[str] = None, signal_id: Optional[str] = None) -> bool:
        try:
            log_file = Path(signals_log_file or self.signals_log_file)

            # Generate signal_id if not provided
            if not signal_id:
                signal_id = generate_signal_id(signal['symbol'], signal['action'])

            signal_entry = {
                "signal_id": signal_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": signal['symbol'],
                "direction": signal['action'],
                "entry": signal['entry_price'],
                "sl": signal['stop_loss'],
                "tp": signal['take_profit'],
                "confidence": signal['score'],
                "max_confidence": signal.get('max_score', 12.3),
                "adx": signal.get('adx', None),
                "rsx": signal.get('rsx', None),
                "strategy": signal.get('strategy_mode', 'unknown'),
                "risk_reward_ratio": round(abs(signal['take_profit'] - signal['entry_price']) / abs(signal['entry_price'] - signal['stop_loss']), 2) if signal['entry_price'] != signal['stop_loss'] else 0,
                "position_size": signal.get('position_size'),
                "quality_tier": signal.get('quality_tier'),
            }

            if signal.get('take_profit_partial') is not None:
                signal_entry['tp_partial'] = signal['take_profit_partial']
                signal_entry['partial_allocation'] = signal.get('partial_allocation')
                signal_entry['partial_rr'] = signal.get('partial_rr')
            if signal.get('take_profit_final') is not None:
                signal_entry['tp_final'] = signal['take_profit_final']
                signal_entry['final_rr'] = signal.get('final_rr')
                signal_entry['final_allocation'] = signal.get('final_allocation')
            if signal.get('time_stop_minutes') is not None:
                signal_entry['time_stop_minutes'] = signal['time_stop_minutes']
            elif signal.get('time_stop_bars') is not None:
                signal_entry['time_stop_bars'] = signal['time_stop_bars']

            # Use file locking to prevent concurrent access issues
            with file_lock(log_file):
                signals = []
                if log_file.exists():
                    signals = json.loads(log_file.read_text())
                signals.append(signal_entry)
                log_file.write_text(json.dumps(signals, indent=2))

            logger.info(f"Signal saved to {log_file} with ID: {signal_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving signal to JSON: {e}")
            return False
    
    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info("Message sent successfully to Telegram")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send message to Telegram: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending message: {e}")
            return False
    
    def send_signal_alert(self, signal: Dict) -> bool:
        try:
            action = signal['action']
            symbol = signal['symbol']
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            take_profit = signal['take_profit']
            score = signal['score']
            max_score = signal.get('max_score', 10.0)
            rsx = signal.get('rsx', signal.get('rsi', 50.0))
            adx = signal['adx']
            
            # Generate unique signal ID for tracking
            signal_id = generate_signal_id(symbol, action)
            
            confidence_pct = (score / max_score) * 100
            
            # Determine quality level
            if score >= 6.0:
                quality = "🔥 STRONG"
                emoji = "🟢🟢" if action == 'LONG' else "🔴🔴"
            elif score >= 4.5:
                quality = "✅ GOOD"
                emoji = "🟢" if action == 'LONG' else "🔴"
            else:
                quality = "⚠️ WEAK"
                emoji = "🟡" if action == 'LONG' else "🟠"
            
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            # Get position tier info
            position_size = signal.get('position_size', 1.0)
            quality_tier = signal.get('quality_tier', quality)
            
            message = f"{emoji} <b>{action} SIGNAL - {symbol}</b> {quality_tier}\n\n"
            message += f"🆔 <code>{signal_id}</code>\n\n"
            message += f"💰 <b>Entry:</b> {entry_price:.6f}\n"
            message += f"🛑 <b>Stop Loss:</b> {stop_loss:.6f}\n"
            message += f"🎯 <b>Take Profit:</b> {take_profit:.6f}\n"
            message += f"📊 <b>Risk/Reward:</b> 1:{rr_ratio:.2f}\n\n"
            message += f"⭐ <b>Signal Strength:</b> {score:.1f}/12.3 ({confidence_pct:.0f}%)\n"
            message += f"📈 <b>RSX:</b> {rsx:.1f} | 💪 <b>ADX:</b> {adx:.1f}\n"
            message += "⏰ <b>Timeframe:</b> 15m | 🔔 MEXC Futures\n\n"
            
            # Add position sizing advice based on tier
            if score >= 7.0:
                message += f"💡 <b>Position Size:</b> {position_size*100:.0f}% (PREMIUM - Full confidence!)"
            elif score >= 6.0:
                message += f"💡 <b>Position Size:</b> {position_size*100:.0f}% (QUALITY - Good setup)"
            else:
                message += f"💡 <b>Position Size:</b> {position_size*100:.0f}% (STANDARD - Be cautious)"
            
            success = self.send_message(message)
            if success:
                self.save_signal_to_json(signal, signal_id=signal_id)
                logger.info(f"Signal sent with ID: {signal_id}")
            return success
            
        except Exception as e:
            logger.error(f"Error formatting signal alert: {e}")
            return False
    
    def send_startup_message(self, pairs: list, available_pairs: dict) -> bool:
        try:
            available_count = sum(1 for v in available_pairs.values() if v)
            total_count = len(pairs)
            
            message = "🤖 <b>Trading Bot Started</b>\n\n"
            message += f"📊 <b>Monitoring {available_count}/{total_count} pairs on MEXC Futures (15m)</b>\n\n"
            
            message += "<b>Available Pairs:</b>\n"
            for pair, available in available_pairs.items():
                if available:
                    message += f"✅ {pair}\n"
            
            unavailable = [pair for pair, available in available_pairs.items() if not available]
            if unavailable:
                message += "\n<b>Unavailable Pairs:</b>\n"
                for pair in unavailable:
                    message += f"❌ {pair}\n"
            
            message += "\n⚙️ <b>Strategy:</b> Confluence (EMA, RSI, MACD, ADX, Volume)\n"
            message += "🎯 <b>Alert Mode:</b> Entry/Exit with TP/SL\n"
            message += "✨ Ready to send trading signals!"
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending startup message: {e}")
            return False
    
    def send_error_message(self, error: str) -> bool:
        try:
            message = f"⚠️ <b>Trading Bot Error</b>\n\n{error}"
            return self.send_message(message)
        except Exception as e:
            logger.error(f"Error sending error message: {e}")
            return False
