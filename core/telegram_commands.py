#!/usr/bin/env python3
"""
Telegram Command Handler - Interactive Bot Control

Provides interactive Telegram commands for managing the trading bots:
- /status - Show all bot status
- /positions - Show open positions
- /performance - Weekly P&L summary
- /daily - Daily P&L summary
- /pause <bot> - Pause a specific bot
- /resume <bot> - Resume a specific bot
- /emergency_stop - Trigger emergency stop
- /reset_emergency - Reset emergency stop
- /help - Show available commands
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


class TelegramCommandHandler:
    """
    Handles incoming Telegram commands for bot control.
    
    Features:
    - Poll for new commands
    - Execute commands and send responses
    - Authorization check (only allowed users)
    
    Usage:
        handler = TelegramCommandHandler(bot_token, chat_id)
        handler.register_command("/status", status_callback)
        handler.start_polling()
    """
    
    def __init__(
        self,
        bot_token: str,
        allowed_chat_ids: List[str],
        poll_interval: int = 5,
    ):
        self.bot_token = bot_token
        self.allowed_chat_ids = [str(cid) for cid in allowed_chat_ids]
        self.poll_interval = poll_interval
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.commands: Dict[str, Callable] = {}
        self.running = False
        self.poll_thread: Optional[threading.Thread] = None
        self.last_update_id = 0
        
        # Register default commands
        self._register_default_commands()
        
        logger.info("TelegramCommandHandler initialized")
    
    def _register_default_commands(self) -> None:
        """Register built-in commands."""
        self.register_command("/help", self._cmd_help)
        self.register_command("/start", self._cmd_help)
    
    def register_command(self, command: str, callback: Callable[[str, str], str]) -> None:
        """
        Register a command handler.
        
        Args:
            command: Command string (e.g., "/status")
            callback: Function(chat_id, args) -> response message
        """
        self.commands[command.lower()] = callback
        logger.debug("Registered command: %s", command)
    
    def _cmd_help(self, chat_id: str, args: str) -> str:
        """Show available commands."""
        lines = [
            "🤖 <b>Trading Bot Commands</b>\n",
            "<b>Status & Monitoring:</b>",
            "  /status - Bot status dashboard",
            "  /positions - Open positions",
            "  /performance - Weekly P&L summary",
            "  /daily - Daily P&L summary",
            "",
            "<b>Control:</b>",
            "  /pause &lt;bot&gt; - Pause a specific bot",
            "  /resume &lt;bot&gt; - Resume a specific bot",
            "  /restart &lt;bot&gt; - Restart a specific bot",
            "",
            "<b>Emergency:</b>",
            "  /emergency_stop - Stop all new positions",
            "  /reset_emergency - Reset emergency stop",
            "",
            "<b>Other:</b>",
            "  /sync - Sync trades from bot stats",
            "  /help - Show this message",
        ]
        return "\n".join(lines)
    
    def send_message(self, chat_id: str, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message to a chat."""
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error("Failed to send message: %s", e)
            return False
    
    def _get_updates(self) -> List[Dict[str, Any]]:
        """Get new updates from Telegram."""
        try:
            url = f"{self.base_url}/getUpdates"
            params = {
                "offset": self.last_update_id + 1,
                "timeout": 30,
                "allowed_updates": ["message"],
            }
            response = requests.get(url, params=params, timeout=35)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    return data.get("result", [])
            
        except requests.Timeout:
            pass  # Normal for long polling
        except Exception as e:
            logger.error("Failed to get updates: %s", e)
        
        return []
    
    def _process_update(self, update: Dict[str, Any]) -> None:
        """Process a single update."""
        update_id = update.get("update_id", 0)
        if update_id > self.last_update_id:
            self.last_update_id = update_id
        
        message = update.get("message", {})
        if not message:
            return
        
        chat = message.get("chat", {})
        chat_id = str(chat.get("id", ""))
        
        # Authorization check
        if chat_id not in self.allowed_chat_ids:
            logger.warning("Unauthorized command attempt from chat_id: %s", chat_id)
            self.send_message(chat_id, "⛔ Unauthorized. This incident will be logged.")
            return
        
        text = message.get("text", "").strip()
        if not text.startswith("/"):
            return
        
        # Parse command and args
        parts = text.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        # Remove @botname suffix if present
        if "@" in command:
            command = command.split("@")[0]
        
        # Execute command
        if command in self.commands:
            try:
                response = self.commands[command](chat_id, args)
                if response:
                    self.send_message(chat_id, response)
            except Exception as e:
                logger.error("Command %s failed: %s", command, e)
                self.send_message(chat_id, f"❌ Command failed: {str(e)[:100]}")
        else:
            self.send_message(chat_id, f"❓ Unknown command: {command}\n\nUse /help for available commands.")
    
    def _poll_loop(self) -> None:
        """Background polling loop."""
        while self.running:
            try:
                updates = self._get_updates()
                for update in updates:
                    if not self.running:
                        break
                    self._process_update(update)
            except Exception as e:
                logger.error("Poll loop error: %s", e)
                time.sleep(5)
    
    def start_polling(self) -> None:
        """Start polling for commands."""
        if self.poll_thread and self.poll_thread.is_alive():
            return
        
        self.running = True
        self.poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.poll_thread.start()
        logger.info("Telegram command polling started")
    
    def stop_polling(self) -> None:
        """Stop polling."""
        self.running = False
        if self.poll_thread:
            self.poll_thread.join(timeout=10)
        logger.info("Telegram command polling stopped")


def create_command_handler(
    orchestrator: Any = None,
    portfolio_manager: Any = None,
    pnl_reporter: Any = None,
) -> Optional[TelegramCommandHandler]:
    """
    Create a TelegramCommandHandler with all commands registered.
    
    Args:
        orchestrator: BotOrchestrator instance
        portfolio_manager: PortfolioRiskManager instance
        pnl_reporter: PnLReporter instance
    
    Returns:
        Configured TelegramCommandHandler or None if credentials missing
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        logger.warning("Telegram credentials not found, command handler disabled")
        return None
    
    handler = TelegramCommandHandler(
        bot_token=bot_token,
        allowed_chat_ids=[chat_id],
    )
    
    # Register status command
    def cmd_status(chat_id: str, args: str) -> str:
        if orchestrator:
            return orchestrator.get_status_message()
        return "❌ Orchestrator not available"
    
    handler.register_command("/status", cmd_status)
    
    # Register positions command
    def cmd_positions(chat_id: str, args: str) -> str:
        if portfolio_manager:
            return portfolio_manager.get_positions_summary()
        return "❌ Portfolio manager not available"
    
    handler.register_command("/positions", cmd_positions)
    
    # Register performance command
    def cmd_performance(chat_id: str, args: str) -> str:
        if pnl_reporter:
            return pnl_reporter.format_weekly_summary_message()
        return "❌ P&L reporter not available"
    
    handler.register_command("/performance", cmd_performance)
    handler.register_command("/weekly", cmd_performance)
    
    # Register daily command
    def cmd_daily(chat_id: str, args: str) -> str:
        if pnl_reporter:
            return pnl_reporter.format_daily_summary_message()
        return "❌ P&L reporter not available"
    
    handler.register_command("/daily", cmd_daily)
    
    # Register pause command
    def cmd_pause(chat_id: str, args: str) -> str:
        if not args:
            return "❌ Usage: /pause <bot_name>"
        if orchestrator:
            success = orchestrator.stop_bot(args.strip())
            return f"{'✅' if success else '❌'} {'Paused' if success else 'Failed to pause'} {args.strip()}"
        return "❌ Orchestrator not available"
    
    handler.register_command("/pause", cmd_pause)
    handler.register_command("/stop", cmd_pause)
    
    # Register resume command
    def cmd_resume(chat_id: str, args: str) -> str:
        if not args:
            return "❌ Usage: /resume <bot_name>"
        if orchestrator:
            success = orchestrator.start_bot(args.strip())
            return f"{'✅' if success else '❌'} {'Resumed' if success else 'Failed to resume'} {args.strip()}"
        return "❌ Orchestrator not available"
    
    handler.register_command("/resume", cmd_resume)
    handler.register_command("/start_bot", cmd_resume)
    
    # Register restart command
    def cmd_restart(chat_id: str, args: str) -> str:
        if not args:
            return "❌ Usage: /restart <bot_name>"
        if orchestrator:
            success = orchestrator.restart_bot(args.strip())
            return f"{'✅' if success else '❌'} {'Restarted' if success else 'Failed to restart'} {args.strip()}"
        return "❌ Orchestrator not available"
    
    handler.register_command("/restart", cmd_restart)
    
    # Register emergency stop command
    def cmd_emergency_stop(chat_id: str, args: str) -> str:
        if portfolio_manager:
            portfolio_manager.trigger_emergency_stop(f"Manual trigger via Telegram")
            return "🚨 Emergency stop activated!\n\nAll new positions blocked."
        return "❌ Portfolio manager not available"
    
    handler.register_command("/emergency_stop", cmd_emergency_stop)
    handler.register_command("/emergency", cmd_emergency_stop)
    
    # Register reset emergency command
    def cmd_reset_emergency(chat_id: str, args: str) -> str:
        if portfolio_manager:
            portfolio_manager.reset_emergency_stop()
            return "✅ Emergency stop reset. Trading resumed."
        return "❌ Portfolio manager not available"
    
    handler.register_command("/reset_emergency", cmd_reset_emergency)
    handler.register_command("/reset", cmd_reset_emergency)
    
    # Register sync command
    def cmd_sync(chat_id: str, args: str) -> str:
        if pnl_reporter:
            count = pnl_reporter.sync_from_bot_stats()
            return f"✅ Synced {count} new trades from bot stats"
        return "❌ P&L reporter not available"
    
    handler.register_command("/sync", cmd_sync)
    
    return handler


if __name__ == "__main__":
    # Test the command handler
    import sys
    
    # Load environment
    try:
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / ".env")
    except ImportError:
        pass
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    handler = create_command_handler()
    
    if handler:
        print("Starting Telegram command polling...")
        print("Send /help to your bot to test")
        handler.start_polling()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            handler.stop_polling()
            print("\nStopped")
    else:
        print("Failed to create command handler. Check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")

