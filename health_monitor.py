"""Health monitoring module for trading bots with heartbeat and error tracking."""

import logging
import time
from datetime import datetime, timezone
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class HealthMonitor:
    """Monitors bot health and sends periodic heartbeat alerts."""
    
    def __init__(self, bot_name: str, notifier: Optional[Any] = None, heartbeat_interval: int = 3600,
                 error_rate_threshold: int = 10, stale_cycle_minutes: int = 15) -> None:
        """
        Initialize health monitor.
        
        Args:
            bot_name: Name of the bot for identification
            notifier: TelegramNotifier instance
            heartbeat_interval: Seconds between heartbeat messages (default: 1 hour)
            error_rate_threshold: Number of errors before sending alert
            stale_cycle_minutes: Minutes without cycle before sending stale alert
        """
        self.bot_name = bot_name
        self.notifier = notifier
        self.heartbeat_interval = heartbeat_interval
        self.error_rate_threshold = error_rate_threshold
        self.stale_cycle_minutes = stale_cycle_minutes
        self.last_heartbeat = time.time()
        self.start_time = time.time()
        self.last_cycle_time = time.time()
        self.cycle_count = 0
        self.error_count = 0
        self.errors_since_last_alert = 0
        self.last_errors: List[str] = []
        self.max_error_history = 10
        self.alert_sent_for_stale = False
        
    def check_heartbeat(self) -> None:
        """Check if it's time to send heartbeat and send if needed."""
        current_time = time.time()
        if current_time - self.last_heartbeat >= self.heartbeat_interval:
            self._send_heartbeat()
            self.last_heartbeat = current_time
    
    def _send_heartbeat(self) -> None:
        """Send heartbeat message to Telegram."""
        if not self.notifier:
            return
            
        uptime_seconds = int(time.time() - self.start_time)
        uptime_hours = uptime_seconds // 3600
        uptime_minutes = (uptime_seconds % 3600) // 60
        
        message = f"💚 <b>{self.bot_name} - Health Check</b>\n\n"
        message += "✅ Status: <b>RUNNING</b>\n"
        message += f"⏱ Uptime: {uptime_hours}h {uptime_minutes}m\n"
        message += f"🔄 Cycles completed: {self.cycle_count}\n"
        message += f"⚠️ Errors: {self.error_count}\n"
        message += f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
        
        if self.last_errors:
            message += "\n\n<b>Recent Errors:</b>\n"
            for err in self.last_errors[-3:]:  # Show last 3 errors
                message += f"• {err}\n"
        
        try:
            self.notifier.send_message(message)
            logger.info("Heartbeat sent successfully")
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")
    
    def record_cycle(self) -> None:
        """Record successful cycle completion."""
        self.cycle_count += 1
        self.last_cycle_time = time.time()
        self.alert_sent_for_stale = False
        self.check_heartbeat()
        self._check_stale_cycle()
    
    def record_error(self, error: str) -> None:
        """Record an error occurrence."""
        self.error_count += 1
        self.errors_since_last_alert += 1
        timestamp = datetime.now(timezone.utc).strftime('%H:%M:%S')
        error_msg = f"[{timestamp}] {error}"
        self.last_errors.append(error_msg)
        
        # Keep only recent errors
        if len(self.last_errors) > self.max_error_history:
            self.last_errors.pop(0)
        
        logger.warning(f"Error recorded: {error}")
        
        # Check if error rate threshold exceeded
        if self.errors_since_last_alert >= self.error_rate_threshold:
            self._send_error_rate_alert()
    
    def _check_stale_cycle(self) -> None:
        """Check if bot hasn't completed a cycle in too long."""
        if self.alert_sent_for_stale:
            return
            
        minutes_since_cycle = (time.time() - self.last_cycle_time) / 60
        if minutes_since_cycle >= self.stale_cycle_minutes:
            self._send_stale_alert(minutes_since_cycle)
            self.alert_sent_for_stale = True
    
    def _send_error_rate_alert(self) -> None:
        """Send alert when error rate threshold is exceeded."""
        if not self.notifier:
            return
            
        message = f"🚨 <b>{self.bot_name} - HIGH ERROR RATE ALERT</b>\n\n"
        message += f"⚠️ <b>{self.errors_since_last_alert} errors</b> detected!\n"
        message += f"📊 Total errors: {self.error_count}\n"
        message += f"🔄 Total cycles: {self.cycle_count}\n\n"
        
        if self.last_errors:
            message += "<b>Recent Errors:</b>\n"
            for err in self.last_errors[-5:]:
                message += f"• {err}\n"
        
        message += f"\n⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
        
        try:
            self.notifier.send_message(message)
            self.errors_since_last_alert = 0
            logger.info("Error rate alert sent")
        except Exception as e:
            logger.error(f"Failed to send error rate alert: {e}")
    
    def _send_stale_alert(self, minutes: float) -> None:
        """Send alert when bot hasn't completed a cycle in too long."""
        if not self.notifier:
            return
            
        message = f"⚠️ <b>{self.bot_name} - STALE CYCLE ALERT</b>\n\n"
        message += f"🕐 No cycle completed in <b>{minutes:.1f} minutes</b>\n"
        message += f"📊 Last cycle: {self.cycle_count}\n"
        message += f"⚠️ Total errors: {self.error_count}\n"
        
        if self.last_errors:
            message += "\n<b>Recent Errors:</b>\n"
            for err in self.last_errors[-3:]:
                message += f"• {err}\n"
        
        message += f"\n⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
        message += "\n\n💡 <b>Action:</b> Check if bot is stuck or has crashed"
        
        try:
            self.notifier.send_message(message)
            logger.info("Stale cycle alert sent")
        except Exception as e:
            logger.error(f"Failed to send stale alert: {e}")
    
    def send_startup_message(self) -> None:
        """Send bot startup notification."""
        if not self.notifier:
            return
            
        message = f"🚀 <b>{self.bot_name} Started</b>\n\n"
        message += "✅ Bot is now monitoring markets\n"
        message += f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        message += f"💚 Heartbeat interval: {self.heartbeat_interval//60} minutes"
        
        try:
            self.notifier.send_message(message)
            logger.info("Startup message sent")
        except Exception as e:
            logger.error(f"Failed to send startup message: {e}")
    
    def send_shutdown_message(self) -> None:
        """Send bot shutdown notification."""
        if not self.notifier:
            return
            
        uptime_seconds = int(time.time() - self.start_time)
        uptime_hours = uptime_seconds // 3600
        uptime_minutes = (uptime_seconds % 3600) // 60
        
        message = f"🛑 <b>{self.bot_name} Stopped</b>\n\n"
        message += f"⏱ Total uptime: {uptime_hours}h {uptime_minutes}m\n"
        message += f"🔄 Total cycles: {self.cycle_count}\n"
        message += f"⚠️ Total errors: {self.error_count}\n"
        message += f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
        
        try:
            self.notifier.send_message(message)
            logger.info("Shutdown message sent")
        except Exception as e:
            logger.error(f"Failed to send shutdown message: {e}")


# RateLimiter re-exported for backwards compatibility
# New code should use: from rate_limit_handler import RateLimitHandler
try:
    from rate_limit_handler import RateLimitHandler as RateLimiter
except ImportError:
    RateLimiter = None  # type: ignore
