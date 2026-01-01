#!/usr/bin/env python3
"""
Bot Orchestrator - Unified Bot Management with Health Monitoring

Provides centralized management for all trading bots:
- Start/stop/restart individual bots
- Health monitoring and automatic restart on crash
- Unified status dashboard
- Cross-bot coordination
"""

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
ORCHESTRATOR_STATE_FILE = BASE_DIR / "orchestrator_state.json"


@dataclass
class BotInfo:
    """Information about a registered bot."""
    name: str
    script_path: str
    working_dir: str
    enabled: bool = True
    auto_restart: bool = True
    max_restarts: int = 5
    restart_cooldown_seconds: int = 60
    health_check_interval: int = 60
    
    # Runtime state
    pid: Optional[int] = None
    status: str = "stopped"  # stopped, running, crashed, disabled
    last_started: Optional[str] = None
    last_health_check: Optional[str] = None
    restart_count: int = 0
    last_error: Optional[str] = None
    cycles_completed: int = 0
    signals_generated: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BotInfo":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class OrchestratorConfig:
    """Orchestrator configuration."""
    # Health check settings
    health_check_interval: int = 60  # seconds
    # Restart settings
    max_total_restarts_per_hour: int = 20
    # Notification settings
    notify_on_crash: bool = True
    notify_on_restart: bool = True
    notify_health_summary_interval: int = 3600  # Every hour
    # Auto-discovery
    auto_discover_bots: bool = True


class BotOrchestrator:
    """
    Centralized bot management and monitoring.
    
    Features:
    - Register and manage multiple bots
    - Health monitoring with automatic restart
    - Unified status dashboard
    - Graceful shutdown coordination
    
    Usage:
        orchestrator = BotOrchestrator()
        orchestrator.register_bot("strat_bot", "strat_bot/strat_bot.py")
        orchestrator.register_bot("volume_bot", "volume_bot/volume_vn_bot.py")
        orchestrator.start_all()
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self.bots: Dict[str, BotInfo] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self.notifier: Optional[Any] = None
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.state_lock = threading.RLock()
        self.restart_history: List[datetime] = []
        self.last_health_summary: datetime = datetime.now(timezone.utc)
        
        # Load state
        self._load_state()
        
        # Auto-discover bots if enabled
        if self.config.auto_discover_bots:
            self._auto_discover_bots()
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        logger.info("BotOrchestrator initialized with %d bots", len(self.bots))
    
    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        logger.info("Received signal %d, initiating graceful shutdown...", signum)
        self.stop_all()
    
    def set_notifier(self, notifier: Any) -> None:
        """Set the Telegram notifier for alerts."""
        self.notifier = notifier
    
    def _load_state(self) -> None:
        """Load orchestrator state from file."""
        if not ORCHESTRATOR_STATE_FILE.exists():
            return
        
        try:
            with open(ORCHESTRATOR_STATE_FILE, 'r') as f:
                data = json.load(f)
            
            for bot_data in data.get("bots", []):
                try:
                    bot = BotInfo.from_dict(bot_data)
                    # Reset runtime state on load
                    bot.pid = None
                    bot.status = "stopped"
                    self.bots[bot.name] = bot
                except (TypeError, KeyError) as e:
                    logger.warning("Failed to load bot: %s", e)
            
            logger.info("Loaded %d bots from state", len(self.bots))
            
        except (json.JSONDecodeError, IOError) as e:
            logger.error("Failed to load orchestrator state: %s", e)
    
    def _save_state(self) -> None:
        """Save orchestrator state to file."""
        temp_file = ORCHESTRATOR_STATE_FILE.with_suffix('.tmp')
        
        try:
            data = {
                "bots": [bot.to_dict() for bot in self.bots.values()],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            temp_file.replace(ORCHESTRATOR_STATE_FILE)
            
        except Exception as e:
            logger.error("Failed to save orchestrator state: %s", e)
            if temp_file.exists():
                temp_file.unlink()
    
    def _auto_discover_bots(self) -> None:
        """Auto-discover bots in the project directory."""
        bot_patterns = [
            ("strat_bot", "strat_bot/strat_bot.py"),
            ("volume_bot", "volume_bot/volume_vn_bot.py"),
            ("harmonic_bot", "harmonic_bot/harmonic_bot.py"),
            ("fib_swing_bot", "fib_swing_bot/fib_swing_bot.py"),
            ("most_bot", "most_bot/most_bot.py"),
            ("psar_bot", "psar_bot/psar_bot.py"),
            ("orb_bot", "orb_bot/orb_bot.py"),
            ("mtf_bot", "mtf_bot/mtf_bot.py"),
            ("liquidation_bot", "liquidation_bot/liquidation_bot.py"),
            ("funding_bot", "funding_bot/funding_bot.py"),
            ("diy_bot", "diy_bot/diy_bot.py"),
            ("volume_profile_bot", "volume_profile_bot/volume_profile_bot.py"),
        ]
        
        for name, script in bot_patterns:
            script_path = BASE_DIR / script
            if script_path.exists() and name not in self.bots:
                self.register_bot(name, script)
                logger.debug("Auto-discovered bot: %s", name)
    
    def register_bot(
        self,
        name: str,
        script_path: str,
        enabled: bool = True,
        auto_restart: bool = True,
    ) -> None:
        """
        Register a bot with the orchestrator.
        
        Args:
            name: Unique bot name
            script_path: Path to bot script (relative to BASE_DIR)
            enabled: Whether bot should be started
            auto_restart: Whether to auto-restart on crash
        """
        full_path = BASE_DIR / script_path
        if not full_path.exists():
            logger.warning("Bot script not found: %s", full_path)
        
        with self.state_lock:
            if name in self.bots:
                # Update existing
                self.bots[name].script_path = script_path
                self.bots[name].enabled = enabled
                self.bots[name].auto_restart = auto_restart
            else:
                self.bots[name] = BotInfo(
                    name=name,
                    script_path=script_path,
                    working_dir=str(full_path.parent),
                    enabled=enabled,
                    auto_restart=auto_restart,
                )
            
            self._save_state()
    
    def start_bot(self, name: str) -> bool:
        """
        Start a specific bot.
        
        Args:
            name: Bot name to start
            
        Returns:
            True if started successfully
        """
        with self.state_lock:
            if name not in self.bots:
                logger.error("Bot not found: %s", name)
                return False
            
            bot = self.bots[name]
            
            if not bot.enabled:
                logger.info("Bot %s is disabled, not starting", name)
                return False
            
            if bot.status == "running" and bot.pid:
                # Check if actually running
                if self._is_process_running(bot.pid):
                    logger.info("Bot %s is already running (PID %d)", name, bot.pid)
                    return True
            
            # Start the bot
            try:
                script_path = BASE_DIR / bot.script_path
                working_dir = script_path.parent
                
                # Start as subprocess
                process = subprocess.Popen(
                    [sys.executable, str(script_path)],
                    cwd=str(working_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
                
                bot.pid = process.pid
                bot.status = "running"
                bot.last_started = datetime.now(timezone.utc).isoformat()
                bot.last_error = None
                
                self.processes[name] = process
                self._save_state()
                
                logger.info("Started bot %s (PID %d)", name, process.pid)
                return True
                
            except Exception as e:
                bot.status = "crashed"
                bot.last_error = str(e)
                logger.error("Failed to start bot %s: %s", name, e)
                return False
    
    def stop_bot(self, name: str, timeout: int = 10) -> bool:
        """
        Stop a specific bot gracefully.
        
        Args:
            name: Bot name to stop
            timeout: Seconds to wait for graceful shutdown
            
        Returns:
            True if stopped successfully
        """
        with self.state_lock:
            if name not in self.bots:
                logger.error("Bot not found: %s", name)
                return False
            
            bot = self.bots[name]
            
            if name in self.processes:
                process = self.processes[name]
                
                try:
                    # Send SIGTERM for graceful shutdown
                    process.terminate()
                    
                    # Wait for process to end
                    try:
                        process.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        # Force kill if didn't stop gracefully
                        process.kill()
                        process.wait(timeout=5)
                    
                    del self.processes[name]
                    
                except Exception as e:
                    logger.error("Error stopping bot %s: %s", name, e)
            
            # Also try to kill by PID if we have one
            if bot.pid and self._is_process_running(bot.pid):
                try:
                    os.kill(bot.pid, signal.SIGTERM)
                    time.sleep(2)
                    if self._is_process_running(bot.pid):
                        os.kill(bot.pid, signal.SIGKILL)
                except OSError:
                    pass
            
            bot.status = "stopped"
            bot.pid = None
            self._save_state()
            
            logger.info("Stopped bot %s", name)
            return True
    
    def restart_bot(self, name: str) -> bool:
        """Restart a specific bot."""
        with self.state_lock:
            if name not in self.bots:
                return False
            
            bot = self.bots[name]
            
            # Check restart limits
            self._cleanup_restart_history()
            if len(self.restart_history) >= self.config.max_total_restarts_per_hour:
                logger.warning("Max restarts per hour reached, not restarting %s", name)
                return False
            
            if bot.restart_count >= bot.max_restarts:
                logger.warning("Max restarts for %s reached (%d), disabling auto-restart",
                              name, bot.restart_count)
                bot.auto_restart = False
                self._save_state()
                return False
            
            # Stop and start
            self.stop_bot(name)
            time.sleep(bot.restart_cooldown_seconds)
            
            success = self.start_bot(name)
            
            if success:
                bot.restart_count += 1
                self.restart_history.append(datetime.now(timezone.utc))
                self._save_state()
                
                if self.config.notify_on_restart and self.notifier:
                    self._send_alert(
                        f"🔄 Bot Restarted: {name}\n"
                        f"Restart count: {bot.restart_count}/{bot.max_restarts}"
                    )
            
            return success
    
    def start_all(self) -> Dict[str, bool]:
        """Start all enabled bots."""
        results = {}
        for name, bot in self.bots.items():
            if bot.enabled:
                results[name] = self.start_bot(name)
        return results
    
    def stop_all(self) -> None:
        """Stop all running bots."""
        self.running = False
        
        for name in list(self.bots.keys()):
            self.stop_bot(name)
        
        logger.info("All bots stopped")
    
    def _is_process_running(self, pid: int) -> bool:
        """Check if a process is running by PID."""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    
    def _cleanup_restart_history(self) -> None:
        """Remove restart entries older than 1 hour."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        self.restart_history = [dt for dt in self.restart_history if dt > cutoff]
    
    def _check_bot_health(self, name: str) -> bool:
        """
        Check if a bot is healthy.
        
        Returns:
            True if healthy, False if crashed/unhealthy
        """
        if name not in self.bots:
            return False
        
        bot = self.bots[name]
        
        if bot.status != "running":
            return bot.status == "stopped" or bot.status == "disabled"
        
        # Check if process is still running
        if bot.pid and not self._is_process_running(bot.pid):
            bot.status = "crashed"
            bot.last_error = "Process not found"
            self._save_state()
            
            logger.warning("Bot %s crashed (PID %d not found)", name, bot.pid)
            
            if self.config.notify_on_crash and self.notifier:
                self._send_alert(
                    f"🚨 Bot Crashed: {name}\n"
                    f"PID: {bot.pid}\n"
                    f"Auto-restart: {'enabled' if bot.auto_restart else 'disabled'}"
                )
            
            return False
        
        # Check subprocess if we have reference
        if name in self.processes:
            process = self.processes[name]
            if process.poll() is not None:
                # Process has exited
                exit_code = process.returncode
                stderr = process.stderr.read() if process.stderr else b""
                
                bot.status = "crashed"
                bot.last_error = f"Exit code {exit_code}: {stderr.decode()[:200]}"
                self._save_state()
                
                logger.warning("Bot %s exited with code %d", name, exit_code)
                
                del self.processes[name]
                return False
        
        bot.last_health_check = datetime.now(timezone.utc).isoformat()
        return True
    
    def _health_check_loop(self) -> None:
        """Background thread for health monitoring."""
        while self.running:
            try:
                for name, bot in list(self.bots.items()):
                    if not self.running:
                        break
                    
                    if bot.status == "running":
                        is_healthy = self._check_bot_health(name)
                        
                        if not is_healthy and bot.auto_restart:
                            logger.info("Attempting to restart crashed bot: %s", name)
                            self.restart_bot(name)
                
                # Send periodic health summary
                now = datetime.now(timezone.utc)
                if (now - self.last_health_summary).seconds >= self.config.notify_health_summary_interval:
                    self._send_health_summary()
                    self.last_health_summary = now
                
            except Exception as e:
                logger.error("Health check error: %s", e)
            
            # Sleep in small increments to allow quick shutdown
            for _ in range(self.config.health_check_interval):
                if not self.running:
                    break
                time.sleep(1)
    
    def start_monitoring(self) -> None:
        """Start the health monitoring background thread."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop the health monitoring thread."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        logger.info("Health monitoring stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all bots."""
        with self.state_lock:
            return {
                "bots": {name: bot.to_dict() for name, bot in self.bots.items()},
                "running_count": sum(1 for b in self.bots.values() if b.status == "running"),
                "total_count": len(self.bots),
                "restarts_last_hour": len(self.restart_history),
                "monitoring_active": self.running,
            }
    
    def get_status_message(self) -> str:
        """Get formatted status message for Telegram."""
        with self.state_lock:
            lines = ["🤖 <b>Bot Status Dashboard</b>\n"]
            
            running = []
            stopped = []
            crashed = []
            
            for name, bot in sorted(self.bots.items()):
                if bot.status == "running":
                    running.append(f"  ✅ {name} (PID {bot.pid})")
                elif bot.status == "crashed":
                    crashed.append(f"  ❌ {name}: {bot.last_error or 'Unknown error'}")
                else:
                    stopped.append(f"  ⏹️ {name}")
            
            if running:
                lines.append(f"<b>Running ({len(running)}):</b>")
                lines.extend(running)
            
            if crashed:
                lines.append(f"\n<b>Crashed ({len(crashed)}):</b>")
                lines.extend(crashed)
            
            if stopped:
                lines.append(f"\n<b>Stopped ({len(stopped)}):</b>")
                lines.extend(stopped)
            
            lines.append(f"\n<b>Summary:</b>")
            lines.append(f"  Running: {len(running)}/{len(self.bots)}")
            lines.append(f"  Restarts (1h): {len(self.restart_history)}")
            lines.append(f"  Monitoring: {'Active' if self.running else 'Inactive'}")
            
            return "\n".join(lines)
    
    def _send_health_summary(self) -> None:
        """Send periodic health summary."""
        if not self.notifier:
            return
        
        message = self.get_status_message()
        try:
            self.notifier.send_message(message, parse_mode="HTML")
        except Exception as e:
            logger.error("Failed to send health summary: %s", e)
    
    def _send_alert(self, message: str) -> None:
        """Send alert via notifier."""
        if self.notifier:
            try:
                self.notifier.send_message(message, parse_mode="HTML")
            except Exception as e:
                logger.error("Failed to send alert: %s", e)


def get_orchestrator() -> BotOrchestrator:
    """Get or create the global BotOrchestrator instance."""
    if not hasattr(get_orchestrator, '_instance'):
        get_orchestrator._instance = BotOrchestrator()
    return get_orchestrator._instance


if __name__ == "__main__":
    # Example usage / CLI
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    parser = argparse.ArgumentParser(description="Bot Orchestrator")
    parser.add_argument("command", choices=["status", "start", "stop", "restart", "monitor"])
    parser.add_argument("--bot", help="Specific bot name (for start/stop/restart)")
    args = parser.parse_args()
    
    orchestrator = get_orchestrator()
    
    if args.command == "status":
        print(orchestrator.get_status_message().replace("<b>", "").replace("</b>", ""))
    
    elif args.command == "start":
        if args.bot:
            success = orchestrator.start_bot(args.bot)
            print(f"{'Started' if success else 'Failed to start'} {args.bot}")
        else:
            results = orchestrator.start_all()
            for name, success in results.items():
                print(f"{'✓' if success else '✗'} {name}")
    
    elif args.command == "stop":
        if args.bot:
            orchestrator.stop_bot(args.bot)
        else:
            orchestrator.stop_all()
        print("Stopped")
    
    elif args.command == "restart":
        if args.bot:
            success = orchestrator.restart_bot(args.bot)
            print(f"{'Restarted' if success else 'Failed to restart'} {args.bot}")
        else:
            print("Please specify --bot for restart")
    
    elif args.command == "monitor":
        print("Starting health monitoring... Press Ctrl+C to stop")
        orchestrator.start_all()
        orchestrator.start_monitoring()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            orchestrator.stop_all()
            orchestrator.stop_monitoring()

