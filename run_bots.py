#!/usr/bin/env python3
"""
Unified Bot Runner - Main Entry Point

This script provides a single entry point to run all trading bots with:
- Health monitoring and automatic restart
- Portfolio-level risk management
- Daily P&L summaries
- Interactive Telegram commands

Usage:
    python run_bots.py                  # Start all bots with monitoring
    python run_bots.py --status         # Show status and exit
    python run_bots.py --bot strat_bot  # Start only specific bot
    python run_bots.py --no-monitor     # Start without health monitoring
"""

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path for imports
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

# Load version if available
VERSION_FILE = BASE_DIR / "VERSION"
VERSION = "unknown"
if VERSION_FILE.exists():
    try:
        VERSION = VERSION_FILE.read_text().strip()
    except Exception:
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_bots")

# Log version on startup
logger.info(f"Trading Bot System v{VERSION}")

# Reduce noise from libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)


def setup_file_logging() -> None:
    """Setup file logging with rotation."""
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"runner_{datetime.now().strftime('%Y%m%d')}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    ))
    
    logging.getLogger().addHandler(file_handler)
    logger.info("File logging enabled: %s", log_file)


def main():
    parser = argparse.ArgumentParser(
        description="Unified Trading Bot Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_bots.py                  # Start all enabled bots
  python run_bots.py --status         # Show current status
  python run_bots.py --bot strat_bot  # Start only strat_bot
  python run_bots.py --no-telegram    # Disable Telegram commands
        """
    )
    
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show status and exit"
    )
    parser.add_argument(
        "--bot", "-b",
        type=str,
        help="Start only a specific bot"
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Disable health monitoring"
    )
    parser.add_argument(
        "--no-telegram",
        action="store_true",
        help="Disable Telegram command handler"
    )
    parser.add_argument(
        "--no-pnl",
        action="store_true",
        help="Disable P&L reporting"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup file logging
    setup_file_logging()
    
    logger.info("=" * 60)
    logger.info("Trading Bot Runner Starting")
    logger.info("=" * 60)
    
    # Import core components
    try:
        from core.portfolio_manager import get_portfolio_manager
        from core.bot_orchestrator import get_orchestrator
        from core.pnl_reporter import get_pnl_reporter
        from core.telegram_commands import create_command_handler
    except ImportError as e:
        logger.error("Failed to import core components: %s", e)
        logger.error("Make sure you're running from the project directory")
        return 1
    
    # Try to import notifier
    notifier = None
    try:
        from notifier import send_telegram_message
        
        class NotifierWrapper:
            def send_message(self, message: str, parse_mode: str = "HTML"):
                send_telegram_message(message.replace("<b>", "*").replace("</b>", "*"))
        
        notifier = NotifierWrapper()
        logger.info("Telegram notifier available")
    except ImportError:
        logger.warning("Notifier not available, notifications disabled")
    
    # Initialize components
    portfolio_manager = get_portfolio_manager()
    orchestrator = get_orchestrator()
    pnl_reporter = get_pnl_reporter() if not args.no_pnl else None
    
    # Set notifiers
    if notifier:
        portfolio_manager.set_notifier(notifier)
        orchestrator.set_notifier(notifier)
        if pnl_reporter:
            pnl_reporter.set_notifier(notifier)
    
    # Status command - just show and exit
    if args.status:
        print(orchestrator.get_status_message().replace("<b>", "").replace("</b>", ""))
        print()
        print(portfolio_manager.get_positions_summary().replace("<b>", "").replace("</b>", ""))
        return 0
    
    # Setup Telegram command handler
    command_handler = None
    if not args.no_telegram:
        command_handler = create_command_handler(
            orchestrator=orchestrator,
            portfolio_manager=portfolio_manager,
            pnl_reporter=pnl_reporter,
        )
        if command_handler:
            command_handler.start_polling()
            logger.info("Telegram command handler started")
    
    # Start P&L reporter scheduler
    if pnl_reporter:
        pnl_reporter.start_scheduler()
        logger.info("P&L reporter scheduler started")
    
    # Define shutdown handler
    running = True
    
    def shutdown_handler(signum, frame):
        nonlocal running
        logger.info("Shutdown signal received...")
        running = False
    
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)
    
    # Start bots
    if args.bot:
        # Start specific bot
        success = orchestrator.start_bot(args.bot)
        if not success:
            logger.error("Failed to start bot: %s", args.bot)
            return 1
        logger.info("Started bot: %s", args.bot)
    else:
        # Start all enabled bots
        results = orchestrator.start_all()
        started = sum(1 for s in results.values() if s)
        logger.info("Started %d/%d bots", started, len(results))
    
    # Start health monitoring
    if not args.no_monitor:
        orchestrator.start_monitoring()
        logger.info("Health monitoring started")
    
    # Send startup notification
    if notifier:
        try:
            status = orchestrator.get_status()
            notifier.send_message(
                f"🚀 *Trading Bots Started*\n\n"
                f"Running: {status['running_count']}/{status['total_count']} bots\n"
                f"Monitoring: {'Active' if not args.no_monitor else 'Disabled'}\n"
                f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
            )
        except Exception as e:
            logger.warning("Failed to send startup notification: %s", e)
    
    logger.info("=" * 60)
    logger.info("All systems running. Press Ctrl+C to stop.")
    logger.info("=" * 60)
    
    # Main loop
    try:
        while running:
            time.sleep(1)
            
            # Periodic tasks
            # (Health monitoring runs in its own thread)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    
    # Cleanup
    logger.info("Shutting down...")
    
    if command_handler:
        command_handler.stop_polling()
    
    if pnl_reporter:
        pnl_reporter.stop_scheduler()
    
    orchestrator.stop_monitoring()
    orchestrator.stop_all()
    
    # Send shutdown notification
    if notifier:
        try:
            notifier.send_message(
                f"⏹️ *Trading Bots Stopped*\n\n"
                f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
            )
        except Exception:
            pass
    
    logger.info("Shutdown complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())

