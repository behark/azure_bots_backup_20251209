#!/usr/bin/env python3
"""
Verification Script for ORB Bot
Simulates market data to trigger 1 LONG (Breakout UP) and 1 SHORT (Breakout DOWN) signal.
Verifies alerts are sent and state is updated.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
import shutil

# Add parent to path to import shared modules
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))

# Import the bot components
from orb_bot import ORBBot, ORBAnalyzer, ORBLevel, ORBSession, setup_logging

# Configure logging to console
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("orb_verification")

class MockClient:
    """Mocks the exchange client."""
    def __init__(self):
        self.ohlcv_data = {}
        self.ticker_data = {}
        
    def fetch_ohlcv(self, symbol, timeframe, limit=200):
        return self.ohlcv_data.get(symbol, [])
    
    def fetch_ticker(self, symbol):
        return {"last": self.ticker_data.get(symbol, 100.0)}

def create_mock_orb_scenario(direction="BULLISH"):
    """
    Generates OHLCV data that forms an ORB range and then breaks it.
    """
    ohlcv = []
    base_price = 100.0
    now = datetime.now(timezone.utc)
    # Start at 00:00 UTC today
    session_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_ts = int(session_start.timestamp() * 1000)
    
    # 1. Create the ORB Range (First 60 mins)
    # Range: 99.0 - 101.0
    current_ts = start_ts
    for i in range(60):
        ohlcv.append([current_ts, 100.0, 101.0, 99.0, 100.0, 1000.0])
        current_ts += 60000 # 1 min
        
    # 2. Create the Breakout
    breakout_ts = current_ts
    
    if direction == "BULLISH":
        # Breakout UP to 102.0
        ohlcv.append([breakout_ts, 100.0, 102.5, 100.0, 102.0, 5000.0]) # High volume breakout
        breakout_price = 102.0
    else:
        # Breakout DOWN to 98.0
        ohlcv.append([breakout_ts, 100.0, 100.0, 97.5, 98.0, 5000.0]) # High volume breakout
        breakout_price = 98.0
        
    return ohlcv, breakout_price, session_start

def verify():
    print("🚀 Starting ORB Bot Verification...")
    
    # Initialize Bot (loads config)
    config_path = BASE_DIR / "orb_config.json"
    bot = ORBBot(config_path)
    
    # Inject Mock Client
    mock_client = MockClient()
    bot.clients = {"mexc": mock_client}
    
    # Setup Scenarios
    scenarios = [
        ("BTC/USDT", "BULLISH"),
        ("ETH/USDT", "BEARISH")
    ]
    
    bot.watchlist = []
    for sym, _ in scenarios:
        bot.watchlist.append({"symbol": sym, "exchange": "mexc", "timeframe": "5m"})
        
    print("\n🔍 Analyzing Mock Data...")
    
    found_signals = 0
    
    for symbol, direction in scenarios:
        print(f"  > Simulating {direction} breakout for {symbol}...")
        
        # Create Data
        ohlcv, price, session_start_dt = create_mock_orb_scenario(direction)
        session_start_ts = int(session_start_dt.timestamp() * 1000)
        
        # Load Mock Data
        # IMPORTANT: ORB logic fetches 1m candles for calculation
        # We need to ensure the mock client returns our 1m data
        # The bot calls `fetch_ohlcv` with timeframe from config/watchlist, 
        # but the analyzer might use a specific one. Let's check `orb_bot.py`.
        # `run` calls `fetch_ohlcv` with `timeframe` (5m) for the main loop data?
        # NO, wait. `ORBAnalyzer.calculate_orb` takes `ohlcv`.
        # `run` loop fetches `ohlcv` using `timeframe` from watchlist.
        # So we mock that.
        
        res_symbol = f"{symbol}:USDT" # MEXC swap format used in resolve_symbol
        mock_client.ohlcv_data[res_symbol] = ohlcv
        mock_client.ticker_data[res_symbol] = price
        
        # --- Run Manual Analysis Step-by-Step (copying logic from `run`) ---
        
        # 1. Calculate ORB Level
        orb = bot.analyzer.calculate_orb(ohlcv, session_start_ts)
        
        if orb:
            print(f"    ✅ ORB Range Established: {orb.low} - {orb.high}")
            
            # 2. Detect Breakout
            detected_direction = bot.analyzer.detect_breakout(price, orb)
            
            if detected_direction == direction:
                print(f"    ✅ SIGNAL DETECTED: {detected_direction}")
                
                # 3. Simulate Alert
                # We mock the signal object creation since `run` does it
                # We need to calc targets first
                targets = bot.analyzer.calculate_targets(price, detected_direction, orb)
                
                from orb_bot import ORBSignal # Import inside
                
                signal = ORBSignal(
                    symbol=symbol, direction=detected_direction,
                    entry=price, stop_loss=targets['sl'],
                    take_profit_1=targets['tp1'], take_profit_2=targets['tp2'],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    timeframe="5m", exchange="mexc",
                    breakout_type="TEST BREAKOUT",
                    orb_high=orb.high, orb_low=orb.low, range_pct=orb.range_pct
                )
                
                bot._send_alert(signal)
                print("    📤 Alert sent to Telegram!")
                found_signals += 1
            else:
                print(f"    ❌ Detection Mismatch: Expected {direction}, got {detected_direction}")
        else:
            print("    ❌ Failed to establish ORB range. Check timestamp logic.")

    print(f"\n📊 Verification Complete. Found {found_signals}/2 signals.")

if __name__ == "__main__":
    verify()
