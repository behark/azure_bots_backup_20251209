#!/usr/bin/env python3
"""
Quick test script to verify Binance API connection
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("BINANCEUSDM_API_KEY")
SECRET = os.getenv("BINANCEUSDM_SECRET")

print("=" * 60)
print("BINANCE API CONNECTION TEST")
print("=" * 60)
print()

# Check if keys are loaded
print("✓ Checking environment variables...")
if not API_KEY or API_KEY == "your_binance_api_key_here":
    print("✗ ERROR: API Key not configured in .env")
    sys.exit(1)

if not SECRET or SECRET == "your_binance_secret_key_here":
    print("✗ ERROR: Secret Key not configured in .env")
    sys.exit(1)

print(f"  API Key: {API_KEY[:20]}...{API_KEY[-10:]}")
print(f"  Secret: {SECRET[:20]}...{SECRET[-10:]}")
print()

# Test CCXT connection
print("✓ Testing CCXT library...")
try:
    import ccxt
    print("  CCXT version:", ccxt.__version__)
except ImportError:
    print("✗ ERROR: CCXT not installed. Run: pip3 install ccxt")
    sys.exit(1)
print()

# Test connection to Binance
print("✓ Connecting to Binance USD-M Futures...")
try:
    exchange = ccxt.binanceusdm({
        'apiKey': API_KEY,
        'secret': SECRET,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
        }
    })
    print("  Exchange initialized:", exchange.id)
except Exception as e:
    print(f"✗ ERROR: Failed to initialize exchange: {e}")
    sys.exit(1)
print()

# Test fetching public market data (doesn't require authentication)
print("✓ Testing public API (no auth required)...")
try:
    ticker = exchange.fetch_ticker('BTC/USDT')
    print(f"  BTC/USDT Last Price: ${ticker['last']:,.2f}")
    print("  ✓ Public API works!")
except Exception as e:
    print(f"✗ ERROR: Failed to fetch ticker: {e}")
    print()
    print("This could mean:")
    print("  - Network connectivity issue")
    print("  - Binance API is down")
    print("  - Rate limiting")
    sys.exit(1)
print()

# Test fetching account data (requires valid authentication)
print("✓ Testing authenticated API (requires valid keys)...")
try:
    # Try to fetch account balance (requires authentication)
    balance = exchange.fetch_balance()
    print("  ✓ Authentication successful!")
    print(f"  Account has {len(balance['info']['assets'])} assets")

    # Show USDT balance if available
    if 'USDT' in balance['total']:
        usdt_balance = balance['total']['USDT']
        print(f"  USDT Balance: {usdt_balance:,.2f}")

except ccxt.AuthenticationError as e:
    print(f"✗ AUTHENTICATION ERROR: {e}")
    print()
    print("This usually means:")
    print("  - Invalid API Key or Secret")
    print("  - Keys don't match (key from one account, secret from another)")
    print("  - Wrong key type (Ed25519 vs HMAC_SHA256)")
    print()
    print("SOLUTION:")
    print("  1. Go to: https://www.binance.com/en/my/settings/api-management")
    print("  2. Delete the current API key")
    print("  3. Create NEW API key with 'HMAC_SHA256' type")
    print("  4. Copy BOTH the API Key and Secret Key")
    print("  5. Update .env file with both keys")
    sys.exit(1)

except ccxt.PermissionDenied as e:
    print(f"✗ PERMISSION ERROR: {e}")
    print()
    print("The API key doesn't have the required permissions.")
    print("Make sure 'Enable Reading' is checked for this API key.")
    sys.exit(1)

except Exception as e:
    print(f"✗ ERROR: {e}")
    print(f"   Error type: {type(e).__name__}")
    sys.exit(1)

print()
print("=" * 60)
print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
print("=" * 60)
print()
print("Your Binance API is working correctly!")
print("You can now run the volume bot:")
print("  ./start_volume_bot.sh")
print()
