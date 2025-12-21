#!/bin/bash
#
# Volume Bot Launcher
# Starts the Volume VN Bot with optimal settings
#

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}    Volume Bot v2.1 - Enhanced Edition${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""

# Set Telegram credentials
export TELEGRAM_BOT_TOKEN="8512614186:AAH-IXvUj1VHmqtr4sCtfn3h3dcRuhIF3Qc"
export TELEGRAM_CHAT_ID="1507876704"

echo -e "${GREEN}✅ Telegram credentials configured${NC}"
echo -e "   Bot Token: ${TELEGRAM_BOT_TOKEN:0:20}..."
echo -e "   Chat ID: $TELEGRAM_CHAT_ID"
echo ""

# Check for exchange API keys in .env file
if [ -f ".env" ]; then
    echo -e "${GREEN}✅ .env file found${NC}"

    # Check if exchange keys are configured
    if grep -q "BINANCEUSDM_API_KEY" .env && grep -q "BINANCEUSDM_SECRET" .env; then
        # Check if they're not empty
        API_KEY=$(grep "BINANCEUSDM_API_KEY" .env | cut -d'=' -f2 | tr -d ' "')
        if [ -z "$API_KEY" ] || [ "$API_KEY" == "your_api_key_here" ]; then
            echo -e "${RED}❌ ERROR: Exchange API keys not configured in .env${NC}"
            echo ""
            echo -e "${YELLOW}⚠️  The volume bot REQUIRES exchange API keys to:${NC}"
            echo "   • Fetch market data (OHLCV candles)"
            echo "   • Get volume information"
            echo "   • Check current prices for TP/SL monitoring"
            echo ""
            echo -e "${YELLOW}Please add your Binance API keys to .env file:${NC}"
            echo "   BINANCEUSDM_API_KEY=your_actual_api_key"
            echo "   BINANCEUSDM_SECRET=your_actual_secret_key"
            echo ""
            echo -e "${BLUE}💡 Get API keys from: https://www.binance.com/en/my/settings/api-management${NC}"
            echo ""
            exit 1
        else
            echo -e "${GREEN}✅ Exchange API keys configured${NC}"
        fi
    else
        echo -e "${RED}❌ ERROR: Exchange API keys missing from .env${NC}"
        echo ""
        echo -e "${YELLOW}Please add these lines to your .env file:${NC}"
        echo "   BINANCEUSDM_API_KEY=your_api_key"
        echo "   BINANCEUSDM_SECRET=your_secret"
        echo ""
        exit 1
    fi
else
    echo -e "${YELLOW}⚠️  No .env file found. Creating one...${NC}"
    cat > .env << 'EOL'
# Telegram Configuration (Already configured)
TELEGRAM_BOT_TOKEN=8512614186:AAH-IXvUj1VHmqtr4sCtfn3h3dcRuhIF3Qc
TELEGRAM_CHAT_ID=1507876704

# Exchange API Keys (REQUIRED - Please add your keys)
BINANCEUSDM_API_KEY=your_api_key_here
BINANCEUSDM_SECRET=your_secret_here

# Optional: Other exchanges
# BYBIT_API_KEY=your_bybit_key
# BYBIT_SECRET=your_bybit_secret
EOL
    echo -e "${GREEN}✅ Created .env file with Telegram credentials${NC}"
    echo ""
    echo -e "${RED}❌ Please edit .env and add your Binance API keys${NC}"
    echo -e "${BLUE}💡 Get API keys from: https://www.binance.com/en/my/settings/api-management${NC}"
    echo ""
    exit 1
fi

# Check if config.json exists
if [ ! -f "config.json" ]; then
    echo -e "${YELLOW}⚠️  config.json not found. Using default settings.${NC}"
    CONFIG_FLAG=""
else
    echo -e "${GREEN}✅ Using optimized config.json${NC}"
    CONFIG_FLAG="--config config.json"
fi

echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ ERROR: Python 3 is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python 3 found: $(python3 --version)${NC}"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Parse command line arguments
RUN_MODE="normal"
EXTRA_ARGS=""

for arg in "$@"; do
    case $arg in
        --once)
            RUN_MODE="test"
            EXTRA_ARGS="$EXTRA_ARGS --once"
            ;;
        --debug)
            EXTRA_ARGS="$EXTRA_ARGS --log-level DEBUG --detailed-logging"
            ;;
        --track)
            RUN_MODE="track"
            EXTRA_ARGS="$EXTRA_ARGS --track"
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $arg"
            ;;
    esac
done

# Show run mode
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
case $RUN_MODE in
    "test")
        echo -e "${YELLOW}🧪 TEST MODE: Running one cycle only${NC}"
        ;;
    "track")
        echo -e "${YELLOW}📊 TRACK MODE: Checking open signals only${NC}"
        ;;
    *)
        echo -e "${GREEN}🚀 PRODUCTION MODE: Running continuously${NC}"
        echo -e "${YELLOW}   Press Ctrl+C to stop${NC}"
        ;;
esac
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""

# Start the bot
echo -e "${GREEN}Starting Volume Bot...${NC}"
echo ""

python3 volume_vn_bot.py $CONFIG_FLAG $EXTRA_ARGS

# Check exit code
EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Bot stopped gracefully${NC}"
else
    echo -e "${RED}❌ Bot exited with error code: $EXIT_CODE${NC}"
    echo -e "${YELLOW}💡 Check logs/volume_vn_errors.log for details${NC}"
fi

exit $EXIT_CODE
