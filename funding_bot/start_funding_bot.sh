#!/bin/bash
#
# Funding Bot Launcher
# Starts the Funding Rate Bot with optimal settings
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
echo -e "${BLUE}    Funding Bot v2.1 - Unified Edition${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""

# Set Telegram credentials
export TELEGRAM_BOT_TOKEN="8202436291:AAGPmO6Dh2FllrmwXvndKuIOhDARMTc33fg"
export TELEGRAM_CHAT_ID="1507876704"

echo -e "${GREEN}✅ Telegram credentials configured${NC}"
echo -e "   Bot Token: ${TELEGRAM_BOT_TOKEN:0:20}..."
echo -e "   Chat ID: $TELEGRAM_CHAT_ID"
echo ""

# Check for .env in parent directory
if [ -f "../.env" ]; then
    echo -e "${GREEN}✅ .env file found in parent directory${NC}"
    export $(grep -v '^#' ../.env | xargs)
else
    echo -e "${YELLOW}⚠️  No .env file found in parent directory.${NC}"
fi

# Check if funding_config.json exists
if [ ! -f "funding_config.json" ]; then
    echo -e "${YELLOW}⚠️  funding_config.json not found. Using defaults.${NC}"
    CONFIG_FLAG=""
else
    echo -e "${GREEN}✅ Using optimized funding_config.json${NC}"
    CONFIG_FLAG="--config funding_config.json"
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
            EXTRA_ARGS="$EXTRA_ARGS --debug"
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
    *)
        echo -e "${GREEN}🚀 PRODUCTION MODE: Running continuously${NC}"
        echo -e "${YELLOW}   Press Ctrl+C to stop${NC}"
        ;;
esac
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""

# Start the bot
echo -e "${GREEN}Starting Funding Bot...${NC}"
echo ""

# Use venv if available
if [ -d "../venv" ]; then
    ../venv/bin/python3 funding_bot.py $CONFIG_FLAG $EXTRA_ARGS
else
    python3 funding_bot.py $CONFIG_FLAG $EXTRA_ARGS
fi

# Check exit code
EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Bot stopped gracefully${NC}"
else
    echo -e "${RED}❌ Bot exited with error code: $EXIT_CODE${NC}"
    echo -e "${YELLOW}💡 Check logs/funding_errors.log for details${NC}"
fi

exit $EXIT_CODE
