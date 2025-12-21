#!/bin/bash
#
# Harmonic Bot Launcher
# Starts the Harmonic Pattern Bot with optimal settings
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
echo -e "${BLUE}    Harmonic Bot v2.3 - Production Edition${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""

# Load environment variables from .env files
# Check local .env first, then parent directory .env
if [ -f ".env" ]; then
    echo -e "${GREEN}✅ Loading .env from harmonic_bot directory${NC}"
    set -a
    source .env
    set +a
elif [ -f "../.env" ]; then
    echo -e "${GREEN}✅ Loading .env from parent directory${NC}"
    set -a
    source ../.env
    set +a
else
    echo -e "${YELLOW}⚠️  No .env file found${NC}"
fi

# Verify Telegram credentials are set
if [ -n "$TELEGRAM_BOT_TOKEN_HARMONIC" ] || [ -n "$TELEGRAM_BOT_TOKEN" ]; then
    echo -e "${GREEN}✅ Telegram credentials loaded from environment${NC}"
    if [ -n "$TELEGRAM_BOT_TOKEN_HARMONIC" ]; then
        echo -e "   Bot Token (HARMONIC): ${TELEGRAM_BOT_TOKEN_HARMONIC:0:20}..."
    elif [ -n "$TELEGRAM_BOT_TOKEN" ]; then
        echo -e "   Bot Token (GENERIC): ${TELEGRAM_BOT_TOKEN:0:20}..."
    fi
    [ -n "$TELEGRAM_CHAT_ID" ] && echo -e "   Chat ID: $TELEGRAM_CHAT_ID"
else
    echo -e "${RED}❌ WARNING: No Telegram credentials found in environment${NC}"
    echo -e "${YELLOW}   The bot will not send notifications!${NC}"
    echo -e "${YELLOW}   Please add TELEGRAM_BOT_TOKEN_HARMONIC and TELEGRAM_CHAT_ID to .env${NC}"
fi
echo ""

# Check if harmonic_config.json exists
if [ ! -f "harmonic_config.json" ]; then
    echo -e "${YELLOW}⚠️  harmonic_config.json not found. Using default settings.${NC}"
    CONFIG_FLAG=""
else
    echo -e "${GREEN}✅ Using optimized harmonic_config.json${NC}"
    CONFIG_FLAG="--config harmonic_config.json"
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
echo -e "${GREEN}Starting Harmonic Bot...${NC}"
echo ""

# Use venv if available
if [ -d "../venv" ]; then
    ../venv/bin/python3 harmonic_bot.py $CONFIG_FLAG $EXTRA_ARGS
else
    python3 harmonic_bot.py $CONFIG_FLAG $EXTRA_ARGS
fi

# Check exit code
EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Bot stopped gracefully${NC}"
else
    echo -e "${RED}❌ Bot exited with error code: $EXIT_CODE${NC}"
    echo -e "${YELLOW}💡 Check logs/harmonic_errors.log for details${NC}"
fi

exit $EXIT_CODE
