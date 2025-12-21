#!/bin/bash

# ============================================================================
# DIY Multi-Indicator Bot Launcher
# ============================================================================
# This script starts the DIY bot with proper validation and error handling
# ============================================================================

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Files and directories
BOT_SCRIPT="diy_bot.py"
CONFIG_FILE="diy_config.json"
WATCHLIST_FILE="diy_watchlist.json"
LOG_DIR="logs"
ENV_FILE="../.env"

# Parse command-line arguments
TEST_MODE=false
DEBUG_MODE=false
ONCE_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            shift
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --once)
            ONCE_MODE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --test      Run in test mode (single cycle)"
            echo "  --debug     Enable debug logging"
            echo "  --once      Run one cycle and exit (same as --test)"
            echo "  -h, --help  Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Test mode implies once mode
if [ "$TEST_MODE" = true ]; then
    ONCE_MODE=true
fi

# Print header
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   DIY Multi-Indicator Bot - v2.0 Production${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ ERROR: python3 is not installed or not in PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Python 3 found: $(python3 --version)"

# Verify bot script exists
if [ ! -f "$BOT_SCRIPT" ]; then
    echo -e "${RED}✗ ERROR: Bot script not found: $BOT_SCRIPT${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Bot script found: $BOT_SCRIPT"

# Verify configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}✗ ERROR: Config file not found: $CONFIG_FILE${NC}"
    echo -e "${YELLOW}  Please create diy_config.json with required settings${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Config file found: $CONFIG_FILE"

# Verify watchlist file exists
if [ ! -f "$WATCHLIST_FILE" ]; then
    echo -e "${RED}✗ ERROR: Watchlist file not found: $WATCHLIST_FILE${NC}"
    exit 1
fi

# Count symbols in watchlist
SYMBOL_COUNT=$(python3 -c "import json; data=json.load(open('$WATCHLIST_FILE')); print(len(data))" 2>/dev/null || echo "0")

if [ "$SYMBOL_COUNT" -eq 0 ]; then
    echo -e "${RED}✗ ERROR: Watchlist is empty or invalid${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Watchlist found: $SYMBOL_COUNT symbols"

# Check .env file
if [ -f "$ENV_FILE" ]; then
    echo -e "${GREEN}✓${NC} Environment file found: $ENV_FILE"
    source "$ENV_FILE"
else
    echo -e "${YELLOW}⚠${NC} Environment file not found: $ENV_FILE"
    echo -e "${YELLOW}  Checking for environment variables...${NC}"
fi

# Verify required environment variables
MISSING_VARS=0

if [ -z "$TELEGRAM_BOT_TOKEN_DIY" ] && [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo -e "${RED}✗ ERROR: TELEGRAM_BOT_TOKEN_DIY or TELEGRAM_BOT_TOKEN not set${NC}"
    MISSING_VARS=$((MISSING_VARS + 1))
else
    echo -e "${GREEN}✓${NC} Telegram bot token found"
fi

if [ -z "$TELEGRAM_CHAT_ID" ]; then
    echo -e "${RED}✗ ERROR: TELEGRAM_CHAT_ID not set${NC}"
    MISSING_VARS=$((MISSING_VARS + 1))
else
    echo -e "${GREEN}✓${NC} Telegram chat ID found"
fi

if [ $MISSING_VARS -gt 0 ]; then
    echo ""
    echo -e "${RED}Missing $MISSING_VARS required environment variable(s)${NC}"
    echo -e "${YELLOW}Please set them in $ENV_FILE or export them${NC}"
    exit 1
fi

# Create logs directory
mkdir -p "$LOG_DIR"
echo -e "${GREEN}✓${NC} Log directory ready: $LOG_DIR"

# Check Python dependencies
echo ""
echo -e "${BLUE}Checking Python dependencies...${NC}"

REQUIRED_PACKAGES=("ccxt" "numpy" "requests")
REQUIRED_IMPORTS=("ccxt" "numpy" "requests")
OPTIONAL_PACKAGES=("python-dotenv")
OPTIONAL_IMPORTS=("dotenv")
MISSING_PACKAGES=()

# Check required packages
for i in "${!REQUIRED_PACKAGES[@]}"; do
    package="${REQUIRED_PACKAGES[$i]}"
    import="${REQUIRED_IMPORTS[$i]}"
    if python3 -c "import ${import}" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $package installed"
    else
        echo -e "${RED}✗${NC} $package NOT installed"
        MISSING_PACKAGES+=("$package")
    fi
done

# Check optional packages (warnings only)
for i in "${!OPTIONAL_PACKAGES[@]}"; do
    package="${OPTIONAL_PACKAGES[$i]}"
    import="${OPTIONAL_IMPORTS[$i]}"
    if python3 -c "import ${import}" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $package installed"
    else
        echo -e "${YELLOW}⚠${NC} $package NOT installed (optional)"
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}Missing packages: ${MISSING_PACKAGES[*]}${NC}"
    echo -e "${YELLOW}Install with: pip3 install ${MISSING_PACKAGES[*]}${NC}"
    exit 1
fi

# Build command
CMD="python3 $BOT_SCRIPT"

if [ "$ONCE_MODE" = true ]; then
    CMD="$CMD --once"
fi

if [ "$DEBUG_MODE" = true ]; then
    CMD="$CMD --debug"
fi

# Display mode
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

if [ "$TEST_MODE" = true ]; then
    echo -e "${YELLOW}MODE: TEST (Single Cycle)${NC}"
elif [ "$DEBUG_MODE" = true ]; then
    echo -e "${YELLOW}MODE: DEBUG${NC}"
else
    echo -e "${GREEN}MODE: PRODUCTION${NC}"
fi

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Display configuration summary
echo -e "${BLUE}Configuration Summary:${NC}"
echo -e "  Symbols: ${SYMBOL_COUNT}"
echo -e "  Config: ${CONFIG_FILE}"
echo -e "  Logs: ${LOG_DIR}/"

# Get some config values
if command -v jq &> /dev/null; then
    CYCLE_INTERVAL=$(jq -r '.execution.cycle_interval_seconds // 60' "$CONFIG_FILE" 2>/dev/null || echo "60")
    MIN_CONFIDENCE=$(jq -r '.analysis.min_confidence_threshold // 60' "$CONFIG_FILE" 2>/dev/null || echo "60")
    echo -e "  Cycle interval: ${CYCLE_INTERVAL}s"
    echo -e "  Min confidence: ${MIN_CONFIDENCE}%"
fi

echo ""

# Final confirmation for production mode
if [ "$TEST_MODE" = false ] && [ "$ONCE_MODE" = false ]; then
    echo -e "${YELLOW}Starting bot in production mode...${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop the bot${NC}"
    echo ""
    sleep 2
fi

# Start the bot
echo -e "${GREEN}Starting DIY Bot...${NC}"
echo -e "${BLUE}Command: $CMD${NC}"
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Execute
exec $CMD
