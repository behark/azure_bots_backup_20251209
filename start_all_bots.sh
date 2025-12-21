#!/usr/bin/env bash
set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$BASE_DIR/venv"

echo -e "${GREEN}=== Starting All Trading Bots ===${NC}"
echo "Base directory: $BASE_DIR"
echo ""

# Check if virtual environment exists
if [[ ! -d "$VENV_DIR" ]]; then
    echo -e "${RED}Error: Virtual environment not found at $VENV_DIR${NC}" >&2
    exit 1
fi

# Check if screen is installed
if ! command -v screen >/dev/null 2>&1; then
    echo -e "${RED}Error: screen command not found. Please install it:${NC}" >&2
    echo "  sudo apt-get install screen   # Debian/Ubuntu" >&2
    echo "  sudo yum install screen        # CentOS/RHEL" >&2
    exit 1
fi

# Define bots with their configurations
# Format: "bot_name:script_path:args"
# Standardized: --interval 300 (5min cycle), --cooldown 30 (30min cooldown)
declare -a BOTS=(
    "diy_bot:diy_bot/diy_bot.py:--interval 300"
    "fib_swing_bot:fib_swing_bot/fib_swing_bot.py:--interval 300"
    "funding_bot:funding_bot/funding_bot.py:--interval 300"
    "harmonic_bot:harmonic_bot/harmonic_bot.py:--interval 300"
    "liquidation_bot:liquidation_bot/liquidation_bot.py:--interval 300 --cooldown 30"
    "most_bot:most_bot/most_bot.py:--interval 300 --cooldown 30"
    "mtf_bot:mtf_bot/mtf_bot.py:--interval 300 --cooldown 30"
    "orb_bot:orb_bot/orb_bot.py:--interval 300"
    "psar_bot:psar_bot/psar_bot.py:--interval 300 --cooldown 30"
    "strat_bot:strat_bot/strat_bot.py:--interval 300 --cooldown 30"
    "volume_bot:volume_bot/volume_vn_bot.py:--interval 300"
    "volume_profile_bot:volume_profile_bot/volume_profile_bot.py:--interval 300"
)

started=0
already_running=0
failed=0

for bot_config in "${BOTS[@]}"; do
    IFS=':' read -r bot_name script_path args <<< "$bot_config"

    full_script_path="$BASE_DIR/$script_path"
    bot_dir="$(dirname "$full_script_path")"
    script_basename="$(basename "$full_script_path")"

    # Check if bot script exists
    if [[ ! -f "$full_script_path" ]]; then
        echo -e "${RED}✗ $bot_name: Script not found at $script_path${NC}"
        ((failed++))
        continue
    fi

    # Check if bot is already running
    if pgrep -f "[p]ython.*${script_basename}" >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠ $bot_name: Already running${NC}"
        ((already_running++))
        continue
    fi

    # Start bot in screen session
    echo -e "${GREEN}▶ Starting $bot_name...${NC}"

    # Build the command to run
    if [[ -n "$args" ]]; then
        cmd="cd '$bot_dir' && source '$VENV_DIR/bin/activate' && python '$script_basename' $args"
    else
        cmd="cd '$bot_dir' && source '$VENV_DIR/bin/activate' && python '$script_basename'"
    fi

    # Start in screen session with logging
    screen -dmS "$bot_name" bash -c "$cmd >> '$bot_dir/${bot_name}.log' 2>&1"

    # Give it a moment to start
    sleep 0.5

    # Verify it started
    if pgrep -f "[p]ython.*${script_basename}" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ $bot_name started successfully${NC}"
        ((started++))
    else
        echo -e "${RED}✗ $bot_name failed to start (check logs at $bot_dir/${bot_name}.log)${NC}"
        ((failed++))
    fi
done

echo ""
echo -e "${GREEN}=== Summary ===${NC}"
echo "Started: $started"
echo "Already running: $already_running"
echo "Failed: $failed"
echo ""

if [[ $started -gt 0 ]]; then
    echo -e "${GREEN}To view running bots, use:${NC}"
    echo "  screen -ls"
    echo ""
    echo -e "${GREEN}To attach to a bot session:${NC}"
    echo "  screen -r <bot_name>"
    echo ""
    echo -e "${GREEN}To detach from a session:${NC}"
    echo "  Press Ctrl+A then D"
fi

if [[ $failed -gt 0 ]]; then
    echo -e "${YELLOW}Some bots failed to start. Check the logs in their respective directories.${NC}"
    exit 1
fi
