#!/usr/bin/env bash
set -uo pipefail # Removed -e so it doesn't crash on "not found" errors

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}=== Force Stopping Bots ===${NC}"

declare -a BOTS=(
    "candlestick_bot"
    "consensus_bot"
    "diy_bot"
    "fib_reversal_bot"
    "fib_swing_bot"
    "funding_bot"
    "harmonic_bot"
    "liquidation_bot"
    "most_bot"
    "mtf_bot"
    "orb_bot"
    "psar_bot"
    "strat_bot"
    "volume_vn_bot"
    "volume_profile_bot"
)

for bot_name in "${BOTS[@]}"; do
    echo -e "Processing: ${bot_name}..."

    # 1. Try to kill the screen (Loose match)
    if screen -list | grep -q "$bot_name"; then
        # Try to quit properly first
        screen -S "$bot_name" -X quit 2>/dev/null || true
        # If it's stubborn, kill the screen session
        screen -S "$bot_name" -p 0 -X kill 2>/dev/null || true
        echo -e "${YELLOW}   > Screen session closed.${NC}"
    fi

    # 2. NUCLEAR OPTION: Find the python process and kill it directly
    # This works even if the screen session wasn't found
    if pgrep -f "python.*${bot_name}.py" > /dev/null; then
        pkill -f "python.*${bot_name}.py"
        echo -e "${GREEN}   > Killed Python process.${NC}"
    else
        echo -e "${RED}   > No running process found.${NC}"
    fi
done

echo ""
echo -e "${YELLOW}Double checking...${NC}"
sleep 1

# Final check
remaining=$(pgrep -f "python.*_bot.py" | wc -l || true)
remaining=$(echo "$remaining" | xargs)

if [[ "$remaining" -eq 0 ]]; then
    echo -e "${GREEN}✅ All bots are definitely dead.${NC}"
else
    echo -e "${RED}⚠️  Warning: $remaining processes still alive:${NC}"
    ps aux | grep -E "python.*_bot\.py" | grep -v grep
fi
