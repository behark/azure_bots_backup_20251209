#!/bin/bash

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load environment variables from parent .env
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Change to bot directory and run in background
cd "$SCRIPT_DIR/orb_bot"
nohup ../venv/bin/python3 orb_bot.py --loop >> logs/orb_bot.log 2>&1 &

echo "✅ ORB Bot started! PID: $!"
echo "📋 Monitor: tail -f orb_bot/logs/orb_bot.log"
echo "🛑 Stop: pkill -f 'orb_bot.py'"
