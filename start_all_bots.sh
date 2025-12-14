#!/bin/bash

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Define Log Directory and PID File
LOG_DIR="logs"
PID_FILE="active_bots.pid"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Clear previous PID file
> "$PID_FILE"

# Ensure git commands inside bots resolve even from subdirectories
if [ -d "$DIR/.git" ]; then
    export GIT_DIR="$DIR/.git"
    export GIT_WORK_TREE="$DIR"
fi

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║          🚀 STARTING THE HIVE MIND (Corrected Paths) 🚀             ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📂 Working Directory: $DIR"

# --- 0. LOAD ENVIRONMENT VARIABLES ---
if [ -f "$DIR/.env" ]; then
    echo "🔑 Loading environment variables from .env..."
    export $(grep -v '^#' "$DIR/.env" | xargs)
fi

# --- 1. VIRTUAL ENVIRONMENT CHECK ---
if [ -d "venv" ]; then
    echo "🔌 Activating virtual environment (venv)..."
    source venv/bin/activate
elif [ -d "../venv" ]; then
    echo "🔌 Activating virtual environment (../venv)..."
    source ../venv/bin/activate
else
    echo "⚠️  WARNING: No 'venv' folder found. Running with system Python."
    echo "    (If imports fail, ensure you are in the correct directory)"
fi

echo "📄 Logs stored in:    $DIR/$LOG_DIR/"
echo ""

# Function to start a bot
# Usage: start_bot "path/to/script.py" "LogName" "Interval" "Cooldown" [extra args...]
start_bot() {
    local script_path="$1"
    local name="$2"
    local interval="${3:-300}" # Default 5 minutes
    local cooldown="${4:-15}"  # Default 15 minutes
    shift 4
    local extra_args=("$@")
    local logfile="$LOG_DIR/${name}.log"

    if [ ! -f "$script_path" ]; then
        echo "  ❌ ERROR: Script not found: $script_path"
        return
    fi

    echo "  🚀 Starting $name..."

    if [ ${#extra_args[@]} -gt 0 ]; then
        nohup python3 "$script_path" "${extra_args[@]}" >> "$logfile" 2>&1 &
    else
        nohup python3 "$script_path" --interval "$interval" --cooldown "$cooldown" --loop >> "$logfile" 2>&1 &
    fi

    pid=$!
    echo $pid >> "$PID_FILE"
}

# --- NEW BOTS ---
echo "🎨 Starting Strategy Bots..."
start_bot "harmonic_bot/harmonic_bot.py"         "harmonic"     300 15
start_bot "candlestick_bot/candlestick_bot.py"   "candlestick"  300 15
start_bot "mtf_bot/mtf_bot.py"                   "mtf"          300 15
start_bot "psar_bot/psar_bot.py"                 "psar"         300 15
start_bot "diy_bot/diy_bot.py"                   "diy"          300 15
start_bot "most_bot/most_bot.py"                 "most"         300 15
start_bot "strat_bot/strat_bot.py"               "strat"        300 15
start_bot "fib_reversal_bot/fib_reversal_bot.py" "fib_reversal" 300 15
start_bot "fib_swing_bot/fib_swing_bot.py"       "fib_swing"    300 15 --interval 300 --loop
start_bot "orb_bot/orb_bot.py"                   "orb"          60 15 --loop

# --- OLD BOTS ---
echo ""
echo "📊 Starting Data Bots..."
start_bot "funding_bot/funding_bot.py"           "funding"      300 15
start_bot "liquidation_bot/liquidation_bot.py"   "liquidation"  300 15 --interval 300 --loop
start_bot "volume_bot/volume_vn_bot.py"          "volume"       60 15 --loop
start_bot "volume_profile_bot/volume_profile_bot.py" "volume_profile" 60 15 --interval 60 --loop

# --- CONSENSUS ---
echo ""
echo "🏆 Starting MASTER MANAGER..."
if [ -f "consensus_bot/consensus_bot.py" ]; then
    nohup python3 consensus_bot/consensus_bot.py --interval 30 --window 30 --min-rr 1.2 --loop >> "$LOG_DIR/consensus.log" 2>&1 &
    pid=$!
    echo $pid >> "$PID_FILE"
    echo "  ✅ Consensus Bot started (PID: $pid)"
else
    echo "  ❌ ERROR: consensus_bot/consensus_bot.py not found!"
fi

echo ""
echo "⏳ Initializing..."
sleep 2

# Validation
RUNNING_COUNT=$(wc -l < "$PID_FILE")
REAL_COUNT=$(ps aux | grep "python3.*_bot.py" | grep -v grep | wc -l)

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🤖 STATUS CHECK:"
echo "   PIDs Recorded: $RUNNING_COUNT / 15"
echo "   Actual Process Count: $REAL_COUNT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "💡 COMMANDS:"
echo "   Monitor Consensus:  tail -f logs/consensus.log"
echo "   Stop All:           ./stop_all_bots.sh"
echo ""
