#!/bin/bash

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║          📊 BOT STATUS CHECK 📊                                     ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

BOT_COUNT=$(ps aux | grep "_bot.py" | grep -v grep | wc -l)

echo "🤖 Total Running Bots: $BOT_COUNT / 12"
echo ""

if [ "$BOT_COUNT" -eq 0 ]; then
    echo "❌ NO BOTS ARE RUNNING!"
    echo ""
    echo "💡 To start all bots, run:"
    echo "   ./start_all_bots.sh"
    echo ""
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🤖 RUNNING BOTS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check each bot individually
check_bot() {
    local bot_name=$1
    local process_name=$2
    
    if ps aux | grep "$process_name" | grep -v grep > /dev/null; then
        local pid=$(ps aux | grep "$process_name" | grep -v grep | awk '{print $2}' | head -1)
        local runtime=$(ps -p $pid -o etime= | xargs)
        echo "  ✅ $bot_name (PID: $pid, Runtime: $runtime)"
    else
        echo "  ❌ $bot_name - NOT RUNNING"
    fi
}

echo ""
echo "🎨 NEW BOTS:"
check_bot "Harmonic Bot      " "harmonic_bot.py"
check_bot "Candlestick Bot   " "candlestick_bot.py"
check_bot "MTF Bot           " "mtf_bot.py"
check_bot "PSAR Bot          " "psar_bot.py"
check_bot "DIY Bot           " "diy_bot.py"
check_bot "MOST Bot          " "most_bot.py"
check_bot "STRAT Bot         " "strat_bot.py"
check_bot "Fib Reversal Bot  " "fib_reversal_bot.py"

echo ""
echo "📊 OLD BOTS:"
check_bot "Funding Bot       " "funding_bot.py"
check_bot "Liquidation Bot   " "liquidation_bot.py"
check_bot "Volume Bot        " "volume_vn_bot.py"

echo ""
echo "🏆 MASTER:"
check_bot "Consensus Bot     " "consensus_bot.py"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$BOT_COUNT" -eq 12 ]; then
    echo ""
    echo "🎉🔥🏆 PERFECT! ALL 12 BOTS ARE RUNNING! 🏆🔥🎉"
    echo ""
elif [ "$BOT_COUNT" -gt 0 ]; then
    echo ""
    echo "⚠️  PARTIAL DEPLOYMENT: $BOT_COUNT / 12 bots running"
    echo ""
    echo "💡 To start missing bots, run:"
    echo "   ./start_all_bots.sh"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "💡 USEFUL COMMANDS:"
echo ""
echo "  Watch Consensus Bot (real-time):"
echo "    tail -f consensus_bot/logs/consensus_bot.log"
echo ""
echo "  View recent consensus alerts:"
echo "    tail -50 consensus_bot/logs/consensus_bot.log | grep 'Consensus alert'"
echo ""
echo "  Stop all bots:"
echo "    ./stop_all_bots.sh"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
