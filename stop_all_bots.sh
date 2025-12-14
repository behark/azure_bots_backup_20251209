#!/bin/bash

PID_FILE="active_bots.pid"

echo "🛑 STOPPING HIVE MIND..."

if [ -f "$PID_FILE" ]; then
    while read pid; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "  ☠️  Killed PID $pid"
        else
            echo "  ⚠️  PID $pid not found (already dead?)"
        fi
    done < "$PID_FILE"

    rm "$PID_FILE"
    echo "✅ Clean shutdown complete."
else
    echo "⚠️  No PID file found. Fallback to nuclear option?"
    echo "   Run: pkill -f '_bot.py'"
fi
