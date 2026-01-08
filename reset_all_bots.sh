#!/usr/bin/env bash
# Reset all bot history, logs, and open trades

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

echo -e "${YELLOW}⚠️  WARNING: This will reset ALL bot history, logs, and open trades!${NC}"
echo ""
echo "This will delete:"
echo "  - All state files (*_state.json)"
echo "  - All signal log files (*_signals.json)"
echo "  - All stats files (logs/*_stats.json)"
echo "  - All open signals"
echo "  - All trade history"
echo ""
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo -e "${YELLOW}Reset cancelled.${NC}"
    exit 0
fi

echo ""
echo -e "${GREEN}🔄 Resetting all bots...${NC}"
echo ""

# Counter for files removed
removed_count=0

# Find and remove all state files
echo "📦 Removing state files..."
while IFS= read -r file; do
    if [ -f "$file" ]; then
        rm -f "$file"
        echo "  ✓ Removed: $file"
        ((removed_count++))
    fi
done < <(find "$BASE_DIR" -name "*_state.json" -type f 2>/dev/null)

# Find and remove all signal log files
echo ""
echo "📨 Removing signal log files..."
while IFS= read -r file; do
    if [ -f "$file" ]; then
        rm -f "$file"
        echo "  ✓ Removed: $file"
        ((removed_count++))
    fi
done < <(find "$BASE_DIR" -name "*_signals.json" -type f 2>/dev/null)

# Find and remove all stats files
echo ""
echo "📊 Removing stats files..."
while IFS= read -r file; do
    if [ -f "$file" ]; then
        rm -f "$file"
        echo "  ✓ Removed: $file"
        ((removed_count++))
    fi
done < <(find "$BASE_DIR" -name "*_stats.json" -type f 2>/dev/null)

# Remove core signal manager state files
echo ""
echo "🔧 Removing core signal manager files..."
for bot_dir in "$BASE_DIR"/*_bot; do
    if [ -d "$bot_dir" ]; then
        bot_name=$(basename "$bot_dir")
        signal_file="$bot_dir/${bot_name}_signals.json"
        if [ -f "$signal_file" ]; then
            rm -f "$signal_file"
            echo "  ✓ Removed: $signal_file"
            ((removed_count++))
        fi
    fi
done

# Also check core directory
if [ -d "$BASE_DIR/core" ]; then
    for signal_file in "$BASE_DIR/core"/*_signals.json; do
        if [ -f "$signal_file" ]; then
            rm -f "$signal_file"
            echo "  ✓ Removed: $signal_file"
            ((removed_count++))
        fi
    done
fi

# Remove portfolio manager state
echo ""
echo "💼 Removing portfolio manager state..."
portfolio_state="$BASE_DIR/portfolio_state.json"
orchestrator_state="$BASE_DIR/orchestrator_state.json"
pnl_state="$BASE_DIR/pnl_state.json"

for state_file in "$portfolio_state" "$orchestrator_state" "$pnl_state"; do
    if [ -f "$state_file" ]; then
        rm -f "$state_file"
        echo "  ✓ Removed: $state_file"
        ((removed_count++))
    fi
done

# Remove logs directory contents (but keep directory)
echo ""
echo "📝 Clearing log files..."
if [ -d "$BASE_DIR/logs" ]; then
    log_count=$(find "$BASE_DIR/logs" -name "*.log" -type f | wc -l)
    if [ "$log_count" -gt 0 ]; then
        find "$BASE_DIR/logs" -name "*.log" -type f -delete
        echo "  ✓ Removed $log_count log files"
        ((removed_count+=log_count))
    fi
fi

# Remove bot-specific log directories
for bot_dir in "$BASE_DIR"/*_bot; do
    if [ -d "$bot_dir/logs" ]; then
        log_count=$(find "$bot_dir/logs" -name "*.log" -type f 2>/dev/null | wc -l)
        if [ "$log_count" -gt 0 ]; then
            find "$bot_dir/logs" -name "*.log" -type f -delete
            echo "  ✓ Removed $log_count log files from $bot_dir/logs"
            ((removed_count+=log_count))
        fi
    fi
done

echo ""
echo -e "${GREEN}✅ Reset complete!${NC}"
echo "   Total files removed: $removed_count"
echo ""
echo -e "${GREEN}All bots will start with fresh state on next run.${NC}"
echo ""
echo "Next steps:"
echo "  1. Start bots: ./start_all_bots.sh"
echo "  2. All new signals will show: 📈 History: 0% Win (0/0) | TP:0 SL:0"
