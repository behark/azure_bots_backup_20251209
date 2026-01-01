#!/bin/bash
# Cleanup script to remove recreatable files, logs, and signal history

set -e

cd "$(dirname "$0")"

echo "Starting cleanup..."

# Remove all .log files (excluding venv and .git)
echo "Removing log files..."
/usr/bin/find . -name "*.log" -not -path "./venv/*" -not -path "./.git/*" -delete 2>/dev/null || true

# Remove all logs directories (contain logs and signal history)
echo "Removing logs directories..."
for dir in */logs; do
    if [ -d "$dir" ]; then
        echo "  Removing $dir"
        rm -rf "$dir"
    fi
done

# Remove signal history JSON files
echo "Removing signal history files..."
/usr/bin/find . -name "*signals.json" -not -path "./venv/*" -not -path "./.git/*" -not -path "./.mypy_cache/*" -delete 2>/dev/null || true

# Remove recreatable directories
echo "Removing recreatable directories..."

# Remove venv (virtual environment - can be recreated with pip install -r requirements.txt)
if [ -d "venv" ]; then
    echo "  Removing venv/ (423M)"
    rm -rf venv
fi

# Remove .mypy_cache (type checking cache - can be recreated)
if [ -d ".mypy_cache" ]; then
    echo "  Removing .mypy_cache/ (48M)"
    rm -rf .mypy_cache
fi

# Remove __pycache__ directories (Python bytecode cache)
echo "Removing __pycache__ directories..."
/usr/bin/find . -type d -name "__pycache__" -not -path "./venv/*" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true

# Remove .pyc files
echo "Removing .pyc files..."
/usr/bin/find . -name "*.pyc" -not -path "./venv/*" -not -path "./.git/*" -delete 2>/dev/null || true

# Remove .pyo files
echo "Removing .pyo files..."
/usr/bin/find . -name "*.pyo" -not -path "./venv/*" -not -path "./.git/*" -delete 2>/dev/null || true

echo ""
echo "Cleanup complete!"
echo "Checking remaining size..."
du -sh . 2>/dev/null || true
