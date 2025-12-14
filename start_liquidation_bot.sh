#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/liquidation_bot"

if [ -d "../venv" ]; then
  source ../venv/bin/activate
  exec python liquidation_bot.py --loop --interval 300
else
  echo "Virtual environment ../venv not found" >&2
  exit 1
fi
