#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/funding_bot"

if [ -d "../venv" ]; then
  source ../venv/bin/activate
  exec python funding_bot.py --loop --interval 300 --cooldown 45
else
  echo "Virtual environment ../venv not found" >&2
  exit 1
fi
