#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/consensus_bot"

if [ -d "../venv" ]; then
  source ../venv/bin/activate
  exec python consensus_bot.py --loop --interval 30 --window 30
else
  echo "Virtual environment ../venv not found" >&2
  exit 1
fi
