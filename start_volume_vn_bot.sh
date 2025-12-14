#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/volume_bot"

if [ -d "../venv" ]; then
  source ../venv/bin/activate
  exec python volume_vn_bot.py --loop
else
  echo "Virtual environment ../venv not found" >&2
  exit 1
fi
