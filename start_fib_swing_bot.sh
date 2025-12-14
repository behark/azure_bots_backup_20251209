#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
exec python fib_swing_bot/fib_swing_bot.py --loop --interval 300
