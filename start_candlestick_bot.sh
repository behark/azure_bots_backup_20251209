#!/bin/bash
cd "$(dirname "$0")/candlestick_bot"
../venv/bin/python3 candlestick_bot.py --loop
