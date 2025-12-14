#!/bin/bash
cd "$(dirname "$0")/mtf_bot"
../venv/bin/python3 mtf_bot.py --loop
