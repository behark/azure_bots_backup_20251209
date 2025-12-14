#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/home/behar/Desktop/azure_bots_backup_20251209"

if [[ ! -d "$BASE_DIR" ]]; then
    echo "Base directory $BASE_DIR not found" >&2
    exit 1
fi

if ! command -v screen >/dev/null 2>&1; then
    echo "screen command not found; please install it to launch bots" >&2
    exit 1
fi

shopt -s nullglob
for bot_dir in "$BASE_DIR"/*_bot; do
    [[ -d "$bot_dir" ]] || continue
    bot_name="$(basename "$bot_dir")"

    bot_script=""
    default_script="$bot_dir/${bot_name}.py"
    if [[ -f "$default_script" ]]; then
        bot_script="$default_script"
    else
        # fallback: pick a single *_bot.py inside directory
        mapfile -t candidates < <(find "$bot_dir" -maxdepth 1 -type f -name '*_bot.py' -print)
        if [[ ${#candidates[@]} -eq 1 ]]; then
            bot_script="${candidates[0]}"
        fi
    fi

    if [[ -z "$bot_script" ]]; then
        echo "Skipping $bot_name: missing bot entry script" >&2
        continue
    fi

    script_basename="$(basename "$bot_script")"
    if pgrep -f "[p]ython .*${script_basename}" >/dev/null 2>&1; then
        echo "$bot_name already running"
        continue
    fi

    echo "Starting $bot_name in screen session $bot_name via $script_basename"
    screen -dmS "$bot_name" bash -c "cd \"$bot_dir\" && python3 \"$bot_script\" >> \"$bot_dir/${bot_name}.log\" 2>&1"
done
