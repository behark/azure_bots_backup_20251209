#!/usr/bin/env python3
"""Validate configuration files before starting bots."""

import json
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    import os
except ImportError:
    print("❌ python-dotenv not installed. Install with: pip install python-dotenv")
    sys.exit(1)


def validate_env():
    """Validate .env file."""
    load_dotenv()
    errors = []
    warnings = []
    
    # Required
    if not os.getenv('TELEGRAM_BOT_TOKEN'):
        errors.append("TELEGRAM_BOT_TOKEN not set")
    if not os.getenv('TELEGRAM_CHAT_ID'):
        errors.append("TELEGRAM_CHAT_ID not set")
    
    # Optional but recommended
    if not os.getenv('MEXC_API_KEY'):
        warnings.append("MEXC_API_KEY not set (required for orb_bot)")
    if not os.getenv('MEXC_API_SECRET'):
        warnings.append("MEXC_API_SECRET not set (required for orb_bot)")
    
    return errors, warnings


def validate_global_config():
    """Validate global_config.json."""
    config_file = Path('global_config.json')
    errors = []
    
    if not config_file.exists():
        errors.append("global_config.json not found")
        return errors
    
    try:
        with open(config_file) as f:
            config = json.load(f)
        
        # Validate structure
        if 'global_risk' not in config:
            errors.append("global_config.json missing 'global_risk' section")
        else:
            # Validate global_risk structure
            risk = config['global_risk']
            required_risk_keys = ['max_drawdown_percent', 'emergency_stop_percent', 'max_stop_loss_percent']
            for key in required_risk_keys:
                if key not in risk:
                    errors.append(f"global_config.json missing 'global_risk.{key}'")
        
        if 'bots' not in config:
            errors.append("global_config.json missing 'bots' section")
            
    except json.JSONDecodeError as e:
        errors.append(f"global_config.json is invalid JSON: {e}")
    except Exception as e:
        errors.append(f"Error reading global_config.json: {e}")
    
    return errors


def validate_bot_configs():
    """Validate bot-specific configuration files."""
    errors = []
    warnings = []
    
    bot_dirs = [
        'diy_bot', 'fib_swing_bot', 'funding_bot', 'harmonic_bot',
        'harmonic_pattern_bot', 'liquidation_bot', 'most_bot',
        'mtf_bot', 'orb_bot', 'psar_bot', 'strat_bot',
        'volume_bot', 'volume_profile_bot'
    ]
    
    for bot_dir in bot_dirs:
        bot_path = Path(bot_dir)
        if not bot_path.exists():
            continue
        
        # Check for watchlist
        watchlist_files = list(bot_path.glob('*watchlist.json'))
        if not watchlist_files:
            warnings.append(f"{bot_dir}: No watchlist.json found")
        
        # Check for config if bot typically has one
        config_files = list(bot_path.glob('*config.json'))
        if bot_dir in ['orb_bot', 'funding_bot', 'harmonic_bot'] and not config_files:
            warnings.append(f"{bot_dir}: No config.json found (may be optional)")
    
    return errors, warnings


def main():
    print("🔍 Validating configuration...")
    print("")
    
    env_errors, env_warnings = validate_env()
    config_errors = validate_global_config()
    bot_errors, bot_warnings = validate_bot_configs()
    
    all_errors = env_errors + config_errors + bot_errors
    all_warnings = env_warnings + bot_warnings
    
    if all_warnings:
        print("⚠️  Warnings:")
        for warning in all_warnings:
            print(f"   - {warning}")
        print("")
    
    if all_errors:
        print("❌ Errors found:")
        for error in all_errors:
            print(f"   - {error}")
        print("")
        print("Please fix the errors above before starting bots.")
        sys.exit(1)
    
    print("✅ Configuration is valid!")
    if all_warnings:
        print("⚠️  Some warnings were found (see above), but configuration is usable.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
