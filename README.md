# Trading Bot System v2.0

A modular, production-ready trading bot framework for cryptocurrency futures trading on MEXC (and compatible exchanges via CCXT).

## 🏗️ Architecture Overview

This system uses a **unified architecture** where all trading bots inherit from a common `BaseBot` class, ensuring consistent behavior, risk management, and signal handling across all strategies.

```
byk/
├── core/                      # Core Framework
│   ├── base_bot.py           # Abstract base class for all bots
│   ├── signal_manager.py     # Unified signal tracking & monitoring
│   ├── risk_manager.py       # Portfolio-level risk protection
│   ├── market_analyzer.py    # Market regime detection
│   └── exchange_client.py    # Rate-limited exchange wrapper
├── strategies/                # Strategy Implementations
│   ├── orb_strategy.py       # Opening Range Breakout
│   ├── psar_strategy.py      # Parabolic SAR
│   └── liquidation_strategy.py
├── common/                    # Shared Utilities
│   ├── emergency_stop.py     # Emergency protection
│   ├── logging_config.py     # Logging setup
│   ├── resilience.py         # Retry & circuit breaker
│   └── types.py              # Type definitions
├── *_bot/                     # Legacy Bot Directories
├── tests/                     # Test Suite
├── global_config.json         # Centralized configuration
├── main.py                    # Unified entry point
└── requirements.txt
```

## ⚡ Quick Start

### 1. Install Dependencies

```bash
cd byk
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the `byk` directory:

```env
MEXC_API_KEY=your_api_key
MEXC_API_SECRET=your_api_secret
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 3. Run a Bot

```bash
# Run a specific strategy
python main.py --bot orb

# Run multiple bots
python main.py --bot orb psar

# Run all enabled bots
python main.py --all

# Run one cycle only (testing)
python main.py --bot orb --once

# Show status of all bots
python main.py --status

# Monitor live signals
python main.py --monitor
```

## 🎯 Available Strategies

### New Architecture (BaseBot)

| Strategy | Description | Key Features |
|----------|-------------|--------------|
| **ORB** | Opening Range Breakout | Session-based, ATR targets |
| **PSAR** | Parabolic SAR | Trailing stops, reversal detection |
| **Liquidation** | Liquidation Heatmap | Volume-based, squeeze detection |

### Legacy Bots (Original Architecture)

| Bot | Description |
|-----|-------------|
| **Funding Bot** | Funding rate arbitrage |
| **Harmonic Bot** | Harmonic pattern detection |
| **Fib Swing Bot** | Fibonacci retracement + swing |
| **Volume Bot** | Volume profile analysis |
| **STRAT Bot** | Rob Smith's 1-2-3 system |
| **MOST Bot** | Moving Stop Loss indicator |
| **MTF Bot** | Multi-timeframe confluence |
| **DIY Bot** | Multi-indicator confluence |

## 🛡️ Risk Management

The system includes comprehensive risk protection:

### Portfolio-Level Protection
- **Maximum Drawdown**: Automatically stops trading at 25% drawdown
- **Position Limits**: Maximum 50 concurrent signals
- **Direction Limits**: Maximum 5 positions in same direction

### Trade-Level Protection
- **Emergency Stop**: Hard 5% stop loss on any trade
- **Maximum Risk**: 2.5% max stop loss per trade
- **Breakeven Stop**: Moves to breakeven at 1.5% profit
- **Trailing Stops**: Locks in profits as price moves

### Symbol-Level Protection
- **Performance Tracking**: Win rate tracked per symbol
- **Auto-Disable**: Symbols below 30% win rate disabled
- **Blacklist**: Problematic symbols (e.g., NIGHT) excluded

## ⚙️ Configuration

### Global Configuration (`global_config.json`)

```json
{
  "global_risk": {
    "max_drawdown_percent": 25.0,
    "emergency_stop_percent": 5.0,
    "max_stop_loss_percent": 2.5,
    "min_risk_reward": 1.2,
    "enable_trailing_stop": true,
    "enable_breakeven_stop": true
  },
  "bots": {
    "orb_bot": {
      "enabled": true,
      "max_open_signals": 30,
      "interval_seconds": 60
    }
  },
  "disabled_symbols": ["NIGHT"]
}
```

### Bot-Specific Configuration

Each bot can have its own configuration file in its directory (e.g., `orb_bot/orb_config.json`).

## 📊 Monitoring

### Real-Time Monitoring

```bash
# Monitor all open signals
python main.py --monitor

# View bot status
python main.py --status
```

### Telegram Notifications

Configure Telegram for real-time alerts:
- New signal notifications with entry, TP, SL levels
- Trade closure notifications with P&L
- Health monitoring alerts

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_core_framework.py -v

# Run with coverage
pytest tests/ --cov=core --cov-report=html
```

## 📁 Project Structure

### Core Framework

| Module | Purpose |
|--------|---------|
| `BaseBot` | Abstract base class with signal generation, monitoring, and lifecycle |
| `SignalManager` | Tracks open signals, handles TP/SL detection, maintains history |
| `RiskManager` | Portfolio risk, drawdown protection, position sizing |
| `MarketAnalyzer` | ADX, volatility, trend detection, regime classification |
| `ExchangeClient` | Rate-limited API calls, retry logic, circuit breaker |

### Creating a New Strategy

```python
from core.base_bot import BaseBot, Signal
from common.types import SignalDirection, SignalType

class MyStrategy(BaseBot):
    def __init__(self, **kwargs):
        super().__init__(
            bot_name="my_strategy",
            strategy_type=SignalType.CUSTOM,
            **kwargs
        )
    
    def generate_signal(self, symbol: str, data: Dict) -> Optional[Signal]:
        # Implement your strategy logic here
        candles = data["candles"]
        current_price = data["current_price"]
        
        # Your analysis logic...
        if should_go_long:
            return Signal(
                signal_id=self.generate_signal_id(symbol, "LONG"),
                symbol=symbol,
                direction=SignalDirection.LONG,
                signal_type=SignalType.CUSTOM,
                entry=current_price,
                stop_loss=stop_price,
                take_profit_1=tp1,
                take_profit_2=tp2,
            )
        return None
```

## 🔄 Migration from Legacy

The legacy bot scripts in `*_bot/` directories remain fully functional. To migrate:

1. Create a new strategy file in `strategies/`
2. Extend `BaseBot` and implement `generate_signal()`
3. Register in `main.py`'s `BOT_REGISTRY`
4. Test with `--once` flag before production

## ⚠️ Critical Fixes Applied (v2.0)

This release includes critical bug fixes:

| Bot | Issue | Fix |
|-----|-------|-----|
| **Fib Swing** | No max stop loss limit | Added 2.5% maximum |
| **Volume** | TP1/TP2 ordering wrong | Fixed order validation |
| **STRAT** | Fixed structure stops too wide | ATR-adjusted stops |
| **MOST** | Trailing stop not persisting | Added state save |
| **Harmonic** | Pattern thresholds too loose | Tightened to 10% |

## 📈 Performance Metrics

Track performance with:

```bash
# View summary for each bot
python main.py --status

# Detailed analysis
python analyze_bot_performance.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes following the BaseBot pattern
4. Add tests
5. Submit a pull request

## 📜 License

Private - All rights reserved.

---

**Version**: 2.0.0  
**Last Updated**: January 2026  
**Architecture**: Unified BaseBot Framework
