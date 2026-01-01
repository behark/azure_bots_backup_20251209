# Bot Performance Analysis - Quick Reference

## 🚀 Quick Start

```bash
# Full analysis
python3 analyze_bot_performance.py

# Generate HTML report
python3 analyze_bot_performance.py --export html

# Analyze specific bot
python3 analyze_bot_performance.py --bot orb --detailed
```

---

## 📊 Current Bot Rankings (Jan 1, 2026)

| Rank | Bot | Win Rate | Total P&L | Trades | Status |
|------|-----|----------|-----------|--------|--------|
| 🥇 1 | PSAR Bot | 100.0% | +3.06% | 3 | ⭐ Excellent |
| 🥈 2 | ORB Bot | 67.7% | +1.77% | 96 | ⭐ Excellent |
| 🥉 3 | Liquidation Bot | 45.4% | +0.59% | 97 | ✅ Good |
| 4 | MTF Bot | 0.0% | -0.34% | 1 | ⏳ Insufficient Data |
| 5 | Harmonic Bot | 18.3% | -26.00% | 231 | ⚠️ Needs Work |
| 6 | MOST Bot | 30.2% | -74.39% | 205 | ⚠️ Needs Work |
| 7 | STRAT Bot | 24.5% | -78.15% | 404 | ⚠️ Needs Work |
| 8 | Volume Bot | 20.5% | -84.76% | 138 | ❌ Poor |
| 9 | Fibonacci Swing Bot | 29.9% | -277.64% | 201 | ❌ Critical |

---

## 🎯 Top Symbols to Trade

### ✅ Best Performers
1. **POWER/USDT** - Strong across multiple bots
2. **ASR/USDT** - High win rates
3. **RLS/USDT** - Consistent profits
4. **CLO/USDT** - Good in ORB & Liquidation
5. **IRYS/USDT** - Strong in multiple strategies

### ❌ Avoid These
1. **NIGHT/USDT** - Massive losses (-125%)
2. **TRUTH/USDT** - Poor across bots
3. **SOON/USDT** - Generally negative
4. **FHE/USDT** - Consistent losses

---

## 📈 Key Metrics at a Glance

### Portfolio Overview
- **Total Trades**: 1,563 (187 open, 1,376 closed)
- **Overall Win Rate**: 29.6%
- **Total P&L**: -535.87%
- **Active Bots**: 9/11

### Best Metrics
- **Best Win Rate**: PSAR Bot (100%)
- **Best Profit Factor**: ORB Bot (1.02)
- **Best Sharpe Ratio**: PSAR Bot (1.49)
- **Most Trades**: STRAT Bot (404)

---

## 🔧 Immediate Actions Required

### 🔴 Critical (Do Now)
- [ ] **Disable Fibonacci Swing Bot** (losing -277%)
- [ ] **Remove NIGHT/USDT** from all watchlists
- [ ] **Reduce position sizes** on losing bots
- [ ] **Implement max drawdown alerts** (25% limit)

### 🟡 Important (This Week)
- [ ] **Optimize Volume Bot** or disable
- [ ] **Review STRAT Bot** parameters
- [ ] **Increase ORB Bot** allocation
- [ ] **Add more symbols** to PSAR Bot

### 🟢 Recommended (This Month)
- [ ] Backtest improvements
- [ ] Implement performance-based sizing
- [ ] Create automated monitoring
- [ ] Research new strategies

---

## 💡 Strategy Recommendations

### Increase Allocation
- **ORB Bot**: 40% (proven winner)
- **PSAR Bot**: 20% (promising)
- **Liquidation Bot**: 20% (break-even potential)

### Reduce/Disable
- **Fibonacci Swing Bot**: 0% (disable)
- **Volume Bot**: 0% (disable until fixed)
- **STRAT Bot**: 5% (reduce significantly)
- **MOST Bot**: 5% (reduce significantly)

---

## 📱 Command Cheat Sheet

### Analysis Commands
```bash
# All bots summary
python3 analyze_bot_performance.py --summary

# Specific bot detailed
python3 analyze_bot_performance.py --bot orb --detailed

# Export formats
python3 analyze_bot_performance.py --export html
python3 analyze_bot_performance.py --export json
python3 analyze_bot_performance.py --export csv
```

### File Locations
```
Reports:
- bot_performance_report.html
- bot_performance_data.json
- bot_performance_summary.csv

Documentation:
- PERFORMANCE_ANALYSIS_README.md (full guide)
- PERFORMANCE_SUMMARY.md (detailed analysis)
- QUICK_REFERENCE.md (this file)

Stats Files:
- {bot_name}/logs/{bot_name}_stats.json
```

---

## 🎓 Understanding Metrics

### Win Rate
- **>60%**: Excellent
- **40-60%**: Good
- **30-40%**: Acceptable
- **<30%**: Poor

### Profit Factor
- **>2.0**: Excellent
- **1.0-2.0**: Good
- **0.5-1.0**: Poor
- **<0.5**: Very Poor

### Sharpe Ratio
- **>1.0**: Excellent
- **0.5-1.0**: Good
- **0-0.5**: Acceptable
- **<0**: Poor

### Max Drawdown
- **<20%**: Excellent
- **20-40%**: Acceptable
- **40-60%**: Concerning
- **>60%**: Critical

---

## 🔍 Bot Strategy Quick Reference

| Bot | Strategy | Timeframe | Avg Duration | Best For |
|-----|----------|-----------|--------------|----------|
| ORB | Opening Range Breakout | 5m | 2.7h | Trending markets |
| PSAR | Parabolic SAR + Trailing | 5m | 0.2h | Strong trends |
| Liquidation | Cluster analysis | 5m | 3.1h | High volatility |
| MOST | Moving stop loss | 5m | 2.5h | Trending markets |
| STRAT | STRAT patterns | 5m | 2.8h | Pattern recognition |
| Harmonic | Harmonic patterns | 5m | 3.8h | Reversal points |
| Fib Swing | Fibonacci levels | 5-15m | 8.9h | Retracements |
| Volume | Volume profile | 5m | 16.5h | Support/resistance |
| MTF | Multi-timeframe | 5m | 0.1h | Trend confirmation |

---

## 📊 Performance Thresholds

### Bot Health Check
✅ **Healthy Bot**:
- Win rate >40%
- Profit factor >1.0
- Max drawdown <40%
- Positive total P&L

⚠️ **Warning Zone**:
- Win rate 30-40%
- Profit factor 0.7-1.0
- Max drawdown 40-60%
- Slightly negative P&L

❌ **Critical Issues**:
- Win rate <30%
- Profit factor <0.7
- Max drawdown >60%
- Significantly negative P&L

---

## 🎯 Success Patterns

### What Works
1. **Clear entry/exit rules** (ORB, PSAR)
2. **Quick execution** (<3h duration)
3. **Good risk/reward** (1:2 minimum)
4. **High TP1 hit rate** (>50%)
5. **Symbol selectivity** (focus on best)

### What Doesn't Work
1. **Complex patterns** (low accuracy)
2. **Long durations** (>10h)
3. **Wide stops** (large losses)
4. **Too many signals** (low quality)
5. **Poor symbol selection** (one-size-fits-all)

---

## 📞 Quick Help

### Common Issues

**"No data for bot"**
- Bot hasn't closed trades yet
- Check: `{bot}/logs/{bot}_stats.json`

**"Stats file not found"**
- Bot not running or misconfigured
- Check logs directory exists

**"Metrics seem wrong"**
- Verify JSON format
- Check for data corruption
- Review bot's SignalStats usage

### Getting Help
1. Check `PERFORMANCE_ANALYSIS_README.md`
2. Review `PERFORMANCE_SUMMARY.md`
3. Examine bot logs
4. Verify stats file format

---

## 🔄 Regular Maintenance

### Daily
- Check open signals
- Monitor new signals
- Review any alerts

### Weekly
- Run performance analysis
- Compare to previous week
- Adjust allocations if needed

### Monthly
- Full performance report
- Strategy optimization
- Portfolio rebalancing
- Review and update watchlists

---

## 📈 Performance Targets

### Short-term (1 Month)
- [ ] Overall win rate >35%
- [ ] At least 3 bots profitable
- [ ] Reduce max portfolio DD to <200%
- [ ] Increase best bot allocation

### Medium-term (3 Months)
- [ ] Overall win rate >40%
- [ ] At least 5 bots profitable
- [ ] Portfolio profit factor >1.0
- [ ] Positive total P&L

### Long-term (6 Months)
- [ ] Overall win rate >45%
- [ ] Portfolio P&L >+50%
- [ ] Sharpe ratio >0.5
- [ ] Consistent monthly profits

---

**Last Updated**: January 1, 2026  
**Next Review**: Weekly  
**Version**: 1.0

---

## 🔗 Related Files

- 📘 [Full Documentation](PERFORMANCE_ANALYSIS_README.md)
- 📊 [Detailed Analysis](PERFORMANCE_SUMMARY.md)
- 🔧 [Analysis Script](analyze_bot_performance.py)
- 📁 [HTML Report](bot_performance_report.html)
- 📄 [JSON Data](bot_performance_data.json)
- 📋 [CSV Summary](bot_performance_summary.csv)
