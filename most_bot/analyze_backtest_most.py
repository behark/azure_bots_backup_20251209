import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import backtrader as bt
import ccxt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
STATS_PATH = BASE_DIR / "logs" / "most_stats.json"


class SignalFeed:
    def __init__(self, stats_path: Path):
        self.stats_path = stats_path

    def load(self) -> pd.DataFrame:
        data = json.loads(self.stats_path.read_text())
        history: List[Dict[str, Any]] = data.get("history", [])
        return pd.DataFrame(history)


class CCXTOHLC(bt.feeds.PandasData):
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
    )


class ReplayStrategy(bt.Strategy):
    params = dict(signals=None)

    def __init__(self):
        self.signals = self.p.signals
        self.sig_idx = 0

    def next(self):
        # Apply signals sequentially on this data feed by time
        while self.sig_idx < len(self.signals):
            sig = self.signals.iloc[self.sig_idx]
            sig_time = pd.to_datetime(sig["created_at"], utc=True).tz_convert(None)
            data_time = pd.Timestamp(self.data.datetime.datetime(0)).tz_localize(None)
            if sig_time > data_time:
                break
            direction = sig["direction"]
            entry = float(sig.get("entry", self.data.close[0]))
            if direction == "BULLISH":
                self.buy(size=1, price=entry)
            else:
                self.sell(size=1, price=entry)
            self.sig_idx += 1


def fetch_ohlcv(symbol: str, timeframe: str = "5m", limit: int = 500) -> pd.DataFrame:
    ex = ccxt.mexc({"enableRateLimit": True, "options": {"defaultType": "swap"}})  # type: ignore[arg-type]
    ex.load_markets()
    mk = symbol if symbol.endswith(":USDT") else symbol.replace("/USDT", "/USDT:USDT")
    ohlcv = ex.fetch_ohlcv(mk, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df.set_index("time", inplace=True)
    return df


def run_backtest():
    feed = SignalFeed(STATS_PATH)
    df = feed.load()
    if df.empty:
        print("No signals to backtest")
        return
    # Use top symbol by count
    top_symbol = df["symbol"].value_counts().idxmax()
    timeframe = df["extra"].apply(lambda x: x.get("timeframe") if isinstance(x, dict) else None).mode().iat[0]
    ohlc = fetch_ohlcv(top_symbol, timeframe)
    data = CCXTOHLC(dataname=ohlc)
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(ReplayStrategy, signals=df[df["symbol"] == top_symbol].sort_values("created_at"))
    cerebro.broker.setcash(10000.0)
    cerebro.run()
    print("Final portfolio value:", cerebro.broker.getvalue())


if __name__ == "__main__":
    run_backtest()
