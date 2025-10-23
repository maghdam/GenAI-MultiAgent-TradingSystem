from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd

from . import data_fetcher


@dataclass
class BacktestParams:
    symbol: str
    timeframe: str = "M5"
    num_bars: int = 1500


def _buy_and_hold_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"error": "No data"}
    start = float(df["close"].iloc[0])
    end = float(df["close"].iloc[-1])
    total_return = (end / start - 1.0) * 100.0
    trades = 1
    win_rate = 100.0 if total_return > 0 else 0.0
    return {
        "Total Return [%]": round(total_return, 2),
        "Number of Trades": trades,
        "Win Rate [%]": round(win_rate, 2),
    }


def run_backtest(params: BacktestParams) -> Dict[str, Any]:
    """Minimal backtest: compute buy-and-hold metrics over the window.

    This is intentionally simple to avoid adding heavy dependencies. It can be
    replaced by a full engine later.
    """
    df, _ = data_fetcher.fetch_data(params.symbol, params.timeframe, params.num_bars)
    return _buy_and_hold_metrics(df)

