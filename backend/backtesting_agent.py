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


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min()) if len(dd) else 0.0


def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()


def _crossover_backtest(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"error": "No data"}

    close = df["close"].astype(float)
    f = _sma(close, fast)
    s = _sma(close, slow)
    long_signal = (f > s) & (f.shift(1) <= s.shift(1))  # golden cross
    exit_signal = (f < s) & (f.shift(1) >= s.shift(1))  # death cross

    # Position state: 1 when in long, else 0
    position = pd.Series(index=close.index, dtype=float)
    position[long_signal] = 1.0
    position[exit_signal] = 0.0
    position = position.ffill().fillna(0.0)

    rets = close.pct_change().fillna(0)
    strat_rets = rets * position
    equity = (1.0 + strat_rets).cumprod()

    # Trades (pair each entry with the next exit; if no later exit, close at last bar)
    entries_idx = list(long_signal[long_signal].index)
    exits_idx = list(exit_signal[exit_signal].index)
    trade_rets: list[float] = []
    if entries_idx:
        import bisect
        exits_idx_sorted = list(sorted(exits_idx))
        for e in entries_idx:
            i = bisect.bisect_right(exits_idx_sorted, e)
            x = exits_idx_sorted[i] if i < len(exits_idx_sorted) else close.index[-1]
            if e >= x:
                continue
            trade_ret = float(close.loc[x] / close.loc[e] - 1.0)
            trade_rets.append(trade_ret)

    num_trades = len(trade_rets)
    wins = sum(1 for r in trade_rets if r > 0)
    win_rate = (wins / num_trades * 100.0) if num_trades > 0 else 0.0
    total_return = float(equity.iloc[-1] - 1.0) * 100.0
    avg_trade = (sum(trade_rets) / num_trades * 100.0) if num_trades > 0 else 0.0

    # Simple Sharpe (per-bar) without risk-free, scaled by sqrt(252*78) ~ per 5m to annual
    import math
    ann_factor = math.sqrt(252 * (24*60/5))  # 5-minute bars
    sharpe = 0.0
    if strat_rets.std(ddof=0) > 0:
        sharpe = float(strat_rets.mean() / strat_rets.std(ddof=0)) * ann_factor

    mdd = _max_drawdown(equity)
    return {
        "Total Return [%]": round(total_return, 4),
        "Number of Trades": float(num_trades),
        "Win Rate [%]": round(win_rate, 2),
        "Avg Trade [%]": round(avg_trade, 4),
        "Max Drawdown [%]": round(mdd * 100.0, 2),
        "Sharpe": round(sharpe, 3),
    }


def run_backtest(params: BacktestParams) -> Dict[str, Any]:
    """Default backtest: SMA crossover with richer metrics.

    If you later install `vectorbt`, this function can detect it and route to a
    vectorbt-based portfolio; otherwise it falls back to our crossover metrics.
    """
    df, _ = data_fetcher.fetch_data(params.symbol, params.timeframe, params.num_bars)
    try:
        import vectorbt as vbt  # type: ignore

        close = df["close"].astype(float)
        fast, slow = 50, 200
        f = close.rolling(fast, min_periods=fast).mean()
        s = close.rolling(slow, min_periods=slow).mean()
        entries = (f > s) & (f.shift(1) <= s.shift(1))
        exits = (f < s) & (f.shift(1) >= s.shift(1))
        pf = vbt.Portfolio.from_signals(close, entries, exits, fees=0.0, slippage=0.0)
        stats = pf.stats()
        # Return a compact subset of commonly used metrics
        out = {
            "Total Return [%]": float(stats.get("Total Return [%]", 0.0)),
            "Sharpe Ratio": float(stats.get("Sharpe Ratio", 0.0)),
            "Max Drawdown [%]": float(stats.get("Max Drawdown [%]", 0.0)),
            "Win Rate [%]": float(stats.get("Win Rate [%]", 0.0)) if isinstance(stats.get("Win Rate [%]"), (int, float)) else None,
            "Number of Trades": float(stats.get("Total Trades", 0.0)) if isinstance(stats.get("Total Trades"), (int, float)) else None,
        }
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items() if v is not None}
    except Exception:
        return _crossover_backtest(df)
