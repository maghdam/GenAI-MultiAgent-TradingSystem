from __future__ import annotations

import types
import math
from pathlib import Path

import backend.data_fetcher as data_fetcher
import pandas as pd
from fastapi import HTTPException


def list_saved_strategy_files() -> dict:
    try:
        root = Path("backend/strategies_generated")
        files = [str(path.name) for path in root.glob("*.py")] if root.exists() else []
        return {"files": files, "cwd": str(Path.cwd())}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def run_saved_strategy_backtest(
    *,
    strategy: str,
    symbol: str,
    timeframe: str = "M5",
    num_bars: int = 1500,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
):
    root = Path("backend/strategies_generated")
    path = root / f"{strategy.lower()}.py"
    if not path.exists():
        raise HTTPException(404, f"Saved strategy '{strategy}' not found.")

    df, _ = data_fetcher.fetch_data(symbol, timeframe, num_bars)
    if df is None or df.empty:
        raise HTTPException(404, "No data fetched for backtest")

    try:
        src = path.read_text(encoding="utf-8")
        mod_globals = {}
        exec(src, mod_globals)
        signals_fn = mod_globals.get("signals")
    except Exception as exc:
        raise HTTPException(400, f"Unable to load strategy module: {exc}") from exc
    if not callable(signals_fn):
        raise HTTPException(400, "This strategy does not define a callable signals(df) for backtesting.")

    try:
        sig = signals_fn(df)
    except Exception as exc:
        raise HTTPException(400, f"signals(df) execution failed: {exc}") from exc

    if "pandas" not in str(type(sig)):
        raise HTTPException(400, "signals(df) must return a pandas Series aligned to df index.")
    try:
        sig = sig.reindex(df.index).fillna(0)
    except Exception as exc:
        raise HTTPException(400, "signals(df) output not aligned to input index.") from exc

    close = df["close"].astype(float)
    rets = close.pct_change().fillna(0)
    pos = sig.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)).ffill().fillna(0.0)
    strat_rets = rets * pos
    prev = pos.shift(1).fillna(0.0)
    flips = (pos != 0.0) & (prev != 0.0) & (pos != prev)
    entries = ((pos != 0.0) & (prev == 0.0)) | flips
    exits = ((pos == 0.0) & (prev != 0.0)) | flips
    cost_side = max(0.0, float(fee_bps) + float(slippage_bps)) / 10_000.0
    cost_series = (entries.astype(float) + exits.astype(float)) * (-cost_side)
    cost_series = cost_series.reindex(strat_rets.index).fillna(0.0)
    strat_rets_net = strat_rets + cost_series
    equity = (1.0 + strat_rets_net).cumprod()

    entries_idx = list(entries[entries].index)
    exits_idx = list(exits[exits].index)
    trade_rets: list[float] = []
    hold_bars_list: list[int] = []
    if entries_idx:
        import bisect

        exits_idx_sorted = list(sorted(exits_idx))
        for entry_index in entries_idx:
            index = bisect.bisect_right(exits_idx_sorted, entry_index)
            exit_index = exits_idx_sorted[index] if index < len(exits_idx_sorted) else close.index[-1]
            if entry_index >= exit_index:
                continue
            direction = float(pos.loc[entry_index])
            grow = close.loc[exit_index] / close.loc[entry_index]
            trade_ret = (grow - 1.0) if direction > 0 else ((1.0 / grow) - 1.0)
            net_trade = ((1.0 + float(trade_ret)) * (1.0 - cost_side) * (1.0 - cost_side)) - 1.0
            trade_rets.append(float(net_trade))
            try:
                start_pos = df.index.get_loc(entry_index)
                end_pos = df.index.get_loc(exit_index)
                hold_bars_list.append(int(max(0, end_pos - start_pos)))
            except Exception:
                pass

    num_trades = len(trade_rets)
    wins = sum(1 for ret in trade_rets if ret > 0)
    win_rate = (wins / num_trades * 100.0) if num_trades > 0 else 0.0
    total_return = float(equity.iloc[-1] - 1.0) * 100.0
    avg_trade = (sum(trade_rets) / num_trades * 100.0) if num_trades > 0 else 0.0

    def _tf_minutes(tf: str) -> int:
        return {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}.get((tf or "M5").upper(), 5)

    ann_factor = math.sqrt(252 * (24 * 60 / max(1, _tf_minutes(timeframe))))
    sharpe = 0.0
    std_bar = float(strat_rets_net.std(ddof=0))
    if std_bar > 0:
        sharpe = float(strat_rets_net.mean() / std_bar) * ann_factor

    try:
        daily = (1.0 + strat_rets_net).resample("1D").prod() - 1.0
        std_d = float(daily.std(ddof=0))
        daily_sharpe = float(daily.mean() / std_d) * math.sqrt(252) if std_d > 0 and len(daily) >= 2 else 0.0
        avg_daily_pct = float(daily.mean() * 100.0) if len(daily) > 0 else 0.0
    except Exception:
        daily_sharpe = 0.0
        avg_daily_pct = 0.0

    trade_sharpe = 0.0
    sqn = 0.0
    avg_hold_bars = float(sum(hold_bars_list) / len(hold_bars_list)) if hold_bars_list else 0.0
    if len(trade_rets) >= 2:
        import statistics

        mu_t = float(statistics.mean(trade_rets))
        try:
            sd_t = float(statistics.pstdev(trade_rets))
        except statistics.StatisticsError:
            sd_t = 0.0
        if sd_t > 0:
            trade_sharpe = mu_t / sd_t
            sqn = (len(trade_rets) ** 0.5) * (mu_t / sd_t)

    try:
        delta_s = (df.index[-1] - df.index[0]).total_seconds()
        days = max(0.0001, delta_s / 86400.0)
    except Exception:
        days = 0.0
    trades_per_day = float(len(trade_rets) / days) if days > 0 else 0.0

    peak = equity.cummax()
    drawdown = (equity / peak) - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    return {
        "strategy": strategy,
        "symbol": symbol,
        "timeframe": timeframe,
        "Total Return [%]": round(total_return, 4),
        "Number of Trades": float(num_trades),
        "Win Rate [%]": round(win_rate, 2),
        "Avg Trade [%]": round(avg_trade, 4),
        "Max Drawdown [%]": round(max_drawdown * 100.0, 2),
        "Sharpe": round(sharpe, 3),
        "Daily Sharpe": round(daily_sharpe, 3),
        "Avg Daily Return [%]": round(avg_daily_pct, 4),
        "Fees [bps]": round(float(fee_bps), 3),
        "Slippage [bps]": round(float(slippage_bps), 3),
        "Trade Sharpe": round(trade_sharpe, 3),
        "SQN": round(sqn, 3),
        "Trades/Day": round(trades_per_day, 3),
        "Avg Hold [bars]": round(avg_hold_bars, 2),
    }


def run_strategy_code_backtest(
    *,
    code: str,
    symbol: str,
    timeframe: str = "M5",
    num_bars: int = 1500,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    strategy_name: str = "draft",
):
    df, _ = data_fetcher.fetch_data(symbol, timeframe, num_bars)
    if df is None or df.empty:
        raise HTTPException(404, "No data fetched for backtest")

    try:
        mod = types.ModuleType("studio_draft_strategy")
        exec(str(code), mod.__dict__)
        signals_fn = getattr(mod, "signals", None)
    except Exception as exc:
        raise HTTPException(400, f"Unable to load draft strategy: {exc}") from exc
    if not callable(signals_fn):
        raise HTTPException(400, "Draft strategy does not define a callable signals(df) for backtesting.")

    try:
        sig = signals_fn(df)
    except Exception as exc:
        raise HTTPException(400, f"signals(df) execution failed: {exc}") from exc

    if "pandas" not in str(type(sig)):
        raise HTTPException(400, "signals(df) must return a pandas Series aligned to df index.")
    try:
        sig = sig.reindex(df.index).fillna(0)
    except Exception as exc:
        raise HTTPException(400, "signals(df) output not aligned to input index.") from exc

    close = df["close"].astype(float)
    rets = close.pct_change().fillna(0)
    pos = sig.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)).ffill().fillna(0.0)
    strat_rets = rets * pos
    prev = pos.shift(1).fillna(0.0)
    flips = (pos != 0.0) & (prev != 0.0) & (pos != prev)
    entries = ((pos != 0.0) & (prev == 0.0)) | flips
    exits = ((pos == 0.0) & (prev != 0.0)) | flips
    cost_side = max(0.0, float(fee_bps) + float(slippage_bps)) / 10_000.0
    cost_series = (entries.astype(float) + exits.astype(float)) * (-cost_side)
    cost_series = cost_series.reindex(strat_rets.index).fillna(0.0)
    strat_rets_net = strat_rets + cost_series
    equity = (1.0 + strat_rets_net).cumprod()

    num_trades = int(entries.sum())
    total_return = float(equity.iloc[-1] - 1.0) * 100.0
    peak = equity.cummax()
    drawdown = (equity / peak) - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    win_rate = 0.0

    return {
        "strategy": strategy_name,
        "symbol": symbol,
        "timeframe": timeframe,
        "Total Return [%]": round(total_return, 4),
        "Number of Trades": float(num_trades),
        "Win Rate [%]": round(win_rate, 2),
        "Max Drawdown [%]": round(max_drawdown * 100.0, 2),
        "Fees [bps]": round(float(fee_bps), 3),
        "Slippage [bps]": round(float(slippage_bps), 3),
        "draft": True,
    }
