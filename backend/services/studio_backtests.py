from __future__ import annotations

import itertools
import math
import re
import statistics
from datetime import timedelta
from pathlib import Path

import backend.data_fetcher as data_fetcher
import pandas as pd
from fastapi import HTTPException

from backend.services.strategy_lifecycle import record_backtest
from backend.services.strategy_sandbox import StrategySandboxError, run_strategy_source


_TIMEFRAME_MINUTES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}

_DEVELOPMENT_FRACTION = 0.70
_MIN_VALIDATION_BARS = {
    "development_backtest": 100,
    "out_of_sample": 50,
    "regime": 100,
}
_WALK_FORWARD_MIN_TRAIN_BARS = 100
_WALK_FORWARD_MIN_TEST_BARS = 30
_MAX_WALK_FORWARD_FOLDS = 8
_MAX_OPTIMIZATION_COMBOS = 100


def list_saved_strategy_files() -> dict:
    try:
        root = Path("backend/strategies_generated")
        files = [str(path.name) for path in root.glob("*.py")] if root.exists() else []
        return {"files": files, "cwd": str(Path.cwd())}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def load_saved_strategy_source(strategy: str) -> dict:
    safe = (strategy or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9_-]+", safe):
        raise HTTPException(400, "Strategy name must contain only letters, numbers, underscores, or hyphens.")

    path = Path("backend/strategies_generated") / f"{safe}.py"
    if not path.exists():
        raise HTTPException(404, f"Saved strategy '{safe}' not found.")

    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(500, f"Could not read saved strategy '{safe}': {exc}") from exc

    return {
        "strategy": safe,
        "filename": path.name,
        "source": source,
    }


def _timeframe_minutes(timeframe: str) -> int:
    value = (timeframe or "M5").upper()
    minutes = _TIMEFRAME_MINUTES.get(value)
    if minutes is None:
        raise HTTPException(400, f"Unsupported backtest timeframe '{value}'.")
    return minutes


def _prepare_market_frame(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise HTTPException(404, "No data fetched for backtest")

    missing_columns = [column for column in ("open", "high", "low", "close") if column not in df.columns]
    if missing_columns:
        raise HTTPException(400, f"Backtest data is missing required columns: {', '.join(missing_columns)}.")

    frame = df.copy()
    try:
        frame.index = pd.to_datetime(frame.index, utc=True, errors="raise")
    except Exception as exc:
        raise HTTPException(400, "Backtest timestamps must be parseable market timestamps.") from exc

    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise HTTPException(400, "Backtest data contains duplicate timestamps.")

    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].isna() | ~frame[column].map(math.isfinite) | (frame[column] <= 0)
        if bool(invalid.any()):
            raise HTTPException(400, f"Backtest data contains invalid {column} prices.")

    if len(frame) < 2:
        raise HTTPException(400, "Backtest requires at least two market bars.")

    _timeframe_minutes(timeframe)
    return frame


def _validation_window(df: pd.DataFrame, validation_kind: str) -> tuple[pd.DataFrame, str, dict]:
    kind = str(validation_kind or "development_backtest").strip().lower()
    allowed = {"development_backtest", "out_of_sample", "regime"}
    if kind not in allowed:
        raise HTTPException(400, f"validation_kind must be one of: {', '.join(sorted(allowed))}")

    split = max(1, min(len(df) - 1, int(len(df) * _DEVELOPMENT_FRACTION)))
    development_bars = split
    holdout_bars = len(df) - split

    if kind == "development_backtest":
        selected = df.iloc[:split].copy()
        selection = "First 70% chronological development window."
    elif kind == "out_of_sample":
        selected = df.iloc[split:].copy()
        selection = "Final 30% chronological holdout; excluded from development tuning."
    else:
        selected = df.copy()
        selection = "Alternate market or non-overlapping period supplied by the operator."

    minimum_bars = int(_MIN_VALIDATION_BARS[kind])
    if len(selected) < minimum_bars:
        raise HTTPException(
            400,
            f"{kind} window contains only {len(selected)} bars; minimum is {minimum_bars}. Request more data.",
        )

    metadata = {
        "Validation Selection": selection,
        "Development Fraction [%]": round(_DEVELOPMENT_FRACTION * 100.0, 2),
        "Holdout Fraction [%]": round((1.0 - _DEVELOPMENT_FRACTION) * 100.0, 2),
        "Development Bars": development_bars,
        "Holdout Bars": holdout_bars,
        "Minimum Bars Required": minimum_bars,
        "Split Timestamp": _timestamp(df.index[split]) if split < len(df) else None,
    }
    return selected, kind, metadata


def _timestamp(value: object) -> str:
    try:
        return pd.Timestamp(value).isoformat()
    except Exception:
        return str(value)


def _normalize_signals(sig: pd.Series, df: pd.DataFrame) -> pd.Series:
    if "pandas" not in str(type(sig)):
        raise HTTPException(400, "signals(df) must return a pandas Series aligned to df index.")
    try:
        aligned = sig.reindex(df.index).fillna(0)
    except Exception as exc:
        raise HTTPException(400, "signals(df) output not aligned to input index.") from exc
    return aligned.apply(lambda value: 1.0 if float(value) > 0 else (-1.0 if float(value) < 0 else 0.0))


def _data_gap_diagnostics(df: pd.DataFrame, timeframe: str) -> dict:
    expected_minutes = float(_timeframe_minutes(timeframe))
    diffs = df.index.to_series().diff().dropna().dt.total_seconds().div(60.0)
    observed_gaps = int((diffs > expected_minutes * 1.5).sum()) if not diffs.empty else 0
    largest_gap = float(diffs.max()) if not diffs.empty else 0.0
    return {
        "Data Timezone": "UTC",
        "Session Assumption": "Broker-observed bars only; calendar gaps are reported and never forward-filled.",
        "Observed Gaps": observed_gaps,
        "Largest Gap [min]": round(largest_gap, 2),
    }


def _safe_cost_bps(value: float, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, f"{label} must be numeric.") from exc
    if not math.isfinite(number) or number < 0:
        raise HTTPException(400, f"{label} must be a finite non-negative value.")
    return number


def _position_fraction(position_size_pct: float) -> float:
    try:
        pct = float(position_size_pct)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, "position_size_pct must be numeric.") from exc
    if not math.isfinite(pct) or pct <= 0 or pct > 100:
        raise HTTPException(400, "position_size_pct must be greater than 0 and at most 100.")
    return pct / 100.0


def _daily_metrics(equity: pd.Series) -> tuple[float, float]:
    if equity.empty:
        return 0.0, 0.0

    daily_equity = equity.resample("1D").last()
    if daily_equity.empty:
        return 0.0, 0.0

    daily_returns = daily_equity.pct_change()
    daily_returns.iloc[0] = float(daily_equity.iloc[0] - 1.0)
    daily_returns = daily_returns.replace([float("inf"), float("-inf")], pd.NA).dropna().astype(float)

    avg_daily_pct = float(daily_returns.mean() * 100.0) if not daily_returns.empty else 0.0
    if len(daily_returns) < 2:
        return 0.0, avg_daily_pct

    std_daily = float(daily_returns.std(ddof=0))
    sharpe = float(daily_returns.mean() / std_daily) * math.sqrt(252) if std_daily > 0 else 0.0
    return sharpe, avg_daily_pct


def _account_backtest(
    *,
    df: pd.DataFrame,
    sig: pd.Series,
    timeframe: str,
    fee_bps: float,
    slippage_bps: float,
    spread_bps: float,
    position_size_pct: float,
) -> dict:
    fee = _safe_cost_bps(fee_bps, "fee_bps")
    slippage = _safe_cost_bps(slippage_bps, "slippage_bps")
    spread = _safe_cost_bps(spread_bps, "spread_bps")
    position_fraction = _position_fraction(position_size_pct)
    timeframe_minutes = _timeframe_minutes(timeframe)

    # Signals are evaluated on a completed bar and become executable only at the
    # next bar open. This prevents a bar's closing information from earning that
    # same bar's return.
    targets = sig.apply(lambda value: 1.0 if value > 0 else (-1.0 if value < 0 else 0.0))
    transaction_cost_fraction = (fee + slippage + spread / 2.0) / 10_000.0

    equity = 1.0
    equity_points: list[float] = [equity]
    equity_index: list[pd.Timestamp] = [pd.Timestamp(df.index[0])]

    current_direction = 0.0
    active_trade: dict | None = None
    trade_returns: list[float] = []
    hold_bars: list[int] = []
    hold_minutes: list[float] = []
    closed_trades = 0
    marked_open_trades = 0

    def mark_equity(price: float) -> float:
        if not active_trade:
            return equity
        units = float(active_trade["units"])
        direction = float(active_trade["direction"])
        entry_price = float(active_trade["entry_price"])
        entry_equity_after_cost = float(active_trade["entry_equity_after_cost"])
        return entry_equity_after_cost + direction * units * (float(price) - entry_price)

    def close_active(exit_price: float, exit_time: pd.Timestamp, exit_bar: int, *, marked: bool) -> float:
        nonlocal active_trade, closed_trades, marked_open_trades
        if not active_trade:
            return equity

        pre_entry_equity = float(active_trade["pre_entry_equity"])
        entry_bar = int(active_trade["entry_bar"])
        entry_time = pd.Timestamp(active_trade["entry_time"])
        units = float(active_trade["units"])
        marked_equity = mark_equity(exit_price)

        if marked:
            final_equity = marked_equity
            marked_open_trades += 1
            bars_held = max(1, exit_bar - entry_bar + 1)
        else:
            exit_notional = abs(units * float(exit_price))
            final_equity = marked_equity - exit_notional * transaction_cost_fraction
            closed_trades += 1
            bars_held = max(0, exit_bar - entry_bar)

        trade_returns.append((final_equity / pre_entry_equity) - 1.0 if pre_entry_equity > 0 else 0.0)
        hold_bars.append(int(bars_held))
        hold_minutes.append(max(0.0, (exit_time - entry_time).total_seconds() / 60.0))
        active_trade = None
        return final_equity

    for bar_index in range(1, len(df)):
        open_price = float(df["open"].iloc[bar_index])
        close_price = float(df["close"].iloc[bar_index])
        bar_time = pd.Timestamp(df.index[bar_index])
        target_direction = float(targets.iloc[bar_index - 1])

        if active_trade:
            equity = mark_equity(open_price)

        if target_direction != current_direction:
            if active_trade:
                equity = close_active(open_price, bar_time, bar_index, marked=False)

            if target_direction != 0.0:
                pre_entry_equity = equity
                entry_notional = pre_entry_equity * position_fraction
                entry_cost = entry_notional * transaction_cost_fraction
                entry_equity_after_cost = pre_entry_equity - entry_cost
                active_trade = {
                    "pre_entry_equity": pre_entry_equity,
                    "entry_equity_after_cost": entry_equity_after_cost,
                    "entry_price": open_price,
                    "entry_time": bar_time,
                    "entry_bar": bar_index,
                    "direction": target_direction,
                    "units": entry_notional / open_price,
                }
                equity = entry_equity_after_cost

            current_direction = target_direction

        if active_trade:
            equity = mark_equity(close_price)

        equity_points.append(float(equity))
        equity_index.append(bar_time)

    if active_trade:
        final_time = pd.Timestamp(df.index[-1]) + timedelta(minutes=int(timeframe_minutes))
        equity = close_active(
            float(df["close"].iloc[-1]),
            final_time,
            len(df) - 1,
            marked=True,
        )
        equity_points[-1] = float(equity)

    equity_curve = pd.Series(equity_points, index=pd.DatetimeIndex(equity_index), dtype=float)
    total_return = float(equity_curve.iloc[-1] - 1.0) * 100.0
    peak = equity_curve.cummax()
    drawdown = (equity_curve / peak) - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    num_trades = len(trade_returns)
    wins = sum(1 for value in trade_returns if value > 0)
    win_rate = (wins / num_trades * 100.0) if num_trades else 0.0
    avg_trade = (statistics.mean(trade_returns) * 100.0) if trade_returns else 0.0

    daily_sharpe, avg_daily_pct = _daily_metrics(equity_curve)
    trade_sharpe = 0.0
    sqn = 0.0
    if len(trade_returns) >= 2:
        mu_trade = float(statistics.mean(trade_returns))
        sd_trade = float(statistics.pstdev(trade_returns))
        if sd_trade > 0:
            trade_sharpe = mu_trade / sd_trade
            sqn = math.sqrt(len(trade_returns)) * trade_sharpe

    observed_days = max(1, len(pd.Index(equity_curve.index.normalize()).unique()))
    trades_per_day = float(num_trades / observed_days)
    avg_hold_bars = float(statistics.mean(hold_bars)) if hold_bars else 0.0
    avg_hold_minutes = float(statistics.mean(hold_minutes)) if hold_minutes else 0.0

    return {
        "Total Return [%]": round(total_return, 4),
        "Number of Trades": float(num_trades),
        "Closed Trades": float(closed_trades),
        "Marked Open Trades": float(marked_open_trades),
        "Win Rate [%]": round(win_rate, 2),
        "Avg Trade [%]": round(avg_trade, 4),
        "Max Drawdown [%]": round(max_drawdown * 100.0, 2),
        "Sharpe": round(daily_sharpe, 3),
        "Daily Sharpe": round(daily_sharpe, 3),
        "Sharpe Basis": "UTC daily equity returns, annualized with sqrt(252).",
        "Avg Daily Return [%]": round(avg_daily_pct, 4),
        "Fees [bps]": round(fee, 3),
        "Slippage [bps]": round(slippage, 3),
        "Spread [bps]": round(spread, 3),
        "Cost Assumption": "Fee and slippage per transaction; half of quoted spread per transaction.",
        "Position Size [%]": round(position_fraction * 100.0, 3),
        "Execution Timing": "Signal on bar close; execute at next bar open.",
        "Trade Sharpe": round(trade_sharpe, 3),
        "SQN": round(sqn, 3),
        "Trades/Day": round(trades_per_day, 3),
        "Observed Trading Days": float(observed_days),
        "Avg Hold [bars]": round(avg_hold_bars, 2),
        "Avg Hold [min]": round(avg_hold_minutes, 2),
    }


def _run_source_on_frame(
    source: str,
    frame: pd.DataFrame,
    params: dict | None = None,
) -> pd.Series:
    try:
        signal = run_strategy_source(source, frame, params or {})
    except (StrategySandboxError, ValueError) as exc:
        raise HTTPException(400, f"Isolated strategy execution failed: {exc}") from exc
    return _normalize_signals(signal, frame)


def _walk_forward_windows(df: pd.DataFrame, folds: int = 3) -> list[tuple[int, int, int]]:
    try:
        fold_count = int(folds)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, "walk_forward_folds must be an integer.") from exc
    if fold_count < 2 or fold_count > _MAX_WALK_FORWARD_FOLDS:
        raise HTTPException(400, f"walk_forward_folds must be between 2 and {_MAX_WALK_FORWARD_FOLDS}.")

    initial_train = max(_WALK_FORWARD_MIN_TRAIN_BARS, len(df) // 2)
    remaining = len(df) - initial_train
    test_size = remaining // fold_count
    if test_size < _WALK_FORWARD_MIN_TEST_BARS:
        required = _WALK_FORWARD_MIN_TRAIN_BARS + fold_count * _WALK_FORWARD_MIN_TEST_BARS
        raise HTTPException(
            400,
            f"Walk-forward requires at least {required} bars for {fold_count} folds; received {len(df)}.",
        )

    windows: list[tuple[int, int, int]] = []
    for fold_index in range(fold_count):
        train_end = initial_train + fold_index * test_size
        test_start = train_end
        test_end = len(df) if fold_index == fold_count - 1 else test_start + test_size
        if test_end - test_start < _WALK_FORWARD_MIN_TEST_BARS:
            raise HTTPException(400, "Walk-forward produced a test fold below the minimum sample size.")
        windows.append((train_end, test_start, test_end))
    return windows


def run_strategy_walk_forward(
    *,
    source: str,
    strategy: str,
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    original_bars: int,
    fee_bps: float,
    slippage_bps: float,
    spread_bps: float,
    position_size_pct: float,
    folds: int = 3,
    strategy_params: dict | None = None,
) -> dict:
    windows = _walk_forward_windows(df, folds)
    fold_results: list[dict] = []
    compounded = 1.0
    total_trades = 0.0
    weighted_wins = 0.0
    sharpes: list[float] = []
    worst_drawdown = 0.0

    for fold_number, (train_end, test_start, test_end) in enumerate(windows, start=1):
        history = df.iloc[:test_end].copy()
        history_signal = _run_source_on_frame(source, history, strategy_params)

        # Include the final training bar as a seed so its completed-bar signal may
        # execute at the first test-bar open without exposing any future bars.
        evaluation_start = max(0, test_start - 1)
        evaluation = df.iloc[evaluation_start:test_end].copy()
        evaluation_signal = history_signal.reindex(evaluation.index).fillna(0.0)
        metrics = _account_backtest(
            df=evaluation,
            sig=evaluation_signal,
            timeframe=timeframe,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            spread_bps=spread_bps,
            position_size_pct=position_size_pct,
        )

        fold_return = float(metrics["Total Return [%]"])
        fold_trades = float(metrics["Number of Trades"])
        fold_win_rate = float(metrics["Win Rate [%]"])
        compounded *= 1.0 + fold_return / 100.0
        total_trades += fold_trades
        weighted_wins += fold_trades * fold_win_rate / 100.0
        sharpes.append(float(metrics["Sharpe"]))
        worst_drawdown = min(worst_drawdown, float(metrics["Max Drawdown [%]"]))

        fold_results.append(
            {
                "Fold": fold_number,
                "Train Bars": train_end,
                "Train Start": _timestamp(df.index[0]),
                "Train End": _timestamp(df.index[train_end - 1]),
                "Test Bars": test_end - test_start,
                "Test Start": _timestamp(df.index[test_start]),
                "Test End": _timestamp(df.index[test_end - 1]),
                **metrics,
            }
        )

    aggregate_win_rate = (weighted_wins / total_trades * 100.0) if total_trades > 0 else 0.0
    aggregate_return = (compounded - 1.0) * 100.0
    average_sharpe = statistics.mean(sharpes) if sharpes else 0.0
    positive_folds = sum(1 for item in fold_results if float(item["Total Return [%]"]) > 0.0)

    return {
        "strategy": strategy,
        "symbol": symbol,
        "timeframe": timeframe,
        "Validation Kind": "walk_forward",
        "Validation Selection": "Expanding chronological train windows with strictly forward test folds.",
        "Walk Forward Folds": len(fold_results),
        "Walk Forward Positive Folds": positive_folds,
        "Walk Forward Minimum Train Bars": _WALK_FORWARD_MIN_TRAIN_BARS,
        "Walk Forward Minimum Test Bars": _WALK_FORWARD_MIN_TEST_BARS,
        "Total Return [%]": round(aggregate_return, 4),
        "Number of Trades": round(total_trades, 4),
        "Win Rate [%]": round(aggregate_win_rate, 2),
        "Sharpe": round(float(average_sharpe), 3),
        "Max Drawdown [%]": round(worst_drawdown, 2),
        "Fetched Bars": original_bars,
        "Selected Bars": len(df),
        "Data Start": _timestamp(df.index[0]),
        "Data End": _timestamp(df.index[-1]),
        "Fold Results": fold_results,
        **_data_gap_diagnostics(df, timeframe),
    }


def optimize_strategy_source(
    *,
    source: str,
    strategy: str,
    symbol: str,
    timeframe: str = "M5",
    num_bars: int = 1500,
    param_grid: dict,
    objective: str = "Sharpe",
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    spread_bps: float = 0.0,
    position_size_pct: float = 100.0,
) -> dict:
    if not isinstance(param_grid, dict) or not param_grid:
        raise HTTPException(400, "param_grid must be a non-empty object of parameter lists.")

    keys: list[str] = []
    values: list[list] = []
    combinations = 1
    for key, candidates in param_grid.items():
        if not isinstance(key, str) or not key.strip():
            raise HTTPException(400, "param_grid keys must be non-empty strings.")
        if not isinstance(candidates, list) or not candidates:
            raise HTTPException(400, f"param_grid['{key}'] must be a non-empty list.")
        keys.append(key)
        values.append(candidates)
        combinations *= len(candidates)

    if combinations > _MAX_OPTIMIZATION_COMBOS:
        raise HTTPException(
            400,
            f"Optimization grid has {combinations} combinations; maximum is {_MAX_OPTIMIZATION_COMBOS}.",
        )

    raw_df, _ = data_fetcher.fetch_data(symbol, timeframe, num_bars)
    full_df = _prepare_market_frame(raw_df, timeframe)
    original_bars = len(full_df)
    development, _, development_meta = _validation_window(full_df, "development_backtest")
    holdout, _, holdout_meta = _validation_window(full_df, "out_of_sample")

    objective_key = str(objective or "Sharpe").strip().lower()
    if objective_key in {"return", "total return", "total_return", "total return [%]"}:
        metric_key = "Total Return [%]"
    elif objective_key in {"drawdown", "max drawdown", "max_drawdown", "max drawdown [%]"}:
        metric_key = "Max Drawdown [%]"
    else:
        metric_key = "Sharpe"

    results: list[dict] = []
    for candidate_values in itertools.product(*values):
        params = {key: value for key, value in zip(keys, candidate_values)}
        try:
            signal = _run_source_on_frame(source, development, params)
            metrics = _account_backtest(
                df=development,
                sig=signal,
                timeframe=timeframe,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                spread_bps=spread_bps,
                position_size_pct=position_size_pct,
            )
        except HTTPException:
            continue
        results.append(
            {
                "params": params,
                "Total Return [%]": metrics["Total Return [%]"],
                "Sharpe": metrics["Sharpe"],
                "Max Drawdown [%]": metrics["Max Drawdown [%]"],
                "Number of Trades": metrics["Number of Trades"],
            }
        )

    if not results:
        raise HTTPException(400, "Optimization produced no valid parameter combinations.")

    # All supported objective values are higher-is-better because drawdown is
    # represented as a negative percentage (e.g. -5% is better than -20%).
    results.sort(key=lambda item: float(item.get(metric_key, float("-inf"))), reverse=True)
    best_params = dict(results[0]["params"])

    development_signal = _run_source_on_frame(source, development, best_params)
    development_result = _backtest_result(
        strategy=strategy,
        symbol=symbol,
        timeframe=timeframe,
        df=development,
        sig=development_signal,
        original_bars=original_bars,
        validation_kind="development_backtest",
        validation_metadata=development_meta,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        position_size_pct=position_size_pct,
    )

    holdout_signal = _run_source_on_frame(source, holdout, best_params)
    holdout_result = _backtest_result(
        strategy=strategy,
        symbol=symbol,
        timeframe=timeframe,
        df=holdout,
        sig=holdout_signal,
        original_bars=original_bars,
        validation_kind="out_of_sample",
        validation_metadata=holdout_meta,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        position_size_pct=position_size_pct,
    )

    return {
        "strategy": strategy,
        "symbol": symbol,
        "timeframe": timeframe,
        "Optimization Objective": metric_key,
        "Optimization Dataset": "development_backtest (first 70%) only",
        "Holdout Used For Selection": False,
        "Optimization Combinations": len(results),
        "Best Parameters": best_params,
        "optimization_results": results,
        "development_result": development_result,
        "holdout_result": holdout_result,
    }


def _backtest_result(
    *,
    strategy: str,
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    sig: pd.Series,
    original_bars: int,
    validation_kind: str,
    validation_metadata: dict,
    fee_bps: float,
    slippage_bps: float,
    spread_bps: float,
    position_size_pct: float,
) -> dict:
    return {
        "strategy": strategy,
        "symbol": symbol,
        "timeframe": timeframe,
        **_account_backtest(
            df=df,
            sig=sig,
            timeframe=timeframe,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            spread_bps=spread_bps,
            position_size_pct=position_size_pct,
        ),
        **_data_gap_diagnostics(df, timeframe),
        "Validation Kind": validation_kind,
        **validation_metadata,
        "Data Start": _timestamp(df.index[0]),
        "Data End": _timestamp(df.index[-1]),
        "Selected Bars": len(df),
        "Fetched Bars": original_bars,
    }


def run_saved_strategy_backtest(
    *,
    strategy: str,
    symbol: str,
    timeframe: str = "M5",
    num_bars: int = 1500,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    spread_bps: float = 0.0,
    position_size_pct: float = 100.0,
    validation_kind: str = "development_backtest",
    walk_forward_folds: int = 3,
):
    root = Path("backend/strategies_generated")
    path = root / f"{strategy.lower()}.py"
    if not path.exists():
        raise HTTPException(404, f"Saved strategy '{strategy}' not found.")

    df, _ = data_fetcher.fetch_data(symbol, timeframe, num_bars)
    df = _prepare_market_frame(df, timeframe)
    original_bars = len(df)

    try:
        src = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(400, f"Unable to read strategy source: {exc}") from exc

    requested_kind = str(validation_kind or "development_backtest").strip().lower()
    if requested_kind == "walk_forward":
        return run_strategy_walk_forward(
            source=src,
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            df=df,
            original_bars=original_bars,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            spread_bps=spread_bps,
            position_size_pct=position_size_pct,
            folds=walk_forward_folds,
        )

    df, validation_kind, validation_metadata = _validation_window(df, requested_kind)
    sig = _run_source_on_frame(src, df)
    result = _backtest_result(
        strategy=strategy,
        symbol=symbol,
        timeframe=timeframe,
        df=df,
        sig=sig,
        original_bars=original_bars,
        validation_kind=validation_kind,
        validation_metadata=validation_metadata,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        position_size_pct=position_size_pct,
    )
    lifecycle = record_backtest(
        strategy=strategy,
        source=src,
        metrics=result,
        validation_kind=validation_kind,
        context={
            "symbol": symbol.upper(),
            "timeframe": timeframe.upper(),
            "data_start": result["Data Start"],
            "data_end": result["Data End"],
            "selected_bars": len(df),
            "validation_selection": result["Validation Selection"],
            "split_timestamp": result["Split Timestamp"],
            "execution_timing": result["Execution Timing"],
            "position_size_pct": result["Position Size [%]"],
        },
    )
    result["Lifecycle"] = lifecycle.model_dump(mode="json")
    return result


def run_strategy_code_backtest(
    *,
    code: str,
    symbol: str,
    timeframe: str = "M5",
    num_bars: int = 1500,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    spread_bps: float = 0.0,
    position_size_pct: float = 100.0,
    strategy_name: str = "draft",
    validation_kind: str = "development_backtest",
    walk_forward_folds: int = 3,
):
    df, _ = data_fetcher.fetch_data(symbol, timeframe, num_bars)
    df = _prepare_market_frame(df, timeframe)
    original_bars = len(df)
    requested_kind = str(validation_kind or "development_backtest").strip().lower()

    if requested_kind == "walk_forward":
        result = run_strategy_walk_forward(
            source=str(code),
            strategy=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            df=df,
            original_bars=original_bars,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            spread_bps=spread_bps,
            position_size_pct=position_size_pct,
            folds=walk_forward_folds,
        )
        result["draft"] = True
        return result

    df, validation_kind, validation_metadata = _validation_window(df, requested_kind)
    sig = _run_source_on_frame(str(code), df)
    result = _backtest_result(
        strategy=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        df=df,
        sig=sig,
        original_bars=original_bars,
        validation_kind=validation_kind,
        validation_metadata=validation_metadata,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        position_size_pct=position_size_pct,
    )
    result["draft"] = True
    return result
