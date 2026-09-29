from __future__ import annotations

import math
import statistics
from datetime import timedelta
from typing import Any

import pandas as pd
from fastapi import HTTPException

from backend import data_fetcher
from backend.domain.models import StrategyAnalysis
from backend.strategies.registry import get_strategy, list_strategies
from backend.services.studio_backtests import (
    _daily_metrics,
    _data_gap_diagnostics,
    _position_fraction,
    _prepare_market_frame,
    _safe_cost_bps,
    _timeframe_minutes,
    _timestamp,
    _validation_window,
)

_RUNTIME_STRATEGIES = {"sma_cross", "rsi_reversal", "breakout"}
_MIN_REGIME_BARS = 50
_REGIME_SEGMENTS = 3


def _direction(signal: str) -> float:
    value = str(signal or "no_trade").strip().lower()
    if value == "long":
        return 1.0
    if value == "short":
        return -1.0
    return 0.0


def _analysis_series(
    *,
    strategy_key: str,
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    params: dict[str, Any],
) -> list[StrategyAnalysis]:
    strategy = get_strategy(strategy_key)
    analyses: list[StrategyAnalysis] = []
    for bar_index in range(len(df)):
        history = df.iloc[: bar_index + 1]
        analyses.append(
            strategy.analyze(
                df=history,
                symbol=symbol.upper(),
                timeframe=timeframe.upper(),
                params=params,
            )
        )
    return analyses


def _protective_exit(
    direction: float,
    price: float,
    stop_loss: float | None,
    take_profit: float | None,
) -> tuple[str | None, float | None]:
    if direction > 0:
        if stop_loss is not None and price <= float(stop_loss):
            return "stop_loss", float(stop_loss)
        if take_profit is not None and price >= float(take_profit):
            return "take_profit", float(take_profit)
    elif direction < 0:
        if stop_loss is not None and price >= float(stop_loss):
            return "stop_loss", float(stop_loss)
        if take_profit is not None and price <= float(take_profit):
            return "take_profit", float(take_profit)
    return None, None


def _runtime_accounting(
    *,
    df: pd.DataFrame,
    analyses: list[StrategyAnalysis],
    start_bar: int,
    end_bar: int,
    timeframe: str,
    fee_bps: float,
    slippage_bps: float,
    spread_bps: float,
    position_size_pct: float,
) -> dict[str, Any]:
    if start_bar < 0 or end_bar > len(df) or end_bar - start_bar < 2:
        raise HTTPException(400, "Runtime strategy validation window must contain at least 2 bars.")
    if len(analyses) != len(df):
        raise HTTPException(500, "Runtime strategy analysis series is not aligned to market data.")

    fee = _safe_cost_bps(fee_bps, "fee_bps")
    slippage = _safe_cost_bps(slippage_bps, "slippage_bps")
    spread = _safe_cost_bps(spread_bps, "spread_bps")
    position_fraction = _position_fraction(position_size_pct)
    timeframe_minutes = int(_timeframe_minutes(timeframe))
    transaction_cost_fraction = (fee + slippage + spread / 2.0) / 10_000.0

    equity = 1.0
    equity_points: list[float] = [equity]
    equity_index: list[pd.Timestamp] = [pd.Timestamp(df.index[start_bar])]
    active_trade: dict[str, Any] | None = None
    trade_returns: list[float] = []
    hold_bars: list[int] = []
    hold_minutes: list[float] = []
    closed_trades = 0
    marked_open_trades = 0
    protective_exits = 0
    signal_flips = 0
    stop_exits = 0
    take_exits = 0

    def mark_equity(price: float) -> float:
        if not active_trade:
            return equity
        return float(active_trade["entry_equity_after_cost"]) + float(active_trade["direction"]) * float(
            active_trade["units"]
        ) * (float(price) - float(active_trade["entry_price"]))

    def close_active(
        exit_price: float,
        exit_time: pd.Timestamp,
        exit_bar: int,
        *,
        marked: bool,
        reason: str,
    ) -> float:
        nonlocal active_trade, closed_trades, marked_open_trades, protective_exits, stop_exits, take_exits
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
            if reason in {"stop_loss", "take_profit"}:
                protective_exits += 1
                if reason == "stop_loss":
                    stop_exits += 1
                else:
                    take_exits += 1

        trade_returns.append((final_equity / pre_entry_equity) - 1.0 if pre_entry_equity > 0 else 0.0)
        hold_bars.append(int(bars_held))
        hold_minutes.append(max(0.0, (exit_time - entry_time).total_seconds() / 60.0))
        active_trade = None
        return final_equity

    def open_active(
        direction: float,
        entry_price: float,
        entry_time: pd.Timestamp,
        entry_bar: int,
        analysis: StrategyAnalysis,
    ) -> float:
        nonlocal active_trade
        pre_entry_equity = equity
        entry_notional = pre_entry_equity * position_fraction
        entry_cost = entry_notional * transaction_cost_fraction
        entry_equity_after_cost = pre_entry_equity - entry_cost
        active_trade = {
            "pre_entry_equity": pre_entry_equity,
            "entry_equity_after_cost": entry_equity_after_cost,
            "entry_price": float(entry_price),
            "entry_time": entry_time,
            "entry_bar": entry_bar,
            "direction": direction,
            "units": entry_notional / float(entry_price),
            "stop_loss": analysis.stop_loss,
            "take_profit": analysis.take_profit,
        }
        return entry_equity_after_cost

    pending = analyses[start_bar]

    for bar_index in range(start_bar + 1, end_bar):
        open_price = float(df["open"].iloc[bar_index])
        close_price = float(df["close"].iloc[bar_index])
        bar_time = pd.Timestamp(df.index[bar_index])
        bar_close_time = bar_time + timedelta(minutes=timeframe_minutes)

        if active_trade:
            equity = mark_equity(open_price)

        desired_direction = _direction(pending.signal)
        if desired_direction != 0.0:
            if active_trade and float(active_trade["direction"]) != desired_direction:
                equity = close_active(
                    open_price,
                    bar_time,
                    bar_index,
                    marked=False,
                    reason="signal_flip",
                )
                signal_flips += 1

            if active_trade:
                active_trade["stop_loss"] = pending.stop_loss
                active_trade["take_profit"] = pending.take_profit
            else:
                equity = open_active(
                    desired_direction,
                    open_price,
                    bar_time,
                    bar_index,
                    pending,
                )

            if active_trade:
                reason, fill_price = _protective_exit(
                    float(active_trade["direction"]),
                    open_price,
                    active_trade.get("stop_loss"),
                    active_trade.get("take_profit"),
                )
                if reason is not None and fill_price is not None:
                    equity = close_active(
                        fill_price,
                        bar_time,
                        bar_index,
                        marked=False,
                        reason=reason,
                    )

        # Runtime no_trade semantics are hold/do-nothing: an existing position
        # and its prior protection remain unchanged.
        if active_trade:
            equity = mark_equity(close_price)
            reason, fill_price = _protective_exit(
                float(active_trade["direction"]),
                close_price,
                active_trade.get("stop_loss"),
                active_trade.get("take_profit"),
            )
            if reason is not None and fill_price is not None:
                equity = close_active(
                    fill_price,
                    bar_close_time,
                    bar_index,
                    marked=False,
                    reason=reason,
                )

        equity_points.append(float(equity))
        equity_index.append(bar_time)
        pending = analyses[bar_index]

    if active_trade:
        final_bar = end_bar - 1
        final_time = pd.Timestamp(df.index[final_bar]) + timedelta(minutes=timeframe_minutes)
        equity = close_active(
            float(df["close"].iloc[final_bar]),
            final_time,
            final_bar,
            marked=True,
            reason="final_mark",
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
    expectancy = (statistics.mean(trade_returns) * 100.0) if trade_returns else 0.0

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
        "Expectancy [%]": round(expectancy, 4),
        "Avg Trade [%]": round(expectancy, 4),
        "Max Drawdown [%]": round(max_drawdown * 100.0, 2),
        "Sharpe": round(daily_sharpe, 3),
        "Daily Sharpe": round(daily_sharpe, 3),
        "Sharpe Basis": "UTC daily equity returns, annualized with sqrt(252).",
        "Avg Daily Return [%]": round(avg_daily_pct, 4),
        "Trade Sharpe": round(trade_sharpe, 3),
        "SQN": round(sqn, 3),
        "Trades/Day": round(trades_per_day, 3),
        "Observed Trading Days": float(observed_days),
        "Avg Hold [bars]": round(avg_hold_bars, 2),
        "Avg Hold [min]": round(avg_hold_minutes, 2),
        "Protective Exits": float(protective_exits),
        "Stop Exits": float(stop_exits),
        "Take-Profit Exits": float(take_exits),
        "Signal Flips": float(signal_flips),
        "Fees [bps]": round(fee, 3),
        "Slippage [bps]": round(slippage, 3),
        "Spread [bps]": round(spread, 3),
        "Cost Assumption": "Fee and slippage per transaction; half of quoted spread per transaction.",
        "Position Size [%]": round(position_fraction * 100.0, 3),
        "Execution Timing": "Completed-bar decision; execute at next bar open.",
        "Runtime no_trade Semantics": "Hold existing position and prior protection; do not flatten.",
        "Protection Semantics": "Check prior protection at next completed-bar close; same-direction decisions refresh protection for subsequent bars.",
    }


def _regime_windows(df: pd.DataFrame) -> list[dict[str, Any]]:
    if len(df) < _REGIME_SEGMENTS * _MIN_REGIME_BARS:
        raise HTTPException(
            400,
            f"Regime analysis requires at least {_REGIME_SEGMENTS * _MIN_REGIME_BARS} bars; received {len(df)}.",
        )

    base = len(df) // _REGIME_SEGMENTS
    raw: list[dict[str, Any]] = []
    for segment in range(_REGIME_SEGMENTS):
        start = segment * base
        end = len(df) if segment == _REGIME_SEGMENTS - 1 else (segment + 1) * base
        close = df["close"].iloc[start:end].astype(float)
        returns = close.pct_change().dropna()
        total_return = float(close.iloc[-1] / close.iloc[0] - 1.0) if len(close) >= 2 else 0.0
        bar_vol = float(returns.std(ddof=0)) if len(returns) >= 2 else 0.0
        trend_scale = bar_vol * math.sqrt(max(1, len(returns)))
        trend_score = total_return / trend_scale if trend_scale > 0 else 0.0
        if trend_score >= 0.75:
            trend = "uptrend"
        elif trend_score <= -0.75:
            trend = "downtrend"
        else:
            trend = "range"
        raw.append(
            {
                "start": start,
                "end": end,
                "trend": trend,
                "bar_vol": bar_vol,
                "total_return": total_return,
                "trend_score": trend_score,
            }
        )

    median_vol = statistics.median(item["bar_vol"] for item in raw)
    for item in raw:
        vol = float(item["bar_vol"])
        if median_vol <= 0:
            vol_label = "normal_vol"
        elif vol > median_vol * 1.25:
            vol_label = "high_vol"
        elif vol < median_vol * 0.75:
            vol_label = "low_vol"
        else:
            vol_label = "normal_vol"
        item["label"] = f"{item['trend']}_{vol_label}"
    return raw


def _sensitivity_cases(strategy_key: str, params: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    cases: list[tuple[str, dict[str, Any]]] = [("baseline", dict(params))]

    def add(label: str, **updates: Any) -> None:
        candidate = dict(params)
        candidate.update(updates)
        if candidate != params and all(candidate != existing for _, existing in cases):
            cases.append((label, candidate))

    if strategy_key == "sma_cross":
        fast = int(params.get("fast", 20))
        slow = int(params.get("slow", 50))
        rr = float(params.get("rr", 2.0))
        add("fast_lower", fast=max(2, int(round(fast * 0.75))))
        add("fast_higher", fast=max(2, min(slow - 1, int(round(fast * 1.25)))))
        add("slow_lower", slow=max(fast + 1, int(round(slow * 0.8))))
        add("slow_higher", slow=max(fast + 1, int(round(slow * 1.2))))
        add("rr_lower", rr=max(0.5, rr * 0.75))
        add("rr_higher", rr=rr * 1.25)
    elif strategy_key == "rsi_reversal":
        length = int(params.get("length", 14))
        lower = float(params.get("lower", 30.0))
        upper = float(params.get("upper", 70.0))
        rr = float(params.get("rr", 1.8))
        add("length_lower", length=max(2, int(round(length * 0.75))))
        add("length_higher", length=max(2, int(round(length * 1.25))))
        add("thresholds_wider", lower=max(1.0, lower - 5.0), upper=min(99.0, upper + 5.0))
        add("thresholds_tighter", lower=min(upper - 1.0, lower + 5.0), upper=max(lower + 1.0, upper - 5.0))
        add("rr_lower", rr=max(0.5, rr * 0.75))
        add("rr_higher", rr=rr * 1.25)
    elif strategy_key == "breakout":
        lookback = int(params.get("lookback", 20))
        rr = float(params.get("rr", 2.2))
        add("lookback_lower", lookback=max(5, int(round(lookback * 0.6))))
        add("lookback_higher", lookback=max(5, int(round(lookback * 1.5))))
        add("rr_lower", rr=max(0.5, rr * 0.75))
        add("rr_higher", rr=rr * 1.25)

    return cases


def _audit_on_frame(
    *,
    strategy_key: str,
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    params: dict[str, Any] | None,
    fee_bps: float,
    slippage_bps: float,
    spread_bps: float,
    position_size_pct: float,
) -> dict[str, Any]:
    normalized = str(strategy_key or "").strip().lower()
    if normalized not in _RUNTIME_STRATEGIES:
        raise HTTPException(400, f"Runtime strategy must be one of: {', '.join(sorted(_RUNTIME_STRATEGIES))}.")

    strategy = get_strategy(normalized)
    effective_params = {**strategy.parameters, **(params or {})}
    prepared = _prepare_market_frame(df, timeframe)
    development, _, development_meta = _validation_window(prepared, "development_backtest")
    holdout, _, holdout_meta = _validation_window(prepared, "out_of_sample")
    split = len(development)

    analyses = _analysis_series(
        strategy_key=normalized,
        df=prepared,
        symbol=symbol,
        timeframe=timeframe,
        params=effective_params,
    )

    full = _runtime_accounting(
        df=prepared,
        analyses=analyses,
        start_bar=0,
        end_bar=len(prepared),
        timeframe=timeframe,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        position_size_pct=position_size_pct,
    )
    development_result = _runtime_accounting(
        df=prepared,
        analyses=analyses,
        start_bar=0,
        end_bar=split,
        timeframe=timeframe,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        position_size_pct=position_size_pct,
    )
    holdout_result = _runtime_accounting(
        df=prepared,
        analyses=analyses,
        start_bar=split,
        end_bar=len(prepared),
        timeframe=timeframe,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        position_size_pct=position_size_pct,
    )
    gross = _runtime_accounting(
        df=prepared,
        analyses=analyses,
        start_bar=0,
        end_bar=len(prepared),
        timeframe=timeframe,
        fee_bps=0.0,
        slippage_bps=0.0,
        spread_bps=0.0,
        position_size_pct=position_size_pct,
    )

    regimes: list[dict[str, Any]] = []
    for window in _regime_windows(prepared):
        metrics = _runtime_accounting(
            df=prepared,
            analyses=analyses,
            start_bar=int(window["start"]),
            end_bar=int(window["end"]),
            timeframe=timeframe,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            spread_bps=spread_bps,
            position_size_pct=position_size_pct,
        )
        regimes.append(
            {
                "Regime": window["label"],
                "Market Return [%]": round(float(window["total_return"]) * 100.0, 4),
                "Trend Score": round(float(window["trend_score"]), 3),
                "Bars": int(window["end"]) - int(window["start"]),
                "Data Start": _timestamp(prepared.index[int(window["start"])]),
                "Data End": _timestamp(prepared.index[int(window["end"]) - 1]),
                **metrics,
            }
        )

    sensitivity: list[dict[str, Any]] = []
    for label, candidate_params in _sensitivity_cases(normalized, effective_params):
        candidate_analyses = _analysis_series(
            strategy_key=normalized,
            df=prepared.iloc[:split].copy(),
            symbol=symbol,
            timeframe=timeframe,
            params=candidate_params,
        )
        candidate_metrics = _runtime_accounting(
            df=prepared.iloc[:split].copy(),
            analyses=candidate_analyses,
            start_bar=0,
            end_bar=split,
            timeframe=timeframe,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            spread_bps=spread_bps,
            position_size_pct=position_size_pct,
        )
        sensitivity.append(
            {
                "Case": label,
                "Parameters": candidate_params,
                "Total Return [%]": candidate_metrics["Total Return [%]"],
                "Number of Trades": candidate_metrics["Number of Trades"],
                "Expectancy [%]": candidate_metrics["Expectancy [%]"],
                "Max Drawdown [%]": candidate_metrics["Max Drawdown [%]"],
                "Sharpe": candidate_metrics["Sharpe"],
                "Trades/Day": candidate_metrics["Trades/Day"],
            }
        )

    return {
        "strategy": normalized,
        "strategy_label": strategy.label,
        "symbol": symbol.upper(),
        "timeframe": timeframe.upper(),
        "Parameters": effective_params,
        "Validation Model": "Trusted runtime StrategyV2 evaluated with completed-bar decisions and next-bar execution.",
        "Full Backtest": full,
        "Development": {
            **development_result,
            **development_meta,
            "Data Start": _timestamp(prepared.index[0]),
            "Data End": _timestamp(prepared.index[split - 1]),
        },
        "Out of Sample": {
            **holdout_result,
            **holdout_meta,
            "Data Start": _timestamp(prepared.index[split]),
            "Data End": _timestamp(prepared.index[-1]),
        },
        "Regime Analysis": regimes,
        "Costs": {
            "Gross Return [%]": gross["Total Return [%]"],
            "Net Return [%]": full["Total Return [%]"],
            "Cost Drag [%]": round(float(gross["Total Return [%]"]) - float(full["Total Return [%]"]), 4),
            "Fees [bps]": full["Fees [bps]"],
            "Slippage [bps]": full["Slippage [bps]"],
            "Spread [bps]": full["Spread [bps]"],
        },
        "Parameter Sensitivity": {
            "Dataset": "Development window (first 70%) only; no holdout parameter selection.",
            "Cases": sensitivity,
        },
        "Fetched Bars": len(prepared),
        "Data Start": _timestamp(prepared.index[0]),
        "Data End": _timestamp(prepared.index[-1]),
        **_data_gap_diagnostics(prepared, timeframe),
    }


def run_runtime_strategy_audit(
    *,
    strategy: str,
    symbol: str,
    timeframe: str = "M5",
    num_bars: int = 1500,
    params: dict[str, Any] | None = None,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    spread_bps: float = 0.0,
    position_size_pct: float = 100.0,
) -> dict[str, Any]:
    raw_df, _ = data_fetcher.fetch_data(symbol, timeframe, num_bars)
    if raw_df is None or raw_df.empty:
        raise HTTPException(503, f"No market data available for {symbol.upper()}:{timeframe.upper()}.")
    return _audit_on_frame(
        strategy_key=strategy,
        symbol=symbol,
        timeframe=timeframe,
        df=raw_df,
        params=params,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        position_size_pct=position_size_pct,
    )


def run_runtime_strategy_matrix(
    *,
    targets: list[tuple[str, str]],
    strategies: list[str] | None = None,
    num_bars: int = 1500,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    spread_bps: float = 0.0,
    position_size_pct: float = 100.0,
) -> dict[str, Any]:
    selected = [str(item).strip().lower() for item in (strategies or sorted(_RUNTIME_STRATEGIES))]
    unknown = [item for item in selected if item not in _RUNTIME_STRATEGIES]
    if unknown:
        raise HTTPException(400, f"Unknown runtime strategies: {', '.join(unknown)}.")
    if not targets:
        raise HTTPException(400, "At least one symbol/timeframe target is required.")

    results: list[dict[str, Any]] = []
    for symbol, timeframe in targets:
        raw_df, _ = data_fetcher.fetch_data(symbol, timeframe, num_bars)
        if raw_df is None or raw_df.empty:
            results.append(
                {
                    "symbol": symbol.upper(),
                    "timeframe": timeframe.upper(),
                    "status": "unavailable",
                    "error": "No market data available.",
                }
            )
            continue
        for strategy_key in selected:
            results.append(
                {
                    "status": "ok",
                    **_audit_on_frame(
                        strategy_key=strategy_key,
                        symbol=symbol,
                        timeframe=timeframe,
                        df=raw_df,
                        params=None,
                        fee_bps=fee_bps,
                        slippage_bps=slippage_bps,
                        spread_bps=spread_bps,
                        position_size_pct=position_size_pct,
                    ),
                }
            )

    return {
        "Strategies": selected,
        "Targets": [{"symbol": symbol.upper(), "timeframe": timeframe.upper()} for symbol, timeframe in targets],
        "Results": results,
    }


def runtime_strategy_keys() -> list[str]:
    registered = {strategy.key for strategy in list_strategies()}
    return sorted(_RUNTIME_STRATEGIES & registered)
