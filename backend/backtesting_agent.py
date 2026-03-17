from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import inspect
import itertools
import os
import pandas as pd
import numpy as np

from . import data_fetcher
from . import optimizer_utils

@dataclass
class BacktestParams:
    symbol: str
    timeframe: str = "M5"
    num_bars: int = 1500
    strategy_name: str = "sma" # default

def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min()) if len(dd) else 0.0


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder-style EWM smoothing (alpha=1/period)."""
    period_i = int(period) if period is not None else 14
    period_i = max(2, period_i)

    s = pd.to_numeric(series, errors="coerce").astype(float)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1 / period_i, adjust=False, min_periods=period_i).mean()
    avg_loss = loss.ewm(alpha=1 / period_i, adjust=False, min_periods=period_i).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out


def _signals_rsi(df: pd.DataFrame, period: int = 14, lower: float = 30.0, upper: float = 70.0) -> pd.Series:
    """Mean-reversion RSI targets: +1 below lower, -1 above upper, else hold prior."""
    if df is None or df.empty:
        return pd.Series([], dtype=float)

    try:
        period_i = max(2, int(period))
    except Exception:
        period_i = 14
    try:
        lower_f = float(lower)
        upper_f = float(upper)
    except Exception:
        lower_f, upper_f = 30.0, 70.0
    if not (lower_f < upper_f):
        lower_f, upper_f = 30.0, 70.0

    r = _rsi(df["close"], period_i)
    sig = pd.Series(np.nan, index=df.index, dtype=float)
    sig[r < lower_f] = 1.0
    sig[r > upper_f] = -1.0
    return sig.ffill().fillna(0.0)


def _fmt(x: Any, decimals: int = 2) -> str:
    try:
        v = float(x)
        if not np.isfinite(v):
            return "n/a"
        return f"{v:.{decimals}f}"
    except Exception:
        return "n/a"

def run_backtest(params: BacktestParams, extra_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Backtest using VectorBT. Supports single run and parameter optimization.
    """
    if extra_params is None:
        extra_params = {}

    # Fetch data
    df, _ = data_fetcher.fetch_data(params.symbol, params.timeframe, params.num_bars)
    if df is None or df.empty:
        return {"error": "No data found"}
        
    try:
        import vectorbt as vbt
    except ImportError:
        return {"error": "vectorbt not installed"}

    close = df["close"].astype(float)

    # Normalize timeframe to a pandas freq string for VectorBT
    freq_map = {
        "M1": "1min",
        "M5": "5min",
        "M15": "15min",
        "M30": "30min",
        "H1": "1h",
        "H4": "4h",
        "D1": "1d",
        "W1": "1w",
    }
    pd_freq = freq_map.get((params.timeframe or "M5").upper(), "5min")

    # Normalize costs (support bps inputs from UI)
    def _cost_fraction(dec_key: str, bps_key: str) -> float:
        if dec_key in extra_params:
            try:
                return float(extra_params.get(dec_key) or 0.0)
            except Exception:
                return 0.0
        if bps_key in extra_params:
            try:
                return float(extra_params.get(bps_key) or 0.0) / 10_000.0
            except Exception:
                return 0.0
        return 0.0

    fees = _cost_fraction("fees", "fee_bps")
    slippage = _cost_fraction("slippage", "slippage_bps")
    
    # 1. Determine Strategy Logic
    # Simple default: SMA crossover if nothing else specified or implmented
    
    is_optimization = any(isinstance(v, list) and len(v) > 1 for v in extra_params.values())

    def _filter_signal_kwargs(signals_fn, raw: Dict[str, Any], *, keep_lists: bool) -> Dict[str, Any]:
        """Filter extra params down to only those accepted by signals(df, ...).

        Also drops engine keys (fees/slippage/etc) that should never be passed into strategies.
        """
        if not callable(signals_fn) or not isinstance(raw, dict) or not raw:
            return {}

        reserved = {
            "fees",
            "fee_bps",
            "slippage",
            "slippage_bps",
            "symbol",
            "symbols",
            "timeframe",
            "timeframes",
            "num_bars",
            "strategy",
            "strategy_name",
            "objective",
            "metric",
        }

        try:
            sig = inspect.signature(signals_fn)
            params = list(sig.parameters.values())
            has_var_kw = any(p.kind == p.VAR_KEYWORD for p in params)

            allowed = set()
            for idx, p in enumerate(params):
                if idx == 0:
                    # Assume first arg is the dataframe
                    continue
                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
                    allowed.add(p.name)

            out: Dict[str, Any] = {}
            for k, v in raw.items():
                if k in reserved:
                    continue
                if not (has_var_kw or k in allowed):
                    continue
                if isinstance(v, list) and not keep_lists:
                    out[k] = v[0] if v else None
                else:
                    out[k] = v
            return out
        except Exception:
            # Fallback: do not block execution if signature inspection fails
            out = {}
            for k, v in raw.items():
                if k in reserved:
                    continue
                if isinstance(v, list) and not keep_lists:
                    out[k] = v[0] if v else None
                else:
                    out[k] = v
            return out

    def _objective_sort(objective: Any):
        obj = (objective or "").strip().lower()
        if obj in ("return", "total return", "total_return", "total return [%]"):
            return ("Total Return [%]", True)
        if obj in ("drawdown", "max drawdown", "max_drawdown", "max drawdown [%]"):
            return ("Max Drawdown [%]", False)
        return ("Sharpe Ratio", True)

    # Load custom strategy once (used for both single-run and optimization)
    requested_strategy = (params.strategy_name or "").strip().lower()
    signals_fn = None
    if requested_strategy and requested_strategy != "sma":
        if requested_strategy == "rsi":
            # Built-in RSI strategy (no saved file needed)
            signals_fn = _signals_rsi
            # Accept "rsi" keyword as alias for period, e.g. "rsi 14"
            if isinstance(extra_params, dict) and "period" not in extra_params and "rsi" in extra_params:
                extra_params["period"] = extra_params.get("rsi")
        else:
            import importlib.util
            from pathlib import Path

            strat_path = Path("backend/strategies_generated") / f"{requested_strategy}.py"
            if not strat_path.exists():
                return {"error": f"Saved strategy '{requested_strategy}' not found."}
            try:
                spec = importlib.util.spec_from_file_location(f"strategies_generated.{requested_strategy}", strat_path)
                if spec is None or spec.loader is None:
                    return {"error": f"Unable to load strategy '{requested_strategy}'."}
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            except Exception as e:
                return {"error": f"Error loading strategy '{requested_strategy}': {e}"}

            signals_fn = getattr(mod, "signals", None)
            if not callable(signals_fn):
                return {"error": f"Strategy '{requested_strategy}' does not define signals(df, ...) for backtesting."}
    
    if is_optimization:
        # Optimization for custom saved strategy (grid search over kwargs)
        if signals_fn is not None:
            kwargs_raw = _filter_signal_kwargs(signals_fn, extra_params, keep_lists=True)
            opt_keys = [k for k, v in kwargs_raw.items() if isinstance(v, list) and len(v) > 1]
            if opt_keys:
                # Guardrail: cap total combinations to avoid freezing the API
                try:
                    max_combos = int(os.getenv("OPT_MAX_COMBOS", "200"))
                except Exception:
                    max_combos = 200

                combos = 1
                for k in opt_keys:
                    combos *= max(1, len(kwargs_raw[k]))
                if combos > max_combos:
                    return {
                        "error": (
                            f"Optimization grid too large ({combos} combos). "
                            f"Reduce ranges/steps or increase OPT_MAX_COMBOS (currently {max_combos})."
                        )
                    }

                objective_key, maximize = _objective_sort(extra_params.get("objective") or extra_params.get("metric"))

                static_kwargs = {k: v for k, v in kwargs_raw.items() if k not in opt_keys}
                results: List[Dict[str, Any]] = []

                for values in itertools.product(*[kwargs_raw[k] for k in opt_keys]):
                    call_kwargs = dict(static_kwargs)
                    for k, v in zip(opt_keys, values):
                        call_kwargs[k] = v

                    try:
                        signal = signals_fn(df, **call_kwargs)
                    except Exception:
                        continue

                    if not isinstance(signal, pd.Series):
                        continue

                    try:
                        signal = signal.reindex(close.index).fillna(0.0).astype(float)
                    except Exception:
                        continue

                    pf = vbt.Portfolio.from_orders(
                        close=close,
                        size=signal,
                        size_type="targetpercent",
                        init_cash=10000.0,
                        fees=fees,
                        slippage=slippage,
                        freq=pd_freq,
                    )

                    try:
                        total_return = float(pf.total_return() * 100.0)
                    except Exception:
                        total_return = float("nan")
                    try:
                        sharpe = float(pf.sharpe_ratio())
                    except Exception:
                        sharpe = float("nan")
                    try:
                        drawdown = float(pf.max_drawdown() * 100.0)
                    except Exception:
                        drawdown = float("nan")
                    try:
                        drawdown = abs(drawdown)
                    except Exception:
                        pass

                    rec: Dict[str, Any] = {
                        "params": ", ".join(f"{k}={call_kwargs[k]}" for k in opt_keys),
                        "Total Return [%]": total_return,
                        "Sharpe Ratio": sharpe,
                        "Max Drawdown [%]": drawdown,
                    }
                    for k in opt_keys:
                        rec[k] = call_kwargs[k]
                    results.append(rec)

                if not results:
                    return {"error": "Optimization produced no valid results."}

                def _val(rec: Dict[str, Any]) -> float:
                    try:
                        v = float(rec.get(objective_key))
                        if np.isnan(v) or np.isinf(v):
                            return float("-inf") if maximize else float("inf")
                        return v
                    except Exception:
                        return float("-inf") if maximize else float("inf")

                results.sort(key=_val, reverse=maximize)
                best = results[0]

                best_extra = dict(extra_params)
                for k in opt_keys:
                    best_extra[k] = best.get(k)
                # Ensure no remaining list params trigger recursion back into optimization
                for k, v in list(best_extra.items()):
                    if isinstance(v, list):
                        best_extra[k] = v[0] if v else None

                best_run = run_backtest(params, best_extra)
                if isinstance(best_run, dict) and best_run.get("error"):
                    return best_run
                best_run["optimization_results"] = results
                best_run["best_result"] = best
                # Prefer metrics from the re-run (matches dashboard/report); keep grid metrics for ranking/table.
                try:
                    rerun_metrics = best_run.get("metrics") if isinstance(best_run, dict) else None
                    if isinstance(rerun_metrics, dict):
                        for k in ("Total Return [%]", "Sharpe Ratio", "Max Drawdown [%]"):
                            if k in rerun_metrics:
                                best[k] = rerun_metrics[k]
                except Exception:
                    pass
                dd = (best_run.get("metrics") or {}).get("Max Drawdown [%]") if isinstance(best_run, dict) else None
                if dd is None:
                    dd = best.get("Max Drawdown [%]")
                try:
                    dd = abs(float(dd))
                except Exception:
                    pass
                best_desc = ", ".join(f"{k}={best.get(k)}" for k in opt_keys)
                tr_val = (best_run.get("metrics") or {}).get("Total Return [%]") if isinstance(best_run, dict) else None
                sr_val = (best_run.get("metrics") or {}).get("Sharpe Ratio") if isinstance(best_run, dict) else None
                if sr_val is None and isinstance(best_run, dict):
                    sr_val = (best_run.get("metrics") or {}).get("Sharpe")
                best_run["message"] = (
                    f"Optimization complete: {len(results)} combos. Best {best_desc} | "
                    f"Return {_fmt(tr_val)}% | Sharpe {_fmt(sr_val)} | "
                    f"MaxDD {_fmt(dd)}%."
                )
                return best_run

        if "fast" in extra_params and "slow" in extra_params:
            fast_windows = extra_params["fast"]
            slow_windows = extra_params["slow"]

            if not isinstance(fast_windows, list) or not isinstance(slow_windows, list):
                return {"error": "Optimization requires lists for both fast and slow windows."}

            try:
                fast_windows = [max(2, int(v)) for v in fast_windows]
                slow_windows = [max(2, int(v)) for v in slow_windows]
            except Exception:
                return {"error": "Invalid fast/slow window values for optimization."}

            if len(fast_windows) < 2 or len(slow_windows) < 2:
                return {"error": "Optimization requires at least 2 values for fast and slow."}

            fast_ma = vbt.MA.run(close, window=fast_windows, short_name="fast")
            slow_ma = vbt.MA.run(close, window=slow_windows, short_name="slow")

            # VectorBT cannot directly align two MAs with different parameter indexes
            # (fast_window vs slow_window). Instead of building a full Portfolio for
            # every combination (slow), compute a fast vectorized approximation to
            # rank combos, then run a full backtest for the best one.
            fast_arr = fast_ma.ma.to_numpy()  # (n_bars, n_fast)
            slow_arr = slow_ma.ma.to_numpy()  # (n_bars, n_slow)
            n_bars = fast_arr.shape[0]
            n_fast = fast_arr.shape[1]
            n_slow = slow_arr.shape[1]

            fast3 = fast_arr[:, :, None]
            slow3 = slow_arr[:, None, :]
            valid = np.isfinite(fast3) & np.isfinite(slow3)
            pos3 = np.where(valid, np.where(fast3 > slow3, 1.0, -1.0), 0.0)  # (n_bars, n_fast, n_slow)
            pos2 = pos3.reshape(n_bars, n_fast * n_slow)

            # Apply previous bar position to current returns
            ret = close.pct_change().fillna(0.0).to_numpy().reshape(-1, 1)
            pos_prev = np.vstack([np.zeros((1, pos2.shape[1])), pos2[:-1]])
            strat_ret = ret * pos_prev

            # Equity curve per combo (normalized)
            equity = np.cumprod(1.0 + strat_ret, axis=0)
            total_return = (equity[-1, :] - 1.0) * 100.0

            # Max drawdown per combo (positive %)
            peak = np.maximum.accumulate(equity, axis=0)
            dd = (equity / np.where(peak == 0, 1.0, peak)) - 1.0
            max_dd = np.abs(np.min(dd, axis=0) * 100.0)

            # Sharpe (annualized, approximate)
            mu = np.mean(strat_ret, axis=0)
            sd = np.std(strat_ret, axis=0, ddof=0)
            sharpe = np.where(sd > 0, mu / sd, 0.0)

            tf_u = (params.timeframe or "M5").upper()
            annual_bars_map = {
                "M1": 252 * 1440,
                "M5": 252 * 288,
                "M15": 252 * 96,
                "M30": 252 * 48,
                "H1": 252 * 24,
                "H4": 252 * 6,
                "D1": 252,
                "W1": 52,
            }
            annual_bars = float(annual_bars_map.get(tf_u, 252 * 288))
            sharpe = sharpe * (annual_bars ** 0.5)

            fast_grid = np.repeat(np.array(fast_windows, dtype=int), n_slow)
            slow_grid = np.tile(np.array(slow_windows, dtype=int), n_fast)
            mask = fast_grid < slow_grid
            if not mask.any():
                return {"error": "No valid fast/slow combinations (fast must be < slow)."}

            results = []
            for f, s, tr, sh, ddv in zip(
                fast_grid[mask],
                slow_grid[mask],
                total_return[mask],
                sharpe[mask],
                max_dd[mask],
            ):
                results.append(
                    {
                        "params": f"({int(f)}, {int(s)})",
                        "Total Return [%]": float(tr),
                        "Sharpe Ratio": float(sh),
                        "Max Drawdown [%]": float(ddv),
                        "Fast": int(f),
                        "Slow": int(s),
                    }
                )
                
            objective_key, maximize = _objective_sort(extra_params.get("objective") or extra_params.get("metric"))

            def _val(rec: Dict[str, Any]) -> float:
                try:
                    v = float(rec.get(objective_key))
                    if np.isnan(v) or np.isinf(v):
                        return float("-inf") if maximize else float("inf")
                    return v
                except Exception:
                    return float("-inf") if maximize else float("inf")

            results.sort(key=_val, reverse=maximize)
            best = results[0] if results else {}

            # Re-run best combination to provide full charts/metrics payload
            best_extra = dict(extra_params)
            if best.get("Fast") is not None:
                best_extra["fast"] = int(best["Fast"])
            if best.get("Slow") is not None:
                best_extra["slow"] = int(best["Slow"])
            for k, v in list(best_extra.items()):
                if isinstance(v, list):
                    best_extra[k] = v[0] if v else None

            best_run = run_backtest(params, best_extra)
            if isinstance(best_run, dict) and best_run.get("error"):
                return best_run
            best_run["optimization_results"] = results
            best_run["best_result"] = best
            # Prefer metrics from the re-run (matches dashboard/report); keep grid metrics for ranking/table.
            try:
                rerun_metrics = best_run.get("metrics") if isinstance(best_run, dict) else None
                if isinstance(rerun_metrics, dict):
                    for k in ("Total Return [%]", "Sharpe Ratio", "Max Drawdown [%]"):
                        if k in rerun_metrics:
                            best[k] = rerun_metrics[k]
            except Exception:
                pass
            dd = (best_run.get("metrics") or {}).get("Max Drawdown [%]") if isinstance(best_run, dict) else None
            if dd is None:
                dd = best.get("Max Drawdown [%]")
            try:
                dd = abs(float(dd))
            except Exception:
                pass
            tr_val = (best_run.get("metrics") or {}).get("Total Return [%]") if isinstance(best_run, dict) else None
            sr_val = (best_run.get("metrics") or {}).get("Sharpe Ratio") if isinstance(best_run, dict) else None
            if sr_val is None and isinstance(best_run, dict):
                sr_val = (best_run.get("metrics") or {}).get("Sharpe")
            best_run["message"] = (
                f"Optimization complete: {len(results)} combos. Best Fast={best.get('Fast')} Slow={best.get('Slow')} | "
                f"Return {_fmt(tr_val)}% | Sharpe {_fmt(sr_val)} | "
                f"MaxDD {_fmt(dd)}%."
            )
            return best_run

    # --- Single Run Logic ---
    # --- Single Run Logic ---
    # --- Single Run Logic ---
    signal = None

    # Check for custom strategy file first
    if signals_fn is not None:
        try:
            kwargs = _filter_signal_kwargs(signals_fn, extra_params, keep_lists=False)
            signal = signals_fn(df, **kwargs) if kwargs else signals_fn(df)
        except Exception as e:
            return {"error": f"signals(df, **params) failed for '{requested_strategy}': {e}"}
        if not isinstance(signal, pd.Series):
            return {"error": "signals(df) must return a pandas Series aligned to df.index."}
        try:
            signal = signal.reindex(close.index).fillna(0.0).astype(float)
        except Exception:
            return {"error": "signals(df) output not aligned to input index."}

    # Fallback to SMA if no custom strategy or failed
    if signal is None:
        fast = int(extra_params.get("fast", 50)) if not isinstance(extra_params.get("fast"), list) else extra_params["fast"][0]
        slow = int(extra_params.get("slow", 200)) if not isinstance(extra_params.get("slow"), list) else extra_params["slow"][0]

        fast_ma = vbt.MA.run(close, window=fast)
        slow_ma = vbt.MA.run(close, window=slow)
        # Convert crossover to target signal (1=Long, -1=Short) without holding neutral?
        # Standard SMA strategy often stays in market. 
        # using entries/exits logic to construct signal:
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        
        # Build a signal series: 1 where long, -1 where short, carry forward
        # vbt.Portfolio.from_signals does this implicitly, but for from_orders we need explicit targets
        # Simple approach: +1 when fast>slow, -1 when fast<slow
        signal = np.where(fast_ma.ma_above(slow_ma), 1.0, -1.0)
        signal = pd.Series(signal, index=close.index)

    # Clean signal to ensure -1, 0, 1
    # Some strategies might return other values, clamp them or assume they are proper targets
    signal = signal.fillna(0.0).astype(float)
    
    # Run Backtest with Target Exposure
    # size=signal: -1 = 100% Short, 0 = Cash, +1 = 100% Long
    pf = vbt.Portfolio.from_orders(
        close=close,
        size=signal,
        size_type='targetpercent',
        init_cash=10000.0,
        fees=fees,
        slippage=slippage,
        freq=pd_freq
    )
    
    # Detailed Data for Charts
    equity = pf.value()
    equity_ser = [{"time": str(t), "value": float(v)} for t, v in equity.items()]
    
    trades = pf.trades.records_readable
    trade_list = []
    if not trades.empty:
        for _, row in trades.iterrows():
            trade_list.append({
                "entry_time": str(row["Entry Timestamp"]),
                "exit_time": str(row["Exit Timestamp"]),
                "entry_price": float(row["Avg Entry Price"]),
                "exit_price": float(row["Avg Exit Price"]),
                "pnl": float(row["PnL"]),
                "return_pct": float(row["Return"] * 100),
                "direction": str(row["Direction"])
            })

    # Stats - Convert all to string or float for JSON serialization
    try:
        raw_stats = pf.stats()
        # Filter and clean
        metrics = {}
        for k, v in raw_stats.items():
            if isinstance(v, (pd.Timestamp, pd.Timedelta)):
                metrics[str(k)] = str(v)
            elif pd.isna(v):
                metrics[str(k)] = 0.0
            else:
                metrics[str(k)] = float(v) if isinstance(v, (int, float, np.number)) else str(v)
    except Exception as e:
        print(f"Stats generation error: {e}")
        metrics = {"Error": str(e)}

    # Return candle data for visualization
    candles = []
    for t, r in df.iterrows():
        candles.append({
            "time": str(t),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": float(r.get("volume", 0))
        })
    
    # Generate Plotly Figure JSON
    plots_json = None
    try:
        # Create the standard 3-pane plot
        fig = pf.plot()
        # Make it lighter? No, keep it standard for now.
        plots_json = fig.to_json()
    except Exception as e:
        print(f"Plot generation error: {e}")

    msg = "Backtest complete."
    try:
        tr = metrics.get("Total Return [%]")
        sr = metrics.get("Sharpe Ratio") or metrics.get("Sharpe")
        dd = metrics.get("Max Drawdown [%]")
        msg = f"Backtest complete. Return {_fmt(tr)}% | Sharpe {_fmt(sr)} | MaxDD {_fmt(dd)}%."
    except Exception:
        pass

    return {
        "message": msg,
        "metrics": metrics,
        "equity": equity_ser,
        "trades": trade_list,
        "candles": candles,
        "plots": plots_json,
    }
