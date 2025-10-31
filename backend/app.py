# MONKEY-PATCH: Fix for Twisted dependency issue in ctrader-open-api
# The version of Twisted required by ctrader-open-api has a bug where
# CertificateOptions is not imported. We inject it here before it's used.
try:
    from twisted.internet import ssl, endpoints
    if not hasattr(endpoints, 'CertificateOptions'):
        endpoints.CertificateOptions = ssl.CertificateOptions
except ImportError:
    pass # If twisted isn't installed, app will fail later with a clearer error.

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import asyncio
import textwrap
import httpx
import os
import threading
import logging
import json
from dataclasses import asdict
from threading import Lock
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel, Field

# ── Our internal modules ──────────────────────────────────────────────────
import backend.ctrader_client as ctd
import backend.data_fetcher as data_fetcher
from backend.symbol_fetcher import get_available_symbols
from backend.indicators import add_indicators
from backend.llm_analyzer import (
    analyze_data_with_llm,
    MODEL_DEFAULT,
    _ollama_generate,
    TradeDecision,
)
from backend.smc_features import (
    detect_bos_choch,
    current_fvg,
    ob_near_price,
    in_premium_discount,
)
from backend.agent_state import recent_signals, all_task_status, reset_all_state
from backend.agent_controller import controller, AgentConfig
from backend.programmer_agent import ProgrammerAgent
from backend.backtesting_agent import run_backtest, BacktestParams
from backend.strategy import (
    get_strategy,
    available_strategies,
    load_generated_strategies,
    get_last_strategy_load_errors,
)
from backend.llm_analyzer import warm_ollama
from backend import web_search

from backend.journal import db as journal_db
from backend.journal.router import router as journal_router

# ──────────────────────────────────────────────────────────────────────────
# FastAPI app & middleware
# ──────────────────────────────────────────────────────────────────────────
app = FastAPI()

app.include_router(journal_router, prefix="/api/journal")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Env (LLM + defaults) ────────────────────────────────────────────────
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava:7b")
DEFAULT_SYMBOL = os.getenv("DEFAULT_SYMBOL", "XAUUSD")

def _clamp_int(x, lo, hi, default):
    try:
        v = int(x)
        return max(lo, min(hi, v))
    except Exception:
        return default


# ──────────────────────────────────────────────────────────────────────────
# Health & startup
# ──────────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "connected": ctd.is_connected()}


@app.on_event("startup")
async def on_startup():
    # Load any user-saved strategies
    try:
        cnt = load_generated_strategies()
        print(f"[startup] Loaded {cnt} generated strategies.")
    except Exception as e:
        print(f"[startup] Failed to load generated strategies: {e}")
    # Pre-load the LLM model to avoid cold-start timeouts on first use
    warm_ollama()
    await asyncio.sleep(5) # Give cTrader client time to connect
    await controller.start_from_config()


# cTrader TCP client — ensure we start it only once even with reloads
_ctrader_thread_lock = Lock()
_ctrader_thread_started = False


def _start_ctrader_once():
    global _ctrader_thread_started
    with _ctrader_thread_lock:
        if _ctrader_thread_started:
            return
        threading.Thread(target=ctd.init_client, daemon=True).start()
        _ctrader_thread_started = True


_start_ctrader_once()


# ──────────────────────────────────────────────────────────────────────────
# LLM status
# ──────────────────────────────────────────────────────────────────────────
@app.get("/api/llm_status")
async def llm_status():
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            t = await c.get(f"{OLLAMA_URL}/api/tags")
            return {"ollama": t.status_code, "model": OLLAMA_MODEL}
    except Exception as e:
        return {"ollama": "unreachable", "error": str(e)}


# ──────────────────────────────────────────────────────────────────────────
# Agent signals feed (UI panel)
# ──────────────────────────────────────────────────────────────────────────
@app.get("/api/agent/signals")
async def agent_signals(n: int = 50):
    return recent_signals(n)


@app.post("/api/agent/reset_state")
async def agent_reset_state():
    """Clear persisted signals, last-bar timestamps, and task status.

    Useful if the dashboard shows stale signals from a prior session, or
    when starting fresh before enabling the agent.
    """
    reset_all_state()
    return {"ok": True}


@app.post("/api/strategies/reload")
@app.get("/api/strategies/reload")
async def strategies_reload():
    try:
        cnt = load_generated_strategies()
        return {"ok": True, "loaded": cnt, "available": available_strategies(), "errors": get_last_strategy_load_errors()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/strategies")
async def strategies_list():
    return {"available": available_strategies(), "errors": get_last_strategy_load_errors()}


@app.get("/api/strategies/files")
async def strategies_files():
    """List .py files visible to the container under strategies_generated.

    Useful to debug host/container volume visibility issues.
    """
    try:
        root = Path("backend/strategies_generated")
        files = [str(p.name) for p in root.glob("*.py")] if root.exists() else []
        return {"files": files, "cwd": str(Path.cwd())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agent/status")
async def agent_status():
    snap = await controller.snapshot()
    cfg = snap.get("config") or {}
    running_pairs = snap.get("running_pairs") or []
    # Ensure tasks is always a list for the frontend
    tasks = all_task_status() or []

    enabled = bool(cfg.get("enabled"))
    watchlist = cfg.get("watchlist") or []
    return {
        "enabled": enabled,
        "running": enabled and len(running_pairs) > 0,
        "watchlist": watchlist,
        "interval_sec": cfg.get("interval_sec"),
        "min_confidence": cfg.get("min_confidence"),
        "trading_mode": cfg.get("trading_mode"),
        "autotrade": cfg.get("autotrade"),
        "lot_size_lots": cfg.get("lot_size_lots"),
        "strategy": cfg.get("strategy"),
        "running_pairs": running_pairs,
        "tasks": tasks,
        "available_strategies": available_strategies(),
    }


@app.get("/api/strategies/backtest")
async def strategies_backtest(strategy: str, symbol: str, timeframe: str = "M5", num_bars: int = 1500):
    """Backtest a saved strategy that exposes a vectorized `signals(df)` function.

    Notes:
    - Loads `backend/strategies_generated/<strategy>.py` and calls `signals(df)`.
    - Supports long/short via sign of the series: >0 long, <0 short, 0 flat.
    - LLM-based strategies with only `trade_decision` are not suitable for fast backtests.
    """
    from pathlib import Path
    import math
    import pandas as pd

    root = Path("backend/strategies_generated")
    path = root / f"{strategy.lower()}.py"
    if not path.exists():
        raise HTTPException(404, f"Saved strategy '{strategy}' not found.")

    df, _ = data_fetcher.fetch_data(symbol, timeframe, num_bars)
    if df is None or df.empty:
        raise HTTPException(404, "No data fetched for backtest")

    # Load module and obtain `signals(df)`
    try:
        src = path.read_text(encoding="utf-8")
        mod_globals = {}
        exec(src, mod_globals)
        signals_fn = mod_globals.get("signals")
    except Exception as e:
        raise HTTPException(400, f"Unable to load strategy module: {e}")
    if not callable(signals_fn):
        raise HTTPException(400, "This strategy does not define a callable signals(df) for backtesting.")

    try:
        sig = signals_fn(df)
    except Exception as e:
        raise HTTPException(400, f"signals(df) execution failed: {e}")

    if 'pandas' not in str(type(sig)):
        raise HTTPException(400, "signals(df) must return a pandas Series aligned to df index.")
    # Align and sanitize
    try:
        sig = sig.reindex(df.index).fillna(0)
    except Exception:
        raise HTTPException(400, "signals(df) output not aligned to input index.")

    close = df["close"].astype(float)
    rets = close.pct_change().fillna(0)
    # Position: 1 for long, -1 for short, 0 flat
    pos = sig.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
    pos = pos.ffill().fillna(0.0)
    strat_rets = rets * pos
    equity = (1.0 + strat_rets).cumprod()

    # Trades metrics from entry/exit transitions
    entries = (pos != 0) & (pos.shift(1).fillna(0) == 0)
    exits = (pos == 0) & (pos.shift(1).fillna(0) != 0)
    entries_idx = list(entries[entries].index)
    exits_idx = list(exits[exits].index)
    trade_rets: list[float] = []
    if entries_idx:
        import bisect
        exits_idx_sorted = list(sorted(exits_idx))
        for e in entries_idx:
            i = bisect.bisect_right(exits_idx_sorted, e)
            x = exits_idx_sorted[i] if i < len(exits_idx_sorted) else close.index[-1]
            if e >= x:
                continue
            # Sign-aware return
            direction = float(pos.loc[e])
            grow = (close.loc[x] / close.loc[e])
            trade_ret = (grow - 1.0) if direction > 0 else ((1.0 / grow) - 1.0)
            trade_rets.append(float(trade_ret))

    num_trades = len(trade_rets)
    wins = sum(1 for r in trade_rets if r > 0)
    win_rate = (wins / num_trades * 100.0) if num_trades > 0 else 0.0
    total_return = float(equity.iloc[-1] - 1.0) * 100.0
    avg_trade = (sum(trade_rets) / num_trades * 100.0) if num_trades > 0 else 0.0

    # Rough annualization assuming 5m bars; a UI control could refine this
    ann_factor = math.sqrt(252 * (24 * 60 / 5))
    sharpe = 0.0
    if strat_rets.std(ddof=0) > 0:
        sharpe = float(strat_rets.mean() / strat_rets.std(ddof=0)) * ann_factor

    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    mdd = float(dd.min()) if len(dd) else 0.0

    return {
        "strategy": strategy,
        "symbol": symbol,
        "timeframe": timeframe,
        "Total Return [%]": round(total_return, 4),
        "Number of Trades": float(num_trades),
        "Win Rate [%]": round(win_rate, 2),
        "Avg Trade [%]": round(avg_trade, 4),
        "Max Drawdown [%]": round(mdd * 100.0, 2),
        "Sharpe": round(sharpe, 3),
    }

@app.post("/api/agent/execute_task")
async def execute_task_endpoint(req: Request):
    """Compatibility endpoint for Strategy Studio tasks.

    Accepts task_type values like:
    - 'calculate_indicator' | 'backtest_strategy' | 'save_strategy' | 'research_strategy' | 'create_strategy'

    Maps them to controller types and normalizes the response to:
    { status: 'success'|'error', message?: str, result?: any }
    """
    body = await req.json()
    t = (body.get("task_type") or "").strip().lower()
    goal = body.get("goal")
    params = body.get("params") or {}

    mapped = (
        "indicator" if t in ("calculate_indicator",)
        else "backtest" if t in ("backtest_strategy",)
        else "strategy"
    )

    # Execute mapped task without relying on autonomous agents
    try:
        # Save Strategy: if explicit request and code provided, persist to file
        if t == "save_strategy":
            name = ((params or {}).get("strategy_name") or "").strip() if isinstance(params, dict) else ""
            code = ((params or {}).get("code") or "") if isinstance(params, dict) else ""
            if not name:
                return {"status": "error", "message": "strategy_name is required"}
            if not code:
                return {"status": "error", "message": "code is required"}
            # sanitize filename
            safe = "".join(ch if ch.isalnum() or ch in ("_","-") else "_" for ch in name)
            dest_dir = Path("backend/strategies_generated")
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / f"{safe}.py"
            code_norm = textwrap.dedent(str(code)).lstrip("\n").replace("\r\n", "\n")
            dest_path.write_text(code_norm, encoding="utf-8")
            # Auto-reload after save so it appears in dropdown without restart
            try:
                load_generated_strategies()
            except Exception:
                pass
            return {"status": "success", "message": f"Strategy saved as {dest_path}"}

        if mapped == "backtest":
            sym = (params.get("symbol") if isinstance(params, dict) else None) or DEFAULT_SYMBOL
            tf = (params.get("timeframe") if isinstance(params, dict) else None) or "M5"
            n = int((params.get("num_bars") if isinstance(params, dict) else 1500) or 1500)
            result = run_backtest(BacktestParams(symbol=sym, timeframe=tf, num_bars=n))
            return {"status": "success", "message": "Backtest complete.", "result": result}

        # strategy or indicator -> generate code snippet for the user
        pa = ProgrammerAgent()
        code = await pa.generate_code(goal or "", mapped)
        return {"status": "success", "message": "Code generated.", "result": {"stdout": code}}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ──────────────────────────────────────────────────────────────────────────
# Symbols & candles
# ──────────────────────────────────────────────────────────────────────────
@app.get("/api/symbols")
async def get_symbols():
    """Return the list of symbols and a safe default for the UI."""
    symbols = sorted(get_available_symbols() or [])  # e.g., ["XAUUSD", "US500", ...]
    default = DEFAULT_SYMBOL if DEFAULT_SYMBOL in symbols else (symbols[0] if symbols else None)
    return {"symbols": symbols, "default": default}


@app.get("/api/symbol_limits")
async def get_symbol_limits(symbol: Optional[str] = None):
    """Return per-symbol volume limits (min/step/max) in lots and API units.

    Lots are standard lots; API units are 0.0001-lot increments where 1.00 lot = 10,000.
    """
    def _one(sym_u: str):
        sid = ctd.symbol_name_to_id.get(sym_u)
        if not sid:
            return None
        min_api = ctd.symbol_min_volume_map.get(sid)
        step_api = ctd.symbol_step_volume_map.get(sid)
        max_api = ctd.symbol_max_volume_map.get(sid)
        hard_min = bool(ctd.symbol_min_verified.get(sid))
        hard_step = bool(ctd.symbol_step_verified.get(sid))
        base_floor_api = int(round(0.01 * ctd._VOLUME_PRECISION))
        eff_min_api = (min_api if hard_min else min((min_api or base_floor_api), base_floor_api))
        eff_step_api = (step_api if hard_step else (step_api if (step_api and step_api <= base_floor_api) else base_floor_api))
        as_lots = lambda x: (x / ctd._VOLUME_PRECISION) if (x is not None) else None
        return {
            "symbol": sym_u,
            "min_lots": as_lots(eff_min_api),
            "step_lots": as_lots(eff_step_api),
            "max_lots": as_lots(max_api),
            "min_api": min_api,
            "step_api": step_api,
            "max_api": max_api,
            "min_source": "verified" if hard_min else "metadata",
            "step_source": "verified" if hard_step else "metadata",
            "effective_min_api": eff_min_api,
            "effective_step_api": eff_step_api,
        }

    if symbol:
        sym_u = symbol.upper()
        data = _one(sym_u)
        if not data:
            raise HTTPException(404, f"Symbol '{symbol}' not found.")
        return data

    out = {}
    for sym_u in sorted(ctd.symbol_name_to_id.keys()):
        data = _one(sym_u)
        if data:
            out[sym_u] = data
    return out


@app.get("/api/candles")
async def get_candles(
    symbol: str = Query(..., min_length=1, description="Required, e.g. XAUUSD"),
    timeframe: str = "M5",
    indicators: List[str] = Query([]),
    num_bars: int = 5000,
):
    if not ctd.is_connected():
        raise HTTPException(status_code=503, detail="cTrader feed not connected yet")

    df, _ = data_fetcher.fetch_data(symbol, timeframe, num_bars)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No candles for symbol '{symbol}' on {timeframe}")

    indicator_data = {}
    if indicators:
        df = add_indicators(df, indicators)
        for ind in indicators:
            key = ind.replace(" ", "_").replace("(", "_").replace("(", "").replace("-", "_")
            if key in df.columns:
                indicator_data[key] = [
                    {"time": int(ts.timestamp()), "value": float(val)}
                    for ts, val in df[key].dropna().items()
                ]

    candles = [
        {
            "time": int(idx.timestamp()),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
        for idx, row in df.iterrows()
    ]

    return {"candles": candles, "indicators": indicator_data}


# ──────────────────────────────────────────────────────────────────────────
# Analysis helpers
# ──────────────────────────────────────────────────────────────────────────

def _scalarize(x):
    """Turn 1‑element numpy/pandas/list into a Python scalar; leave others as-is."""
    try:
        if x is None:
            return None
        if hasattr(x, "size") and getattr(x, "size", None) == 1:
            try:
                return x.item()
            except Exception:
                pass
        if isinstance(x, (pd.Series, pd.Index)) and len(x) == 1:
            return x.iloc[0]
        if isinstance(x, (list, tuple)) and len(x) == 1:
            return x[0]
    except Exception:
        pass
    return x


def _truthy(x) -> bool:
    """Safe boolean truthiness for numpy/pandas."""
    if x is None:
        return False
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (list, tuple, dict, set, str, pd.Series, pd.Index)):
        return len(x) > 0
    if hasattr(x, "size"):
        return x.size > 0
    return True



# ──────────────────────────────────────────────────────────────────────────
# LLM Analysis endpoint
# ──────────────────────────────────────────────────────────────────────────


@app.post("/api/analyze")
async def analyze(req: Request):
    body = await req.json()
    symbol     = (body.get("symbol") or "").strip()
    timeframe  = body.get("timeframe", "M5")
    indicators = body.get("indicators", [])
    strategy_name = body.get("strategy") or controller.config.strategy
    num_bars   = body.get("num_bars", 1000)

    # NEW: per-request LLM knobs (fall back to env defaults)
    model      = (body.get("model") or OLLAMA_MODEL).strip()
    if not model:
        model = OLLAMA_MODEL
    options    = (body.get("options") or {}).copy()
    max_tokens = _clamp_int(body.get("max_tokens", options.get("num_predict", 256)), 32, 1024, 256)
    options["num_predict"] = max_tokens

    if not symbol:
        raise HTTPException(status_code=400, detail="symbol is required")

    df, _ = data_fetcher.fetch_data(symbol, timeframe, num_bars)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}:{timeframe}")

    if indicators:
        df = add_indicators(df, indicators)

    try:
        strategy = get_strategy(strategy_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    fig = strategy.build_figure(df) if getattr(strategy, "requires_figure", False) else None

    try:
        td = await strategy.analyze(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            indicators=indicators,
            fig=fig,
            model=model,
            options=options,
            ollama_url=OLLAMA_URL,
        )
        return {"analysis": td.dict()}
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ──────────────────────────────────────────────────────────────────────────
# Trading endpoints
# ──────────────────────────────────────────────────────────────────────────
class PlaceOrderRequest(BaseModel):
    symbol: str
    direction: str = Field(..., alias="action", pattern="^(BUY|SELL)$")
    order_type: str = "MARKET"
    volume: float = 1.0
    volume_mode: Optional[str] = Field(
        default="lots", description="'lots' or 'api' (0.0001-lot units)"
    )
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    rationale: Optional[str] = None


# Helper to enforce single-position policy for manual/chat orders
def _close_opposites_and_wait(symbol: str, desired_dir: str, *, timeout_sec: int = 12) -> dict | None:
    """Close opposite-direction positions for `symbol` and wait until cleared.

    Returns a same-direction open position dict if one exists after closing opposites,
    otherwise None. Uses blocking sleeps as this is called from sync endpoints.
    """
    import time

    sym_u = (symbol or "").upper()
    desired_dir_l = (desired_dir or "").lower()

    try:
        rows = ctd.get_open_positions() or []
    except Exception:
        rows = []

    # Close any opposite positions
    for p in rows:
        if (str(p.get("symbol_name", "")).upper() == sym_u) and ((p.get("direction") or "").lower() != desired_dir_l):
            try:
                close_def = ctd.close_position(
                    client=ctd.client,
                    account_id=ctd.ACCOUNT_ID,
                    position_id=p.get("position_id"),
                    volume_lots=p.get("volume_lots"),
                )
                close_ack = ctd.wait_for_deferred(close_def, timeout=25)
                if isinstance(close_ack, dict) and close_ack.get("status") == "failed":
                    # Best-effort; continue attempting others
                    pass
            except Exception:
                # Best-effort; continue attempting others
                pass

    # Wait briefly until opposites are gone (broker latency)
    deadline = time.time() + max(1, int(timeout_sec))
    while time.time() < deadline:
        try:
            remaining = [
                p for p in (ctd.get_open_positions() or [])
                if (str(p.get("symbol_name", "")).upper() == sym_u)
                and ((p.get("direction") or "").lower() != desired_dir_l)
            ]
        except Exception:
            remaining = []
        if not remaining:
            break
        time.sleep(1)

    # Return any same-direction position if present (to avoid opening duplicates)
    try:
        rows2 = ctd.get_open_positions() or []
    except Exception:
        rows2 = []
    for p in rows2:
        if (str(p.get("symbol_name", "")).upper() == sym_u) and ((p.get("direction") or "").lower() == desired_dir_l):
            return p
    return None


@app.post("/api/execute_trade")
def execute_trade(order: PlaceOrderRequest):
    try:
        if not ctd.symbol_name_to_id:
            raise HTTPException(503, "Symbols not loaded yet.")

        symbol_key = order.symbol.upper()
        if symbol_key not in ctd.symbol_name_to_id:
            raise HTTPException(404, f"Symbol '{order.symbol}' not found.")

        symbol_id = ctd.symbol_name_to_id[symbol_key]
        print(f"[ORDER DEBUG] Sending order: {order=}, {symbol_id=}")

        # Accept volume in lots (default) or raw API units, and coerce to broker limits
        try:
            if (order.volume_mode or "lots").lower() == "api":
                vol_api = int(order.volume)
                # Snap API volume using the same step/min rules
                lots_guess = vol_api / ctd._VOLUME_PRECISION
                volume_raw, lots_final = ctd.coerce_volume_lots_to_units(symbol_id, lots_guess)
            else:
                volume_raw, lots_final = ctd.coerce_volume_lots_to_units(symbol_id, order.volume)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Enforce single-position policy for manual orders
        same_dir_pos = _close_opposites_and_wait(order.symbol, order.direction)

        # If there is already a same-direction position, optionally amend SL/TP and return
        if same_dir_pos is not None:
            amended = False
            try:
                if order.stop_loss is not None or order.take_profit is not None:
                    sl_norm, tp_norm = ctd.normalize_sltp_for_side(
                        side=order.direction,
                        entry_price=same_dir_pos.get("entry_price"),
                        sl=order.stop_loss,
                        tp=order.take_profit,
                    )
                    if sl_norm is None and tp_norm is None:
                        raise ValueError("Provided SL/TP invalid for side and entry price.")
                    d_amend = ctd.modify_position_sltp(
                        client=ctd.client,
                        account_id=ctd.ACCOUNT_ID,
                        position_id=same_dir_pos.get("position_id"),
                        stop_loss=sl_norm,
                        take_profit=tp_norm,
                        symbol_id=symbol_id,
                    )
                    amend_ack = ctd.wait_for_deferred(d_amend, timeout=25)
                    if isinstance(amend_ack, dict) and amend_ack.get("status") == "failed":
                        print(f"[ERROR] SL/TP amend rejected: {amend_ack}")
                    else:
                        amended = True
            except Exception:
                pass
            # Journal the intent even if we didn't open a new position
            try:
                journal_db.add_trade_entry(
                    symbol=order.symbol,
                    direction=order.direction,
                    volume=order.volume,
                    entry_price=None,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                    rationale=(order.rationale or "") + (" • amended existing position" if amended else " • existing position kept"),
                )
            except Exception:
                pass
            return {
                "status": "success",
                "submitted": False,
                "details": {"note": "Existing position in same direction; not opening a new one.", "amended_sl_tp": amended},
            }

        deferred = ctd.place_order(
            client=ctd.client,
            account_id=ctd.ACCOUNT_ID,
            symbol_id=symbol_id,
            order_type=order.order_type,
            side=order.direction,
            volume=volume_raw,
            price=order.entry_price,
            stop_loss=None if order.order_type == "MARKET" else order.stop_loss,
            take_profit=None if order.order_type == "MARKET" else order.take_profit,
        )

        result = ctd.wait_for_deferred(deferred, timeout=25)

        # Journal the trade
        journal_db.add_trade_entry(
            symbol=order.symbol,
            direction=order.direction,
            volume=order.volume,
            entry_price=result.get('price'), # Use actual executed price
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            rationale=order.rationale
        )

        if order.order_type.upper() == "MARKET" and (order.stop_loss or order.take_profit):
            print("[INFO] Waiting to amend SL/TP after market execution...")
            import time

            attempts = int(getattr(ctd, "AMEND_POLL_ATTEMPTS", 25))
            interval = float(getattr(ctd, "AMEND_POLL_INTERVAL", 2.0))
            for attempt in range(attempts):
                time.sleep(max(0.5, interval))
                for p in ctd.get_open_positions():
                    if (
                        p["symbol_name"].upper() == order.symbol.upper()
                        and p["direction"].upper() == order.direction.upper()
                    ):
                        print("[INFO] Found market position, amending SL/TP.")
                        sl_norm, tp_norm = ctd.normalize_sltp_for_side(
                            side=order.direction,
                            entry_price=p["entry_price"],
                            sl=order.stop_loss,
                            tp=order.take_profit,
                        )
                        if sl_norm is None and tp_norm is None:
                            return {
                                "status": "success",
                                "submitted": True,
                                "details": {"status": "failed", "error": "Provided SL/TP invalid for side and entry price."},
                            }
                        d2 = ctd.modify_position_sltp(
                            client=ctd.client,
                            account_id=ctd.ACCOUNT_ID,
                            position_id=p["position_id"],
                            stop_loss=sl_norm,
                            take_profit=tp_norm,
                            symbol_id=symbol_id,
                        )
                        amend_result = ctd.wait_for_deferred(d2, timeout=25)
                        return {
                            "status": "success",
                            "submitted": True,
                            "details": {"status": "ok", "amended_sl_tp": True, "amend_result": str(amend_result)},
                        }
                print(f"[WARN] Position not found yet, retrying ({attempt+1}/{attempts})...")

            return {
                "status": "success",
                "submitted": True,
                "details": {"status": "failed", "error": "Position not found after MARKET execution"},
            }

        return {
            "status": "success",
            "submitted": True,
            "details": {**result, "submitted_volume_lots": lots_final},
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Failed placing order: {e}")
        raise HTTPException(500, detail=str(e))


@app.get("/api/open_positions")
async def open_positions():
    return ctd.get_open_positions()


@app.get("/api/pending_orders")
async def pending_orders():
    pending = ctd.get_pending_orders()
    return [
        {
            "order_id": o.orderId,
            "symbol": ctd.symbol_map.get(o.tradeData.symbolId),
            "type": "LIMIT" if o.orderType == 2 else "STOP",
            "side": "buy" if o.tradeData.tradeSide == 1 else "sell",
            "price": getattr(o, "limitPrice", getattr(o, "stopPrice", 0)) / 100_000,
            # Convert API volume units back to lots
            "volume": o.tradeData.volume / ctd._VOLUME_PRECISION,
        }
        for o in pending
    ]


# ──────────────────────────────────────────────────────────────────────────
# Agent config endpoints
# ──────────────────────────────────────────────────────────────────────────
class WatchlistItemIn(BaseModel):
    symbol: str = Field(..., min_length=1)
    timeframe: str = Field(..., min_length=1)
    lot_size: Optional[float] = Field(None, gt=0)


class AgentConfigIn(BaseModel):
    enabled: bool
    watchlist: List[WatchlistItemIn]
    interval_sec: int = 60
    min_confidence: float = 0.65
    trading_mode: str = "paper"
    autotrade: bool = False
    lot_size_lots: float = 0.01
    strategy: str = "smc"


@app.get("/api/agent/config")
async def get_agent_cfg():
    cfg = controller.config
    return {
        "enabled": cfg.enabled,
        "watchlist": [
            {"symbol": item.symbol, "timeframe": item.timeframe, "lot_size": item.lot_size}
            for item in cfg.watchlist
        ],
        "interval_sec": cfg.interval_sec,
        "min_confidence": cfg.min_confidence,
        "trading_mode": cfg.trading_mode,
        "autotrade": cfg.autotrade,
        "lot_size_lots": cfg.lot_size_lots,
        "strategy": cfg.strategy,
    }


@app.post("/api/agent/config")
async def set_agent_cfg(cfg: AgentConfigIn):
    watch_entries = [
        {
            "symbol": item.symbol,
            "timeframe": item.timeframe,
            "lot_size": item.lot_size if item.lot_size is not None else cfg.lot_size_lots,
        }
        for item in cfg.watchlist
    ]
    new_cfg = AgentConfig(
        enabled=cfg.enabled,
        watchlist=watch_entries,
        interval_sec=cfg.interval_sec,
        min_confidence=cfg.min_confidence,
        trading_mode=cfg.trading_mode,
        autotrade=cfg.autotrade,
        lot_size_lots=cfg.lot_size_lots,
        strategy=cfg.strategy,
    )
    await controller.apply_config(new_cfg)
    return {"ok": True}


@app.post("/api/agent/watchlist/add")
async def agent_add_pair(symbol: str, timeframe: str, lot_size: Optional[float] = None):
    cfg = controller.config
    # Normalize current watchlist to list of tuples, uppercase, filter empties
    current: list[dict[str, object]] = [
        {"symbol": item.symbol, "timeframe": item.timeframe, "lot_size": item.lot_size}
        for item in (cfg.watchlist or [])
    ]

    sym_u = (symbol or "").strip().upper()
    tf_u = (timeframe or "").strip().upper()
    if not sym_u or not tf_u:
        raise HTTPException(status_code=400, detail="symbol and timeframe are required")

    try:
        lot = float(lot_size) if lot_size is not None else float(cfg.lot_size_lots)
    except (TypeError, ValueError):
        lot = float(cfg.lot_size_lots)
    if lot <= 0:
        lot = float(cfg.lot_size_lots)

    updated = False
    for entry in current:
        if entry["symbol"] == sym_u and entry["timeframe"] == tf_u:
            entry["lot_size"] = lot
            updated = True
            break
    if not updated:
        current.append({"symbol": sym_u, "timeframe": tf_u, "lot_size": lot})

    await controller.apply_config(
        AgentConfig(
            enabled=cfg.enabled,
            watchlist=current,
            interval_sec=cfg.interval_sec,
            min_confidence=cfg.min_confidence,
            trading_mode=cfg.trading_mode,
            autotrade=cfg.autotrade,
            lot_size_lots=cfg.lot_size_lots,
            strategy=cfg.strategy,
        )
    )
    return {"ok": True}


@app.post("/api/agent/watchlist/remove")
async def agent_remove_pair(symbol: str, timeframe: str):
    cfg = controller.config
    sym_u = (symbol or "").strip().upper()
    tf_u = (timeframe or "").strip().upper()

    next_list: list[dict[str, object]] = []
    for item in (cfg.watchlist or []):
        if item.symbol == sym_u and item.timeframe == tf_u:
            continue
        next_list.append({"symbol": item.symbol, "timeframe": item.timeframe, "lot_size": item.lot_size})

    await controller.apply_config(
        AgentConfig(
            enabled=cfg.enabled,
            watchlist=next_list,
            interval_sec=cfg.interval_sec,
            min_confidence=cfg.min_confidence,
            trading_mode=cfg.trading_mode,
            autotrade=cfg.autotrade,
            lot_size_lots=cfg.lot_size_lots,
            strategy=cfg.strategy,
        )
    )
    return {"ok": True}

# ──────────────────────────────────────────────────────────────────────────
# Chat endpoint
# ──────────────────────────────────────────────────────────────────────────
# Chat over sockets removed



# In-memory store for pending trade confirmations
_pending_trades = {}

async def _get_intent(message: str) -> dict:
    """Use an LLM to determine the user's intent and extract entities."""
    prompt = f"""You are a helpful trading assistant. Your job is to determine the user's intent and extract relevant information from their message. 

The user said: '{message}'

Possible intents are 'get_price', 'run_analysis', 'start_agents', 'stop_agents', 'get_agent_status', 'place_order', 'confirm_action', 'cancel_action', 'get_news', 'help', or 'unknown'.

- If the intent is 'get_price', extract the 'symbol'.
- If the intent is 'run_analysis', extract the 'symbol', 'timeframe', and 'strategy'.
- If the intent is 'place_order', extract the 'symbol', 'direction' (buy/sell), and 'volume' (in lots).
- If the intent is 'get_news', extract the 'topic' or 'symbol'.

Respond with ONLY a single JSON object in the following format and nothing else:
{{"intent": "<intent>", "symbol": "<symbol or null>", "topic": "<topic or null>", "timeframe": "<timeframe or null>", "strategy": "<strategy or null>"}}"""

    try:
        response_text = _ollama_generate(
            prompt=prompt,
            model=MODEL_DEFAULT, 
            timeout=300, # A longer timeout for chat responses
            json_only=True,
            options_overrides={"num_predict": 48}
        )
        return json.loads(response_text)
    except Exception as e:
        logging.error(f"Error getting intent from LLM: {e}")
        return {"intent": "error", "detail": str(e)}

async def _summarize_analysis(analysis: TradeDecision) -> str:
    """Use an LLM to generate a human-readable summary of the analysis."""
    if not analysis.reasons:
        return "The analysis did not provide specific reasons for its decision."

    reasons_text = '\n'.join([f'- {r}' for r in analysis.reasons])
    prompt = f"""You are a trading analyst. Your colleague has produced the following trading signal and technical reasons.

Signal: {analysis.signal}
Confidence: {analysis.confidence}
SL: {analysis.sl}
TP: {analysis.tp}

Technical Reasons:
{reasons_text}

Your task is to write a concise, one-paragraph explanation of this trade idea for a report. Combine the technical reasons into a fluent, easy-to-understand narrative. Do not just list the reasons; synthesize them."""

    try:
        summary = _ollama_generate(
            prompt=prompt,
            model=MODEL_DEFAULT,
            timeout=45, # Longer timeout for a more detailed response
            json_only=False, # We want a text response
            options_overrides={"num_predict": 128}
        )
        return summary
    except Exception as e:
        logging.error(f"Error summarizing analysis: {e}")
        return "I was unable to generate a summary for the analysis."

async def process_message(message: str, sid) -> str:
    """Processes an incoming chat message and returns a response."""
    logging.info(f"Processing chat message: '{message}'")

    # First, check if this is a confirmation for a pending trade
    if sid in _pending_trades and message.lower() in ["yes", "y"]:
        trade_details = _pending_trades.pop(sid)
        try:
            symbol_id = ctd.symbol_name_to_id[trade_details['symbol'].upper()]
            volume_units, lots_final = ctd.coerce_volume_lots_to_units(symbol_id, trade_details['volume'])
            # Enforce single-position policy before executing chat-initiated trades
            desired_dir = str(trade_details['direction']).lower()
            same_dir_pos = _close_opposites_and_wait(trade_details['symbol'], desired_dir)
            if same_dir_pos is not None:
                return f"Existing {desired_dir.upper()} position for {trade_details['symbol'].upper()} detected. Not opening a new one."

            deferred = ctd.place_order(
                client=ctd.client,
                account_id=ctd.ACCOUNT_ID,
                symbol_id=symbol_id,
                order_type="MARKET",
                side=trade_details['direction'].upper(),
                volume=volume_units
            )
            result = ctd.wait_for_deferred(deferred, timeout=15)

            # Journal the trade
            journal_db.add_trade_entry(
                symbol=trade_details['symbol'],
                direction=trade_details['direction'].upper(),
                volume=trade_details['volume'],
                entry_price=result.get('price'),
                rationale="Trade placed via chatbot"
            )

            return f"✅ Trade executed successfully! Lots: {lots_final:.2f}. Details: {result}"
        except Exception as e:
            logging.error(f"Chat: Error executing trade: {e}")
            return f"❌ Failed to execute trade. Error: {e}"
    elif sid in _pending_trades:
        _pending_trades.pop(sid) # Clear pending trade on any other message
        return "Trade cancelled."
    
    intent_data = await _get_intent(message)
    intent = intent_data.get("intent")

    if intent == "get_price":
        symbol = intent_data.get("symbol")
        if not symbol:
            return "I see you want a price, but which symbol? Please be more specific."
        
        try:
            df, _ = data_fetcher.fetch_data(symbol.upper(), "M1", 2)
            if df.empty:
                return f"Sorry, I couldn't find any data for {symbol.upper()}."
            latest_price = df['close'].iloc[-1]
            return f"The latest price for {symbol.upper()} is {latest_price:.5f}."
        except Exception as e:
            logging.error(f"Chat: Error fetching price for {symbol}: {e}")
            return f"I ran into an error trying to get the price for {symbol.upper()}."

    elif intent == "run_analysis":
        symbol = intent_data.get("symbol")
        if not symbol:
            return "I can run an analysis, but I need to know for which symbol."
        
        # Provide defaults for timeframe and strategy if not extracted
        timeframe = intent_data.get("timeframe") or "H1"
        strategy_name = intent_data.get("strategy") or "smc"

        try:
            df, _ = data_fetcher.fetch_data(symbol.upper(), timeframe, 1000)
            if df.empty:
                return f"Sorry, I couldn't get any data to analyze for {symbol.upper()}."

            strategy = get_strategy(strategy_name)
            analysis_result = await strategy.analyze(df=df, symbol=symbol.upper(), timeframe=timeframe)

            # Format the result for the user
            sig = analysis_result.signal.replace('_', ' ').title()
            conf = f"{analysis_result.confidence * 100:.0f}%" if analysis_result.confidence is not None else "N/A"

            # NEW: Generate a human-readable summary
            summary = await _summarize_analysis(analysis_result)

            return f"""Analysis Complete for {symbol.upper()}:

Signal: {sig}
Confidence: {conf}
SL: {analysis_result.sl}
TP: {analysis_result.tp}

Summary:
{summary}"""

        except Exception as e:
            logging.error(f"Chat: Error running analysis: {e}")
            return f"I hit an error while trying to analyze {symbol.upper()}: {e}"

    elif intent == "start_agents":
        current_config = controller.config
        new_config = AgentConfig(**asdict(current_config))
        new_config.enabled = True
        await controller.apply_config(new_config)
        return "🤖 Agents have been enabled. They will start running based on the current watchlist."

    elif intent == "stop_agents":
        await controller.stop_all()
        current_config = controller.config
        new_config = AgentConfig(**asdict(current_config))
        new_config.enabled = False
        await controller.apply_config(new_config)
        return "🛑 Agents have been disabled and all running tasks have been stopped."

    elif intent == "place_order":
        symbol = intent_data.get("symbol")
        direction = intent_data.get("direction")
        volume = intent_data.get("volume")

        if not all([symbol, direction, volume]):
            return "I can place a trade, but I need a symbol, direction (buy/sell), and volume (in lots)."

        try:
            volume = float(volume)
        except (ValueError, TypeError):
            return f"The volume '{volume}' doesn't look like a valid number."

        # Store the pending trade and ask for confirmation
        _pending_trades[sid] = {
            "symbol": symbol,
            "direction": direction,
            "volume": volume
        }
        return f"You want to {direction.upper()} {volume} lots of {symbol.upper()}. Is this correct? (yes/no)"

    elif intent == "get_news":
        topic = intent_data.get("topic") or intent_data.get("symbol")
        # If no specific topic, default to a general one
        if not topic:
            topic = "global financial markets"
        
        # This can be slow, so let the user know we're working on it
        # (Socket notification removed)
        
        summary = await web_search.get_news_summary(topic)
        return summary

    elif intent == "get_agent_status":
        snap = await controller.snapshot()
        cfg = snap.get("config") or {}
        running_pairs = snap.get("running_pairs") or []
        enabled = cfg.get("enabled", False)

        status_text = "✅ Enabled" if enabled else "❌ Disabled"
        pairs_text = ', '.join([f'{s}/{tf}' for s, tf in running_pairs]) if running_pairs else "none"
        watchlist_items: list[str] = []
        for entry in cfg.get("watchlist") or []:
            if isinstance(entry, dict):
                sym = entry.get("symbol")
                tf = entry.get("timeframe")
                lot = entry.get("lot_size")
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                sym, tf = entry[0], entry[1]
                lot = entry[2] if len(entry) >= 3 else None
            else:
                continue
            sym_u = (sym or "").strip().upper()
            tf_u = (tf or "").strip().upper()
            if not sym_u or not tf_u:
                continue
            try:
                lot_val = float(lot) if lot is not None else None
            except (TypeError, ValueError):
                lot_val = None
            if lot_val is not None:
                watchlist_items.append(f"{sym_u}/{tf_u} ({lot_val:.2f})")
            else:
                watchlist_items.append(f"{sym_u}/{tf_u}")
        watchlist_text = ', '.join(watchlist_items) if watchlist_items else "not set"


        return f"""Agent Status:
- Main switch: {status_text}
- Running pairs: {pairs_text}
- Watchlist: {watchlist_text}
- Strategy: {cfg.get('strategy')}
- Confidence threshold: {cfg.get('min_confidence')}"""

    elif intent == "help":
        return """I can help with the following:
- Get asset prices (e.g., 'price of gold?')
- Run analysis (e.g., 'analyze EURUSD on H1 with smc')
- Start, stop, or get the status of the trading agents.
- Execute trades with confirmation (e.g., 'buy 0.1 lots of EURUSD')
"""
    
    elif intent == "error":
        return f"Sorry, I had trouble understanding that. The AI model reported an error: {intent_data.get('detail')}"

    else: # 'unknown' or anything else
        return "I'm not sure how to help with that yet. You can ask me for the price of a symbol, like 'price of EURUSD'."


# Socket disconnect removed


class ChatStreamRequest(BaseModel):
    message: str = Field(..., min_length=1)
    client_id: Optional[str] = None


@app.post("/api/chat/stream")
async def chat_stream(req: ChatStreamRequest):
    """Streaming chat endpoint.

    Uses existing process_message logic to produce a reply string, then streams
    it to the client in small chunks to enable a typing effect without websockets.
    """
    sid = (req.client_id or "http").strip()

    async def gen():
        try:
            reply = await process_message(req.message, sid)
        except Exception as e:
            logging.exception("chat_stream error")
            reply = f"An error occurred while processing your message: {e}"

        # Stream the reply in small chunks to the client
        chunk_size = 64
        for i in range(0, len(reply), chunk_size):
            chunk = reply[i:i+chunk_size]
            yield chunk
            await asyncio.sleep(0)  # give control back to event loop

    return StreamingResponse(gen(), media_type="text/plain")

