# app.py

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import asyncio
import httpx
import os
import threading
from threading import Lock
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel, Field

# ── Our internal modules ──────────────────────────────────────────────────
import backend.ctrader_client as ctd
import backend.data_fetcher as data_fetcher
from backend.symbol_fetcher import get_available_symbols
from backend.indicators import add_indicators
from backend.llm_analyzer import analyze_data_with_llm
from backend.smc_features import (
    detect_bos_choch,
    current_fvg,
    ob_near_price,
    in_premium_discount,
)
from backend.agent_state import recent_signals, all_task_status
from backend.agents.runner import run_agents
from backend.agent_controller import controller, AgentConfig
from backend.strategy import get_strategy, available_strategies

# ──────────────────────────────────────────────────────────────────────────
# FastAPI app & middleware
# ──────────────────────────────────────────────────────────────────────────
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

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
async def _maybe_start_agents():
    # Start background agents (optional)
    if os.getenv("AGENTS_ENABLED", "false").lower() == "true":
        asyncio.create_task(run_agents())  # don't block server


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


@app.get("/api/agent/status")
async def agent_status():
    snap = await controller.snapshot()
    cfg = snap.get("config") or {}
    running_pairs = snap.get("running_pairs") or []
    tasks = all_task_status()

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


# ──────────────────────────────────────────────────────────────────────────
# UI: serve index.html
# ──────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home():
    return Path("templates/index.html").read_text()


# ──────────────────────────────────────────────────────────────────────────
# Symbols & candles
# ──────────────────────────────────────────────────────────────────────────
@app.get("/api/symbols")
async def get_symbols():
    """Return the list of symbols and a safe default for the UI."""
    symbols = get_available_symbols()  # e.g., ["XAUUSD", "US500", ...]
    default = DEFAULT_SYMBOL if DEFAULT_SYMBOL in symbols else (symbols[0] if symbols else None)
    return {"symbols": symbols, "default": default}


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
            key = ind.replace(" ", "_").replace("(", "_").replace(")", "").replace("-", "_")
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
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


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

        # cTrader volume units: 1 lot = 100,000 units
        volume_raw = order.volume * 100_000

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

        if order.order_type.upper() == "MARKET" and (order.stop_loss or order.take_profit):
            print("[INFO] Waiting to amend SL/TP after market execution...")
            import time

            for attempt in range(5):
                time.sleep(2)
                for p in ctd.get_open_positions():
                    if (
                        p["symbol_name"].upper() == order.symbol.upper()
                        and p["direction"].upper() == order.direction.upper()
                    ):
                        print("[INFO] Found market position, amending SL/TP.")
                        amend_result = ctd.modify_position_sltp(
                            client=ctd.client,
                            account_id=ctd.ACCOUNT_ID,
                            position_id=p["position_id"],
                            stop_loss=order.stop_loss,
                            take_profit=order.take_profit,
                        )
                        return {
                            "status": "success",
                            "submitted": True,
                            "details": {"status": "ok", "amended_sl_tp": True, "amend_result": str(amend_result)},
                        }
                print(f"[WARN] Position not found yet, retrying ({attempt+1}/5)...")

            return {
                "status": "success",
                "submitted": True,
                "details": {"status": "failed", "error": "Position not found after MARKET execution"},
            }

        return {"status": "success", "submitted": True, "details": result}

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
            # Convert units back to lots
            "volume": o.tradeData.volume / 100_000,
        }
        for o in pending
    ]


# ──────────────────────────────────────────────────────────────────────────
# Agent config endpoints
# ──────────────────────────────────────────────────────────────────────────
class AgentConfigIn(BaseModel):
    enabled: bool
    watchlist: List[Tuple[str, str]]
    interval_sec: int = 60
    min_confidence: float = 0.65
    trading_mode: str = "paper"
    autotrade: bool = False
    lot_size_lots: float = 0.10
    strategy: str = "smc"


@app.get("/api/agent/config")
async def get_agent_cfg():
    cfg = controller.config
    return {
        "enabled": cfg.enabled,
        "watchlist": cfg.watchlist,
        "interval_sec": cfg.interval_sec,
        "min_confidence": cfg.min_confidence,
        "trading_mode": cfg.trading_mode,
        "autotrade": cfg.autotrade,
        "lot_size_lots": cfg.lot_size_lots,
        "strategy": cfg.strategy,
    }


@app.post("/api/agent/config")
async def set_agent_cfg(cfg: AgentConfigIn):
    new_cfg = AgentConfig(
        enabled=cfg.enabled,
        watchlist=[(s, tf) for s, tf in cfg.watchlist],
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
async def agent_add_pair(symbol: str, timeframe: str):
    cfg = controller.config
    wl = set(cfg.watchlist)
    wl.add((symbol, timeframe))
    await controller.apply_config(
        AgentConfig(
            enabled=cfg.enabled,
            watchlist=list(wl),
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
    wl = set(cfg.watchlist)
    wl.discard((symbol, timeframe))
    await controller.apply_config(
        AgentConfig(
            enabled=cfg.enabled,
            watchlist=list(wl),
            interval_sec=cfg.interval_sec,
            min_confidence=cfg.min_confidence,
            trading_mode=cfg.trading_mode,
            autotrade=cfg.autotrade,
            lot_size_lots=cfg.lot_size_lots,
            strategy=cfg.strategy,
        )
    )
    return {"ok": True}
