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
from backend.llm_analyzer import analyze_chart_with_llm
from backend.smc_features import (
    detect_bos_choch,
    current_fvg,
    ob_near_price,
    in_premium_discount,
)
from backend.agent_state import recent_signals
from backend.agents.runner import run_agents
from backend.agent_controller import controller, AgentConfig

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

    # NEW: per-request LLM knobs (fall back to env defaults)
    model      = (body.get("model") or OLLAMA_MODEL).strip()
    options    = body.get("options") or {}
    max_bars   = _clamp_int(body.get("max_bars", 200), 50, 1000, 200)   # candles used in the image + text
    max_tokens = _clamp_int(body.get("max_tokens", options.get("num_predict", 256)), 32, 1024, 256)

    if not symbol:
        raise HTTPException(status_code=400, detail="symbol is required")

    # pull a bit more than we show to the LLM if you want, but trim before passing on
    df, _ = data_fetcher.fetch_data(symbol, timeframe, num_bars=max(1000, max_bars))
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}:{timeframe}")

    if indicators:
        df = add_indicators(df, indicators)

    # keep the LLM context lean
    df_llm = df.tail(max_bars).copy()

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Candles",
        )
    )

    # 1) BOS/CHoCH
    structure = _scalarize(detect_bos_choch(df))
    if _truthy(structure):
        fig.add_annotation(
            text=str(structure),
            x=df.index[-1],
            y=df["close"].iloc[-1],
            showarrow=True,
            arrowhead=2,
            bgcolor="black",
            font=dict(color="white"),
            yshift=30,
        )

    # 2) FVG (need >= 3 rows)
    if len(df) >= 3:
        fvg = _scalarize(current_fvg(df))
        if _truthy(fvg):
            c0, c2 = df.iloc[-1], df.iloc[-3]
            fig.add_shape(
                type="rect",
                x0=df.index[-3],
                x1=df.index[-1],
                y0=c2["high"],
                y1=c0["low"],
                fillcolor="rgba(255, 165, 0, 0.3)",
                line=dict(width=0),
                layer="below",
            )
            fig.add_annotation(
                text=str(fvg),
                x=df.index[-2],
                y=(float(c2["high"]) + float(c0["low"])) / 2.0,
                showarrow=False,
                font=dict(color="orange", size=12),
            )

    # 3) Near OB tag
    near_ob = _scalarize(ob_near_price(df))
    if _truthy(near_ob):
        fig.add_annotation(
            text="Near OB",
            x=df.index[-1],
            y=df["close"].iloc[-1],
            showarrow=True,
            arrowhead=1,
            font=dict(color="blue"),
            yshift=-30,
        )

    # 4) Premium/Discount
    zone = _scalarize(in_premium_discount(df))
    if isinstance(zone, (bytes, bytearray)):
        zone = zone.decode(errors="ignore")
    if zone is not None:
        zone = str(zone)
    if zone in ("premium", "discount"):
        swing_hi = float(df["high"].iloc[-50:].max())
        swing_lo = float(df["low"].iloc[-50:].min())
        if np.isfinite(swing_hi) and np.isfinite(swing_lo) and swing_hi > swing_lo:
            mid = (swing_hi + swing_lo) / 2.0
            fig.add_hline(y=mid, line=dict(dash="dot", color="gray"), name="Equilibrium")
            fig.add_annotation(
                text=zone.upper(),
                x=df.index[-1],
                y=mid,
                showarrow=False,
                font=dict(size=11, color="gray"),
                yshift=-40,
            )

    td = await analyze_chart_with_llm(fig=fig, df=df, symbol=symbol, timeframe=timeframe, indicators=indicators)
    return {"analysis": td.dict()}


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

        volume_raw = order.volume * 10_000_000

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
            "volume": o.tradeData.volume / 10_000_000,
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
