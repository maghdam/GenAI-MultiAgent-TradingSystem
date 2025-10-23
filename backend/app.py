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
import httpx
import os
import threading
import logging
import json
from dataclasses import asdict
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
from backend.strategy import get_strategy, available_strategies
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
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    rationale: Optional[str] = None


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
        try:
            volume_raw = ctd.volume_lots_to_units(symbol_id, order.volume)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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
    # Normalize current watchlist to list of tuples, uppercase, filter empties
    current: list[tuple[str, str]] = []
    for item in (cfg.watchlist or []):
        try:
            s, tf = item
        except Exception:
            continue
        s = (s or "").strip().upper()
        tf = (tf or "").strip().upper()
        if s and tf:
            current.append((s, tf))

    # Add the requested pair (normalized)
    sym_u = (symbol or "").strip().upper()
    tf_u = (timeframe or "").strip().upper()
    if sym_u and tf_u and (sym_u, tf_u) not in current:
        current.append((sym_u, tf_u))

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

    # Normalize and filter out the requested pair
    next_list: list[tuple[str, str]] = []
    for item in (cfg.watchlist or []):
        try:
            s, tf = item
        except Exception:
            continue
        s_u = (s or "").strip().upper()
        tf_u2 = (tf or "").strip().upper()
        if s_u and tf_u2 and not (s_u == sym_u and tf_u2 == tf_u):
            next_list.append((s_u, tf_u2))

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
            volume_units = ctd.volume_lots_to_units(symbol_id, trade_details['volume'])

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

            return f"✅ Trade executed successfully! Details: {result}"
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
        watchlist_text = ', '.join([f'{s}/{tf}' for s, tf in cfg.get('watchlist', [])]) if cfg.get('watchlist') else "not set"


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

