import asyncio
import os
import time
import traceback
from typing import List, Tuple

import plotly.graph_objects as go

import backend.data_fetcher as data_fetcher
from backend.llm_analyzer import analyze_chart_with_llm
from backend.agent_state import push_signal, get_last_bar_ts, set_last_bar_ts
import backend.ctrader_client as ctd


# ---------- utilities ----------
def _parse_watchlist(s: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok: 
            continue
        if ":" in tok:
            sym, tf = tok.split(":", 1)
            out.append((sym.strip(), tf.strip().upper()))
    return out

def _tf_seconds(tf: str) -> int:
    tf = tf.upper()
    return {
        "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
        "H1": 3600, "H4": 14400, "D1": 86400
    }.get(tf, 300)

async def _wait_ready(symbol: str, timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    sym = (symbol or "").upper()
    while time.time() < deadline:
        if ctd.is_connected() and sym in ctd.symbol_name_to_id:
            return True
        await asyncio.sleep(1)
    return False

def _push_err(symbol: str, timeframe: str, msg: str, tag: str = "error"):
    push_signal({
        "ts": time.time(),
        "symbol": symbol,
        "timeframe": timeframe,
        "signal": "error",
        "rationale": f"[{tag}] {msg}",
    })

async def _call_analyzer(fig, df, symbol, timeframe, strategy):
    """Works with or without 'strategy' kwarg (depending on your analyzer version)."""
    try:
        return await analyze_chart_with_llm(
            fig=fig, df=df, symbol=symbol, timeframe=timeframe,
            indicators=[], strategy=strategy
        )
    except TypeError:
        return await analyze_chart_with_llm(
            fig=fig, df=df, symbol=symbol, timeframe=timeframe,
            indicators=[]
        )

def _position_for_symbol(symbol: str):
    rows = ctd.get_open_positions() or []
    sym_u = symbol.upper()
    for p in rows:
        if (p.get("symbol_name","").upper() == sym_u):
            return p  # {symbol_name, position_id, direction, entry_price, volume_lots}
    return None


# ---------- core step (run once per new bar) ----------
async def _scan_once(symbol: str, timeframe: str, min_conf: float, auto_trade: bool,
                     lot_size_lots: float, strategy: str):
    # startup races: ensure feed + symbol are ready
    if not await _wait_ready(symbol, timeout=20):
        _push_err(symbol, timeframe, "cTrader not ready or symbol not loaded", "not_ready")
        return

    try:
        df, _ = data_fetcher.fetch_data(symbol, timeframe, num_bars=500)
    except Exception as e:
        _push_err(symbol, timeframe, str(e), "fetch_fail")
        return

    if df.empty:
        _push_err(symbol, timeframe, "empty dataframe", "no_data")
        return

    # compute the bar-close timestamp for the last candle
    tf_sec = _tf_seconds(timeframe)
    last_idx = df.index[-1].to_pydatetime()
    last_ts = int(last_idx.timestamp())
    last_bar_close = (last_ts // tf_sec) * tf_sec  # end time of the bucket

    # skip if we've already handled this bar
    prev = get_last_bar_ts(symbol, timeframe)
    if prev is not None and prev >= last_bar_close:
        return

    # --- build fig & analyze ---
    try:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="Candles"
        ))

        td = await _call_analyzer(fig, df, symbol, timeframe, strategy)
    except Exception as e:
        _push_err(symbol, timeframe, f"LLM/analyze error: {e}", "llm_error")
        return

    sig = {
        "ts": time.time(),
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy": strategy,
        **td.dict()
    }
    push_signal(sig)

    # --- trade management (simple & safe) ---
    # Only in live+autotrade. Market entry; if already in same direction, amend SL/TP.
    if auto_trade and td.signal in ("long", "short") and (td.confidence or 0) >= min_conf:
        desired_side = "BUY" if td.signal == "long" else "SELL"
        pos = _position_for_symbol(symbol)
        try:
            if pos is None:
                # open new
                sid = ctd.symbol_name_to_id.get(symbol.upper())
                if not sid:
                    _push_err(symbol, timeframe, "symbol id not found", "order_fail")
                else:
                    vol_units = int(float(lot_size_lots) * 10_000_000)
                    d = ctd.place_order(
                        client=ctd.client, account_id=ctd.ACCOUNT_ID, symbol_id=sid,
                        order_type="MARKET", side=desired_side, volume=vol_units,
                        price=None, stop_loss=td.sl, take_profit=td.tp
                    )
                    result = ctd.wait_for_deferred(d, timeout=25)
                    push_signal({**sig, "rationale": (sig.get("rationale") or "") + " • order submitted", "order_result": str(result)})
            else:
                # if same direction -> amend SL/TP; if opposite, we only signal for now
                if (pos.get("direction","").lower() == ("buy" if desired_side=="BUY" else "sell")):
                    try:
                        ctd.modify_position_sltp(
                            client=ctd.client,
                            account_id=ctd.ACCOUNT_ID,
                            position_id=pos["position_id"],
                            stop_loss=td.sl,
                            take_profit=td.tp
                        )
                        push_signal({**sig, "rationale": (sig.get("rationale") or "") + " • amended SL/TP"})
                    except Exception as e:
                        _push_err(symbol, timeframe, f"amend error: {e}", "order_amend_fail")
                else:
                    # Opposite signal while holding a position — just surface it for now.
                    # (Implement close_position in ctrader_client before auto-closing.)
                    push_signal({**sig, "rationale": (sig.get("rationale") or "") + " • opposite signal while in position"})
        except Exception as e:
            _push_err(symbol, timeframe, f"order error: {e}", "order_fail")

    # mark this bar as processed
    set_last_bar_ts(symbol, timeframe, last_bar_close)


# ---------- controller entry ----------
async def run_symbol(symbol: str, timeframe: str, interval: int, min_conf: float,
                     auto_trade: bool, lot_size_lots: float, strategy: str, stop: asyncio.Event):
    timeframe = (timeframe or "M5").upper()
    tf_sec = _tf_seconds(timeframe)
    # Poll fast enough to catch the bar close but not too often
    poll = max(5, min(30, tf_sec // 6))

    while not stop.is_set():
        try:
            await _scan_once(symbol, timeframe, min_conf, auto_trade, lot_size_lots, strategy)
        except asyncio.CancelledError:
            break
        except Exception:
            _push_err(symbol, timeframe, traceback.format_exc().splitlines()[-1], "loop_crash")

        # sleep until next poll or stop
        try:
            await asyncio.wait_for(stop.wait(), timeout=poll)
        except asyncio.TimeoutError:
            pass


# Legacy env-based runner (kept for compatibility)
async def run_agents():
    watch = os.getenv("AGENT_WATCHLIST", "EURUSD:M15")
    interval = int(os.getenv("AGENT_INTERVAL_SEC", "60"))
    min_conf = float(os.getenv("AGENT_MIN_CONFIDENCE", "0.65"))
    mode = os.getenv("TRADING_MODE", "paper").lower()
    auto_trade = mode == "live" and os.getenv("AGENT_AUTOTRADE", "false").lower() == "true"

    tasks = []
    for sym, tf in _parse_watchlist(watch):
        # reuse the bar-close driven logic; interval is ignored
        tasks.append(asyncio.create_task(run_symbol(
            sym, tf, interval, min_conf, auto_trade, lot_size_lots=0.10, strategy="smc", stop=asyncio.Event()
        )))
    await asyncio.gather(*tasks)
