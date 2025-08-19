# backend/agents/runner.py
import asyncio, os, time
from typing import List, Tuple
import plotly.graph_objects as go

from backend.data_fetcher import fetch_data
from backend.llm_analyzer import analyze_chart_with_llm
from backend.agent_state import push_signal


from backend.ctrader_client import place_order, wait_for_deferred, symbol_name_to_id, client, ACCOUNT_ID


from backend.ctrader_client import place_order, wait_for_deferred, symbol_name_to_id, client, ACCOUNT_ID


def _parse_watchlist(s: str) -> List[Tuple[str, str]]:
    # "EURUSD:M15,GBPUSD:M5" -> [("EURUSD","M15"), ("GBPUSD","M5")]
    out = []
    for token in (s or "").split(","):
        if not token.strip(): continue
        if ":" in token:
            sym, tf = token.strip().split(":", 1)
            out.append((sym.strip(), tf.strip()))
    return out

async def _scan_symbol(symbol: str, timeframe: str, interval: int, min_conf: float, auto_trade: bool):
    while True:
        try:
            df, _ = fetch_data(symbol, timeframe, num_bars=500)
            if df.empty:
                await asyncio.sleep(interval); continue

            # minimal fig (same as /api/analyze)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Candles"
            ))

            td = await analyze_chart_with_llm(fig=fig, df=df, symbol=symbol, timeframe=timeframe, indicators=[])

            sig = {
                "ts": time.time(),
                "symbol": symbol,
                "timeframe": timeframe,
                **td.dict(),
            }
            push_signal(sig)

            # optional auto-trade (MARKET) if confidence high enough
            if auto_trade and td.signal in ("long","short") and (td.confidence or 0) >= min_conf:
                side = "BUY" if td.signal == "long" else "SELL"
                sid = symbol_name_to_id.get(symbol.upper())
                if sid:
                    # 0.01 lot by default; adjust to your account
                    volume_units = int(0.01 * 10_000_000)
                    d = place_order(
                        client=client, account_id=ACCOUNT_ID, symbol_id=sid,
                        order_type="MARKET", side=side, volume=volume_units,
                        price=None,  # market
                        stop_loss=td.sl, take_profit=td.tp
                    )
                    result = wait_for_deferred(d, timeout=25)
                    sig["order_result"] = str(result)
        except Exception as e:
            push_signal({"ts": time.time(), "symbol": symbol, "timeframe": timeframe,
                         "signal": "error", "rationale": str(e)})
        finally:
            await asyncio.sleep(interval)

async def run_agents():
    # envs with sane defaults
    watch = os.getenv("AGENT_WATCHLIST", "EURUSD:M15")
    interval = int(os.getenv("AGENT_INTERVAL_SEC", "60"))
    min_conf = float(os.getenv("AGENT_MIN_CONFIDENCE", "0.65"))
    mode = os.getenv("TRADING_MODE", "paper").lower()
    auto_trade = mode == "live" and os.getenv("AGENT_AUTOTRADE", "false").lower() == "true"

    tasks = []
    for sym, tf in _parse_watchlist(watch):
        tasks.append(asyncio.create_task(_scan_symbol(sym, tf, interval, min_conf, auto_trade)))

    # keep tasks alive
    await asyncio.gather(*tasks)




async def _scan_once(symbol: str, timeframe: str, min_conf: float, auto_trade: bool,
                     lot_size_lots: float, strategy: str):
    df, _ = fetch_data(symbol, timeframe, num_bars=500)
    if df.empty:
        push_signal({"ts": time.time(), "symbol": symbol, "timeframe": timeframe,
                     "signal": "no_data", "rationale": "empty dataframe"})
        return

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"],
                                 low=df["low"], close=df["close"], name="Candles"))

    td = await analyze_chart_with_llm(fig=fig, df=df, symbol=symbol, timeframe=timeframe,
                                      indicators=[], strategy=strategy)
    sig = {"ts": time.time(), "symbol": symbol, "timeframe": timeframe, "strategy": strategy, **td.dict()}
    push_signal(sig)

    if auto_trade and td.signal in ("long","short") and (td.confidence or 0) >= min_conf:
        side = "BUY" if td.signal == "long" else "SELL"
        sid = symbol_name_to_id.get(symbol.upper())
        if sid:
            vol_units = int(float(lot_size_lots) * 10_000_000)
            d = place_order(client=client, account_id=ACCOUNT_ID, symbol_id=sid,
                            order_type="MARKET", side=side, volume=vol_units,
                            price=None, stop_loss=td.sl, take_profit=td.tp)
            result = wait_for_deferred(d, timeout=25)
            sig["order_result"] = str(result)

async def run_symbol(symbol: str, timeframe: str, interval: int, min_conf: float,
                     auto_trade: bool, lot_size_lots: float, strategy: str, stop: asyncio.Event):
    while not stop.is_set():
        try:
            await _scan_once(symbol, timeframe, min_conf, auto_trade, lot_size_lots, strategy)
        except asyncio.CancelledError:
            break
        except Exception as e:
            push_signal({"ts": time.time(), "symbol": symbol, "timeframe": timeframe,
                         "signal": "error", "rationale": str(e)})
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass