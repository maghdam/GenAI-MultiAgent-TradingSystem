import asyncio
import os
import time
import traceback
from typing import List, Tuple

import backend.data_fetcher as data_fetcher
from backend.agent_state import (
    push_signal,
    get_last_bar_ts,
    set_last_bar_ts,
    update_task_status,
)
import backend.ctrader_client as ctd
from backend.strategy import get_strategy


# ---------- status helpers ----------
def _status(symbol: str, timeframe: str, *, strategy_name: str | None = None, **fields) -> None:
    if strategy_name is not None:
        fields.setdefault("strategy_name", strategy_name)
    update_task_status(symbol, timeframe, **fields)


def _push_err(symbol: str, timeframe: str, msg: str, tag: str, strategy_name: str | None = None) -> None:
    now = time.time()
    _status(
        symbol,
        timeframe,
        state="error",
        last_error=msg,
        last_error_tag=tag,
        last_error_ts=now,
        strategy_name=strategy_name,
    )
    payload = {
        "ts": now,
        "symbol": symbol,
        "timeframe": timeframe,
        "signal": "error",
        "rationale": f"[{tag}] {msg}",
    }
    if strategy_name:
        payload["strategy"] = strategy_name
    push_signal(payload)


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
    tf = (tf or "").upper()
    return {
        "M1": 60,
        "M5": 300,
        "M15": 900,
        "M30": 1_800,
        "H1": 3_600,
        "H4": 14_400,
        "D1": 86_400,
    }.get(tf, 300)


async def _wait_ready(symbol: str, timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    sym = (symbol or "").upper()
    while time.time() < deadline:
        if ctd.is_connected() and sym in ctd.symbol_name_to_id:
            return True
        await asyncio.sleep(1)
    return False


def _position_for_symbol(symbol: str):
    rows = ctd.get_open_positions() or []
    sym_u = (symbol or "").upper()
    for p in rows:
        if p.get("symbol_name", "").upper() == sym_u:
            return p
    return None


# ---------- core scan ----------
async def _scan_once(symbol: str, timeframe: str, min_conf: float, auto_trade: bool,
                     lot_size_lots: float, strategy_name: str):
    _status(symbol, timeframe, state="priming", strategy_name=strategy_name)

    if not await _wait_ready(symbol, timeout=20):
        _push_err(symbol, timeframe, "cTrader not ready or symbol not loaded", "not_ready", strategy_name)
        return

    try:
        _status(symbol, timeframe, state="fetching_data", strategy_name=strategy_name)
        df, _ = data_fetcher.fetch_data(symbol, timeframe, num_bars=500)
    except Exception as e:
        _push_err(symbol, timeframe, str(e), "fetch_fail", strategy_name)
        return

    if df.empty:
        _push_err(symbol, timeframe, "empty dataframe", "no_data", strategy_name)
        return

    tf_sec = _tf_seconds(timeframe)
    last_idx = df.index[-1].to_pydatetime()
    last_ts = int(last_idx.timestamp())
    last_bar_close = (last_ts // tf_sec) * tf_sec

    prev = get_last_bar_ts(symbol, timeframe)
    if prev is not None and prev >= last_bar_close:
        _status(
            symbol,
            timeframe,
            state="waiting_new_bar",
            last_bar_ts=prev,
            last_seen_bar_ts=last_bar_close,
            strategy_name=strategy_name,
        )
        return

    try:
        strategy_obj = get_strategy(strategy_name)
    except Exception as e:
        _push_err(symbol, timeframe, f"strategy error: {e}", "strategy_load", strategy_name)
        return

    fig = None
    if getattr(strategy_obj, "requires_figure", False):
        try:
            fig = strategy_obj.build_figure(df)
        except Exception as e:
            _push_err(symbol, timeframe, f"figure build error: {e}", "strategy_fig", strategy_name)
            return

    try:
        _status(
            symbol,
            timeframe,
            state="analyzing",
            last_bar_ts=last_bar_close,
            strategy_name=strategy_name,
        )
        td = await strategy_obj.analyze(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            indicators=[],
            fig=fig,
        )
    except Exception as e:
        _push_err(symbol, timeframe, f"strategy analyze error: {e}", "strategy_analyze", strategy_name)
        return

    sig = {
        "ts": time.time(),
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy": strategy_name,
        **td.dict(),
    }
    push_signal(sig)
    _status(
        symbol,
        timeframe,
        state="signal_emitted",
        last_signal=td.signal,
        last_confidence=td.confidence,
        last_signal_ts=sig["ts"],
        strategy_name=strategy_name,
    )

    if auto_trade and td.signal in ("long", "short") and (td.confidence or 0) >= min_conf:
        desired_side = "BUY" if td.signal == "long" else "SELL"
        desired_dir = "buy" if desired_side == "BUY" else "sell"
        pos = _position_for_symbol(symbol)
        try:
            if pos is not None:
                pos_dir = (pos.get("direction") or "").lower()
                if pos_dir != desired_dir:
                    _status(
                        symbol,
                        timeframe,
                        state="closing_position",
                        position_id=pos.get("position_id"),
                        existing_direction=pos_dir,
                        desired_direction=desired_dir,
                        strategy_name=strategy_name,
                    )
                    try:
                        close_def = ctd.close_position(
                            client=ctd.client,
                            account_id=ctd.ACCOUNT_ID,
                            position_id=pos["position_id"],
                        )
                        close_result = ctd.wait_for_deferred(close_def, timeout=25)
                        push_signal({
                            **sig,
                            "rationale": (sig.get("rationale") or "") + " • closed opposite position",
                            "close_result": str(close_result),
                        })
                        _status(
                            symbol,
                            timeframe,
                            state="position_closed",
                            position_id=pos.get("position_id"),
                            close_result=str(close_result),
                            strategy_name=strategy_name,
                        )
                        pos = None
                    except Exception as e:
                        _push_err(symbol, timeframe, f"close error: {e}", "order_close_fail", strategy_name)
                        pos = None
                        return
            if pos is not None and (pos.get("direction", "").lower() == desired_dir):
                try:
                    _status(
                        symbol,
                        timeframe,
                        state="amending_position",
                        position_id=pos.get("position_id"),
                        strategy_name=strategy_name,
                    )
                    ctd.modify_position_sltp(
                        client=ctd.client,
                        account_id=ctd.ACCOUNT_ID,
                        position_id=pos["position_id"],
                        stop_loss=td.sl,
                        take_profit=td.tp,
                    )
                    push_signal({**sig, "rationale": (sig.get("rationale") or "") + " • amended SL/TP"})
                    _status(
                        symbol,
                        timeframe,
                        state="position_amended",
                        position_id=pos.get("position_id"),
                        strategy_name=strategy_name,
                    )
                except Exception as e:
                    _push_err(symbol, timeframe, f"amend error: {e}", "order_amend_fail", strategy_name)
            elif pos is None:
                sid = ctd.symbol_name_to_id.get(symbol.upper())
                if not sid:
                    _push_err(symbol, timeframe, "symbol id not found", "order_fail", strategy_name)
                else:
                    vol_units = int(float(lot_size_lots) * 100_000)
                    _status(
                        symbol,
                        timeframe,
                        state="opening_position",
                        desired_direction=desired_dir,
                        volume_lots=lot_size_lots,
                        strategy_name=strategy_name,
                    )
                    d = ctd.place_order(
                        client=ctd.client,
                        account_id=ctd.ACCOUNT_ID,
                        symbol_id=sid,
                        order_type="MARKET",
                        side=desired_side,
                        volume=vol_units,
                        price=None,
                        stop_loss=td.sl,
                        take_profit=td.tp,
                    )
                    result = ctd.wait_for_deferred(d, timeout=25)
                    push_signal({**sig, "rationale": (sig.get("rationale") or "") + " • order submitted", "order_result": str(result)})
        except Exception as e:
            _push_err(symbol, timeframe, f"order error: {e}", "order_fail", strategy_name)

    set_last_bar_ts(symbol, timeframe, last_bar_close)
    _status(
        symbol,
        timeframe,
        state="processed_bar",
        last_bar_ts=last_bar_close,
        strategy_name=strategy_name,
    )


# ---------- controller entry ----------
async def run_symbol(symbol: str, timeframe: str, interval: int, min_conf: float,
                     auto_trade: bool, lot_size_lots: float, strategy: str, stop: asyncio.Event):
    timeframe = (timeframe or "M5").upper()
    tf_sec = _tf_seconds(timeframe)
    configured_interval = max(0, int(interval or 0))
    default_poll = max(5, min(30, tf_sec // 6))
    poll = max(1, configured_interval) if configured_interval > 0 else default_poll

    _status(
        symbol,
        timeframe,
        state="running",
        timeframe_seconds=tf_sec,
        poll_seconds=poll,
        configured_interval_seconds=configured_interval if configured_interval > 0 else None,
        min_conf=min_conf,
        auto_trade=auto_trade,
        lot_size_lots=lot_size_lots,
        strategy=strategy,
        interval_setting=interval,
    )

    while not stop.is_set():
        try:
            await _scan_once(symbol, timeframe, min_conf, auto_trade, lot_size_lots, strategy)
            _status(
                symbol,
                timeframe,
                state="waiting_new_bar",
                next_poll_seconds=poll,
                last_run_ts=time.time(),
                strategy=strategy,
            )
        except asyncio.CancelledError:
            _status(symbol, timeframe, state="cancelled", strategy=strategy)
            break
        except Exception:
            err = traceback.format_exc().splitlines()[-1]
            _push_err(symbol, timeframe, err, "loop_crash", strategy)

        try:
            await asyncio.wait_for(stop.wait(), timeout=poll)
        except asyncio.TimeoutError:
            pass

    _status(symbol, timeframe, state="stopped", stopped_ts=time.time(), strategy=strategy)


# ---------- env-driven legacy entry ----------
async def run_agents():
    watch = os.getenv("AGENT_WATCHLIST", "EURUSD:M15")
    interval = int(os.getenv("AGENT_INTERVAL_SEC", "60"))
    min_conf = float(os.getenv("AGENT_MIN_CONFIDENCE", "0.65"))
    mode = os.getenv("TRADING_MODE", "paper").lower()
    auto_trade = mode == "live" and os.getenv("AGENT_AUTOTRADE", "false").lower() == "true"

    tasks = []
    for sym, tf in _parse_watchlist(watch):
        tasks.append(asyncio.create_task(run_symbol(
            sym,
            tf,
            interval,
            min_conf,
            auto_trade,
            lot_size_lots=0.10,
            strategy="smc",
            stop=asyncio.Event(),
        )))
    await asyncio.gather(*tasks)
