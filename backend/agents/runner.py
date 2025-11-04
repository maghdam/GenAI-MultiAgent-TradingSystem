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
from backend.journal import db as journal_db

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
                     lot_size_lots: float, strategy_name: str, order_type: str = "MARKET"):
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
        try:
            sym_u = (symbol or "").upper()

            # 1) Close opposite-direction positions first
            rows = ctd.get_open_positions() or []
            for p in rows:
                if (str(p.get("symbol_name", "")).upper() == sym_u) and ((p.get("direction") or "").lower() != desired_dir):
                    _status(symbol, timeframe, state="closing_position", position_id=p.get("position_id"), existing_direction=p.get("direction"), desired_direction=desired_dir, strategy_name=strategy_name)
                    try:
                        close_def = ctd.close_position(
                            client=ctd.client,
                            account_id=ctd.ACCOUNT_ID,
                            position_id=p["position_id"],
                            volume_lots=p.get("volume_lots"),
                        )
                        close_ack = ctd.wait_for_deferred(close_def, timeout=25)
                        if isinstance(close_ack, dict) and close_ack.get("status") == "failed":
                            _push_err(symbol, timeframe, f"close rejected: {close_ack.get('error')}", "order_close_fail", strategy_name)
                            return
                        push_signal({**sig, "rationale": (sig.get("rationale") or "") + " • closed opposite position"})
                    except Exception as e:
                        _push_err(symbol, timeframe, f"close error: {e}", "order_close_fail", strategy_name)
                        return

            # 2) Wait until opposites are gone (broker latency)
            for _ in range(12):  # ~12 seconds
                remaining = [
                    p for p in (ctd.get_open_positions() or [])
                    if (str(p.get("symbol_name", "")).upper() == sym_u) and ((p.get("direction") or "").lower() != desired_dir)
                ]
                if not remaining:
                    break
                await asyncio.sleep(1)
            else:
                # Fall-through of for-else means still remaining
                _status(symbol, timeframe, state="closing_position", desired_direction=desired_dir, remaining_opposites=len(remaining), strategy_name=strategy_name)
                push_signal({**sig, "rationale": (sig.get("rationale") or "") + " • still closing opposite position; not opening new"})
                return

            # 3) If a same-direction position exists, amend SL/TP; else open a new position
            rows2 = (ctd.get_open_positions() or [])
            same = None
            for p in rows2:
                if (str(p.get("symbol_name", "")).upper() == sym_u) and ((p.get("direction") or "").lower() == desired_dir):
                    same = p
                    break

            if same is not None:
                try:
                    if td.sl is not None or td.tp is not None:
                        # Normalize SL/TP relative to the actual entry and side
                        sl_norm, tp_norm = ctd.normalize_sltp_for_side(
                            side=desired_side,
                            entry_price=same.get("entry_price"),
                            sl=td.sl,
                            tp=td.tp,
                        )
                        if sl_norm is None and tp_norm is None:
                            return
                        d_amend = ctd.modify_position_sltp(
                            client=ctd.client,
                            account_id=ctd.ACCOUNT_ID,
                            position_id=same["position_id"],
                            stop_loss=sl_norm,
                            take_profit=tp_norm,
                            symbol_id=same.get("symbol_id"),
                        )
                        ack = ctd.wait_for_deferred(d_amend, timeout=25)
                        if isinstance(ack, dict) and ack.get("status") == "failed":
                            _push_err(symbol, timeframe, f"amend rejected: {ack.get('error')}", "order_amend_fail", strategy_name)
                            return
                        # Verify broker reflects amend
                        ok = False
                        for _ in range(5):
                            time.sleep(0.5)
                            for p2 in (ctd.get_open_positions() or []):
                                if int(p2.get("position_id") or 0) == int(same.get("position_id") or 0):
                                    sl_ok = (sl_norm is None) or (abs(float(p2.get("stop_loss") or 0) - float(sl_norm)) < 1e-6)
                                    tp_ok = (tp_norm is None) or (abs(float(p2.get("take_profit") or 0) - float(tp_norm)) < 1e-6)
                                    if sl_ok and tp_ok:
                                        ok = True
                                    break
                            if ok:
                                break
                        if ok:
                            push_signal({**sig, "rationale": (sig.get("rationale") or "") + " • amended SL/TP"})
                        else:
                            # Partial repair: try only missing side(s)
                            sl_missing = (sl_norm is not None) and (abs(float((same.get("stop_loss") or 0)) - float(sl_norm)) >= 1e-6)
                            tp_missing = (tp_norm is not None) and (abs(float((same.get("take_profit") or 0)) - float(tp_norm)) >= 1e-6)
                            if sl_missing or tp_missing:
                                try:
                                    d_fix = ctd.modify_position_sltp(
                                        client=ctd.client,
                                        account_id=ctd.ACCOUNT_ID,
                                        position_id=same["position_id"],
                                        stop_loss=(sl_norm if sl_missing else None),
                                        take_profit=(tp_norm if tp_missing else None),
                                        symbol_id=same.get("symbol_id"),
                                    )
                                    ack_fix = ctd.wait_for_deferred(d_fix, timeout=25)
                                    # Verify again
                                    ok2 = False
                                    for _ in range(5):
                                        time.sleep(0.5)
                                        for p2 in (ctd.get_open_positions() or []):
                                            if int(p2.get("position_id") or 0) == int(same.get("position_id") or 0):
                                                sl_ok2 = (sl_norm is None) or (abs(float(p2.get("stop_loss") or 0) - float(sl_norm)) < 1e-6)
                                                tp_ok2 = (tp_norm is None) or (abs(float(p2.get("take_profit") or 0) - float(tp_norm)) < 1e-6)
                                                ok2 = sl_ok2 and tp_ok2
                                                break
                                        if ok2:
                                            break
                                    if ok2:
                                        push_signal({**sig, "rationale": (sig.get("rationale") or "") + " • amended SL/TP"})
                                    else:
                                        # Accept partial success if at least one side is set
                                        part_ok = False
                                        for p2 in (ctd.get_open_positions() or []):
                                            if int(p2.get("position_id") or 0) == int(same.get("position_id") or 0):
                                                sl_ok3 = (sl_norm is None) or (abs(float(p2.get("stop_loss") or 0) - float(sl_norm)) < 1e-6)
                                                tp_ok3 = (tp_norm is None) or (abs(float(p2.get("take_profit") or 0) - float(tp_norm)) < 1e-6)
                                                part_ok = sl_ok3 or tp_ok3
                                                break
                                        if part_ok:
                                            push_signal({**sig, "rationale": (sig.get("rationale") or "") + " • partially amended SL/TP"})
                                        else:
                                            _push_err(symbol, timeframe, "amend did not apply after verification", "order_amend_fail", strategy_name)
                                except Exception:
                                    _push_err(symbol, timeframe, "amend did not apply after verification", "order_amend_fail", strategy_name)
                            else:
                                _push_err(symbol, timeframe, "amend did not apply after verification", "order_amend_fail", strategy_name)
                except Exception as e:
                    _push_err(symbol, timeframe, f"amend error: {e}", "order_amend_fail", strategy_name)
            else:
                sid = ctd.symbol_name_to_id.get(sym_u)
                if not sid:
                    _push_err(symbol, timeframe, "symbol id not found", "order_fail", strategy_name)
                    return
                try:
                    vol_units, lots_final = ctd.coerce_volume_lots_to_units(sid, lot_size_lots)
                except ValueError as e:
                    _push_err(symbol, timeframe, str(e), "order_fail", strategy_name)
                    return

                _status(symbol, timeframe, state="opening_position", desired_direction=desired_dir, volume_lots=lots_final, strategy_name=strategy_name)
                # Choose order type and price
                ot = (order_type or "MARKET").upper()
                px = None
                try:
                    last_high = float(df["high"].iloc[-1])
                    last_low = float(df["low"].iloc[-1])
                    last_close = float(df["close"].iloc[-1])
                except Exception:
                    last_high = last_low = last_close = None
                if ot == "STOP":
                    px = (last_high if desired_side == "BUY" else last_low)
                elif ot == "LIMIT":
                    # basic pullback: buy near low, sell near high
                    px = (last_low if desired_side == "BUY" else last_high)

                d = ctd.place_order(
                    client=ctd.client,
                    account_id=ctd.ACCOUNT_ID,
                    symbol_id=sid,
                    order_type=ot,
                    side=desired_side,
                    volume=vol_units,
                    price=px,
                    stop_loss=(td.sl if ot in ("LIMIT", "STOP") else None),
                    take_profit=(td.tp if ot in ("LIMIT", "STOP") else None),
                )
                result = ctd.wait_for_deferred(d, timeout=25)

                # Fallback: if SL/TP not applied, poll for the new position and amend
                try:
                    if td.sl is not None or td.tp is not None:
                        for _ in range(int(getattr(ctd, "AMEND_POLL_ATTEMPTS", 25))):
                            await asyncio.sleep(max(0.5, float(getattr(ctd, "AMEND_POLL_INTERVAL", 2.0))))
                            for p in (ctd.get_open_positions() or []):
                                if (str(p.get("symbol_name", "")).upper() == sym_u) and ((p.get("direction") or "").lower() == desired_dir):
                                    # Normalize relative to actual entry and side
                                    sl_norm, tp_norm = ctd.normalize_sltp_for_side(
                                        side=desired_side,
                                        entry_price=p.get("entry_price"),
                                        sl=td.sl,
                                        tp=td.tp,
                                    )
                                    if sl_norm is None and tp_norm is None:
                                        break
                                    d2 = ctd.modify_position_sltp(
                                        client=ctd.client,
                                        account_id=ctd.ACCOUNT_ID,
                                        position_id=p["position_id"],
                                        stop_loss=sl_norm,
                                        take_profit=tp_norm,
                                        symbol_id=p.get("symbol_id"),
                                    )
                                    ack2 = ctd.wait_for_deferred(d2, timeout=25)
                                    if isinstance(ack2, dict) and ack2.get("status") == "failed":
                                        _push_err(symbol, timeframe, f"amend rejected: {ack2.get('error')}", "order_amend_fail", strategy_name)
                                        break
                                    # Verify
                                    ok = False
                                    for _ in range(5):
                                        time.sleep(0.5)
                                        for p3 in (ctd.get_open_positions() or []):
                                            if int(p3.get("position_id") or 0) == int(p.get("position_id") or 0):
                                                sl_ok = (sl_norm is None) or (abs(float(p3.get("stop_loss") or 0) - float(sl_norm)) < 1e-6)
                                                tp_ok = (tp_norm is None) or (abs(float(p3.get("take_profit") or 0) - float(tp_norm)) < 1e-6)
                                                if sl_ok and tp_ok:
                                                    ok = True
                                                break
                                        if ok:
                                            break
                                    if not ok:
                                        _push_err(symbol, timeframe, "amend did not apply after verification", "order_amend_fail", strategy_name)
                                    break
                            else:
                                continue
                            break
                except Exception as e:
                    _push_err(symbol, timeframe, f"amend fallback error: {e}", "order_amend_fail", strategy_name)

                try:
                    # Determine entry price: prefer result; fallback to scanning open positions
                    entry_px = None
                    if isinstance(result, dict):
                        entry_px = result.get('price') or None
                    if entry_px in (None, 0, 0.0):
                        for _ in range(6):  # ~3s
                            await asyncio.sleep(0.5)
                            for p in (ctd.get_open_positions() or []):
                                if (str(p.get("symbol_name", "")).upper() == sym_u) and ((p.get("direction") or "").lower() == desired_dir):
                                    entry_px = p.get("entry_price")
                                    break
                            if entry_px not in (None, 0, 0.0):
                                break
                    # Journal the trade after it's confirmed (entry may still be None for pending orders)
                    journal_db.add_trade_entry(
                        symbol=symbol,
                        direction=desired_side,
                        volume=lot_size_lots,
                        entry_price=entry_px,
                        stop_loss=td.sl,
                        take_profit=td.tp,
                        rationale=td.rationale or (td.reasons[0] if getattr(td, 'reasons', None) else None),
                    )
                except Exception as e:
                    _push_err(symbol, timeframe, f"journaling error: {e}", "journal_fail", strategy_name)

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
                     auto_trade: bool, lot_size_lots: float, strategy: str, order_type: str, stop: asyncio.Event):
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
            await _scan_once(symbol, timeframe, min_conf, auto_trade, lot_size_lots, strategy, order_type)
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



