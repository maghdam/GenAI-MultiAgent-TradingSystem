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
    # Global scan concurrency gate (sequential by default)
    # Tuned via env: AGENT_MAX_CONCURRENT_SCANS (default 1)
    global _SCAN_SEM
    try:
        _SCAN_SEM
    except NameError:
        try:
            limit = int(os.getenv("AGENT_MAX_CONCURRENT_SCANS", "1") or "1")
        except Exception:
            limit = 1
        if limit <= 0:
            limit = 1
        _SCAN_SEM = asyncio.Semaphore(limit)
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

    # Optional per-pair stagger to spread load after bar close
    try:
        max_off = int(float(os.getenv("AGENT_STAGGER_MAX_SEC", "0") or 0))
    except Exception:
        max_off = 0
    if max_off > 0:
        try:
            import hashlib
            key = f"{symbol}:{timeframe}"
            dig = hashlib.sha1(key.encode("utf-8")).digest()
            delay = int.from_bytes(dig[:2], "big") % (max_off + 1)
        except Exception:
            delay = abs(hash((symbol, timeframe))) % (max_off + 1)
        if delay > 0:
            await asyncio.sleep(delay)

    try:
        # Acquire scan semaphore for heavy analysis/order section
        async with _SCAN_SEM:
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
        # Wrap analysis within the same scan semaphore context
        async with _SCAN_SEM:
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
        last_error=None,
        last_error_tag=None,
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
                        async with _SCAN_SEM:
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
                            await asyncio.sleep(0.5)
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
                                        await asyncio.sleep(0.5)
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

                async with _SCAN_SEM:
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
                    # Helpers for default SL/TP when missing
                    def _risk_param(sym: str, key: str, default_val: str) -> str:
                        # Per-symbol override: KEY_{SYMBOL}, else KEY
                        sym_u = (sym or "").upper()
                        env_key_sym = f"{key}_{sym_u}"
                        v = os.getenv(env_key_sym)
                        if v is None:
                            v = os.getenv(key, default_val)
                        return str(v)

                    def _derive_default_sltp(side: str, entry_price: float | None) -> tuple[float | None, float | None]:
                        try:
                            mode = (_risk_param(symbol, "SMC_RISK_MODE", "atr") or "atr").strip().lower()
                            rr = float(_risk_param(symbol, "SMC_RR", "2.0") or 2.0)
                        except Exception:
                            mode = "atr"; rr = 2.0
                        if entry_price is None or entry_price == 0:
                            try:
                                entry_price = float(df["close"].iloc[-1])
                            except Exception:
                                return None, None
                        if mode == "swing":
                            try:
                                lb = int(_risk_param(symbol, "SMC_SWING_LOOKBACK", "10") or 10)
                                pad = float(_risk_param(symbol, "SMC_TICK_PCT", "0.0005") or 0.0005)
                            except Exception:
                                lb = 10; pad = 0.0005
                            hi = float(df["high"].iloc[-lb:].max())
                            lo = float(df["low"].iloc[-lb:].min())
                            if side == "BUY":
                                sl = lo * (1.0 - pad)
                                tp = entry_price + rr * (entry_price - sl)
                            else:
                                sl = hi * (1.0 + pad)
                                tp = entry_price - rr * (sl - entry_price)
                            return sl, tp
                        # default: ATR
                        try:
                            n = int(_risk_param(symbol, "SMC_ATR_LEN", "14") or 14)
                            mult = float(_risk_param(symbol, "SMC_ATR_MULT", "1.0") or 1.0)
                        except Exception:
                            n = 14; mult = 1.0
                        try:
                            high = df["high"].astype(float)
                            low = df["low"].astype(float)
                            close = df["close"].astype(float)
                            prev_close = close.shift(1)
                            tr = (high - low).to_frame(name="hl")
                            tr["hc"] = (high - prev_close).abs()
                            tr["lc"] = (low - prev_close).abs()
                            tr_max = tr.max(axis=1)
                            atr = tr_max.rolling(n, min_periods=n).mean().iloc[-1]
                        except Exception:
                            return None, None
                        if atr is None or atr == 0:
                            return None, None
                        if side == "BUY":
                            sl = entry_price - mult * float(atr)
                            tp = entry_price + rr * (entry_price - sl)
                        else:
                            sl = entry_price + mult * float(atr)
                            tp = entry_price - rr * (sl - entry_price)
                        return sl, tp

                    if td.sl is not None or td.tp is not None:
                        interval = max(0.5, float(getattr(ctd, "AMEND_POLL_INTERVAL", 2.0)))
                        try:
                            max_dur = float(os.getenv("SLTP_AMEND_MAX_DURATION_SEC", "600"))
                        except Exception:
                            max_dur = 600.0
                        deadline = time.time() + max(30.0, max_dur)
                        ok = False
                        while time.time() < deadline:
                            await asyncio.sleep(interval)
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
                                    # Verify with symbol-aware rounding
                                    tol = _amend_tolerance(p.get("symbol_id"))
                                    expect_sl = _round_for_symbol(p.get("symbol_id"), sl_norm)
                                    expect_tp = _round_for_symbol(p.get("symbol_id"), tp_norm)
                                    for _ in range(8):
                                        await asyncio.sleep(0.5)
                                        for p3 in (ctd.get_open_positions() or []):
                                            if int(p3.get("position_id") or 0) == int(p.get("position_id") or 0):
                                                sl_ok = (expect_sl is None) or (abs(float(p3.get("stop_loss") or 0) - float(expect_sl)) <= tol)
                                                tp_ok = (expect_tp is None) or (abs(float(p3.get("take_profit") or 0) - float(expect_tp)) <= tol)
                                                if sl_ok and tp_ok:
                                                    ok = True
                                                break
                                        if ok:
                                            break
                            if ok:
                                break
                        if not ok:
                            # Partial repair: try to amend only missing sides
                            try:
                                last_pos = None
                                for p4 in (ctd.get_open_positions() or []):
                                    if int(p4.get("position_id") or 0) == int(p.get("position_id") or 0):
                                        last_pos = p4
                                        break
                                if last_pos is not None:
                                    has_sl = (expect_sl is None) or (abs(float(last_pos.get("stop_loss") or 0) - float(expect_sl)) <= tol)
                                    has_tp = (expect_tp is None) or (abs(float(last_pos.get("take_profit") or 0) - float(expect_tp)) <= tol)
                                    if not (has_sl and has_tp):
                                        d_fix = ctd.modify_position_sltp(
                                            client=ctd.client,
                                            account_id=ctd.ACCOUNT_ID,
                                            position_id=p.get("position_id"),
                                            stop_loss=(expect_sl if not has_sl else None),
                                            take_profit=(expect_tp if not has_tp else None),
                                            symbol_id=p.get("symbol_id"),
                                        )
                                        _ = ctd.wait_for_deferred(d_fix, timeout=25)
                                        # Verify again (short)
                                        ok2 = False
                                        for _ in range(5):
                                            await asyncio.sleep(0.5)
                                            for p5 in (ctd.get_open_positions() or []):
                                                if int(p5.get("position_id") or 0) == int(p.get("position_id") or 0):
                                                    sl_ok2 = (expect_sl is None) or (abs(float(p5.get("stop_loss") or 0) - float(expect_sl)) <= tol)
                                                    tp_ok2 = (expect_tp is None) or (abs(float(p5.get("take_profit") or 0) - float(expect_tp)) <= tol)
                                                    ok2 = sl_ok2 and tp_ok2
                                                    break
                                            if ok2:
                                                break
                                        if not ok2:
                                            _push_err(symbol, timeframe, "amend did not apply after verification", "order_amend_fail", strategy_name)
                                else:
                                    _push_err(symbol, timeframe, "position not found after amend", "order_amend_fail", strategy_name)
                            except Exception as _e:
                                _push_err(symbol, timeframe, f"partial amend error: {_e}", "order_amend_fail", strategy_name)
                    else:
                        # Strategy didn't provide SL/TP. Attempt to derive defaults and amend once we know entry
                        # Prefer broker-reflected entry from open positions (decoded), not raw ack
                        entry_px_for_defaults = None
                        for _ in range(6):  # ~3s total
                            await asyncio.sleep(0.5)
                            found = False
                            for p in (ctd.get_open_positions() or []):
                                if (str(p.get("symbol_name", "")).upper() == sym_u) and ((p.get("direction") or "").lower() == desired_dir):
                                    entry_px_for_defaults = p.get("entry_price")
                                    found = True
                                    break
                            if found and entry_px_for_defaults not in (None, 0, 0.0):
                                break
                        if entry_px_for_defaults in (None, 0, 0.0):
                            # Fallback to ack price only if broker scan failed
                            if isinstance(result, dict):
                                try:
                                    entry_px_for_defaults = float(result.get('price') or 0) or None
                                except Exception:
                                    entry_px_for_defaults = None
                        sl_raw, tp_raw = _derive_default_sltp(desired_side, entry_px_for_defaults)
                        if sl_raw is not None or tp_raw is not None:
                            # Step-up backoff: try base, then widen distances if amend verification fails
                            widen_factors = []
                            try:
                                widen_env = os.getenv("SMC_BACKOFF_FACTORS", "1.25,1.5")
                                if widen_env:
                                    widen_factors = [float(x) for x in str(widen_env).split(',') if x.strip()]
                            except Exception:
                                widen_factors = [1.25, 1.5]

                            async def _attempt_amend(sl_val: float | None, tp_val: float | None) -> bool:
                                sl_n, tp_n = ctd.normalize_sltp_for_side(
                                    side=desired_side,
                                    entry_price=entry_px_for_defaults,
                                    sl=sl_val,
                                    tp=tp_val,
                                )
                                if sl_n is None and tp_n is None:
                                    return False
                                pos_id_target = None
                                for p in (ctd.get_open_positions() or []):
                                    if (str(p.get("symbol_name", "")).upper() == sym_u) and ((p.get("direction") or "").lower() == desired_dir):
                                        pos_id_target = p.get("position_id")
                                        break
                                if not pos_id_target:
                                    return False
                                d_mod = ctd.modify_position_sltp(
                                    client=ctd.client,
                                    account_id=ctd.ACCOUNT_ID,
                                    position_id=pos_id_target,
                                    stop_loss=sl_n,
                                    take_profit=tp_n,
                                    symbol_id=sid,
                                )
                                _ = ctd.wait_for_deferred(d_mod, timeout=25)
                                # Verify
                                ok = False
                                tol = _amend_tolerance(sid)
                                for _ in range(5):
                                    await asyncio.sleep(0.5)
                                    for p3 in (ctd.get_open_positions() or []):
                                        if int(p3.get("position_id") or 0) == int(pos_id_target or 0):
                                            sl_ok = (sl_n is None) or (abs(float(p3.get("stop_loss") or 0) - float(sl_n)) <= tol)
                                            tp_ok = (tp_n is None) or (abs(float(p3.get("take_profit") or 0) - float(tp_n)) <= tol)
                                            ok = sl_ok and tp_ok
                                            break
                                    if ok:
                                        break
                                return ok

                            ok_base = await _attempt_amend(sl_raw, tp_raw)
                            if not ok_base:
                                # Widen and retry
                                for f in widen_factors:
                                    # Recompute widened SL/TP
                                    try:
                                        mode_now = (_risk_param(symbol, "SMC_RISK_MODE", "atr") or "atr").strip().lower()
                                    except Exception:
                                        mode_now = "atr"
                                    if mode_now == "swing":
                                        try:
                                            pad0 = float(_risk_param(symbol, "SMC_TICK_PCT", "0.0005") or 0.0005)
                                        except Exception:
                                            pad0 = 0.0005
                                        pad = pad0 * float(f)
                                        lb = int(_risk_param(symbol, "SMC_SWING_LOOKBACK", "10") or 10)
                                        hi = float(df["high"].iloc[-lb:].max())
                                        lo = float(df["low"].iloc[-lb:].min())
                                        if desired_side == "BUY":
                                            sl_w = lo * (1.0 - pad)
                                            tp_w = entry_px_for_defaults + float(_risk_param(symbol, "SMC_RR", "2.0") or 2.0) * (entry_px_for_defaults - sl_w)
                                        else:
                                            sl_w = hi * (1.0 + pad)
                                            tp_w = entry_px_for_defaults - float(_risk_param(symbol, "SMC_RR", "2.0") or 2.0) * (sl_w - entry_px_for_defaults)
                                    else:
                                        # ATR widen
                                        n = int(_risk_param(symbol, "SMC_ATR_LEN", "14") or 14)
                                        high = df["high"].astype(float)
                                        low = df["low"].astype(float)
                                        close = df["close"].astype(float)
                                        prev_close = close.shift(1)
                                        tr = (high - low).to_frame(name="hl")
                                        tr["hc"] = (high - prev_close).abs()
                                        tr["lc"] = (low - prev_close).abs()
                                        tr_max = tr.max(axis=1)
                                        atr = float(tr_max.rolling(n, min_periods=n).mean().iloc[-1])
                                        mult0 = float(_risk_param(symbol, "SMC_ATR_MULT", "1.0") or 1.0)
                                        mult_w = mult0 * float(f)
                                        rr = float(_risk_param(symbol, "SMC_RR", "2.0") or 2.0)
                                        if desired_side == "BUY":
                                            sl_w = entry_px_for_defaults - mult_w * atr
                                            tp_w = entry_px_for_defaults + rr * (entry_px_for_defaults - sl_w)
                                        else:
                                            sl_w = entry_px_for_defaults + mult_w * atr
                                            tp_w = entry_px_for_defaults - rr * (sl_w - entry_px_for_defaults)
                                        if await _attempt_amend(sl_w, tp_w):
                                            break
                except Exception as e:
                    _push_err(symbol, timeframe, f"amend fallback error: {e}", "order_amend_fail", strategy_name)

                try:
                    # Determine entry price: prefer result; fallback to scanning open positions
                    # Prefer broker-reflected entry for journaling too
                    entry_px = None
                    for _ in range(6):
                        await asyncio.sleep(0.5)
                        got = False
                        for p in (ctd.get_open_positions() or []):
                            if (str(p.get("symbol_name", "")).upper() == sym_u) and ((p.get("direction") or "").lower() == desired_dir):
                                entry_px = p.get("entry_price")
                                got = True
                                break
                        if got and entry_px not in (None, 0, 0.0):
                            break
                    if entry_px in (None, 0, 0.0) and isinstance(result, dict):
                        try:
                            entry_px = float(result.get('price') or 0) or None
                        except Exception:
                            entry_px = None
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
        last_error=None,
        last_error_tag=None,
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



# ---------- status helpers ----------
# (existing code)
def _digits_for_symbol_id(symbol_id: int | None) -> int:
    try:
        sid = int(symbol_id) if symbol_id is not None else None
        if sid is not None and sid in ctd.symbol_money_digits_map:
            return int(ctd.symbol_money_digits_map[sid])
        if sid is not None and sid in ctd.symbol_digits_map:
            return int(ctd.symbol_digits_map[sid])
    except Exception:
        pass
    return 2

def _round_for_symbol(symbol_id: int | None, px: float | None) -> float | None:
    if px is None:
        return None
    try:
        d = _digits_for_symbol_id(symbol_id)
        return round(float(px), d)
    except Exception:
        return px


def _amend_tolerance(symbol_id: int | None) -> float:
    """Return tolerance for SL/TP verification; allow env overrides for coarse-precision symbols (indices/metals)."""
    base = 0.5 * (10 ** (-_digits_for_symbol_id(symbol_id)))
    override = None
    try:
        raw = os.getenv("SMC_AMEND_TOL")
        if raw:
            override = float(raw)
    except Exception:
        override = None
    try:
        sym = ctd.symbol_map.get(symbol_id)
        if sym:
            raw_sym = os.getenv(f"SMC_AMEND_TOL_{sym.upper()}")
            if raw_sym:
                override = max(float(raw_sym), override) if override is not None else float(raw_sym)
    except Exception:
        pass
    return max(base, override) if override is not None else base
