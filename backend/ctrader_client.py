# backend/ctrader_client.py
# ---------------------------------------------------------------------------

from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAApplicationAuthReq,
    ProtoOAAccountAuthReq,
    ProtoOASymbolsListReq,
    ProtoOAReconcileReq,
    ProtoOAGetTrendbarsReq,
    ProtoOANewOrderReq,
    ProtoOAAmendOrderReq,
    ProtoOAAmendPositionSLTPReq,
    ProtoOAClosePositionReq,
)
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (
    ProtoOAOrderType,
    ProtoOATradeSide,
    ProtoOATrendbarPeriod,
)
from google.protobuf.json_format import MessageToDict

from twisted.internet import reactor
import asyncio
import os
import threading
import time
from datetime import datetime, timezone, timedelta
import calendar, time, threading, os, json, math

# Try to find a .env (root), otherwise fall back to backend/.env
from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv() or "backend/.env"
load_dotenv(dotenv_path)

# ── Credentials & client ───────────────────────────────────────────────────
CLIENT_ID     = os.getenv("CTRADER_CLIENT_ID")
CLIENT_SECRET = os.getenv("CTRADER_CLIENT_SECRET")
ACCESS_TOKEN  = os.getenv("CTRADER_ACCESS_TOKEN")
ACCOUNT_ID    = int(os.getenv("CTRADER_ACCOUNT_ID"))
HOST_TYPE     = (os.getenv("CTRADER_HOST_TYPE") or "demo").lower()

host = EndPoints.PROTOBUF_LIVE_HOST if HOST_TYPE == "live" else EndPoints.PROTOBUF_DEMO_HOST
client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)

# ── symbol maps ────────────────────────────────────────────────────────────
symbol_map            : dict[int, str] = {}   # {id: name}
symbol_name_to_id     : dict[str, int] = {}   # {name.upper(): id}
symbol_digits_map     : dict[int, int] = {}   # {id: digits}
symbol_money_digits_map: dict[int, int] = {}  # {id: moneyDigits}
symbol_min_volume_map : dict[int, int] = {}   # {id: minimum volume in API units}
symbol_step_volume_map: dict[int, int] = {}   # {id: step volume in API units}
symbol_max_volume_map: dict[int, int] = {}   # {id: maximum volume in units}
symbol_lot_size_map: dict[int, int] = {}   # {id: lot size in units}
symbol_min_verified: dict[int, bool] = {}     # {id: True if learned from broker error}
symbol_step_verified: dict[int, bool] = {}    # {id: True if learned from broker error}

# Optional fallback if cTrader rejects symbol list (e.g., invalid credentials or maintenance)
_fallback_symbols_cfg = os.getenv("CTRADER_FALLBACK_SYMBOLS", "XAUUSD,EURUSD,GBPUSD,US500").strip()
FALLBACK_SYMBOLS = [s.strip().upper() for s in _fallback_symbols_cfg.split(",") if s.strip()]

# connection state
CONNECTED = False

_PRICE_FACTOR = 100_000
_VOLUME_PRECISION = 10_000         # API volume units = 0.0001 lots (1 lot = 10,000 units)

# Track the last order's symbol so we can reconcile broker-side volume
# requirements if an immediate TRADING_BAD_VOLUME error arrives.
_LAST_ORDER_CTX: dict[str, int] = {"symbol_id": -1}

def _px(x):
    """Pass-through float price (no integer scaling)."""
    try:
        return float(x) if x is not None else None
    except Exception:
        return None

def _px_sym(symbol_id: int | None, x):
    """Return a float price rounded to per-symbol money digits (no integer scaling)."""
    if x is None:
        return None
    try:
        sid = int(symbol_id) if symbol_id is not None else None
        if sid is not None and sid in symbol_money_digits_map:
            digits = int(symbol_money_digits_map[sid])
        else:
            digits = int(symbol_digits_map.get(sid, 5)) if sid is not None else 5
    except Exception:
        digits = 5
    try:
        return round(float(x), digits)
    except Exception:
        return float(x)

def _decode_px(symbol_id: int | None, raw):
    """Decode cTrader price; avoid over-dividing indices like US30.
    Divide only if value looks like a scaled integer using moneyDigits."""
    if raw is None:
        return None
    try:
        v = float(raw)
    except Exception:
        return raw
    try:
        sid = int(symbol_id) if symbol_id is not None else None
        md = int(symbol_money_digits_map.get(sid, symbol_digits_map.get(sid, 2))) if sid is not None else 2
    except Exception:
        md = 2
    try:
        threshold = float(10 ** (md + 3))
    except Exception:
        threshold = 100000.0
    if v >= threshold:
        return v / (10 ** md)
    return v

def _price_factor_for_symbol(symbol_id: int | None) -> int:
    """Return price scaling factor: prefer moneyDigits if known; else digits; fallback 1e5."""
    try:
        if symbol_id is None:
            return _PRICE_FACTOR
        sid = int(symbol_id)
        if sid in symbol_money_digits_map:
            return 10 ** int(symbol_money_digits_map[sid])
        d = int(symbol_digits_map.get(sid, 5))
        return 10 ** d
    except Exception:
        return _PRICE_FACTOR

def normalize_sltp_for_side(*, side: str, entry_price: float | None, sl: float | None, tp: float | None) -> tuple[float | None, float | None]:
    """Return (sl, tp) corrected for order side using entry_price as a hint.

    - For BUY: SL must be < entry, TP must be > entry.
    - For SELL: SL must be > entry, TP must be < entry.
    - If both are provided but obviously reversed relative to entry, swap them.
    - If a single value violates the rule, drop it (return None) to avoid broker rejection.
    """
    try:
        s = (side or "").upper()
        e = float(entry_price) if entry_price is not None else None
        sl_v = None if sl is None else float(sl)
        tp_v = None if tp is None else float(tp)
    except Exception:
        return sl, tp

    if e is None:
        # Without entry hint, cannot validate; return as-is
        return sl, tp

    if sl_v is not None and tp_v is not None:
        if s == "BUY" and (sl_v > e and tp_v < e):
            sl_v, tp_v = tp_v, sl_v
        elif s == "SELL" and (sl_v < e and tp_v > e):
            sl_v, tp_v = tp_v, sl_v

    # Drop invalid singles after potential swap
    if s == "BUY":
        if sl_v is not None and sl_v >= e:
            sl_v = None
        if tp_v is not None and tp_v <= e:
            tp_v = None
    elif s == "SELL":
        if sl_v is not None and sl_v <= e:
            sl_v = None
        if tp_v is not None and tp_v >= e:
            tp_v = None

    return sl_v, tp_v


def _lots_to_units(lots: float | int | None, symbol_id: int) -> int | None:
    """Convert a lot amount to the cTrader API volume units.

    Notes:
    - cTrader OpenAPI expects volume as an integer in 0.0001-lot increments.
      In other words, 1.00 lot == 10,000 API volume units.
    - This is independent of symbol contract size; do NOT multiply by contract size here.
    """
    if lots is None:
        return None
    return int(round(float(lots) * _VOLUME_PRECISION))


def volume_lots_to_units(symbol_id: int, lots: float | int | None) -> int:
    """Convert lots from UI to cTrader API volume units and validate against symbol limits.

    - API volume = lots * 10_000 (integer)
    - Symbol metadata (min/step/max) is returned in 0.01-lot units ("cent-lots").
      Convert to API volume units by multiplying by 100.
    """
    try:
        lots_val = float(lots)
    except (TypeError, ValueError):
        raise ValueError("Lot size must be numeric") from None

    if lots_val <= 0:
        raise ValueError("Lot size must be greater than zero")

    api_volume = int(round(lots_val * _VOLUME_PRECISION))

    min_api = symbol_min_volume_map.get(symbol_id)
    step_api = symbol_step_volume_map.get(symbol_id)
    max_api = symbol_max_volume_map.get(symbol_id)

    if min_api is not None and api_volume < min_api:
        raise ValueError(
            f"Lot size too small; minimum is {min_api / _VOLUME_PRECISION:.2f} lots"
        )

    if max_api is not None and api_volume > max_api:
        raise ValueError(
            f"Lot size too large; maximum is {max_api / _VOLUME_PRECISION:.2f} lots"
        )

    if step_api:
        if api_volume % step_api:
            raise ValueError(
                f"Lot size must align to step {step_api / _VOLUME_PRECISION:.2f} lots"
            )

    print(
        f"[VOLUME] symbol_id={symbol_id} lots={lots_val} api_volume={api_volume} (api-units) min={min_api} step={step_api} max={max_api}"
    )
    return api_volume


def coerce_volume_lots_to_units(symbol_id: int, lots: float | int | None) -> tuple[int, float]:
    """Coerce a requested lot amount up to broker-allowed min/step and return API units.

    Returns (api_volume, lots_final). Rounds up to nearest allowed step and clamps
    to [min,max] if limits are known. Does not raise on small/misaligned sizes.
    """
    try:
        lots_val = float(lots)
    except (TypeError, ValueError):
        raise ValueError("Lot size must be numeric") from None

    if lots_val <= 0:
        raise ValueError("Lot size must be greater than zero")

    api_volume = int(round(lots_val * _VOLUME_PRECISION))

    min_api = symbol_min_volume_map.get(symbol_id)
    step_api = symbol_step_volume_map.get(symbol_id)
    max_api = symbol_max_volume_map.get(symbol_id)
    hard_min = bool(symbol_min_verified.get(symbol_id))
    hard_step = bool(symbol_step_verified.get(symbol_id))
    base_floor_api = int(round(0.01 * _VOLUME_PRECISION))  # 0.01 lots

    # Apply min: treat metadata mins as soft; enforce only verified mins.
    if hard_min and (min_api is not None) and api_volume < min_api:
        api_volume = int(min_api)
    else:
        if api_volume < base_floor_api:
            api_volume = base_floor_api

    # Snap to step by rounding up
    if step_api and step_api > 0:
        if hard_step or step_api <= base_floor_api:
            rem = api_volume % step_api
            if rem:
                api_volume += step_api - rem

    # Apply max (after step rounding)
    if max_api is not None and api_volume > max_api:
        api_volume = int(max_api)

    lots_final = api_volume / _VOLUME_PRECISION
    print(
        f"[VOLUME] COERCE symbol_id={symbol_id} requested_lots={lots_val} -> lots={lots_final:.2f} api_volume={api_volume} min={min_api} step={step_api} max={max_api} hard_min={hard_min} hard_step={hard_step}"
    )
    return api_volume, lots_final

# ── helpers ────────────────────────────────────────────────────────────────
def pips_to_relative(pips: int, digits: int) -> int:
    """Convert pips → 1/100000 units (works for 2–5 digit symbols)."""
    return pips * 10 ** (6 - digits)

def on_error(failure):
    print("[ERROR]", failure)

# ── bootstrapping: symbols ─────────────────────────────────────────────────
def _install_fallback_symbols(reason: str | None = None):
    global symbol_map, symbol_name_to_id, symbol_digits_map, symbol_min_volume_map, symbol_step_volume_map, symbol_max_volume_map, symbol_lot_size_map, symbol_min_verified, symbol_step_verified
    print(f"[WARN] Using fallback symbols ({reason or 'unknown error'})")
    symbol_map.clear(); symbol_name_to_id.clear(); symbol_digits_map.clear()
    symbol_min_volume_map.clear(); symbol_step_volume_map.clear(); symbol_max_volume_map.clear(); symbol_lot_size_map.clear(); symbol_min_verified.clear(); symbol_step_verified.clear()
    default_min = int(round(0.01 * _VOLUME_PRECISION))
    default_max = int(round(100.00 * _VOLUME_PRECISION))
    for idx, name in enumerate(FALLBACK_SYMBOLS, start=1):
        symbol_map[idx] = name
        symbol_name_to_id[name] = idx
        symbol_digits_map[idx] = 5
        # Assume 0.01 lot minimum/step when metadata is unavailable
        symbol_lot_size_map[idx] = 100_000 if name in ["EURUSD", "GBPUSD"] else 100
        symbol_min_volume_map[idx] = default_min
        symbol_step_volume_map[idx] = default_min
        symbol_min_verified[idx] = False
        symbol_step_verified[idx] = False
        symbol_max_volume_map[idx] = default_max
    if symbol_map:
        print(f"[INFO] Loaded {len(symbol_map)} fallback symbols: {', '.join(symbol_map.values())}")

def symbols_response_cb(res):
    global symbol_map, symbol_name_to_id, symbol_digits_map, symbol_min_volume_map, symbol_step_volume_map, symbol_max_volume_map, symbol_lot_size_map, symbol_min_verified, symbol_step_verified
    symbol_map.clear(); symbol_name_to_id.clear(); symbol_digits_map.clear()
    symbol_min_volume_map.clear(); symbol_step_volume_map.clear(); symbol_max_volume_map.clear(); symbol_lot_size_map.clear(); symbol_min_verified.clear(); symbol_step_verified.clear()

    symbols = Protobuf.extract(res)
    # Some responses are error envelopes instead of the expected list
    if hasattr(symbols, "errorCode") or symbols.__class__.__name__ == "ProtoOAErrorRes":
        err = f"{getattr(symbols, 'errorCode', 'ERR')} {getattr(symbols, 'description', '').strip()}".strip()
        _install_fallback_symbols(err or "ProtoOAErrorRes")
        return

    loaded = 0
    for s in getattr(symbols, "symbol", []):
        digits = getattr(s, "digits", getattr(s, "pipPosition", 5))
        symbol_map[s.symbolId]                  = s.symbolName
        symbol_name_to_id[s.symbolName.upper()] = s.symbolId
        symbol_digits_map[s.symbolId]           = digits
        min_vol_raw = getattr(s, "minVolume", None) or getattr(s, "min_volume", None)
        step_vol_raw = getattr(s, "stepVolume", None) or getattr(s, "step_volume", None)
        max_vol_raw = getattr(s, "maxVolume", None) or getattr(s, "max_volume", None)
        # Symbol volume thresholds are provided by API in cent-lots (0.01-lot) units; convert to API units (0.0001 lots) by multiplying by 100.
        # Fall back conservatively if missing (assume 1.00 lot min/max reasonable defaults).
        min_api = (
            int(min_vol_raw) * 100 if min_vol_raw is not None else int(round(1.00 * _VOLUME_PRECISION))
        )
        step_api = (
            int(step_vol_raw) * 100 if step_vol_raw is not None else min_api
        )
        max_api = (
            int(max_vol_raw) * 100 if max_vol_raw is not None else int(round(100.00 * _VOLUME_PRECISION))
        )
        
        symbol_min_volume_map[s.symbolId] = min_api
        symbol_step_volume_map[s.symbolId] = step_api
        symbol_max_volume_map[s.symbolId] = max_api
        symbol_min_verified[s.symbolId] = False
        symbol_step_verified[s.symbolId] = False
        loaded += 1

    if loaded == 0:
        _install_fallback_symbols("empty symbol list")
    else:
        print(f"[DEBUG] Loaded {loaded} symbols.")

def account_auth_cb(_):
    req = ProtoOASymbolsListReq(
        ctidTraderAccountId=ACCOUNT_ID,
        includeArchivedSymbols=False,
    )
    client.send(req).addCallbacks(symbols_response_cb, on_error)

def app_auth_cb(_):
    req = ProtoOAAccountAuthReq(
        ctidTraderAccountId=ACCOUNT_ID,
        accessToken=ACCESS_TOKEN,
    )
    client.send(req).addCallbacks(account_auth_cb, on_error)

def _on_connected(_):
    global CONNECTED
    CONNECTED = True
    req = ProtoOAApplicationAuthReq(clientId=CLIENT_ID, clientSecret=CLIENT_SECRET)
    client.send(req).addCallbacks(app_auth_cb, on_error)

def _on_disconnected(c, reason):
    global CONNECTED
    CONNECTED = False
    print("[INFO] Disconnected:", reason)


def _log_event(event) -> None:
    try:
        payload = MessageToDict(event, preserving_proto_field_name=True)
    except Exception as e:
        payload = {"decode_error": str(e)}
    name = getattr(event, "__class__", type("x", (), {})).__name__
    summary = _format_payload(payload)

    if name == "ProtoOAExecutionEvent":
        print(f"[CTRADER EXECUTION] {summary}")
    elif name == "ProtoOAErrorRes":
        print(f"[CTRADER ERROR] {summary}")
    elif name == "ProtoOAOrderErrorEvent":
        # Opportunistically adjust volume constraints if broker rejects for BAD_VOLUME
        print(f"[CTRADER ORDER_ERROR] {summary}")
        try:
            err = (payload or {}).get("errorCode") or ""
            desc = (payload or {}).get("description") or ""
            if str(err).upper() == "TRADING_BAD_VOLUME" and desc:
                import re
                m = re.search(r"minimum allowed volume\s*=\s*([0-9]+(?:\.[0-9]+)?)", str(desc), re.IGNORECASE)
                if m:
                    # Error text uses cent-lots (0.01-lot) units; convert to API units (x100)
                    min_api = int(round(float(m.group(1)) * 100))
                    sid = int(_LAST_ORDER_CTX.get("symbol_id", -1))
                    if sid in symbol_map:
                        prev = symbol_min_volume_map.get(sid)
                        symbol_min_volume_map[sid] = max(min_api, prev or 0)
                        symbol_min_verified[sid] = True
                        print(f"[VOLUME] Updated min for {symbol_map[sid]} to {min_api} (API units)")
                m2 = re.search(r"step\s*=?\s*([0-9]+(?:\.[0-9]+)?)", str(desc), re.IGNORECASE)
                if m2:
                    # Error text uses cent-lots; convert to API units (x100)
                    step_api = int(round(float(m2.group(1)) * 100))
                    sid = int(_LAST_ORDER_CTX.get("symbol_id", -1))
                    if sid in symbol_map and step_api > 0:
                        symbol_step_volume_map[sid] = step_api
                        symbol_step_verified[sid] = True
                        print(f"[VOLUME] Updated step for {symbol_map[sid]} to {step_api} (API units)")
        except Exception as e:
            print(f"[WARN] Unable to parse order error for volume hints: {e}")
    elif name == "ProtoOAAccountLogoutRes":
        print(f"[CTRADER LOGOUT] {summary}")
    elif name == "ProtoOAGetTrendbarsRes" and isinstance(payload, dict):
        bars = payload.get("trendbar") or payload.get("trendBar") or []
        count = len(bars) if isinstance(bars, list) else 0
        first_ts = last_ts = None
        if count:
            first = bars[0] if isinstance(bars[0], dict) else {}
            last = bars[-1] if isinstance(bars[-1], dict) else {}
            first_ts = first.get("utcTimestampInMinutes") or first.get("utc_timestamp_in_minutes")
            last_ts = last.get("utcTimestampInMinutes") or last.get("utc_timestamp_in_minutes")
        print(f"[CTRADER TREND] bars={count} ts_range={first_ts}->{last_ts}")
    else:
        print(f"[CTRADER EVENT] {name}: {summary}")


def _format_payload(payload) -> str:
    try:
        txt = json.dumps(payload, ensure_ascii=False)
    except Exception:
        txt = str(payload)
    if len(txt) > 600:
        return f"{txt[:600]}... (truncated, {len(txt)} chars)"
    return txt


def init_client():
    client.setConnectedCallback(_on_connected)
    client.setDisconnectedCallback(_on_disconnected)
    def _on_message(_client, message):
        try:
            event = Protobuf.extract(message)
        except Exception as e:
            print(f"[CTRADER EVENT] decode_error: {e}")
            return
        _log_event(event)

    client.setMessageReceivedCallback(_on_message)
    client.startService()
    reactor.run(installSignalHandlers=False)

def is_connected() -> bool:
    return CONNECTED

# ── OHLC fetch (local event; handles errors; timeout) ──────────────────────
# --- replace get_ohlc_data() body with this version ---
def get_ohlc_data(symbol: str, tf: str = "D1", n: int = 10):
    ev = threading.Event()
    out: list[dict] = []
    err_txt = None

    sid = symbol_name_to_id.get(symbol.upper())
    if sid is None:
        raise ValueError(f"Unknown symbol '{symbol}'")

    now = datetime.utcnow()
    req = ProtoOAGetTrendbarsReq(
        symbolId=sid,
        ctidTraderAccountId=ACCOUNT_ID,
        period=getattr(ProtoOATrendbarPeriod, tf),
        fromTimestamp=int(calendar.timegm((now - timedelta(weeks=52)).utctimetuple())) * 1000,
        toTimestamp=int(calendar.timegm(now.utctimetuple())) * 1000,
    )

    def _tb(tb):
        ts = datetime.fromtimestamp(tb.utcTimestampInMinutes * 60, timezone.utc)
        return dict(
            time=ts.isoformat(),
            open=(tb.low + tb.deltaOpen)/100_000,
            high=(tb.low + tb.deltaHigh)/100_000,
            low=tb.low/100_000,
            close=(tb.low + tb.deltaClose)/100_000,
            volume=tb.volume,
        )

    def _ok(res):
        nonlocal err_txt
        obj = Protobuf.extract(res)
        # Got an API error, not trendbars
        if getattr(obj, "__class__", type("x",(object,),{})).__name__ == "ProtoOAErrorRes" or hasattr(obj, "errorCode"):
            err_txt = f"{getattr(obj,'errorCode','ERR')} {getattr(obj,'description','')}".strip()
            ev.set(); return
        try:
            out.extend(map(_tb, obj.trendbar))
        finally:
            ev.set()

    def _err(f):
        nonlocal err_txt
        err_txt = str(f); ev.set()

    d = client.send(req, timeout=15)   # a bit longer
    d.addCallbacks(_ok, _err)

    if not ev.wait(15):
        err_txt = err_txt or "trendbars timeout"

    if not out:
        raise RuntimeError(f"trendbars error: {err_txt or 'empty response'}")

    return out[-n:] if n else out


# --- replace get_pending_orders() and get_open_positions() callbacks similarly ---
def get_open_positions():
    ev = threading.Event(); rows=[]; err=None
    def _ok(res):
        nonlocal err
        obj = Protobuf.extract(res)
        if getattr(obj,"__class__",type("x",(object,),{})).__name__ == "ProtoOAErrorRes" or hasattr(obj,"errorCode"):
            err = f"{getattr(obj,'errorCode','ERR')} {getattr(obj,'description','')}".strip(); ev.set(); return
        for p in obj.position:
            td = p.tradeData
            sid = getattr(td, "symbolId", None)
            md = getattr(p, "moneyDigits", None)
            try:
                if md is not None:
                    md_int = int(md)
                    # Track money digits per symbol
                    if sid is not None:
                        symbol_money_digits_map[int(sid)] = md_int
                    pf = 10 ** md_int
                else:
                    pf = _price_factor_for_symbol(sid)
            except Exception:
                pf = _price_factor_for_symbol(sid)
            rows.append(dict(
                symbol_name=symbol_map.get(sid, str(sid)),
                symbol_id=sid,
                digits=symbol_digits_map.get(sid),
                money_digits=symbol_money_digits_map.get(int(sid)) if sid is not None else None,
                position_id=p.positionId,
                direction="buy" if td.tradeSide==ProtoOATradeSide.BUY else "sell",
                entry_price=_decode_px(sid, getattr(p, "price", 0)),
                volume_lots=td.volume/_VOLUME_PRECISION,
                stop_loss=_decode_px(sid, getattr(p, "stopLoss", None)) if getattr(p, "stopLoss", None) not in (None, 0) else None,
                take_profit=_decode_px(sid, getattr(p, "takeProfit", None)) if getattr(p, "takeProfit", None) not in (None, 0) else None,
            ))
        ev.set()
    def _err(f): nonlocal err; err=str(f); ev.set()
    d = client.send(ProtoOAReconcileReq(ctidTraderAccountId=ACCOUNT_ID), timeout=10)
    d.addCallbacks(_ok,_err); ev.wait(10)
    return rows  # (log err if you like)


def get_pending_orders():
    ev = threading.Event(); out=[]; err=None
    def _ok(res):
        nonlocal err
        obj = Protobuf.extract(res)
        if getattr(obj,"__class__",type("x",(object,),{})).__name__ == "ProtoOAErrorRes" or hasattr(obj,"errorCode"):
            err = f"{getattr(obj,'errorCode','ERR')} {getattr(obj,'description','')}".strip(); ev.set(); return
        out.extend(obj.order); ev.set()
    def _err(f): nonlocal err; err=str(f); ev.set()
    d = client.send(ProtoOAReconcileReq(ctidTraderAccountId=ACCOUNT_ID), timeout=10)
    d.addCallbacks(_ok,_err); ev.wait(10)
    return out


# ── place order ───────────────────────────────────────────────────────────-
def place_order(
    *, client, account_id, symbol_id,
    order_type, side, volume,
    price=None, stop_loss=None, take_profit=None,
    client_msg_id=None,
):
    req = ProtoOANewOrderReq(
        ctidTraderAccountId=account_id,
        symbolId=symbol_id,
        orderType=ProtoOAOrderType.Value(order_type.upper()),
        tradeSide=ProtoOATradeSide.Value(side.upper()),
        volume=int(volume),  # 1 lot = 10,000,000 units (already scaled by caller)
    )

    # Capture context for potential TRADING_BAD_VOLUME correction
    try:
        global _LAST_ORDER_CTX
        _LAST_ORDER_CTX = {"symbol_id": int(symbol_id), "ts": int(time.time())}
    except Exception:
        pass

    # Absolute price fields for LIMIT/STOP
    if order_type.upper() == "LIMIT":
        if price is None:
            raise ValueError("Limit order requires price.")
        req.limitPrice = _px_sym(symbol_id, price)
    elif order_type.upper() == "STOP":
        if price is None:
            raise ValueError("Stop order requires price.")
        req.stopPrice = _px_sym(symbol_id, price)

    sl_price = stop_loss
    tp_price = take_profit

    if order_type.upper() in ("LIMIT", "STOP"):
        if stop_loss   is not None: req.stopLoss   = _px_sym(symbol_id, stop_loss)
        if take_profit is not None: req.takeProfit = _px_sym(symbol_id, take_profit)
    else:
        # MARKET: defer SL/TP to post-fill amendment so we can use absolute prices
        stop_loss = None
        take_profit = None

    print(
        f"[DEBUG] Sending order: {order_type=} {side=} volume={volume} price={price} SL={stop_loss} TP={take_profit}"
    )
    d = client.send(req, client_msg_id=client_msg_id, timeout=12)

    # Optionally amend SL/TP post-fill for MARKET
    if order_type.upper() == "MARKET":
        def _delayed_sltp(res):
            info: dict[str, object] = {}
            pos_id = None
            sid_hint = symbol_id
            entry_hint = None
            try:
                event = Protobuf.extract(res)
                info["ack"] = MessageToDict(event, preserving_proto_field_name=True)
                if getattr(event, "rejectReason", 0):
                    info["status"] = "order_rejected"
                    info["reject_reason"] = int(getattr(event, "rejectReason", 0))
                    print(f"[ERROR] Order rejected: {info['reject_reason']} ack={info['ack']}")
                    return info
                if getattr(event, "executionType", None) is not None:
                    info["execution_type"] = int(getattr(event, "executionType"))
                if getattr(event, "orderStatus", None) is not None:
                    info["order_status"] = int(getattr(event, "orderStatus"))
                # Try to capture the created/filled position directly from the event
                pos = getattr(event, "position", None)
                if pos is not None:
                    try:
                        pos_id = int(getattr(pos, "positionId", 0) or 0)
                    except Exception:
                        pos_id = None
                    try:
                        td = getattr(pos, "tradeData", None)
                        sid_hint = int(getattr(td, "symbolId", sid_hint) or sid_hint)
                    except Exception:
                        sid_hint = symbol_id
                    try:
                        md = getattr(pos, "moneyDigits", None)
                        if md is not None:
                            symbol_digits_map[int(sid_hint)] = int(md)
                    except Exception:
                        pass
                    try:
                        entry_hint = float(getattr(pos, "price", 0.0) or 0.0)
                    except Exception:
                        entry_hint = None
            except Exception as e:
                info["ack_parse_error"] = str(e)
                print(f"[WARN] Unable to parse order acknowledgement: {e}")

            # Run the amend in a background thread so we don't block the reactor
            def _amend_worker():
                try:
                    # Prefer direct amend by positionId only if we have a filled price
                    can_direct = bool(pos_id) and (entry_hint not in (None, 0, 0.0))
                    if can_direct:
                        entry_for_norm = entry_hint
                        sl_norm, tp_norm = normalize_sltp_for_side(side=side, entry_price=entry_for_norm, sl=sl_price, tp=tp_price)
                        if sl_norm is None and tp_norm is None:
                            print("[INFO] No valid SL/TP provided for amend; skipping.")
                            return
                        amend_def = modify_position_sltp(
                            client=client,
                            account_id=account_id,
                            position_id=pos_id,
                            stop_loss=sl_norm,
                            take_profit=tp_norm,
                            symbol_id=sid_hint,
                        )
                        ack = wait_for_deferred(amend_def, timeout=25)
                        # Verify broker state reflects SL/TP
                        ok = False
                        sl_ok = False
                        tp_ok = False
                        for _ in range(5):
                            time.sleep(0.5)
                            for p in (get_open_positions() or []):
                                if int(p.get("position_id") or 0) == int(pos_id):
                                    sl_ok = (sl_norm is None) or (abs(float(p.get("stop_loss") or 0) - float(sl_norm)) < 1e-6)
                                    tp_ok = (tp_norm is None) or (abs(float(p.get("take_profit") or 0) - float(tp_norm)) < 1e-6)
                                    ok = sl_ok and tp_ok
                                    break
                            if ok:
                                break
                        print(f"[INFO] SL/TP amend sent for position {pos_id} ({symbol_map.get(sid_hint, sid_hint)}): SL={sl_norm} TP={tp_norm} entry={entry_hint} ack={ack} verified={ok} (sl_ok={sl_ok} tp_ok={tp_ok})")
                        if ok:
                            return
                        # Partial repair: try to amend only missing side(s)
                        sl_missing = (sl_norm is not None) and (not sl_ok)
                        tp_missing = (tp_norm is not None) and (not tp_ok)
                        if not (sl_missing or tp_missing):
                            return
                        amend_def2 = modify_position_sltp(
                            client=client,
                            account_id=account_id,
                            position_id=pos_id,
                            stop_loss=(sl_norm if sl_missing else None),
                            take_profit=(tp_norm if tp_missing else None),
                            symbol_id=sid_hint,
                        )
                        ack2 = wait_for_deferred(amend_def2, timeout=25)
                        # Verify again
                        ok2 = False
                        sl_ok2 = sl_ok
                        tp_ok2 = tp_ok
                        for _ in range(5):
                            time.sleep(0.5)
                            for p in (get_open_positions() or []):
                                if int(p.get("position_id") or 0) == int(pos_id):
                                    if sl_missing:
                                        sl_ok2 = (abs(float(p.get("stop_loss") or 0) - float(sl_norm)) < 1e-6)
                                    if tp_missing:
                                        tp_ok2 = (abs(float(p.get("take_profit") or 0) - float(tp_norm)) < 1e-6)
                                    ok2 = sl_ok2 and tp_ok2
                                    break
                            if ok2:
                                break
                        print(f"[INFO] SL/TP partial amend for position {pos_id}: ack2={ack2} verified={ok2} (sl_ok={sl_ok2} tp_ok={tp_ok2})")
                        return
                except Exception as e:
                    print(f"[ERROR] Failed direct SL/TP amend by positionId: {e}")
                    # Fall through to scanning

                # Fallback: scan for the opened position and amend (time-budgeted)
                interval = max(0.5, float(AMEND_POLL_INTERVAL))
                try:
                    max_dur = float(os.getenv("SLTP_AMEND_MAX_DURATION_SEC", "600"))
                except Exception:
                    max_dur = 600.0
                deadline = time.time() + max(30.0, max_dur)
                while time.time() < deadline:
                    time.sleep(interval)
                    try:
                        open_pos = get_open_positions()
                    except Exception as e:
                        print(f"[WARN] get_open_positions failed: {e}")
                        open_pos = []
                    for p in open_pos:
                        is_target = False
                        if pos_id:
                            is_target = int(p.get("position_id") or 0) == pos_id
                        else:
                            is_target = (
                                str(p.get("symbol_name", "")).upper() == str(symbol_map.get(symbol_id, symbol_id)).upper()
                                and str(p.get("direction", "")).upper() == side.upper()
                            )
                        if is_target:
                            entry_px = p.get("entry_price")
                            sl_norm, tp_norm = normalize_sltp_for_side(side=side, entry_price=entry_px, sl=sl_price, tp=tp_price)
                            if sl_norm is None and tp_norm is None:
                                print("[INFO] No valid SL/TP provided for amend; skipping.")
                                return
                            try:
                                amend_def = modify_position_sltp(
                                    client=client,
                                    account_id=account_id,
                                    position_id=p.get("position_id"),
                                    stop_loss=sl_norm,
                                    take_profit=tp_norm,
                                    symbol_id=p.get("symbol_id", symbol_id),
                                )
                                ack = wait_for_deferred(amend_def, timeout=25)
                                # Verify broker state reflects SL/TP
                                ok = False
                                sl_ok = False
                                tp_ok = False
                                for _ in range(5):
                                    time.sleep(0.5)
                                    for p2 in (get_open_positions() or []):
                                        if int(p2.get("position_id") or 0) == int(p.get("position_id") or 0):
                                            sl_ok = (sl_norm is None) or (abs(float(p2.get("stop_loss") or 0) - float(sl_norm)) < 1e-6)
                                            tp_ok = (tp_norm is None) or (abs(float(p2.get("take_profit") or 0) - float(tp_norm)) < 1e-6)
                                            ok = sl_ok and tp_ok
                                            break
                                    if ok:
                                        break
                                print(f"[INFO] SL/TP amend sent for position {p.get('position_id')} ({p.get('symbol_name')}): SL={sl_norm} TP={tp_norm} entry={entry_px} ack={ack} verified={ok} (sl_ok={sl_ok} tp_ok={tp_ok})")
                                if ok:
                                    return
                                # Partial repair: try to amend only missing side(s)
                                sl_missing = (sl_norm is not None) and (not sl_ok)
                                tp_missing = (tp_norm is not None) and (not tp_ok)
                                if sl_missing or tp_missing:
                                    amend_def2 = modify_position_sltp(
                                        client=client,
                                        account_id=account_id,
                                        position_id=p.get("position_id"),
                                        stop_loss=(sl_norm if sl_missing else None),
                                        take_profit=(tp_norm if tp_missing else None),
                                        symbol_id=p.get("symbol_id", symbol_id),
                                    )
                                    ack2 = wait_for_deferred(amend_def2, timeout=25)
                                    # Verify again
                                    ok2 = False
                                    sl_ok2 = sl_ok
                                    tp_ok2 = tp_ok
                                    for _ in range(5):
                                        time.sleep(0.5)
                                        for p3 in (get_open_positions() or []):
                                            if int(p3.get("position_id") or 0) == int(p.get("position_id") or 0):
                                                if sl_missing:
                                                    sl_ok2 = (abs(float(p3.get("stop_loss") or 0) - float(sl_norm)) < 1e-6)
                                                if tp_missing:
                                                    tp_ok2 = (abs(float(p3.get("take_profit") or 0) - float(tp_norm)) < 1e-6)
                                                ok2 = sl_ok2 and tp_ok2
                                                break
                                        if ok2:
                                            break
                                    print(f"[INFO] SL/TP partial amend for position {p.get('position_id')}: ack2={ack2} verified={ok2} (sl_ok={sl_ok2} tp_ok={tp_ok2})")
                            except Exception as e:
                                print(f"[ERROR] Failed to submit SL/TP amendment: {e}")
                            return
                print("[WARN] Could not find opened position to amend SL/TP within max duration.")

            try:
                threading.Thread(target=_amend_worker, daemon=True).start()
            except Exception as e:
                print(f"[WARN] Failed to start amend worker: {e}")
            return info
        d.addCallback(_delayed_sltp)

    return d

def _decode_px_1e5(raw):
    if raw in (None, 0):
        return None
    try:
        return float(raw) / _PRICE_FACTOR
    except Exception:
        try:
            return float(raw)
        except Exception:
            return None

# ── amend helpers ──────────────────────────────────────────────────────────
def modify_position_sltp(client, account_id, position_id, stop_loss=None, take_profit=None, symbol_id=None):
    req = ProtoOAAmendPositionSLTPReq(ctidTraderAccountId = account_id, positionId = position_id)
    if stop_loss   is not None: req.stopLoss   = _px_sym(symbol_id, stop_loss)
    if take_profit is not None: req.takeProfit = _px_sym(symbol_id, take_profit)
    # Increase timeout to avoid default 5s cancellation
    return client.send(req, timeout=20)


def close_position(*, client, account_id, position_id, volume_lots=None):
    req = ProtoOAClosePositionReq(
        ctidTraderAccountId=account_id,
        positionId=position_id,
    )
    # Volume is specified in 0.0001 lots for the API.
    vol_units = _lots_to_units(volume_lots, -1) if volume_lots is not None else None
    if vol_units is not None:
        req.volume = vol_units
    return client.send(req, timeout=20)

def modify_pending_order_sltp(client, account_id, order_id, version, stop_loss=None, take_profit=None, symbol_id=None):
    req = ProtoOAAmendOrderReq(
        ctidTraderAccountId = account_id,
        orderId             = order_id,
        version             = version,
    )
    if stop_loss   is not None: req.stopLoss   = _px_sym(symbol_id, stop_loss)
    if take_profit is not None: req.takeProfit = _px_sym(symbol_id, take_profit)
    return client.send(req, timeout=20)

# ── blocking helper used by FastAPI layer ─────────────────────────────────
def wait_for_deferred(deferred, timeout=40):
    ev = threading.Event()
    outcome = {"resolved": False, "value": None, "error": None}

    def _ok(res):
        outcome["resolved"] = True
        outcome["value"] = res
        ev.set()
        return res

    def _err(err):
        outcome["resolved"] = True
        outcome["error"] = err
        ev.set()
        return err

    deferred.addCallbacks(_ok, _err)

    if not ev.wait(timeout):
        msg = "deferred timeout"
        print(f"[FATAL] Deferred result timeout or failure: {msg}")
        return {"status": "failed", "error": msg}

    if outcome["error"] is not None:
        err = outcome["error"]
        err_txt = getattr(getattr(err, "value", err), "message", None) or str(err)
        print(f"[FATAL] Deferred result timeout or failure: {err_txt}")
        return {"status": "failed", "error": err_txt}

    return outcome["value"]
# Polling config for post-fill SL/TP amendment
try:
    AMEND_POLL_ATTEMPTS = int(os.getenv("SLTP_AMEND_POLL_ATTEMPTS", "25"))  # ~50s @ 2s
except Exception:
    AMEND_POLL_ATTEMPTS = 25
try:
    AMEND_POLL_INTERVAL = float(os.getenv("SLTP_AMEND_POLL_INTERVAL", "2.0"))
except Exception:
    AMEND_POLL_INTERVAL = 2.0
