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
symbol_min_volume_map : dict[int, int] = {}   # {id: minimum volume in units}
symbol_step_volume_map: dict[int, int] = {}   # {id: step volume in units}

# Optional fallback if cTrader rejects symbol list (e.g., invalid credentials or maintenance)
_fallback_symbols_cfg = os.getenv("CTRADER_FALLBACK_SYMBOLS", "XAUUSD,EURUSD,GBPUSD,US500").strip()
FALLBACK_SYMBOLS = [s.strip().upper() for s in _fallback_symbols_cfg.split(",") if s.strip()]

# connection state
CONNECTED = False

_PRICE_FACTOR = 100_000
_UNITS_PER_LOT = 100_000           # base units (1 lot)
_VOLUME_PRECISION = 100            # cTrader volumes are expressed in 1/100 units
_API_VOLUME_PER_LOT = _UNITS_PER_LOT * _VOLUME_PRECISION  # 10,000,000

def _px(x):
    """Convert float price → cTrader int format (e.g., 1.0935 → 109350)."""
    return int(round(float(x) * _PRICE_FACTOR)) if x is not None else None


def _lots_to_units(lots: float | int | None) -> int | None:
    if lots is None:
        return None
    return int(round(float(lots) * _UNITS_PER_LOT))


def volume_lots_to_units(symbol_id: int, lots: float | int | None) -> int:
    try:
        lots_val = float(lots)
    except (TypeError, ValueError):
        raise ValueError("Lot size must be numeric") from None

    if lots_val <= 0:
        raise ValueError("Lot size must be greater than zero")

    api_volume = int(round(lots_val * _API_VOLUME_PER_LOT))
    min_api = symbol_min_volume_map.get(symbol_id) or (_API_VOLUME_PER_LOT // 100)
    step_api = symbol_step_volume_map.get(symbol_id) or min_api

    if api_volume < min_api:
        raise ValueError(
            f"Lot size too small for symbol; minimum is {min_api / _API_VOLUME_PER_LOT:.2f} lots"
        )

    if api_volume % step_api:
        raise ValueError(
            f"Lot size must align to step {step_api / _API_VOLUME_PER_LOT:.2f} lots"
        )

    print(
        f"[VOLUME] symbol_id={symbol_id} lots={lots_val} api_volume={api_volume} min={min_api} step={step_api}"
    )
    return api_volume

# ── helpers ────────────────────────────────────────────────────────────────
def pips_to_relative(pips: int, digits: int) -> int:
    """Convert pips → 1/100000 units (works for 2–5 digit symbols)."""
    return pips * 10 ** (6 - digits)

def on_error(failure):
    print("[ERROR]", failure)

# ── bootstrapping: symbols ─────────────────────────────────────────────────
def _install_fallback_symbols(reason: str | None = None):
    global symbol_map, symbol_name_to_id, symbol_digits_map, symbol_min_volume_map, symbol_step_volume_map
    print(f"[WARN] Using fallback symbols ({reason or 'unknown error'})")
    symbol_map.clear(); symbol_name_to_id.clear(); symbol_digits_map.clear()
    symbol_min_volume_map.clear(); symbol_step_volume_map.clear()
    for idx, name in enumerate(FALLBACK_SYMBOLS, start=1):
        symbol_map[idx] = name
        symbol_name_to_id[name] = idx
        symbol_digits_map[idx] = 5
        symbol_min_volume_map[idx] = _API_VOLUME_PER_LOT // 100  # 0.01 lot
        symbol_step_volume_map[idx] = _API_VOLUME_PER_LOT // 100
    if symbol_map:
        print(f"[INFO] Loaded {len(symbol_map)} fallback symbols: {', '.join(symbol_map.values())}")


def symbols_response_cb(res):
    global symbol_map, symbol_name_to_id, symbol_digits_map, symbol_min_volume_map, symbol_step_volume_map
    symbol_map.clear(); symbol_name_to_id.clear(); symbol_digits_map.clear()
    symbol_min_volume_map.clear(); symbol_step_volume_map.clear()

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
        min_api = max(1, int(min_vol_raw or (_API_VOLUME_PER_LOT // 100)))
        step_api = max(1, int(step_vol_raw or min_vol_raw or (_API_VOLUME_PER_LOT // 100)))
        symbol_min_volume_map[s.symbolId] = min_api
        symbol_step_volume_map[s.symbolId] = step_api
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
            rows.append(dict(
                symbol_name=symbol_map.get(td.symbolId, str(td.symbolId)),
                position_id=p.positionId,
                direction="buy" if td.tradeSide==ProtoOATradeSide.BUY else "sell",
                entry_price=getattr(p,"price",0),
                volume_lots=td.volume/100_000,
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

    # Absolute price fields for LIMIT/STOP
    if order_type.upper() == "LIMIT":
        if price is None:
            raise ValueError("Limit order requires price.")
        req.limitPrice = _px(price)
    elif order_type.upper() == "STOP":
        if price is None:
            raise ValueError("Stop order requires price.")
        req.stopPrice = _px(price)

    sl_price = stop_loss
    tp_price = take_profit

    if order_type.upper() in ("LIMIT", "STOP"):
        if stop_loss   is not None: req.stopLoss   = _px(stop_loss)
        if take_profit is not None: req.takeProfit = _px(take_profit)
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
            except Exception as e:
                info["ack_parse_error"] = str(e)
                print(f"[WARN] Unable to parse order acknowledgement: {e}")

            time.sleep(8)
            open_pos = get_open_positions()
            for p in open_pos:
                if (
                    p["symbol_name"].upper() == symbol_map[symbol_id].upper()
                    and p["direction"].upper() == side.upper()
                ):
                    info["status"] = "position_found"
                    info["position_id"] = p["position_id"]
                    if sl_price is None and tp_price is None:
                        return info
                    amend_res = modify_position_sltp(
                        client=client,
                        account_id=account_id,
                        position_id=p["position_id"],
                        stop_loss=sl_price,
                        take_profit=tp_price,
                    )
                    info["amend_submitted"] = True
                    info["amend_result"] = str(amend_res)
                    return info
            info.setdefault("status", "position_not_found")
            return info
        d.addCallback(_delayed_sltp)

    return d

# ── amend helpers ──────────────────────────────────────────────────────────
def modify_position_sltp(client, account_id, position_id, stop_loss=None, take_profit=None):
    req = ProtoOAAmendPositionSLTPReq(ctidTraderAccountId = account_id, positionId = position_id)
    if stop_loss   is not None: req.stopLoss   = _px(stop_loss)
    if take_profit is not None: req.takeProfit = _px(take_profit)
    return client.send(req)


def close_position(*, client, account_id, position_id, volume_lots=None):
    req = ProtoOAClosePositionReq(
        ctidTraderAccountId=account_id,
        positionId=position_id,
    )
    vol_units = _lots_to_units(volume_lots)
    if vol_units is not None:
        req.volume = vol_units
    return client.send(req)

def modify_pending_order_sltp(client, account_id, order_id, version, stop_loss=None, take_profit=None):
    req = ProtoOAAmendOrderReq(
        ctidTraderAccountId = account_id,
        orderId             = order_id,
        version             = version,
    )
    if stop_loss   is not None: req.stopLoss   = _px(stop_loss)
    if take_profit is not None: req.takeProfit = _px(take_profit)
    return client.send(req)

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
