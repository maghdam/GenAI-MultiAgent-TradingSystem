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
)
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (
    ProtoOAOrderType,
    ProtoOATradeSide,
    ProtoOATrendbarPeriod,
)
from twisted.internet import reactor
from datetime import datetime, timezone, timedelta
import calendar, time, threading, os

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
symbol_map        : dict[int, str] = {}   # {id: name}
symbol_name_to_id : dict[str, int] = {}   # {name.upper(): id}
symbol_digits_map : dict[int, int] = {}   # {id: digits}

# connection state
CONNECTED = False

_PRICE_FACTOR = 100_000

def _px(x):
    """Convert float price → cTrader int format (e.g., 1.0935 → 109350)."""
    return int(round(float(x) * _PRICE_FACTOR)) if x is not None else None

# ── helpers ────────────────────────────────────────────────────────────────
def pips_to_relative(pips: int, digits: int) -> int:
    """Convert pips → 1/100000 units (works for 2–5 digit symbols)."""
    return pips * 10 ** (6 - digits)

def on_error(failure):
    print("[ERROR]", failure)

# ── bootstrapping: symbols ─────────────────────────────────────────────────
def symbols_response_cb(res):
    global symbol_map, symbol_name_to_id, symbol_digits_map
    symbol_map.clear(); symbol_name_to_id.clear(); symbol_digits_map.clear()

    symbols = Protobuf.extract(res)
    for s in symbols.symbol:
        digits = getattr(s, "digits", getattr(s, "pipPosition", 5))
        symbol_map[s.symbolId]                  = s.symbolName
        symbol_name_to_id[s.symbolName.upper()] = s.symbolId
        symbol_digits_map[s.symbolId]           = digits

    print(f"[DEBUG] Loaded {len(symbol_map)} symbols.")

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

def init_client():
    client.setConnectedCallback(_on_connected)
    client.setDisconnectedCallback(_on_disconnected)
    client.setMessageReceivedCallback(lambda c, m: None)
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
                volume_lots=td.volume/10_000_000,
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
        req.limitPrice = float(price)
    elif order_type.upper() == "STOP":
        if price is None:
            raise ValueError("Stop order requires price.")
        req.stopPrice = float(price)

    if order_type.upper() in ("LIMIT", "STOP"):
        if stop_loss   is not None: req.stopLoss   = float(stop_loss)
        if take_profit is not None: req.takeProfit = float(take_profit)
    else:
        # MARKET: relative distances (1/100000 units)
        digits = symbol_digits_map.get(symbol_id, 5)
        if stop_loss   is not None: req.relativeStopLoss   = pips_to_relative(int(stop_loss),   digits)
        if take_profit is not None: req.relativeTakeProfit = pips_to_relative(int(take_profit), digits)

    print(
        f"[DEBUG] Sending order: {order_type=} {side=} price={price} SL={stop_loss} TP={take_profit}"
    )
    d = client.send(req, client_msg_id=client_msg_id, timeout=12)

    # Optionally amend SL/TP post-fill for MARKET
    if order_type.upper() == "MARKET":
        def _delayed_sltp(_):
            time.sleep(8)
            open_pos = get_open_positions()
            for p in open_pos:
                if (
                    p["symbol_name"].upper() == symbol_map[symbol_id].upper()
                    and p["direction"].upper() == side.upper()
                ):
                    return modify_position_sltp(
                        client=client,
                        account_id=account_id,
                        position_id=p["position_id"],
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                    )
            return {"status": "position_not_found"}
        d.addCallback(_delayed_sltp)

    return d

# ── amend helpers ──────────────────────────────────────────────────────────
def modify_position_sltp(client, account_id, position_id, stop_loss=None, take_profit=None):
    req = ProtoOAAmendPositionSLTPReq(ctidTraderAccountId = account_id, positionId = position_id)
    if stop_loss   is not None: req.stopLoss   = _px(stop_loss)
    if take_profit is not None: req.takeProfit = _px(take_profit)
    return client.send(req)

def modify_pending_order_sltp(client, account_id, order_id, version, stop_loss=None, take_profit=None):
    req = ProtoOAAmendOrderReq(
        ctidTraderAccountId = account_id,
        orderId             = order_id,
        version             = version,
    )
    if stop_loss   is not None: req.stopLoss   = stop_loss
    if take_profit is not None: req.takeProfit = take_profit
    return client.send(req)

# ── blocking helper used by FastAPI layer ─────────────────────────────────
def wait_for_deferred(deferred, timeout=40):
    try:
        return deferred.result(timeout=timeout)
    except Exception as e:
        print(f"[FATAL] Deferred result timeout or failure: {e}")
        return {"status": "failed", "error": str(e)}
