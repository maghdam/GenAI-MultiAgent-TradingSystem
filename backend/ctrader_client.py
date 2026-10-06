# backend/ctrader_client.py
# ---------------------------------------------------------------------------

from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAApplicationAuthReq,
    ProtoOAAccountAuthReq,
    ProtoOAGetAccountListByAccessTokenReq,
    ProtoOATraderReq,
    ProtoOAAssetListReq,
    ProtoOAGetPositionUnrealizedPnLReq,
    ProtoOASymbolsListReq,
    ProtoOASymbolByIdReq,
    ProtoOAAssetClassListReq,
    ProtoOAReconcileReq,
    ProtoOAGetTrendbarsReq,
    ProtoOANewOrderReq,
    ProtoOAAmendOrderReq,
    ProtoOAAmendPositionSLTPReq,
    ProtoOAClosePositionReq,
    ProtoOADealListByPositionIdReq,
)
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (
    ProtoOAOrderType,
    ProtoOATradeSide,
    ProtoOATrendbarPeriod,
)
from google.protobuf.json_format import MessageToDict

from twisted.application.internet import ClientService
from twisted.internet import reactor
import asyncio
import os
import threading
import time
from collections import deque
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
if HOST_TYPE not in {"demo", "live"}:
    HOST_TYPE = "demo"


class _TradeAgentTcpProtocol(TcpProtocol):
    """Isolate OpenApiPy transport state per connection.

    Upstream TcpProtocol keeps its outbound queue/task/timestamp as class
    attributes. With simultaneous Live + Demo clients that makes different
    sockets share the same send queue, so a Demo probe request can be emitted
    by the Live protocol. Shadow those fields on each protocol instance.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._send_queue = deque([])
        self._send_task = None
        self._lastSendMessageTime = None


def _new_client(host_type: str):
    host = (
        EndPoints.PROTOBUF_LIVE_HOST
        if str(host_type).lower() == "live"
        else EndPoints.PROTOBUF_DEMO_HOST
    )
    return Client(host, EndPoints.PROTOBUF_PORT, _TradeAgentTcpProtocol)


client = _new_client(HOST_TYPE)
CLIENT_HOST_TYPE = HOST_TYPE

# ── symbol maps ────────────────────────────────────────────────────────────
symbol_map            : dict[int, str] = {}   # {id: name}
symbol_name_to_id     : dict[str, int] = {}   # {name.upper(): id}
symbol_digits_map     : dict[int, int] = {}   # {id: digits}
symbol_money_digits_map: dict[int, int] = {}  # {id: moneyDigits}
symbol_min_volume_map : dict[int, int] = {}   # {id: minimum volume in API units}
symbol_step_volume_map: dict[int, int] = {}   # {id: step volume in API units}
symbol_max_volume_map: dict[int, int] = {}   # {id: maximum volume in units}
symbol_lot_size_map: dict[int, float] = {}   # {id: one lot's underlying units}
symbol_min_verified: dict[int, bool] = {}     # {id: True if learned from broker error}
symbol_step_verified: dict[int, bool] = {}    # {id: True if learned from broker error}
SYMBOL_METADATA_READY: bool = False        # full contract metadata loaded for current session

# Optional fallback if cTrader rejects symbol list (e.g., invalid credentials or maintenance)
_fallback_symbols_cfg = os.getenv("CTRADER_FALLBACK_SYMBOLS", "XAUUSD,EURUSD,GBPUSD,US500").strip()
FALLBACK_SYMBOLS = [s.strip().upper() for s in _fallback_symbols_cfg.split(",") if s.strip()]

# connection state
CONNECTED = False
AUTHORIZED = False
AUTH_ERROR = None
LAST_AUTH_ATTEMPT_AT = None
ACCOUNT_IS_DEMO: bool | None = None
ACCOUNT_VERIFICATION_ERROR: str | None = None
AVAILABLE_ACCOUNTS: list[dict[str, object]] = []
_DEMO_DIRECTORY_PROBE_CLIENT = None
_demo_directory_probe_lock = threading.Lock()
ACTIVE_ACCOUNT_ID: int | None = None
ACTIVE_HOST_TYPE: str | None = None
ACCOUNT_SWITCH_IN_PROGRESS: bool = False
ACCOUNT_SWITCH_TARGET_ID: int | None = None
ACCOUNT_SWITCH_ERROR: str | None = None
_transport_lock = threading.Lock()

_asset_name_cache: dict[int, str] = {}
_asset_cache_account_id: int | None = None
_asset_cache_lock = threading.Lock()

_PRICE_FACTOR = 100_000
_PROTOCOL_VOLUME_SCALE = 100       # cTrader volume fields are cents of measurement units.
_AUTH_RESPONSE_TIMEOUT_SECONDS = 15   # Cross-host auth can exceed the library's 5s default.


def _request_client_msg_id(stage: str, account_id: int | None = None) -> str:
    """Create a correlation id that is safe to expose in diagnostics."""
    account_part = str(int(account_id)) if account_id is not None else "none"
    return f"tradeagent:{stage}:{account_part}:{time.monotonic_ns()}"


# Track the last order's symbol so we can reconcile broker-side volume
# requirements if an immediate TRADING_BAD_VOLUME error arrives.
_LAST_ORDER_CTX: dict[str, int] = {"symbol_id": -1}

def _normalize_host_type(value: str) -> str:
    host_type = str(value or "").strip().lower()
    if host_type not in {"demo", "live"}:
        raise ValueError("cTrader account type must be 'demo' or 'live'.")
    return host_type


def _account_id_int(value) -> int | None:
    try:
        account_id = int(value)
    except (TypeError, ValueError):
        return None
    return account_id if account_id > 0 else None


def _refresh_account_flags() -> None:
    desired = _account_id_int(ACCOUNT_ID)
    active = _account_id_int(ACTIVE_ACCOUNT_ID)
    for row in AVAILABLE_ACCOUNTS:
        row_id = _account_id_int(row.get("account_id"))
        row["selected"] = bool(desired is not None and row_id == desired)
        row["active"] = bool(active is not None and row_id == active)


def _is_current_session(source_client=None, expected_account_id: int | None = None) -> bool:
    if source_client is not None and source_client is not client:
        return False
    if expected_account_id is not None and _account_id_int(ACCOUNT_ID) != int(expected_account_id):
        return False
    return True


def _clear_symbol_metadata() -> None:
    global SYMBOL_METADATA_READY
    symbol_map.clear()
    symbol_name_to_id.clear()
    symbol_digits_map.clear()
    symbol_money_digits_map.clear()
    symbol_min_volume_map.clear()
    symbol_step_volume_map.clear()
    symbol_max_volume_map.clear()
    symbol_lot_size_map.clear()
    symbol_min_verified.clear()
    symbol_step_verified.clear()
    SYMBOL_METADATA_READY = False


def _px(x):
    """Pass-through float price (no integer scaling)."""
    try:
        return float(x) if x is not None else None
    except Exception:
        return None

def _px_sym(symbol_id: int | None, x):
    """Return an absolute trading price rounded to ProtoOASymbol.digits.

    ProtoOAPosition.moneyDigits applies only to monetary fields (swap,
    commission, margin, etc.) and must never be used as price precision.
    """
    if x is None:
        return None
    try:
        sid = int(symbol_id) if symbol_id is not None else None
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


def _protocol_volume_per_lot(symbol_id: int) -> float:
    """Return cTrader protocol-volume units for one lot of the symbol.

    Open API volume is expressed in cents of the symbol measurement unit.
    ProtoOASymbol.lotSize is also in cents, while symbol_lot_size_map stores
    the decoded measurement units per lot.
    """
    lot_size_units = symbol_lot_size_map.get(int(symbol_id))
    try:
        lot_size_units = float(lot_size_units)
    except (TypeError, ValueError):
        lot_size_units = 0.0
    if lot_size_units <= 0:
        raise ValueError(f"Broker lotSize metadata is unavailable for symbol_id={symbol_id}")
    return lot_size_units * _PROTOCOL_VOLUME_SCALE


def _lots_to_units(lots: float | int | None, symbol_id: int) -> int | None:
    """Convert lots to cTrader protocol volume (cents of measurement units)."""
    if lots is None:
        return None
    return int(round(float(lots) * _protocol_volume_per_lot(symbol_id)))


def protocol_volume_to_lots(symbol_id: int, volume: float | int | None) -> float | None:
    """Convert cTrader protocol volume back to lots using the symbol contract."""
    if volume is None:
        return None
    return float(volume) / _protocol_volume_per_lot(symbol_id)


def volume_lots_to_units(symbol_id: int, lots: float | int | None) -> int:
    """Convert a lot amount to cTrader protocol volume and validate broker limits."""
    try:
        lots_val = float(lots)
    except (TypeError, ValueError):
        raise ValueError("Lot size must be numeric") from None

    if lots_val <= 0:
        raise ValueError("Lot size must be greater than zero")

    api_volume = int(round(lots_val * _protocol_volume_per_lot(symbol_id)))

    min_api = symbol_min_volume_map.get(symbol_id)
    step_api = symbol_step_volume_map.get(symbol_id)
    max_api = symbol_max_volume_map.get(symbol_id)

    if min_api is not None and api_volume < min_api:
        minimum_lots = protocol_volume_to_lots(symbol_id, min_api)
        raise ValueError(f"Lot size too small; minimum is {minimum_lots:.4f} lots")

    if max_api is not None and api_volume > max_api:
        maximum_lots = protocol_volume_to_lots(symbol_id, max_api)
        raise ValueError(f"Lot size too large; maximum is {maximum_lots:.4f} lots")

    if step_api:
        base = int(min_api or 0)
        if (api_volume - base) % int(step_api):
            step_lots = protocol_volume_to_lots(symbol_id, step_api)
            raise ValueError(f"Lot size must align to step {step_lots:.4f} lots")

    print(
        f"[VOLUME] symbol_id={symbol_id} lots={lots_val} protocol_volume={api_volume} "
        f"min={min_api} step={step_api} max={max_api}"
    )
    return api_volume


def coerce_volume_lots_to_units(symbol_id: int, lots: float | int | None) -> tuple[int, float]:
    """Coerce lots to broker min/step/max and return (protocol_volume, lots)."""
    try:
        lots_val = float(lots)
    except (TypeError, ValueError):
        raise ValueError("Lot size must be numeric") from None

    if lots_val <= 0:
        raise ValueError("Lot size must be greater than zero")

    per_lot = _protocol_volume_per_lot(symbol_id)
    api_volume = int(round(lots_val * per_lot))
    min_api = symbol_min_volume_map.get(symbol_id)
    step_api = symbol_step_volume_map.get(symbol_id)
    max_api = symbol_max_volume_map.get(symbol_id)

    base_floor_api = max(1, int(round(0.01 * per_lot)))
    floor = int(min_api) if min_api is not None else base_floor_api
    api_volume = max(api_volume, floor, base_floor_api)

    if step_api and int(step_api) > 0:
        step = int(step_api)
        base = int(min_api or 0)
        rem = (api_volume - base) % step
        if rem:
            api_volume += step - rem

    if max_api is not None and api_volume > int(max_api):
        api_volume = int(max_api)
        if step_api and int(step_api) > 0:
            step = int(step_api)
            base = int(min_api or 0)
            rem = (api_volume - base) % step
            if rem:
                api_volume -= rem

    lots_final = api_volume / per_lot
    print(
        f"[VOLUME] COERCE symbol_id={symbol_id} requested_lots={lots_val} -> "
        f"lots={lots_final:.4f} protocol_volume={api_volume} min={min_api} step={step_api} max={max_api}"
    )
    return api_volume, lots_final

# ── helpers ────────────────────────────────────────────────────────────────
def pips_to_relative(pips: int, digits: int) -> int:
    """Convert pips → 1/100000 units (works for 2–5 digit symbols)."""
    return pips * 10 ** (6 - digits)

def on_error(failure, *, stage: str | None = None):
    global AUTH_ERROR, ACCOUNT_SWITCH_IN_PROGRESS, ACCOUNT_SWITCH_TARGET_ID, ACCOUNT_SWITCH_ERROR
    raw = str(failure)
    err_msg = f"{stage}: {raw}" if stage else raw
    AUTH_ERROR = err_msg
    if ACCOUNT_SWITCH_IN_PROGRESS:
        ACCOUNT_SWITCH_IN_PROGRESS = False
        ACCOUNT_SWITCH_TARGET_ID = None
        ACCOUNT_SWITCH_ERROR = err_msg
    print("[ERROR]", err_msg)


def _session_error(
    failure,
    source_client=None,
    expected_account_id: int | None = None,
    *,
    stage: str | None = None,
):
    """Ignore late failures emitted by a superseded client/account session."""
    if not _is_current_session(source_client, expected_account_id):
        return failure
    on_error(failure, stage=stage)
    return failure

# ── bootstrapping: symbols ─────────────────────────────────────────────────
def _install_fallback_symbols(reason: str | None = None):
    print(f"[WARN] Using fallback symbols ({reason or 'unknown error'})")
    _clear_symbol_metadata()

    for idx, name in enumerate(FALLBACK_SYMBOLS, start=1):
        symbol_map[idx] = name
        symbol_name_to_id[name] = idx
        symbol_digits_map[idx] = 5
        lot_size_units = 100_000.0 if name in ["EURUSD", "GBPUSD"] else 100.0
        symbol_lot_size_map[idx] = lot_size_units
        per_lot = lot_size_units * _PROTOCOL_VOLUME_SCALE
        symbol_min_volume_map[idx] = int(round(0.01 * per_lot))
        symbol_step_volume_map[idx] = int(round(0.01 * per_lot))
        symbol_max_volume_map[idx] = int(round(100.0 * per_lot))
        symbol_min_verified[idx] = False
        symbol_step_verified[idx] = False
    if symbol_map:
        print(f"[INFO] Loaded {len(symbol_map)} fallback symbols: {', '.join(symbol_map.values())}")

def symbols_response_cb(res, source_client=None, expected_account_id: int | None = None):
    if not _is_current_session(source_client, expected_account_id):
        return None
    active_client = source_client or client
    _clear_symbol_metadata()

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
        # Open API already reports min/step/max volume in cents of the
        # symbol measurement unit. Preserve those protocol values exactly.
        min_api = int(min_vol_raw) if min_vol_raw is not None else 1
        step_api = int(step_vol_raw) if step_vol_raw is not None else min_api
        max_api = int(max_vol_raw) if max_vol_raw is not None else max(min_api, step_api)
        
        symbol_min_volume_map[s.symbolId] = min_api
        symbol_step_volume_map[s.symbolId] = step_api
        symbol_max_volume_map[s.symbolId] = max_api
        symbol_min_verified[s.symbolId] = False
        symbol_step_verified[s.symbolId] = False
        loaded += 1

    if loaded == 0:
        _install_fallback_symbols("empty symbol list")
    else:
        print(f"[DEBUG] Deep Discovery Loaded {loaded} symbols.")
        # The list response contains light symbols. Request full contracts so
        # sizing/P&L can use lotSize rather than assuming every instrument is FX.
        req = ProtoOASymbolByIdReq(
            ctidTraderAccountId=ACCOUNT_ID,
            symbolId=list(symbol_map.keys()),
        )
        active_client.send(req).addCallbacks(
            lambda response: symbol_details_response_cb(
                response,
                active_client,
                expected_account_id,
            ),
            lambda failure: _session_error(
                failure,
                active_client,
                expected_account_id,
            ),
        )


def symbol_details_response_cb(res, source_client=None, expected_account_id: int | None = None):
    global SYMBOL_METADATA_READY
    if not _is_current_session(source_client, expected_account_id):
        return None
    payload = Protobuf.extract(res)
    if hasattr(payload, "errorCode") or payload.__class__.__name__ == "ProtoOAErrorRes":
        SYMBOL_METADATA_READY = False
        err = f"{getattr(payload, 'errorCode', 'ERR')} {getattr(payload, 'description', '').strip()}".strip()
        print(f"[WARN] Full symbol contract metadata unavailable: {err or 'ProtoOAErrorRes'}")
        return

    detailed = 0
    for item in getattr(payload, "symbol", []):
        sid = int(item.symbolId)
        digits = getattr(item, "digits", None)
        lot_size_raw = getattr(item, "lotSize", None)
        min_volume_raw = getattr(item, "minVolume", None)
        step_volume_raw = getattr(item, "stepVolume", None)
        max_volume_raw = getattr(item, "maxVolume", None)
        if digits is not None:
            symbol_digits_map[sid] = int(digits)
        # cTrader encodes lotSize in cents of an underlying unit.
        if lot_size_raw:
            symbol_lot_size_map[sid] = float(lot_size_raw) / 100.0
        # minVolume/stepVolume/maxVolume are already protocol volume (cents
        # of measurement units), so do not apply another scale factor.
        if min_volume_raw is not None:
            symbol_min_volume_map[sid] = int(min_volume_raw)
            symbol_min_verified[sid] = True
        if step_volume_raw is not None:
            symbol_step_volume_map[sid] = int(step_volume_raw)
            symbol_step_verified[sid] = True
        if max_volume_raw is not None:
            symbol_max_volume_map[sid] = int(max_volume_raw)
        if (
            lot_size_raw
            and min_volume_raw is not None
            and step_volume_raw is not None
            and max_volume_raw is not None
        ):
            detailed += 1
    expected = len(symbol_map)
    SYMBOL_METADATA_READY = expected > 0 and detailed == expected
    print(f"[DEBUG] Loaded complete contract metadata for {detailed}/{expected} symbols.")

def account_auth_cb(_, source_client=None, expected_account_id: int | None = None):
    global AUTHORIZED, AUTH_ERROR, ACTIVE_ACCOUNT_ID, ACTIVE_HOST_TYPE
    global ACCOUNT_SWITCH_IN_PROGRESS, ACCOUNT_SWITCH_TARGET_ID, ACCOUNT_SWITCH_ERROR
    global ACCOUNT_VERIFICATION_ERROR, LAST_AUTH_ATTEMPT_AT
    if not _is_current_session(source_client, expected_account_id):
        return None
    active_client = source_client or client
    active_account_id = _account_id_int(expected_account_id or ACCOUNT_ID)
    if (
        AUTHORIZED
        and active_account_id is not None
        and ACTIVE_ACCOUNT_ID == active_account_id
        and ACTIVE_HOST_TYPE == CLIENT_HOST_TYPE
    ):
        # Spotware's Python sample handles ProtoOAAccountAuthRes from the
        # message stream. The SDK may also resolve the request Deferred for
        # the same response, so keep this handler idempotent.
        return None
    AUTHORIZED = True
    AUTH_ERROR = None
    ACCOUNT_VERIFICATION_ERROR = None
    LAST_AUTH_ATTEMPT_AT = datetime.now(timezone.utc)
    ACTIVE_ACCOUNT_ID = active_account_id
    ACTIVE_HOST_TYPE = CLIENT_HOST_TYPE
    ACCOUNT_SWITCH_IN_PROGRESS = False
    ACCOUNT_SWITCH_TARGET_ID = None
    ACCOUNT_SWITCH_ERROR = None
    _refresh_account_flags()
    print(
        f"[CTRADER AUTH] account authorized id={active_account_id} "
        f"host={ACTIVE_HOST_TYPE or 'unknown'}."
    )
    if CLIENT_HOST_TYPE == "live":
        # Refresh the Demo-sourced token directory only after Live account auth.
        # Starting a second Demo connection during cross-host authentication can
        # overlap with teardown of the previous Demo transport and interfere with
        # the sensitive Live auth sequence.
        _start_demo_directory_probe()

    # Phase 1: Fetch asset classes
    req = ProtoOAAssetClassListReq(
        ctidTraderAccountId=active_account_id,
    )
    active_client.send(req).addCallbacks(
        lambda response: asset_class_response_cb(
            response,
            active_client,
            active_account_id,
        ),
        lambda failure: _session_error(
            failure,
            active_client,
            active_account_id,
        ),
    )


def asset_class_response_cb(res, source_client=None, expected_account_id: int | None = None):
    if not _is_current_session(source_client, expected_account_id):
        return None
    active_client = source_client or client
    account_id = _account_id_int(expected_account_id or ACCOUNT_ID)
    # We could log or filter here, but for now we just move to Phase 2: Symbols
    # Some brokers require specific symbols list requests, but global is usually fine.
    # By fetching assets first, we ensure the account session is synchronized.
    req = ProtoOASymbolsListReq(
        ctidTraderAccountId=account_id,
        includeArchivedSymbols=True, # Attempting maximum coverage
    )
    active_client.send(req).addCallbacks(
        lambda response: symbols_response_cb(
            response,
            active_client,
            account_id,
        ),
        lambda failure: _session_error(
            failure,
            active_client,
            account_id,
        ),
    )

def _apply_account_directory(accounts) -> None:
    AVAILABLE_ACCOUNTS.clear()
    for account in accounts:
        account_id = int(getattr(account, "ctidTraderAccountId", 0) or 0)
        if account_id <= 0:
            continue
        is_live = bool(getattr(account, "isLive", False))
        trader_login_raw = getattr(account, "traderLogin", None)
        trader_login = int(trader_login_raw) if trader_login_raw is not None else None
        broker_title = str(getattr(account, "brokerTitleShort", "") or "").strip() or None
        AVAILABLE_ACCOUNTS.append(
            {
                "account_id": account_id,
                "account_type": "live" if is_live else "demo",
                "is_live": is_live,
                "trader_login": trader_login,
                "broker_title": broker_title,
                "selected": False,
                "active": False,
            }
        )
    _refresh_account_flags()


def account_list_response_cb(res, source_client=None):
    global ACCOUNT_IS_DEMO, ACCOUNT_VERIFICATION_ERROR, AUTH_ERROR
    if source_client is not None and source_client is not client:
        return None
    active_client = source_client or client
    payload = Protobuf.extract(res)
    accounts = list(getattr(payload, "ctidTraderAccount", []) or [])
    _apply_account_directory(accounts)
    print(f"[CTRADER DIRECTORY] demo host returned {len(accounts)} token-granted account(s).")

    selected = next(
        (
            account
            for account in accounts
            if int(getattr(account, "ctidTraderAccountId", 0) or 0) == int(ACCOUNT_ID)
        ),
        None,
    )
    if selected is None:
        global ACCOUNT_SWITCH_IN_PROGRESS, ACCOUNT_SWITCH_TARGET_ID, ACCOUNT_SWITCH_ERROR
        ACCOUNT_IS_DEMO = None
        ACCOUNT_VERIFICATION_ERROR = "Configured cTrader account was not returned for the access token."
        AUTH_ERROR = ACCOUNT_VERIFICATION_ERROR
        if ACCOUNT_SWITCH_IN_PROGRESS:
            ACCOUNT_SWITCH_IN_PROGRESS = False
            ACCOUNT_SWITCH_TARGET_ID = None
            ACCOUNT_SWITCH_ERROR = ACCOUNT_VERIFICATION_ERROR
        print(f"[SAFETY] {ACCOUNT_VERIFICATION_ERROR}")
        return None

    is_live = bool(getattr(selected, "isLive", True))
    ACCOUNT_IS_DEMO = not is_live
    expected_host = "live" if is_live else "demo"
    if CLIENT_HOST_TYPE != expected_host:
        ACCOUNT_VERIFICATION_ERROR = (
            f"Selected cTrader account requires the {expected_host} host; "
            f"current transport host is {CLIENT_HOST_TYPE}."
        )
    else:
        ACCOUNT_VERIFICATION_ERROR = None

    _refresh_account_flags()
    if ACCOUNT_VERIFICATION_ERROR:
        AUTH_ERROR = ACCOUNT_VERIFICATION_ERROR
        if ACCOUNT_SWITCH_IN_PROGRESS:
            ACCOUNT_SWITCH_IN_PROGRESS = False
            ACCOUNT_SWITCH_TARGET_ID = None
            ACCOUNT_SWITCH_ERROR = ACCOUNT_VERIFICATION_ERROR
        print(f"[SAFETY] {ACCOUNT_VERIFICATION_ERROR}")
        return None

    account_id = int(ACCOUNT_ID)
    req = ProtoOAAccountAuthReq(
        ctidTraderAccountId=account_id,
        accessToken=ACCESS_TOKEN,
    )
    return active_client.send(
        req,
        clientMsgId=_request_client_msg_id(expected_host + "-account-auth", account_id),
        responseTimeoutInSeconds=_AUTH_RESPONSE_TIMEOUT_SECONDS,
    ).addCallbacks(
        lambda response: account_auth_cb(response, active_client, account_id),
        lambda failure: _session_error(
            failure,
            active_client,
            account_id,
            stage=f"cTrader {expected_host} account auth {account_id}",
        ),
    )


def account_list_error_cb(failure, source_client=None):
    global ACCOUNT_IS_DEMO, ACCOUNT_VERIFICATION_ERROR, AUTH_ERROR
    global ACCOUNT_SWITCH_IN_PROGRESS, ACCOUNT_SWITCH_TARGET_ID, ACCOUNT_SWITCH_ERROR
    if source_client is not None and source_client is not client:
        return failure
    AVAILABLE_ACCOUNTS.clear()
    ACCOUNT_IS_DEMO = None
    ACCOUNT_VERIFICATION_ERROR = f"Unable to verify cTrader account type: {failure}"
    AUTH_ERROR = ACCOUNT_VERIFICATION_ERROR
    if ACCOUNT_SWITCH_IN_PROGRESS:
        ACCOUNT_SWITCH_IN_PROGRESS = False
        ACCOUNT_SWITCH_TARGET_ID = None
        ACCOUNT_SWITCH_ERROR = ACCOUNT_VERIFICATION_ERROR
    print(f"[SAFETY] {ACCOUNT_VERIFICATION_ERROR}")
    return failure



def _demo_directory_probe_is_current(probe_client) -> bool:
    with _demo_directory_probe_lock:
        return _DEMO_DIRECTORY_PROBE_CLIENT is probe_client


def _finish_demo_directory_probe(probe_client, *, error: str | None = None) -> None:
    global _DEMO_DIRECTORY_PROBE_CLIENT
    with _demo_directory_probe_lock:
        if _DEMO_DIRECTORY_PROBE_CLIENT is not probe_client:
            return
        _DEMO_DIRECTORY_PROBE_CLIENT = None
    _stop_client_service(probe_client)
    if error:
        print(f"[WARN] cTrader demo account-directory probe failed: {error}")


def _cancel_demo_directory_probe() -> bool:
    global _DEMO_DIRECTORY_PROBE_CLIENT
    with _demo_directory_probe_lock:
        probe_client = _DEMO_DIRECTORY_PROBE_CLIENT
        _DEMO_DIRECTORY_PROBE_CLIENT = None
    if probe_client is None:
        return False
    _stop_client_service(probe_client)
    return True


def _demo_directory_probe_error(failure, probe_client, *, stage: str):
    if not _demo_directory_probe_is_current(probe_client):
        return None
    current_stage = str(getattr(probe_client, "_tradeagent_directory_stage", ""))
    if current_stage != stage:
        return None
    _finish_demo_directory_probe(probe_client, error=f"{stage}: {failure}")
    return None


def _demo_directory_probe_send_account_list(probe_client):
    if not _demo_directory_probe_is_current(probe_client):
        return None
    if getattr(probe_client, "_tradeagent_directory_stage", None) == "account_list":
        return None
    probe_client._tradeagent_directory_stage = "account_list"
    print("[CTRADER DIRECTORY] demo directory probe authorized; requesting token accounts.")
    req = ProtoOAGetAccountListByAccessTokenReq(accessToken=ACCESS_TOKEN)
    request_id = _request_client_msg_id("demo-directory-account-list")
    probe_client._tradeagent_directory_client_msg_id = request_id
    return probe_client.send(
        req,
        clientMsgId=request_id,
        responseTimeoutInSeconds=15,
    ).addCallbacks(
        lambda _: None,
        lambda failure: _demo_directory_probe_error(
            failure,
            probe_client,
            stage="account_list",
        ),
    )


def _demo_directory_probe_message_received(source_client, message, probe_client) -> None:
    if source_client is not probe_client or not _demo_directory_probe_is_current(probe_client):
        return
    try:
        payload = Protobuf.extract(message)
    except Exception:
        return

    message_name = payload.__class__.__name__
    stage = str(getattr(probe_client, "_tradeagent_directory_stage", ""))

    if message_name == "ProtoOAApplicationAuthRes":
        if stage == "app_auth":
            _demo_directory_probe_send_account_list(probe_client)
        return

    if message_name == "ProtoOAGetAccountListByAccessTokenRes":
        if stage == "account_list":
            accounts = list(getattr(payload, "ctidTraderAccount", []) or [])
            _apply_account_directory(accounts)
            print(
                f"[CTRADER DIRECTORY] demo directory probe returned {len(accounts)} "
                "token-granted account(s)."
            )
            _finish_demo_directory_probe(probe_client)
        return

    if message_name != "ProtoOAErrorRes":
        return

    client_msg_id = str(getattr(message, "clientMsgId", None) or "")
    expected_msg_id = str(
        getattr(probe_client, "_tradeagent_directory_client_msg_id", "") or ""
    )
    matches_request = bool(
        client_msg_id
        and expected_msg_id
        and client_msg_id == expected_msg_id
    )
    print(
        f"[CTRADER DIRECTORY ERROR CONTEXT] "
        f"client_msg_id={client_msg_id or '<none>'} "
        f"expected_client_msg_id={expected_msg_id or '<none>'} "
        f"stage={stage or 'unknown'} matched={matches_request}"
    )
    if not matches_request:
        # cTrader can emit connection-level errors that are not responses to
        # the probe's current request. Do not tear down account discovery
        # unless the error is correlated to the outstanding probe request.
        return
    error_code = str(getattr(payload, "errorCode", "") or "").strip().upper()
    description = str(getattr(payload, "description", "") or "").strip()
    if stage == "app_auth" and error_code in {
        "ALREADY_LOGGED_IN",
        "CH_CLIENT_ALREADY_AUTHENTICATED",
    }:
        _demo_directory_probe_send_account_list(probe_client)
        return
    _finish_demo_directory_probe(
        probe_client,
        error=f"{stage or 'unknown'}: cTrader error {error_code} {description}".strip(),
    )


def _demo_directory_probe_connected(connected_client, probe_client):
    if connected_client is not probe_client or not _demo_directory_probe_is_current(probe_client):
        return None
    probe_client._tradeagent_directory_stage = "app_auth"
    print("[CTRADER DIRECTORY] demo directory probe connected; authorizing application.")
    req = ProtoOAApplicationAuthReq(clientId=CLIENT_ID, clientSecret=CLIENT_SECRET)
    request_id = _request_client_msg_id("demo-directory-app-auth")
    probe_client._tradeagent_directory_client_msg_id = request_id
    return probe_client.send(
        req,
        clientMsgId=request_id,
        responseTimeoutInSeconds=15,
    ).addCallbacks(
        lambda _: None,
        lambda failure: _demo_directory_probe_error(
            failure,
            probe_client,
            stage="app_auth",
        ),
    )


def _demo_directory_probe_disconnected(disconnected_client, reason, probe_client) -> None:
    if disconnected_client is probe_client and _demo_directory_probe_is_current(probe_client):
        _finish_demo_directory_probe(
            probe_client,
            error=f"disconnected before account discovery completed: {reason}",
        )


def _start_demo_directory_probe() -> bool:
    """Refresh the token-granted account directory through cTrader's Demo endpoint."""
    global _DEMO_DIRECTORY_PROBE_CLIENT
    if CLIENT_HOST_TYPE != "live" or not getattr(reactor, "running", False):
        return False
    with _demo_directory_probe_lock:
        if _DEMO_DIRECTORY_PROBE_CLIENT is not None:
            return False
        probe_client = _new_client("demo")
        _DEMO_DIRECTORY_PROBE_CLIENT = probe_client

    probe_client.setConnectedCallback(
        lambda connected_client: _demo_directory_probe_connected(
            connected_client,
            probe_client,
        )
    )
    probe_client.setDisconnectedCallback(
        lambda disconnected_client, reason: _demo_directory_probe_disconnected(
            disconnected_client,
            reason,
            probe_client,
        )
    )
    probe_client.setMessageReceivedCallback(
        lambda source_client, message: _demo_directory_probe_message_received(
            source_client,
            message,
            probe_client,
        )
    )
    try:
        probe_client.startService()
    except Exception as exc:
        _finish_demo_directory_probe(probe_client, error=str(exc))
        return False
    return True


def app_auth_cb(_, source_client=None):
    global LAST_AUTH_ATTEMPT_AT, ACCOUNT_IS_DEMO, ACCOUNT_VERIFICATION_ERROR
    if source_client is not None and source_client is not client:
        return None
    active_client = source_client or client
    LAST_AUTH_ATTEMPT_AT = datetime.now(timezone.utc)
    ACCOUNT_VERIFICATION_ERROR = None

    if CLIENT_HOST_TYPE == "live":
        # Spotware's own multi-environment sample obtains the token account
        # directory through the Demo client. Authenticate the selected Live
        # account first; the Demo-only directory refresh starts after that
        # succeeds so cross-host authentication is serialized.
        ACCOUNT_IS_DEMO = False
        account_id = int(ACCOUNT_ID)
        req = ProtoOAAccountAuthReq(
            ctidTraderAccountId=account_id,
            accessToken=ACCESS_TOKEN,
        )
        print(
            f"[CTRADER AUTH] application authorized host=live; "
            f"sending account auth id={account_id}."
        )
        return active_client.send(
            req,
            clientMsgId=_request_client_msg_id("live-account-auth", account_id),
            responseTimeoutInSeconds=_AUTH_RESPONSE_TIMEOUT_SECONDS,
        ).addCallbacks(
            lambda response: account_auth_cb(response, active_client, account_id),
            lambda failure: _session_error(
                failure,
                active_client,
                account_id,
                stage=f"cTrader live account auth {account_id}",
            ),
        )

    AVAILABLE_ACCOUNTS.clear()
    ACCOUNT_IS_DEMO = None
    req = ProtoOAGetAccountListByAccessTokenReq(
        accessToken=ACCESS_TOKEN,
    )
    return active_client.send(req).addCallbacks(
        lambda response: account_list_response_cb(response, active_client),
        lambda failure: account_list_error_cb(failure, active_client),
    )


def _on_connected(connected_client):
    global CONNECTED, AUTHORIZED, AUTH_ERROR, ACCOUNT_IS_DEMO, ACCOUNT_VERIFICATION_ERROR
    if connected_client is not client:
        return
    _clear_asset_cache()
    _clear_symbol_metadata()
    CONNECTED = True
    AUTHORIZED = False
    AUTH_ERROR = None
    ACCOUNT_IS_DEMO = None
    ACCOUNT_VERIFICATION_ERROR = None
    host_type = CLIENT_HOST_TYPE
    print(f"[CTRADER AUTH] connected host={host_type}; sending application auth.")
    req = ProtoOAApplicationAuthReq(clientId=CLIENT_ID, clientSecret=CLIENT_SECRET)
    connected_client.send(
        req,
        clientMsgId=_request_client_msg_id(f"{host_type}-application-auth"),
        responseTimeoutInSeconds=_AUTH_RESPONSE_TIMEOUT_SECONDS,
    ).addCallbacks(
        lambda response: app_auth_cb(response, connected_client),
        lambda failure: _session_error(
            failure,
            connected_client,
            stage=f"cTrader {host_type} application auth",
        ),
    )


def _on_disconnected(disconnected_client, reason):
    global CONNECTED, AUTHORIZED, ACCOUNT_IS_DEMO, ACTIVE_ACCOUNT_ID, ACTIVE_HOST_TYPE
    if disconnected_client is not client:
        return
    _clear_asset_cache()
    _clear_symbol_metadata()
    AVAILABLE_ACCOUNTS.clear()
    CONNECTED = False
    AUTHORIZED = False
    ACCOUNT_IS_DEMO = None
    ACTIVE_ACCOUNT_ID = None
    ACTIVE_HOST_TYPE = None
    print("[INFO] Disconnected:", reason)


def _log_event(event) -> None:
    name = getattr(event, "__class__", type("x", (), {})).__name__

    large_collection_attrs = {
        "ProtoOASymbolsListRes": "symbol",
        "ProtoOASymbolByIdRes": "symbol",
        "ProtoOAAssetClassListRes": "assetClass",
        "ProtoOAAssetListRes": "asset",
    }
    if name in large_collection_attrs:
        attr_name = large_collection_attrs[name]
        items = getattr(event, attr_name, []) or []
        print(f"[CTRADER EVENT] {name}: count={len(items)}")
        return

    if name == "ProtoOAGetTrendbarsRes":
        bars = getattr(event, "trendbar", []) or []
        first = bars[0] if bars else None
        last = bars[-1] if bars else None
        first_ts = (
            getattr(first, "utcTimestampInMinutes", None)
            if first is not None
            else None
        )
        last_ts = (
            getattr(last, "utcTimestampInMinutes", None)
            if last is not None
            else None
        )
        print(f"[CTRADER TREND] bars={len(bars)} ts_range={first_ts}->{last_ts}")
        return

    try:
        payload = MessageToDict(event, preserving_proto_field_name=True)
    except Exception as e:
        payload = {"decode_error": str(e)}
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
                    # Broker volume hints use the same protocol volume unit:
                    # cents of the symbol measurement unit.
                    min_api = int(round(float(m.group(1))))
                    sid = int(_LAST_ORDER_CTX.get("symbol_id", -1))
                    if sid in symbol_map:
                        prev = symbol_min_volume_map.get(sid)
                        symbol_min_volume_map[sid] = max(min_api, prev or 0)
                        symbol_min_verified[sid] = True
                        print(f"[VOLUME] Updated min for {symbol_map[sid]} to {min_api} (API units)")
                m2 = re.search(r"step\s*=?\s*([0-9]+(?:\.[0-9]+)?)", str(desc), re.IGNORECASE)
                if m2:
                    step_api = int(round(float(m2.group(1))))
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


def _redact_sensitive_payload(value):
    """Remove credential material before broker messages are written to logs."""
    sensitive_keys = {
        "accesstoken",
        "refreshtoken",
        "clientsecret",
        "authorizationcode",
    }
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            normalized = str(key).replace("_", "").lower()
            redacted[key] = "<redacted>" if normalized in sensitive_keys else _redact_sensitive_payload(item)
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_sensitive_payload(item) for item in value)
    return value


def _format_payload(payload) -> str:
    payload = _redact_sensitive_payload(payload)
    try:
        txt = json.dumps(payload, ensure_ascii=False)
    except Exception:
        txt = str(payload)
    if len(txt) > 600:
        return f"{txt[:600]}... (truncated, {len(txt)} chars)"
    return txt


def _configure_client_callbacks(target_client) -> None:
    target_client.setConnectedCallback(_on_connected)
    target_client.setDisconnectedCallback(_on_disconnected)

    def _on_message(source_client, message):
        if source_client is not client:
            return
        try:
            event = Protobuf.extract(message)
        except Exception as e:
            print(f"[CTRADER EVENT] decode_error: {e}")
            return
        _log_event(event)
        if event.__class__.__name__ == "ProtoOAErrorRes":
            client_msg_id = getattr(message, "clientMsgId", None) or "<none>"
            print(f"[CTRADER ERROR CONTEXT] client_msg_id={client_msg_id}")

        # Spotware's official OpenApiPy samples treat successful account auth
        # as a broker event. Do not require the SDK request Deferred to resolve:
        # some Live sessions deliver ProtoOAAccountAuthRes without satisfying
        # that correlation path.
        if event.__class__.__name__ == "ProtoOAAccountAuthRes":
            event_account_id = _account_id_int(
                getattr(event, "ctidTraderAccountId", None)
            )
            if event_account_id == _account_id_int(ACCOUNT_ID):
                account_auth_cb(
                    event,
                    source_client,
                    event_account_id,
                )

    target_client.setMessageReceivedCallback(_on_message)


def _stop_client_service(target_client) -> None:
    try:
        if getattr(target_client, "running", False):
            # ctrader-open-api's Client.stopService() skips disconnected
            # services. Call the Twisted base implementation so a stale
            # reconnect loop cannot survive a Demo/Live host switch.
            ClientService.stopService(target_client)
    except Exception as exc:
        print(f"[WARN] Unable to stop previous cTrader client service: {exc}")


def configure_target_account(account_id: int, account_type: str) -> None:
    """Configure the desired account before the transport thread starts."""
    global ACCOUNT_ID, HOST_TYPE, client, CLIENT_HOST_TYPE
    global CONNECTED, AUTHORIZED, ACCOUNT_IS_DEMO
    global ACTIVE_ACCOUNT_ID, ACTIVE_HOST_TYPE
    global ACCOUNT_SWITCH_IN_PROGRESS, ACCOUNT_SWITCH_TARGET_ID, ACCOUNT_SWITCH_ERROR
    target_id = _account_id_int(account_id)
    if target_id is None:
        raise ValueError("cTrader account ID must be a positive integer.")
    target_host = _normalize_host_type(account_type)

    with _transport_lock:
        if getattr(client, "running", False):
            raise RuntimeError("cTrader transport is already running.")
        ACCOUNT_ID = target_id
        HOST_TYPE = target_host
        CONNECTED = False
        AUTHORIZED = False
        ACCOUNT_IS_DEMO = None
        ACTIVE_ACCOUNT_ID = None
        ACTIVE_HOST_TYPE = None
        ACCOUNT_SWITCH_IN_PROGRESS = False
        ACCOUNT_SWITCH_TARGET_ID = None
        ACCOUNT_SWITCH_ERROR = None
        AVAILABLE_ACCOUNTS.clear()
        _clear_asset_cache()
        _clear_symbol_metadata()
        if CLIENT_HOST_TYPE != target_host:
            client = _new_client(target_host)
            CLIENT_HOST_TYPE = target_host


def _switch_account_on_reactor(account_id: int, account_type: str) -> None:
    global client, CLIENT_HOST_TYPE, ACCOUNT_IS_DEMO
    target_host = _normalize_host_type(account_type)
    target_id = int(account_id)
    current_client = client

    if CLIENT_HOST_TYPE == target_host:
        if bool(getattr(current_client, "isConnected", False)) or CONNECTED:
            ACCOUNT_IS_DEMO = target_host == "demo"
            req = ProtoOAAccountAuthReq(
                ctidTraderAccountId=target_id,
                accessToken=ACCESS_TOKEN,
            )
            current_client.send(
                req,
                clientMsgId=_request_client_msg_id(
                    f"{target_host}-account-auth",
                    target_id,
                ),
                responseTimeoutInSeconds=_AUTH_RESPONSE_TIMEOUT_SECONDS,
            ).addCallbacks(
                lambda response: account_auth_cb(
                    response,
                    current_client,
                    target_id,
                ),
                lambda failure: _session_error(
                    failure,
                    current_client,
                    target_id,
                    stage=f"cTrader {target_host} account auth {target_id}",
                ),
            )
        elif not getattr(current_client, "running", False):
            _configure_client_callbacks(current_client)
            current_client.startService()
        # A running but temporarily disconnected ClientService will reconnect
        # itself; _on_connected() will authenticate the desired account.
        return

    if target_host == "demo":
        _cancel_demo_directory_probe()
    next_client = _new_client(target_host)
    _configure_client_callbacks(next_client)
    client = next_client
    CLIENT_HOST_TYPE = target_host
    _stop_client_service(current_client)
    next_client.startService()


def switch_account(account_id: int, account_type: str) -> dict[str, object]:
    """Begin a fail-closed account switch without restarting Twisted's reactor."""
    global ACCOUNT_ID, HOST_TYPE, AUTHORIZED, AUTH_ERROR, ACCOUNT_IS_DEMO
    global ACTIVE_ACCOUNT_ID, ACTIVE_HOST_TYPE
    global ACCOUNT_SWITCH_IN_PROGRESS, ACCOUNT_SWITCH_TARGET_ID, ACCOUNT_SWITCH_ERROR
    global ACCOUNT_VERIFICATION_ERROR, CONNECTED

    target_id = _account_id_int(account_id)
    if target_id is None:
        raise ValueError("cTrader account ID must be a positive integer.")
    target_host = _normalize_host_type(account_type)

    with _transport_lock:
        if ACCOUNT_SWITCH_IN_PROGRESS:
            if ACCOUNT_SWITCH_TARGET_ID == target_id:
                return {
                    "switch_started": False,
                    "already_switching": True,
                    "target_account_id": target_id,
                    "target_host_type": target_host,
                }
            raise RuntimeError("Another cTrader account switch is already in progress.")

        if (
            AUTHORIZED
            and ACTIVE_ACCOUNT_ID == target_id
            and ACTIVE_HOST_TYPE == target_host
        ):
            ACCOUNT_ID = target_id
            HOST_TYPE = target_host
            _refresh_account_flags()
            return {
                "switch_started": False,
                "already_active": True,
                "target_account_id": target_id,
                "target_host_type": target_host,
            }

        cross_host = CLIENT_HOST_TYPE != target_host
        ACCOUNT_ID = target_id
        HOST_TYPE = target_host
        AUTHORIZED = False
        AUTH_ERROR = "cTrader account switch in progress."
        ACCOUNT_IS_DEMO = None
        ACCOUNT_VERIFICATION_ERROR = "cTrader account switch in progress."
        ACTIVE_ACCOUNT_ID = None
        ACTIVE_HOST_TYPE = None
        ACCOUNT_SWITCH_IN_PROGRESS = True
        ACCOUNT_SWITCH_TARGET_ID = target_id
        ACCOUNT_SWITCH_ERROR = None
        if cross_host:
            CONNECTED = False
        _clear_asset_cache()
        _clear_symbol_metadata()
        _refresh_account_flags()

        if getattr(reactor, "running", False):
            reactor.callFromThread(
                _switch_account_on_reactor,
                target_id,
                target_host,
            )
        else:
            _switch_account_on_reactor(target_id, target_host)

    return {
        "switch_started": True,
        "already_active": False,
        "cross_host": cross_host,
        "target_account_id": target_id,
        "target_host_type": target_host,
    }


def init_client():
    current_client = client
    _configure_client_callbacks(current_client)
    current_client.startService()
    if not getattr(reactor, "running", False):
        reactor.run(installSignalHandlers=False)


def is_connected() -> bool:
    return CONNECTED

def is_authorized() -> bool:
    return AUTHORIZED

def is_symbol_metadata_ready() -> bool:
    return bool(CONNECTED and AUTHORIZED and SYMBOL_METADATA_READY)


def get_auth_error() -> str | None:
    return AUTH_ERROR

def get_last_auth_attempt() -> datetime | None:
    return LAST_AUTH_ATTEMPT_AT


def is_account_confirmed() -> bool:
    """Return whether the configured cTrader account is the authenticated active session."""
    if ACTIVE_HOST_TYPE not in {"demo", "live"} or ACCOUNT_IS_DEMO is None:
        return False
    expected_demo = ACTIVE_HOST_TYPE == "demo"
    return bool(
        CONNECTED
        and AUTHORIZED
        and ACTIVE_ACCOUNT_ID == _account_id_int(ACCOUNT_ID)
        and ACCOUNT_IS_DEMO is expected_demo
    )


def is_demo_account_confirmed() -> bool:
    """Backward-compatible demo-specific view of the generic account confirmation."""
    return bool(is_account_confirmed() and ACTIVE_HOST_TYPE == "demo")


def get_account_verification_error() -> str | None:
    return ACCOUNT_VERIFICATION_ERROR


def get_active_account_id() -> int | None:
    return _account_id_int(ACTIVE_ACCOUNT_ID)


def get_active_host_type() -> str:
    return ACTIVE_HOST_TYPE if ACTIVE_HOST_TYPE in {"demo", "live"} else "unknown"


def is_account_switch_in_progress() -> bool:
    return bool(ACCOUNT_SWITCH_IN_PROGRESS)


def get_account_switch_target_id() -> int | None:
    return _account_id_int(ACCOUNT_SWITCH_TARGET_ID)


def get_account_switch_error() -> str | None:
    return ACCOUNT_SWITCH_ERROR


def get_available_accounts() -> list[dict[str, object]]:
    """Return broker-reported accounts authorized by the current access token."""
    return [dict(account) for account in AVAILABLE_ACCOUNTS]


def _trendbar_lookback_minutes(tf: str, n: int | None) -> int:
    tf_min_map = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
    minutes_needed = (n or 1000) * tf_min_map.get(tf.upper(), 5)
    # A short request made on a weekend/holiday otherwise returns no bars and
    # forces a second 12-second request. One week covers the latest session.
    return max(int(minutes_needed * 1.2), 7 * 24 * 60)



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
    buffer_minutes = _trendbar_lookback_minutes(tf, n)
    from_time = now - timedelta(minutes=buffer_minutes)

    req = ProtoOAGetTrendbarsReq(
        symbolId=sid,
        ctidTraderAccountId=ACCOUNT_ID,
        period=getattr(ProtoOATrendbarPeriod, tf),
        fromTimestamp=int(calendar.timegm(from_time.utctimetuple())) * 1000,
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

    print(f"[DEBUG] OHLC Fetch: {symbol} {tf} | window={buffer_minutes}m")
    d = client.send(req, responseTimeoutInSeconds=10)
    d.addCallbacks(_ok, _err)

    if not ev.wait(12):
        err_txt = "trendbars timeout (no event set after 12s)"
        print(f"[WARN] OHLC timeout for {symbol} {tf}")

    # --- FALLBACK ---
    if not out and not err_txt:
        print(f"[DEBUG] OHLC tight window empty for {symbol} {tf}. Trying 60-day fallback...")
        ev = threading.Event()
        from_time_fb = now - timedelta(days=60)
        req_fb = ProtoOAGetTrendbarsReq(
            symbolId=sid,
            ctidTraderAccountId=ACCOUNT_ID,
            period=getattr(ProtoOATrendbarPeriod, tf),
            fromTimestamp=int(calendar.timegm(from_time_fb.utctimetuple())) * 1000,
            toTimestamp=int(calendar.timegm(now.utctimetuple())) * 1000,
        )
        d_fb = client.send(req_fb, responseTimeoutInSeconds=10)
        d_fb.addCallbacks(_ok, _err)
        if not ev.wait(12):
            print(f"[WARN] OHLC fallback timeout for {symbol} {tf}")

    if not out:
        msg = f"trendbars error for {symbol} {tf}: {err_txt or 'empty response after fallback'}"
        print(f"[ERROR] {msg}")
        raise RuntimeError(msg)

    print(f"[DEBUG] OHLC Success for {symbol} {tf}: returned {len(out)} bars")
    return out[-n:] if n else out


# --- reconcile snapshot helpers ---
def _parse_reconcile_positions(obj):
    rows = []
    for p in getattr(obj, "position", []):
        td = p.tradeData
        sid = getattr(td, "symbolId", None)
        md = getattr(p, "moneyDigits", None)
        try:
            if md is not None and sid is not None:
                symbol_money_digits_map[int(sid)] = int(md)
        except Exception:
            pass

        rows.append(dict(
            symbol_name=symbol_map.get(sid, str(sid)),
            symbol_id=sid,
            digits=symbol_digits_map.get(sid),
            money_digits=symbol_money_digits_map.get(int(sid)) if sid is not None else None,
            position_id=p.positionId,
            direction="buy" if td.tradeSide == ProtoOATradeSide.BUY else "sell",
            entry_price=_decode_px(sid, getattr(p, "price", 0)),
            volume_lots=protocol_volume_to_lots(int(sid), td.volume) if sid is not None else None,
            stop_loss=_decode_px(sid, getattr(p, "stopLoss", None)) if getattr(p, "stopLoss", None) not in (None, 0) else None,
            take_profit=_decode_px(sid, getattr(p, "takeProfit", None)) if getattr(p, "takeProfit", None) not in (None, 0) else None,
        ))
    return rows


def get_reconcile_snapshot():
    ev = threading.Event()
    positions = []
    orders = []
    err = None

    def _ok(res):
        nonlocal err
        obj = Protobuf.extract(res)
        if getattr(obj, "__class__", type("x", (object,), {})).__name__ == "ProtoOAErrorRes" or hasattr(obj, "errorCode"):
            err = f"{getattr(obj, 'errorCode', 'ERR')} {getattr(obj, 'description', '')}".strip()
            ev.set()
            return
        positions.extend(_parse_reconcile_positions(obj))
        orders.extend(getattr(obj, "order", []))
        ev.set()

    def _err(f):
        nonlocal err
        err = str(f)
        ev.set()

    d = client.send(ProtoOAReconcileReq(ctidTraderAccountId=ACCOUNT_ID), responseTimeoutInSeconds=10)
    d.addCallbacks(_ok, _err)
    if not ev.wait(10):
        err = "reconcile timeout"
    return {"positions": positions, "orders": orders, "error": err}


def get_open_positions():
    return get_reconcile_snapshot()["positions"]


def get_pending_orders():
    return get_reconcile_snapshot()["orders"]


def _extract_response_or_raise(raw, label: str):
    if isinstance(raw, dict) and raw.get("status") == "failed":
        raise RuntimeError(f"{label} failed: {raw.get('error') or 'unknown error'}")
    event = Protobuf.extract(raw)
    if getattr(event, "__class__", type("x", (object,), {})).__name__ == "ProtoOAErrorRes" or hasattr(event, "errorCode"):
        code = getattr(event, "errorCode", "ERR")
        description = getattr(event, "description", "")
        raise RuntimeError(f"{label} failed: {code} {description}".strip())
    return event


def _clear_asset_cache() -> None:
    global _asset_cache_account_id
    with _asset_cache_lock:
        _asset_name_cache.clear()
        _asset_cache_account_id = None


def _get_deposit_currency(deposit_asset_id: int) -> str:
    """Resolve the account deposit currency with a session-scoped asset cache."""
    global _asset_cache_account_id

    account_id = int(ACCOUNT_ID)
    asset_id = int(deposit_asset_id)
    with _asset_cache_lock:
        if _asset_cache_account_id == account_id:
            cached = _asset_name_cache.get(asset_id)
            if cached:
                return cached
        else:
            _asset_name_cache.clear()
            _asset_cache_account_id = None

        assets_raw = wait_for_deferred(
            client.send(
                ProtoOAAssetListReq(ctidTraderAccountId=account_id),
                responseTimeoutInSeconds=10,
            ),
            timeout=12,
        )
        assets_event = _extract_response_or_raise(assets_raw, "cTrader asset list")
        names: dict[int, str] = {}
        for asset in list(getattr(assets_event, "asset", []) or []):
            try:
                current_id = int(getattr(asset, "assetId", 0) or 0)
            except (TypeError, ValueError):
                continue
            current_name = str(getattr(asset, "name", "") or "").strip().upper()
            if current_id > 0 and current_name:
                names[current_id] = current_name

        _asset_name_cache.clear()
        _asset_name_cache.update(names)
        _asset_cache_account_id = account_id
        currency = _asset_name_cache.get(asset_id)
        if not currency:
            raise RuntimeError(
                f"cTrader deposit asset {asset_id} was not returned by the asset list."
            )
        return currency


def _decode_money(raw, money_digits: int) -> float:
    return float(raw or 0) / float(10 ** max(int(money_digits or 0), 0))


def get_account_snapshot() -> dict[str, object]:
    """Return authoritative monetary state for the authenticated active cTrader account."""
    if not is_account_confirmed():
        reason = (
            get_account_verification_error()
            or "The selected cTrader account is not the authenticated active account."
        )
        raise RuntimeError(f"cTrader account snapshot blocked: {reason}")

    trader_raw = wait_for_deferred(
        client.send(
            ProtoOATraderReq(ctidTraderAccountId=ACCOUNT_ID),
            responseTimeoutInSeconds=10,
        ),
        timeout=12,
    )
    trader_event = _extract_response_or_raise(trader_raw, "cTrader trader snapshot")
    trader = getattr(trader_event, "trader", None)
    if trader is None:
        raise RuntimeError("cTrader trader snapshot did not include trader account data.")

    money_digits = int(getattr(trader, "moneyDigits", 0) or 0)
    balance = _decode_money(getattr(trader, "balance", 0), money_digits)
    deposit_asset_id = int(getattr(trader, "depositAssetId", 0) or 0)
    if deposit_asset_id <= 0:
        raise RuntimeError("cTrader trader snapshot did not include a valid deposit asset id.")

    currency = _get_deposit_currency(deposit_asset_id)

    pnl_raw = wait_for_deferred(
        client.send(
            ProtoOAGetPositionUnrealizedPnLReq(ctidTraderAccountId=ACCOUNT_ID),
            responseTimeoutInSeconds=10,
        ),
        timeout=12,
    )
    pnl_event = _extract_response_or_raise(pnl_raw, "cTrader unrealized P&L")
    pnl_money_digits = int(getattr(pnl_event, "moneyDigits", 0) or 0)
    unrealized_pnl = sum(
        _decode_money(getattr(item, "netUnrealizedPnL", 0), pnl_money_digits)
        for item in list(getattr(pnl_event, "positionUnrealizedPnL", []) or [])
    )

    return {
        "account_id": int(ACCOUNT_ID),
        "currency": currency,
        "balance": float(balance),
        "unrealized_pnl": float(unrealized_pnl),
        "equity": float(balance + unrealized_pnl),
        "money_digits": money_digits,
        "deposit_asset_id": deposit_asset_id,
        "source": "ctrader",
        "verified": True,
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


def get_deals_by_position_id(position_id: int, *, from_timestamp: int | None = None, to_timestamp: int | None = None):
    """Return cTrader execution deals for one position id.

    Historical deal data is the authoritative source for actual broker close
    price and realized P&L after a demo position disappears from reconcile.

    Some released cTrader OpenApiPy protobuf schemas still mark fromTimestamp
    and toTimestamp as required for ProtoOADealListByPositionIdReq even though
    the current public API docs describe them as optional. TcpProtocol queues
    serialization asynchronously, so leaving a required proto2 field unset can
    surface only as a response timeout. Always populate both fields.
    """
    now_ms = int(time.time() * 1000)
    # Historical endpoints reject future period boundaries.
    end_ms = now_ms if to_timestamp is None else min(now_ms, max(0, int(to_timestamp)))
    start_ms = (
        max(0, end_ms - 2 * 24 * 60 * 60 * 1000)
        if from_timestamp is None
        else max(0, int(from_timestamp))
    )
    # Server contract caps timestamps at 19 Jan 2038.
    end_ms = min(end_ms, 2_147_483_646_000)
    if end_ms < start_ms:
        start_ms, end_ms = end_ms, start_ms

    req = ProtoOADealListByPositionIdReq(
        ctidTraderAccountId=ACCOUNT_ID,
        positionId=int(position_id),
        fromTimestamp=start_ms,
        toTimestamp=end_ms,
    )

    raw = wait_for_deferred(client.send(req, responseTimeoutInSeconds=20), timeout=25)
    if isinstance(raw, dict) and raw.get("status") == "failed":
        raise RuntimeError(f"cTrader deal history failed: {raw.get('error') or 'unknown error'}")

    event = Protobuf.extract(raw)
    if getattr(event, "__class__", type("x", (object,), {})).__name__ == "ProtoOAErrorRes" or hasattr(event, "errorCode"):
        code = getattr(event, "errorCode", "ERR")
        description = getattr(event, "description", "")
        raise RuntimeError(f"cTrader deal history failed: {code} {description}".strip())

    return list(getattr(event, "deal", []) or [])


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
        volume=int(volume),  # cTrader protocol volume: cents of measurement units
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
    d = client.send(req, clientMsgId=client_msg_id, responseTimeoutInSeconds=12)

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
                        info["position_id"] = pos_id or None
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
                            symbol_money_digits_map[int(sid_hint)] = int(md)
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
    return client.send(req, responseTimeoutInSeconds=20)


def close_position(*, client, account_id, position_id, symbol_id, volume_lots):
    req = ProtoOAClosePositionReq(
        ctidTraderAccountId=account_id,
        positionId=position_id,
    )
    # ProtoOAClosePositionReq.volume is required and uses the same protocol
    # volume representation (0.01 of a measurement unit) as order volume.
    req.volume = volume_lots_to_units(int(symbol_id), volume_lots)
    return client.send(req, responseTimeoutInSeconds=20)

def modify_pending_order_sltp(client, account_id, order_id, version, stop_loss=None, take_profit=None, symbol_id=None):
    req = ProtoOAAmendOrderReq(
        ctidTraderAccountId = account_id,
        orderId             = order_id,
        version             = version,
    )
    if stop_loss   is not None: req.stopLoss   = _px_sym(symbol_id, stop_loss)
    if take_profit is not None: req.takeProfit = _px_sym(symbol_id, take_profit)
    return client.send(req, responseTimeoutInSeconds=20)

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
