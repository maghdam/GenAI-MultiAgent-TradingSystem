from __future__ import annotations

import os
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import quote

import httpx

from backend.security.credential_store import get_alpaca_secret


_DEFAULT_PAPER_BASE_URL = "https://paper-api.alpaca.markets"


class AlpacaPaperError(RuntimeError):
    """Raised when Alpaca Paper cannot safely return or mutate broker truth."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        ambiguous: bool = False,
        broker_details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.ambiguous = ambiguous
        self.broker_details = dict(broker_details or {})


def _base_url() -> str:
    return (os.getenv("ALPACA_PAPER_BASE_URL") or _DEFAULT_PAPER_BASE_URL).strip().rstrip("/")


def _credentials() -> tuple[str, str]:
    key_id = (get_alpaca_secret("ALPACA_API_KEY_ID") or "").strip()
    secret_key = (get_alpaca_secret("ALPACA_API_SECRET_KEY") or "").strip()
    if not key_id or not secret_key:
        raise AlpacaPaperError(
            "Alpaca Paper credentials are not configured. "
            "Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY."
        )
    return key_id, secret_key


def _request_json(path: str, *, params: dict[str, object] | None = None) -> Any:
    key_id, secret_key = _credentials()
    headers = {
        "APCA-API-KEY-ID": key_id,
        "APCA-API-SECRET-KEY": secret_key,
        "Accept": "application/json",
    }
    try:
        with httpx.Client(base_url=_base_url(), timeout=10.0, headers=headers) as client:
            response = client.get(path, params=params)
    except httpx.HTTPError as exc:
        raise AlpacaPaperError(
            f"Alpaca Paper request failed before a response was received for {path}."
        ) from exc

    if response.status_code >= 400:
        raise AlpacaPaperError(
            f"Alpaca Paper request returned HTTP {response.status_code} for {path}.",
            status_code=int(response.status_code),
        )

    try:
        return response.json()
    except ValueError as exc:
        raise AlpacaPaperError(
            f"Alpaca Paper returned a non-JSON response for {path}."
        ) from exc


def _request_optional_json(
    path: str,
    *,
    params: dict[str, object] | None = None,
) -> Any | None:
    try:
        return _request_json(path, params=params)
    except AlpacaPaperError as exc:
        if exc.status_code == 404:
            return None
        raise


def _safe_broker_error_details(response: httpx.Response) -> dict[str, object]:
    try:
        payload = response.json()
    except ValueError:
        return {}
    if not isinstance(payload, dict):
        return {}

    details: dict[str, object] = {}
    if payload.get("code") is not None:
        details["code"] = payload.get("code")
    if payload.get("message") is not None:
        details["message"] = str(payload.get("message"))
    return details


def _position_key(symbol_or_asset_id: str) -> str:
    normalized = str(symbol_or_asset_id or "").strip()
    if not normalized:
        raise ValueError("Alpaca symbol_or_asset_id must not be empty.")
    return normalized.replace("/", "")


def _request_mutation_json(
    method: str,
    path: str,
    *,
    params: dict[str, object] | None = None,
    body: dict[str, object] | None = None,
) -> Any:
    key_id, secret_key = _credentials()
    headers = {
        "APCA-API-KEY-ID": key_id,
        "APCA-API-SECRET-KEY": secret_key,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(base_url=_base_url(), timeout=10.0, headers=headers) as client:
            response = client.request(method, path, params=params, json=body)
    except httpx.HTTPError as exc:
        raise AlpacaPaperError(
            f"Alpaca Paper {method.upper()} failed before a response was received for {path}.",
            ambiguous=True,
        ) from exc

    if response.status_code >= 400:
        raise AlpacaPaperError(
            f"Alpaca Paper {method.upper()} returned HTTP {response.status_code} for {path}.",
            status_code=int(response.status_code),
            ambiguous=False,
            broker_details=_safe_broker_error_details(response),
        )

    try:
        return response.json()
    except ValueError as exc:
        raise AlpacaPaperError(
            f"Alpaca Paper returned a non-JSON response for {path}.",
            ambiguous=True,
        ) from exc


def _normalize_order(item: dict[str, object]) -> dict[str, object]:
    """Preserve broker-side parent/child identity without trusting it as protection.

    Alpaca returns nested bracket legs only when requested. A missing or
    malformed legs collection cannot be silently interpreted as protected.
    """
    fields = (
        "id", "client_order_id", "symbol", "asset_class", "side", "type",
        "time_in_force", "qty", "notional", "status", "filled_qty",
        "filled_avg_price", "submitted_at", "filled_at", "canceled_at",
        "expired_at", "failed_at", "order_class", "group_id",
        "group_role", "limit_price", "stop_price", "replaces",
        "replaced_by", "replaced_at",
    )
    normalized = {field: item.get(field) for field in fields}
    raw_legs = item.get("legs")
    if raw_legs is None:
        normalized["legs"] = None
    elif isinstance(raw_legs, list):
        children: list[dict[str, object]] = []
        for leg in raw_legs:
            if not isinstance(leg, dict) or leg.get("legs") not in (None, []):
                raise AlpacaPaperError("Alpaca Paper bracket legs had an unexpected shape.")
            children.append({field: leg.get(field) for field in fields})
            # Alpaca order legs are at most one level deep.
            children[-1]["legs"] = None
        normalized["legs"] = children
    else:
        raise AlpacaPaperError("Alpaca Paper bracket legs had an unexpected shape.")
    return normalized


def _normalize_position(item: dict[str, object]) -> dict[str, object]:
    return {
        "asset_id": item.get("asset_id"),
        "symbol": item.get("symbol"),
        "asset_class": item.get("asset_class"),
        "exchange": item.get("exchange"),
        "side": item.get("side"),
        "qty": item.get("qty"),
        "qty_available": item.get("qty_available"),
        "avg_entry_price": item.get("avg_entry_price"),
        "current_price": item.get("current_price"),
        "market_value": item.get("market_value"),
        "cost_basis": item.get("cost_basis"),
        "unrealized_pl": item.get("unrealized_pl"),
        "unrealized_plpc": item.get("unrealized_plpc"),
    }


def get_account() -> dict[str, object]:
    payload = _request_json("/v2/account")
    if not isinstance(payload, dict):
        raise AlpacaPaperError("Alpaca Paper account response had an unexpected shape.")

    return {
        "provider": "alpaca",
        "environment": "paper",
        "account_id": payload.get("id"),
        "account_number": payload.get("account_number"),
        "status": payload.get("status"),
        "crypto_status": payload.get("crypto_status"),
        "currency": payload.get("currency"),
        "equity": payload.get("equity"),
        "cash": payload.get("cash"),
        "buying_power": payload.get("buying_power"),
        "portfolio_value": payload.get("portfolio_value"),
        "trading_blocked": bool(payload.get("trading_blocked", False)),
        "account_blocked": bool(payload.get("account_blocked", False)),
        "trade_suspended_by_user": bool(payload.get("trade_suspended_by_user", False)),
        "shorting_enabled": bool(payload.get("shorting_enabled", False)),
    }


def list_positions() -> list[dict[str, object]]:
    payload = _request_json("/v2/positions")
    if not isinstance(payload, list):
        raise AlpacaPaperError("Alpaca Paper positions response had an unexpected shape.")

    rows: list[dict[str, object]] = []
    for item in payload:
        if isinstance(item, dict):
            rows.append(_normalize_position(item))
    return rows


def list_orders(*, status: str = "open", limit: int = 50) -> list[dict[str, object]]:
    normalized_status = str(status or "open").lower()
    if normalized_status not in {"open", "closed", "all"}:
        raise ValueError("Alpaca order status must be open, closed, or all.")
    normalized_limit = int(limit)
    if normalized_limit < 1 or normalized_limit > 500:
        raise ValueError("Alpaca order limit must be between 1 and 500.")

    payload = _request_json(
        "/v2/orders",
        params={
            "status": normalized_status,
            "limit": normalized_limit,
            "direction": "desc",
            "nested": True,
        },
    )
    if not isinstance(payload, list):
        raise AlpacaPaperError("Alpaca Paper orders response had an unexpected shape.")

    rows: list[dict[str, object]] = []
    for item in payload:
        if isinstance(item, dict):
            rows.append(_normalize_order(item))
    return rows


def get_order_by_client_order_id(
    client_order_id: str, *, nested: bool = False,
) -> dict[str, object] | None:
    normalized = str(client_order_id or "").strip()
    if not normalized:
        raise ValueError("Alpaca client_order_id must not be empty.")
    params: dict[str, object] = {"client_order_id": normalized}
    if nested:
        params["nested"] = True
    payload = _request_optional_json(
        "/v2/orders:by_client_order_id",
        params=params,
    )
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise AlpacaPaperError("Alpaca Paper order lookup response had an unexpected shape.")
    return _normalize_order(payload)


def get_order_by_id(order_id: str, *, nested: bool = False) -> dict[str, object] | None:
    normalized = str(order_id or "").strip()
    if not normalized:
        raise ValueError("Alpaca order_id must not be empty.")
    payload = _request_optional_json(
        f"/v2/orders/{quote(normalized, safe='')}",
        params={"nested": True} if nested else None,
    )
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise AlpacaPaperError("Alpaca Paper order lookup response had an unexpected shape.")
    return _normalize_order(payload)


def get_open_position(symbol_or_asset_id: str) -> dict[str, object] | None:
    encoded = quote(_position_key(symbol_or_asset_id), safe="")
    payload = _request_optional_json(f"/v2/positions/{encoded}")
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise AlpacaPaperError("Alpaca Paper position lookup response had an unexpected shape.")
    return _normalize_position(payload)


def build_equity_bracket_market_order(
    *,
    symbol: str,
    side: str,
    client_order_id: str,
    qty: float,
    reference_price: float,
    stop_loss: float,
    take_profit: float,
) -> dict[str, object]:
    """Build, but do NOT submit, a broker-native US-equity bracket request.

    Phase 10.14 is a fail-closed prerequisite. This pure validator must not
    be wired into runtime execution until durable parent/leg reconciliation
    and protected-position exit semantics are implemented and tested.
    """

    normalized_symbol = str(symbol or "").strip().upper()
    normalized_side = str(side or "").strip().lower()
    client_id = str(client_order_id or "").strip()
    if (
        not normalized_symbol
        or "/" in normalized_symbol
        or not normalized_symbol.isascii()
        or not normalized_symbol.replace(".", "").replace("-", "").isalnum()
    ):
        raise ValueError("A US-equity symbol is required for bracket protection.")
    if normalized_side not in {"buy", "sell"}:
        raise ValueError("Equity bracket side must be buy or sell.")
    if not client_id or len(client_id) > 48:
        raise ValueError("Equity bracket requires a durable client_order_id (maximum 48 characters).")

    def positive_decimal(field: str, raw: object) -> Decimal:
        if isinstance(raw, bool):
            raise ValueError(f"{field} must be a positive finite number.")
        try:
            value = Decimal(str(raw))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a positive finite number.") from exc
        if not value.is_finite() or value <= 0:
            raise ValueError(f"{field} must be a positive finite number.")
        return value

    quantity = positive_decimal("qty", qty)
    if quantity != quantity.to_integral_value():
        raise ValueError("Protected US-equity bracket entries require whole-share qty; notional/fractional entries are unsupported.")
    reference = positive_decimal("reference_price", reference_price)
    stop = positive_decimal("stop_loss", stop_loss)
    target = positive_decimal("take_profit", take_profit)

    for name, price in (("stop_loss", stop), ("take_profit", target)):
        # Alpaca's US-equity price increments are cents above $1 and
        # at most four decimal places below $1. Reject instead of rounding
        # silently, which could change the intended risk.
        places = 2 if price >= 1 else 4
        if price * (Decimal(10) ** places) != (price * (Decimal(10) ** places)).to_integral_value():
            raise ValueError(f"{name} has unsupported US-equity price precision.")

    if normalized_side == "buy" and not (stop < reference < target):
        raise ValueError("Buy bracket requires stop_loss < reference_price < take_profit.")
    if normalized_side == "sell" and not (target < reference < stop):
        raise ValueError("Sell bracket requires take_profit < reference_price < stop_loss.")

    return {
        "symbol": normalized_symbol,
        "side": normalized_side,
        "type": "market",
        "time_in_force": "gtc",
        "order_class": "bracket",
        "client_order_id": client_id,
        "qty": format(quantity, "f"),
        "take_profit": {"limit_price": format(target, "f")},
        # Stop MARKET, not stop-limit: protects against gaps better but
        # still has fill-price/slippage risk and is not a guaranteed price.
        "stop_loss": {"stop_price": format(stop, "f")},
    }


def submit_market_order(
    *,
    symbol: str,
    asset_class: str,
    side: str,
    client_order_id: str,
    qty: float | None = None,
    notional: float | None = None,
) -> dict[str, object]:
    normalized_symbol = str(symbol or "").strip().upper()
    normalized_asset_class = str(asset_class or "").strip().lower()
    normalized_side = str(side or "").strip().lower()
    normalized_client_order_id = str(client_order_id or "").strip()

    if not normalized_symbol:
        raise ValueError("Alpaca symbol must not be empty.")
    if normalized_asset_class not in {"us_equity", "crypto"}:
        raise ValueError("Alpaca asset_class must be us_equity or crypto.")
    if normalized_side not in {"buy", "sell"}:
        raise ValueError("Alpaca side must be buy or sell.")
    if not normalized_client_order_id:
        raise ValueError("Alpaca client_order_id must not be empty.")
    if (qty is None) == (notional is None):
        raise ValueError("Provide exactly one of qty or notional.")
    if qty is not None and float(qty) <= 0:
        raise ValueError("Alpaca qty must be greater than zero.")
    if notional is not None and float(notional) <= 0:
        raise ValueError("Alpaca notional must be greater than zero.")

    body: dict[str, object] = {
        "symbol": normalized_symbol,
        "side": normalized_side,
        "type": "market",
        "time_in_force": "gtc" if normalized_asset_class == "crypto" else "day",
        "client_order_id": normalized_client_order_id,
    }
    if qty is not None:
        body["qty"] = str(qty)
    else:
        body["notional"] = str(notional)

    payload = _request_mutation_json("POST", "/v2/orders", body=body)
    if not isinstance(payload, dict):
        raise AlpacaPaperError(
            "Alpaca Paper order submission response had an unexpected shape.",
            ambiguous=True,
        )
    return _normalize_order(payload)


def close_position(symbol_or_asset_id: str) -> dict[str, object]:
    encoded = quote(_position_key(symbol_or_asset_id), safe="")
    payload = _request_mutation_json(
        "DELETE",
        f"/v2/positions/{encoded}",
        params={"percentage": 100},
    )
    if not isinstance(payload, dict):
        raise AlpacaPaperError(
            "Alpaca Paper close-position response had an unexpected shape.",
            ambiguous=True,
        )
    return _normalize_order(payload)
