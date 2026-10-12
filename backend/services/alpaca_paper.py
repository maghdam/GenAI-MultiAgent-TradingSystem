from __future__ import annotations

import os
from typing import Any

import httpx

from backend.security.credential_store import get_alpaca_secret


_DEFAULT_PAPER_BASE_URL = "https://paper-api.alpaca.markets"


class AlpacaPaperError(RuntimeError):
    """Raised when the read-only Alpaca Paper integration cannot return broker truth."""


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
            f"Alpaca Paper request returned HTTP {response.status_code} for {path}."
        )

    try:
        return response.json()
    except ValueError as exc:
        raise AlpacaPaperError(
            f"Alpaca Paper returned a non-JSON response for {path}."
        ) from exc


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
        if not isinstance(item, dict):
            continue
        rows.append(
            {
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
        )
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
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "id": item.get("id"),
                "client_order_id": item.get("client_order_id"),
                "symbol": item.get("symbol"),
                "asset_class": item.get("asset_class"),
                "side": item.get("side"),
                "type": item.get("type"),
                "time_in_force": item.get("time_in_force"),
                "qty": item.get("qty"),
                "notional": item.get("notional"),
                "status": item.get("status"),
                "filled_qty": item.get("filled_qty"),
                "filled_avg_price": item.get("filled_avg_price"),
                "submitted_at": item.get("submitted_at"),
                "filled_at": item.get("filled_at"),
            }
        )
    return rows
