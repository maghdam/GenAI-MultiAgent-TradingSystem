from __future__ import annotations

import importlib

from fastapi.testclient import TestClient
import pytest

from backend.security import credential_store
from backend.services import alpaca_paper


def test_alpaca_windows_credentials_precede_environment(monkeypatch) -> None:
    monkeypatch.setattr(credential_store, "_is_windows", lambda: True)
    monkeypatch.setattr(
        credential_store,
        "read_windows_alpaca_credential",
        lambda name: "vault-value" if name == "ALPACA_API_KEY_ID" else None,
    )

    result = credential_store.get_alpaca_secret_with_source(
        "ALPACA_API_KEY_ID",
        environ={"ALPACA_API_KEY_ID": "env-value"},
    )

    assert result.value == "vault-value"
    assert result.source == "windows_credential_manager"


def test_alpaca_environment_is_non_windows_fallback(monkeypatch) -> None:
    monkeypatch.setattr(credential_store, "_is_windows", lambda: False)

    result = credential_store.get_alpaca_secret_with_source(
        "ALPACA_API_SECRET_KEY",
        environ={"ALPACA_API_SECRET_KEY": "env-secret"},
    )

    assert result.value == "env-secret"
    assert result.source == "environment"


def test_alpaca_unknown_credential_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported Alpaca credential"):
        credential_store.get_alpaca_secret("ALPACA_PASSWORD", environ={})


def test_alpaca_account_normalization(monkeypatch) -> None:
    monkeypatch.setattr(
        alpaca_paper,
        "_request_json",
        lambda path, params=None: {
            "id": "account-uuid",
            "account_number": "PA123",
            "status": "ACTIVE",
            "crypto_status": "ACTIVE",
            "currency": "USD",
            "equity": "123456.78",
            "cash": "1000.00",
            "buying_power": "250000.00",
            "portfolio_value": "123456.78",
            "trading_blocked": False,
            "account_blocked": False,
            "trade_suspended_by_user": False,
            "shorting_enabled": True,
            "ignored_secret_like_field": "not-forwarded",
        },
    )

    payload = alpaca_paper.get_account()

    assert payload["provider"] == "alpaca"
    assert payload["environment"] == "paper"
    assert payload["account_id"] == "account-uuid"
    assert payload["status"] == "ACTIVE"
    assert payload["equity"] == "123456.78"
    assert payload["shorting_enabled"] is True
    assert "ignored_secret_like_field" not in payload


def test_alpaca_positions_normalization(monkeypatch) -> None:
    monkeypatch.setattr(
        alpaca_paper,
        "_request_json",
        lambda path, params=None: [
            {
                "asset_id": "asset-1",
                "symbol": "AAPL",
                "asset_class": "us_equity",
                "exchange": "NASDAQ",
                "side": "long",
                "qty": "2",
                "qty_available": "2",
                "avg_entry_price": "100",
                "current_price": "110",
                "market_value": "220",
                "cost_basis": "200",
                "unrealized_pl": "20",
                "unrealized_plpc": "0.1",
                "raw_field": "not-forwarded",
            }
        ],
    )

    rows = alpaca_paper.list_positions()

    assert rows == [
        {
            "asset_id": "asset-1",
            "symbol": "AAPL",
            "asset_class": "us_equity",
            "exchange": "NASDAQ",
            "side": "long",
            "qty": "2",
            "qty_available": "2",
            "avg_entry_price": "100",
            "current_price": "110",
            "market_value": "220",
            "cost_basis": "200",
            "unrealized_pl": "20",
            "unrealized_plpc": "0.1",
        }
    ]


def test_alpaca_orders_validate_filters_and_normalize(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_request(path: str, *, params=None):
        captured["path"] = path
        captured["params"] = params
        return [
            {
                "id": "order-1",
                "client_order_id": "tradeagent-1",
                "symbol": "SPY",
                "asset_class": "us_equity",
                "side": "buy",
                "type": "limit",
                "time_in_force": "day",
                "qty": "1",
                "notional": None,
                "status": "filled",
                "filled_qty": "1",
                "filled_avg_price": "500",
                "submitted_at": "2026-10-12T12:00:00Z",
                "filled_at": "2026-10-12T12:00:01Z",
            }
        ]

    monkeypatch.setattr(alpaca_paper, "_request_json", _fake_request)

    rows = alpaca_paper.list_orders(status="all", limit=25)

    assert captured["path"] == "/v2/orders"
    assert captured["params"] == {
        "status": "all",
        "limit": 25,
        "direction": "desc",
        "nested": True,
    }
    assert rows[0]["client_order_id"] == "tradeagent-1"
    assert rows[0]["status"] == "filled"

    with pytest.raises(ValueError, match="open, closed, or all"):
        alpaca_paper.list_orders(status="pending")
    with pytest.raises(ValueError, match="between 1 and 500"):
        alpaca_paper.list_orders(limit=501)


def test_alpaca_http_request_uses_paper_host_and_header_credentials(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return {"status": "ACTIVE"}

    class _Client:
        def __init__(self, *, base_url, timeout, headers):
            captured["base_url"] = base_url
            captured["timeout"] = timeout
            captured["headers"] = headers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, path, params=None):
            captured["path"] = path
            captured["params"] = params
            return _Response()

    monkeypatch.setattr(alpaca_paper, "get_alpaca_secret", lambda name: {
        "ALPACA_API_KEY_ID": "key-id",
        "ALPACA_API_SECRET_KEY": "secret-key",
    }[name])
    monkeypatch.setattr(alpaca_paper.httpx, "Client", _Client)
    monkeypatch.delenv("ALPACA_PAPER_BASE_URL", raising=False)

    result = alpaca_paper._request_json("/v2/account")

    assert result == {"status": "ACTIVE"}
    assert captured["base_url"] == "https://paper-api.alpaca.markets"
    assert captured["path"] == "/v2/account"
    headers = captured["headers"]
    assert headers["APCA-API-KEY-ID"] == "key-id"
    assert headers["APCA-API-SECRET-KEY"] == "secret-key"


def test_alpaca_api_endpoints_are_read_only_and_delegated(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_EVENT_INTELLIGENCE_ON_BOOT", "0")

    router_module = importlib.import_module("backend.api.router")
    app_module = importlib.import_module("backend.app")

    monkeypatch.setattr(
        router_module,
        "get_alpaca_paper_account",
        lambda: {"provider": "alpaca", "environment": "paper", "status": "ACTIVE"},
    )
    monkeypatch.setattr(
        router_module,
        "list_alpaca_paper_positions",
        lambda: [{"symbol": "AAPL", "qty": "2"}],
    )
    monkeypatch.setattr(
        router_module,
        "list_alpaca_paper_orders",
        lambda *, status, limit: [{"symbol": "SPY", "status": status, "limit": limit}],
    )

    with TestClient(app_module.app) as client:
        account = client.get("/api/alpaca/paper/account")
        positions = client.get("/api/alpaca/paper/positions")
        orders = client.get("/api/alpaca/paper/orders?status=all&limit=10")

    assert account.status_code == 200
    assert account.json()["status"] == "ACTIVE"
    assert positions.status_code == 200
    assert positions.json()[0]["symbol"] == "AAPL"
    assert orders.status_code == 200
    assert orders.json() == [{"symbol": "SPY", "status": "all", "limit": 10}]


def test_alpaca_api_fails_closed_without_credentials(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_EVENT_INTELLIGENCE_ON_BOOT", "0")

    router_module = importlib.import_module("backend.api.router")
    app_module = importlib.import_module("backend.app")

    def _missing():
        raise alpaca_paper.AlpacaPaperError("Alpaca Paper credentials are not configured.")

    monkeypatch.setattr(router_module, "get_alpaca_paper_account", _missing)

    with TestClient(app_module.app) as client:
        response = client.get("/api/alpaca/paper/account")

    assert response.status_code == 503
    assert response.json()["detail"] == "Alpaca Paper credentials are not configured."
