from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from backend import ctrader_client as ctd
from backend.adapters.ctrader import CTraderBrokerAdapter
from backend.api.router import v2_set_config
from backend.domain.models import EngineConfig


class _Deferred:
    def __init__(self) -> None:
        self.callbacks = None

    def addCallbacks(self, success, error):
        self.callbacks = (success, error)
        return self


def test_account_list_confirms_demo_before_account_authorization(monkeypatch) -> None:
    sent = []
    monkeypatch.setattr(ctd, "ACCOUNT_ID", 123)
    monkeypatch.setattr(ctd, "HOST_TYPE", "demo")
    monkeypatch.setattr(
        ctd.Protobuf,
        "extract",
        lambda _: SimpleNamespace(
            ctidTraderAccount=[SimpleNamespace(ctidTraderAccountId=123, isLive=False)]
        ),
    )
    monkeypatch.setattr(ctd.client, "send", lambda request: sent.append(request) or _Deferred())
    monkeypatch.setattr(ctd, "ACCOUNT_IS_DEMO", None)
    monkeypatch.setattr(ctd, "ACCOUNT_VERIFICATION_ERROR", None)
    monkeypatch.setattr(ctd, "AUTH_ERROR", None)

    ctd.account_list_response_cb(object())

    assert ctd.ACCOUNT_IS_DEMO is True
    assert ctd.ACCOUNT_VERIFICATION_ERROR is None
    assert len(sent) == 1
    assert sent[0].ctidTraderAccountId == 123


def test_account_list_blocks_live_account_without_authorizing(monkeypatch) -> None:
    sent = []
    monkeypatch.setattr(ctd, "ACCOUNT_ID", 123)
    monkeypatch.setattr(ctd, "HOST_TYPE", "demo")
    monkeypatch.setattr(
        ctd.Protobuf,
        "extract",
        lambda _: SimpleNamespace(
            ctidTraderAccount=[SimpleNamespace(ctidTraderAccountId=123, isLive=True)]
        ),
    )
    monkeypatch.setattr(ctd.client, "send", lambda request: sent.append(request) or _Deferred())
    monkeypatch.setattr(ctd, "ACCOUNT_IS_DEMO", None)
    monkeypatch.setattr(ctd, "ACCOUNT_VERIFICATION_ERROR", None)
    monkeypatch.setattr(ctd, "AUTH_ERROR", None)

    ctd.account_list_response_cb(object())

    assert ctd.ACCOUNT_IS_DEMO is False
    assert "live" in (ctd.ACCOUNT_VERIFICATION_ERROR or "").lower()
    assert sent == []


def test_demo_order_is_blocked_without_verified_demo_account(monkeypatch) -> None:
    monkeypatch.setattr(ctd, "is_demo_account_confirmed", lambda: False)
    monkeypatch.setattr(ctd, "get_account_verification_error", lambda: "account type unknown")
    monkeypatch.setattr(ctd, "place_order", lambda **kwargs: pytest.fail("order must stay blocked"))

    with pytest.raises(RuntimeError, match="account type unknown"):
        CTraderBrokerAdapter().place_demo_market_order(
            symbol="XAUUSD",
            direction="long",
            quantity_lots=0.1,
        )


def test_verified_demo_order_uses_broker_symbol_and_lot_volume(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(ctd, "is_demo_account_confirmed", lambda: True)
    monkeypatch.setattr(ctd, "symbol_name_to_id", {"XAUUSD": 7})
    monkeypatch.setattr(ctd, "ACCOUNT_ID", 123)
    monkeypatch.setattr(ctd, "volume_lots_to_units", lambda symbol_id, lots: 250)
    monkeypatch.setattr(ctd, "place_order", lambda **kwargs: captured.update(kwargs) or object())
    monkeypatch.setattr(
        ctd,
        "wait_for_deferred",
        lambda deferred, timeout: {"status": "executed", "position_id": 456, "ack": {"ok": True}},
    )

    result = CTraderBrokerAdapter().place_demo_market_order(
        symbol="xauusd",
        direction="short",
        quantity_lots=0.025,
        stop_loss=101.0,
        take_profit=98.0,
    )

    assert captured["account_id"] == 123
    assert captured["symbol_id"] == 7
    assert captured["side"] == "SELL"
    assert captured["volume"] == 250
    assert result["account_type"] == "demo"
    assert result["position_id"] == 456


def test_demo_confirmation_requires_connection_authorization_host_and_account_type(monkeypatch) -> None:
    monkeypatch.setattr(ctd, "CONNECTED", True)
    monkeypatch.setattr(ctd, "AUTHORIZED", True)
    monkeypatch.setattr(ctd, "HOST_TYPE", "demo")
    monkeypatch.setattr(ctd, "ACCOUNT_IS_DEMO", True)

    assert ctd.is_demo_account_confirmed() is True

    monkeypatch.setattr(ctd, "ACCOUNT_IS_DEMO", False)
    assert ctd.is_demo_account_confirmed() is False


def test_config_api_rejects_live_execution_request() -> None:
    with pytest.raises(HTTPException, match="Live-account execution is not supported") as exc:
        asyncio.run(v2_set_config(EngineConfig(allow_live=True)))

    assert exc.value.status_code == 400
