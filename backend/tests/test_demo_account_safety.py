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


def test_sync_demo_position_targets_updates_and_verifies_broker(monkeypatch) -> None:
    state = {
        "symbol_name": "XAUUSD",
        "symbol_id": 7,
        "position_id": 456,
        "direction": "buy",
        "entry_price": 100.0,
        "stop_loss": None,
        "take_profit": None,
    }
    captured = {}

    monkeypatch.setattr(ctd, "is_demo_account_confirmed", lambda: True)
    monkeypatch.setattr(ctd, "get_account_verification_error", lambda: None)
    monkeypatch.setattr(ctd, "symbol_name_to_id", {"XAUUSD": 7})
    monkeypatch.setattr(ctd, "symbol_digits_map", {7: 2})
    monkeypatch.setattr(ctd, "ACCOUNT_ID", 123)
    monkeypatch.setattr(ctd, "get_open_positions", lambda: [dict(state)])

    def _modify(**kwargs):
        captured.update(kwargs)
        state["stop_loss"] = kwargs["stop_loss"]
        state["take_profit"] = kwargs["take_profit"]
        return object()

    monkeypatch.setattr(ctd, "modify_position_sltp", _modify)
    monkeypatch.setattr(ctd, "wait_for_deferred", lambda deferred, timeout: {"status": "ok"})

    result = CTraderBrokerAdapter().sync_demo_position_targets(
        symbol="XAUUSD",
        direction="long",
        stop_loss=99.0,
        take_profit=102.0,
        position_id=456,
    )

    assert result["verified"] is True
    assert result["status"] == "synced"
    assert captured["position_id"] == 456
    assert captured["stop_loss"] == 99.0
    assert captured["take_profit"] == 102.0


def test_sync_demo_position_targets_skips_amend_when_already_synced(monkeypatch) -> None:
    monkeypatch.setattr(ctd, "is_demo_account_confirmed", lambda: True)
    monkeypatch.setattr(ctd, "get_account_verification_error", lambda: None)
    monkeypatch.setattr(ctd, "symbol_name_to_id", {"XAUUSD": 7})
    monkeypatch.setattr(ctd, "symbol_digits_map", {7: 2})
    monkeypatch.setattr(
        ctd,
        "get_open_positions",
        lambda: [
            {
                "symbol_name": "XAUUSD",
                "symbol_id": 7,
                "position_id": 456,
                "direction": "buy",
                "entry_price": 100.0,
                "stop_loss": 99.0,
                "take_profit": 102.0,
            }
        ],
    )
    monkeypatch.setattr(
        ctd,
        "modify_position_sltp",
        lambda **kwargs: pytest.fail("already-synced targets must not be amended"),
    )

    result = CTraderBrokerAdapter().sync_demo_position_targets(
        symbol="XAUUSD",
        direction="long",
        stop_loss=99.0,
        take_profit=102.0,
    )

    assert result["verified"] is True
    assert result["status"] == "already_synced"
    assert result["position_id"] == 456


def test_sync_demo_position_targets_surfaces_ctrader_reject_reason(monkeypatch) -> None:
    state = {
        "symbol_name": "NAS100",
        "symbol_id": 116,
        "position_id": 56980461,
        "direction": "buy",
        "entry_price": 29486.2,
        "stop_loss": None,
        "take_profit": None,
    }
    event = SimpleNamespace(
        errorCode="TRADING_BAD_STOPS",
        description="Protection price is invalid",
        rejectReason=0,
        executionType=None,
    )

    monkeypatch.setattr(ctd, "is_demo_account_confirmed", lambda: True)
    monkeypatch.setattr(ctd, "get_account_verification_error", lambda: None)
    monkeypatch.setattr(ctd, "symbol_name_to_id", {"NAS100": 116})
    monkeypatch.setattr(ctd, "symbol_digits_map", {116: 2})
    monkeypatch.setattr(ctd, "ACCOUNT_ID", 123)
    monkeypatch.setattr(ctd, "get_open_positions", lambda: [dict(state)])
    monkeypatch.setattr(ctd, "modify_position_sltp", lambda **kwargs: object())
    monkeypatch.setattr(ctd, "wait_for_deferred", lambda deferred, timeout: object())
    monkeypatch.setattr(ctd.Protobuf, "extract", lambda raw: event)
    monkeypatch.setattr(
        ctd,
        "MessageToDict",
        lambda event, preserving_proto_field_name=True: {
            "errorCode": "TRADING_BAD_STOPS",
            "description": "Protection price is invalid",
        },
    )

    with pytest.raises(RuntimeError, match="TRADING_BAD_STOPS"):
        CTraderBrokerAdapter().sync_demo_position_targets(
            symbol="NAS100",
            direction="long",
            stop_loss=29476.48,
            take_profit=29511.75,
            position_id=56980461,
        )


def test_sync_demo_position_targets_verification_error_includes_observed_values(monkeypatch) -> None:
    state = {
        "symbol_name": "US30",
        "symbol_id": 117,
        "position_id": 56980462,
        "direction": "buy",
        "entry_price": 51686.0,
        "stop_loss": None,
        "take_profit": None,
    }
    event = SimpleNamespace(errorCode=None, description=None, rejectReason=0, executionType=4)

    monkeypatch.setattr(ctd, "is_demo_account_confirmed", lambda: True)
    monkeypatch.setattr(ctd, "get_account_verification_error", lambda: None)
    monkeypatch.setattr(ctd, "symbol_name_to_id", {"US30": 117})
    monkeypatch.setattr(ctd, "symbol_digits_map", {117: 2})
    monkeypatch.setattr(ctd, "ACCOUNT_ID", 123)
    monkeypatch.setattr(ctd, "get_open_positions", lambda: [dict(state)])
    monkeypatch.setattr(ctd, "modify_position_sltp", lambda **kwargs: object())
    monkeypatch.setattr(ctd, "wait_for_deferred", lambda deferred, timeout: object())
    monkeypatch.setattr(ctd.Protobuf, "extract", lambda raw: event)
    monkeypatch.setattr(
        ctd,
        "MessageToDict",
        lambda event, preserving_proto_field_name=True: {"executionType": "ORDER_REPLACED"},
    )
    monkeypatch.setattr("backend.adapters.ctrader.time.sleep", lambda seconds: None)

    with pytest.raises(RuntimeError, match="observed_sl=None observed_tp=None"):
        CTraderBrokerAdapter().sync_demo_position_targets(
            symbol="US30",
            direction="long",
            stop_loss=51692.65,
            take_profit=51771.85,
            position_id=56980462,
        )
