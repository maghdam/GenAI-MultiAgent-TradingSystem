from __future__ import annotations

from copy import deepcopy

import pytest

from backend.services import alpaca_bracket_observations, alpaca_paper
from backend.storage.db import init_db
from backend.storage.repositories import (
    create_order_intent,
    get_order_intent_by_id,
    latest_alpaca_bracket_observation,
    list_trade_audits,
    record_alpaca_bracket_observation,
)


def _intent(**details_override):
    details = {
        "broker_provider": "alpaca",
        "broker_environment": "paper",
        "asset_class": "us_equity",
        "protection_mode": "equity_bracket",
        "client_order_id": "tradeagent-alpaca-bracket-1",
    }
    details.update(details_override)
    return create_order_intent(
        symbol="SPY", timeframe="M5", strategy="sma_cross",
        direction="long", intent_type="open", status="accepted",
        confidence=0.85, entry_price=101.0, stop_loss=99.0,
        take_profit=104.0, quantity=2.0, rationale="test-protection-truth",
        details=details,
    )


def _order():
    return {
        "id": "parent-1", "client_order_id": "tradeagent-alpaca-bracket-1",
        "symbol": "SPY", "asset_class": "us_equity", "side": "buy",
        "order_class": "bracket", "group_role": "entry",
        "group_id": "group-1", "status": "filled", "qty": "2", "filled_qty": "2",
        "legs": [
            {
                "id": "limit-1", "symbol": "SPY", "asset_class": "us_equity",
                "side": "sell", "group_id": "group-1", "group_role": "take_profit",
                "type": "limit", "limit_price": "104", "status": "new",
                "qty": "2", "filled_qty": "0",
            },
            {
                "id": "stop-1", "symbol": "SPY", "asset_class": "us_equity",
                "side": "sell", "group_id": "group-1", "group_role": "stop_loss",
                "type": "stop", "stop_price": "99", "status": "new",
                "qty": "2", "filled_qty": "0",
            },
        ],
    }


def _position():
    return {
        "symbol": "SPY", "asset_class": "us_equity", "side": "long",
        "qty": "2", "avg_entry_price": "101",
    }


def test_exact_nested_parent_and_two_exit_legs_are_durably_observed(monkeypatch):
    intent = _intent()
    calls = []
    def lookup(client_id, *, nested=False):
        calls.append((client_id, nested))
        return alpaca_paper._normalize_order(_order())
    monkeypatch.setattr(alpaca_bracket_observations, "get_order_by_client_order_id", lookup)
    monkeypatch.setattr(alpaca_bracket_observations, "get_open_position", lambda symbol: _position())
    observed = alpaca_bracket_observations.observe_paper_equity_bracket(intent_id=intent.id)
    assert calls == [("tradeagent-alpaca-bracket-1", True)]
    assert observed["verified_at_observation"] is True
    assert observed["current_protection_verified"] is False
    assert observed["parent_order_id"] == "parent-1"
    assert observed["stop_order_id"] == "stop-1"
    assert observed["take_profit_order_id"] == "limit-1"
    assert observed["broker_snapshot"]["broker_parent"]["legs"][1]["id"] == "stop-1"
    assert latest_alpaca_bracket_observation(intent.id)["id"] == observed["id"]
    assert get_order_intent_by_id(intent.id).status == "accepted"
    assert list_trade_audits(1)[0].event_type == "alpaca_paper_bracket_protection_observed"


def test_successful_snapshot_never_masks_new_missing_protection_after_restart(monkeypatch):
    intent = _intent()
    state = {"parent": _order()}
    monkeypatch.setattr(
        alpaca_bracket_observations, "get_order_by_client_order_id",
        lambda client_id, *, nested: alpaca_paper._normalize_order(state["parent"]),
    )
    monkeypatch.setattr(alpaca_bracket_observations, "get_open_position", lambda symbol: _position())
    first = alpaca_bracket_observations.observe_paper_equity_bracket(intent_id=intent.id)
    assert first["verified_at_observation"]
    # Calling idempotent schema migration again simulates a restarted service;
    # old DB observation remains historical even when legs disappear.
    init_db()
    state["parent"] = {**_order(), "legs": []}
    second = alpaca_bracket_observations.observe_paper_equity_bracket(intent_id=intent.id)
    assert second["id"] > first["id"]
    assert second["verified_at_observation"] is False
    assert second["stop_order_id"] is None
    assert second["take_profit_order_id"] is None
    assert latest_alpaca_bracket_observation(intent.id)["id"] == second["id"]
    assert get_order_intent_by_id(intent.id).status == "accepted"
    assert list_trade_audits(1)[0].event_type == "alpaca_paper_bracket_protection_unverified"


@pytest.mark.parametrize("bad_mode", [
    {"protection_mode": "simple"},
    {"asset_class": "crypto"},
    {"broker_environment": "live"},
    {"broker_provider": "ctrader"},
])
def test_wrong_venue_scope_never_reaches_broker(monkeypatch, bad_mode):
    intent = _intent(**bad_mode)
    monkeypatch.setattr(
        alpaca_bracket_observations, "get_order_by_client_order_id",
        lambda *a, **k: pytest.fail("wrong-provenance read must not reach broker"),
    )
    with pytest.raises(ValueError, match="Only explicitly reserved"):
        alpaca_bracket_observations.observe_paper_equity_bracket(intent_id=intent.id)
    assert latest_alpaca_bracket_observation(intent.id) is None


def test_unreserved_client_id_cannot_be_looked_up(monkeypatch):
    intent = _intent(client_order_id="")
    monkeypatch.setattr(
        alpaca_bracket_observations, "get_order_by_client_order_id",
        lambda *a, **k: pytest.fail("missing ID cannot reach broker"),
    )
    with pytest.raises(ValueError, match="no durable reserved"):
        alpaca_bracket_observations.observe_paper_equity_bracket(intent_id=intent.id)
    assert latest_alpaca_bracket_observation(intent.id) is None


@pytest.mark.parametrize("situation", ["network_error", "parent_absent", "stop_missing", "position_absent", "filled_stop"])
def test_broker_unknown_or_lost_protection_is_durably_unverified(monkeypatch, situation):
    intent = _intent()
    parent = _order()
    if situation == "stop_missing":
        parent["legs"] = parent["legs"][:1]
    elif situation == "filled_stop":
        parent["legs"][1]["status"] = "filled"
        parent["legs"][1]["filled_qty"] = "2"
    if situation == "network_error":
        def lookup(*args, **kwargs):
            raise alpaca_paper.AlpacaPaperError("timeout", ambiguous=True)
    elif situation == "parent_absent":
        lookup = lambda *args, **kwargs: None
    else:
        lookup = lambda *args, **kwargs: alpaca_paper._normalize_order(parent)
    monkeypatch.setattr(alpaca_bracket_observations, "get_order_by_client_order_id", lookup)
    monkeypatch.setattr(
        alpaca_bracket_observations, "get_open_position",
        lambda symbol: None if situation == "position_absent" else _position(),
    )
    obs = alpaca_bracket_observations.observe_paper_equity_bracket(intent_id=intent.id)
    assert not obs["verified_at_observation"]
    assert not obs["current_protection_verified"]
    assert obs["parent_order_id"] is None and obs["stop_order_id"] is None
    assert latest_alpaca_bracket_observation(intent.id)["id"] == obs["id"]
    assert "unverified" in list_trade_audits(1)[0].event_type


def test_unverified_observations_reject_forged_leg_ids():
    intent = _intent()
    with pytest.raises(ValueError, match="must not carry"):
        record_alpaca_bracket_observation(
            intent_id=intent.id, client_order_id="tradeagent-alpaca-bracket-1",
            verified_at_observation=False, reason="unknown", stop_order_id="forged-id",
        )


def test_recorded_broker_proof_is_historical_not_authority_to_trade(monkeypatch):
    intent = _intent()
    monkeypatch.setattr(alpaca_bracket_observations, "get_order_by_client_order_id", lambda *a, **k: _order())
    monkeypatch.setattr(alpaca_bracket_observations, "get_open_position", lambda symbol: _position())
    monkeypatch.setattr(
        alpaca_paper, "_request_mutation_json",
        lambda *a, **k: pytest.fail("read-only reconciliation must not submit/cancel/close"),
    )
    obs = alpaca_bracket_observations.observe_paper_equity_bracket(intent_id=intent.id)
    assert obs["verified_at_observation"]
    assert obs["current_protection_verified"] is False
    assert get_order_intent_by_id(intent.id).status == "accepted"
