from __future__ import annotations

from copy import deepcopy

import pytest

from backend.services import (
    alpaca_bracket_observations, alpaca_bracket_reconcile, alpaca_paper,
)
from backend.storage.db import init_db
from backend.storage.repositories import (
    create_order_intent, get_order_intent_by_id,
    first_verified_alpaca_bracket_observation,
    latest_alpaca_bracket_observation, list_trade_audits,
)


def _intent(*, state="parent_acknowledged_protection_unverified", **extra):
    details = {
        "broker_provider": "alpaca", "broker_environment": "paper",
        "asset_class": "us_equity", "protection_mode": "equity_bracket",
        "client_order_id": "ta-ap-brkt-check",
        "outcome_state": state, "reconciliation_resolved": False,
        "automatic_retry": False, "broker_native_protection_verified": False,
    }
    details.update(extra)
    return create_order_intent(
        symbol="SPY", timeframe="M5", strategy="manual", direction="long",
        intent_type="open", status="accepted", confidence=1,
        entry_price=101, stop_loss=99, take_profit=104, quantity=2,
        rationale="explicit mock reconciliation", details=details,
    )


def _parent():
    return {
        "id": "parent-A", "client_order_id": "ta-ap-brkt-check",
        "order_class": "bracket", "asset_class": "us_equity",
        "symbol": "SPY", "side": "buy", "status": "filled",
        "group_role": "entry", "group_id": "group-A",
        "qty": "2", "filled_qty": "2",
        "legs": [
            {"id": "target-A", "symbol": "SPY", "asset_class": "us_equity",
             "side": "sell", "type": "limit", "limit_price": "104",
             "group_role": "take_profit", "group_id": "group-A",
             "status": "new", "qty": "2", "filled_qty": "0"},
            {"id": "stop-A", "symbol": "SPY", "asset_class": "us_equity",
             "side": "sell", "type": "stop", "stop_price": "99",
             "group_role": "stop_loss", "group_id": "group-A",
             "status": "new", "qty": "2", "filled_qty": "0"},
        ],
    }


def _position():
    return {
        "symbol": "SPY", "asset_class": "us_equity",
        "side": "long", "qty": "2", "avg_entry_price": "101",
    }


@pytest.fixture
def broker(monkeypatch):
    state = {"parent": _parent(), "position": _position(), "error": None}
    calls = []
    monkeypatch.setattr(
        alpaca_bracket_reconcile, "_base_url",
        lambda: "https://paper-api.alpaca.markets",
    )

    def lookup(client_id, *, nested=False):
        calls.append((client_id, nested))
        if state["error"] == "parent":
            raise alpaca_paper.AlpacaPaperError("Alpaca timeout", ambiguous=True)
        return (alpaca_paper._normalize_order(deepcopy(state["parent"]))
                if state["parent"] is not None else None)

    def position(symbol):
        if state["error"] == "position":
            raise alpaca_paper.AlpacaPaperError("Alpaca position timeout")
        return deepcopy(state["position"])

    monkeypatch.setattr(alpaca_bracket_observations, "get_order_by_client_order_id", lookup)
    monkeypatch.setattr(alpaca_bracket_observations, "get_open_position", position)
    monkeypatch.setattr(
        alpaca_paper, "_request_mutation_json",
        lambda *args, **kwargs: pytest.fail("Reconciliation must never mutate broker orders"),
    )
    return state, calls


def _check(intent):
    return alpaca_bracket_reconcile.reconcile_dispatched_paper_equity_bracket(intent_id=intent.id)


def test_filled_parent_and_independent_active_legs_snapshot_stays_unresolved(broker):
    state, calls = broker
    intent = _intent()
    result = _check(intent)
    assert calls == [("ta-ap-brkt-check", True)]
    assert result["classification"] == "complete_bracket_verified_at_observation_only"
    assert result["verified_at_observation"] is True
    assert result["current_protection_verified"] is False
    assert result["reconciliation_resolved"] is False
    assert result["automatic_retry"] is False
    assert result["parent_order_id"] == "parent-A"
    assert result["stop_order_id"] == "stop-A"
    assert result["take_profit_order_id"] == "target-A"
    assert get_order_intent_by_id(intent.id).details["broker_native_protection_verified"] is False
    assert get_order_intent_by_id(intent.id).status == "accepted"
    assert latest_alpaca_bracket_observation(intent.id)["verified_at_observation"] is True
    assert list_trade_audits(1)[0].event_type == "alpaca_paper_bracket_readonly_reconciled"


@pytest.mark.parametrize("change,classification", [
    ("pending", "parent_pending_or_unknown_unverified"),
    ("partial", "partial_fill_unverified"),
    ("no_parent", "broker_parent_unknown_unverified"),
    ("no_position", "filled_parent_exits_or_position_unverified"),
    ("no_stop", "filled_parent_exits_or_position_unverified"),
    ("stop_pending_cancel", "filled_parent_exits_or_position_unverified"),
    ("stop_partially_filled", "filled_parent_exits_or_position_unverified"),
    ("rejected", "parent_terminal_unverified"),
    ("parent_lookup_error", "broker_parent_unknown_unverified"),
    ("position_lookup_error", "filled_parent_exits_or_position_unverified"),
])
def test_unconfirmed_or_ambiguous_broker_state_never_claims_protection(
    broker, change, classification
):
    state, _ = broker
    intent = _intent()
    if change == "pending":
        state["parent"]["status"] = "new"
        state["parent"]["filled_qty"] = "0"
    elif change == "partial":
        state["parent"]["status"] = "partially_filled"
        state["parent"]["filled_qty"] = "1"
        state["position"]["qty"] = "1"
    elif change == "no_parent":
        state["parent"] = None
    elif change == "no_position":
        state["position"] = None
    elif change == "no_stop":
        state["parent"]["legs"] = state["parent"]["legs"][:1]
    elif change == "stop_pending_cancel":
        state["parent"]["legs"][1]["status"] = "pending_cancel"
    elif change == "stop_partially_filled":
        state["parent"]["legs"][1]["filled_qty"] = "1"
    elif change == "rejected":
        state["parent"]["status"] = "rejected"
        state["parent"]["filled_qty"] = "0"
    elif change == "parent_lookup_error":
        state["error"] = "parent"
    elif change == "position_lookup_error":
        state["error"] = "position"
    result = _check(intent)
    assert result["classification"] == classification
    assert result["verified_at_observation"] is False
    assert result["current_protection_verified"] is False
    assert result["parent_order_id"] is None
    assert result["stop_order_id"] is None
    assert result["take_profit_order_id"] is None
    assert latest_alpaca_bracket_observation(intent.id)["verified_at_observation"] is False
    assert get_order_intent_by_id(intent.id).status == "accepted"


@pytest.mark.parametrize("field", ["id", "stop_id", "target_id"])
def test_verified_identity_change_after_restart_is_durably_flagged(
    broker, field
):
    state, _ = broker
    intent = _intent()
    first = _check(intent)
    assert first["verified_at_observation"]
    init_db()  # persisted ledger reloaded as on a service restart
    if field == "id":
        state["parent"]["id"] = "parent-B"
    elif field == "stop_id":
        state["parent"]["legs"][1]["id"] = "stop-B"
    else:
        state["parent"]["legs"][0]["id"] = "target-B"
    changed = _check(intent)
    assert not changed["verified_at_observation"]
    assert "identity changed" in changed["reason"]
    assert changed["classification"] == "filled_parent_exits_or_position_unverified"
    assert all(changed[field] is None for field in (
        "parent_order_id", "stop_order_id", "take_profit_order_id"
    ))
    baseline = first_verified_alpaca_bracket_observation(intent.id)
    assert baseline["id"] == first["observation_id"]
    assert latest_alpaca_bracket_observation(intent.id)["id"] == changed["observation_id"]
    # Repeated reads must not quietly adopt a new parent or stop/target ID.
    assert not _check(intent)["verified_at_observation"]


def test_protection_lost_after_successful_observation_is_not_masked_by_history(broker):
    state, _ = broker
    intent = _intent()
    assert _check(intent)["verified_at_observation"]
    init_db()
    state["parent"]["legs"][1]["status"] = "canceled"
    lost = _check(intent)
    assert not lost["verified_at_observation"]
    assert lost["stop_order_id"] is None
    assert lost["classification"] == "filled_parent_exits_or_position_unverified"
    assert not lost["current_protection_verified"]


@pytest.mark.parametrize("outcome", [
    "bracket_reserved_not_submitted",
    "unknown_synthetic_state",
    "",
])
def test_not_dispatched_or_unknown_claim_state_is_rejected_before_broker(
    broker, outcome
):
    state, calls = broker
    intent = _intent(state=outcome)
    with pytest.raises(ValueError, match="Only already-dispatched"):
        _check(intent)
    assert calls == []
    assert latest_alpaca_bracket_observation(intent.id) is None


@pytest.mark.parametrize("extra", [
    {"broker_environment": "live"},
    {"broker_provider": "ctrader"},
    {"asset_class": "crypto"},
    {"protection_mode": "simple"},
    {"client_order_id": ""},
])
def test_wrong_provenance_is_rejected_without_broker_reads(broker, extra):
    _, calls = broker
    intent = _intent(**extra)
    with pytest.raises(ValueError, match="Only already-dispatched"):
        _check(intent)
    assert calls == []


def test_non_paper_hostname_blocks_before_any_broker_lookups(broker, monkeypatch):
    _, calls = broker
    intent = _intent()
    monkeypatch.setattr(
        alpaca_bracket_reconcile, "_base_url",
        lambda: "https://api.alpaca.markets",
    )
    with pytest.raises(alpaca_paper.AlpacaPaperError, match="official HTTPS"):
        _check(intent)
    assert calls == []
    assert latest_alpaca_bracket_observation(intent.id) is None


def test_reconciliation_of_broker_rejection_remains_unknown_not_retryable(broker):
    state, calls = broker
    state["parent"] = None
    intent = _intent(state="broker_rejected_no_retry")
    result = _check(intent)
    assert result["classification"] == "broker_parent_unknown_unverified"
    assert result["automatic_retry"] is False
    assert calls == [("ta-ap-brkt-check", True)]
    assert get_order_intent_by_id(intent.id).details["outcome_state"] == "broker_rejected_no_retry"


def test_full_broker_truth_is_rechecked_every_call_not_trusted_from_old_snapshot(broker):
    _, calls = broker
    intent = _intent(state="ambiguous_post_submission_unknown")
    a = _check(intent)
    b = _check(intent)
    assert a["observation_id"] != b["observation_id"]
    assert len(calls) == 2
    assert all(x["current_protection_verified"] is False for x in (a,b))
    assert get_order_intent_by_id(intent.id).details["outcome_state"] == "ambiguous_post_submission_unknown"
