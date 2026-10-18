from __future__ import annotations

import pytest

from backend.services import alpaca_bracket_reservation, alpaca_paper
from backend.storage.db import init_db
from backend.storage.repositories import (
    claim_reserved_alpaca_bracket_dispatch,
    create_order_intent,
    get_order_intent_by_id,
    list_order_intents,
    list_trade_audits,
)


def _ready_account():
    return {"provider": "alpaca", "environment": "paper", "status": "ACTIVE",
            "trading_blocked": False, "account_blocked": False,
            "trade_suspended_by_user": False, "shorting_enabled": True}


def _request(**kw):
    args = dict(
        symbol="SPY", side="buy", qty=2, reference_price=101.0,
        stop_loss=99.0, take_profit=104.0, timeframe="M5",
        strategy="sma_cross", rationale="read-only reservation test",
    )
    args.update(kw)
    return args


@pytest.fixture
def broker_truth(monkeypatch):
    monkeypatch.setattr(alpaca_bracket_reservation, "_execution_ready_account", _ready_account)
    monkeypatch.setattr(alpaca_bracket_reservation, "get_open_position", lambda symbol: None)
    monkeypatch.setattr(alpaca_bracket_reservation, "list_orders", lambda **kw: [])
    monkeypatch.setattr(
        alpaca_paper, "_request_mutation_json",
        lambda *a, **kw: pytest.fail("reservation and claim cannot call broker mutations"),
    )


def test_atomic_reserved_client_identity_and_request_durable_without_post(broker_truth):
    result = alpaca_bracket_reservation.reserve_paper_equity_bracket(**_request())
    assert result["status"] == "reserved_not_submitted"
    assert result["broker_mutation_permitted"] is False
    client_id = result["client_order_id"]
    assert client_id.startswith("ta-ap-brkt-") and len(client_id) <= 48
    stored = get_order_intent_by_id(result["intent_id"])
    assert stored.status == "accepted"
    assert stored.details["broker_provider"] == "alpaca"
    assert stored.details["broker_environment"] == "paper"
    assert stored.details["protection_mode"] == "equity_bracket"
    assert stored.details["outcome_state"] == "bracket_reserved_not_submitted"
    assert stored.details["client_order_id"] == client_id
    assert stored.details["broker_mutation_permitted"] is False
    assert stored.details["bracket_order_request"]["order_class"] == "bracket"
    assert stored.details["bracket_order_request"]["client_order_id"] == client_id
    assert stored.details["bracket_order_request"]["take_profit"]["limit_price"] == "104.0"
    assert stored.details["bracket_order_request"]["stop_loss"]["stop_price"] == "99.0"
    assert list_trade_audits(1)[0].event_type == "alpaca_paper_bracket_reserved"


def test_reservation_fails_closed_on_same_symbol_but_allows_other_symbol(broker_truth):
    one = alpaca_bracket_reservation.reserve_paper_equity_bracket(**_request())
    duplicate = alpaca_bracket_reservation.reserve_paper_equity_bracket(**_request())
    other = alpaca_bracket_reservation.reserve_paper_equity_bracket(**_request(symbol="QQQ"))
    assert one["status"] == "reserved_not_submitted"
    assert duplicate["status"] == "blocked"
    assert duplicate["intent_id"] is None
    assert other["status"] == "reserved_not_submitted"
    assert other["client_order_id"] != one["client_order_id"]
    assert len(list_order_intents(10)) == 2


@pytest.mark.parametrize("override", [
    {"symbol": "BTC/USD"},
    {"qty": 0.5},
    {"qty": 0},
    {"stop_loss": 101.0},
    {"take_profit": 100.0},
    {"stop_loss": float("nan")},
    {"take_profit": float("inf")},
])
def test_rejected_protected_brackets_create_no_durable_intent(broker_truth, override):
    with pytest.raises(ValueError):
        alpaca_bracket_reservation.reserve_paper_equity_bracket(**_request(**override))
    assert not list_order_intents(10)


def test_broker_position_and_open_order_block_without_durable_reservation(monkeypatch, broker_truth):
    monkeypatch.setattr(alpaca_bracket_reservation, "get_open_position", lambda symbol: {"symbol": symbol})
    assert alpaca_bracket_reservation.reserve_paper_equity_bracket(**_request())["status"] == "blocked"
    monkeypatch.setattr(alpaca_bracket_reservation, "get_open_position", lambda symbol: None)
    monkeypatch.setattr(alpaca_bracket_reservation, "list_orders", lambda **kw: [{"symbol": "SPY"}])
    assert alpaca_bracket_reservation.reserve_paper_equity_bracket(**_request())["status"] == "blocked"
    assert not list_order_intents(10)


def test_not_ready_paper_or_unavailable_broker_truth_never_reserves(monkeypatch, broker_truth):
    monkeypatch.setattr(
        alpaca_bracket_reservation, "_execution_ready_account",
        lambda: (_ for _ in ()).throw(alpaca_paper.AlpacaPaperError("Paper account unavailable")),
    )
    with pytest.raises(alpaca_paper.AlpacaPaperError):
        alpaca_bracket_reservation.reserve_paper_equity_bracket(**_request())
    monkeypatch.setattr(alpaca_bracket_reservation, "_execution_ready_account", _ready_account)
    monkeypatch.setattr(
        alpaca_bracket_reservation, "list_orders",
        lambda **kw: (_ for _ in ()).throw(alpaca_paper.AlpacaPaperError("broker orders unavailable")),
    )
    with pytest.raises(alpaca_paper.AlpacaPaperError):
        alpaca_bracket_reservation.reserve_paper_equity_bracket(**_request())
    assert not list_order_intents(10)


def test_short_bracket_requires_broker_shorting_permission(monkeypatch, broker_truth):
    monkeypatch.setattr(
        alpaca_bracket_reservation, "_execution_ready_account",
        lambda: {**_ready_account(), "shorting_enabled": False},
    )
    with pytest.raises(alpaca_paper.AlpacaPaperError, match="short"):
        alpaca_bracket_reservation.reserve_paper_equity_bracket(**_request(side="sell",stop_loss=104,take_profit=99))
    assert not list_order_intents(10)


def test_one_shot_dispatch_claim_stays_durable_across_restart_and_never_submits(broker_truth):
    reserved = alpaca_bracket_reservation.reserve_paper_equity_bracket(**_request())
    key = reserved["intent_id"]
    claimed = claim_reserved_alpaca_bracket_dispatch(key)
    assert claimed is not None
    assert claimed.status == "accepted"
    assert claimed.details["outcome_state"] == "bracket_dispatch_claimed_submission_unknown"
    assert claimed.details["automatic_retry"] is False
    assert claimed.details["broker_mutation_permitted"] is False
    init_db()
    assert claim_reserved_alpaca_bracket_dispatch(key) is None
    assert get_order_intent_by_id(key).details["client_order_id"] == reserved["client_order_id"]
    assert list_trade_audits(1)[0].event_type == "alpaca_paper_bracket_dispatch_claimed"


def test_claim_wrong_provenance_or_simple_order_is_denied(broker_truth):
    old = create_order_intent(
        symbol="AAPL", timeframe="D1", strategy="manual", direction="long",
        intent_type="open", status="accepted", confidence=1,
        entry_price=10, stop_loss=9, take_profit=11, quantity=1,
        rationale="legacy", details={
            "broker_provider":"alpaca", "broker_environment":"paper",
            "asset_class":"us_equity", "protection_mode":"simple",
            "outcome_state":"bracket_reserved_not_submitted",
            "client_order_id":"legacy-test", "bracket_order_request":{"fake":True},
        },
    )
    assert claim_reserved_alpaca_bracket_dispatch(old.id) is None
    assert get_order_intent_by_id(old.id).details["outcome_state"] == "bracket_reserved_not_submitted"


def test_existing_nonterminal_simple_paper_intent_blocks_reservation(broker_truth):
    existing = create_order_intent(
        symbol="SPY", timeframe="D1", strategy="manual", direction="long",
        intent_type="open", status="accepted", confidence=1.0,
        entry_price=10, stop_loss=9, take_profit=11, quantity=1,
        rationale="pending simple",
        details={"broker_provider":"alpaca","broker_environment":"paper",
                 "asset_class":"us_equity","outcome_state":"broker_new",
                 "reconciliation_resolved":False},
    )
    assert alpaca_bracket_reservation.reserve_paper_equity_bracket(**_request())["status"] == "blocked"
    assert get_order_intent_by_id(existing.id).status == "accepted"


def test_reservation_is_not_an_automatic_entry_or_verified_protection(broker_truth):
    result = alpaca_bracket_reservation.reserve_paper_equity_bracket(**_request())
    intent = get_order_intent_by_id(result["intent_id"])
    assert intent.details["outcome_state"] == "bracket_reserved_not_submitted"
    assert intent.details["reconciliation_resolved"] is False
    assert "broker_order" not in intent.details
    assert "broker_native_protection_verified" not in intent.details
    assert result["broker_mutation_permitted"] is False
