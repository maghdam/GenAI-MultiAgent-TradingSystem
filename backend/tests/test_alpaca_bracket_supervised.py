from __future__ import annotations

from copy import deepcopy

import pytest

from backend.domain.models import EngineConfig
from backend.services import (
    alpaca_bracket_reservation, alpaca_bracket_supervised, alpaca_paper,
)
from backend.storage.db import init_db
from backend.storage.repositories import (
    get_order_intent_by_id, latest_alpaca_bracket_observation,
    list_trade_audits,
)


def _account():
    return {"provider": "alpaca", "environment": "paper", "status": "ACTIVE",
            "trading_blocked": False, "account_blocked": False,
            "trade_suspended_by_user": False, "shorting_enabled": True}


def _parent(cid, *, status="new"):
    return {"id": "parent-123", "client_order_id": cid, "order_class": "bracket",
            "asset_class": "us_equity", "symbol": "SPY", "side": "buy",
            "status": status, "qty": "2", "filled_qty": "0", "legs": None}


def _reserve(monkeypatch):
    monkeypatch.setattr(alpaca_bracket_reservation, "_execution_ready_account", _account)
    monkeypatch.setattr(alpaca_bracket_reservation, "get_open_position", lambda symbol: None)
    monkeypatch.setattr(alpaca_bracket_reservation, "list_orders", lambda **kwargs: [])
    result = alpaca_bracket_reservation.reserve_paper_equity_bracket(
        symbol="SPY", side="buy", qty=2, reference_price=101,
        stop_loss=99, take_profit=104, timeframe="M5", strategy="manual",
        rationale="supervised-only-test",
    )
    return result


def _setup(monkeypatch):
    reserved = _reserve(monkeypatch)
    monkeypatch.setattr(
        alpaca_bracket_supervised, "load_engine_config",
        lambda defaults: EngineConfig(kill_switch=False),
    )
    monkeypatch.setattr(alpaca_bracket_supervised, "_execution_ready_account", _account)
    monkeypatch.setattr(alpaca_bracket_supervised, "get_open_position", lambda symbol: None)
    monkeypatch.setattr(alpaca_bracket_supervised, "list_orders", lambda **kwargs: [])
    monkeypatch.setattr(alpaca_bracket_supervised, "list_positions", lambda: [])
    monkeypatch.setattr(
        alpaca_bracket_supervised, "get_order_by_client_order_id",
        lambda cid, *, nested: None,
    )
    return reserved


def _args(reserved, **kw):
    return {
        "intent_id": reserved["intent_id"],
        "expected_client_order_id": reserved["client_order_id"],
        "operator_confirmation": f"SUBMIT PAPER BRACKET {reserved['intent_id']}",
        "operator_reason": "Operator supervised bracket experiment",
        **kw,
    }


@pytest.mark.parametrize("blocker", ["max_zero", "symbol_zero", "capacity", "same_symbol", "truncated"])
def test_dispatch_position_capacity_and_inventory_fail_closed(monkeypatch, blocker):
    reserved = _setup(monkeypatch)
    monkeypatch.setattr(alpaca_bracket_supervised, "submit_equity_bracket_order",
                        lambda **kw: pytest.fail("Risk-blocked POST"))
    if blocker in {"max_zero", "symbol_zero"}:
        config = EngineConfig(kill_switch=False, **{
            "max_open_positions" if blocker == "max_zero" else "max_positions_per_symbol": 0})
        monkeypatch.setattr(alpaca_bracket_supervised, "load_engine_config", lambda defaults: config)
    elif blocker == "capacity":
        monkeypatch.setattr(alpaca_bracket_supervised, "list_positions",
                            lambda: [{"symbol": symbol} for symbol in ("QQQ", "AAPL", "MSFT")])
    elif blocker == "same_symbol":
        monkeypatch.setattr(alpaca_bracket_supervised, "list_positions", lambda: [{"symbol": "SPY"}])
    else:
        monkeypatch.setattr(alpaca_bracket_supervised, "list_orders",
                            lambda **kw: [{"symbol": "QQQ"}] * 500)
    if blocker in {"max_zero", "symbol_zero", "truncated"}:
        with pytest.raises(alpaca_paper.AlpacaPaperError):
            alpaca_bracket_supervised.submit_supervised_paper_equity_bracket(**_args(reserved))
    else:
        assert alpaca_bracket_supervised.submit_supervised_paper_equity_bracket(**_args(reserved))["status"] == "blocked"
    assert get_order_intent_by_id(reserved["intent_id"]).details["outcome_state"] == "bracket_reserved_not_submitted"


def test_supervised_bracket_posts_exactly_once_and_records_unprotected_parent(monkeypatch):
    reserved = _setup(monkeypatch)
    transmitted = []
    def fake_post(**kwargs):
        transmitted.append(kwargs)
        return _parent(kwargs["client_order_id"])
    monkeypatch.setattr(alpaca_bracket_supervised, "submit_equity_bracket_order", fake_post)
    monkeypatch.setattr(
        alpaca_bracket_supervised, "observe_paper_equity_bracket",
        lambda *, intent_id: {"id": 11, "verified_at_observation": False},
    )
    outcome = alpaca_bracket_supervised.submit_supervised_paper_equity_bracket(**_args(reserved))
    assert len(transmitted) == 1
    assert transmitted[0]["client_order_id"] == reserved["client_order_id"]
    stored = get_order_intent_by_id(reserved["intent_id"]).details["bracket_order_request"]
    sent_payload = alpaca_paper.build_equity_bracket_market_order(**transmitted[0])
    assert sent_payload == stored  # exact durable request, including price strings
    assert outcome["status"] == "reconciliation_required"
    assert outcome["broker_native_protection_verified"] is False
    assert outcome["observed_legs_at_snapshot"] is False
    record = get_order_intent_by_id(reserved["intent_id"])
    assert record.status == "accepted"
    assert record.details["outcome_state"] == "parent_acknowledged_protection_unverified"
    assert record.details["reconciliation_resolved"] is False
    assert record.details["automatic_retry"] is False
    init_db()
    with pytest.raises(ValueError, match="unclaimed"):
        alpaca_bracket_supervised.submit_supervised_paper_equity_bracket(**_args(reserved))
    assert len(transmitted) == 1


def test_confirmation_and_exact_client_id_required_before_claim(monkeypatch):
    reserved = _setup(monkeypatch)
    monkeypatch.setattr(alpaca_bracket_supervised, "submit_equity_bracket_order",
                        lambda **kw: pytest.fail("unauthorized POST"))
    for invalid in [
        _args(reserved, operator_confirmation=""),
        _args(reserved, expected_client_order_id="wrong"),
        _args(reserved, operator_reason=""),
    ]:
        with pytest.raises(ValueError):
            alpaca_bracket_supervised.submit_supervised_paper_equity_bracket(**invalid)
    assert get_order_intent_by_id(reserved["intent_id"]).details["outcome_state"] == "bracket_reserved_not_submitted"


def test_kill_switch_rejects_new_entry_before_claim(monkeypatch):
    reserved = _setup(monkeypatch)
    monkeypatch.setattr(alpaca_bracket_supervised, "load_engine_config", lambda defaults: EngineConfig())
    monkeypatch.setattr(alpaca_bracket_supervised, "submit_equity_bracket_order",
                        lambda **kw: pytest.fail("kill switch POST"))
    with pytest.raises(alpaca_paper.AlpacaPaperError, match="Kill switch"):
        alpaca_bracket_supervised.submit_supervised_paper_equity_bracket(**_args(reserved))
    assert get_order_intent_by_id(reserved["intent_id"]).details["outcome_state"] == "bracket_reserved_not_submitted"


@pytest.mark.parametrize("blocker", ["position", "order", "broker_unavailable", "account_unavailable"])
def test_broker_readiness_or_existing_exposure_blocks_before_claim(monkeypatch, blocker):
    reserved = _setup(monkeypatch)
    monkeypatch.setattr(alpaca_bracket_supervised, "submit_equity_bracket_order",
                        lambda **kw: pytest.fail("blocked POST"))
    if blocker == "position":
        monkeypatch.setattr(alpaca_bracket_supervised, "get_open_position", lambda symbol: {"symbol":symbol})
    elif blocker == "order":
        monkeypatch.setattr(alpaca_bracket_supervised, "list_orders", lambda **kw: [{"symbol":"SPY"}])
    elif blocker == "broker_unavailable":
        def error(*args, **kwargs):
            raise alpaca_paper.AlpacaPaperError("broker position unavailable")
        monkeypatch.setattr(alpaca_bracket_supervised, "get_open_position", error)
    else:
        def error():
            raise alpaca_paper.AlpacaPaperError("Paper account unavailable")
        monkeypatch.setattr(alpaca_bracket_supervised, "_execution_ready_account", error)
    if blocker.endswith("unavailable"):
        with pytest.raises(alpaca_paper.AlpacaPaperError):
            alpaca_bracket_supervised.submit_supervised_paper_equity_bracket(**_args(reserved))
    else:
        assert alpaca_bracket_supervised.submit_supervised_paper_equity_bracket(**_args(reserved))["status"] == "blocked"
    assert get_order_intent_by_id(reserved["intent_id"]).details["outcome_state"] == "bracket_reserved_not_submitted"


@pytest.mark.parametrize("situation", ["ambiguous_absent", "ambiguous_lookup_error", "definite_reject"])
def test_post_failure_consumes_claim_no_resubmission_after_restart(monkeypatch, situation):
    reserved = _setup(monkeypatch)
    calls = []
    def post(**kwargs):
        calls.append(kwargs)
        if situation == "definite_reject":
            raise alpaca_paper.AlpacaPaperError("reject", status_code=403)
        raise alpaca_paper.AlpacaPaperError("timeout", ambiguous=True)
    monkeypatch.setattr(alpaca_bracket_supervised, "submit_equity_bracket_order", post)
    if situation == "ambiguous_lookup_error":
        hits = {"n":0}
        def lookup(cid, *, nested):
            hits["n"] += 1
            if hits["n"] > 1:
                raise alpaca_paper.AlpacaPaperError("lookup unavailable")
            return None
        monkeypatch.setattr(alpaca_bracket_supervised, "get_order_by_client_order_id", lookup)
    outcome = alpaca_bracket_supervised.submit_supervised_paper_equity_bracket(**_args(reserved))
    assert outcome["status"] == "reconciliation_required"
    assert outcome["automatic_retry"] is False
    assert get_order_intent_by_id(reserved["intent_id"]).status == "accepted"
    init_db()
    with pytest.raises(ValueError, match="unclaimed"):
        alpaca_bracket_supervised.submit_supervised_paper_equity_bracket(**_args(reserved))
    assert len(calls) == 1


def test_pre_post_existing_exact_client_id_forbids_post_and_marks_unresolved(monkeypatch):
    reserved = _setup(monkeypatch)
    monkeypatch.setattr(
        alpaca_bracket_supervised, "get_order_by_client_order_id",
        lambda cid, *, nested: _parent(cid),
    )
    monkeypatch.setattr(alpaca_bracket_supervised, "submit_equity_bracket_order",
                        lambda **kw: pytest.fail("existing ID POST"))
    outcome = alpaca_bracket_supervised.submit_supervised_paper_equity_bracket(**_args(reserved))
    assert outcome["outcome_state"] == "existing_broker_order_reconciliation_required"
    assert not outcome["broker_native_protection_verified"]
    assert get_order_intent_by_id(reserved["intent_id"]).status == "accepted"


def test_low_level_exact_native_bracket_payload_and_paper_only_hostname(monkeypatch):
    sent = []
    monkeypatch.setattr(alpaca_paper, "_base_url", lambda: "https://paper-api.alpaca.markets")
    def mutation(method, path, *, body, **kwargs):
        sent.append((method, path, body))
        return {
            "id":"broker-parent", "client_order_id":body["client_order_id"],
            "order_class":"bracket", "asset_class":"us_equity",
            "symbol":body["symbol"], "side":body["side"], "status":"new",
        }
    monkeypatch.setattr(alpaca_paper, "_request_mutation_json", mutation)
    kwargs = dict(symbol="SPY",side="buy",qty=2,reference_price=101,
                  stop_loss=99,take_profit=104,client_order_id="ta-test-01")
    normalized = alpaca_paper.submit_equity_bracket_order(**kwargs)
    assert normalized["id"] == "broker-parent"
    assert len(sent) == 1
    assert sent[0][0:2] == ("POST", "/v2/orders")
    assert sent[0][2]["order_class"] == "bracket"
    assert sent[0][2]["stop_loss"] == {"stop_price": "99"}
    assert sent[0][2]["take_profit"] == {"limit_price": "104"}
    assert sent[0][2]["time_in_force"] == "gtc"
    monkeypatch.setattr(alpaca_paper, "_base_url", lambda: "https://api.alpaca.markets")
    with pytest.raises(alpaca_paper.AlpacaPaperError, match="Paper endpoint"):
        alpaca_paper.submit_equity_bracket_order(**kwargs)
    assert len(sent) == 1


@pytest.mark.parametrize("malformed", [
    {"id": None}, {"client_order_id":"wrong"}, {"order_class":"simple"},
    {"asset_class":"crypto"}, {"symbol":"QQQ"}, {"side":"sell"},
])
def test_low_level_mismatched_broker_ack_is_ambiguous_without_retry(monkeypatch, malformed):
    monkeypatch.setattr(alpaca_paper, "_base_url", lambda: "https://paper-api.alpaca.markets")
    def mutation(method, path, *, body, **kwargs):
        return {**_parent(body["client_order_id"]),**malformed}
    monkeypatch.setattr(alpaca_paper, "_request_mutation_json", mutation)
    with pytest.raises(alpaca_paper.AlpacaPaperError) as exc:
        alpaca_paper.submit_equity_bracket_order(
            symbol="SPY", side="buy", qty=2, reference_price=101,
            stop_loss=99, take_profit=104, client_order_id="ta-test",
        )
    assert exc.value.ambiguous


def test_no_automatic_runtime_entry_or_manual_simple_reconciler_claim(monkeypatch):
    # The new module has no API route and cannot override runtime lockout.
    from backend.services import alpaca_runtime
    assert "submit_supervised_paper_equity_bracket" not in (
        alpaca_runtime.execute_alpaca_runtime_signal.__code__.co_names
    )


@pytest.mark.parametrize("numeric_inputs", [
    {"reference_price": 101, "stop_loss": 99, "take_profit": 104, "qty": 2},
    {"reference_price": 101.0, "stop_loss": 99.0, "take_profit": 104.0, "qty": 2},
])
def test_valid_reserved_numeric_representation_survives_sqlite_roundtrip(
    monkeypatch, numeric_inputs
):
    monkeypatch.setattr(alpaca_bracket_reservation, "_execution_ready_account", _account)
    monkeypatch.setattr(alpaca_bracket_reservation, "get_open_position", lambda symbol: None)
    monkeypatch.setattr(alpaca_bracket_reservation, "list_orders", lambda **kwargs: [])
    reserved = alpaca_bracket_reservation.reserve_paper_equity_bracket(
        symbol="SPY", side="buy", timeframe="M5", strategy="manual",
        rationale="serialized-order-numeric-equivalence", **numeric_inputs,
    )
    init_db()  # simulate roundtrip / restart between reserve and dispatch
    monkeypatch.setattr(
        alpaca_bracket_supervised, "load_engine_config",
        lambda defaults: EngineConfig(kill_switch=False),
    )
    monkeypatch.setattr(alpaca_bracket_supervised, "_execution_ready_account", _account)
    monkeypatch.setattr(alpaca_bracket_supervised, "get_open_position", lambda symbol: None)
    monkeypatch.setattr(alpaca_bracket_supervised, "list_orders", lambda **kwargs: [])
    monkeypatch.setattr(
        alpaca_bracket_supervised, "get_order_by_client_order_id",
        lambda cid, *, nested: None,
    )
    transmitted = []
    def fake_submit(**kwargs):
        transmitted.append(alpaca_paper.build_equity_bracket_market_order(**kwargs))
        return _parent(kwargs["client_order_id"])
    monkeypatch.setattr(alpaca_bracket_supervised, "submit_equity_bracket_order", fake_submit)
    monkeypatch.setattr(
        alpaca_bracket_supervised, "observe_paper_equity_bracket",
        lambda *, intent_id: {"id": 31, "verified_at_observation": False},
    )
    result = alpaca_bracket_supervised.submit_supervised_paper_equity_bracket(
        **_args(reserved)
    )
    assert result["status"] == "reconciliation_required"
    assert len(transmitted) == 1
    assert transmitted[0] == get_order_intent_by_id(
        reserved["intent_id"]
    ).details["bracket_order_request"]


@pytest.mark.parametrize("tamper", [
    "stop_loss",
    "quantity",
    "client_order_id",
    "extra_field",
])
def test_mutated_persisted_request_fails_closed_before_dispatch_claim(
    monkeypatch, tamper
):
    from backend.storage.repositories import update_order_intent_status
    reserved = _setup(monkeypatch)
    intent = get_order_intent_by_id(reserved["intent_id"])
    request = deepcopy(intent.details["bracket_order_request"])
    if tamper == "stop_loss":
        request["stop_loss"]["stop_price"] = "98"
    elif tamper == "quantity":
        request["qty"] = "3"
    elif tamper == "client_order_id":
        request["client_order_id"] = "altered-id"
    else:
        request["unexpected"] = "forbidden"
    update_order_intent_status(
        reserved["intent_id"], "accepted", {"bracket_order_request": request},
        reason="test_tampered_prospective_bracket",
    )
    monkeypatch.setattr(
        alpaca_bracket_supervised, "submit_equity_bracket_order",
        lambda **kw: pytest.fail("tampered request must never POST"),
    )
    with pytest.raises(ValueError, match="Durable bracket request differs"):
        alpaca_bracket_supervised.submit_supervised_paper_equity_bracket(
            **_args(reserved)
        )
    observed = get_order_intent_by_id(reserved["intent_id"])
    assert observed.details["outcome_state"] == "bracket_reserved_not_submitted"
