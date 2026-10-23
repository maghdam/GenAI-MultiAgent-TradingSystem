from copy import deepcopy

import pytest

from backend.services import alpaca_bracket_exit as service, alpaca_paper as broker
from backend.storage.alpaca_bracket_exits import exit_actions
from backend.storage.db import init_db
from backend.tests.test_alpaca_bracket_reconcile import _intent, _parent, _position


@pytest.fixture
def scenario(monkeypatch):
    intent = _intent()
    state = {"parent": _parent(), "position": _position(), "orders": [], "close": None,
             "account_id": "paper-account", "calls": []}
    monkeypatch.setattr(broker, "_base_url", lambda: "https://paper-api.alpaca.markets")
    monkeypatch.setattr(broker, "get_account", lambda: {
        "environment": "paper", "account_id": state["account_id"], "status": "ACTIVE"})
    monkeypatch.setattr(broker, "get_order_by_client_order_id",
                        lambda cid, **kw: deepcopy(state["parent"] if cid == "ta-ap-brkt-check" else state["close"]))
    monkeypatch.setattr(broker, "get_open_position", lambda symbol: deepcopy(state["position"]))
    monkeypatch.setattr(broker, "list_orders", lambda **kw: deepcopy(state["orders"]))
    monkeypatch.setattr(broker, "cancel_paper_bracket_order", lambda oid: state["calls"].append(("cancel", oid)))
    monkeypatch.setattr(broker, "submit_paper_bracket_close", lambda **kw: state["calls"].append(("close", kw)))
    monkeypatch.setattr(broker, "_request_mutation_json", lambda *a, **kw: pytest.fail("Real mutation prohibited"))
    return intent, state


def advance(intent):
    return service.advance_supervised_paper_bracket_exit(
        intent_id=intent.id, operator_confirmation=f"CANCEL AND FLATTEN PAPER BRACKET {intent.id}",
        operator_reason="Supervised mocked exit with interruption acknowledged")


def canceled(state):
    for leg in state["parent"]["legs"]:
        leg["status"] = "canceled"


def test_cancel_ack_wait_terminal_then_one_close_and_recover(scenario):
    intent, s = scenario
    assert advance(intent)["status"] == "acknowledged_reconciliation_required"
    assert s["calls"] == [("cancel", "target-A")]
    init_db()
    assert advance(intent)["status"] == "cancel_reconciliation_required"
    canceled(s)
    assert advance(intent)["status"] == "acknowledged_reconciliation_required"
    request = s["calls"][-1][1]
    assert request["qty"] == "2" and request["side"] == "sell"
    assert advance(intent)["status"] == "close_reconciliation_required"
    s["close"] = {**request, "id": "close-A", "type": "market", "asset_class": "us_equity",
                  "status": "partially_filled", "filled_qty": "1"}
    s["position"]["qty"] = "1"
    assert service.inspect_paper_bracket_exit(intent_id=intent.id)["assessment"]["remaining_quantity"] == "1"
    assert advance(intent)["status"] == "close_reconciliation_required"
    s["close"].update(status="filled", filled_qty="2")
    s["position"] = None
    assert advance(intent)["status"] == "flat_terminal_at_observation"
    assert len(s["calls"]) == 2


@pytest.mark.parametrize("action", ["cancel", "close"])
def test_lost_ack_and_restart_never_repeat(scenario, monkeypatch, action):
    intent, s = scenario
    if action == "close":
        canceled(s)
    def timeout(*args, **kwargs):
        s["calls"].append(action)
        raise broker.AlpacaPaperError("lost acknowledgement", ambiguous=True)
    monkeypatch.setattr(broker, "cancel_paper_bracket_order" if action == "cancel" else "submit_paper_bracket_close", timeout)
    assert advance(intent)["status"] == "blocked"
    init_db()
    advance(intent)
    assert s["calls"] == [action]
    assert exit_actions(intent.id)


def test_claim_committed_before_crash(scenario, monkeypatch):
    intent, s = scenario
    canceled(s)
    def crash(**kw):
        assert "close" in exit_actions(intent.id)
        raise SystemExit("process died")
    monkeypatch.setattr(broker, "submit_paper_bracket_close", crash)
    with pytest.raises(SystemExit):
        advance(intent)
    init_db()
    assert advance(intent)["status"] == "close_reconciliation_required"


@pytest.mark.parametrize("failure", ["live", "account", "replacement", "missing", "unknown",
                                    "foreign", "truncated", "quantity", "reversed", "oco_overfill"])
def test_untrusted_truth_blocks_all_mutations(scenario, monkeypatch, failure):
    intent, s = scenario
    if failure == "live":
        monkeypatch.setattr(broker, "_base_url", lambda: "https://api.alpaca.markets")
    elif failure == "account":
        s["account_id"] = ""
    elif failure == "replacement":
        s["parent"]["legs"][1]["replaced_by"] = "new-stop"
    elif failure == "missing":
        s["parent"]["legs"].pop()
    elif failure == "unknown":
        s["parent"]["legs"][0]["status"] = "mystery"
    elif failure == "foreign":
        s["orders"] = [{"id": "external", "symbol": "SPY"}]
    elif failure == "truncated":
        s["orders"] = [{"id": "other", "symbol": "QQQ"}] * 500
    elif failure == "quantity":
        s["position"]["qty"] = "1"
    elif failure == "reversed":
        s["position"].update(side="short", qty="-2")
    else:
        for leg in s["parent"]["legs"]:
            leg.update(status="filled", filled_qty="2")
    assert advance(intent)["status"] == "blocked"
    assert s["calls"] == []


def test_account_and_identity_pinned_across_restart(scenario):
    intent, s = scenario
    service.inspect_paper_bracket_exit(intent_id=intent.id)
    init_db()
    s["account_id"] = "another-account"
    assert advance(intent)["status"] == "blocked"
    s["account_id"] = "paper-account"
    s["parent"]["legs"][1]["id"] = "different-stop"
    assert advance(intent)["status"] == "blocked"
    assert not s["calls"]


def test_flat_orphan_cancel_and_oco_race_never_extra_close(scenario):
    intent, s = scenario
    s["parent"]["legs"][0].update(status="filled", filled_qty="2")
    s["position"] = None
    advance(intent)
    assert s["calls"] == [("cancel", "stop-A")]
    s["parent"]["legs"][1]["status"] = "canceled"
    assert advance(intent)["status"] == "flat_terminal_at_observation"
    assert len(s["calls"]) == 1


def test_partial_entry_cancel_then_close_actual_remaining(scenario):
    intent, s = scenario
    s["parent"].update(status="partially_filled", filled_qty="1")
    s["position"]["qty"] = "1"
    advance(intent)
    assert s["calls"] == [("cancel", "parent-A")]
    s["parent"]["status"] = "canceled"
    canceled(s)
    advance(intent)
    assert s["calls"][-1][1]["qty"] == "1"


def test_pending_cancel_and_cancel_fill_race(scenario):
    intent, s = scenario
    s["parent"]["legs"][0]["status"] = "pending_cancel"
    assert advance(intent)["status"] == "cancel_reconciliation_required"
    assert not s["calls"]
    s["parent"]["legs"][0].update(status="filled", filled_qty="2")
    s["parent"]["legs"][1]["status"] = "canceled"
    s["position"] = None
    assert advance(intent)["status"] == "flat_terminal_at_observation"
    assert not s["calls"]


def test_confirmation_required_and_inspection_never_mutates(scenario):
    intent, s = scenario
    with pytest.raises(ValueError):
        service.advance_supervised_paper_bracket_exit(intent_id=intent.id,
                                                     operator_confirmation="yes", operator_reason="test reason")
    assert service.inspect_paper_bracket_exit(intent_id=intent.id)["status"] == "reconciliation_required"
    assert not s["calls"]


def test_activation_gate_cannot_be_enabled_by_environment(monkeypatch):
    from backend.services.alpaca_bracket_readiness import paper_bracket_automation_readiness
    monkeypatch.setenv("ALPACA_AUTOTRADE", "true")
    result = paper_bracket_automation_readiness()
    assert not result["automatic_trading_allowed"]
    assert not result["activation_available"]
    assert result["blockers"]


@pytest.mark.parametrize("status", [204, 500])
def test_cancel_transport_204_and_server_ambiguity(monkeypatch, status):
    import httpx
    class Client:
        def __init__(self, **kw):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def request(self, method, path, **kw):
            assert method == "DELETE" and path == "/v2/orders/order-A"
            return httpx.Response(status)
    monkeypatch.setattr(broker, "_base_url", lambda: "https://paper-api.alpaca.markets")
    monkeypatch.setattr(broker, "_credentials", lambda: ("mock", "mock"))
    monkeypatch.setattr(broker.httpx, "Client", Client)
    if status == 204:
        assert broker.cancel_paper_bracket_order("order-A") is None
    else:
        with pytest.raises(broker.AlpacaPaperError) as error:
            broker.cancel_paper_bracket_order("order-A")
        assert error.value.ambiguous


def test_two_workers_claim_same_action_once(scenario):
    from concurrent.futures import ThreadPoolExecutor
    from backend.storage.alpaca_bracket_exits import claim_exit_action
    from backend.storage import db
    intent, _ = scenario
    service.inspect_paper_bracket_exit(intent_id=intent.id)
    def claim(_):
        try:
            return claim_exit_action(intent.id, "close", {"qty": "2"})
        finally:
            conn = getattr(db._LOCAL, "connection", None)
            if conn is not None:
                conn.close()
                db._LOCAL.connection = None
    with ThreadPoolExecutor(max_workers=2) as workers:
        assert sorted(workers.map(claim, range(2))) == [False, True]


@pytest.mark.parametrize("field,value", [("id", "changed"), ("qty", "3"), ("side", "buy"),
                                        ("filled_qty", "NaN"), ("status", "replaced")])
def test_recovered_close_identity_and_quantities_cannot_drift(scenario, field, value):
    intent, s = scenario
    canceled(s)
    advance(intent)
    request = s["calls"][-1][1]
    s["close"] = {**request, "id": "close-A", "type": "market", "asset_class": "us_equity",
                  "status": "new", "filled_qty": "0"}
    assert service.inspect_paper_bracket_exit(intent_id=intent.id)["status"] == "reconciliation_required"
    init_db()
    s["close"][field] = value
    assert advance(intent)["status"] == "blocked"
    assert len(s["calls"]) == 1


def test_old_bracket_blocks_legacy_close_beyond_recent_intents(scenario):
    from backend.services.alpaca_execution import _unresolved_intent_for_symbol
    from backend.storage.db import get_db
    intent, _ = scenario
    with get_db() as db:
        # Fill the UI page with unrelated intents without erasing the old claim.
        db.executemany(
            "INSERT INTO order_intents (created_at, symbol, timeframe, strategy, direction, "
            "intent_type, status, confidence, rationale, details_json) "
            "VALUES ('2026-10-21', 'QQQ', 'M5', 'manual', 'long', 'open', 'failed', 1, '', '{}')",
            [() for _ in range(205)],
        )
        db.commit()
    assert _unresolved_intent_for_symbol("spy").id == intent.id
