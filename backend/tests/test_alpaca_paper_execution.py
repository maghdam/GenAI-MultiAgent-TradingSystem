from __future__ import annotations

import importlib

from fastapi.testclient import TestClient

from backend.services import alpaca_execution, alpaca_paper
from backend.storage.repositories import (
    list_order_intents,
    list_trade_audits,
)


def _ready_account() -> dict[str, object]:
    return {
        "provider": "alpaca",
        "environment": "paper",
        "status": "ACTIVE",
        "trading_blocked": False,
        "account_blocked": False,
        "trade_suspended_by_user": False,
    }


def _order(
    *,
    status: str = "filled",
    client_order_id: str = "tradeagent-alpaca-test",
    order_id: str = "order-1",
    symbol: str = "AAPL",
    side: str = "buy",
) -> dict[str, object]:
    return {
        "id": order_id,
        "client_order_id": client_order_id,
        "symbol": symbol,
        "asset_class": "us_equity",
        "side": side,
        "type": "market",
        "time_in_force": "day",
        "qty": "0.01",
        "notional": None,
        "status": status,
        "filled_qty": "0.01" if status == "filled" else "0",
        "filled_avg_price": "250" if status == "filled" else None,
        "submitted_at": "2026-10-16T12:00:00Z",
        "filled_at": "2026-10-16T12:00:01Z" if status == "filled" else None,
        "canceled_at": None,
        "expired_at": None,
        "failed_at": None,
    }


def _position(symbol: str = "AAPL") -> dict[str, object]:
    return {
        "asset_id": "asset-aapl",
        "symbol": symbol,
        "asset_class": "us_equity",
        "exchange": "NASDAQ",
        "side": "long",
        "qty": "1",
        "qty_available": "1",
        "avg_entry_price": "200",
        "current_price": "250",
        "market_value": "250",
        "cost_basis": "200",
        "unrealized_pl": "50",
        "unrealized_plpc": "0.25",
    }


def test_low_level_market_order_uses_explicit_client_id_and_equity_day_tif(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _mutate(method, path, *, params=None, body=None):
        captured.update({"method": method, "path": path, "params": params, "body": body})
        return {
            "id": "order-1",
            "client_order_id": body["client_order_id"],
            "symbol": body["symbol"],
            "asset_class": "us_equity",
            "side": body["side"],
            "type": body["type"],
            "time_in_force": body["time_in_force"],
            "qty": body.get("qty"),
            "notional": body.get("notional"),
            "status": "new",
            "filled_qty": "0",
        }

    monkeypatch.setattr(alpaca_paper, "_request_mutation_json", _mutate)

    order = alpaca_paper.submit_market_order(
        symbol="AAPL",
        asset_class="us_equity",
        side="buy",
        client_order_id="tradeagent-alpaca-123",
        qty=0.01,
    )

    assert captured["method"] == "POST"
    assert captured["path"] == "/v2/orders"
    assert captured["body"] == {
        "symbol": "AAPL",
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
        "client_order_id": "tradeagent-alpaca-123",
        "qty": "0.01",
    }
    assert order["client_order_id"] == "tradeagent-alpaca-123"
    assert order["status"] == "new"


def test_low_level_crypto_market_order_uses_gtc_and_notional(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _mutate(method, path, *, params=None, body=None):
        captured["body"] = body
        return {
            "id": "crypto-order",
            "client_order_id": body["client_order_id"],
            "symbol": body["symbol"],
            "asset_class": "crypto",
            "side": body["side"],
            "type": body["type"],
            "time_in_force": body["time_in_force"],
            "qty": None,
            "notional": body.get("notional"),
            "status": "new",
            "filled_qty": "0",
        }

    monkeypatch.setattr(alpaca_paper, "_request_mutation_json", _mutate)

    alpaca_paper.submit_market_order(
        symbol="BTC/USD",
        asset_class="crypto",
        side="buy",
        client_order_id="tradeagent-alpaca-crypto",
        notional=5.0,
    )

    assert captured["body"]["time_in_force"] == "gtc"
    assert captured["body"]["notional"] == "5.0"
    assert "qty" not in captured["body"]


def test_low_level_close_encodes_symbol_and_requests_full_flatten(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _mutate(method, path, *, params=None, body=None):
        captured.update({"method": method, "path": path, "params": params})
        return {
            "id": "close-order",
            "client_order_id": "broker-close-id",
            "symbol": "BTC/USD",
            "asset_class": "crypto",
            "side": "sell",
            "type": "market",
            "time_in_force": "gtc",
            "qty": "0.01",
            "notional": None,
            "status": "new",
            "filled_qty": "0",
        }

    monkeypatch.setattr(alpaca_paper, "_request_mutation_json", _mutate)

    order = alpaca_paper.close_position("BTC/USD")

    assert captured == {
        "method": "DELETE",
        "path": "/v2/positions/BTC%2FUSD",
        "params": {"percentage": 100},
    }
    assert order["id"] == "close-order"


def test_filled_market_order_is_durably_identified_and_audited(monkeypatch) -> None:
    monkeypatch.setattr(alpaca_execution, "get_account", _ready_account)
    monkeypatch.setattr(alpaca_execution, "get_open_position", lambda _symbol: None)

    captured: dict[str, object] = {}

    def _submit(**kwargs):
        captured.update(kwargs)
        return _order(client_order_id=str(kwargs["client_order_id"]))

    monkeypatch.setattr(alpaca_execution, "submit_market_order", _submit)

    result = alpaca_execution.submit_paper_market_order(
        symbol="AAPL",
        asset_class="us_equity",
        side="buy",
        qty=0.01,
        notional=None,
        timeframe="D1",
        strategy="manual",
        rationale="field-safe paper test",
    )

    assert result["status"] == "executed"
    assert result["action_taken"] is True
    assert str(captured["client_order_id"]).startswith("tradeagent-alpaca-")

    intent = list_order_intents(1)[0]
    assert intent.status == "executed"
    assert intent.details["broker_provider"] == "alpaca"
    assert intent.details["broker_environment"] == "paper"
    assert intent.details["client_order_id"] == captured["client_order_id"]
    assert intent.details["broker_order"]["status"] == "filled"

    events = {item.event_type for item in list_trade_audits(10)}
    assert "alpaca_paper_order_reserved" in events
    assert "alpaca_paper_order_filled" in events


def test_existing_position_blocks_entry_only_market_mutation(monkeypatch) -> None:
    monkeypatch.setattr(alpaca_execution, "get_account", _ready_account)
    monkeypatch.setattr(alpaca_execution, "get_open_position", lambda _symbol: _position())

    def _must_not_submit(**kwargs):
        raise AssertionError("Broker submission must be blocked when position already exists")

    monkeypatch.setattr(alpaca_execution, "submit_market_order", _must_not_submit)

    result = alpaca_execution.submit_paper_market_order(
        symbol="AAPL",
        asset_class="us_equity",
        side="buy",
        qty=0.01,
        notional=None,
        timeframe="D1",
        strategy="manual",
        rationale="entry-only guard",
    )

    assert result["status"] == "blocked"
    assert result["intent_id"] is None
    assert result["position"]["symbol"] == "AAPL"
    assert list_order_intents(10) == []


def test_nonterminal_order_blocks_duplicate_symbol_submission(monkeypatch) -> None:
    monkeypatch.setattr(alpaca_execution, "get_account", _ready_account)
    monkeypatch.setattr(alpaca_execution, "get_open_position", lambda _symbol: None)
    calls = {"submit": 0}

    def _submit(**kwargs):
        calls["submit"] += 1
        return _order(
            status="new",
            client_order_id=str(kwargs["client_order_id"]),
        )

    monkeypatch.setattr(alpaca_execution, "submit_market_order", _submit)

    first = alpaca_execution.submit_paper_market_order(
        symbol="AAPL",
        asset_class="us_equity",
        side="buy",
        qty=0.01,
        notional=None,
        timeframe="D1",
        strategy="manual",
        rationale="first",
    )
    second = alpaca_execution.submit_paper_market_order(
        symbol="AAPL",
        asset_class="us_equity",
        side="buy",
        qty=0.01,
        notional=None,
        timeframe="D1",
        strategy="manual",
        rationale="duplicate",
    )

    assert first["status"] == "accepted"
    assert second["status"] == "blocked"
    assert second["intent_id"] == first["intent_id"]
    assert calls["submit"] == 1


def test_ambiguous_submit_recovers_exact_client_order_id(monkeypatch) -> None:
    monkeypatch.setattr(alpaca_execution, "get_account", _ready_account)
    monkeypatch.setattr(alpaca_execution, "get_open_position", lambda _symbol: None)

    def _ambiguous(**kwargs):
        raise alpaca_paper.AlpacaPaperError("timeout", ambiguous=True)

    monkeypatch.setattr(alpaca_execution, "submit_market_order", _ambiguous)
    monkeypatch.setattr(
        alpaca_execution,
        "get_order_by_client_order_id",
        lambda client_order_id: _order(client_order_id=client_order_id),
    )
    monkeypatch.setattr(alpaca_execution, "sleep", lambda _seconds: None)

    result = alpaca_execution.submit_paper_market_order(
        symbol="AAPL",
        asset_class="us_equity",
        side="buy",
        qty=0.01,
        notional=None,
        timeframe="D1",
        strategy="manual",
        rationale="ambiguous but recoverable",
    )

    assert result["status"] == "executed"
    intent = list_order_intents(1)[0]
    assert intent.details["reconciliation_source"] == "post_submit_lookup"
    assert intent.details["reconciliation_resolved"] is True


def test_ambiguous_submit_with_no_broker_order_fails_without_auto_retry(monkeypatch) -> None:
    monkeypatch.setattr(alpaca_execution, "get_account", _ready_account)
    monkeypatch.setattr(alpaca_execution, "get_open_position", lambda _symbol: None)
    monkeypatch.setattr(
        alpaca_execution,
        "submit_market_order",
        lambda **kwargs: (_ for _ in ()).throw(
            alpaca_paper.AlpacaPaperError("timeout", ambiguous=True)
        ),
    )
    monkeypatch.setattr(alpaca_execution, "get_order_by_client_order_id", lambda _cid: None)
    monkeypatch.setattr(alpaca_execution, "sleep", lambda _seconds: None)

    result = alpaca_execution.submit_paper_market_order(
        symbol="AAPL",
        asset_class="us_equity",
        side="buy",
        qty=0.01,
        notional=None,
        timeframe="D1",
        strategy="manual",
        rationale="ambiguous absent",
    )

    assert result["status"] == "failed"
    intent = list_order_intents(1)[0]
    assert intent.details["outcome_state"] == "submission_resolved_no_order"
    assert intent.details["retryable"] is True
    assert intent.details["automatic_retry"] is False


def test_reconcile_survives_restart_style_nonterminal_intent(monkeypatch) -> None:
    monkeypatch.setattr(alpaca_execution, "get_account", _ready_account)
    monkeypatch.setattr(alpaca_execution, "get_open_position", lambda _symbol: None)
    monkeypatch.setattr(alpaca_execution, "list_positions", lambda: [])
    state = {"order": None}

    def _submit(**kwargs):
        state["order"] = _order(
            status="new",
            client_order_id=str(kwargs["client_order_id"]),
        )
        return state["order"]

    monkeypatch.setattr(alpaca_execution, "submit_market_order", _submit)

    created = alpaca_execution.submit_paper_market_order(
        symbol="AAPL",
        asset_class="us_equity",
        side="buy",
        qty=0.01,
        notional=None,
        timeframe="D1",
        strategy="manual",
        rationale="restart recovery",
    )
    assert created["status"] == "accepted"

    monkeypatch.setattr(
        alpaca_execution,
        "get_order_by_client_order_id",
        lambda client_order_id: _order(client_order_id=client_order_id),
    )
    reconciled = alpaca_execution.reconcile_paper_execution()

    assert reconciled["checked_intents"] == 1
    assert reconciled["executed"] == 1
    intent = list_order_intents(1)[0]
    assert intent.status == "executed"
    assert intent.details["reconciliation_source"] == "manual_reconcile"


def test_reconcile_missing_order_keeps_visible_position_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(alpaca_execution, "get_account", _ready_account)
    state = {"preflight": True}

    def _position_truth(_symbol):
        if state["preflight"]:
            return None
        return _position()

    monkeypatch.setattr(alpaca_execution, "get_open_position", _position_truth)
    monkeypatch.setattr(alpaca_execution, "list_positions", lambda: [_position()])
    monkeypatch.setattr(
        alpaca_execution,
        "submit_market_order",
        lambda **kwargs: _order(
            status="new",
            client_order_id=str(kwargs["client_order_id"]),
        ),
    )

    created = alpaca_execution.submit_paper_market_order(
        symbol="AAPL",
        asset_class="us_equity",
        side="buy",
        qty=0.01,
        notional=None,
        timeframe="D1",
        strategy="manual",
        rationale="visible-position recovery",
    )
    assert created["status"] == "accepted"

    state["preflight"] = False
    monkeypatch.setattr(alpaca_execution, "get_order_by_client_order_id", lambda _cid: None)
    monkeypatch.setattr(alpaca_execution, "sleep", lambda _seconds: None)

    reconciled = alpaca_execution.reconcile_paper_execution()

    assert reconciled["unresolved"] == 1
    intent = list_order_intents(1)[0]
    assert intent.status == "accepted"
    assert intent.details["outcome_state"] == "position_visible_order_unresolved"
    assert intent.details["automatic_retry"] is False


def test_close_already_flat_is_a_noop_without_intent(monkeypatch) -> None:
    monkeypatch.setattr(alpaca_execution, "get_account", _ready_account)
    monkeypatch.setattr(alpaca_execution, "get_open_position", lambda _symbol: None)

    result = alpaca_execution.close_paper_position(
        symbol_or_asset_id="AAPL",
        timeframe="D1",
        strategy="manual",
        rationale="close",
    )

    assert result["status"] == "already_flat"
    assert result["action_taken"] is False
    assert list_order_intents(10) == []


def test_full_close_filled_is_recorded_as_flat_execution(monkeypatch) -> None:
    monkeypatch.setattr(alpaca_execution, "get_account", _ready_account)
    monkeypatch.setattr(alpaca_execution, "get_open_position", lambda _symbol: _position())
    monkeypatch.setattr(
        alpaca_execution,
        "close_position",
        lambda _symbol: _order(
            status="filled",
            client_order_id="broker-close-client",
            order_id="close-order",
            side="sell",
        ),
    )

    result = alpaca_execution.close_paper_position(
        symbol_or_asset_id="AAPL",
        timeframe="D1",
        strategy="manual",
        rationale="full close",
    )

    assert result["status"] == "executed"
    intent = list_order_intents(1)[0]
    assert intent.intent_type == "close"
    assert intent.direction == "flat"
    assert intent.details["close_semantics"] == "full_position_to_flat"
    assert intent.details["broker_order"]["id"] == "close-order"
    assert any(
        audit.event_type == "alpaca_paper_position_close_filled"
        for audit in list_trade_audits(10)
    )


def test_ambiguous_close_still_open_remains_unresolved_and_blocks_repeat(monkeypatch) -> None:
    monkeypatch.setattr(alpaca_execution, "get_account", _ready_account)
    calls = {"position": 0}

    def _position_truth(_symbol):
        calls["position"] += 1
        return _position()

    monkeypatch.setattr(alpaca_execution, "get_open_position", _position_truth)
    monkeypatch.setattr(
        alpaca_execution,
        "close_position",
        lambda _symbol: (_ for _ in ()).throw(
            alpaca_paper.AlpacaPaperError("timeout", ambiguous=True)
        ),
    )

    result = alpaca_execution.close_paper_position(
        symbol_or_asset_id="AAPL",
        timeframe="D1",
        strategy="manual",
        rationale="ambiguous close",
    )
    repeated = alpaca_execution.close_paper_position(
        symbol_or_asset_id="AAPL",
        timeframe="D1",
        strategy="manual",
        rationale="repeat close",
    )

    assert result["status"] == "reconciliation_required"
    assert repeated["status"] == "blocked"
    assert repeated["intent_id"] == result["intent_id"]
    intent = list_order_intents(1)[0]
    assert intent.status == "accepted"
    assert intent.details["reconciliation_resolved"] is False
    assert intent.details["automatic_retry"] is False


def test_reconcile_ambiguous_close_can_confirm_flat_from_position_truth(monkeypatch) -> None:
    monkeypatch.setattr(alpaca_execution, "get_account", _ready_account)
    monkeypatch.setattr(alpaca_execution, "list_positions", lambda: [])
    state = {"open": True}

    monkeypatch.setattr(
        alpaca_execution,
        "get_open_position",
        lambda _symbol: _position() if state["open"] else None,
    )
    monkeypatch.setattr(
        alpaca_execution,
        "close_position",
        lambda _symbol: (_ for _ in ()).throw(
            alpaca_paper.AlpacaPaperError("timeout", ambiguous=True)
        ),
    )

    initial = alpaca_execution.close_paper_position(
        symbol_or_asset_id="AAPL",
        timeframe="D1",
        strategy="manual",
        rationale="ambiguous close",
    )
    assert initial["status"] == "reconciliation_required"

    state["open"] = False
    reconciled = alpaca_execution.reconcile_paper_execution()

    assert reconciled["executed"] == 1
    intent = list_order_intents(1)[0]
    assert intent.status == "executed"
    assert intent.details["outcome_state"] == "flat_confirmed_by_reconciliation"
    assert any(
        audit.event_type == "alpaca_paper_position_flat_confirmed"
        for audit in list_trade_audits(10)
    )


def test_alpaca_execution_api_delegates_without_touching_ctrader(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_EVENT_INTELLIGENCE_ON_BOOT", "0")

    router_module = importlib.import_module("backend.api.router")
    app_module = importlib.import_module("backend.app")

    monkeypatch.setattr(
        router_module,
        "submit_alpaca_paper_market_order",
        lambda **kwargs: {
            "action_taken": True,
            "intent_id": 1,
            "status": "executed",
            "summary": "filled",
            "broker_order": {"id": "order-1"},
        },
    )
    monkeypatch.setattr(
        router_module,
        "close_alpaca_paper_position",
        lambda **kwargs: {
            "action_taken": True,
            "intent_id": 2,
            "status": "executed",
            "summary": "flat",
            "broker_order": {"id": "close-1"},
        },
    )
    monkeypatch.setattr(
        router_module,
        "reconcile_alpaca_paper_execution",
        lambda **kwargs: {
            "provider": "alpaca",
            "environment": "paper",
            "checked_intents": 0,
        },
    )

    with TestClient(app_module.app) as client:
        opened = client.post(
            "/api/alpaca/paper/orders/market",
            json={
                "symbol": "AAPL",
                "asset_class": "us_equity",
                "side": "buy",
                "notional": 5,
                "timeframe": "D1",
                "strategy": "manual",
            },
        )
        closed = client.post(
            "/api/alpaca/paper/positions/close",
            json={
                "symbol_or_asset_id": "AAPL",
                "timeframe": "D1",
                "strategy": "manual",
            },
        )
        reconciled = client.post("/api/alpaca/paper/reconcile?limit=25")

    assert opened.status_code == 200
    assert opened.json()["status"] == "executed"
    assert closed.status_code == 200
    assert closed.json()["status"] == "executed"
    assert reconciled.status_code == 200
    assert reconciled.json()["environment"] == "paper"
