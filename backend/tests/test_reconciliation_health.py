from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from backend.domain.models import BrokerStatus
from backend.services import reconciliation_health
from backend.services.reconciliation_health import build_reconciliation_health
from backend.storage.db import get_db
from backend.storage.repositories import (
    create_order_intent,
    open_paper_position,
    update_order_intent_status,
)


BROKER_ID = 960001


def _ready_broker(*, open_positions: int = 1) -> BrokerStatus:
    return BrokerStatus(
        connected=True,
        socket_connected=True,
        account_authorized=True,
        symbols_loaded=100,
        open_positions=open_positions,
        pending_orders=0,
        ready=True,
        market_data_ready=True,
        broker_mode="ctrader_demo",
        account_id=123,
        account_type="demo",
        demo_account_confirmed=True,
        execution_ready=True,
    )


def _unavailable_broker() -> BrokerStatus:
    return BrokerStatus(
        connected=False,
        socket_connected=False,
        account_authorized=False,
        symbols_loaded=0,
        open_positions=0,
        pending_orders=0,
        ready=False,
        market_data_ready=False,
        broker_mode="ctrader",
        account_type="unknown",
        demo_account_confirmed=False,
        execution_ready=False,
    )


def _broker_row(
    *,
    position_id: int = BROKER_ID,
    symbol: str = "NAS100",
    direction: str = "buy",
) -> dict:
    return {
        "symbol": symbol,
        "direction": direction,
        "volume_lots": 0.10,
        "entry_price": 29486.2,
        "stop_loss": 29476.4,
        "take_profit": 29511.7,
        "position_id": position_id,
    }


def _local(*, broker_position_id: int | None = BROKER_ID):
    return open_paper_position(
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        direction="long",
        quantity=0.10,
        entry_price=29486.2,
        stop_loss=29476.4,
        take_profit=29511.7,
        broker_position_id=broker_position_id,
    )


def _executed_intent(*, position_id: int = BROKER_ID):
    intent = create_order_intent(
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        direction="long",
        intent_type="open",
        status="accepted",
        confidence=0.90,
        entry_price=29486.2,
        stop_loss=29476.4,
        take_profit=29511.7,
        quantity=0.10,
        rationale="reconciliation health fixture",
        details={},
    )
    return update_order_intent_status(
        intent.id,
        "executed",
        {
            "broker_order": {
                "position_id": position_id,
                "symbol": "NAS100",
                "direction": "long",
                "quantity_lots": 0.10,
            }
        },
        reason="fixture_executed",
    )


def _table_counts() -> dict[str, int]:
    tables = ("paper_positions", "order_intents", "order_intent_transitions")
    with get_db() as db:
        return {
            table: int(db.execute(f"SELECT COUNT(*) AS total FROM {table}").fetchone()["total"])
            for table in tables
        }


def test_reconciliation_health_reports_healthy_canonical_match_and_ignores_manual_broker(monkeypatch) -> None:
    local = _local()
    monkeypatch.setattr(reconciliation_health, "get_broker_status", _ready_broker)
    monkeypatch.setattr(
        reconciliation_health,
        "list_positions",
        lambda: [
            _broker_row(),
            _broker_row(position_id=970001, symbol="XAUUSD"),
        ],
    )

    report = build_reconciliation_health()

    assert report.status == "healthy"
    assert report.broker_truth_available is True
    assert report.local_managed_positions == 1
    assert report.canonical_broker_positions == 1
    assert report.ignored_broker_positions == 1
    assert len(report.items) == 1
    item = report.items[0]
    assert item.status == "healthy"
    assert item.local_position_id == local.id
    assert item.broker_position_id == BROKER_ID
    assert item.match_status == "id_match"


def test_reconciliation_health_reports_unavailable_without_guessing_broker_state(monkeypatch) -> None:
    local = _local()
    monkeypatch.setattr(reconciliation_health, "get_broker_status", _unavailable_broker)
    monkeypatch.setattr(
        reconciliation_health,
        "list_positions",
        lambda: pytest.fail("broker positions must not be requested when execution truth is unavailable"),
    )

    report = build_reconciliation_health()

    assert report.status == "unavailable"
    assert report.broker_truth_available is False
    assert report.canonical_broker_positions is None
    assert report.local_managed_positions == 1
    assert report.items[0].local_position_id == local.id
    assert report.items[0].status == "unavailable"
    assert "restore" in report.items[0].action_required.lower()


def test_reconciliation_health_reports_missing_broker_for_persisted_local_identity(monkeypatch) -> None:
    local = _local()
    monkeypatch.setattr(reconciliation_health, "get_broker_status", _ready_broker)
    monkeypatch.setattr(reconciliation_health, "list_positions", lambda: [])

    report = build_reconciliation_health()

    assert report.status == "missing_broker"
    assert report.canonical_broker_positions == 0
    assert len(report.items) == 1
    item = report.items[0]
    assert item.status == "missing_broker"
    assert item.local_position_id == local.id
    assert item.broker_position_id == BROKER_ID
    assert item.match_status == "id_not_found"
    assert "do not silently switch" in item.action_required.lower()


def test_reconciliation_health_reports_missing_local_only_for_canonical_tradeagent_broker(monkeypatch) -> None:
    intent = _executed_intent()
    monkeypatch.setattr(reconciliation_health, "get_broker_status", _ready_broker)
    monkeypatch.setattr(
        reconciliation_health,
        "list_positions",
        lambda: [
            _broker_row(),
            _broker_row(position_id=970001, symbol="XAUUSD"),
        ],
    )

    report = build_reconciliation_health()

    assert report.status == "missing_local"
    assert report.local_managed_positions == 0
    assert report.canonical_broker_positions == 1
    assert report.ignored_broker_positions == 1
    assert len(report.items) == 1
    item = report.items[0]
    assert item.status == "missing_local"
    assert item.scope == "broker_position"
    assert item.broker_position_id == BROKER_ID
    assert item.intent_id == intent.id
    assert "tracker recovery" in item.action_required.lower()


def test_reconciliation_health_reports_degraded_legacy_tracker_without_mutating_id(monkeypatch) -> None:
    intent = _executed_intent()
    legacy = _local(broker_position_id=None)
    monkeypatch.setattr(reconciliation_health, "get_broker_status", _ready_broker)
    monkeypatch.setattr(reconciliation_health, "list_positions", lambda: [_broker_row()])

    report = build_reconciliation_health()

    assert report.status == "degraded"
    assert len(report.items) == 1
    item = report.items[0]
    assert item.status == "degraded"
    assert item.local_position_id == legacy.id
    assert item.broker_position_id == BROKER_ID
    assert item.intent_id == intent.id
    assert item.match_status == "legacy_identity_unpersisted"
    with get_db() as db:
        persisted = db.execute(
            "SELECT broker_position_id FROM paper_positions WHERE id = ?",
            (legacy.id,),
        ).fetchone()
    assert persisted["broker_position_id"] is None


def test_reconciliation_health_reports_unresolved_identity_conflict(monkeypatch) -> None:
    local = _local()
    monkeypatch.setattr(reconciliation_health, "get_broker_status", _ready_broker)
    monkeypatch.setattr(
        reconciliation_health,
        "list_positions",
        lambda: [_broker_row(direction="sell")],
    )

    report = build_reconciliation_health()

    assert report.status == "unresolved"
    item = report.items[0]
    assert item.status == "unresolved"
    assert item.local_position_id == local.id
    assert item.match_status == "id_mismatch"
    assert "canonical persisted identity" in item.action_required.lower()


def test_reconciliation_health_reports_unresolved_ambiguous_legacy_candidates(monkeypatch) -> None:
    _executed_intent()
    first = _local(broker_position_id=None)
    second = _local(broker_position_id=None)
    monkeypatch.setattr(reconciliation_health, "get_broker_status", _ready_broker)
    monkeypatch.setattr(reconciliation_health, "list_positions", lambda: [_broker_row()])

    report = build_reconciliation_health()

    assert report.status == "unresolved"
    assert len(report.items) == 1
    item = report.items[0]
    assert item.scope == "broker_position"
    assert item.match_status == "legacy_local_ambiguous"
    assert item.local_position_id is None
    assert {first.id, second.id} == {
        row["id"]
        for row in [
            dict(get_db().execute("SELECT id FROM paper_positions WHERE id = ?", (first.id,)).fetchone()),
            dict(get_db().execute("SELECT id FROM paper_positions WHERE id = ?", (second.id,)).fetchone()),
        ]
    }


def test_reconciliation_health_api_is_read_only(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_EVENT_INTELLIGENCE_ON_BOOT", "0")
    _local()
    before = _table_counts()

    monkeypatch.setattr(reconciliation_health, "get_broker_status", _ready_broker)
    monkeypatch.setattr(reconciliation_health, "list_positions", lambda: [_broker_row()])

    from backend.app import app

    with TestClient(app) as client:
        response = client.get("/api/reports/reconciliation-health")

    after = _table_counts()
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["broker_truth_available"] is True
    assert payload["items"][0]["match_status"] == "id_match"
    assert before == after
