from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from backend.domain.models import BrokerStatus
from backend.services import protection_health
from backend.services.protection_health import build_protection_health
from backend.storage.db import get_db
from backend.storage.repositories import (
    create_order_intent,
    open_paper_position,
    update_order_intent_status,
)


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
    position_id: int,
    symbol: str = "NAS100",
    direction: str = "buy",
    stop_loss: float | None = 29476.4,
    take_profit: float | None = 29511.7,
) -> dict:
    return {
        "symbol": symbol,
        "direction": direction,
        "volume_lots": 0.10,
        "entry_price": 29486.2,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "position_id": position_id,
    }


def _local(
    *,
    broker_position_id: int | None,
    symbol: str = "NAS100",
    stop_loss: float | None = 29476.4,
    take_profit: float | None = 29511.7,
):
    return open_paper_position(
        symbol=symbol,
        timeframe="M5",
        strategy="breakout",
        direction="long",
        quantity=0.10,
        entry_price=29486.2,
        stop_loss=stop_loss,
        take_profit=take_profit,
        broker_position_id=broker_position_id,
    )


def _executed_intent(*, position_id: int, symbol: str = "NAS100"):
    intent = create_order_intent(
        symbol=symbol,
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
        rationale="protection health fixture",
        details={},
    )
    return update_order_intent_status(
        intent.id,
        "executed",
        {
            "broker_order": {
                "position_id": position_id,
                "symbol": symbol,
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


def test_protection_health_reports_full_partial_unprotected_and_coverage(monkeypatch) -> None:
    full = _local(broker_position_id=980001, symbol="NAS100")
    partial = _local(broker_position_id=980002, symbol="US30")
    unprotected = _local(
        broker_position_id=980003,
        symbol="XAUUSD",
        stop_loss=2900.0,
        take_profit=3100.0,
    )
    monkeypatch.setattr(protection_health, "get_broker_status", lambda: _ready_broker(open_positions=4))
    monkeypatch.setattr(
        protection_health,
        "list_positions",
        lambda: [
            _broker_row(position_id=980001, symbol="NAS100"),
            _broker_row(position_id=980002, symbol="US30", take_profit=None),
            _broker_row(position_id=980003, symbol="XAUUSD", stop_loss=None, take_profit=None),
            _broker_row(position_id=989999, symbol="EURUSD"),
        ],
    )

    report = build_protection_health()

    assert report.status == "unprotected"
    assert report.broker_truth_available is True
    assert report.managed_positions == 3
    assert report.assessable_positions == 3
    assert report.fully_protected_positions == 1
    assert report.full_protection_coverage_pct == pytest.approx(100.0 / 3.0)
    assert report.ignored_broker_positions == 1
    assert report.counts == {
        "fully_protected": 1,
        "partially_protected": 1,
        "unprotected": 1,
        "unavailable": 0,
        "identity_unresolved": 0,
    }
    by_local = {item.local_position_id: item for item in report.items}
    assert by_local[full.id].status == "fully_protected"
    assert by_local[partial.id].status == "partially_protected"
    assert by_local[unprotected.id].status == "unprotected"
    assert by_local[unprotected.id].broker_stop_loss is None
    assert by_local[unprotected.id].broker_take_profit is None


def test_local_requested_targets_do_not_mask_unprotected_broker_truth(monkeypatch) -> None:
    local = _local(
        broker_position_id=981001,
        stop_loss=29000.0,
        take_profit=30000.0,
    )
    monkeypatch.setattr(protection_health, "get_broker_status", _ready_broker)
    monkeypatch.setattr(
        protection_health,
        "list_positions",
        lambda: [
            _broker_row(
                position_id=981001,
                stop_loss=None,
                take_profit=None,
            )
        ],
    )

    report = build_protection_health()

    assert report.status == "unprotected"
    item = report.items[0]
    assert item.local_position_id == local.id
    assert item.status == "unprotected"
    assert item.broker_stop_loss is None
    assert item.broker_take_profit is None
    assert "local requested" in item.action_required.lower() or "existing protection" in item.action_required.lower()


def test_protection_health_reports_unavailable_without_requesting_positions(monkeypatch) -> None:
    local = _local(broker_position_id=982001)
    monkeypatch.setattr(protection_health, "get_broker_status", _unavailable_broker)
    monkeypatch.setattr(
        protection_health,
        "list_positions",
        lambda: pytest.fail("positions must not be requested when broker execution truth is unavailable"),
    )

    report = build_protection_health()

    assert report.status == "unavailable"
    assert report.broker_truth_available is False
    assert report.managed_positions == 1
    assert report.assessable_positions == 0
    assert report.full_protection_coverage_pct is None
    assert report.items[0].local_position_id == local.id
    assert report.items[0].status == "unavailable"


def test_protection_health_reports_unavailable_when_broker_position_read_fails(monkeypatch) -> None:
    _local(broker_position_id=983001)
    monkeypatch.setattr(protection_health, "get_broker_status", _ready_broker)

    def _fail():
        raise RuntimeError("simulated broker read failure")

    monkeypatch.setattr(protection_health, "list_positions", _fail)

    report = build_protection_health()

    assert report.status == "unavailable"
    assert report.broker_truth_available is False
    assert "simulated broker read failure" in report.summary


def test_protection_health_marks_persisted_identity_conflict_unresolved(monkeypatch) -> None:
    local = _local(broker_position_id=984001)
    monkeypatch.setattr(protection_health, "get_broker_status", _ready_broker)
    monkeypatch.setattr(
        protection_health,
        "list_positions",
        lambda: [_broker_row(position_id=984001, direction="sell")],
    )

    report = build_protection_health()

    assert report.status == "identity_unresolved"
    assert report.assessable_positions == 0
    assert report.full_protection_coverage_pct is None
    item = report.items[0]
    assert item.local_position_id == local.id
    assert item.broker_sync_status == "id_mismatch"
    assert item.status == "identity_unresolved"


def test_canonical_broker_only_position_is_measured_from_broker_protection(monkeypatch) -> None:
    intent = _executed_intent(position_id=985001)
    monkeypatch.setattr(protection_health, "get_broker_status", lambda: _ready_broker(open_positions=2))
    monkeypatch.setattr(
        protection_health,
        "list_positions",
        lambda: [
            _broker_row(position_id=985001),
            _broker_row(position_id=985999, symbol="EURUSD"),
        ],
    )

    report = build_protection_health()

    assert report.status == "fully_protected"
    assert report.managed_positions == 1
    assert report.assessable_positions == 1
    assert report.fully_protected_positions == 1
    assert report.full_protection_coverage_pct == 100.0
    assert report.ignored_broker_positions == 1
    item = report.items[0]
    assert item.scope == "broker_position"
    assert item.local_position_id is None
    assert item.broker_position_id == 985001
    assert item.intent_id == intent.id
    assert item.status == "fully_protected"


def test_legacy_tracker_identity_remains_unresolved_even_when_broker_has_targets(monkeypatch) -> None:
    intent = _executed_intent(position_id=986001)
    legacy = _local(broker_position_id=None)
    monkeypatch.setattr(protection_health, "get_broker_status", _ready_broker)
    monkeypatch.setattr(
        protection_health,
        "list_positions",
        lambda: [_broker_row(position_id=986001)],
    )

    report = build_protection_health()

    assert report.status == "identity_unresolved"
    assert report.assessable_positions == 0
    item = report.items[0]
    assert item.local_position_id == legacy.id
    assert item.intent_id == intent.id
    assert item.broker_position_id == 986001
    assert item.broker_stop_loss == pytest.approx(29476.4)
    assert item.broker_take_profit == pytest.approx(29511.7)
    assert item.broker_sync_status == "legacy_identity_unpersisted"
    with get_db() as db:
        row = db.execute(
            "SELECT broker_position_id FROM paper_positions WHERE id = ?",
            (legacy.id,),
        ).fetchone()
    assert row["broker_position_id"] is None


def test_unavailable_with_no_known_managed_positions_keeps_zero_totals(monkeypatch) -> None:
    monkeypatch.setattr(protection_health, "get_broker_status", _unavailable_broker)
    monkeypatch.setattr(
        protection_health,
        "list_positions",
        lambda: pytest.fail("positions must not be requested while unavailable"),
    )

    report = build_protection_health()

    assert report.status == "unavailable"
    assert report.managed_positions == 0
    assert report.assessable_positions == 0
    assert report.fully_protected_positions == 0
    assert report.items == []
    assert report.full_protection_coverage_pct is None


def test_protection_health_api_is_read_only(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_EVENT_INTELLIGENCE_ON_BOOT", "0")
    _local(broker_position_id=987001)
    before = _table_counts()

    monkeypatch.setattr(protection_health, "get_broker_status", _ready_broker)
    monkeypatch.setattr(
        protection_health,
        "list_positions",
        lambda: [_broker_row(position_id=987001)],
    )

    from backend.app import app

    with TestClient(app) as client:
        response = client.get("/api/reports/protection-health")

    after = _table_counts()
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "fully_protected"
    assert payload["full_protection_coverage_pct"] == 100.0
    assert payload["items"][0]["broker_sync_status"] == "id_match"
    assert before == after
