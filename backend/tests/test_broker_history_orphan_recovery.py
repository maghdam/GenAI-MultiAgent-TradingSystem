from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from backend.domain.models import BrokerAccountSnapshot
from backend.services import broker_ledger, execution_engine
from backend.services.journal_statement_comparison import build_journal_export
from backend.storage.repositories import (
    create_order_intent,
    list_broker_deals,
    list_paper_positions,
    list_trade_audits,
    update_order_intent_status,
)


def _candidate(
    *,
    broker_position_id: int,
    client_order_id: str,
    net_profit: float = -20.39,
) -> dict:
    return {
        "broker_position_id": broker_position_id,
        "opening_order_id": 73809759,
        "client_order_id": client_order_id,
        "symbol": "XAUUSD",
        "direction": "short",
        "quantity_lots": 0.10,
        "opened_at": datetime.fromisoformat("2026-10-07T06:28:29.000000"),
        "closed_at": datetime.fromisoformat("2026-10-07T06:29:25.249000"),
        "entry_price": 4133.41,
        "exit_price": 4135.86,
        "deals": [
            {
                "deal_id": 63499285,
                "execution_price": 4135.86,
                "execution_at": datetime.fromisoformat("2026-10-07T06:29:25.249000"),
                "closed_volume_api": 1000.0,
                "closed_volume_lots": 0.10,
                "gross_profit": net_profit,
                "swap": 0.0,
                "commission": 0.0,
                "pnl_conversion_fee": 0.0,
                "net_profit": net_profit,
            }
        ],
    }


def _ready_history(monkeypatch, candidate: dict) -> None:
    monkeypatch.setattr(
        broker_ledger,
        "get_broker_status",
        lambda: SimpleNamespace(execution_ready=True),
    )
    monkeypatch.setattr(
        broker_ledger,
        "get_broker_account_snapshot",
        lambda: BrokerAccountSnapshot(currency="CHF", verified=True),
    )
    monkeypatch.setattr(
        broker_ledger,
        "get_account_trade_history",
        lambda **kwargs: [candidate],
    )
    monkeypatch.setattr(broker_ledger, "list_positions", lambda: [])


def test_legacy_numeric_client_id_collision_recovers_without_false_intent_link(monkeypatch) -> None:
    collision = create_order_intent(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        direction="short",
        intent_type="open",
        status="accepted",
        confidence=0.85,
        entry_price=4303.48,
        stop_loss=4299.63,
        take_profit=4311.18,
        quantity=0.10,
        rationale="older runtime intent",
        details={"source": "auto"},
    )
    update_order_intent_status(
        collision.id,
        "executed",
        {
            "opened_position_id": 33,
            "broker_order": {
                "position_id": 57330292,
                "symbol": "XAUUSD",
            },
        },
        reason="older_runtime_execution",
    )

    candidate = _candidate(
        broker_position_id=57868693,
        client_order_id=f"tradeagent-intent-{collision.id}",
    )
    _ready_history(monkeypatch, candidate)

    first = broker_ledger.recover_tradeagent_closed_history(
        window_days=30,
        now=datetime.fromisoformat("2026-10-07T12:00:00"),
    )

    assert first["recovered"] == 1
    positions = list_paper_positions("closed")
    assert len(positions) == 1
    recovered = positions[0]
    assert recovered.broker_position_id == 57868693
    assert recovered.strategy == "ctrader_recovered"
    assert recovered.timeframe == "UNKNOWN"
    assert recovered.realized_pnl == pytest.approx(-20.39)
    assert recovered.realized_pnl_source == "ctrader_deal"

    deals = list_broker_deals(local_position_id=recovered.id)
    assert [row["deal_id"] for row in deals] == [63499285]

    audit = next(
        row
        for row in list_trade_audits(20)
        if row.event_type == "ctrader_closed_orphan_recovered"
    )
    assert audit.intent_id is None
    assert audit.details["local_intent_linked"] is False
    assert audit.details["client_order_id"] == f"tradeagent-intent-{collision.id}"

    journal = build_journal_export(
        all_time=True,
        now=datetime.fromisoformat("2026-10-08T00:00:00"),
    )
    row = next(item for item in journal.rows if item.broker_position_id == 57868693)
    assert row.realized_pnl == pytest.approx(-20.39)
    assert row.realized_pnl_basis == "broker_deals"
    assert row.execution_source == "unknown"

    second = broker_ledger.recover_tradeagent_closed_history(
        window_days=30,
        now=datetime.fromisoformat("2026-10-07T12:00:00"),
    )
    assert second["recovered"] == 0
    assert second["already_tracked"] == 1
    assert len(list_paper_positions("closed")) == 1
    assert len(list_broker_deals(local_position_id=recovered.id)) == 1


def test_unique_persisted_client_id_restores_original_metadata(monkeypatch) -> None:
    intent = create_order_intent(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        direction="short",
        intent_type="open",
        status="accepted",
        confidence=0.85,
        entry_price=4133.41,
        stop_loss=4135.84,
        take_profit=4126.04,
        quantity=0.10,
        rationale="manual confirmation",
        details={"source": "manual"},
    )
    client_order_id = f"tradeagent-intent-{intent.id}-abc123def456"
    update_order_intent_status(
        intent.id,
        "accepted",
        {
            "client_msg_id": client_order_id,
            "outcome_state": "submission_reserved",
        },
        reason="submission_reserved",
    )

    _ready_history(
        monkeypatch,
        _candidate(
            broker_position_id=57868693,
            client_order_id=client_order_id,
        ),
    )

    result = broker_ledger.recover_tradeagent_closed_history(
        window_days=30,
        now=datetime.fromisoformat("2026-10-07T12:00:00"),
    )
    assert result["recovered"] == 1

    recovered = list_paper_positions("closed")[0]
    assert recovered.timeframe == "M5"
    assert recovered.strategy == "sma_cross"

    audit = next(
        row
        for row in list_trade_audits(20)
        if row.event_type == "ctrader_closed_orphan_recovered"
    )
    assert audit.intent_id == intent.id
    assert audit.details["local_intent_linked"] is True
    assert audit.details["execution_source"] == "manual"

    journal = build_journal_export(
        all_time=True,
        now=datetime.fromisoformat("2026-10-08T00:00:00"),
    )
    row = next(item for item in journal.rows if item.broker_position_id == 57868693)
    assert row.execution_source == "manual"


def test_ctrader_client_order_ids_are_unique_across_runtime_generations() -> None:
    first = execution_engine._new_ctrader_client_msg_id(844)
    second = execution_engine._new_ctrader_client_msg_id(844)

    assert first.startswith("tradeagent-intent-844-")
    assert second.startswith("tradeagent-intent-844-")
    assert first != second
    assert len(first) <= 50
    assert len(second) <= 50
