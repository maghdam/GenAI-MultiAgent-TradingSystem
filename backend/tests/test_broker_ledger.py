from __future__ import annotations

from datetime import datetime

import pytest

from backend.services.broker_ledger import reconcile_closed_demo_history
from backend.storage.repositories import (
    close_paper_position,
    create_order_intent,
    list_paper_positions,
    list_trade_audits,
    open_paper_position,
    update_order_intent_status,
)


def test_closed_demo_history_replaces_estimated_pnl_with_ctrader_deal(monkeypatch) -> None:
    intent = create_order_intent(
        symbol="US30",
        timeframe="M5",
        strategy="breakout",
        direction="long",
        intent_type="open",
        status="accepted",
        confidence=0.8,
        entry_price=51646.4,
        stop_loss=51630.0,
        take_profit=51690.0,
        quantity=0.1,
        rationale="test",
        details={},
    )
    position = open_paper_position(
        symbol="US30",
        timeframe="M5",
        strategy="breakout",
        direction="long",
        quantity=0.1,
        entry_price=51646.4,
        stop_loss=51630.0,
        take_profit=51690.0,
        account_currency="CHF",
        cash_per_price_unit_per_lot=1.0,
        instrument_spec_source="ctrader_contract",
    )
    update_order_intent_status(
        intent.id,
        "executed",
        {
            "opened_position_id": position.id,
            "execution_mode": "ctrader_demo",
            "broker_order": {
                "position_id": 700009,
                "symbol": "US30",
                "quantity_lots": 0.1,
            },
        },
        reason="ctrader_demo_order_executed",
    )

    # Simulate the old local-bar estimate being written first.
    close_paper_position(position.id, 51697.0, "broker_position_closed")
    initially_closed = list_paper_positions("closed")[0]
    assert initially_closed.realized_pnl > 0
    assert initially_closed.realized_pnl_source == "paper_estimate"

    broker_closed_at = datetime.fromisoformat("2026-09-23T19:03:11.014000")
    monkeypatch.setattr(
        "backend.services.broker_ledger.get_closed_position_summary",
        lambda broker_position_id: {
            "status": "found",
            "position_id": broker_position_id,
            "exit_price": 51637.5,
            "closed_at": broker_closed_at,
            "gross_profit": -0.73,
            "swap": 0.0,
            "commission": 0.0,
            "pnl_conversion_fee": 0.0,
            "net_profit": -0.73,
            "closed_volume_api": 10,
            "deal_ids": [88001],
        },
    )

    result = reconcile_closed_demo_history(limit=20)

    assert result["reconciled"] == 1
    corrected = list_paper_positions("closed")[0]
    assert corrected.broker_position_id == 700009
    assert corrected.exit_price == pytest.approx(51637.5)
    assert corrected.realized_pnl == pytest.approx(-0.73)
    assert corrected.realized_pnl_source == "ctrader_deal"
    assert corrected.closed_at == broker_closed_at

    close_audit = next(
        row
        for row in list_trade_audits(20)
        if row.event_type == "paper_position_closed" and row.position_id == position.id
    )
    assert close_audit.details["realized_pnl"] == pytest.approx(-0.73)
    assert close_audit.details["realized_pnl_source"] == "ctrader_deal"
    assert close_audit.details["broker_position_id"] == 700009

    reconciled_audit = next(
        row
        for row in list_trade_audits(20)
        if row.event_type == "ctrader_demo_close_reconciled" and row.position_id == position.id
    )
    assert reconciled_audit.details["broker_exit_price"] == pytest.approx(51637.5)
    assert reconciled_audit.details["broker_net_profit"] == pytest.approx(-0.73)


def test_closed_history_skips_paper_position_without_broker_identity(monkeypatch) -> None:
    position = open_paper_position(
        symbol="EURUSD",
        timeframe="M5",
        strategy="sma_cross",
        direction="long",
        quantity=0.1,
        entry_price=1.10,
        stop_loss=1.09,
        take_profit=1.12,
    )
    close_paper_position(position.id, 1.11, "take_profit")
    monkeypatch.setattr(
        "backend.services.broker_ledger.get_closed_position_summary",
        lambda broker_position_id: pytest.fail("pure paper trade must not query broker history"),
    )

    result = reconcile_closed_demo_history(limit=20)

    assert result["reconciled"] == 0
    assert result["missing_broker_id"] == 1
