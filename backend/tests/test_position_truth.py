from __future__ import annotations

from datetime import UTC, datetime

from backend.domain.models import PaperPosition
from backend.services.position_truth import attach_broker_truth


def _position(*, broker_position_id: int | None = 900001) -> PaperPosition:
    return PaperPosition(
        id=1,
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        direction="long",
        quantity=0.1,
        status="open",
        entry_price=29487.5,
        current_price=29490.0,
        stop_loss=29470.0,
        take_profit=29520.0,
        opened_at=datetime(2026, 9, 27, 12, 0, 0),
        broker_position_id=broker_position_id,
    )


def test_attach_broker_truth_exposes_matched_ctrader_snapshot() -> None:
    synced_at = datetime(2026, 9, 27, 16, 30, 0)
    positions = attach_broker_truth(
        [_position()],
        [
            {
                "symbol": "XAUUSD",
                "direction": "buy",
                "volume_lots": 0.1,
                "entry_price": 29486.2,
                "stop_loss": 29471.0,
                "take_profit": 29518.0,
                "position_id": 900001,
            }
        ],
        synced_at=synced_at,
    )

    position = positions[0]
    assert position.entry_price == 29487.5
    assert position.stop_loss == 29470.0
    assert position.take_profit == 29520.0
    assert position.broker_entry_price == 29486.2
    assert position.broker_stop_loss == 29471.0
    assert position.broker_take_profit == 29518.0
    assert position.broker_protection_status == "protected"
    assert position.broker_last_synced_at == synced_at
    assert position.broker_sync_status == "id_match"


def test_attach_broker_truth_does_not_fallback_when_persisted_id_is_missing() -> None:
    positions = attach_broker_truth(
        [_position()],
        [
            {
                "symbol": "XAUUSD",
                "direction": "buy",
                "entry_price": 30000.0,
                "stop_loss": 29900.0,
                "take_profit": 30200.0,
                "position_id": 900002,
            }
        ],
    )

    position = positions[0]
    assert position.broker_entry_price is None
    assert position.broker_stop_loss is None
    assert position.broker_take_profit is None
    assert position.broker_protection_status == "unavailable"
    assert position.broker_last_synced_at is None
    assert position.broker_sync_status == "id_not_found"


def test_attach_broker_truth_marks_paper_position_as_not_applicable() -> None:
    position = attach_broker_truth([_position(broker_position_id=None)], None)[0]

    assert position.broker_entry_price is None
    assert position.broker_protection_status == "unavailable"
    assert position.broker_last_synced_at is None
    assert position.broker_sync_status == "paper_only"
