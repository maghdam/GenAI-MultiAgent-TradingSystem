from __future__ import annotations

from datetime import UTC, datetime

from backend.domain.models import PaperPosition
from backend.services.broker_position_match import match_broker_position


def _position(*, broker_position_id: int | None) -> PaperPosition:
    return PaperPosition(
        id=1,
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        direction="long",
        quantity=0.1,
        status="open",
        entry_price=100.0,
        current_price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        opened_at=datetime.now(UTC).replace(tzinfo=None),
        broker_position_id=broker_position_id,
    )


def test_persisted_broker_id_wins_over_same_symbol_direction_fallback() -> None:
    rows = [
        {"position_id": 222, "symbol": "XAUUSD", "direction": "buy"},
        {"position_id": 111, "symbol": "XAUUSD", "direction": "buy"},
    ]

    result = match_broker_position(_position(broker_position_id=111), rows)

    assert result.status == "id_match"
    assert result.row is not None
    assert result.row["position_id"] == 111


def test_persisted_broker_id_does_not_fall_back_to_different_position() -> None:
    rows = [
        {"position_id": 222, "symbol": "XAUUSD", "direction": "buy"},
    ]

    result = match_broker_position(_position(broker_position_id=111), rows)

    assert result.status == "id_not_found"
    assert result.row is None


def test_persisted_broker_id_rejects_symbol_or_direction_mismatch() -> None:
    rows = [
        {"position_id": 111, "symbol": "US30", "direction": "buy"},
    ]

    result = match_broker_position(_position(broker_position_id=111), rows)

    assert result.status == "id_mismatch"
    assert result.row is None


def test_legacy_position_uses_unique_symbol_direction_fallback() -> None:
    rows = [
        {"position_id": 333, "symbol": "XAUUSD", "direction": "buy"},
        {"position_id": 444, "symbol": "US30", "direction": "buy"},
    ]

    result = match_broker_position(_position(broker_position_id=None), rows)

    assert result.status == "legacy_match"
    assert result.row is not None
    assert result.row["position_id"] == 333


def test_legacy_position_refuses_ambiguous_symbol_direction_fallback() -> None:
    rows = [
        {"position_id": 333, "symbol": "XAUUSD", "direction": "buy"},
        {"position_id": 444, "symbol": "XAUUSD", "direction": "buy"},
    ]

    result = match_broker_position(_position(broker_position_id=None), rows)

    assert result.status == "legacy_ambiguous"
    assert result.row is None
    assert result.candidates == 2
