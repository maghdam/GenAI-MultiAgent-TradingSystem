from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Dict, Iterable, List

from backend.domain.models import PaperPosition
from backend.services.broker_position_match import match_broker_position


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _protection_status(stop_loss: float | None, take_profit: float | None) -> str:
    if stop_loss is not None and take_profit is not None:
        return "protected"
    if stop_loss is not None or take_profit is not None:
        return "partial"
    return "unprotected"


def attach_broker_truth(
    positions: Iterable[PaperPosition],
    broker_rows: Iterable[Dict[str, Any]] | None,
    *,
    synced_at: datetime | None = None,
) -> List[PaperPosition]:
    """Attach read-only cTrader truth without overwriting the local position ledger."""
    rows = None if broker_rows is None else list(broker_rows)
    observed_at = synced_at or datetime.now(UTC).replace(tzinfo=None)
    enriched: List[PaperPosition] = []

    for position in positions:
        if position.broker_position_id is None:
            enriched.append(
                position.model_copy(
                    update={
                        "broker_protection_status": "unavailable",
                        "broker_sync_status": "paper_only",
                    }
                )
            )
            continue

        if rows is None:
            enriched.append(
                position.model_copy(
                    update={
                        "broker_protection_status": "unavailable",
                        "broker_sync_status": "unavailable",
                    }
                )
            )
            continue

        match = match_broker_position(position, rows)
        if not match.matched:
            enriched.append(
                position.model_copy(
                    update={
                        "broker_protection_status": "unavailable",
                        "broker_sync_status": match.status,
                    }
                )
            )
            continue

        row = match.row or {}
        stop_loss = _optional_float(row.get("stop_loss"))
        take_profit = _optional_float(row.get("take_profit"))
        enriched.append(
            position.model_copy(
                update={
                    "broker_entry_price": _optional_float(row.get("entry_price")),
                    "broker_stop_loss": stop_loss,
                    "broker_take_profit": take_profit,
                    "broker_protection_status": _protection_status(stop_loss, take_profit),
                    "broker_last_synced_at": observed_at,
                    "broker_sync_status": match.status,
                }
            )
        )

    return enriched
