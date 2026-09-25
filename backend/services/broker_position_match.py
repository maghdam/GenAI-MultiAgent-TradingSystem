from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

from backend.domain.models import PaperPosition


@dataclass(frozen=True)
class BrokerPositionMatch:
    row: Dict[str, Any] | None
    status: str
    candidates: int = 0

    @property
    def matched(self) -> bool:
        return self.row is not None


def _row_position_id(row: Dict[str, Any]) -> int | None:
    try:
        value = int(row.get("position_id") or 0)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _row_matches_identity(row: Dict[str, Any], position: PaperPosition) -> bool:
    expected_side = "buy" if position.direction == "long" else "sell"
    return (
        str(row.get("symbol") or "").upper() == position.symbol.upper()
        and str(row.get("direction") or "").lower() == expected_side
    )


def match_broker_position(
    position: PaperPosition,
    broker_rows: Iterable[Dict[str, Any]],
) -> BrokerPositionMatch:
    """Match a local position to broker state without silently changing identity.

    Persisted cTrader position id is authoritative. Symbol+direction matching is
    permitted only for legacy local rows that do not yet have a broker id, and
    only when that fallback is unambiguous.
    """
    rows = list(broker_rows)
    persisted_id = int(position.broker_position_id or 0)

    if persisted_id > 0:
        for row in rows:
            if _row_position_id(row) != persisted_id:
                continue
            if not _row_matches_identity(row, position):
                return BrokerPositionMatch(row=None, status="id_mismatch", candidates=1)
            return BrokerPositionMatch(row=row, status="id_match", candidates=1)
        return BrokerPositionMatch(row=None, status="id_not_found", candidates=0)

    candidates = [row for row in rows if _row_matches_identity(row, position)]
    if len(candidates) == 1:
        return BrokerPositionMatch(row=candidates[0], status="legacy_match", candidates=1)
    if len(candidates) > 1:
        return BrokerPositionMatch(row=None, status="legacy_ambiguous", candidates=len(candidates))
    return BrokerPositionMatch(row=None, status="legacy_not_found", candidates=0)
