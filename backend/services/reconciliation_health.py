from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

from backend.domain.models import (
    BrokerStatus,
    PaperPosition,
    ReconciliationHealthItem,
    ReconciliationHealthResponse,
)
from backend.services.broker import get_broker_status, list_positions
from backend.services.broker_position_match import match_broker_position
from backend.services.reconciler import _is_canonical_tradeagent_recovery_intent
from backend.storage.repositories import list_order_intents, list_paper_positions


_STATUS_PRIORITY = {
    "healthy": 0,
    "degraded": 1,
    "unavailable": 2,
    "missing_broker": 3,
    "missing_local": 4,
    "unresolved": 5,
}


def _direction(row: dict[str, Any]) -> str:
    return "long" if str(row.get("direction") or "").lower() == "buy" else "short"


def _broker_id(row: dict[str, Any]) -> int | None:
    try:
        value = int(row.get("position_id") or 0)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _overall_status(items: list[ReconciliationHealthItem]) -> str:
    if not items:
        return "healthy"
    return max(items, key=lambda item: _STATUS_PRIORITY[item.status]).status


def _counts(items: list[ReconciliationHealthItem]) -> dict[str, int]:
    counts = Counter(item.status for item in items)
    return {
        status: int(counts.get(status, 0))
        for status in _STATUS_PRIORITY
    }


def _unavailable_response(
    *,
    broker: BrokerStatus,
    local_rows: list[PaperPosition],
    reason: str,
) -> ReconciliationHealthResponse:
    tracked = [
        position
        for position in local_rows
        if int(position.broker_position_id or 0) > 0
    ]
    items = [
        ReconciliationHealthItem(
            status="unavailable",
            scope="local_tracker",
            symbol=position.symbol.upper(),
            direction=position.direction,
            local_position_id=position.id,
            broker_position_id=int(position.broker_position_id or 0),
            match_status="broker_truth_unavailable",
            message=(
                f"Broker truth is unavailable for tracked position {position.id} "
                f"({position.symbol}:{position.timeframe})."
            ),
            action_required=(
                "Restore the confirmed cTrader demo broker connection/readiness and refresh "
                "reconciliation health before treating broker/local state as verified."
            ),
        )
        for position in tracked
    ]
    if not items:
        items.append(
            ReconciliationHealthItem(
                status="unavailable",
                scope="broker_position",
                symbol="",
                direction="long",
                match_status="broker_truth_unavailable",
                message="Broker truth is currently unavailable, so reconciliation health cannot be verified.",
                action_required=(
                    "Restore the confirmed cTrader demo broker connection/readiness and refresh "
                    "reconciliation health."
                ),
            )
        )
    return ReconciliationHealthResponse(
        status="unavailable",
        broker_truth_available=False,
        broker_execution_ready=bool(broker.execution_ready),
        checked_at=datetime.now(UTC).replace(tzinfo=None),
        local_managed_positions=len(tracked),
        canonical_broker_positions=None,
        ignored_broker_positions=0,
        counts=_counts(items),
        items=items,
        summary=reason,
    )


def build_reconciliation_health() -> ReconciliationHealthResponse:
    """Read-only broker-vs-local health using existing canonical identity rules.

    This function never reconciles, adopts, closes, amends, or otherwise mutates
    broker/local trading state. Persisted broker position IDs remain authoritative
    for existing trackers. Broker-only rows are considered TradeAgent-managed only
    when the existing restart-recovery intent identity rule confirms them.
    """

    local_rows = list_paper_positions("open")
    intents = list_order_intents(500)
    broker = get_broker_status()

    if not broker.execution_ready:
        reason = (
            "cTrader execution truth is unavailable; broker/local reconciliation "
            "cannot be verified without a ready confirmed cTrader connection."
        )
        return _unavailable_response(broker=broker, local_rows=local_rows, reason=reason)

    try:
        broker_rows = list(list_positions())
    except Exception as exc:
        return _unavailable_response(
            broker=broker,
            local_rows=local_rows,
            reason=f"Could not read cTrader positions: {exc}",
        )

    positive_locals = [
        position
        for position in local_rows
        if int(position.broker_position_id or 0) > 0
    ]
    local_by_broker: dict[int, list[PaperPosition]] = defaultdict(list)
    for position in positive_locals:
        local_by_broker[int(position.broker_position_id or 0)].append(position)

    broker_by_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in broker_rows:
        position_id = _broker_id(row)
        if position_id is not None:
            broker_by_id[position_id].append(row)

    canonical_broker: dict[int, tuple[dict[str, Any], Any]] = {}
    for row in broker_rows:
        position_id = _broker_id(row)
        if position_id is None:
            continue
        matching_intent = next(
            (
                intent
                for intent in intents
                if _is_canonical_tradeagent_recovery_intent(intent, row)
            ),
            None,
        )
        if matching_intent is not None:
            canonical_broker[position_id] = (row, matching_intent)

    scoped_broker_ids = set(local_by_broker) | set(canonical_broker)
    ignored_broker_positions = sum(
        1
        for row in broker_rows
        if (_broker_id(row) or 0) not in scoped_broker_ids
    )

    items: list[ReconciliationHealthItem] = []
    handled_local_ids: set[int] = set()

    # Existing persisted local broker IDs are authoritative.
    for broker_position_id, local_group in sorted(local_by_broker.items()):
        broker_group = broker_by_id.get(broker_position_id, [])

        if len(local_group) > 1:
            for position in local_group:
                handled_local_ids.add(position.id)
                items.append(
                    ReconciliationHealthItem(
                        status="unresolved",
                        scope="local_tracker",
                        symbol=position.symbol.upper(),
                        direction=position.direction,
                        local_position_id=position.id,
                        broker_position_id=broker_position_id,
                        match_status="duplicate_local_broker_id",
                        message=(
                            f"Multiple local trackers claim broker position {broker_position_id}."
                        ),
                        action_required=(
                            "Do not auto-adopt or submit replacement orders; inspect the duplicate "
                            "local trackers and reconcile them against canonical broker truth."
                        ),
                    )
                )
            continue

        position = local_group[0]
        handled_local_ids.add(position.id)

        if len(broker_group) > 1:
            items.append(
                ReconciliationHealthItem(
                    status="unresolved",
                    scope="local_tracker",
                    symbol=position.symbol.upper(),
                    direction=position.direction,
                    local_position_id=position.id,
                    broker_position_id=broker_position_id,
                    match_status="duplicate_broker_position_id",
                    message=f"Broker truth returned duplicate rows for position {broker_position_id}.",
                    action_required=(
                        "Treat broker identity as unresolved and refresh broker truth before any "
                        "automatic recovery or replacement order."
                    ),
                )
            )
            continue

        match = match_broker_position(position, broker_rows)
        if match.status == "id_match":
            items.append(
                ReconciliationHealthItem(
                    status="healthy",
                    scope="local_tracker",
                    symbol=position.symbol.upper(),
                    direction=position.direction,
                    local_position_id=position.id,
                    broker_position_id=broker_position_id,
                    match_status=match.status,
                    message="Local tracker and broker position identity agree.",
                    action_required="No action required.",
                )
            )
        elif match.status == "id_not_found":
            items.append(
                ReconciliationHealthItem(
                    status="missing_broker",
                    scope="local_tracker",
                    symbol=position.symbol.upper(),
                    direction=position.direction,
                    local_position_id=position.id,
                    broker_position_id=broker_position_id,
                    match_status=match.status,
                    message=(
                        f"Local tracker {position.id} references broker position "
                        f"{broker_position_id}, but that broker position is not open."
                    ),
                    action_required=(
                        "Refresh/reconcile broker close truth before allowing a replacement order; "
                        "do not silently switch to another same-symbol broker position."
                    ),
                )
            )
        else:
            items.append(
                ReconciliationHealthItem(
                    status="unresolved",
                    scope="local_tracker",
                    symbol=position.symbol.upper(),
                    direction=position.direction,
                    local_position_id=position.id,
                    broker_position_id=broker_position_id,
                    match_status=match.status,
                    message=(
                        f"Persisted broker identity for local tracker {position.id} is ambiguous "
                        f"or conflicts with current broker truth."
                    ),
                    action_required=(
                        "Keep canonical persisted identity unchanged and resolve the broker/local "
                        "identity conflict before automatic recovery or new demo submission."
                    ),
                )
            )

    # Canonically TradeAgent-owned broker positions without a persisted positive local ID.
    for broker_position_id, (row, intent) in sorted(canonical_broker.items()):
        if broker_position_id in local_by_broker:
            continue

        symbol = str(row.get("symbol") or "").upper()
        direction = _direction(row)
        legacy_matches = [
            position
            for position in local_rows
            if position.id not in handled_local_ids
            and position.broker_position_id is None
            and position.symbol.upper() == symbol
            and position.direction == direction
            and position.timeframe.upper() == intent.timeframe.upper()
            and position.strategy == intent.strategy
        ]
        conflicting_local = [
            position
            for position in local_rows
            if position.id not in handled_local_ids
            and position.symbol.upper() == symbol
            and position.timeframe.upper() == intent.timeframe.upper()
            and position not in legacy_matches
        ]

        if len(legacy_matches) == 1:
            position = legacy_matches[0]
            handled_local_ids.add(position.id)
            items.append(
                ReconciliationHealthItem(
                    status="degraded",
                    scope="local_tracker",
                    symbol=symbol,
                    direction=direction,
                    local_position_id=position.id,
                    broker_position_id=broker_position_id,
                    intent_id=intent.id,
                    match_status="legacy_identity_unpersisted",
                    message=(
                        "A unique legacy local tracker matches the canonical TradeAgent broker "
                        "position, but the broker position ID is not persisted on the tracker."
                    ),
                    action_required=(
                        "Run normal reconciliation/recovery to attach the canonical broker position "
                        "ID; the health report itself will not mutate state."
                    ),
                )
            )
        elif len(legacy_matches) > 1 or conflicting_local:
            items.append(
                ReconciliationHealthItem(
                    status="unresolved",
                    scope="broker_position",
                    symbol=symbol,
                    direction=direction,
                    broker_position_id=broker_position_id,
                    intent_id=intent.id,
                    match_status="legacy_local_ambiguous",
                    message=(
                        "Canonical broker position has ambiguous/conflicting local tracker candidates."
                    ),
                    action_required=(
                        "Do not auto-adopt the broker position or create another demo order; inspect "
                        "the local candidates and canonical intent identity first."
                    ),
                )
            )
        else:
            items.append(
                ReconciliationHealthItem(
                    status="missing_local",
                    scope="broker_position",
                    symbol=symbol,
                    direction=direction,
                    broker_position_id=broker_position_id,
                    intent_id=intent.id,
                    match_status="canonical_broker_without_local_tracker",
                    message=(
                        "A canonical TradeAgent-managed broker position is open without a local tracker."
                    ),
                    action_required=(
                        "Run normal tracker recovery from the canonical intent/broker identity before "
                        "allowing any new demo submission for this position."
                    ),
                )
            )

    overall = _overall_status(items)
    canonical_count = len(set(broker_by_id).intersection(scoped_broker_ids))
    if not items:
        summary = "No TradeAgent-managed open demo positions currently require reconciliation."
    elif overall == "healthy":
        summary = "All TradeAgent-managed local trackers agree with current broker position identity."
    else:
        summary = (
            f"Broker/local reconciliation health is {overall}; inspect the classified items before "
            "treating demo position state as fully reconciled."
        )

    return ReconciliationHealthResponse(
        status=overall,
        broker_truth_available=True,
        broker_execution_ready=True,
        checked_at=datetime.now(UTC).replace(tzinfo=None),
        local_managed_positions=len(positive_locals),
        canonical_broker_positions=canonical_count,
        ignored_broker_positions=ignored_broker_positions,
        counts=_counts(items),
        items=items,
        summary=summary,
    )
