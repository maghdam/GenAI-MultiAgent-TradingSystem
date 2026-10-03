from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

from backend.domain.models import (
    BrokerStatus,
    PaperPosition,
    ProtectionHealthItem,
    ProtectionHealthResponse,
)
from backend.services.broker import get_broker_status, list_positions
from backend.services.broker_position_match import match_broker_position
from backend.services.position_truth import extract_broker_protection
from backend.services.reconciler import _is_canonical_tradeagent_recovery_intent
from backend.storage.repositories import list_order_intents, list_paper_positions


_STATUS_PRIORITY = {
    "fully_protected": 0,
    "partially_protected": 1,
    "unprotected": 2,
    "unavailable": 3,
    "identity_unresolved": 4,
}


def _direction(row: dict[str, Any]) -> str:
    return "long" if str(row.get("direction") or "").lower() == "buy" else "short"


def _broker_id(row: dict[str, Any]) -> int | None:
    try:
        value = int(row.get("position_id") or 0)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _health_status(protection_status: str) -> str:
    return {
        "protected": "fully_protected",
        "partial": "partially_protected",
        "unprotected": "unprotected",
    }[protection_status]


def _overall_status(items: list[ProtectionHealthItem]) -> str:
    if not items:
        return "fully_protected"
    return max(items, key=lambda item: _STATUS_PRIORITY[item.status]).status


def _counts(items: list[ProtectionHealthItem]) -> dict[str, int]:
    counts = Counter(item.status for item in items)
    return {
        status: int(counts.get(status, 0))
        for status in _STATUS_PRIORITY
    }


def _response(
    *,
    broker: BrokerStatus,
    broker_truth_available: bool,
    items: list[ProtectionHealthItem],
    ignored_broker_positions: int,
    summary: str,
    status_override: str | None = None,
    managed_positions_override: int | None = None,
) -> ProtectionHealthResponse:
    counts = _counts(items)
    assessable = (
        counts["fully_protected"]
        + counts["partially_protected"]
        + counts["unprotected"]
    )
    fully = counts["fully_protected"]
    coverage = (fully / assessable * 100.0) if assessable else None
    return ProtectionHealthResponse(
        status=status_override or _overall_status(items),
        broker_truth_available=broker_truth_available,
        broker_execution_ready=bool(broker.execution_ready),
        checked_at=datetime.now(UTC).replace(tzinfo=None),
        managed_positions=(
            len(items)
            if managed_positions_override is None
            else managed_positions_override
        ),
        assessable_positions=assessable,
        fully_protected_positions=fully,
        full_protection_coverage_pct=coverage,
        ignored_broker_positions=ignored_broker_positions,
        counts=counts,
        items=items,
        summary=summary,
    )


def _unavailable_response(
    *,
    broker: BrokerStatus,
    local_rows: list[PaperPosition],
    reason: str,
) -> ProtectionHealthResponse:
    tracked = [
        position
        for position in local_rows
        if int(position.broker_position_id or 0) > 0
    ]
    items = [
        ProtectionHealthItem(
            status="unavailable",
            scope="local_tracker",
            symbol=position.symbol.upper(),
            direction=position.direction,
            local_position_id=position.id,
            broker_position_id=int(position.broker_position_id or 0),
            broker_sync_status="broker_truth_unavailable",
            message=(
                f"Broker protection truth is unavailable for tracked position {position.id} "
                f"({position.symbol}:{position.timeframe})."
            ),
            action_required=(
                "Restore the confirmed cTrader demo broker connection/readiness and refresh "
                "protection health before treating this position as broker-protected."
            ),
        )
        for position in tracked
    ]
    return _response(
        broker=broker,
        broker_truth_available=False,
        items=items,
        ignored_broker_positions=0,
        summary=reason,
        status_override="unavailable",
        managed_positions_override=len(tracked),
    )


def _protection_item(
    *,
    row: dict[str, Any],
    scope: str,
    local_position_id: int | None,
    broker_position_id: int,
    intent_id: int | None,
    broker_sync_status: str,
) -> ProtectionHealthItem:
    stop_loss, take_profit, protection_status = extract_broker_protection(row)
    status = _health_status(protection_status)
    symbol = str(row.get("symbol") or "").upper()
    direction = _direction(row)

    if status == "fully_protected":
        message = "Broker truth confirms both stop-loss and take-profit protection."
        action_required = "No action required."
    elif status == "partially_protected":
        message = "Broker truth confirms only one protective target."
        action_required = (
            "Use the existing reconciliation/protection-sync path to restore the missing broker "
            "target; do not infer full protection from local requested SL/TP values."
        )
    else:
        message = "Broker truth confirms no stop-loss or take-profit on the open position."
        action_required = (
            "Treat the position as unprotected and use the existing protection repair/fail-safe "
            "path; this report will not amend or close the broker position."
        )

    return ProtectionHealthItem(
        status=status,
        scope=scope,
        symbol=symbol,
        direction=direction,
        local_position_id=local_position_id,
        broker_position_id=broker_position_id,
        intent_id=intent_id,
        broker_sync_status=broker_sync_status,
        broker_stop_loss=stop_loss,
        broker_take_profit=take_profit,
        message=message,
        action_required=action_required,
    )


def build_protection_health() -> ProtectionHealthResponse:
    """Report read-only broker protection health using canonical identity.

    Local requested SL/TP values are never treated as proof of broker protection.
    The function does not reconcile, amend, close, adopt, or submit broker orders.
    """

    local_rows = list_paper_positions("open")
    intents = list_order_intents(500)
    broker = get_broker_status()

    if not broker.execution_ready:
        return _unavailable_response(
            broker=broker,
            local_rows=local_rows,
            reason=(
                "cTrader demo protection truth is unavailable; protection health cannot be "
                "verified without a ready confirmed demo connection."
            ),
        )

    try:
        broker_rows = list(list_positions())
    except Exception as exc:
        return _unavailable_response(
            broker=broker,
            local_rows=local_rows,
            reason=f"Could not read cTrader demo positions for protection health: {exc}",
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

    items: list[ProtectionHealthItem] = []
    handled_local_ids: set[int] = set()

    for broker_position_id, local_group in sorted(local_by_broker.items()):
        broker_group = broker_by_id.get(broker_position_id, [])

        if len(local_group) > 1:
            for position in local_group:
                handled_local_ids.add(position.id)
                items.append(
                    ProtectionHealthItem(
                        status="identity_unresolved",
                        scope="local_tracker",
                        symbol=position.symbol.upper(),
                        direction=position.direction,
                        local_position_id=position.id,
                        broker_position_id=broker_position_id,
                        broker_sync_status="duplicate_local_broker_id",
                        message=(
                            f"Multiple local trackers claim broker position {broker_position_id}; "
                            "broker protection cannot be attributed safely."
                        ),
                        action_required=(
                            "Resolve canonical broker/local identity before treating protection as verified."
                        ),
                    )
                )
            continue

        position = local_group[0]
        handled_local_ids.add(position.id)

        if len(broker_group) > 1:
            items.append(
                ProtectionHealthItem(
                    status="identity_unresolved",
                    scope="local_tracker",
                    symbol=position.symbol.upper(),
                    direction=position.direction,
                    local_position_id=position.id,
                    broker_position_id=broker_position_id,
                    broker_sync_status="duplicate_broker_position_id",
                    message=(
                        f"Broker truth returned duplicate rows for position {broker_position_id}; "
                        "protection identity is unresolved."
                    ),
                    action_required=(
                        "Refresh and resolve broker identity before treating SL/TP state as verified."
                    ),
                )
            )
            continue

        match = match_broker_position(position, broker_rows)
        if not match.matched:
            items.append(
                ProtectionHealthItem(
                    status="identity_unresolved",
                    scope="local_tracker",
                    symbol=position.symbol.upper(),
                    direction=position.direction,
                    local_position_id=position.id,
                    broker_position_id=broker_position_id,
                    broker_sync_status=match.status,
                    message=(
                        "Persisted broker identity is missing or conflicts with current broker truth; "
                        "protection cannot be verified safely."
                    ),
                    action_required=(
                        "Keep the persisted broker position identity unchanged and resolve broker/local "
                        "identity before evaluating protection."
                    ),
                )
            )
            continue

        items.append(
            _protection_item(
                row=match.row or {},
                scope="local_tracker",
                local_position_id=position.id,
                broker_position_id=broker_position_id,
                intent_id=None,
                broker_sync_status=match.status,
            )
        )

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

        if legacy_matches or conflicting_local:
            local_id = legacy_matches[0].id if len(legacy_matches) == 1 and not conflicting_local else None
            stop_loss, take_profit, _ = extract_broker_protection(row)
            items.append(
                ProtectionHealthItem(
                    status="identity_unresolved",
                    scope="local_tracker" if local_id is not None else "broker_position",
                    symbol=symbol,
                    direction=direction,
                    local_position_id=local_id,
                    broker_position_id=broker_position_id,
                    intent_id=intent.id,
                    broker_sync_status=(
                        "legacy_identity_unpersisted"
                        if local_id is not None
                        else "legacy_local_ambiguous"
                    ),
                    broker_stop_loss=stop_loss,
                    broker_take_profit=take_profit,
                    message=(
                        "Broker protection values are visible, but broker/local tracker identity is "
                        "not durably resolved."
                    ),
                    action_required=(
                        "Run the normal identity recovery/reconciliation path first; do not treat "
                        "visible SL/TP values as verified protection for an unresolved local identity."
                    ),
                )
            )
            continue

        items.append(
            _protection_item(
                row=row,
                scope="broker_position",
                local_position_id=None,
                broker_position_id=broker_position_id,
                intent_id=intent.id,
                broker_sync_status="canonical_intent_match",
            )
        )

    overall = _overall_status(items)
    if not items:
        summary = "No TradeAgent-managed open demo positions currently require protection measurement."
    elif overall == "fully_protected":
        summary = "All assessable TradeAgent-managed demo positions are fully protected at the broker."
    else:
        summary = (
            f"Protection health is {overall}; inspect per-position broker truth before treating "
            "all managed demo positions as fully protected."
        )

    return _response(
        broker=broker,
        broker_truth_available=True,
        items=items,
        ignored_broker_positions=ignored_broker_positions,
        summary=summary,
    )
