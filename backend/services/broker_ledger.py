from __future__ import annotations

from typing import Any, Dict

from backend.domain.models import PaperPosition
from backend.services.broker import (
    get_broker_status,
    get_closed_position_summary,
    get_position_close_deals,
)
from backend.storage.repositories import (
    broker_realized_pnl_for_position,
    close_paper_position,
    list_broker_deals,
    list_order_intents,
    list_paper_positions,
    list_trade_audits,
    record_broker_deals,
    reconcile_closed_paper_position_from_broker,
    set_paper_position_broker_id,
    update_open_paper_position_from_broker_partial,
)


def resolve_broker_position_id(position: PaperPosition) -> int | None:
    if position.broker_position_id:
        return int(position.broker_position_id)

    for intent in list_order_intents(500):
        details = intent.details if isinstance(intent.details, dict) else {}
        try:
            opened_position_id = int(details.get("opened_position_id") or 0)
        except (TypeError, ValueError):
            opened_position_id = 0
        if opened_position_id != position.id:
            continue
        broker_order = details.get("broker_order")
        if not isinstance(broker_order, dict):
            continue
        try:
            broker_position_id = int(broker_order.get("position_id") or 0)
        except (TypeError, ValueError):
            broker_position_id = 0
        if broker_position_id > 0:
            return broker_position_id

    # Recovered trackers intentionally point back to the original intent, so
    # their broker id is recorded on the recovery audit instead of on
    # opened_position_id.
    for audit in list_trade_audits(500):
        if audit.position_id != position.id:
            continue
        details = audit.details if isinstance(audit.details, dict) else {}
        try:
            broker_position_id = int(details.get("broker_position_id") or 0)
        except (TypeError, ValueError):
            broker_position_id = 0
        if broker_position_id > 0:
            return broker_position_id

    return None


def _broker_summary_details(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "broker_exit_price": summary.get("exit_price"),
        "broker_gross_profit": summary.get("gross_profit"),
        "broker_swap": summary.get("swap"),
        "broker_commission": summary.get("commission"),
        "broker_pnl_conversion_fee": summary.get("pnl_conversion_fee"),
        "broker_net_profit": summary.get("net_profit"),
        "broker_deal_ids": summary.get("deal_ids") or [],
        "broker_closed_at": (
            summary.get("closed_at").isoformat()
            if getattr(summary.get("closed_at"), "isoformat", None)
            else summary.get("closed_at")
        ),
    }


def _persist_summary_deals(
    position: PaperPosition,
    broker_position_id: int,
    summary: Dict[str, Any] | None,
) -> Dict[str, Any]:
    if not isinstance(summary, dict):
        return {"inserted": 0, "deal_ids": [], "net_profit": 0.0, "closed_volume_lots": 0.0}
    deals = summary.get("deals")
    if not isinstance(deals, list) or not deals:
        return {"inserted": 0, "deal_ids": [], "net_profit": 0.0, "closed_volume_lots": 0.0}
    return record_broker_deals(
        local_position_id=position.id,
        broker_position_id=int(broker_position_id),
        symbol=position.symbol,
        account_currency=position.account_currency,
        deals=deals,
    )


def _tracked_initial_quantity(position: PaperPosition) -> float | None:
    """Return the quantity recorded when this local tracker was created."""
    for audit in list_trade_audits(1000):
        if audit.position_id != position.id or audit.event_type != "paper_position_opened":
            continue
        details = audit.details if isinstance(audit.details, dict) else {}
        try:
            quantity = float(details.get("quantity"))
        except (TypeError, ValueError):
            continue
        if quantity > 0:
            return quantity
    return None


def reconcile_open_position_ledger(
    position: PaperPosition,
    broker_position: Dict[str, Any],
) -> Dict[str, Any]:
    """Synchronize partial broker closes into the immutable local deal ledger.

    History is queried only when the broker reports less remaining volume than
    the local tracker. This keeps normal reconciliation lightweight while
    repeatedly retrying a partial close until its authoritative deal appears.
    """
    try:
        observed_broker_id = int(broker_position.get("position_id") or 0)
    except (TypeError, ValueError):
        observed_broker_id = 0
    canonical_broker_id = int(
        position.broker_position_id
        or resolve_broker_position_id(position)
        or 0
    )
    if (
        canonical_broker_id > 0
        and observed_broker_id > 0
        and observed_broker_id != canonical_broker_id
    ):
        return {
            "status": "identity_mismatch",
            "reason": "Broker position id does not match the canonical local tracker.",
            "canonical_broker_position_id": canonical_broker_id,
            "observed_broker_position_id": observed_broker_id,
        }

    broker_position_id = canonical_broker_id or observed_broker_id
    if broker_position_id <= 0:
        return {"status": "unavailable", "reason": "Broker position id is unavailable."}

    broker_symbol = str(broker_position.get("symbol") or "").strip().upper()
    broker_side = str(broker_position.get("direction") or "").strip().lower()
    normalized_direction = {
        "buy": "long",
        "long": "long",
        "sell": "short",
        "short": "short",
    }.get(broker_side)
    if (
        (broker_symbol and broker_symbol != position.symbol.upper())
        or (normalized_direction and normalized_direction != position.direction)
    ):
        return {
            "status": "identity_mismatch",
            "reason": "Broker symbol/direction does not match the canonical local tracker.",
            "broker_position_id": broker_position_id,
            "local_symbol": position.symbol.upper(),
            "broker_symbol": broker_symbol or None,
            "local_direction": position.direction,
            "broker_direction": normalized_direction,
        }

    try:
        remaining_quantity = float(broker_position.get("volume_lots"))
    except (TypeError, ValueError):
        return {"status": "unavailable", "reason": "Broker remaining quantity is unavailable."}
    if remaining_quantity <= 0:
        return {"status": "unavailable", "reason": "Broker remaining quantity is not positive."}

    local_quantity = float(position.quantity)
    tolerance = 1e-8
    reduction = local_quantity - remaining_quantity
    if remaining_quantity > local_quantity + tolerance:
        return {
            "status": "volume_increase_mismatch",
            "broker_position_id": broker_position_id,
            "local_quantity": local_quantity,
            "remaining_quantity": remaining_quantity,
        }
    if reduction <= tolerance:
        return {
            "status": "in_sync",
            "broker_position_id": broker_position_id,
            "remaining_quantity": remaining_quantity,
            "realized_pnl": float(position.realized_pnl),
            "inserted_deals": 0,
            "deal_ids": [],
        }

    before_rows = list_broker_deals(local_position_id=position.id)
    before_closed_lots = sum(float(row.get("closed_volume_lots") or 0.0) for row in before_rows)

    deals = get_position_close_deals(
        broker_position_id,
        symbol=position.symbol,
        opened_at_hint=position.opened_at,
    )
    recorded = record_broker_deals(
        local_position_id=position.id,
        broker_position_id=broker_position_id,
        symbol=position.symbol,
        account_currency=position.account_currency,
        deals=deals,
    )

    all_rows = list_broker_deals(local_position_id=position.id)
    total_closed_lots = sum(float(row.get("closed_volume_lots") or 0.0) for row in all_rows)
    realized_pnl = broker_realized_pnl_for_position(position.id)
    newly_closed_lots = max(0.0, total_closed_lots - before_closed_lots)

    tracked_initial_quantity = _tracked_initial_quantity(position)
    if (
        tracked_initial_quantity is not None
        and tracked_initial_quantity + tolerance >= local_quantity
    ):
        required_closed_lots = max(0.0, tracked_initial_quantity - remaining_quantity)
    else:
        # Fail closed when the tracker's original quantity cannot be recovered:
        # require the already-booked volume plus the newly observed reduction.
        required_closed_lots = before_closed_lots + reduction

    if total_closed_lots + tolerance < required_closed_lots:
        return {
            "status": "pending_deal_history",
            "broker_position_id": broker_position_id,
            "local_quantity": local_quantity,
            "remaining_quantity": remaining_quantity,
            "observed_reduction": reduction,
            "newly_closed_lots": newly_closed_lots,
            "total_closed_lots": total_closed_lots,
            "required_closed_lots": required_closed_lots,
            "tracked_initial_quantity": tracked_initial_quantity,
            "inserted_deals": recorded["inserted"],
            "deal_ids": recorded["deal_ids"],
            "action_required": (
                "Keep the canonical tracker open and retry broker deal-history reconciliation; "
                "do not synthesize realized P&L or treat the residual broker position as closed."
            ),
        }

    updated = update_open_paper_position_from_broker_partial(
        position.id,
        remaining_quantity=remaining_quantity,
        realized_pnl=realized_pnl,
        broker_position_id=broker_position_id,
    )
    return {
        "status": "partial_close_synced",
        "broker_position_id": broker_position_id,
        "previous_quantity": local_quantity,
        "remaining_quantity": float(updated.quantity),
        "realized_pnl": float(updated.realized_pnl),
        "inserted_deals": recorded["inserted"],
        "deal_ids": recorded["deal_ids"],
        "total_closed_lots": total_closed_lots,
        "required_closed_lots": required_closed_lots,
        "tracked_initial_quantity": tracked_initial_quantity,
    }

def close_local_position_from_broker(
    position: PaperPosition,
    *,
    fallback_price: float,
    fallback_reason: str,
) -> PaperPosition:
    broker_position_id = resolve_broker_position_id(position)
    if broker_position_id and position.broker_position_id != broker_position_id:
        position = set_paper_position_broker_id(position.id, broker_position_id)

    if broker_position_id:
        try:
            summary = get_closed_position_summary(
                broker_position_id,
                closed_at_hint=position.closed_at,
                symbol=position.symbol,
            )
        except Exception:
            summary = None
        if summary and summary.get("exit_price") is not None:
            _persist_summary_deals(position, broker_position_id, summary)
            return close_paper_position(
                position.id,
                float(summary["exit_price"]),
                fallback_reason,
                realized_pnl_override=float(summary.get("net_profit") or 0.0),
                closed_at_override=summary.get("closed_at"),
                realized_pnl_source="ctrader_deal",
            )

    return close_paper_position(
        position.id,
        fallback_price,
        fallback_reason,
        realized_pnl_source="paper_estimate",
    )


def close_local_position_after_broker_close(
    position: PaperPosition,
    *,
    broker_close: Dict[str, Any] | None,
    fallback_price: float,
    reason: str,
) -> PaperPosition:
    broker_close = broker_close if isinstance(broker_close, dict) else {}
    if broker_close.get("status") != "closed" or broker_close.get("verified") is not True:
        raise RuntimeError(
            "Local cTrader tracker close blocked because broker close is not verified."
        )

    canonical_broker_id = int(
        position.broker_position_id
        or resolve_broker_position_id(position)
        or 0
    )
    broker_position_id = int(broker_close.get("position_id") or 0)
    if canonical_broker_id <= 0 or broker_position_id != canonical_broker_id:
        raise RuntimeError(
            "Local cTrader tracker close blocked because broker position identity does not match: "
            f"local={canonical_broker_id or None} broker={broker_position_id or None}."
        )

    summary = broker_close.get("close_summary")
    # The real close adapter always includes close_summary (possibly None).
    # Do not issue a second historical request for mocked/legacy close payloads.
    if not isinstance(summary, dict) and not broker_close and broker_position_id:
        try:
            summary = get_closed_position_summary(
                broker_position_id,
                closed_at_hint=position.closed_at,
                symbol=position.symbol,
            )
        except Exception:
            summary = None

    if isinstance(summary, dict) and summary.get("exit_price") is not None:
        _persist_summary_deals(position, broker_position_id, summary)
        return close_paper_position(
            position.id,
            float(summary["exit_price"]),
            reason,
            realized_pnl_override=float(summary.get("net_profit") or 0.0),
            closed_at_override=summary.get("closed_at"),
            realized_pnl_source="ctrader_deal",
        )

    return close_paper_position(
        position.id,
        fallback_price,
        reason,
        realized_pnl_source="paper_estimate",
    )


def reconcile_closed_history(limit: int = 100) -> Dict[str, Any]:
    status = get_broker_status()
    if not status.execution_ready:
        return {
            "checked": 0,
            "reconciled": 0,
            "missing_broker_id": 0,
            "unavailable": 0,
            "duplicate_broker_id": 0,
            "ready": False,
            "reason": "cTrader account is not execution-ready yet.",
            "errors": [],
        }

    positions = [
        position
        for position in list_paper_positions("closed")[: max(1, int(limit))]
        if position.realized_pnl_source != "ctrader_deal"
    ]
    checked = 0
    reconciled = 0
    missing_broker_id = 0
    unavailable = 0
    duplicate_broker_id = 0
    errors: list[Dict[str, Any]] = []
    seen_broker_ids: set[int] = set()

    for position in positions:
        checked += 1
        broker_position_id = resolve_broker_position_id(position)
        if not broker_position_id:
            missing_broker_id += 1
            continue
        if broker_position_id in seen_broker_ids:
            duplicate_broker_id += 1
            continue
        seen_broker_ids.add(broker_position_id)

        try:
            summary = get_closed_position_summary(
                broker_position_id,
                closed_at_hint=position.closed_at,
                symbol=position.symbol,
            )
        except Exception as exc:
            unavailable += 1
            errors.append(
                {
                    "position_id": position.id,
                    "broker_position_id": broker_position_id,
                    "symbol": position.symbol,
                    "error": str(exc),
                }
            )
            continue
        if not summary or summary.get("exit_price") is None:
            unavailable += 1
            errors.append(
                {
                    "position_id": position.id,
                    "broker_position_id": broker_position_id,
                    "symbol": position.symbol,
                    "error": "No closing cTrader deal was returned for this position.",
                }
            )
            continue

        _persist_summary_deals(position, broker_position_id, summary)
        reconcile_closed_paper_position_from_broker(
            position.id,
            exit_price=float(summary["exit_price"]),
            realized_pnl=float(summary.get("net_profit") or 0.0),
            closed_at=summary.get("closed_at"),
            broker_position_id=broker_position_id,
            broker_details=_broker_summary_details(summary),
        )
        reconciled += 1

    return {
        "checked": checked,
        "reconciled": reconciled,
        "missing_broker_id": missing_broker_id,
        "unavailable": unavailable,
        "duplicate_broker_id": duplicate_broker_id,
        "ready": True,
        "reason": "",
        "errors": errors[:10],
    }


# Backward-compatible wrappers for pre-Phase-10.4 callers/tests.
def reconcile_open_demo_position_ledger(
    position: PaperPosition,
    broker_position: Dict[str, Any],
) -> Dict[str, Any]:
    return reconcile_open_position_ledger(position, broker_position)


def reconcile_closed_demo_history(limit: int = 100) -> Dict[str, Any]:
    return reconcile_closed_history(limit)
