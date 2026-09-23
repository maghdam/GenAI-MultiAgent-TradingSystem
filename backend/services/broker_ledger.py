from __future__ import annotations

from typing import Any, Dict

from backend.domain.models import PaperPosition
from backend.services.broker import get_closed_position_summary
from backend.storage.repositories import (
    close_paper_position,
    list_order_intents,
    list_paper_positions,
    list_trade_audits,
    reconcile_closed_paper_position_from_broker,
    set_paper_position_broker_id,
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
            summary = get_closed_position_summary(broker_position_id)
        except Exception:
            summary = None
        if summary and summary.get("exit_price") is not None:
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
    broker_position_id = int(
        broker_close.get("position_id")
        or position.broker_position_id
        or resolve_broker_position_id(position)
        or 0
    )
    if broker_position_id and position.broker_position_id != broker_position_id:
        position = set_paper_position_broker_id(position.id, broker_position_id)

    summary = broker_close.get("close_summary")
    # The real close adapter always includes close_summary (possibly None).
    # Do not issue a second historical request for mocked/legacy close payloads.
    if not isinstance(summary, dict) and not broker_close and broker_position_id:
        try:
            summary = get_closed_position_summary(broker_position_id)
        except Exception:
            summary = None

    if isinstance(summary, dict) and summary.get("exit_price") is not None:
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


def reconcile_closed_demo_history(limit: int = 100) -> Dict[str, Any]:
    positions = [
        position
        for position in list_paper_positions("closed")[: max(1, int(limit))]
        if position.realized_pnl_source != "ctrader_deal"
    ]
    checked = 0
    reconciled = 0
    missing_broker_id = 0
    unavailable = 0

    for position in positions:
        checked += 1
        broker_position_id = resolve_broker_position_id(position)
        if not broker_position_id:
            missing_broker_id += 1
            continue

        try:
            summary = get_closed_position_summary(broker_position_id)
        except Exception:
            unavailable += 1
            continue
        if not summary or summary.get("exit_price") is None:
            unavailable += 1
            continue

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
    }
