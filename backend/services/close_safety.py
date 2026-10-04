from __future__ import annotations

from typing import Any, Dict

from backend.domain.models import PaperEvent, PaperPosition
from backend.services.broker import (
    CTraderCloseOutcomeAmbiguous,
    CTraderCloseRejected,
    close_position,
    get_closed_position_summary,
    list_positions,
)
from backend.services.broker_ledger import close_local_position_after_broker_close
from backend.storage.repositories import (
    add_paper_event,
    add_trade_audit,
    list_paper_events,
    log_incident,
)


_AMBIGUOUS_EVENT = "ctrader_demo_close_ambiguous"
_RESOLVED_EVENT = "ctrader_demo_close_reconciled"


def _event_matches_position(event: PaperEvent, position: PaperPosition) -> bool:
    details = event.details if isinstance(event.details, dict) else {}
    try:
        local_id = int(details.get("position_id") or 0)
    except (TypeError, ValueError):
        local_id = 0
    try:
        broker_id = int(details.get("broker_position_id") or 0)
    except (TypeError, ValueError):
        broker_id = 0
    return (
        local_id == int(position.id)
        and broker_id == int(position.broker_position_id or 0)
    )


def unresolved_close_event(position: PaperPosition) -> PaperEvent | None:
    if not position.broker_position_id:
        return None
    for event in list_paper_events(500):
        if not _event_matches_position(event, position):
            continue
        if event.event_type in {_RESOLVED_EVENT, "ctrader_demo_close_verified"}:
            return None
        if event.event_type in {_AMBIGUOUS_EVENT, "ctrader_demo_close_rejected"}:
            return event
    return None


def _canonical_broker_row(position: PaperPosition) -> Dict[str, Any] | None:
    broker_id = int(position.broker_position_id or 0)
    if broker_id <= 0:
        raise RuntimeError(
            f"Tracked cTrader position {position.id} has no canonical broker position id."
        )
    for row in list_positions() or []:
        try:
            row_id = int(row.get("position_id") or 0)
        except (TypeError, ValueError):
            continue
        if row_id == broker_id:
            return dict(row)
    return None


def _verified_absence_payload(
    position: PaperPosition,
    *,
    quantity_lots: float,
) -> Dict[str, Any]:
    broker_id = int(position.broker_position_id or 0)
    close_summary = None
    try:
        close_summary = get_closed_position_summary(
            broker_id,
            closed_at_hint=position.closed_at,
            symbol=position.symbol,
        )
    except Exception:
        close_summary = None
    return {
        "status": "closed",
        "symbol": position.symbol,
        "position_id": broker_id,
        "quantity_lots": float(quantity_lots),
        "verified": True,
        "ack": {},
        "reconciled_from_broker": True,
        "close_summary": close_summary,
    }


def record_ambiguous_close(
    position: PaperPosition,
    *,
    error: CTraderCloseOutcomeAmbiguous,
    phase: str,
    quantity_lots: float,
) -> None:
    details = {
        "position_id": position.id,
        "broker_position_id": int(position.broker_position_id or 0),
        "symbol": position.symbol,
        "timeframe": position.timeframe,
        "phase": phase,
        "quantity_lots": float(quantity_lots),
        "outcome_state": "ambiguous_post_submit",
        "failure_kind": error.failure_kind,
        "submission_may_have_succeeded": True,
        "automatic_retry": False,
        "tracking_retained": True,
        "error": str(error),
        "ack": error.ack,
        "action_required": (
            "Reconcile the canonical broker position. Do not send another close "
            "while the post-submission close outcome is unresolved."
        ),
    }
    add_paper_event(
        _AMBIGUOUS_EVENT,
        "cTrader close outcome is ambiguous after broker submission.",
        details,
    )
    log_incident(
        "error",
        "ctrader_demo_close_ambiguous",
        f"cTrader close outcome is ambiguous for {position.symbol}:{position.timeframe}; automatic re-close is blocked.",
        details,
    )
    add_trade_audit(
        event_type="ctrader_demo_close_ambiguous",
        symbol=position.symbol,
        timeframe=position.timeframe,
        strategy=position.strategy,
        position_id=position.id,
        summary="Close was submitted but broker outcome is ambiguous; retained canonical tracking and blocked resubmission.",
        details=details,
    )


def record_rejected_close(
    position: PaperPosition,
    *,
    error: CTraderCloseRejected,
    phase: str,
    quantity_lots: float,
) -> None:
    details = {
        "position_id": position.id,
        "broker_position_id": int(position.broker_position_id or 0),
        "symbol": position.symbol,
        "timeframe": position.timeframe,
        "phase": phase,
        "quantity_lots": float(quantity_lots),
        "outcome_state": "broker_rejected",
        "submission_may_have_succeeded": False,
        "automatic_retry": False,
        "tracking_retained": True,
        "error": str(error),
        "ack": error.ack,
        "action_required": (
            "Inspect the broker rejection and reconcile the canonical position "
            "before retrying on a later trading cycle."
        ),
    }
    add_paper_event(
        "ctrader_demo_close_rejected",
        "Broker explicitly rejected a cTrader close request.",
        details,
    )
    log_incident(
        "error",
        "ctrader_demo_close_rejected",
        f"Broker rejected the cTrader close for {position.symbol}:{position.timeframe}; local tracking remains open.",
        details,
    )
    add_trade_audit(
        event_type="ctrader_demo_close_rejected",
        symbol=position.symbol,
        timeframe=position.timeframe,
        strategy=position.strategy,
        position_id=position.id,
        summary="Broker rejected the close; local tracker remained open.",
        details=details,
    )


def attempt_verified_close(
    position: PaperPosition,
    *,
    fallback_price: float,
    reason: str,
    phase: str,
    quantity_lots: float | None = None,
) -> Dict[str, Any]:
    """Close one tracked cTrader position without ever synthesizing broker truth."""

    broker_id = int(position.broker_position_id or 0)
    if broker_id <= 0:
        raise RuntimeError(
            f"cTrader close blocked: tracked position {position.id} has no canonical broker id."
        )
    quantity = float(quantity_lots or position.quantity)

    unresolved = unresolved_close_event(position)
    if unresolved is not None:
        prior_ambiguous = unresolved.event_type == _AMBIGUOUS_EVENT
        pending_status = "ambiguous_pending" if prior_ambiguous else "rejected_pending"
        try:
            broker_row = _canonical_broker_row(position)
        except Exception as exc:
            details = {
                "position_id": position.id,
                "broker_position_id": broker_id,
                "phase": phase,
                "prior_close_event": unresolved.event_type,
                "error": str(exc),
                "automatic_retry": False,
                "tracking_retained": True,
                "close_request_sent": False,
            }
            log_incident(
                "error",
                "ctrader_demo_close_reconciliation_deferred",
                f"Could not reconcile prior cTrader close failure for {position.symbol}:{position.timeframe}.",
                details,
            )
            return {
                "status": pending_status,
                "closed": False,
                "retryable": False,
                **details,
            }

        if broker_row is not None:
            details = {
                "position_id": position.id,
                "broker_position_id": broker_id,
                "phase": phase,
                "prior_close_event": unresolved.event_type,
                "automatic_retry": False,
                "tracking_retained": True,
                "close_request_sent": False,
                "broker_position_still_open": True,
            }
            log_incident(
                "warning",
                (
                    "ctrader_demo_close_ambiguity_pending"
                    if prior_ambiguous
                    else "ctrader_demo_close_rejection_pending"
                ),
                (
                    f"Canonical broker position is still open after an ambiguous close for {position.symbol}:{position.timeframe}; no second close was sent."
                    if prior_ambiguous
                    else f"Canonical broker position is still open after a rejected close for {position.symbol}:{position.timeframe}; automatic re-close remains blocked."
                ),
                details,
            )
            return {
                "status": pending_status,
                "closed": False,
                "retryable": False,
                **details,
            }

        broker_close = _verified_absence_payload(position, quantity_lots=quantity)
        closed = close_local_position_after_broker_close(
            position,
            broker_close=broker_close,
            fallback_price=fallback_price,
            reason=reason,
        )
        details = {
            "position_id": position.id,
            "closed_position_id": closed.id,
            "broker_position_id": broker_id,
            "phase": phase,
            "automatic_retry": False,
            "close_request_sent": False,
            "reconciled_from_broker": True,
            "broker_close": broker_close,
        }
        add_paper_event(
            _RESOLVED_EVENT,
            "Resolved ambiguous cTrader close from broker position absence.",
            details,
        )
        log_incident(
            "warning",
            "ctrader_demo_close_reconciled",
            f"Resolved ambiguous close from broker truth for {position.symbol}:{position.timeframe}; local tracker is now closed.",
            details,
        )
        add_trade_audit(
            event_type="ctrader_demo_close_reconciled",
            symbol=position.symbol,
            timeframe=position.timeframe,
            strategy=position.strategy,
            position_id=closed.id,
            summary="Broker position absence resolved the prior ambiguous close without resubmitting.",
            details=details,
        )
        return {
            "status": "closed",
            "closed": True,
            "retryable": False,
            "position": closed,
            **details,
        }

    try:
        broker_close = close_position(
            symbol=position.symbol,
            position_id=broker_id,
            quantity_lots=quantity,
        )
    except CTraderCloseRejected as exc:
        record_rejected_close(
            position,
            error=exc,
            phase=phase,
            quantity_lots=quantity,
        )
        return {
            "status": "rejected",
            "closed": False,
            "retryable": False,
            "position_id": position.id,
            "broker_position_id": broker_id,
            "tracking_retained": True,
            "close_request_sent": True,
        }
    except CTraderCloseOutcomeAmbiguous as exc:
        record_ambiguous_close(
            position,
            error=exc,
            phase=phase,
            quantity_lots=quantity,
        )
        return {
            "status": "ambiguous_pending",
            "closed": False,
            "retryable": False,
            "position_id": position.id,
            "broker_position_id": broker_id,
            "tracking_retained": True,
            "close_request_sent": True,
            "failure_kind": exc.failure_kind,
        }
    except Exception as exc:
        details = {
            "position_id": position.id,
            "broker_position_id": broker_id,
            "phase": phase,
            "error": str(exc),
            "tracking_retained": True,
            "close_request_sent": False,
        }
        log_incident(
            "error",
            "ctrader_demo_close_failed",
            f"cTrader close failed before a verified broker close for {position.symbol}:{position.timeframe}.",
            details,
        )
        add_trade_audit(
            event_type="ctrader_demo_close_failed",
            symbol=position.symbol,
            timeframe=position.timeframe,
            strategy=position.strategy,
            position_id=position.id,
            summary="cTrader close failed; local tracker remained open.",
            details=details,
        )
        return {
            "status": "failed",
            "closed": False,
            "retryable": True,
            **details,
        }

    closed = close_local_position_after_broker_close(
        position,
        broker_close=broker_close,
        fallback_price=fallback_price,
        reason=reason,
    )
    details = {
        "position_id": position.id,
        "closed_position_id": closed.id,
        "broker_position_id": broker_id,
        "phase": phase,
        "automatic_retry": False,
        "close_request_sent": True,
        "broker_close": broker_close,
    }
    add_paper_event(
        "ctrader_demo_close_verified",
        "Broker-confirmed cTrader close completed.",
        details,
    )
    return {
        "status": "closed",
        "closed": True,
        "retryable": False,
        "position": closed,
        **details,
    }


# Backward-compatible aliases for pre-Phase-10.4 callers/tests.
unresolved_demo_close_event = unresolved_close_event
record_ambiguous_demo_close = record_ambiguous_close
record_rejected_demo_close = record_rejected_close
attempt_verified_demo_close = attempt_verified_close
