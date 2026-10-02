from __future__ import annotations

from math import isclose
from typing import Any, Dict

from backend.domain.models import PaperPosition
from backend.services.close_safety import attempt_verified_demo_close
from backend.storage.repositories import add_trade_audit, log_incident


def broker_protection_matches(
    position: PaperPosition,
    broker_row: Dict[str, Any],
) -> bool:
    """Return whether broker-confirmed SL/TP matches the tracked intended targets."""

    def _same(actual: object, expected: float | None) -> bool:
        if expected is None:
            return actual in (None, 0, 0.0)
        if actual in (None, 0, 0.0):
            return False
        try:
            return isclose(
                float(actual),
                float(expected),
                rel_tol=1e-9,
                abs_tol=max(1e-6, abs(float(expected)) * 1e-9),
            )
        except (TypeError, ValueError):
            return False

    return _same(broker_row.get("stop_loss"), position.stop_loss) and _same(
        broker_row.get("take_profit"), position.take_profit
    )


def fail_safe_close_unverified_demo_position(
    position: PaperPosition,
    *,
    broker_row: Dict[str, Any],
    fallback_price: float,
    protection_error: Exception,
    phase: str,
) -> Dict[str, Any]:
    """Close one canonical demo position when broker protection cannot be verified.

    A failed close retains the local tracker. The caller decides when a later
    protection retry is allowed; this helper never submits another open order or
    changes local SL/TP.
    """

    local_broker_id = int(position.broker_position_id or 0)
    row_broker_id = int(broker_row.get("position_id") or 0)
    if local_broker_id <= 0 or row_broker_id != local_broker_id:
        raise RuntimeError(
            "Protection fail-safe close blocked because the canonical broker "
            f"position id does not match: local={local_broker_id or None} "
            f"broker={row_broker_id or None}."
        )

    try:
        quantity_lots = float(broker_row.get("volume_lots") or position.quantity)
    except (TypeError, ValueError):
        quantity_lots = float(position.quantity)

    failure_kind = getattr(protection_error, "failure_kind", "protection_sync_failed")
    details = {
        "position_id": position.id,
        "broker_position_id": local_broker_id,
        "phase": phase,
        "protection_error": str(protection_error),
        "protection_failure_kind": failure_kind,
        "intended_stop_loss": position.stop_loss,
        "intended_take_profit": position.take_profit,
        "observed_stop_loss": broker_row.get("stop_loss"),
        "observed_take_profit": broker_row.get("take_profit"),
        "quantity_lots": quantity_lots,
        "action_required": (
            "Reconcile the canonical broker position/protection state. "
            "Do not submit a competing open order while protection is unresolved."
        ),
    }

    log_incident(
        "error",
        "ctrader_demo_protection_unverified",
        f"Broker protection is unverified for {position.symbol}:{position.timeframe}; fail-safe close will be attempted.",
        details,
    )
    add_trade_audit(
        event_type="ctrader_demo_protection_unverified",
        symbol=position.symbol,
        timeframe=position.timeframe,
        strategy=position.strategy,
        position_id=position.id,
        summary="Broker SL/TP could not be verified; attempting fail-safe close.",
        details=details,
    )

    close_result = attempt_verified_demo_close(
        position,
        fallback_price=fallback_price,
        reason="broker_protection_unverified_failsafe",
        phase=phase,
        quantity_lots=quantity_lots,
    )
    if not close_result.get("closed"):
        failed = {
            **details,
            "status": (
                "failsafe_close_ambiguous"
                if close_result.get("status") == "ambiguous_pending"
                else "failsafe_close_failed"
            ),
            "closed": False,
            "tracking_retained": True,
            "close_result": close_result,
        }
        log_incident(
            "error",
            "ctrader_demo_protection_failsafe_close_failed",
            f"Fail-safe close did not reach verified broker closure for unverified cTrader demo protection on {position.symbol}:{position.timeframe}.",
            failed,
        )
        add_trade_audit(
            event_type="ctrader_demo_protection_failsafe_close_failed",
            symbol=position.symbol,
            timeframe=position.timeframe,
            strategy=position.strategy,
            position_id=position.id,
            summary="Fail-safe close did not verify broker closure; retained the canonical local tracker for broker-truth recovery.",
            details=failed,
        )
        return failed

    closed_position = close_result["position"]
    succeeded = {
        **details,
        "status": "failsafe_closed",
        "closed": True,
        "tracking_retained": False,
        "broker_close": close_result.get("broker_close") or {},
        "close_result": close_result,
        "closed_position_id": closed_position.id,
    }
    log_incident(
        "warning",
        "ctrader_demo_protection_failsafe_closed",
        f"Closed cTrader demo position because broker protection could not be verified for {position.symbol}:{position.timeframe}.",
        succeeded,
    )
    add_trade_audit(
        event_type="ctrader_demo_protection_failsafe_closed",
        symbol=position.symbol,
        timeframe=position.timeframe,
        strategy=position.strategy,
        position_id=position.id,
        summary="Closed cTrader demo position because broker SL/TP could not be verified.",
        details=succeeded,
    )
    return succeeded
