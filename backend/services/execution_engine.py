from __future__ import annotations

from dataclasses import dataclass
from time import sleep
from uuid import uuid4

from backend.domain.models import EngineConfig, PaperPosition, StrategyAnalysis, WatchlistItem
from backend.services.broker import (
    CTraderCloseOutcomeAmbiguous,
    CTraderCloseRejected,
    CTraderOrderAcknowledgementTimeout,
    CTraderProtectionSyncFailure,
    close_position,
    get_broker_account_snapshot,
    get_broker_status,
    get_symbol_execution_readiness,
    get_instrument_spec,
    list_positions,
    place_market_order,
    sync_position_targets,
)
from backend.services.paper_book import apply_mark, reconcile_position
from backend.services.broker_ledger import (
    close_local_position_from_broker,
    reconcile_open_position_ledger,
)
from backend.services.broker_position_match import match_broker_position
from backend.services.financial_units import resolve_monetary_basis
from backend.services.live_trading_guard import (
    get_live_trading_armed_account_id,
    live_entry_block_reason,
)
from backend.services.quantity_rules import derive_auto_quantity, evaluate_order_quantity
from backend.services.close_safety import (
    attempt_verified_close,
    record_ambiguous_close,
    record_rejected_close,
)
from backend.services.protection_safety import fail_safe_close_unverified_position
from backend.services.risk_engine import evaluate_risk
from backend.storage.db import SQLiteBusyError
from backend.storage.repositories import (
    add_trade_audit,
    close_paper_position,
    create_decision_record,
    create_order_intent,
    get_open_position,
    list_order_intents,
    list_paper_positions,
    log_incident,
    open_paper_position,
    update_order_intent_status,
    update_paper_position_targets,
    set_paper_position_broker_id,
)


@dataclass
class ExecutionResult:
    action_taken: bool
    intent_id: int | None
    status: str
    summary: str
    position_id: int | None = None
    mode: str = "paper_only"
    broker_position_id: int | None = None
    retryable: bool = False


class BrokerPositionIdentityError(RuntimeError):
    pass


def _ctrader_account_type() -> str:
    account_type = str(getattr(get_broker_status(), "account_type", "unknown") or "unknown").lower()
    return account_type if account_type in {"demo", "live"} else "unknown"


def _ctrader_account_type_from_order(broker_order: dict[str, object] | None = None) -> str:
    if broker_order:
        account_type = str(broker_order.get("account_type") or "").lower()
        if account_type in {"demo", "live"}:
            return account_type
    return _ctrader_account_type()


def _ctrader_execution_mode(broker_order: dict[str, object] | None = None) -> str:
    return "live_enabled" if _ctrader_account_type_from_order(broker_order) == "live" else "demo_enabled"


def _ctrader_execution_tag(broker_order: dict[str, object] | None = None) -> str:
    account_type = _ctrader_account_type_from_order(broker_order)
    return f"ctrader_{account_type}" if account_type in {"demo", "live"} else "ctrader_unknown"


def _canonical_broker_row_for_position(position: PaperPosition) -> dict[str, object]:
    match = match_broker_position(position, list_positions())
    if match.status in {"id_mismatch", "legacy_ambiguous"} or not match.matched:
        raise BrokerPositionIdentityError(
            f"Could not safely identify canonical broker position for {position.symbol}:{position.timeframe}."
        )
    row = dict(match.row or {})
    row_id = _broker_position_id(row)
    if row_id is None:
        raise BrokerPositionIdentityError(
            f"Canonical broker position for {position.symbol}:{position.timeframe} has no valid id."
        )
    if position.broker_position_id is not None and int(position.broker_position_id) != row_id:
        raise BrokerPositionIdentityError(
            f"Canonical broker position id changed for {position.symbol}:{position.timeframe}."
        )
    return row


def _broker_position_id(row: dict[str, object]) -> int | None:
    try:
        value = int(row.get("position_id") or 0)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _ctrader_new_entry_inventory_gate() -> tuple[bool, dict[str, object], str | None]:
    """Fail closed when broker exposure is not fully represented by local trackers."""
    try:
        broker_rows = [dict(row) for row in (list_positions() or [])]
    except Exception as exc:
        details = {
            "broker_inventory_available": False,
            "broker_submission_suppressed": True,
            "automatic_adoption": False,
            "error": str(exc),
        }
        return (
            False,
            details,
            "cTrader broker position inventory is unavailable; new entries are blocked until broker truth can be verified.",
        )

    try:
        local_rows = list_paper_positions("open")
    except Exception as exc:
        details = {
            "broker_inventory_available": True,
            "local_tracker_inventory_available": False,
            "broker_submission_suppressed": True,
            "automatic_adoption": False,
            "error": str(exc),
        }
        return (
            False,
            details,
            "Local broker-tracker inventory is unavailable; new cTrader entries are blocked until position identity can be verified.",
        )

    tracked_ids = sorted(
        {
            int(position.broker_position_id)
            for position in local_rows
            if position.broker_position_id is not None and int(position.broker_position_id) > 0
        }
    )
    broker_ids: list[int] = []
    missing_id_count = 0
    for row in broker_rows:
        position_id = _broker_position_id(row)
        if position_id is None:
            missing_id_count += 1
            continue
        broker_ids.append(position_id)

    broker_ids = sorted(set(broker_ids))
    tracked_id_set = set(tracked_ids)
    untracked_ids = [position_id for position_id in broker_ids if position_id not in tracked_id_set]
    details = {
        "broker_inventory_available": True,
        "local_tracker_inventory_available": True,
        "broker_open_position_count": len(broker_rows),
        "broker_open_position_ids": broker_ids,
        "local_open_tracker_count": len(local_rows),
        "local_tracked_broker_position_ids": tracked_ids,
        "untracked_broker_position_ids": untracked_ids,
        "broker_positions_missing_id": missing_id_count,
        "broker_submission_suppressed": bool(untracked_ids or missing_id_count),
        "automatic_adoption": False,
    }
    if untracked_ids or missing_id_count:
        return (
            False,
            details,
            "Untracked cTrader broker exposure exists; new entries are blocked until reconciliation resolves broker identity.",
        )
    return True, details, None


def _matching_new_broker_positions(
    rows: list[dict[str, object]],
    *,
    baseline_ids: set[int],
    symbol: str,
    direction: str,
    quantity_lots: float,
) -> list[dict[str, object]]:
    expected_side = "buy" if direction == "long" else "sell"
    tolerance = max(1e-6, abs(float(quantity_lots)) * 1e-3)
    matches: list[dict[str, object]] = []
    for row in rows:
        position_id = _broker_position_id(row)
        if position_id is None or position_id in baseline_ids:
            continue
        if str(row.get("symbol") or "").upper() != symbol.upper():
            continue
        if str(row.get("direction") or "").lower() != expected_side:
            continue
        try:
            broker_quantity = float(row.get("volume_lots"))
        except (TypeError, ValueError):
            continue
        if abs(broker_quantity - float(quantity_lots)) > tolerance:
            continue
        matches.append(row)
    return matches


def _reconcile_order_ack_timeout(
    *,
    baseline_available: bool,
    baseline_ids: set[int],
    symbol: str,
    direction: str,
    quantity_lots: float,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    if not baseline_available:
        return None, {
            "status": "baseline_unavailable",
            "candidate_count": 0,
            "automatic_adoption": False,
        }

    last_rows: list[dict[str, object]] = []
    for attempt in range(3):
        try:
            last_rows = [dict(row) for row in (list_positions() or [])]
        except Exception as exc:
            return None, {
                "status": "reconcile_unavailable",
                "error": str(exc),
                "candidate_count": 0,
                "automatic_adoption": False,
            }
        candidates = _matching_new_broker_positions(
            last_rows,
            baseline_ids=baseline_ids,
            symbol=symbol,
            direction=direction,
            quantity_lots=quantity_lots,
        )
        if len(candidates) == 1:
            return candidates[0], {
                "status": "broker_position_confirmed",
                "attempt": attempt + 1,
                "candidate_count": 1,
                "broker_position_id": _broker_position_id(candidates[0]),
                "automatic_adoption": True,
            }
        if len(candidates) > 1:
            return None, {
                "status": "multiple_new_candidates",
                "attempt": attempt + 1,
                "candidate_count": len(candidates),
                "candidate_position_ids": [
                    _broker_position_id(row) for row in candidates
                ],
                "automatic_adoption": False,
            }
        if attempt < 2:
            sleep(0.2)

    return None, {
        "status": "broker_position_not_observed",
        "attempt": 3,
        "candidate_count": 0,
        "observed_position_ids": [
            position_id
            for row in last_rows
            if (position_id := _broker_position_id(row)) is not None
        ],
        "automatic_adoption": False,
    }


def _unresolved_order_ack_timeout(symbol: str, timeframe: str):
    for intent in list_order_intents(200):
        if (
            intent.intent_type == "open"
            and intent.status in {"accepted", "failed"}
            and intent.symbol.upper() == symbol.upper()
            and intent.timeframe.upper() == timeframe.upper()
        ):
            details = intent.details if isinstance(intent.details, dict) else {}
            outcome_state = str(details.get("outcome_state") or "")
            if (
                outcome_state == "ambiguous_post_submit"
                and details.get("ambiguity_resolved") is not True
            ):
                return intent
            if outcome_state in {
                "submission_reserved",
                "broker_confirmed_tracking_pending",
            }:
                return intent
    return None


def _resolve_reserved_ctrader_submission(intent) -> ExecutionResult:
    details = intent.details if isinstance(intent.details, dict) else {}
    baseline_available = bool(details.get("baseline_snapshot_available"))
    baseline_ids = {
        int(value)
        for value in (details.get("baseline_position_ids") or [])
        if str(value).isdigit() and int(value) > 0
    }
    try:
        quantity_lots = float(intent.quantity or 0.0)
    except (TypeError, ValueError):
        quantity_lots = 0.0

    confirmed_row, reconciliation = _reconcile_order_ack_timeout(
        baseline_available=baseline_available,
        baseline_ids=baseline_ids,
        symbol=intent.symbol,
        direction=intent.direction,
        quantity_lots=quantity_lots,
    )

    if confirmed_row is not None:
        broker_position_id = _broker_position_id(confirmed_row)
        broker_order = {
            "status": "executed_reconciled",
            "account_type": _ctrader_account_type(),
            "symbol": intent.symbol.upper(),
            "direction": intent.direction,
            "quantity_lots": float(confirmed_row.get("volume_lots") or quantity_lots),
            "position_id": broker_position_id,
            "entry_price": confirmed_row.get("entry_price"),
            "ack": {},
            "reconciled_from_broker": True,
            "reconciliation": reconciliation,
            "client_msg_id": details.get("client_msg_id"),
        }
        try:
            update_order_intent_status(
                intent.id,
                "failed",
                {
                    "broker_order": broker_order,
                    "outcome_state": "broker_confirmed_tracking_pending",
                    "submission_may_have_succeeded": True,
                    "ambiguity_resolved": True,
                    "broker_position_confirmed": True,
                    "tracking_retained": True,
                    "failsafe_closed": False,
                    "automatic_retry": False,
                    "retryable": False,
                    "persistence_recovered": True,
                    "reconciliation": reconciliation,
                },
                reason="sqlite_persistence_recovered_broker_position",
            )
        except SQLiteBusyError as exc:
            log_incident(
                "error",
                "sqlite_persistence_busy_post_submit",
                f"SQLite remained busy while reconciling a submitted cTrader order for {intent.symbol}:{intent.timeframe}.",
                {
                    "intent_id": intent.id,
                    "phase": "submission_reservation_recovery",
                    "broker_position_id": broker_position_id,
                    "error": str(exc),
                    "automatic_resubmission": False,
                    "action_required": "Keep automatic resubmission blocked and retry local persistence only.",
                },
            )
            return ExecutionResult(
                action_taken=False,
                intent_id=intent.id,
                status="persistence_pending",
                summary="Broker position is confirmed but SQLite persistence is still busy; automatic resubmission remains blocked.",
                mode=_ctrader_execution_mode(),
                broker_position_id=broker_position_id,
                retryable=False,
            )

        log_incident(
            "warning",
            "sqlite_persistence_post_submit_reconciled",
            f"Recovered broker truth after SQLite interrupted the cTrader-order handoff for {intent.symbol}:{intent.timeframe}.",
            {
                "intent_id": intent.id,
                "broker_position_id": broker_position_id,
                "reconciliation": reconciliation,
                "automatic_resubmission": False,
                "action_required": "Recover the canonical local tracker from the persisted broker position before any new order.",
            },
        )
        return ExecutionResult(
            action_taken=False,
            intent_id=intent.id,
            status="blocked",
            summary="Broker position confirmed after persistence recovery; tracker recovery is required before any new order.",
            mode=_ctrader_execution_mode(),
            broker_position_id=broker_position_id,
            retryable=False,
        )

    if reconciliation.get("status") == "broker_position_not_observed":
        try:
            update_order_intent_status(
                intent.id,
                "failed",
                {
                    "outcome_state": "submission_resolved_no_position",
                    "submission_may_have_succeeded": True,
                    "ambiguity_resolved": True,
                    "broker_position_confirmed": False,
                    "tracking_retained": False,
                    "automatic_retry": False,
                    "retryable": True,
                    "persistence_recovered": True,
                    "reconciliation": reconciliation,
                },
                reason="sqlite_persistence_submission_resolved_no_position",
            )
        except SQLiteBusyError as exc:
            log_incident(
                "error",
                "sqlite_persistence_busy_post_submit",
                f"SQLite remained busy while resolving a submitted cTrader order for {intent.symbol}:{intent.timeframe}.",
                {
                    "intent_id": intent.id,
                    "phase": "submission_reservation_recovery",
                    "error": str(exc),
                    "automatic_resubmission": False,
                    "action_required": "Keep automatic resubmission blocked and retry local persistence only.",
                },
            )
            return ExecutionResult(
                action_taken=False,
                intent_id=intent.id,
                status="persistence_pending",
                summary="SQLite persistence is still busy; automatic resubmission remains blocked.",
                mode=_ctrader_execution_mode(),
                retryable=False,
            )

        log_incident(
            "warning",
            "sqlite_persistence_submission_resolved",
            f"Resolved SQLite-interrupted cTrader submission for {intent.symbol}:{intent.timeframe}; no broker position remains.",
            {
                "intent_id": intent.id,
                "reconciliation": reconciliation,
                "automatic_resubmission": False,
                "action_required": "The interrupted submission is resolved; a later engine cycle may evaluate a fresh signal.",
            },
        )
        return ExecutionResult(
            action_taken=False,
            intent_id=intent.id,
            status="deferred",
            summary="Interrupted submission resolved with no broker position; a later cycle may evaluate a fresh order.",
            mode=_ctrader_execution_mode(),
            retryable=True,
        )

    log_incident(
        "error",
        "sqlite_persistence_post_submit_unresolved",
        f"Could not safely resolve an SQLite-interrupted cTrader submission for {intent.symbol}:{intent.timeframe}.",
        {
            "intent_id": intent.id,
            "reconciliation": reconciliation,
            "automatic_resubmission": False,
            "action_required": "Keep automatic resubmission blocked until broker truth is unique and durable persistence succeeds.",
        },
    )
    return ExecutionResult(
        action_taken=False,
        intent_id=intent.id,
        status="blocked",
        summary="Prior cTrader submission remains unresolved; automatic resubmission is blocked.",
        mode=_ctrader_execution_mode(),
        retryable=False,
    )


def _journal_rejection_summary(
    analysis: StrategyAnalysis,
    reasons: list[str],
    details: dict[str, object] | None = None,
) -> str:
    """Return a compact operator-facing rejection reason for the Trade Journal."""
    evidence = details or {}
    reason = next((str(value).strip() for value in reasons if str(value).strip()), "Execution gate rejected the signal.")
    lowered = reason.lower()

    if "signal strength is below" in lowered or "confidence is below" in lowered:
        try:
            confidence = float(evidence.get("confidence", analysis.confidence))
            minimum = float(evidence["min_confidence"])
            return f"Rejected - signal strength {confidence:.0%} < minimum {minimum:.0%}."
        except (KeyError, TypeError, ValueError):
            pass

    if "inside cooldown" in lowered:
        try:
            minutes = int(evidence["cooldown_minutes"])
            return f"Rejected - {analysis.symbol} in {minutes}-minute cooldown."
        except (KeyError, TypeError, ValueError):
            return f"Rejected - {analysis.symbol} is still in cooldown."

    if "market bar is stale" in lowered:
        return f"Rejected - stale {analysis.timeframe} market bar."

    if "max daily trade count" in lowered:
        return "Rejected - max daily trade count reached."

    if "daily loss cap" in lowered:
        return "Rejected - daily loss cap reached."

    return f"Rejected - {reason.rstrip('.')}."


def _refresh_open_position(
    item: WatchlistItem,
    mark_price: float,
    *,
    ctrader_execution: bool = False,
) -> PaperPosition | None:
    position = get_open_position(item.symbol.upper(), item.timeframe.upper())
    if not position:
        return None

    if not ctrader_execution:
        reconcile_position(position, mark_price)
        return get_open_position(item.symbol.upper(), item.timeframe.upper())

    # In cTrader mode the broker is the execution source of truth. Never simulate
    # a local-only SL/TP exit while the broker position is still open.
    apply_mark(position, mark_price)
    status = get_broker_status()
    if status.execution_ready:
        match = match_broker_position(position, list_positions())
        if match.status in {"id_mismatch", "legacy_ambiguous"}:
            message = f"Could not safely identify broker position for {position.symbol}:{position.timeframe}."
            log_incident(
                "error",
                "ctrader_position_identity_ambiguous",
                message,
                {
                    "position_id": position.id,
                    "broker_position_id": position.broker_position_id,
                    "match_status": match.status,
                    "candidates": match.candidates,
                },
            )
            raise BrokerPositionIdentityError(message)
        if not match.matched:
            close_local_position_from_broker(
                position,
                fallback_price=mark_price,
                fallback_reason="broker_position_closed",
            )
            return None
        if match.status == "legacy_match":
            position = set_paper_position_broker_id(
                position.id,
                int(match.row.get("position_id") or 0),
            )

        ledger_sync = reconcile_open_position_ledger(position, match.row)
        if ledger_sync.get("status") == "partial_close_synced":
            add_trade_audit(
                event_type="ctrader_partial_close_synced",
                symbol=position.symbol,
                timeframe=position.timeframe,
                strategy=position.strategy,
                position_id=position.id,
                summary="Synchronized broker partial close before strategy execution.",
                details=ledger_sync,
            )
        elif ledger_sync.get("status") == "pending_deal_history":
            log_incident(
                "warning",
                "ctrader_partial_close_history_pending",
                f"Broker volume decreased for {position.symbol}:{position.timeframe}, but close deal history is not complete yet.",
                {"position_id": position.id, **ledger_sync},
            )
        elif ledger_sync.get("status") in {"identity_mismatch", "volume_increase_mismatch", "unavailable"}:
            message = f"Could not safely reconcile broker volume for {position.symbol}:{position.timeframe}."
            log_incident(
                "error",
                "ctrader_volume_reconciliation_failed",
                message,
                {"position_id": position.id, **ledger_sync},
            )
            raise BrokerPositionIdentityError(message)

    return get_open_position(item.symbol.upper(), item.timeframe.upper())


def execute_paper_signal(
    *,
    config: EngineConfig,
    watch_item: WatchlistItem,
    analysis: StrategyAnalysis,
    mark_price: float,
    bar_timestamp=None,
    bar_snapshot=None,
    quantity: float | None = None,
    source: str = "auto",
) -> ExecutionResult:
    ctrader_execution = bool(config.ctrader_autotrade and watch_item.trading_enabled)
    try:
        position = _refresh_open_position(watch_item, mark_price, ctrader_execution=ctrader_execution)
    except BrokerPositionIdentityError as exc:
        return ExecutionResult(
            action_taken=False,
            intent_id=None,
            status="failed",
            summary=str(exc),
            mode=_ctrader_execution_mode(),
            retryable=True,
        )

    # Keep an already-open cTrader position protected at the broker even when the
    # current strategy result is no_trade or later fails a new-entry risk gate.
    # The local ledger is not allowed to drift silently from broker SL/TP.
    if ctrader_execution and position:
        try:
            protection = sync_position_targets(
                symbol=position.symbol,
                direction=position.direction,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                position_id=position.broker_position_id,
                reference_price=mark_price,
            )
            if protection.get("status") in {"exit_due_stop_loss", "exit_due_take_profit"}:
                reason = (
                    "broker_stop_loss"
                    if protection.get("status") == "exit_due_stop_loss"
                    else "broker_take_profit"
                )
                close_result = attempt_verified_close(
                    position,
                    fallback_price=mark_price,
                    reason=reason,
                    phase="pre_signal_protective_exit",
                    quantity_lots=float(protection.get("quantity_lots") or position.quantity),
                )
                if close_result.get("closed"):
                    closed = close_result["position"]
                    add_trade_audit(
                        event_type="ctrader_protective_exit",
                        symbol=position.symbol,
                        timeframe=position.timeframe,
                        strategy=position.strategy,
                        position_id=closed.id,
                        summary="Closed cTrader position because an intended protective target was already crossed.",
                        details={"protection": protection, "close_result": close_result},
                    )
                    return ExecutionResult(
                        action_taken=True,
                        intent_id=None,
                        status="executed",
                        summary=reason,
                        position_id=closed.id,
                        mode=_ctrader_execution_mode(),
                        broker_position_id=position.broker_position_id,
                        retryable=False,
                    )
                return ExecutionResult(
                    action_taken=False,
                    intent_id=None,
                    status=str(close_result.get("status") or "failed"),
                    summary="Broker close was not verified; local tracker remains open.",
                    position_id=position.id,
                    mode=_ctrader_execution_mode(),
                    broker_position_id=position.broker_position_id,
                    retryable=bool(close_result.get("retryable")),
                )
            if protection.get("status") == "synced":
                add_trade_audit(
                    event_type="ctrader_protection_repaired",
                    symbol=position.symbol,
                    timeframe=position.timeframe,
                    strategy=position.strategy,
                    position_id=position.id,
                    summary="Repaired broker SL/TP from the local tracking position.",
                    details=protection,
                )
        except CTraderProtectionSyncFailure as exc:
            try:
                broker_row = _canonical_broker_row_for_position(position)
                failsafe = fail_safe_close_unverified_position(
                    position,
                    broker_row=broker_row,
                    fallback_price=mark_price,
                    protection_error=exc,
                    phase="pre_signal_protection",
                )
            except Exception as failsafe_exc:
                log_incident(
                    "error",
                    "ctrader_protection_failsafe_unresolved",
                    f"Protection failure could not be safely resolved for {position.symbol}:{position.timeframe}.",
                    {
                        "position_id": position.id,
                        "broker_position_id": position.broker_position_id,
                        "protection_error": str(exc),
                        "failsafe_error": str(failsafe_exc),
                        "retryable": False,
                    },
                )
                return ExecutionResult(
                    action_taken=False,
                    intent_id=None,
                    status="protection_failsafe_pending",
                    summary=str(failsafe_exc),
                    position_id=position.id,
                    mode=_ctrader_execution_mode(),
                    broker_position_id=position.broker_position_id,
                    retryable=False,
                )

            if failsafe.get("closed"):
                return ExecutionResult(
                    action_taken=True,
                    intent_id=None,
                    status="failed",
                    summary="cTrader position was closed by fail-safe because broker protection could not be verified.",
                    position_id=int(failsafe.get("closed_position_id") or position.id),
                    mode=_ctrader_execution_mode(),
                    broker_position_id=position.broker_position_id,
                    retryable=False,
                )
            return ExecutionResult(
                action_taken=False,
                intent_id=None,
                status="protection_failsafe_pending",
                summary="Broker protection failed and the fail-safe close failed; canonical tracking was retained.",
                position_id=position.id,
                mode=_ctrader_execution_mode(),
                broker_position_id=position.broker_position_id,
                retryable=False,
            )
        except Exception as exc:
            log_incident(
                "error",
                "ctrader_protection_sync_failed",
                f"Could not synchronize broker protection for {position.symbol}:{position.timeframe}",
                {"position_id": position.id, "error": str(exc)},
            )
            add_trade_audit(
                event_type="ctrader_protection_sync_failed",
                symbol=position.symbol,
                timeframe=position.timeframe,
                strategy=position.strategy,
                position_id=position.id,
                summary="Broker SL/TP synchronization failed; execution paused for this symbol.",
                details={"error": str(exc)},
            )
            return ExecutionResult(
                action_taken=False,
                intent_id=None,
                status="failed",
                summary=str(exc),
                position_id=position.id,
                mode=_ctrader_execution_mode(),
                retryable=True,
            )

    # Broker symbol metadata arrives asynchronously after account authorization.
    # If cTrader execution is enabled, never attempt an order until the exact
    # symbol contract is loaded. Mark this as retryable so the engine can
    # revisit the same bar without risking a duplicate order.
    if ctrader_execution and analysis.signal != "no_trade":
        broker_ready, broker_reason = get_symbol_execution_readiness(analysis.symbol)
        if not broker_ready:
            log_incident(
                "warning",
                "ctrader_symbol_not_ready",
                f"Deferred cTrader execution for {analysis.symbol}:{analysis.timeframe}",
                {"reason": broker_reason, "retryable": True},
            )
            add_trade_audit(
                event_type="ctrader_order_deferred",
                symbol=analysis.symbol,
                timeframe=analysis.timeframe,
                strategy=analysis.strategy,
                summary="cTrader order deferred until broker symbol metadata is ready.",
                details={"reason": broker_reason},
            )
            return ExecutionResult(
                action_taken=False,
                intent_id=None,
                status="deferred",
                summary=broker_reason,
                mode=_ctrader_execution_mode(),
                retryable=True,
            )

    monetary_basis = resolve_monetary_basis(config)
    if ctrader_execution and analysis.signal != "no_trade":
        account_snapshot = get_broker_account_snapshot()
        monetary_basis = resolve_monetary_basis(
            config,
            ctrader_execution=True,
            account_snapshot=account_snapshot,
        )
        if not monetary_basis.verified:
            reason = monetary_basis.reason or "cTrader account monetary snapshot is unavailable."
            log_incident(
                "warning",
                "ctrader_account_snapshot_not_ready",
                f"Deferred cTrader execution for {analysis.symbol}:{analysis.timeframe}",
                {"reason": reason, **monetary_basis.as_details(), "retryable": True},
            )
            add_trade_audit(
                event_type="ctrader_order_deferred",
                symbol=analysis.symbol,
                timeframe=analysis.timeframe,
                strategy=analysis.strategy,
                summary="cTrader order deferred until broker monetary account data is verified.",
                details={"reason": reason, **monetary_basis.as_details()},
            )
            return ExecutionResult(
                action_taken=False,
                intent_id=None,
                status="deferred",
                summary=reason,
                mode=_ctrader_execution_mode(),
                retryable=True,
            )

    instrument = get_instrument_spec(analysis.symbol, monetary_basis.currency or config.account_currency)
    configured_quantity = watch_item.lot_size if source != "manual" else None
    sizing = (
        derive_auto_quantity(config, analysis, mark_price, monetary_basis=monetary_basis)
        if source != "manual" and configured_quantity is None and quantity is None
        else None
    )
    if sizing is not None and not sizing.accepted:
        evidence = {
            **sizing.details,
            "analysis": analysis.model_dump(mode="json"),
            "source": source,
            "instrument_spec": instrument.model_dump(mode="json"),
            "sizing_reasons": sizing.reasons,
        }
        decision = create_decision_record(
            correlation_id=str(uuid4()),
            decision_type="paper_execution_gate",
            symbol=analysis.symbol,
            timeframe=analysis.timeframe,
            strategy=analysis.strategy,
            outcome="rejected_sizing",
            summary=sizing.reasons[0] if sizing.reasons else "Sizing rejected.",
            evidence=evidence,
        )
        intent = create_order_intent(
            symbol=analysis.symbol,
            timeframe=analysis.timeframe,
            strategy=analysis.strategy,
            direction=analysis.signal,
            intent_type="skip",
            status="rejected",
            confidence=analysis.confidence,
            entry_price=analysis.entry_price or mark_price,
            stop_loss=analysis.stop_loss,
            take_profit=analysis.take_profit,
            quantity=None,
            rationale="; ".join(sizing.reasons),
            details=evidence,
            decision_id=decision.id,
        )
        log_incident(
            "warning",
            "sizing_rejected",
            f"Blocked automatic sizing for {analysis.symbol}:{analysis.timeframe}",
            {"reasons": sizing.reasons, "intent_id": intent.id, "decision_id": decision.id},
        )
        add_trade_audit(
            event_type="paper_sizing_rejected",
            symbol=analysis.symbol,
            timeframe=analysis.timeframe,
            strategy=analysis.strategy,
            intent_id=intent.id,
            summary=_journal_rejection_summary(analysis, sizing.reasons, sizing.details),
            details={"reasons": sizing.reasons, **sizing.details},
        )
        return ExecutionResult(
            action_taken=False,
            intent_id=intent.id,
            status="rejected",
            summary=sizing.reasons[0] if sizing.reasons else "sizing rejected",
        )
    requested_quantity = (
        float(quantity if quantity is not None else (config.paper_trade_size or 1.0))
        if source == "manual"
        else float(
            quantity
            if quantity is not None
            else (
                configured_quantity
                if configured_quantity is not None
                else ((sizing.requested_quantity if sizing else None) or (config.paper_trade_size or 1.0))
            )
        )
    )
    quantity_decision = evaluate_order_quantity(analysis.symbol, requested_quantity, source)

    if not quantity_decision.accepted:
        evidence = {
            **(sizing.details if sizing else {}),
            **quantity_decision.details,
            "analysis": analysis.model_dump(mode="json"),
            "source": source,
            "instrument_spec": instrument.model_dump(mode="json"),
            "quantity_reasons": quantity_decision.reasons,
        }
        decision = create_decision_record(
            correlation_id=str(uuid4()),
            decision_type="paper_execution_gate",
            symbol=analysis.symbol,
            timeframe=analysis.timeframe,
            strategy=analysis.strategy,
            outcome="rejected_quantity",
            summary=quantity_decision.reasons[0] if quantity_decision.reasons else "Quantity rejected.",
            evidence=evidence,
        )
        intent = create_order_intent(
            symbol=analysis.symbol,
            timeframe=analysis.timeframe,
            strategy=analysis.strategy,
            direction=analysis.signal,
            intent_type="skip",
            status="rejected",
            confidence=analysis.confidence,
            entry_price=analysis.entry_price or mark_price,
            stop_loss=analysis.stop_loss,
            take_profit=analysis.take_profit,
            quantity=requested_quantity,
            rationale="; ".join(quantity_decision.reasons),
            details=evidence,
            decision_id=decision.id,
        )
        log_incident(
            "info",
            "quantity_rejected",
            f"Rejected {analysis.signal} for {analysis.symbol}:{analysis.timeframe} due to quantity constraints",
            {"reasons": quantity_decision.reasons, "intent_id": intent.id},
        )
        add_trade_audit(
            event_type="paper_signal_rejected",
            symbol=analysis.symbol,
            timeframe=analysis.timeframe,
            strategy=analysis.strategy,
            intent_id=intent.id,
            summary=_journal_rejection_summary(analysis, quantity_decision.reasons, quantity_decision.details),
            details={
                "reasons": quantity_decision.reasons,
                **(sizing.details if sizing else {}),
                **quantity_decision.details,
            },
        )
        return ExecutionResult(
            action_taken=False,
            intent_id=intent.id,
            status="rejected",
            summary=quantity_decision.reasons[0] if quantity_decision.reasons else "quantity rejected",
        )

    trade_quantity = float(quantity_decision.final_quantity or requested_quantity)
    risk = evaluate_risk(
        config=config,
        watch_item=watch_item,
        analysis=analysis,
        existing_position=position,
        mark_price=mark_price,
        bar_timestamp=bar_timestamp,
        bar_snapshot=bar_snapshot,
        source=source,
        monetary_basis=monetary_basis,
    )

    # Live arming gates only *new* real-money entries. Existing broker-backed
    # positions must remain manageable after a restart/disarm so protection,
    # reconciliation, and verified close operations are never disabled.
    if (
        ctrader_execution
        and position is None
        and risk.accepted
        and risk.intent_type == "open"
    ):
        broker = get_broker_status()
        live_block_reason = live_entry_block_reason(
            broker.account_type,
            broker.account_id,
        )
        if live_block_reason:
            risk.accepted = False
            risk.intent_type = "skip"
            risk.reasons.append(live_block_reason)
            risk.details["live_trading_armed"] = False
            risk.details["live_trading_armed_account_id"] = get_live_trading_armed_account_id()
            risk.details["active_ctrader_account_id"] = broker.account_id
            risk.details["active_ctrader_account_type"] = broker.account_type

    # A broker position that is not represented by a canonical local tracker is
    # unresolved account exposure. New entries must fail closed until that
    # exposure is reconciled; existing tracked positions remain manageable.
    if (
        ctrader_execution
        and position is None
        and risk.accepted
        and risk.intent_type == "open"
    ):
        inventory_ok, inventory_details, inventory_reason = _ctrader_new_entry_inventory_gate()
        risk.details["broker_entry_inventory"] = inventory_details
        if not inventory_ok:
            reason = inventory_reason or (
                "cTrader broker exposure is unresolved; new entries are blocked until broker identity is verified."
            )
            risk.accepted = False
            risk.intent_type = "skip"
            risk.reasons.append(reason)
            incident_code = (
                "ctrader_untracked_broker_exposure_blocks_entry"
                if inventory_details.get("broker_inventory_available")
                and inventory_details.get("local_tracker_inventory_available", True)
                else "ctrader_broker_inventory_unavailable_blocks_entry"
            )
            log_incident(
                "error",
                incident_code,
                f"Blocked new cTrader entry for {analysis.symbol}:{analysis.timeframe}.",
                {
                    **inventory_details,
                    "symbol": analysis.symbol.upper(),
                    "timeframe": analysis.timeframe.upper(),
                    "strategy": analysis.strategy,
                    "action_required": (
                        "Resolve or close untracked broker exposure and rerun reconciliation before enabling new entries."
                    ),
                },
            )

    flipped = False
    intent_quantity = position.quantity if position and position.direction == analysis.signal else trade_quantity
    intent_reasons = list(risk.reasons)
    if sizing:
        intent_reasons = [*sizing.reasons, *intent_reasons]
    if quantity_decision.details.get("quantity_normalized"):
        intent_reasons = [*quantity_decision.reasons, *intent_reasons]

    if (
        ctrader_execution
        and position is None
        and risk.accepted
        and risk.intent_type == "open"
    ):
        unresolved_ack = _unresolved_order_ack_timeout(
            analysis.symbol,
            analysis.timeframe,
        )
        if unresolved_ack is not None:
            unresolved_details = (
                unresolved_ack.details if isinstance(unresolved_ack.details, dict) else {}
            )
            if unresolved_details.get("outcome_state") == "submission_reserved":
                return _resolve_reserved_ctrader_submission(unresolved_ack)
            return ExecutionResult(
                action_taken=False,
                intent_id=unresolved_ack.id,
                status="blocked",
                summary=(
                    "Automatic cTrader order blocked because a prior post-submission "
                    "outcome or broker-confirmed tracking handoff is unresolved. "
                    "Reconcile broker truth before any new order is allowed."
                ),
                mode=_ctrader_execution_mode(),
                broker_position_id=(
                    int((unresolved_details.get("broker_order") or {}).get("position_id") or 0)
                    or None
                ),
                retryable=False,
            )

    decision_evidence = {
        **(sizing.details if sizing else {}),
        **quantity_decision.details,
        **risk.details,
        "analysis": analysis.model_dump(mode="json"),
        "source": source,
        "instrument_spec": instrument.model_dump(mode="json"),
        "risk_reasons": risk.reasons,
    }
    decision_outcome = f"accepted_{risk.intent_type}" if risk.accepted else "rejected_risk"
    try:
        decision = create_decision_record(
            correlation_id=str(uuid4()),
            decision_type="paper_execution_gate",
            symbol=analysis.symbol,
            timeframe=analysis.timeframe,
            strategy=analysis.strategy,
            outcome=decision_outcome,
            summary=(risk.reasons[0] if risk.reasons else decision_outcome),
            evidence=decision_evidence,
        )
        intent = create_order_intent(
            symbol=analysis.symbol,
            timeframe=analysis.timeframe,
            strategy=analysis.strategy,
            direction=analysis.signal,
            intent_type=risk.intent_type,
            status="accepted" if risk.accepted else "rejected",
            confidence=analysis.confidence,
            entry_price=analysis.entry_price or mark_price,
            stop_loss=analysis.stop_loss,
            take_profit=analysis.take_profit,
            quantity=intent_quantity,
            rationale="; ".join(intent_reasons),
            details=decision_evidence,
            decision_id=decision.id,
        )
    except SQLiteBusyError as exc:
        log_incident(
            "error",
            "sqlite_persistence_busy_pre_submit",
            f"Blocked execution for {analysis.symbol}:{analysis.timeframe} because SQLite durable persistence is busy.",
            {
                "phase": "decision_or_intent_persistence",
                "error": str(exc),
                "broker_submission_suppressed": True,
                "automatic_resubmission": False,
                "retryable_persistence": True,
                "action_required": "Release the SQLite writer lock and retry a fresh engine cycle; no broker order was submitted.",
            },
        )
        return ExecutionResult(
            action_taken=False,
            intent_id=None,
            status="deferred",
            summary="SQLite persistence is busy; broker submission was suppressed before any cTrader order.",
            mode=_ctrader_execution_mode() if ctrader_execution else "paper_only",
            retryable=True,
        )

    lifecycle_version_hash: str | None = None
    lifecycle_details = risk.details.get("strategy_lifecycle")
    if isinstance(lifecycle_details, dict) and lifecycle_details.get("governed"):
        lifecycle_version_hash = str(lifecycle_details.get("version_hash") or "").strip() or None

    if not risk.accepted:
        log_incident(
            "info",
            "signal_rejected",
            f"Rejected {analysis.signal} for {analysis.symbol}:{analysis.timeframe}",
            {"reasons": risk.reasons, "intent_id": intent.id},
        )
        add_trade_audit(
            event_type="paper_signal_rejected",
            symbol=analysis.symbol,
            timeframe=analysis.timeframe,
            strategy=analysis.strategy,
            intent_id=intent.id,
            summary=_journal_rejection_summary(analysis, risk.reasons, risk.details),
            details={"reasons": risk.reasons, **risk.details},
        )
        return ExecutionResult(
            action_taken=False,
            intent_id=intent.id,
            status="rejected",
            summary=risk.reasons[0] if risk.reasons else "signal rejected",
        )

    if ctrader_execution and position and position.direction != analysis.signal:
        update_order_intent_status(
            intent.id,
            "failed",
            {"reason": "ctrader_signal_flip_requires_reconciliation"},
            reason="ctrader_signal_flip_blocked",
        )
        log_incident(
            "warning",
            "ctrader_signal_flip_blocked",
            f"Blocked cTrader signal flip for {analysis.symbol}:{analysis.timeframe}",
            {"intent_id": intent.id, "position_id": position.id},
        )
        add_trade_audit(
            event_type="ctrader_order_blocked",
            symbol=analysis.symbol,
            timeframe=analysis.timeframe,
            strategy=analysis.strategy,
            intent_id=intent.id,
            position_id=position.id,
            summary="cTrader order blocked because the opposite broker position must be reconciled first.",
            details={"direction": analysis.signal},
        )
        return ExecutionResult(
            action_taken=False,
            intent_id=intent.id,
            status="failed",
            summary="cTrader signal flip blocked pending broker reconciliation.",
            position_id=position.id,
            mode=_ctrader_execution_mode(),
        )

    if position and position.direction != analysis.signal:
        closed = close_paper_position(position.id, mark_price, "signal_flip")
        update_order_intent_status(
            intent.id,
            "accepted",
            {"closed_position_id": closed.id, "flip": True},
            reason="conflicting_position_closed",
        )
        flipped = True
        add_trade_audit(
            event_type="paper_signal_flip",
            symbol=analysis.symbol,
            timeframe=analysis.timeframe,
            strategy=analysis.strategy,
            intent_id=intent.id,
            position_id=closed.id,
            summary="Closed existing paper position on signal flip.",
            details={"close_reason": "signal_flip"},
        )
        position = None

    if position and position.direction == analysis.signal:
        broker_protection = None
        if ctrader_execution:
            try:
                broker_protection = sync_position_targets(
                    symbol=position.symbol,
                    direction=position.direction,
                    stop_loss=analysis.stop_loss,
                    take_profit=analysis.take_profit,
                    position_id=position.broker_position_id,
                    reference_price=mark_price,
                )
                if broker_protection.get("status") in {"exit_due_stop_loss", "exit_due_take_profit"}:
                    reason = (
                        "broker_stop_loss"
                        if broker_protection.get("status") == "exit_due_stop_loss"
                        else "broker_take_profit"
                    )
                    close_result = attempt_verified_close(
                        position,
                        fallback_price=mark_price,
                        reason=reason,
                        phase="signal_target_protective_exit",
                        quantity_lots=float(broker_protection.get("quantity_lots") or position.quantity),
                    )
                    if close_result.get("closed"):
                        closed = close_result["position"]
                        update_order_intent_status(
                            intent.id,
                            "executed",
                            {
                                "closed_position_id": closed.id,
                                "broker_protection": broker_protection,
                                "close_result": close_result,
                            },
                            reason="protective_exit_before_target_update",
                        )
                        add_trade_audit(
                            event_type="ctrader_protective_exit",
                            symbol=analysis.symbol,
                            timeframe=analysis.timeframe,
                            strategy=analysis.strategy,
                            intent_id=intent.id,
                            position_id=closed.id,
                            summary="Closed cTrader position because the refreshed target was already crossed.",
                            details={"protection": broker_protection, "close_result": close_result},
                        )
                        return ExecutionResult(
                            action_taken=True,
                            intent_id=intent.id,
                            status="executed",
                            summary=reason,
                            position_id=closed.id,
                            mode=_ctrader_execution_mode(),
                            broker_position_id=position.broker_position_id,
                            retryable=False,
                        )

                    update_order_intent_status(
                        intent.id,
                        "failed",
                        {
                            "broker_protection": broker_protection,
                            "close_result": close_result,
                            "tracking_retained": True,
                        },
                        reason="protective_exit_close_not_verified",
                    )
                    return ExecutionResult(
                        action_taken=False,
                        intent_id=intent.id,
                        status=str(close_result.get("status") or "failed"),
                        summary="Broker close was not verified; local tracker remains open.",
                        position_id=position.id,
                        mode=_ctrader_execution_mode(),
                        broker_position_id=position.broker_position_id,
                        retryable=bool(close_result.get("retryable")),
                    )
            except CTraderProtectionSyncFailure as exc:
                try:
                    broker_row = _canonical_broker_row_for_position(position)
                    failsafe = fail_safe_close_unverified_position(
                        position,
                        broker_row=broker_row,
                        fallback_price=mark_price,
                        protection_error=exc,
                        phase="signal_target_update",
                    )
                except Exception as failsafe_exc:
                    failsafe = {
                        "closed": False,
                        "tracking_retained": True,
                        "failsafe_error": str(failsafe_exc),
                    }

                details = {
                    "broker": "ctrader",
                    "error": str(exc),
                    "protection_failure_kind": exc.failure_kind,
                    "broker_position_id": position.broker_position_id,
                    "failsafe": failsafe,
                    "local_targets_updated": False,
                }
                update_order_intent_status(
                    intent.id,
                    "failed",
                    details,
                    reason=(
                        "ctrader_target_update_failsafe_closed"
                        if failsafe.get("closed")
                        else "ctrader_target_update_failsafe_pending"
                    ),
                )
                log_incident(
                    "error" if not failsafe.get("closed") else "warning",
                    "ctrader_target_update_rejected",
                    f"Broker target update failed for {analysis.symbol}:{analysis.timeframe}; fail-safe policy applied.",
                    {"intent_id": intent.id, "position_id": position.id, **details},
                )
                add_trade_audit(
                    event_type="ctrader_target_update_rejected",
                    symbol=analysis.symbol,
                    timeframe=analysis.timeframe,
                    strategy=analysis.strategy,
                    intent_id=intent.id,
                    position_id=position.id,
                    summary=(
                        "Broker target update failed; local targets were not changed and fail-safe policy was applied."
                    ),
                    details=details,
                )
                return ExecutionResult(
                    action_taken=bool(failsafe.get("closed")),
                    intent_id=intent.id,
                    status="failed" if failsafe.get("closed") else "protection_failsafe_pending",
                    summary=(
                        "cTrader position was closed by fail-safe because broker protection could not be verified."
                        if failsafe.get("closed")
                        else "Broker protection failed and the fail-safe close failed; canonical tracking was retained."
                    ),
                    position_id=int(failsafe.get("closed_position_id") or position.id),
                    mode=_ctrader_execution_mode(),
                    broker_position_id=position.broker_position_id,
                    retryable=False,
                )
            except Exception as exc:
                update_order_intent_status(
                    intent.id,
                    "failed",
                    {"broker": "ctrader", "error": str(exc)},
                    reason="ctrader_target_update_failed",
                )
                log_incident(
                    "error",
                    "ctrader_target_update_failed",
                    f"Could not update broker targets for {analysis.symbol}:{analysis.timeframe}",
                    {"intent_id": intent.id, "position_id": position.id, "error": str(exc)},
                )
                add_trade_audit(
                    event_type="ctrader_target_update_failed",
                    symbol=analysis.symbol,
                    timeframe=analysis.timeframe,
                    strategy=analysis.strategy,
                    intent_id=intent.id,
                    position_id=position.id,
                    summary="Kept previous local targets because broker target update failed.",
                    details={"error": str(exc)},
                )
                return ExecutionResult(
                    action_taken=False,
                    intent_id=intent.id,
                    status="failed",
                    summary=str(exc),
                    position_id=position.id,
                    mode=_ctrader_execution_mode(),
                )

        update_paper_position_targets(position.id, analysis.stop_loss, analysis.take_profit)
        update_order_intent_status(
            intent.id,
            "executed",
            {"updated_position_id": position.id, "broker_protection": broker_protection or {}},
            reason="position_targets_updated",
        )
        add_trade_audit(
            event_type="paper_signal_update",
            symbol=analysis.symbol,
            timeframe=analysis.timeframe,
            strategy=analysis.strategy,
            intent_id=intent.id,
            position_id=position.id,
            summary="Updated existing paper position targets.",
            details={"stop_loss": analysis.stop_loss, "take_profit": analysis.take_profit},
        )
        return ExecutionResult(
            action_taken=True,
            intent_id=intent.id,
            status="executed",
            summary="position updated",
            position_id=position.id,
        )

    broker_order = None
    unprotected_close_error: str | None = None
    unprotected_close_failure: Exception | None = None
    unprotected_close_phase: str | None = None
    ack_timeout_reconciled = False
    if ctrader_execution:
        baseline_rows: list[dict[str, object]] = []
        baseline_error: str | None = None
        try:
            baseline_rows = [dict(row) for row in (list_positions() or [])]
        except Exception as exc:
            baseline_error = str(exc)
        baseline_ids = {
            position_id
            for row in baseline_rows
            if (position_id := _broker_position_id(row)) is not None
        }
        client_msg_id = f"tradeagent-intent-{intent.id}"
        try:
            update_order_intent_status(
                intent.id,
                "accepted",
                {
                    "outcome_state": "submission_reserved",
                    "submission_may_have_succeeded": True,
                    "ambiguity_resolved": False,
                    "automatic_retry": False,
                    "retryable": False,
                    "client_msg_id": client_msg_id,
                    "baseline_snapshot_available": baseline_error is None,
                    "baseline_snapshot_error": baseline_error,
                    "baseline_position_ids": sorted(baseline_ids),
                },
                reason="ctrader_order_submission_reserved",
            )
        except SQLiteBusyError as exc:
            log_incident(
                "error",
                "sqlite_persistence_busy_pre_submit",
                f"Blocked cTrader submission for {analysis.symbol}:{analysis.timeframe} because the durable submission reservation could not be written.",
                {
                    "intent_id": intent.id,
                    "phase": "submission_reservation",
                    "error": str(exc),
                    "broker_submission_suppressed": True,
                    "automatic_resubmission": False,
                    "retryable_persistence": True,
                    "action_required": "Release the SQLite writer lock and retry a fresh engine cycle; no broker order was submitted.",
                },
            )
            return ExecutionResult(
                action_taken=False,
                intent_id=intent.id,
                status="deferred",
                summary="SQLite persistence is busy; cTrader order submission was blocked before routing.",
                mode=_ctrader_execution_mode(),
                retryable=True,
            )

        try:
            broker_order = place_market_order(
                symbol=analysis.symbol,
                direction=analysis.signal,
                quantity_lots=trade_quantity,
                stop_loss=analysis.stop_loss,
                take_profit=analysis.take_profit,
                client_msg_id=client_msg_id,
            )
        except CTraderOrderAcknowledgementTimeout as exc:
            confirmed_row, reconciliation = _reconcile_order_ack_timeout(
                baseline_available=baseline_error is None,
                baseline_ids=baseline_ids,
                symbol=analysis.symbol,
                direction=analysis.signal,
                quantity_lots=trade_quantity,
            )
            if confirmed_row is not None:
                broker_order = {
                    "status": "executed_reconciled",
                    "account_type": _ctrader_account_type(),
                    "symbol": analysis.symbol.upper(),
                    "direction": analysis.signal,
                    "quantity_lots": float(confirmed_row.get("volume_lots") or trade_quantity),
                    "position_id": _broker_position_id(confirmed_row),
                    "entry_price": confirmed_row.get("entry_price"),
                    "ack": {},
                    "acknowledgement_timeout": True,
                    "reconciled_from_broker": True,
                    "client_msg_id": client_msg_id,
                    "reconciliation": reconciliation,
                }
                ack_timeout_reconciled = True
                log_incident(
                    "warning",
                    "ctrader_order_ack_timeout_reconciled",
                    f"cTrader order acknowledgement timed out for {analysis.symbol}:{analysis.timeframe}, but broker reconciliation confirmed one new position.",
                    {
                        "intent_id": intent.id,
                        "client_msg_id": client_msg_id,
                        "baseline_position_ids": sorted(baseline_ids),
                        "reconciliation": reconciliation,
                        "automatic_retry": False,
                    },
                )
                add_trade_audit(
                    event_type="ctrader_order_ack_timeout_reconciled",
                    symbol=analysis.symbol,
                    timeframe=analysis.timeframe,
                    strategy=analysis.strategy,
                    intent_id=intent.id,
                    summary="Broker truth confirmed one new cTrader position after the order acknowledgement timed out.",
                    details={
                        "client_msg_id": client_msg_id,
                        "baseline_position_ids": sorted(baseline_ids),
                        "reconciliation": reconciliation,
                        "quantity": trade_quantity,
                    },
                )
            else:
                details = {
                    "broker": "ctrader",
                    "account_type": _ctrader_account_type(),
                    "error": str(exc),
                    "outcome_state": "ambiguous_post_submit",
                    "acknowledgement_timeout": True,
                    "submission_may_have_succeeded": True,
                    "ambiguity_resolved": False,
                    "retryable": False,
                    "automatic_retry": False,
                    "client_msg_id": client_msg_id,
                    "baseline_snapshot_available": baseline_error is None,
                    "baseline_snapshot_error": baseline_error,
                    "baseline_position_ids": sorted(baseline_ids),
                    "reconciliation": reconciliation,
                }
                update_order_intent_status(
                    intent.id,
                    "failed",
                    details,
                    reason="ctrader_order_ack_timeout_ambiguous",
                )
                log_incident(
                    "error",
                    "ctrader_order_ack_timeout_ambiguous",
                    f"cTrader order acknowledgement timed out after submission for {analysis.symbol}:{analysis.timeframe}; automatic resubmission is blocked.",
                    {"intent_id": intent.id, **details},
                )
                add_trade_audit(
                    event_type="ctrader_order_ack_timeout_ambiguous",
                    symbol=analysis.symbol,
                    timeframe=analysis.timeframe,
                    strategy=analysis.strategy,
                    intent_id=intent.id,
                    summary="cTrader order acknowledgement timed out after submission; broker outcome remains ambiguous and automatic retry is blocked.",
                    details=details,
                )
                return ExecutionResult(
                    action_taken=False,
                    intent_id=intent.id,
                    status="failed",
                    summary=(
                        "cTrader order acknowledgement timed out after submission; "
                        "broker outcome is ambiguous and automatic retry is blocked."
                    ),
                    mode=_ctrader_execution_mode(),
                    retryable=False,
                )
        except Exception as exc:
            update_order_intent_status(
                intent.id,
                "failed",
                {
                    "broker": "ctrader",
                    "account_type": _ctrader_account_type(),
                    "error": str(exc),
                    "outcome_state": "submission_failed",
                    "submission_may_have_succeeded": False,
                    "ambiguity_resolved": True,
                    "automatic_retry": False,
                    "retryable": True,
                },
                reason="ctrader_order_failed",
            )
            log_incident(
                "error",
                "ctrader_order_failed",
                f"cTrader order failed for {analysis.symbol}:{analysis.timeframe}",
                {"intent_id": intent.id, "error": str(exc)},
            )
            add_trade_audit(
                event_type="ctrader_order_failed",
                symbol=analysis.symbol,
                timeframe=analysis.timeframe,
                strategy=analysis.strategy,
                intent_id=intent.id,
                summary="cTrader order was not executed.",
                details={"error": str(exc), "quantity": trade_quantity},
            )
            return ExecutionResult(
                action_taken=False,
                intent_id=intent.id,
                status="failed",
                summary=str(exc),
                mode=_ctrader_execution_mode(),
            )

    if broker_order:
        broker_position_id = int(broker_order.get("position_id") or 0)
        try:
            update_order_intent_status(
                intent.id,
                "accepted",
                {
                    "broker_order": broker_order,
                    "outcome_state": "broker_confirmed_tracking_pending",
                    "submission_may_have_succeeded": True,
                    "ambiguity_resolved": True,
                    "broker_position_confirmed": broker_position_id > 0,
                    "tracking_retained": True,
                    "automatic_retry": False,
                    "retryable": False,
                    "acknowledgement_timeout": ack_timeout_reconciled,
                },
                reason="ctrader_broker_confirmed_tracking_pending",
            )
        except SQLiteBusyError as exc:
            failsafe_close: dict[str, object] = {
                "attempted": False,
                "closed": False,
                "broker_position_id": broker_position_id or None,
            }
            if broker_position_id > 0:
                failsafe_close["attempted"] = True
                try:
                    close_result = close_position(
                        symbol=analysis.symbol,
                        position_id=broker_position_id,
                        quantity_lots=float(broker_order.get("quantity_lots") or trade_quantity),
                    )
                except Exception as close_exc:
                    failsafe_close["error"] = str(close_exc)
                    failsafe_close["outcome"] = (
                        "ambiguous_post_submit"
                        if isinstance(close_exc, CTraderCloseOutcomeAmbiguous)
                        else (
                            "broker_rejected"
                            if isinstance(close_exc, CTraderCloseRejected)
                            else "close_failed"
                        )
                    )
                else:
                    failsafe_close.update(
                        {
                            "closed": bool(
                                isinstance(close_result, dict)
                                and close_result.get("status") == "closed"
                                and close_result.get("verified") is True
                            ),
                            "outcome": "verified_closed",
                            "result": close_result,
                        }
                    )

            log_incident(
                "error",
                "sqlite_persistence_busy_post_submit",
                f"SQLite became busy after cTrader submission for {analysis.symbol}:{analysis.timeframe}; automatic resubmission is blocked.",
                {
                    "intent_id": intent.id,
                    "phase": "broker_confirmed_handoff",
                    "broker_position_id": broker_position_id or None,
                    "error": str(exc),
                    "failsafe_close": failsafe_close,
                    "automatic_resubmission": False,
                    "action_required": (
                        "Keep automatic resubmission blocked. Once SQLite is writable, reconcile the reserved submission "
                        "against broker truth before creating or routing any new order."
                    ),
                },
            )
            return ExecutionResult(
                action_taken=True,
                intent_id=intent.id,
                status="persistence_pending",
                summary=(
                    "Broker submission occurred but SQLite could not persist the canonical handoff; "
                    "fail-safe close was attempted and automatic resubmission is blocked."
                ),
                mode=_ctrader_execution_mode(),
                broker_position_id=broker_position_id or None,
                retryable=False,
            )

        try:
            broker_protection = sync_position_targets(
                symbol=analysis.symbol,
                direction=analysis.signal,
                stop_loss=analysis.stop_loss,
                take_profit=analysis.take_profit,
                position_id=broker_order.get("position_id"),
                reference_price=mark_price,
            )
            broker_order["protection"] = broker_protection
            if broker_protection.get("status") in {"exit_due_stop_loss", "exit_due_take_profit"}:
                try:
                    broker_close = close_position(
                        symbol=analysis.symbol,
                        position_id=int(broker_order.get("position_id") or 0),
                        quantity_lots=float(broker_order.get("quantity_lots") or trade_quantity),
                    )
                except Exception as close_exc:
                    unprotected_close_error = str(close_exc)
                    unprotected_close_failure = close_exc
                    unprotected_close_phase = "immediate_protective_exit"
                    broker_order["protective_close_failed"] = True
                    broker_order["protective_close_error"] = unprotected_close_error
                    broker_order["protection_verified"] = False
                else:
                    broker_order["protective_close"] = broker_close
                    broker_order["protection_verified"] = False
                    update_order_intent_status(
                        intent.id,
                        "failed" if ack_timeout_reconciled else "executed",
                        {
                            "broker_order": broker_order,
                            "outcome_state": (
                                "ack_timeout_reconciled_protective_exit"
                                if ack_timeout_reconciled
                                else "executed"
                            ),
                            "acknowledgement_timeout": ack_timeout_reconciled,
                            "ambiguity_resolved": ack_timeout_reconciled,
                            "retryable": False,
                            "tracking_retained": False,
                        },
                        reason=(
                            "ctrader_order_ack_timeout_reconciled_protective_exit"
                            if ack_timeout_reconciled
                            else "ctrader_immediate_protective_exit"
                        ),
                    )
                    add_trade_audit(
                        event_type="ctrader_protective_exit",
                        symbol=analysis.symbol,
                        timeframe=analysis.timeframe,
                        strategy=analysis.strategy,
                        intent_id=intent.id,
                        summary="Closed newly opened cTrader position because its protective target was already crossed.",
                        details={"protection": broker_protection, "broker_close": broker_close},
                    )
                    return ExecutionResult(
                        action_taken=True,
                        intent_id=intent.id,
                        status="failed" if ack_timeout_reconciled else "executed",
                        summary=broker_protection.get("status") or "protective exit",
                        mode=_ctrader_execution_mode(),
                        broker_position_id=int(broker_order.get("position_id") or 0) or None,
                        retryable=False,
                    )
            else:
                broker_order["protection_verified"] = True
        except Exception as exc:
            broker_order["protection_verified"] = False
            broker_order["protection_error"] = str(exc)
            broker_position_id = int(broker_order.get("position_id") or 0)
            log_incident(
                "error",
                "ctrader_order_unprotected",
                f"cTrader order opened but broker SL/TP could not be verified for {analysis.symbol}:{analysis.timeframe}",
                {
                    "intent_id": intent.id,
                    "broker_position_id": broker_position_id or None,
                    "error": str(exc),
                },
            )
            add_trade_audit(
                event_type="ctrader_order_unprotected",
                symbol=analysis.symbol,
                timeframe=analysis.timeframe,
                strategy=analysis.strategy,
                intent_id=intent.id,
                summary="cTrader order opened without verified broker protection; fail-safe close will be attempted.",
                details={
                    "broker_position_id": broker_position_id or None,
                    "error": str(exc),
                    "stop_loss": analysis.stop_loss,
                    "take_profit": analysis.take_profit,
                },
            )
            try:
                if broker_position_id <= 0:
                    raise RuntimeError("cTrader order did not return a valid broker position id for fail-safe close.")
                broker_close = close_position(
                    symbol=analysis.symbol,
                    position_id=broker_position_id,
                    quantity_lots=float(broker_order.get("quantity_lots") or trade_quantity),
                )
            except Exception as close_exc:
                unprotected_close_error = str(close_exc)
                unprotected_close_failure = close_exc
                unprotected_close_phase = "unprotected_position_failsafe"
                broker_order["failsafe_closed"] = False
                broker_order["failsafe_close_error"] = unprotected_close_error
                log_incident(
                    "error",
                    "ctrader_unprotected_failsafe_close_failed",
                    f"Fail-safe close failed for unprotected cTrader position {analysis.symbol}:{analysis.timeframe}",
                    {
                        "intent_id": intent.id,
                        "broker_position_id": broker_position_id or None,
                        "protection_error": str(exc),
                        "close_error": unprotected_close_error,
                    },
                )
                add_trade_audit(
                    event_type="ctrader_unprotected_failsafe_close_failed",
                    symbol=analysis.symbol,
                    timeframe=analysis.timeframe,
                    strategy=analysis.strategy,
                    intent_id=intent.id,
                    summary="Fail-safe close failed; local tracking will be retained so protection can be retried.",
                    details={
                        "broker_position_id": broker_position_id or None,
                        "protection_error": str(exc),
                        "close_error": unprotected_close_error,
                    },
                )
            else:
                broker_order["failsafe_close"] = broker_close
                broker_order["failsafe_closed"] = True
                update_order_intent_status(
                    intent.id,
                    "failed",
                    {
                        "broker_order": broker_order,
                        "protection_error": str(exc),
                        "failsafe_closed": True,
                        "failsafe_close": broker_close,
                    },
                    reason="ctrader_unprotected_failsafe_closed",
                )
                log_incident(
                    "warning",
                    "ctrader_unprotected_failsafe_closed",
                    f"Closed unprotected cTrader position for {analysis.symbol}:{analysis.timeframe}",
                    {
                        "intent_id": intent.id,
                        "broker_position_id": broker_position_id,
                        "protection_error": str(exc),
                        "broker_close": broker_close,
                    },
                )
                add_trade_audit(
                    event_type="ctrader_unprotected_failsafe_closed",
                    symbol=analysis.symbol,
                    timeframe=analysis.timeframe,
                    strategy=analysis.strategy,
                    intent_id=intent.id,
                    summary="Closed cTrader position because broker SL/TP could not be verified.",
                    details={
                        "broker_position_id": broker_position_id,
                        "protection_error": str(exc),
                        "broker_close": broker_close,
                    },
                )
                return ExecutionResult(
                    action_taken=True,
                    intent_id=intent.id,
                    status="failed",
                    summary="cTrader position was closed by fail-safe because broker protection could not be verified.",
                    mode=_ctrader_execution_mode(),
                    broker_position_id=broker_position_id,
                    retryable=False,
                )

    try:
        created = open_paper_position(
            symbol=analysis.symbol,
            timeframe=analysis.timeframe,
            strategy=analysis.strategy,
            direction=analysis.signal,
            quantity=(
                float((broker_order or {}).get("quantity_lots") or trade_quantity)
                if ack_timeout_reconciled
                else trade_quantity
            ),
            entry_price=(
                float((broker_order or {}).get("entry_price"))
                if ack_timeout_reconciled and (broker_order or {}).get("entry_price") is not None
                else (analysis.entry_price or mark_price)
            ),
            stop_loss=analysis.stop_loss,
            take_profit=analysis.take_profit,
            lifecycle_version_hash=lifecycle_version_hash,
            account_currency=monetary_basis.currency or config.account_currency,
            cash_per_price_unit_per_lot=float(instrument.cash_per_price_unit_per_lot or 1.0),
            instrument_spec_source=instrument.source if instrument.valuation_ready else "unvalued_fallback",
            broker_position_id=(
                int(broker_order.get("position_id") or 0)
                if broker_order and broker_order.get("position_id")
                else None
            ),
        )
    except SQLiteBusyError as exc:
        broker_position_id = (
            int((broker_order or {}).get("position_id") or 0)
            if broker_order
            else 0
        )
        log_incident(
            "error",
            "sqlite_persistence_busy_tracking_deferred",
            f"Deferred local tracker persistence for {analysis.symbol}:{analysis.timeframe} because SQLite is busy.",
            {
                "intent_id": intent.id,
                "phase": "local_tracker_persistence",
                "broker_position_id": broker_position_id or None,
                "broker_confirmed_handoff": bool(broker_order),
                "error": str(exc),
                "automatic_resubmission": False if broker_order else True,
                "action_required": (
                    "Recover the canonical local tracker from the persisted broker-confirmed intent before any new cTrader order."
                    if broker_order
                    else "Release the SQLite writer lock and retry the paper-only persistence step."
                ),
            },
        )
        return ExecutionResult(
            action_taken=bool(broker_order),
            intent_id=intent.id,
            status="persistence_pending",
            summary=(
                "Broker position is durably identified but local tracker persistence is deferred."
                if broker_order
                else "SQLite persistence is busy; local paper position was not created."
            ),
            mode=_ctrader_execution_mode() if broker_order else "paper_only",
            broker_position_id=broker_position_id or None,
            retryable=not bool(broker_order),
        )

    if broker_order and unprotected_close_error is not None:
        close_outcome_state = "close_failed"
        retryable_close = True
        quantity_lots = float((broker_order or {}).get("quantity_lots") or created.quantity)
        if isinstance(unprotected_close_failure, CTraderCloseOutcomeAmbiguous):
            record_ambiguous_close(
                created,
                error=unprotected_close_failure,
                phase=unprotected_close_phase or "untracked_close_handoff",
                quantity_lots=quantity_lots,
            )
            close_outcome_state = "ambiguous_post_submit"
            retryable_close = False
        elif isinstance(unprotected_close_failure, CTraderCloseRejected):
            record_rejected_close(
                created,
                error=unprotected_close_failure,
                phase=unprotected_close_phase or "untracked_close_handoff",
                quantity_lots=quantity_lots,
            )
            close_outcome_state = "broker_rejected"
            retryable_close = False

        update_order_intent_status(
            intent.id,
            "failed",
            {
                "opened_position_id": created.id,
                "execution_mode": _ctrader_execution_tag(broker_order),
                "broker_order": broker_order,
                "failsafe_close_error": unprotected_close_error,
                "close_outcome_state": close_outcome_state,
                "close_retryable": retryable_close,
                "tracking_retained": True,
            },
            reason="ctrader_unprotected_failsafe_close_failed",
        )
        add_trade_audit(
            event_type="ctrader_unprotected_tracking_retained",
            symbol=analysis.symbol,
            timeframe=analysis.timeframe,
            strategy=analysis.strategy,
            intent_id=intent.id,
            position_id=created.id,
            summary="Retained local tracking for an unprotected cTrader position after the fail-safe close also failed.",
            details={
                "broker_position_id": (broker_order or {}).get("position_id"),
                "close_error": unprotected_close_error,
                "stop_loss": analysis.stop_loss,
                "take_profit": analysis.take_profit,
            },
        )
        return ExecutionResult(
            action_taken=True,
            intent_id=intent.id,
            status=(
                "close_ambiguous_pending"
                if close_outcome_state == "ambiguous_post_submit"
                else ("close_rejected" if close_outcome_state == "broker_rejected" else "failed")
            ),
            summary="Broker close was not verified; canonical local tracking was retained.",
            position_id=created.id,
            mode=_ctrader_execution_mode(),
            broker_position_id=(broker_order or {}).get("position_id"),
            retryable=retryable_close,
        )

    if ack_timeout_reconciled:
        update_order_intent_status(
            intent.id,
            "failed",
            {
                "opened_position_id": created.id,
                "execution_mode": _ctrader_execution_tag(broker_order),
                "broker_order": broker_order or {},
                "outcome_state": "ack_timeout_reconciled_broker_position",
                "acknowledgement_timeout": True,
                "submission_may_have_succeeded": True,
                "ambiguity_resolved": True,
                "broker_position_confirmed": True,
                "retryable": False,
                "automatic_retry": False,
                "tracking_retained": True,
            },
            reason="ctrader_order_ack_timeout_reconciled",
        )
        add_trade_audit(
            event_type="ctrader_order_ack_timeout_tracking_retained",
            symbol=analysis.symbol,
            timeframe=analysis.timeframe,
            strategy=analysis.strategy,
            intent_id=intent.id,
            position_id=created.id,
            summary="Tracked the broker-confirmed cTrader position after an acknowledgement timeout without resubmitting the order.",
            details={
                "broker_position_id": (broker_order or {}).get("position_id"),
                "broker_order": broker_order or {},
                "automatic_retry": False,
            },
        )
        return ExecutionResult(
            action_taken=True,
            intent_id=intent.id,
            status="failed",
            summary=(
                "Order acknowledgement timed out, but broker reconciliation confirmed "
                "the cTrader position; local tracking was created and automatic retry stayed blocked."
            ),
            position_id=created.id,
            mode=_ctrader_execution_mode(),
            broker_position_id=(broker_order or {}).get("position_id"),
            retryable=False,
        )

    try:
        update_order_intent_status(
            intent.id,
            "executed",
            {
                "opened_position_id": created.id,
                "execution_mode": _ctrader_execution_tag(broker_order) if broker_order else "paper",
                "broker_order": broker_order or {},
            },
            reason="ctrader_order_executed" if broker_order else "paper_position_opened",
        )
        add_trade_audit(
            event_type="ctrader_order_executed" if broker_order else "paper_signal_open",
            symbol=analysis.symbol,
            timeframe=analysis.timeframe,
            strategy=analysis.strategy,
            intent_id=intent.id,
            position_id=created.id,
            summary=(
                "Executed cTrader order and opened the local tracking position."
                if broker_order
                else "Opened new paper position from accepted signal."
            ),
            details={"entry_price": created.entry_price, "quantity": created.quantity, "broker_order": broker_order or {}},
        )
    except SQLiteBusyError as exc:
        log_incident(
            "error",
            "sqlite_persistence_busy_post_tracking",
            f"SQLite became busy after the local tracker was created for {analysis.symbol}:{analysis.timeframe}.",
            {
                "intent_id": intent.id,
                "position_id": created.id,
                "broker_position_id": (broker_order or {}).get("position_id"),
                "error": str(exc),
                "automatic_resubmission": False,
                "action_required": "Keep the canonical local tracker; retry only intent/audit persistence after SQLite recovers.",
            },
        )
        return ExecutionResult(
            action_taken=True,
            intent_id=intent.id,
            status="persistence_pending",
            summary="Local tracker is durable; final intent/audit persistence is deferred and no new order is allowed.",
            position_id=created.id,
            mode=_ctrader_execution_mode() if broker_order else "paper_only",
            broker_position_id=(broker_order or {}).get("position_id"),
            retryable=False,
        )

    return ExecutionResult(
        action_taken=True,
        intent_id=intent.id,
        status="executed",
        summary="position flipped and opened" if flipped else "position opened",
        position_id=created.id,
        mode=_ctrader_execution_mode() if broker_order else "paper_only",
        broker_position_id=(broker_order or {}).get("position_id"),
    )
