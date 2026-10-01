from __future__ import annotations

from dataclasses import dataclass
from time import sleep
from uuid import uuid4

from backend.domain.models import EngineConfig, PaperPosition, StrategyAnalysis, WatchlistItem
from backend.services.broker import (
    DemoOrderAcknowledgementTimeout,
    close_demo_position,
    get_broker_account_snapshot,
    get_broker_status,
    get_demo_symbol_execution_readiness,
    get_instrument_spec,
    list_positions,
    place_demo_market_order,
    sync_demo_position_targets,
)
from backend.services.paper_book import apply_mark, reconcile_position
from backend.services.broker_ledger import (
    close_local_position_after_broker_close,
    close_local_position_from_broker,
    reconcile_open_demo_position_ledger,
)
from backend.services.broker_position_match import match_broker_position
from backend.services.financial_units import resolve_monetary_basis
from backend.services.quantity_rules import derive_auto_quantity, evaluate_order_quantity
from backend.services.risk_engine import evaluate_risk
from backend.storage.repositories import (
    add_trade_audit,
    close_paper_position,
    create_decision_record,
    create_order_intent,
    get_open_position,
    list_order_intents,
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


def _broker_position_id(row: dict[str, object]) -> int | None:
    try:
        value = int(row.get("position_id") or 0)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


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
            and intent.status == "failed"
            and intent.symbol.upper() == symbol.upper()
            and intent.timeframe.upper() == timeframe.upper()
        ):
            details = intent.details if isinstance(intent.details, dict) else {}
            if (
                details.get("outcome_state") == "ambiguous_post_submit"
                and details.get("ambiguity_resolved") is not True
            ):
                return intent
    return None


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
    demo_execution: bool = False,
) -> PaperPosition | None:
    position = get_open_position(item.symbol.upper(), item.timeframe.upper())
    if not position:
        return None

    if not demo_execution:
        reconcile_position(position, mark_price)
        return get_open_position(item.symbol.upper(), item.timeframe.upper())

    # In demo mode the broker is the execution source of truth. Never simulate
    # a local-only SL/TP exit while the broker position is still open.
    apply_mark(position, mark_price)
    status = get_broker_status()
    if status.execution_ready:
        match = match_broker_position(position, list_positions())
        if match.status in {"id_mismatch", "legacy_ambiguous"}:
            message = f"Could not safely identify broker position for {position.symbol}:{position.timeframe}."
            log_incident(
                "error",
                "ctrader_demo_position_identity_ambiguous",
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

        ledger_sync = reconcile_open_demo_position_ledger(position, match.row)
        if ledger_sync.get("status") == "partial_close_synced":
            add_trade_audit(
                event_type="ctrader_demo_partial_close_synced",
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
                "ctrader_demo_partial_close_history_pending",
                f"Broker volume decreased for {position.symbol}:{position.timeframe}, but close deal history is not complete yet.",
                {"position_id": position.id, **ledger_sync},
            )
        elif ledger_sync.get("status") in {"volume_increase_mismatch", "unavailable"}:
            message = f"Could not safely reconcile broker volume for {position.symbol}:{position.timeframe}."
            log_incident(
                "error",
                "ctrader_demo_volume_reconciliation_failed",
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
    demo_execution = bool(config.demo_autotrade and watch_item.trading_enabled)
    try:
        position = _refresh_open_position(watch_item, mark_price, demo_execution=demo_execution)
    except BrokerPositionIdentityError as exc:
        return ExecutionResult(
            action_taken=False,
            intent_id=None,
            status="failed",
            summary=str(exc),
            mode="demo_enabled",
            retryable=True,
        )

    # Keep an already-open demo position protected at the broker even when the
    # current strategy result is no_trade or later fails a new-entry risk gate.
    # The local ledger is not allowed to drift silently from broker SL/TP.
    if demo_execution and position:
        try:
            protection = sync_demo_position_targets(
                symbol=position.symbol,
                direction=position.direction,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                position_id=position.broker_position_id,
                reference_price=mark_price,
            )
            if protection.get("status") in {"exit_due_stop_loss", "exit_due_take_profit"}:
                broker_close = close_demo_position(
                    symbol=position.symbol,
                    position_id=int(protection.get("position_id") or 0),
                    quantity_lots=float(protection.get("quantity_lots") or position.quantity),
                )
                reason = (
                    "broker_stop_loss"
                    if protection.get("status") == "exit_due_stop_loss"
                    else "broker_take_profit"
                )
                closed = close_local_position_after_broker_close(
                    position,
                    broker_close=broker_close,
                    fallback_price=mark_price,
                    reason=reason,
                )
                add_trade_audit(
                    event_type="ctrader_demo_protective_exit",
                    symbol=position.symbol,
                    timeframe=position.timeframe,
                    strategy=position.strategy,
                    position_id=closed.id,
                    summary="Closed cTrader demo position because an intended protective target was already crossed.",
                    details={"protection": protection, "broker_close": broker_close},
                )
                return ExecutionResult(
                    action_taken=True,
                    intent_id=None,
                    status="executed",
                    summary=reason,
                    position_id=closed.id,
                    mode="demo_enabled",
                    broker_position_id=int(protection.get("position_id") or 0) or None,
                )
            if protection.get("status") == "synced":
                add_trade_audit(
                    event_type="ctrader_demo_protection_repaired",
                    symbol=position.symbol,
                    timeframe=position.timeframe,
                    strategy=position.strategy,
                    position_id=position.id,
                    summary="Repaired broker SL/TP from the local tracking position.",
                    details=protection,
                )
        except Exception as exc:
            log_incident(
                "error",
                "ctrader_demo_protection_sync_failed",
                f"Could not synchronize broker protection for {position.symbol}:{position.timeframe}",
                {"position_id": position.id, "error": str(exc)},
            )
            add_trade_audit(
                event_type="ctrader_demo_protection_sync_failed",
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
                mode="demo_enabled",
                retryable=True,
            )

    # Broker symbol metadata arrives asynchronously after account authorization.
    # If demo execution is enabled, never attempt an order until the exact
    # symbol contract is loaded. Mark this as retryable so the engine can
    # revisit the same bar without risking a duplicate order.
    if demo_execution and analysis.signal != "no_trade":
        broker_ready, broker_reason = get_demo_symbol_execution_readiness(analysis.symbol)
        if not broker_ready:
            log_incident(
                "warning",
                "ctrader_demo_symbol_not_ready",
                f"Deferred cTrader demo execution for {analysis.symbol}:{analysis.timeframe}",
                {"reason": broker_reason, "retryable": True},
            )
            add_trade_audit(
                event_type="ctrader_demo_order_deferred",
                symbol=analysis.symbol,
                timeframe=analysis.timeframe,
                strategy=analysis.strategy,
                summary="Demo order deferred until broker symbol metadata is ready.",
                details={"reason": broker_reason},
            )
            return ExecutionResult(
                action_taken=False,
                intent_id=None,
                status="deferred",
                summary=broker_reason,
                mode="demo_enabled",
                retryable=True,
            )

    monetary_basis = resolve_monetary_basis(config)
    if demo_execution and analysis.signal != "no_trade":
        account_snapshot = get_broker_account_snapshot()
        monetary_basis = resolve_monetary_basis(
            config,
            demo_execution=True,
            account_snapshot=account_snapshot,
        )
        if not monetary_basis.verified:
            reason = monetary_basis.reason or "cTrader account monetary snapshot is unavailable."
            log_incident(
                "warning",
                "ctrader_demo_account_snapshot_not_ready",
                f"Deferred cTrader demo execution for {analysis.symbol}:{analysis.timeframe}",
                {"reason": reason, **monetary_basis.as_details(), "retryable": True},
            )
            add_trade_audit(
                event_type="ctrader_demo_order_deferred",
                symbol=analysis.symbol,
                timeframe=analysis.timeframe,
                strategy=analysis.strategy,
                summary="Demo order deferred until broker monetary account data is verified.",
                details={"reason": reason, **monetary_basis.as_details()},
            )
            return ExecutionResult(
                action_taken=False,
                intent_id=None,
                status="deferred",
                summary=reason,
                mode="demo_enabled",
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
    flipped = False
    intent_quantity = position.quantity if position and position.direction == analysis.signal else trade_quantity
    intent_reasons = list(risk.reasons)
    if sizing:
        intent_reasons = [*sizing.reasons, *intent_reasons]
    if quantity_decision.details.get("quantity_normalized"):
        intent_reasons = [*quantity_decision.reasons, *intent_reasons]

    if (
        demo_execution
        and position is None
        and risk.accepted
        and risk.intent_type == "open"
    ):
        unresolved_ack = _unresolved_order_ack_timeout(
            analysis.symbol,
            analysis.timeframe,
        )
        if unresolved_ack is not None:
            return ExecutionResult(
                action_taken=False,
                intent_id=unresolved_ack.id,
                status="blocked",
                summary=(
                    "Automatic demo order blocked because a prior post-submission "
                    "acknowledgement timeout is still ambiguous. Reconcile broker truth "
                    "before any new order is allowed."
                ),
                mode="demo_enabled",
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

    if demo_execution and position and position.direction != analysis.signal:
        update_order_intent_status(
            intent.id,
            "failed",
            {"reason": "demo_signal_flip_requires_reconciliation"},
            reason="demo_signal_flip_blocked",
        )
        log_incident(
            "warning",
            "demo_signal_flip_blocked",
            f"Blocked cTrader demo signal flip for {analysis.symbol}:{analysis.timeframe}",
            {"intent_id": intent.id, "position_id": position.id},
        )
        add_trade_audit(
            event_type="ctrader_demo_order_blocked",
            symbol=analysis.symbol,
            timeframe=analysis.timeframe,
            strategy=analysis.strategy,
            intent_id=intent.id,
            position_id=position.id,
            summary="Demo order blocked because the opposite broker position must be reconciled first.",
            details={"direction": analysis.signal},
        )
        return ExecutionResult(
            action_taken=False,
            intent_id=intent.id,
            status="failed",
            summary="Demo signal flip blocked pending broker reconciliation.",
            position_id=position.id,
            mode="demo_enabled",
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
        if demo_execution:
            try:
                broker_protection = sync_demo_position_targets(
                    symbol=position.symbol,
                    direction=position.direction,
                    stop_loss=analysis.stop_loss,
                    take_profit=analysis.take_profit,
                    position_id=position.broker_position_id,
                    reference_price=mark_price,
                )
                if broker_protection.get("status") in {"exit_due_stop_loss", "exit_due_take_profit"}:
                    broker_close = close_demo_position(
                        symbol=position.symbol,
                        position_id=int(broker_protection.get("position_id") or 0),
                        quantity_lots=float(broker_protection.get("quantity_lots") or position.quantity),
                    )
                    reason = (
                        "broker_stop_loss"
                        if broker_protection.get("status") == "exit_due_stop_loss"
                        else "broker_take_profit"
                    )
                    closed = close_local_position_after_broker_close(
                    position,
                    broker_close=broker_close,
                    fallback_price=mark_price,
                    reason=reason,
                )
                    update_order_intent_status(
                        intent.id,
                        "executed",
                        {
                            "closed_position_id": closed.id,
                            "broker_protection": broker_protection,
                            "broker_close": broker_close,
                        },
                        reason="protective_exit_before_target_update",
                    )
                    add_trade_audit(
                        event_type="ctrader_demo_protective_exit",
                        symbol=analysis.symbol,
                        timeframe=analysis.timeframe,
                        strategy=analysis.strategy,
                        intent_id=intent.id,
                        position_id=closed.id,
                        summary="Closed cTrader demo position because the refreshed target was already crossed.",
                        details={"protection": broker_protection, "broker_close": broker_close},
                    )
                    return ExecutionResult(
                        action_taken=True,
                        intent_id=intent.id,
                        status="executed",
                        summary=reason,
                        position_id=closed.id,
                        mode="demo_enabled",
                        broker_position_id=int(broker_protection.get("position_id") or 0) or None,
                    )
            except Exception as exc:
                update_order_intent_status(
                    intent.id,
                    "failed",
                    {"broker": "ctrader", "error": str(exc)},
                    reason="ctrader_demo_target_update_failed",
                )
                log_incident(
                    "error",
                    "ctrader_demo_target_update_failed",
                    f"Could not update broker targets for {analysis.symbol}:{analysis.timeframe}",
                    {"intent_id": intent.id, "position_id": position.id, "error": str(exc)},
                )
                add_trade_audit(
                    event_type="ctrader_demo_target_update_failed",
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
                    mode="demo_enabled",
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
    ack_timeout_reconciled = False
    if demo_execution:
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
            broker_order = place_demo_market_order(
                symbol=analysis.symbol,
                direction=analysis.signal,
                quantity_lots=trade_quantity,
                stop_loss=analysis.stop_loss,
                take_profit=analysis.take_profit,
                client_msg_id=client_msg_id,
            )
        except DemoOrderAcknowledgementTimeout as exc:
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
                    "account_type": "demo",
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
                    "ctrader_demo_order_ack_timeout_reconciled",
                    f"cTrader demo order acknowledgement timed out for {analysis.symbol}:{analysis.timeframe}, but broker reconciliation confirmed one new position.",
                    {
                        "intent_id": intent.id,
                        "client_msg_id": client_msg_id,
                        "baseline_position_ids": sorted(baseline_ids),
                        "reconciliation": reconciliation,
                        "automatic_retry": False,
                    },
                )
                add_trade_audit(
                    event_type="ctrader_demo_order_ack_timeout_reconciled",
                    symbol=analysis.symbol,
                    timeframe=analysis.timeframe,
                    strategy=analysis.strategy,
                    intent_id=intent.id,
                    summary="Broker truth confirmed one new demo position after the order acknowledgement timed out.",
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
                    "account_type": "demo",
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
                    reason="ctrader_demo_order_ack_timeout_ambiguous",
                )
                log_incident(
                    "error",
                    "ctrader_demo_order_ack_timeout_ambiguous",
                    f"cTrader demo order acknowledgement timed out after submission for {analysis.symbol}:{analysis.timeframe}; automatic resubmission is blocked.",
                    {"intent_id": intent.id, **details},
                )
                add_trade_audit(
                    event_type="ctrader_demo_order_ack_timeout_ambiguous",
                    symbol=analysis.symbol,
                    timeframe=analysis.timeframe,
                    strategy=analysis.strategy,
                    intent_id=intent.id,
                    summary="Demo order acknowledgement timed out after submission; broker outcome remains ambiguous and automatic retry is blocked.",
                    details=details,
                )
                return ExecutionResult(
                    action_taken=False,
                    intent_id=intent.id,
                    status="failed",
                    summary=(
                        "cTrader demo order acknowledgement timed out after submission; "
                        "broker outcome is ambiguous and automatic retry is blocked."
                    ),
                    mode="demo_enabled",
                    retryable=False,
                )
        except Exception as exc:
            update_order_intent_status(
                intent.id,
                "failed",
                {"broker": "ctrader", "account_type": "demo", "error": str(exc)},
                reason="ctrader_demo_order_failed",
            )
            log_incident(
                "error",
                "ctrader_demo_order_failed",
                f"cTrader demo order failed for {analysis.symbol}:{analysis.timeframe}",
                {"intent_id": intent.id, "error": str(exc)},
            )
            add_trade_audit(
                event_type="ctrader_demo_order_failed",
                symbol=analysis.symbol,
                timeframe=analysis.timeframe,
                strategy=analysis.strategy,
                intent_id=intent.id,
                summary="cTrader demo order was not executed.",
                details={"error": str(exc), "quantity": trade_quantity},
            )
            return ExecutionResult(
                action_taken=False,
                intent_id=intent.id,
                status="failed",
                summary=str(exc),
                mode="demo_enabled",
            )

    if broker_order:
        try:
            broker_protection = sync_demo_position_targets(
                symbol=analysis.symbol,
                direction=analysis.signal,
                stop_loss=analysis.stop_loss,
                take_profit=analysis.take_profit,
                position_id=broker_order.get("position_id"),
                reference_price=mark_price,
            )
            broker_order["protection"] = broker_protection
            if broker_protection.get("status") in {"exit_due_stop_loss", "exit_due_take_profit"}:
                broker_close = close_demo_position(
                    symbol=analysis.symbol,
                    position_id=int(broker_order.get("position_id") or 0),
                    quantity_lots=float(broker_order.get("quantity_lots") or trade_quantity),
                )
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
                        "ctrader_demo_order_ack_timeout_reconciled_protective_exit"
                        if ack_timeout_reconciled
                        else "ctrader_demo_immediate_protective_exit"
                    ),
                )
                add_trade_audit(
                    event_type="ctrader_demo_protective_exit",
                    symbol=analysis.symbol,
                    timeframe=analysis.timeframe,
                    strategy=analysis.strategy,
                    intent_id=intent.id,
                    summary="Closed newly opened demo position because its protective target was already crossed.",
                    details={"protection": broker_protection, "broker_close": broker_close},
                )
                return ExecutionResult(
                    action_taken=True,
                    intent_id=intent.id,
                    status="failed" if ack_timeout_reconciled else "executed",
                    summary=broker_protection.get("status") or "protective exit",
                    mode="demo_enabled",
                    broker_position_id=int(broker_order.get("position_id") or 0) or None,
                    retryable=False,
                )
            broker_order["protection_verified"] = True
        except Exception as exc:
            broker_order["protection_verified"] = False
            broker_order["protection_error"] = str(exc)
            broker_position_id = int(broker_order.get("position_id") or 0)
            log_incident(
                "error",
                "ctrader_demo_order_unprotected",
                f"Demo order opened but broker SL/TP could not be verified for {analysis.symbol}:{analysis.timeframe}",
                {
                    "intent_id": intent.id,
                    "broker_position_id": broker_position_id or None,
                    "error": str(exc),
                },
            )
            add_trade_audit(
                event_type="ctrader_demo_order_unprotected",
                symbol=analysis.symbol,
                timeframe=analysis.timeframe,
                strategy=analysis.strategy,
                intent_id=intent.id,
                summary="Demo order opened without verified broker protection; fail-safe close will be attempted.",
                details={
                    "broker_position_id": broker_position_id or None,
                    "error": str(exc),
                    "stop_loss": analysis.stop_loss,
                    "take_profit": analysis.take_profit,
                },
            )
            try:
                if broker_position_id <= 0:
                    raise RuntimeError("cTrader demo order did not return a valid broker position id for fail-safe close.")
                broker_close = close_demo_position(
                    symbol=analysis.symbol,
                    position_id=broker_position_id,
                    quantity_lots=float(broker_order.get("quantity_lots") or trade_quantity),
                )
            except Exception as close_exc:
                unprotected_close_error = str(close_exc)
                broker_order["failsafe_closed"] = False
                broker_order["failsafe_close_error"] = unprotected_close_error
                log_incident(
                    "error",
                    "ctrader_demo_unprotected_failsafe_close_failed",
                    f"Fail-safe close failed for unprotected cTrader demo position {analysis.symbol}:{analysis.timeframe}",
                    {
                        "intent_id": intent.id,
                        "broker_position_id": broker_position_id or None,
                        "protection_error": str(exc),
                        "close_error": unprotected_close_error,
                    },
                )
                add_trade_audit(
                    event_type="ctrader_demo_unprotected_failsafe_close_failed",
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
                    reason="ctrader_demo_unprotected_failsafe_closed",
                )
                log_incident(
                    "warning",
                    "ctrader_demo_unprotected_failsafe_closed",
                    f"Closed unprotected cTrader demo position for {analysis.symbol}:{analysis.timeframe}",
                    {
                        "intent_id": intent.id,
                        "broker_position_id": broker_position_id,
                        "protection_error": str(exc),
                        "broker_close": broker_close,
                    },
                )
                add_trade_audit(
                    event_type="ctrader_demo_unprotected_failsafe_closed",
                    symbol=analysis.symbol,
                    timeframe=analysis.timeframe,
                    strategy=analysis.strategy,
                    intent_id=intent.id,
                    summary="Closed cTrader demo position because broker SL/TP could not be verified.",
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
                    summary="Demo position was closed by fail-safe because broker protection could not be verified.",
                    mode="demo_enabled",
                    broker_position_id=broker_position_id,
                    retryable=False,
                )

    created = open_paper_position(
        symbol=analysis.symbol,
        timeframe=analysis.timeframe,
        strategy=analysis.strategy,
        direction=analysis.signal,
        quantity=trade_quantity,
        entry_price=analysis.entry_price or mark_price,
        stop_loss=analysis.stop_loss,
        take_profit=analysis.take_profit,
        lifecycle_version_hash=lifecycle_version_hash,
        account_currency=monetary_basis.currency or config.account_currency,
        cash_per_price_unit_per_lot=float(instrument.cash_per_price_unit_per_lot or 1.0),
        instrument_spec_source=instrument.source if instrument.valuation_ready else "unvalued_fallback",
        broker_position_id=(
            int(broker_order.get("position_id") or 0) if broker_order and broker_order.get("position_id") else None
        ),
    )
    if broker_order and unprotected_close_error is not None:
        update_order_intent_status(
            intent.id,
            "failed",
            {
                "opened_position_id": created.id,
                "execution_mode": "ctrader_demo",
                "broker_order": broker_order,
                "failsafe_close_error": unprotected_close_error,
                "tracking_retained": True,
            },
            reason="ctrader_demo_unprotected_failsafe_close_failed",
        )
        add_trade_audit(
            event_type="ctrader_demo_unprotected_tracking_retained",
            symbol=analysis.symbol,
            timeframe=analysis.timeframe,
            strategy=analysis.strategy,
            intent_id=intent.id,
            position_id=created.id,
            summary="Retained local tracking for an unprotected demo position after the fail-safe close also failed.",
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
            status="failed",
            summary="Broker protection failed and the fail-safe close failed; local tracking was retained for retry.",
            position_id=created.id,
            mode="demo_enabled",
            broker_position_id=(broker_order or {}).get("position_id"),
            retryable=True,
        )

    if ack_timeout_reconciled:
        update_order_intent_status(
            intent.id,
            "failed",
            {
                "opened_position_id": created.id,
                "execution_mode": "ctrader_demo",
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
            reason="ctrader_demo_order_ack_timeout_reconciled",
        )
        add_trade_audit(
            event_type="ctrader_demo_order_ack_timeout_tracking_retained",
            symbol=analysis.symbol,
            timeframe=analysis.timeframe,
            strategy=analysis.strategy,
            intent_id=intent.id,
            position_id=created.id,
            summary="Tracked the broker-confirmed demo position after an acknowledgement timeout without resubmitting the order.",
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
                "the demo position; local tracking was created and automatic retry stayed blocked."
            ),
            position_id=created.id,
            mode="demo_enabled",
            broker_position_id=(broker_order or {}).get("position_id"),
            retryable=False,
        )

    update_order_intent_status(
        intent.id,
        "executed",
        {
            "opened_position_id": created.id,
            "execution_mode": "ctrader_demo" if broker_order else "paper",
            "broker_order": broker_order or {},
        },
        reason="ctrader_demo_order_executed" if broker_order else "paper_position_opened",
    )
    add_trade_audit(
        event_type="ctrader_demo_order_executed" if broker_order else "paper_signal_open",
        symbol=analysis.symbol,
        timeframe=analysis.timeframe,
        strategy=analysis.strategy,
        intent_id=intent.id,
        position_id=created.id,
        summary=(
            "Executed cTrader demo order and opened the local tracking position."
            if broker_order
            else "Opened new paper position from accepted signal."
        ),
        details={"entry_price": created.entry_price, "quantity": created.quantity, "broker_order": broker_order or {}},
    )
    return ExecutionResult(
        action_taken=True,
        intent_id=intent.id,
        status="executed",
        summary="position flipped and opened" if flipped else "position opened",
        position_id=created.id,
        mode="demo_enabled" if broker_order else "paper_only",
        broker_position_id=(broker_order or {}).get("position_id"),
    )
