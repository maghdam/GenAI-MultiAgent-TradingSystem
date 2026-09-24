from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from backend.domain.models import EngineConfig, PaperPosition, StrategyAnalysis, WatchlistItem
from backend.services.broker import (
    close_demo_position,
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
)
from backend.services.quantity_rules import derive_auto_quantity, evaluate_order_quantity
from backend.services.risk_engine import evaluate_risk
from backend.storage.repositories import (
    add_trade_audit,
    close_paper_position,
    create_decision_record,
    create_order_intent,
    get_open_position,
    log_incident,
    open_paper_position,
    update_order_intent_status,
    update_paper_position_targets,
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
        expected_side = "buy" if position.direction == "long" else "sell"
        broker_match = next(
            (
                row
                for row in list_positions()
                if str(row.get("symbol") or "").upper() == position.symbol.upper()
                and str(row.get("direction") or "").lower() == expected_side
            ),
            None,
        )
        if broker_match is None:
            close_local_position_from_broker(
                position,
                fallback_price=mark_price,
                fallback_reason="broker_position_closed",
            )
            return None

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
    position = _refresh_open_position(watch_item, mark_price, demo_execution=demo_execution)

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

    instrument = get_instrument_spec(analysis.symbol, config.account_currency)
    configured_quantity = watch_item.lot_size if source != "manual" else None
    sizing = (
        derive_auto_quantity(config, analysis, mark_price)
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
            summary="Automatic execution blocked by monetary sizing requirements.",
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
            summary="Signal rejected by symbol quantity rules.",
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
    )
    flipped = False
    intent_quantity = position.quantity if position and position.direction == analysis.signal else trade_quantity
    intent_reasons = list(risk.reasons)
    if sizing:
        intent_reasons = [*sizing.reasons, *intent_reasons]
    if quantity_decision.details.get("quantity_normalized"):
        intent_reasons = [*quantity_decision.reasons, *intent_reasons]

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
            summary="Signal rejected by V2 risk engine.",
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
    if demo_execution:
        try:
            broker_order = place_demo_market_order(
                symbol=analysis.symbol,
                direction=analysis.signal,
                quantity_lots=trade_quantity,
                stop_loss=analysis.stop_loss,
                take_profit=analysis.take_profit,
                client_msg_id=f"tradeagent-intent-{intent.id}",
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
                    "executed",
                    {"broker_order": broker_order},
                    reason="ctrader_demo_immediate_protective_exit",
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
                    status="executed",
                    summary=broker_protection.get("status") or "protective exit",
                    mode="demo_enabled",
                    broker_position_id=int(broker_order.get("position_id") or 0) or None,
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

    created = open_paper_position(
        symbol=analysis.symbol,
        timeframe=analysis.timeframe,
        strategy=analysis.strategy,
        direction=analysis.signal,
        quantity=trade_quantity,
        entry_price=analysis.entry_price or mark_price,
        stop_loss=analysis.stop_loss,
        take_profit=analysis.take_profit,
        account_currency=config.account_currency,
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
