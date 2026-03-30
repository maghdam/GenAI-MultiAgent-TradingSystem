from __future__ import annotations

from dataclasses import dataclass

from backend.domain.models import EngineConfig, PaperPosition, StrategyAnalysis, WatchlistItem
from backend.services.paper_book import reconcile_position
from backend.services.quantity_rules import derive_auto_quantity, evaluate_order_quantity
from backend.services.risk_engine import evaluate_risk
from backend.storage.repositories import (
    add_trade_audit,
    close_paper_position,
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
    intent_id: int
    status: str
    summary: str
    position_id: int | None = None


def _refresh_open_position(item: WatchlistItem, mark_price: float) -> PaperPosition | None:
    position = get_open_position(item.symbol.upper(), item.timeframe.upper())
    if not position:
        return None
    reconcile_position(position, mark_price)
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
    position = _refresh_open_position(watch_item, mark_price)
    sizing = (
        derive_auto_quantity(config, analysis, mark_price)
        if source != "manual"
        else None
    )
    requested_quantity = (
        float(quantity if quantity is not None else (config.paper_trade_size or 1.0))
        if source == "manual"
        else float((sizing.requested_quantity if sizing else None) or (config.paper_trade_size or 1.0))
    )
    quantity_decision = evaluate_order_quantity(analysis.symbol, requested_quantity, source)

    if not quantity_decision.accepted:
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
            details={
                **(sizing.details if sizing else {}),
                **quantity_decision.details,
                "analysis": analysis.model_dump(mode="json"),
                "source": source,
            },
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
        details={
            **(sizing.details if sizing else {}),
            **quantity_decision.details,
            **risk.details,
            "analysis": analysis.model_dump(mode="json"),
            "source": source,
        },
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

    if position and position.direction != analysis.signal:
        closed = close_paper_position(position.id, mark_price, "signal_flip")
        update_order_intent_status(intent.id, "accepted", {"closed_position_id": closed.id, "flip": True})
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
        update_paper_position_targets(position.id, analysis.stop_loss, analysis.take_profit)
        update_order_intent_status(intent.id, "executed", {"updated_position_id": position.id})
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

    created = open_paper_position(
        symbol=analysis.symbol,
        timeframe=analysis.timeframe,
        strategy=analysis.strategy,
        direction=analysis.signal,
        quantity=trade_quantity,
        entry_price=analysis.entry_price or mark_price,
        stop_loss=analysis.stop_loss,
        take_profit=analysis.take_profit,
    )
    update_order_intent_status(intent.id, "executed", {"opened_position_id": created.id})
    add_trade_audit(
        event_type="paper_signal_open",
        symbol=analysis.symbol,
        timeframe=analysis.timeframe,
        strategy=analysis.strategy,
        intent_id=intent.id,
        position_id=created.id,
        summary="Opened new paper position from accepted signal.",
        details={"entry_price": created.entry_price, "quantity": created.quantity},
    )
    return ExecutionResult(
        action_taken=True,
        intent_id=intent.id,
        status="executed",
        summary="position flipped and opened" if flipped else "position opened",
        position_id=created.id,
    )
