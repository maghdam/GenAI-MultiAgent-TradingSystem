from __future__ import annotations

from time import sleep
from typing import Any
from uuid import uuid4

from backend.services.alpaca_paper import (
    AlpacaPaperError,
    close_position,
    get_account,
    get_open_position,
    get_order_by_client_order_id,
    get_order_by_id,
    list_positions,
    submit_market_order,
)
from backend.storage.repositories import (
    add_trade_audit,
    create_order_intent,
    list_order_intents,
    update_order_intent_status,
    unresolved_alpaca_intent_for_symbol,
)


_TERMINAL_SUCCESS = {"filled"}
_TERMINAL_CANCELLED = {"canceled", "expired"}
_TERMINAL_FAILED = {"rejected", "suspended"}
_OPEN_BROKER_STATES = {
    "new",
    "partially_filled",
    "accepted",
    "pending_new",
    "accepted_for_bidding",
    "stopped",
    "pending_cancel",
    "pending_replace",
    "held",
    "done_for_day",
    "calculated",
}


def _new_client_order_id(intent_id: int) -> str:
    return f"tradeagent-alpaca-{int(intent_id)}-{uuid4().hex[:16]}"


def _execution_ready_account() -> dict[str, object]:
    account = get_account()
    if str(account.get("environment") or "") != "paper":
        raise AlpacaPaperError("Alpaca execution is restricted to the Paper environment.")
    if str(account.get("status") or "").upper() != "ACTIVE":
        raise AlpacaPaperError("Alpaca Paper account is not ACTIVE.")
    if bool(account.get("trading_blocked")):
        raise AlpacaPaperError("Alpaca Paper account is trading-blocked.")
    if bool(account.get("account_blocked")):
        raise AlpacaPaperError("Alpaca Paper account is account-blocked.")
    if bool(account.get("trade_suspended_by_user")):
        raise AlpacaPaperError("Alpaca Paper trading is suspended by the user.")
    return account


def _unresolved_intent_for_symbol(symbol: str):
    return unresolved_alpaca_intent_for_symbol(str(symbol or ""))


def _lookup_order_with_retry(client_order_id: str) -> dict[str, object] | None:
    last: dict[str, object] | None = None
    for attempt in range(3):
        last = get_order_by_client_order_id(client_order_id)
        if last is not None:
            return last
        if attempt < 2:
            sleep(0.2)
    return last


def _close_fill_pnl_details(intent, order: dict[str, object]) -> dict[str, object]:
    if intent.intent_type != "close" or str(order.get("status") or "").lower() != "filled":
        return {}
    details = intent.details if isinstance(intent.details, dict) else {}
    position = details.get("position_before_close")
    if not isinstance(position, dict):
        return {}
    try:
        entry_price = float(position.get("avg_entry_price") or 0.0)
        exit_price = float(order.get("filled_avg_price") or 0.0)
        filled_qty = float(order.get("filled_qty") or 0.0)
    except (TypeError, ValueError):
        return {}
    if entry_price <= 0 or exit_price <= 0 or filled_qty <= 0:
        return {}

    side = str(position.get("side") or "").lower()
    direction = 1.0 if side == "long" else -1.0 if side == "short" else 0.0
    if direction == 0.0:
        return {}
    realized = (exit_price - entry_price) * filled_qty * direction
    return {
        "realized_pnl_estimate": realized,
        "realized_pnl_currency": "USD",
        "realized_pnl_source": "alpaca_full_close_fill_estimate",
        "position_entry_price": entry_price,
        "close_fill_price": exit_price,
        "close_filled_qty": filled_qty,
    }


def _audit_for_truth(intent, order: dict[str, object], *, source: str) -> None:
    broker_status = str(order.get("status") or "").lower()
    if broker_status == "filled":
        event_type = (
            "alpaca_paper_position_close_filled"
            if intent.intent_type == "close"
            else "alpaca_paper_order_filled"
        )
        summary = "Alpaca Paper broker order filled."
    elif broker_status in _TERMINAL_CANCELLED:
        event_type = "alpaca_paper_order_cancelled"
        summary = "Alpaca Paper broker order reached a cancelled terminal state."
    elif broker_status in _TERMINAL_FAILED:
        event_type = "alpaca_paper_order_failed"
        summary = "Alpaca Paper broker order reached a failed terminal state."
    else:
        event_type = "alpaca_paper_order_acknowledged"
        summary = "Alpaca Paper broker order acknowledged and remains non-terminal."

    add_trade_audit(
        event_type=event_type,
        symbol=intent.symbol,
        timeframe=intent.timeframe,
        strategy=intent.strategy,
        intent_id=intent.id,
        summary=summary,
        details={
            "broker_provider": "alpaca",
            "broker_environment": "paper",
            "broker_order": order,
            "reconciliation_source": source,
            **(
                {"runtime_execution": intent.details.get("runtime_execution")}
                if isinstance(intent.details, dict) and intent.details.get("runtime_execution")
                else {}
            ),
            **_close_fill_pnl_details(intent, order),
        },
    )


def _apply_order_truth(
    intent,
    order: dict[str, object],
    *,
    source: str,
) -> dict[str, object]:
    broker_status = str(order.get("status") or "").lower()
    details = {
        "broker_provider": "alpaca",
        "broker_environment": "paper",
        "broker_order": order,
        "client_order_id": order.get("client_order_id")
        or (intent.details or {}).get("client_order_id"),
        "reconciliation_source": source,
        "reconciliation_resolved": broker_status
        in (_TERMINAL_SUCCESS | _TERMINAL_CANCELLED | _TERMINAL_FAILED),
        "outcome_state": f"broker_{broker_status or 'unknown'}",
        "automatic_retry": False,
    }

    if broker_status in _TERMINAL_SUCCESS:
        updated = update_order_intent_status(
            intent.id,
            "executed",
            details,
            reason="alpaca_paper_broker_filled",
        )
    elif broker_status in _TERMINAL_CANCELLED:
        updated = update_order_intent_status(
            intent.id,
            "cancelled",
            details,
            reason="alpaca_paper_broker_cancelled",
        )
    elif broker_status in _TERMINAL_FAILED:
        updated = update_order_intent_status(
            intent.id,
            "failed",
            {**details, "retryable": True},
            reason="alpaca_paper_broker_failed",
        )
    else:
        updated = update_order_intent_status(
            intent.id,
            "accepted",
            {**details, "reconciliation_resolved": False},
            reason="alpaca_paper_broker_acknowledged",
        )

    _audit_for_truth(updated, order, source=source)
    return {
        "action_taken": broker_status in (_TERMINAL_SUCCESS | _OPEN_BROKER_STATES),
        "intent_id": updated.id,
        "status": updated.status,
        "summary": (
            "Alpaca Paper order filled."
            if updated.status == "executed"
            else "Alpaca Paper order acknowledged; reconciliation remains active."
            if updated.status == "accepted"
            else f"Alpaca Paper order {updated.status}."
        ),
        "broker_order": order,
    }


def _mark_submission_absent(intent, *, source: str) -> dict[str, object]:
    updated = update_order_intent_status(
        intent.id,
        "failed",
        {
            "broker_provider": "alpaca",
            "broker_environment": "paper",
            "outcome_state": "submission_resolved_no_order",
            "reconciliation_source": source,
            "reconciliation_resolved": True,
            "retryable": True,
            "automatic_retry": False,
        },
        reason="alpaca_paper_submission_resolved_no_order",
    )
    add_trade_audit(
        event_type="alpaca_paper_submission_not_found",
        symbol=updated.symbol,
        timeframe=updated.timeframe,
        strategy=updated.strategy,
        intent_id=updated.id,
        summary="Alpaca Paper reconciliation found no broker order for the reserved client order id.",
        details={
            "broker_provider": "alpaca",
            "broker_environment": "paper",
            "client_order_id": updated.details.get("client_order_id"),
            "reconciliation_source": source,
            "automatic_retry": False,
        },
    )
    return {
        "action_taken": False,
        "intent_id": updated.id,
        "status": updated.status,
        "summary": "No Alpaca Paper broker order was found; the reservation was resolved without automatic retry.",
        "broker_order": None,
    }


def submit_paper_market_order(
    *,
    symbol: str,
    asset_class: str,
    side: str,
    qty: float | None,
    notional: float | None,
    timeframe: str,
    strategy: str,
    rationale: str,
    confidence: float = 1.0,
    entry_price: float | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    execution_details: dict[str, Any] | None = None,
) -> dict[str, object]:
    normalized_symbol = str(symbol or "").strip().upper()
    normalized_side = str(side or "").strip().lower()
    normalized_asset_class = str(asset_class or "").strip().lower()
    direction = "long" if normalized_side == "buy" else "short"

    account = _execution_ready_account()
    if normalized_asset_class == "crypto" and normalized_side == "sell":
        raise ValueError(
            "Phase 10.9 does not open short crypto positions; use the explicit close endpoint to flatten an existing crypto position."
        )
    if (
        normalized_asset_class == "us_equity"
        and normalized_side == "sell"
        and not bool(account.get("shorting_enabled"))
    ):
        raise AlpacaPaperError("Alpaca Paper account does not permit short equity entries.")
    if normalized_side == "sell" and notional is not None:
        raise ValueError("Short equity entry requires explicit qty sizing; notional sell entries are not enabled.")

    existing_position = get_open_position(normalized_symbol)
    if existing_position is not None:
        return {
            "action_taken": False,
            "intent_id": None,
            "status": "blocked",
            "summary": (
                "An Alpaca Paper position already exists for this symbol. "
                "Phase 10.9 market submission is entry-only; use the explicit close endpoint to flatten it."
            ),
            "broker_order": None,
            "position": existing_position,
        }

    unresolved = _unresolved_intent_for_symbol(normalized_symbol)
    if unresolved is not None:
        return {
            "action_taken": False,
            "intent_id": unresolved.id,
            "status": "blocked",
            "summary": (
                "An unresolved Alpaca Paper intent already exists for this symbol; "
                "reconcile it before submitting another order."
            ),
            "broker_order": unresolved.details.get("broker_order"),
        }

    intent = create_order_intent(
        symbol=normalized_symbol,
        timeframe=timeframe,
        strategy=strategy,
        direction=direction,
        intent_type="open",
        status="pending",
        confidence=float(confidence),
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        quantity=qty,
        rationale=rationale or "Manual Alpaca Paper market order.",
        details={
            "broker_provider": "alpaca",
            "broker_environment": "paper",
            "asset_class": asset_class,
            "side": normalized_side,
            "qty": qty,
            "notional": notional,
            "automatic_retry": False,
            **(execution_details or {}),
        },
    )
    client_order_id = _new_client_order_id(intent.id)
    intent = update_order_intent_status(
        intent.id,
        "accepted",
        {
            "client_order_id": client_order_id,
            "outcome_state": "submission_reserved",
            "reconciliation_resolved": False,
        },
        reason="alpaca_paper_submission_reserved",
    )
    add_trade_audit(
        event_type="alpaca_paper_order_reserved",
        symbol=intent.symbol,
        timeframe=intent.timeframe,
        strategy=intent.strategy,
        intent_id=intent.id,
        summary="Reserved durable Alpaca Paper order identity before broker submission.",
        details={
            "client_order_id": client_order_id,
            "asset_class": asset_class,
            "side": normalized_side,
            "qty": qty,
            "notional": notional,
            **(execution_details or {}),
        },
    )

    try:
        order = submit_market_order(
            symbol=normalized_symbol,
            asset_class=asset_class,
            side=normalized_side,
            client_order_id=client_order_id,
            qty=qty,
            notional=notional,
        )
        return _apply_order_truth(intent, order, source="submission_response")
    except AlpacaPaperError as exc:
        if not exc.ambiguous:
            updated = update_order_intent_status(
                intent.id,
                "failed",
                {
                    "outcome_state": "submission_rejected",
                    "reconciliation_resolved": True,
                    "retryable": True,
                    "broker_error": str(exc),
                    "broker_status_code": exc.status_code,
                    "broker_details": exc.broker_details,
                    "automatic_retry": False,
                },
                reason="alpaca_paper_submission_rejected",
            )
            add_trade_audit(
                event_type="alpaca_paper_order_failed",
                symbol=updated.symbol,
                timeframe=updated.timeframe,
                strategy=updated.strategy,
                intent_id=updated.id,
                summary="Alpaca Paper rejected the order submission.",
                details={
                    "client_order_id": client_order_id,
                    "broker_error": str(exc),
                    "broker_status_code": exc.status_code,
                    "broker_details": exc.broker_details,
                },
            )
            return {
                "action_taken": False,
                "intent_id": updated.id,
                "status": updated.status,
                "summary": "Alpaca Paper rejected the order submission.",
                "broker_order": None,
            }

        try:
            order = _lookup_order_with_retry(client_order_id)
        except AlpacaPaperError:
            updated = update_order_intent_status(
                intent.id,
                "accepted",
                {
                    "outcome_state": "ambiguous_post_submit",
                    "reconciliation_resolved": False,
                    "submission_may_have_succeeded": True,
                    "automatic_retry": False,
                },
                reason="alpaca_paper_submission_ambiguous",
            )
            return {
                "action_taken": False,
                "intent_id": updated.id,
                "status": "reconciliation_required",
                "summary": "Alpaca Paper submission outcome is ambiguous; automatic resubmission is blocked.",
                "broker_order": None,
            }
        if order is None:
            return _mark_submission_absent(intent, source="post_submit_lookup")
        return _apply_order_truth(intent, order, source="post_submit_lookup")


def close_paper_position(
    *,
    symbol_or_asset_id: str,
    timeframe: str,
    strategy: str,
    rationale: str,
    confidence: float = 1.0,
    execution_details: dict[str, Any] | None = None,
) -> dict[str, object]:
    _execution_ready_account()
    position = get_open_position(symbol_or_asset_id)
    if position is None:
        return {
            "action_taken": False,
            "intent_id": None,
            "status": "already_flat",
            "summary": "Alpaca Paper position is already flat.",
            "broker_order": None,
            "position": None,
        }

    symbol = str(position.get("symbol") or symbol_or_asset_id).upper()
    unresolved = _unresolved_intent_for_symbol(symbol)
    if unresolved is not None:
        return {
            "action_taken": False,
            "intent_id": unresolved.id,
            "status": "blocked",
            "summary": (
                "An unresolved Alpaca Paper intent already exists for this symbol; "
                "reconcile it before closing again."
            ),
            "broker_order": unresolved.details.get("broker_order"),
            "position": position,
        }

    try:
        quantity = float(position.get("qty_available") or position.get("qty") or 0.0)
    except (TypeError, ValueError):
        quantity = 0.0

    intent = create_order_intent(
        symbol=symbol,
        timeframe=timeframe,
        strategy=strategy,
        direction="flat",
        intent_type="close",
        status="pending",
        confidence=float(confidence),
        entry_price=None,
        stop_loss=None,
        take_profit=None,
        quantity=quantity if quantity > 0 else None,
        rationale=rationale or "Manual Alpaca Paper full position close.",
        details={
            "broker_provider": "alpaca",
            "broker_environment": "paper",
            "symbol_or_asset_id": symbol_or_asset_id,
            "asset_id": position.get("asset_id"),
            "position_before_close": position,
            "close_semantics": "full_position_to_flat",
            "automatic_retry": False,
            **(execution_details or {}),
        },
    )
    intent = update_order_intent_status(
        intent.id,
        "accepted",
        {
            "outcome_state": "close_reserved",
            "reconciliation_resolved": False,
        },
        reason="alpaca_paper_close_reserved",
    )
    add_trade_audit(
        event_type="alpaca_paper_close_reserved",
        symbol=intent.symbol,
        timeframe=intent.timeframe,
        strategy=intent.strategy,
        intent_id=intent.id,
        summary="Reserved durable Alpaca Paper full-close intent before broker mutation.",
        details={
            "symbol_or_asset_id": symbol_or_asset_id,
            "asset_id": position.get("asset_id"),
            "position_before_close": position,
            **(execution_details or {}),
        },
    )

    try:
        order = close_position(symbol_or_asset_id)
        result = _apply_order_truth(intent, order, source="close_response")
        result["position_before_close"] = position
        return result
    except AlpacaPaperError as exc:
        if not exc.ambiguous:
            updated = update_order_intent_status(
                intent.id,
                "failed",
                {
                    "outcome_state": "close_rejected",
                    "reconciliation_resolved": True,
                    "retryable": True,
                    "broker_error": str(exc),
                    "broker_status_code": exc.status_code,
                    "broker_details": exc.broker_details,
                    "automatic_retry": False,
                },
                reason="alpaca_paper_close_rejected",
            )
            return {
                "action_taken": False,
                "intent_id": updated.id,
                "status": updated.status,
                "summary": "Alpaca Paper rejected the position close.",
                "broker_order": None,
                "position_before_close": position,
            }

        try:
            current = get_open_position(symbol_or_asset_id)
        except AlpacaPaperError:
            updated = update_order_intent_status(
                intent.id,
                "accepted",
                {
                    "outcome_state": "ambiguous_close",
                    "reconciliation_resolved": False,
                    "automatic_retry": False,
                },
                reason="alpaca_paper_close_ambiguous",
            )
            return {
                "action_taken": False,
                "intent_id": updated.id,
                "status": "reconciliation_required",
                "summary": "Alpaca Paper close outcome is ambiguous; automatic repeat-close is blocked.",
                "broker_order": None,
                "position_before_close": position,
            }

        if current is None:
            updated = update_order_intent_status(
                intent.id,
                "executed",
                {
                    "outcome_state": "flat_confirmed_after_ambiguous_close",
                    "reconciliation_resolved": True,
                    "position_after_close": None,
                    "automatic_retry": False,
                },
                reason="alpaca_paper_flat_confirmed",
            )
            add_trade_audit(
                event_type="alpaca_paper_position_flat_confirmed",
                symbol=updated.symbol,
                timeframe=updated.timeframe,
                strategy=updated.strategy,
                intent_id=updated.id,
                summary="Broker position truth confirmed flat after an ambiguous Alpaca Paper close response.",
                details={"symbol_or_asset_id": symbol_or_asset_id},
            )
            return {
                "action_taken": True,
                "intent_id": updated.id,
                "status": updated.status,
                "summary": "Alpaca Paper position is confirmed flat.",
                "broker_order": None,
                "position_before_close": position,
            }

        updated = update_order_intent_status(
            intent.id,
            "accepted",
            {
                "outcome_state": "ambiguous_close_position_still_open",
                "reconciliation_resolved": False,
                "position_after_close": current,
                "automatic_retry": False,
            },
            reason="alpaca_paper_close_still_ambiguous",
        )
        return {
            "action_taken": False,
            "intent_id": updated.id,
            "status": "reconciliation_required",
            "summary": (
                "Alpaca Paper position is still visible after an ambiguous close; "
                "automatic repeat-close remains blocked."
            ),
            "broker_order": None,
            "position_before_close": position,
            "position_after_close": current,
        }


def reconcile_paper_execution(*, limit: int = 200) -> dict[str, object]:
    _execution_ready_account()
    checked = 0
    executed = 0
    accepted = 0
    failed = 0
    cancelled = 0
    unresolved = 0
    results: list[dict[str, object]] = []

    for intent in list_order_intents(limit):
        details = intent.details if isinstance(intent.details, dict) else {}
        if (
            intent.status != "accepted"
            or details.get("broker_provider") != "alpaca"
            or details.get("broker_environment") != "paper"
        ):
            continue

        # Bracket reservations/parent fills must NEVER be classified by the
        # simple-order reconciler. In particular, a filled parent alone does
        # not establish active protective exit legs. Phase 10.17 has no
        # bracket sender; the dedicated lifecycle must own future truth.
        if details.get("protection_mode") == "equity_bracket":
            checked += 1
            unresolved += 1
            results.append(
                {
                    "intent_id": intent.id,
                    "status": "bracket_reconciliation_deferred",
                    "summary": (
                        "Protected bracket intent is excluded from simple-order "
                        "reconciliation; no broker mutation or order-status "
                        "transition was performed."
                    ),
                }
            )
            continue

        checked += 1
        broker_order = details.get("broker_order") if isinstance(details.get("broker_order"), dict) else {}
        client_order_id = str(
            details.get("client_order_id")
            or broker_order.get("client_order_id")
            or ""
        ).strip()
        broker_order_id = str(broker_order.get("id") or "").strip()

        order = None
        if client_order_id:
            try:
                order = _lookup_order_with_retry(client_order_id)
            except AlpacaPaperError:
                unresolved += 1
                results.append(
                    {
                        "intent_id": intent.id,
                        "status": "reconciliation_required",
                        "summary": "Broker order lookup is unavailable.",
                    }
                )
                continue
        elif broker_order_id:
            try:
                order = get_order_by_id(broker_order_id)
            except AlpacaPaperError:
                unresolved += 1
                results.append(
                    {
                        "intent_id": intent.id,
                        "status": "reconciliation_required",
                        "summary": "Broker order-id lookup is unavailable.",
                    }
                )
                continue

        if order is not None:
            result = _apply_order_truth(intent, order, source="manual_reconcile")
            results.append(result)
            if result["status"] == "executed":
                executed += 1
            elif result["status"] == "accepted":
                accepted += 1
            elif result["status"] == "cancelled":
                cancelled += 1
            else:
                failed += 1
            continue

        if intent.intent_type != "close":
            try:
                position = get_open_position(intent.symbol)
            except AlpacaPaperError:
                unresolved += 1
                results.append(
                    {
                        "intent_id": intent.id,
                        "status": "reconciliation_required",
                        "summary": "Position truth is unavailable after missing order lookup.",
                    }
                )
                continue
            if position is not None:
                updated = update_order_intent_status(
                    intent.id,
                    "accepted",
                    {
                        "outcome_state": "position_visible_order_unresolved",
                        "reconciliation_resolved": False,
                        "position_truth": position,
                        "automatic_retry": False,
                    },
                    reason="alpaca_paper_position_visible_order_unresolved",
                )
                unresolved += 1
                results.append(
                    {
                        "intent_id": updated.id,
                        "status": "reconciliation_required",
                        "summary": "Broker position exists but exact order truth is unavailable.",
                    }
                )
                continue

        if intent.intent_type == "close":
            symbol_or_asset_id = str(details.get("symbol_or_asset_id") or intent.symbol)
            try:
                position = get_open_position(symbol_or_asset_id)
            except AlpacaPaperError:
                unresolved += 1
                results.append(
                    {
                        "intent_id": intent.id,
                        "status": "reconciliation_required",
                        "summary": "Position truth is unavailable for close reconciliation.",
                    }
                )
                continue
            if position is None:
                updated = update_order_intent_status(
                    intent.id,
                    "executed",
                    {
                        "outcome_state": "flat_confirmed_by_reconciliation",
                        "reconciliation_resolved": True,
                        "automatic_retry": False,
                    },
                    reason="alpaca_paper_reconcile_flat_confirmed",
                )
                add_trade_audit(
                    event_type="alpaca_paper_position_flat_confirmed",
                    symbol=updated.symbol,
                    timeframe=updated.timeframe,
                    strategy=updated.strategy,
                    intent_id=updated.id,
                    summary="Alpaca Paper reconciliation confirmed the requested position is flat.",
                    details={"symbol_or_asset_id": symbol_or_asset_id},
                )
                executed += 1
                results.append(
                    {
                        "intent_id": updated.id,
                        "status": updated.status,
                        "summary": "Alpaca Paper position confirmed flat.",
                    }
                )
            else:
                updated = update_order_intent_status(
                    intent.id,
                    "accepted",
                    {
                        "outcome_state": "close_unresolved_position_still_open",
                        "reconciliation_resolved": False,
                        "position_after_close": position,
                        "automatic_retry": False,
                    },
                    reason="alpaca_paper_reconcile_close_still_unresolved",
                )
                unresolved += 1
                results.append(
                    {
                        "intent_id": updated.id,
                        "status": "reconciliation_required",
                        "summary": (
                            "Position remains open and no terminal close-order truth is available; "
                            "automatic repeat-close remains blocked."
                        ),
                    }
                )
            continue

        result = _mark_submission_absent(intent, source="manual_reconcile")
        failed += 1
        results.append(result)

    positions = list_positions()
    return {
        "provider": "alpaca",
        "environment": "paper",
        "checked_intents": checked,
        "executed": executed,
        "accepted": accepted,
        "failed": failed,
        "cancelled": cancelled,
        "unresolved": unresolved,
        "open_position_count": len(positions),
        "open_positions": positions,
        "results": results,
        "automatic_resubmission": False,
    }
