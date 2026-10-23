"""Explicit, operator-supervised cancel-and-flatten steps. No engine callers.

Each invocation performs at most one broker mutation. Cancellation can expose
the position until a later close fills; this workflow is not unattended-safe.
Restart recovery reads exact IDs and never repeats a consumed mutation claim.
"""
from __future__ import annotations

from backend.services import alpaca_paper as broker
from backend.services.alpaca_bracket_lifecycle import TERMINAL, KNOWN, assess_lifecycle, number
from backend.services.alpaca_bracket_reconcile import _DISPATCHED_STATES
from backend.storage.alpaca_bracket_exits import bind_exit_session, claim_exit_action, exit_actions
from backend.storage.repositories import (
    add_trade_audit, first_verified_alpaca_bracket_observation, get_order_intent_by_id,
)


def _eligible(intent):
    d = intent.details
    if (intent.status != "accepted" or intent.intent_type != "open"
            or intent.direction not in {"long", "short"} or not isinstance(d, dict)
            or d.get("broker_provider") != "alpaca" or d.get("broker_environment") != "paper"
            or d.get("asset_class") != "us_equity" or d.get("protection_mode") != "equity_bracket"
            or d.get("outcome_state") not in _DISPATCHED_STATES or not d.get("client_order_id")):
        raise ValueError("Only dispatched, unresolved Paper equity brackets are eligible")


def _audit(intent, result):
    result.update(automatic_retry=False, automatic_trading_allowed=False,
                  current_protection_verified=False)
    add_trade_audit(event_type="alpaca_bracket_exit_step", symbol=intent.symbol,
                    timeframe=intent.timeframe, strategy=intent.strategy, intent_id=intent.id,
                    summary=result["status"], details=result)
    return result


def _read(intent):
    broker.require_official_paper_endpoint()
    account = broker.get_account()
    if (account.get("environment") != "paper" or not account.get("account_id")
            or account.get("status") != "ACTIVE" or account.get("trading_blocked")
            or account.get("account_blocked") or account.get("trade_suspended_by_user")):
        raise ValueError("Active unblocked Paper account with exact identity required")
    parent = broker.get_order_by_client_order_id(intent.details["client_order_id"], nested=True)
    actions = exit_actions(intent.id)
    close_request = actions.get("close")
    close = None
    close_filled = "0"
    if close_request:
        close = broker.get_order_by_client_order_id(close_request["client_order_id"])
        if close is not None:
            if (not close.get("id") or close.get("client_order_id") != close_request["client_order_id"]
                    or close.get("symbol") != intent.symbol or close.get("asset_class") != "us_equity"
                    or close.get("side") != close_request["side"] or close.get("type") != "market"
                    or close.get("order_class") not in (None, "", "simple")
                    or close.get("replaced_by") or close.get("replaces")
                    or number(close.get("qty")) != number(close_request["qty"])
                    or close.get("status") not in KNOWN
                    or not 0 <= number(close.get("filled_qty")) <= number(close_request["qty"])
                    or (close.get("status") == "filled"
                        and number(close.get("filled_qty")) != number(close_request["qty"]))):
                raise ValueError("Close identity, state or fill quantity mismatch")
            close_filled = close["filled_qty"]
    position = broker.get_open_position(intent.symbol)
    baseline = first_verified_alpaca_bracket_observation(intent.id)
    expected = None if baseline is None else {
        "parent": baseline["parent_order_id"], "stop": baseline["stop_order_id"],
        "target": baseline["take_profit_order_id"],
    }
    acknowledged = intent.details.get("broker_parent_order")
    if isinstance(acknowledged, dict) and isinstance(parent, dict) and acknowledged.get("id") != parent.get("id"):
        raise ValueError("Parent differs from original acknowledgement")
    assessment = assess_lifecycle(
        parent=parent, position=position, client_order_id=intent.details["client_order_id"],
        symbol=intent.symbol, side="buy" if intent.direction == "long" else "sell",
        quantity=intent.quantity, stop_loss=intent.stop_loss, take_profit=intent.take_profit,
        expected_ids=expected, close_filled=close_filled,
    )
    if not assessment["manageable"]:
        raise ValueError(assessment.get("reason", "Unresolved bracket ownership"))
    bind_exit_session(intent.id, account["account_id"], {
        "ids": assessment["ids"], "client_order_id": intent.details["client_order_id"],
        "symbol": intent.symbol, "direction": intent.direction,
        "quantity": str(intent.quantity), "stop_loss": str(intent.stop_loss),
        "take_profit": str(intent.take_profit),
    })
    if close is not None:
        claim_exit_action(intent.id, "close_identity", {"id": close["id"]})
        if exit_actions(intent.id)["close_identity"]["id"] != close["id"]:
            raise ValueError("Close broker identity changed after observation")
    open_orders = broker.list_orders(status="open", limit=500)
    if len(open_orders) >= 500:
        raise ValueError("Open order list may be truncated")
    known_ids = set(assessment["ids"].values())
    if close is not None:
        known_ids.add(close["id"])
    for order in open_orders:
        for item in [order, *(order.get("legs") or [])]:
            if not item.get("symbol") or not item.get("id"):
                raise ValueError("Malformed open order inventory")
            if item["symbol"] == intent.symbol and item["id"] not in known_ids:
                raise ValueError("Unowned same-symbol order blocks cancellation and flatten")
            if item["symbol"] == intent.symbol and item.get("status") not in TERMINAL:
                original = next((x for x in [parent, *parent["legs"]] if x["id"] == item["id"]), None)
                if original and original["status"] in TERMINAL:
                    raise ValueError("Contradictory order snapshots; reread required")
    return parent, assessment, actions, close


def inspect_paper_bracket_exit(*, intent_id: int):
    """Read-only restart inspection, including ambiguous/partial close fills."""
    intent = get_order_intent_by_id(intent_id)
    _eligible(intent)
    try:
        _, assessment, actions, close = _read(intent)
        complete = (assessment["classification"] == "flat_terminal"
                    and ("close" not in actions or close is not None and close["status"] in TERMINAL))
        return _audit(intent, {"status": "flat_terminal_at_observation" if complete else "reconciliation_required",
                               "assessment": assessment, "close_order": close,
                               "close_claimed": "close" in actions})
    except (broker.AlpacaPaperError, ValueError) as exc:
        return _audit(intent, {"status": "blocked", "reason": str(exc)})


def advance_supervised_paper_bracket_exit(*, intent_id: int, operator_confirmation: str,
                                         operator_reason: str):
    """Cancel native group, verify terminal states, then close remaining shares.

    Operator must supervise the unprotected cancellation-to-fill interval and
    prevent concurrent external activity. Kill switch does not block exits.
    A consumed claim with no broker evidence requires manual recovery.
    """
    if operator_confirmation != f"CANCEL AND FLATTEN PAPER BRACKET {intent_id}":
        raise ValueError("Explicit cancel-and-flatten confirmation required")
    if len(str(operator_reason or "").strip()) < 8:
        raise ValueError("Meaningful operator reason required")
    intent = get_order_intent_by_id(intent_id)
    _eligible(intent)
    try:
        parent, assessment, actions, close = _read(intent)
        if "close" in actions:
            status = ("flat_terminal_at_observation" if assessment["classification"] == "flat_terminal"
                      and close is not None and close["status"] in TERMINAL else "close_reconciliation_required")
            return _audit(intent, {"status": status, "assessment": assessment, "close_order": close})
        outstanding = [o for o in [parent, *parent["legs"]] if o["status"] not in TERMINAL]
        if outstanding:
            order = outstanding[0]
            key = "cancel:" + order["id"]
            if order["status"] in {"pending_cancel", "pending_replace"} or key in actions:
                return _audit(intent, {"status": "cancel_reconciliation_required", "assessment": assessment})
            request = {"order_id": order["id"], "operator_reason": operator_reason.strip()}
        elif number(assessment["remaining_quantity"]) == 0:
            return _audit(intent, {"status": "flat_terminal_at_observation", "assessment": assessment})
        else:
            key = "close"
            request = {"symbol": intent.symbol, "side": "sell" if intent.direction == "long" else "buy",
                       "qty": assessment["remaining_quantity"],
                       "client_order_id": f"ta-bracket-exit-{intent.id}",
                       "operator_reason": operator_reason.strip()}
            if broker.get_order_by_client_order_id(request["client_order_id"]) is not None:
                raise ValueError("Unclaimed close client ID already exists")
        if not claim_exit_action(intent.id, key, request):
            return _audit(intent, {"status": "action_already_claimed"})
        # Claim survives a process death, audit failure or any HTTP outcome.
        _audit(intent, {"status": "mutation_claimed", "action": key, "request": request})
        if key == "close":
            broker.submit_paper_bracket_close(**{k: v for k, v in request.items() if k != "operator_reason"})
        else:
            broker.cancel_paper_bracket_order(request["order_id"])
        return _audit(intent, {"status": "acknowledged_reconciliation_required", "action": key,
                               "assessment": assessment})
    except (broker.AlpacaPaperError, ValueError) as exc:
        return _audit(intent, {"status": "blocked", "reason": str(exc)})
