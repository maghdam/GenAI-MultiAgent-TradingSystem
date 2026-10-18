"""Read-only, append-only Alpaca Paper equity-bracket evidence reconciliation.

This module observes reserved bracket intents only. No submit/cancel/close
calls are possible through this service, and a stored historical proof must
never authorize new trading or imply current broker-side protection.
"""

from __future__ import annotations

from backend.services.alpaca_paper import (
    AlpacaPaperError,
    get_open_position,
    get_order_by_client_order_id,
)
from backend.services.alpaca_protection import assess_equity_bracket_truth
from backend.storage.repositories import (
    add_trade_audit,
    get_order_intent_by_id,
    record_alpaca_bracket_observation,
)


def observe_paper_equity_bracket(*, intent_id: int) -> dict[str, object]:
    """Record broker parent/leg and position truth for an explicit Paper bracket intent.

    A successful historical observation does not change the durable intent's
    terminal status or disable any global fail-closed automatic-entry gate.
    """
    intent = get_order_intent_by_id(intent_id)
    details = intent.details if isinstance(intent.details, dict) else {}
    if (
        intent.intent_type != "open"
        or details.get("broker_provider") != "alpaca"
        or details.get("broker_environment") != "paper"
        or details.get("asset_class") != "us_equity"
        or details.get("protection_mode") != "equity_bracket"
        or intent.direction not in {"long", "short"}
    ):
        raise ValueError("Only explicitly reserved Alpaca Paper US-equity bracket entries may be inspected.")

    client_id = str(details.get("client_order_id") or "").strip()
    if not client_id:
        raise ValueError("Protected Alpaca order intent has no durable reserved client_order_id.")
    side = "buy" if intent.direction == "long" else "sell"
    parent: dict[str, object] | None = None
    position: dict[str, object] | None = None
    try:
        parent = get_order_by_client_order_id(client_id, nested=True)
        position = get_open_position(intent.symbol)
        assessment = assess_equity_bracket_truth(
            parent=parent,
            position=position,
            expected_client_order_id=client_id,
            symbol=intent.symbol,
            side=side,
            quantity=intent.quantity,
            stop_loss=intent.stop_loss,
            take_profit=intent.take_profit,
        )
        verified = assessment.verified
        reason = assessment.reason
        parent_id = assessment.parent_order_id
        stop_id = assessment.stop_order_id
        target_id = assessment.take_profit_order_id
    except AlpacaPaperError:
        # Lookup timeouts, malformed broker data, and absent truth are all
        # UNVERIFIED; an ambiguous lookup never means the order was absent.
        verified = False
        reason = "Alpaca Paper broker truth unavailable; bracket protection could not be verified."
        parent_id = stop_id = target_id = None

    observation = record_alpaca_bracket_observation(
        intent_id=intent.id,
        client_order_id=client_id,
        verified_at_observation=verified,
        reason=reason,
        parent_order_id=parent_id,
        stop_order_id=stop_id,
        take_profit_order_id=target_id,
        broker_parent=parent,
        broker_position=position,
    )
    add_trade_audit(
        event_type=(
            "alpaca_paper_bracket_protection_observed"
            if verified else "alpaca_paper_bracket_protection_unverified"
        ),
        symbol=intent.symbol,
        timeframe=intent.timeframe,
        strategy=intent.strategy,
        intent_id=intent.id,
        summary=reason,
        details={
            "observation_id": observation["id"],
            "verified_at_observation": verified,
            "client_order_id": client_id,
            "parent_order_id": parent_id,
            "stop_order_id": stop_id,
            "take_profit_order_id": target_id,
            "current_protection_verified": False,
            "broker_mutation_permitted": False,
        },
    )
    return observation
