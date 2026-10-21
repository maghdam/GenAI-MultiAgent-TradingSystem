"""Explicitly supervised, one-shot Alpaca Paper US-equity bracket dispatch.

No API, engine, scheduler, or automatic-entry caller is wired to this module.
The durable dispatch claim is consumed BEFORE a single broker POST. Ambiguous
outcomes are never retried as another POST. A parent acknowledgement NEVER
marks broker-native SL/TP protection as currently verified.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from backend.domain.models import EngineConfig
from backend.services.alpaca_bracket_observations import observe_paper_equity_bracket
from backend.services.alpaca_execution import _execution_ready_account
from backend.services.alpaca_paper import (
    AlpacaPaperError,
    build_equity_bracket_market_order,
    get_open_position,
    get_order_by_client_order_id,
    list_orders,
    list_positions,
    submit_equity_bracket_order,
)
from backend.storage.repositories import (
    add_trade_audit,
    claim_reserved_alpaca_bracket_dispatch,
    get_order_intent_by_id,
    load_engine_config,
    list_paper_positions,
    update_order_intent_status,
)


def _record_post_outcome(intent, *, outcome: str, summary: str,
                         broker_order: dict[str, object] | None = None,
                         broker_status_code: int | None = None) -> dict[str, object]:
    """Keep every claimed bracket unresolved, even if its parent is filled."""
    updates: dict[str, object] = {
        "outcome_state": outcome,
        "reconciliation_resolved": False,
        "automatic_retry": False,
        "broker_mutation_permitted": False,
        "broker_native_protection_verified": False,
    }
    if broker_order is not None:
        updates["broker_parent_order"] = broker_order
    if broker_status_code is not None:
        updates["broker_status_code"] = broker_status_code
    updated = update_order_intent_status(
        intent.id, "accepted", updates,
        reason="alpaca_paper_bracket_" + outcome,
    )
    add_trade_audit(
        event_type="alpaca_paper_bracket_" + outcome,
        symbol=intent.symbol, timeframe=intent.timeframe,
        strategy=intent.strategy, intent_id=intent.id,
        summary=summary,
        details={
            "client_order_id": updated.details["client_order_id"],
            "broker_order_id": broker_order.get("id") if broker_order else None,
            "broker_status_code": broker_status_code,
            "automatic_retry": False,
            "broker_native_protection_verified": False,
        },
    )
    return {
        "intent_id": updated.id,
        "status": "reconciliation_required",
        "outcome_state": outcome,
        "summary": summary,
        "broker_order": broker_order,
        "broker_native_protection_verified": False,
        "automatic_retry": False,
    }


def _read_after_claim(intent, *, source: str) -> dict[str, object] | None:
    """Read exact client identity. Missing lookup is NEVER a retry license."""
    client_id = intent.details["client_order_id"]
    try:
        parent = get_order_by_client_order_id(client_id, nested=True)
    except AlpacaPaperError:
        return _record_post_outcome(
            intent, outcome="submission_unknown_broker_lookup_unavailable",
            summary="Exact Alpaca Paper bracket lookup unavailable; no POST attempted or retried.",
        )
    if parent is None:
        return None
    return _record_post_outcome(
        intent, outcome="existing_broker_order_reconciliation_required",
        summary="Broker already has the reserved client order ID; a second POST is forbidden.",
        broker_order=parent,
    )


def submit_supervised_paper_equity_bracket(
    *,
    intent_id: int,
    expected_client_order_id: str,
    operator_confirmation: str,
    operator_reason: str,
) -> dict[str, object]:
    """One explicit operator-confirmed dispatch attempt for a durable reservation.

    A claim cannot be consumed twice, including after a restart or a POST
    timeout. This never routes autonomous strategies to Alpaca. Existing
    broker positions/open orders or unavailable preflight truth prevent POST.
    """
    if operator_confirmation != f"SUBMIT PAPER BRACKET {int(intent_id)}":
        raise ValueError("Explicit confirmation of the Paper bracket intent ID is required.")
    if len(str(operator_reason or "").strip()) < 8:
        raise ValueError("A meaningful operator reason is required.")
    intent = get_order_intent_by_id(intent_id)
    details = intent.details if isinstance(intent.details, dict) else {}
    cid = str(details.get("client_order_id") or "")
    if (
        intent.status != "accepted" or intent.intent_type != "open"
        or details.get("broker_provider") != "alpaca"
        or details.get("broker_environment") != "paper"
        or details.get("asset_class") != "us_equity"
        or details.get("protection_mode") != "equity_bracket"
        or details.get("outcome_state") != "bracket_reserved_not_submitted"
        or not cid or expected_client_order_id != cid
    ):
        raise ValueError("No matching, unclaimed, reserved Alpaca Paper equity bracket exists.")
    side = "buy" if intent.direction == "long" else "sell" if intent.direction == "short" else ""
    stored = details.get("bracket_order_request")
    if not isinstance(stored, dict):
        raise ValueError("Durable bracket request differs from the strictly validated order payload.")
    stored_stop = stored.get("stop_loss")
    stored_target = stored.get("take_profit")
    if not isinstance(stored_stop, dict) or not isinstance(stored_target, dict):
        raise ValueError("Durable bracket request differs from the strictly validated order payload.")
    # The original order payload may contain "2", "99", "104" while SQLite
    # stores numeric intent columns as 2.0, 99.0, 104.0. Rebuilding from
    # SQLite float representations changes *strings*, not economic values.
    # Rebuild instead from the exact reserved request, then independently
    # compare all economic values to the durable numeric intent columns.
    # Never relax symbol, class, side, order type, TIF, client ID, or keys.
    try:
        prospective = build_equity_bracket_market_order(
            symbol=intent.symbol, side=side, qty=stored.get("qty"),
            reference_price=intent.entry_price,
            stop_loss=stored_stop.get("stop_price"),
            take_profit=stored_target.get("limit_price"),
            client_order_id=cid,
        )
        match_numeric_intent = all(
            Decimal(str(persisted)).is_finite()
            and Decimal(str(persisted)) == Decimal(str(requested))
            for persisted, requested in (
                (intent.quantity, prospective["qty"]),
                (intent.stop_loss, prospective["stop_loss"]["stop_price"]),
                (intent.take_profit, prospective["take_profit"]["limit_price"]),
            )
        )
    except (TypeError, ValueError, InvalidOperation, KeyError) as exc:
        raise ValueError(
            "Durable bracket request differs from the strictly validated order payload."
        ) from exc
    if prospective != stored or not match_numeric_intent:
        raise ValueError("Durable bracket request differs from the strictly validated order payload.")
    config = load_engine_config(EngineConfig())
    if config.kill_switch:
        raise AlpacaPaperError("Kill switch must be OFF for a supervised new Paper bracket entry.")
    if config.max_open_positions <= 0 or config.max_positions_per_symbol <= 0:
        raise AlpacaPaperError("Configured position limits prohibit new bracket entries.")
    account = _execution_ready_account()
    if side == "sell" and not account.get("shorting_enabled"):
        raise AlpacaPaperError("Alpaca Paper account does not allow short US-equity entries.")
    if get_open_position(intent.symbol) is not None:
        return {"status": "blocked", "intent_id": intent.id,
                "summary": "An open Alpaca position blocks new bracket dispatch.",
                "broker_native_protection_verified": False}
    open_orders = list_orders(status="open", limit=500)
    if len(open_orders) >= 500 or any(not item.get("symbol") for item in open_orders):
        raise AlpacaPaperError("Incomplete broker open-order inventory blocks bracket dispatch.")
    if any(
        str(x.get("symbol") or "").upper() == intent.symbol.upper()
        for x in open_orders
    ):
        return {"status": "blocked", "intent_id": intent.id,
                "summary": "An open Alpaca order blocks new bracket dispatch.",
                "broker_native_protection_verified": False}

    # The durable single-use claim is committed before any broker POST.
    positions = list_positions()
    if any(not p.get("symbol") for p in positions):
        raise AlpacaPaperError("Incomplete broker position inventory blocks bracket dispatch.")
    local_positions = list_paper_positions("open")
    if any(p.symbol.upper() == intent.symbol.upper() for p in local_positions):
        return {"status": "blocked", "intent_id": intent.id,
                "summary": "A local position already occupies this symbol.",
                "broker_native_protection_verified": False}
    occupied = {str(p["symbol"]).upper() for p in [*positions, *open_orders]}
    claimed = claim_reserved_alpaca_bracket_dispatch(
        intent.id, max_open_positions=config.max_open_positions,
        occupied_symbols=occupied, local_open_count=len(local_positions),
    )
    if claimed is None:
        return {"status": "blocked", "intent_id": intent.id,
                "summary": "Bracket dispatch was already claimed or is no longer eligible.",
                "broker_native_protection_verified": False}
    add_trade_audit(
        event_type="alpaca_paper_bracket_supervised_dispatch_authorized",
        symbol=intent.symbol, timeframe=intent.timeframe,
        strategy=intent.strategy, intent_id=intent.id,
        summary="One supervised Paper bracket dispatch claimed after preflight.",
        details={"client_order_id": cid,
                 "operator_reason": operator_reason.strip(),
                 "automatic_retry": False},
    )

    already = _read_after_claim(claimed, source="pre_post_read")
    if already is not None:
        return already
    try:
        acknowledged = submit_equity_bracket_order(
            symbol=intent.symbol, side=side, qty=prospective["qty"],
            reference_price=intent.entry_price,
            stop_loss=prospective["stop_loss"]["stop_price"],
            take_profit=prospective["take_profit"]["limit_price"],
            client_order_id=cid,
        )
    except AlpacaPaperError as exc:
        if exc.ambiguous:
            # Ambiguous POST: lookup by SAME id, never POST again.
            found = _read_after_claim(claimed, source="ambiguous_post_lookup")
            if found is not None:
                return found
            return _record_post_outcome(
                claimed, outcome="ambiguous_post_submission_unknown",
                summary="Paper bracket POST outcome unknown; exact lookup found no order, but this is NOT proof of absence. Never resubmit automatically.",
                broker_status_code=exc.status_code,
            )
        return _record_post_outcome(
            claimed, outcome="broker_rejected_no_retry",
            summary="Alpaca rejected the bracket attempt; the consumed claim cannot be retried.",
            broker_status_code=exc.status_code,
        )

    outcome = _record_post_outcome(
        claimed, outcome="parent_acknowledged_protection_unverified",
        summary="Alpaca acknowledged an equity bracket parent; active broker exit legs and position are NOT yet established.",
        broker_order=acknowledged,
    )
    # A new read-only observation is historical evidence only. Its result
    # never changes intent to executed or authorizes unattended entry.
    observation = observe_paper_equity_bracket(intent_id=intent.id)
    outcome["observation_id"] = observation["id"]
    outcome["observed_legs_at_snapshot"] = observation["verified_at_observation"]
    return outcome
