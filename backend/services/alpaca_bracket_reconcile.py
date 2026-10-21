"""Explicit read-only lifecycle reconciliation for already-dispatched Alpaca brackets.

Runs only when deliberately invoked for a single accepted Paper equity bracket
whose one-shot claim was consumed. Every call makes NEW broker reads and
appends historical evidence; no saved proof is an ongoing protection guarantee.
There is no API/engine/scheduler caller, no broker mutation, and no status
transition to executed or permission to re-submit an ambiguous bracket.
"""

from __future__ import annotations

from urllib.parse import urlparse

from backend.services.alpaca_bracket_observations import observe_paper_equity_bracket
from backend.services.alpaca_paper import AlpacaPaperError, _base_url
from backend.storage.repositories import (
    add_trade_audit,
    get_order_intent_by_id,
)


# These are only post-claim states produced by the Phase 10.17/10.18
# single-use dispatcher. Never inspect an unsubmitted reservation as a trade.
_DISPATCHED_STATES = frozenset({
    "bracket_dispatch_claimed_submission_unknown",
    "submission_unknown_broker_lookup_unavailable",
    "existing_broker_order_reconciliation_required",
    "ambiguous_post_submission_unknown",
    "broker_rejected_no_retry",
    "parent_acknowledged_protection_unverified",
})


def reconcile_dispatched_paper_equity_bracket(*, intent_id: int) -> dict[str, object]:
    """Fresh read of exact bracket parent/legs and separate broker position.

    `verified_at_observation` is historical evidence, not a persistent risk
    gate. Partial fills, changed/held/filled protective legs, missing position,
    missing parent, lookup errors, or unknown broker state remain unresolved.
    """
    intent = get_order_intent_by_id(intent_id)
    details = intent.details if isinstance(intent.details, dict) else {}
    if (
        intent.status != "accepted"
        or intent.intent_type != "open"
        or intent.direction not in {"long", "short"}
        or details.get("broker_provider") != "alpaca"
        or details.get("broker_environment") != "paper"
        or details.get("asset_class") != "us_equity"
        or details.get("protection_mode") != "equity_bracket"
        or details.get("outcome_state") not in _DISPATCHED_STATES
        or not str(details.get("client_order_id") or "").strip()
    ):
        raise ValueError(
            "Only already-dispatched, unresolved Alpaca Paper US-equity brackets may be reconciled."
        )
    endpoint = urlparse(_base_url())
    if endpoint.scheme != "https" or endpoint.netloc != "paper-api.alpaca.markets" or endpoint.path not in ("", "/"):
        raise AlpacaPaperError(
            "Read-only bracket reconciliation requires the official HTTPS Alpaca Paper endpoint."
        )

    # The observer obtains fresh exact nested client-order truth plus separate
    # position truth and records a new immutable observation and audit. It
    # also flags changed parent/SL/TP identities from the first proven set.
    observation = observe_paper_equity_bracket(intent_id=intent.id)
    parent = observation["broker_snapshot"]["broker_parent"]
    if observation["verified_at_observation"]:
        classification = "complete_bracket_verified_at_observation_only"
    elif not isinstance(parent, dict):
        classification = "broker_parent_unknown_unverified"
    elif str(parent.get("status") or "").lower() == "filled":
        classification = "filled_parent_exits_or_position_unverified"
    elif str(parent.get("status") or "").lower() == "partially_filled" or (
        str(parent.get("filled_qty") or "") not in ("", "0", "0.0", "None")
    ):
        classification = "partial_fill_unverified"
    elif str(parent.get("status") or "").lower() in {
        "canceled", "expired", "rejected", "replaced", "done_for_day"
    }:
        classification = "parent_terminal_unverified"
    else:
        classification = "parent_pending_or_unknown_unverified"

    # This result cannot be held as continuously verified. Even when a single
    # instant was proven, another broker read is required before any later
    # risk decision, and separate exit/partial-fill management is still absent.
    result: dict[str, object] = {
        "intent_id": intent.id,
        "status": "reconciliation_required",
        "classification": classification,
        "observation_id": observation["id"],
        "observed_at": observation["observed_at"],
        "verified_at_observation": observation["verified_at_observation"],
        "reason": observation["reason"],
        "parent_order_id": observation["parent_order_id"],
        "stop_order_id": observation["stop_order_id"],
        "take_profit_order_id": observation["take_profit_order_id"],
        "current_protection_verified": False,
        "reconciliation_resolved": False,
        "automatic_retry": False,
        "broker_mutation_permitted": False,
    }
    add_trade_audit(
        event_type="alpaca_paper_bracket_readonly_reconciled",
        symbol=intent.symbol,
        timeframe=intent.timeframe,
        strategy=intent.strategy,
        intent_id=intent.id,
        summary=(
            "Read-only bracket snapshot classified as " + classification
            + "; unresolved and no order mutation."
        ),
        details={
            "observation_id": observation["id"],
            "classification": classification,
            "verified_at_observation": observation["verified_at_observation"],
            "current_protection_verified": False,
            "reconciliation_resolved": False,
            "automatic_retry": False,
            "broker_mutation_permitted": False,
        },
    )
    return result
