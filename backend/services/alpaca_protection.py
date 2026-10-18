"""Read-only broker-truth proof for Alpaca Paper US-equity bracket exits.

This module does not submit, cancel, or reconcile orders and is deliberately
not an auto-entry enablement switch. Missing, incomplete, or mismatched
evidence means protection is NOT VERIFIED.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any


@dataclass(frozen=True)
class BracketTruth:
    verified: bool
    reason: str
    parent_order_id: str | None = None
    stop_order_id: str | None = None
    take_profit_order_id: str | None = None


def _decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = Decimal(str(value))
    except (TypeError, ValueError, InvalidOperation):
        return None
    return parsed if parsed.is_finite() and parsed >= 0 else None


def assess_equity_bracket_truth(
    *,
    parent: dict[str, object] | None,
    position: dict[str, object] | None,
    expected_client_order_id: str,
    symbol: str,
    side: str,
    quantity: int,
    stop_loss: float,
    take_profit: float,
) -> BracketTruth:
    """Require exact filled parent + two active, unfired exit legs + position.

    This deliberately recognizes only full-fill, zero-exit-fill snapshots.
    Partial fills, pending/held exits, replacements and OCO exits require a
    later lifecycle implementation, not optimistic 'verified' classification.
    Bracket legs alone are insufficient without independently checked position.
    """
    def no(reason: str) -> BracketTruth:
        return BracketTruth(False, reason)

    if not isinstance(parent, dict) or not isinstance(position, dict):
        return no("Exact bracket parent AND open broker position truth are required.")

    identity = str(expected_client_order_id or "").strip()
    normalized_symbol = str(symbol or "").strip().upper()
    normalized_side = str(side or "").strip().lower()
    qty = _decimal(quantity)
    stop = _decimal(stop_loss)
    target = _decimal(take_profit)
    if (
        not identity or not normalized_symbol or normalized_side not in {"buy", "sell"}
        or qty is None or qty <= 0 or qty != qty.to_integral_value()
        or stop is None or stop <= 0 or target is None or target <= 0
    ):
        return no("Expected broker bracket identity, quantity or protective prices are invalid.")

    parent_id = str(parent.get("id") or "").strip()
    if not parent_id or str(parent.get("client_order_id") or "") != identity:
        return no("Bracket parent identity does not match the reserved client order ID.")
    if (
        str(parent.get("order_class") or "").lower() != "bracket"
        or str(parent.get("asset_class") or "").lower() != "us_equity"
        or str(parent.get("symbol") or "").upper() != normalized_symbol
        or str(parent.get("side") or "").lower() != normalized_side
        or str(parent.get("status") or "").lower() != "filled"
        or str(parent.get("group_role") or "entry").lower() != "entry"
        or _decimal(parent.get("qty")) != qty
        or _decimal(parent.get("filled_qty")) != qty
    ):
        return no("US-equity bracket parent is not an exact fully filled entry.")

    direction = "long" if normalized_side == "buy" else "short"
    entry_price = _decimal(position.get("avg_entry_price"))
    if (
        str(position.get("symbol") or "").upper() != normalized_symbol
        or str(position.get("side") or "").lower() != direction
        or str(position.get("asset_class") or "").lower() != "us_equity"
        or _decimal(position.get("qty")) != qty
        or entry_price is None or entry_price <= 0
    ):
        return no("The open broker position does not match the filled bracket parent.")
    if (
        (normalized_side == "buy" and not stop < entry_price < target)
        or (normalized_side == "sell" and not target < entry_price < stop)
    ):
        return no("Stop/target prices do not protect the actual broker entry-price direction.")

    legs = parent.get("legs")
    if not isinstance(legs, list) or len(legs) != 2 or any(not isinstance(x, dict) for x in legs):
        return no("Broker returned fewer or more than two identifiable bracket exit legs.")
    exit_side = "sell" if normalized_side == "buy" else "buy"
    ids: set[str] = {parent_id}
    found: dict[str, str] = {}
    group = str(parent.get("group_id") or "").strip()
    for leg in legs:
        leg_id = str(leg.get("id") or "").strip()
        if not leg_id or leg_id in ids:
            return no("A bracket exit leg lacks a unique broker order ID.")
        ids.add(leg_id)
        if (
            str(leg.get("symbol") or "").upper() != normalized_symbol
            or str(leg.get("asset_class") or "").lower() != "us_equity"
            or str(leg.get("side") or "").lower() != exit_side
            or str(leg.get("status") or "").lower() != "new"
            or _decimal(leg.get("qty")) != qty
            or _decimal(leg.get("filled_qty")) != Decimal(0)
            or leg.get("replaced_by") not in (None, "")
        ):
            return no("A bracket exit leg is not active, unfilled, unreplaced, or size-matched.")

        leg_group = str(leg.get("group_id") or "").strip()
        if group or leg_group:
            if not group or leg_group != group:
                return no("Bracket parent and exit-leg group identities do not match.")

        order_type = str(leg.get("type") or "").lower()
        if order_type == "limit" and _decimal(leg.get("limit_price")) == target:
            role = "take_profit"
        elif order_type == "stop" and _decimal(leg.get("stop_price")) == stop:
            role = "stop_loss"
        else:
            return no("Bracket stop or take-profit price/type does not match the strategy targets.")
        if str(leg.get("group_role") or role).lower() != role or role in found:
            return no("Bracket exit legs have missing, incorrect or duplicate roles.")
        found[role] = leg_id

    if set(found) != {"stop_loss", "take_profit"}:
        return no("Both independent stop and take-profit broker legs are required.")
    return BracketTruth(
        verified=True,
        reason="Exact filled US-equity bracket parent, position, and two active exit legs verified.",
        parent_order_id=parent_id,
        stop_order_id=found["stop_loss"],
        take_profit_order_id=found["take_profit"],
    )
