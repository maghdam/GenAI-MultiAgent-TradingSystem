"""Pure, conservative accounting of native bracket entry and exit fills.

This is snapshot evidence, never an authorization to trade. Broker-side OCO
resizing is checked, not emulated by sending a second protective order.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation


TERMINAL = frozenset({"filled", "canceled", "expired", "rejected"})
ACTIVE = frozenset({"new", "partially_filled"})
KNOWN = TERMINAL | ACTIVE | {"accepted", "pending_new", "held", "pending_cancel",
                              "pending_replace", "done_for_day", "stopped", "suspended",
                              "calculated", "accepted_for_bidding"}


def number(value):
    if value is None or isinstance(value, bool):
        raise ValueError("Missing or invalid broker quantity")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("Invalid broker quantity") from exc
    if not result.is_finite():
        raise ValueError("Non-finite broker quantity")
    return result


def assess_lifecycle(*, parent, position, client_order_id, symbol, side,
                     quantity, stop_loss, take_profit, expected_ids=None,
                     close_filled="0"):
    """Reconcile signed positions against cumulative fills, including partials.

    Unknown/replaced orders, overfills, mismatched identities and external
    position changes block management. Inactive protection is reported even
    when exact ownership and fill accounting can still be established.
    """
    result = {"manageable": False, "protected_at_observation": False,
              "classification": "unresolved", "automatic_trading_allowed": False}
    try:
        if not isinstance(parent, dict) or side not in {"buy", "sell"}:
            raise ValueError("Exact bracket parent and entry side required")
        qty = number(quantity)
        if qty <= 0 or qty != qty.to_integral_value():
            raise ValueError("Invalid reserved quantity")
        if (parent.get("client_order_id") != client_order_id
                or parent.get("order_class") != "bracket"
                or parent.get("symbol") != symbol or parent.get("side") != side
                or parent.get("asset_class") != "us_equity"
                or parent.get("group_role") not in (None, "", "entry")
                or number(parent.get("qty")) != qty):
            raise ValueError("Parent differs from reserved bracket")
        legs = parent.get("legs")
        if not isinstance(legs, list) or len(legs) != 2:
            raise ValueError("Both exact protective legs required")
        roles = {}
        for leg in legs:
            if not isinstance(leg, dict):
                raise ValueError("Malformed protective leg")
            role = {"limit": "target", "stop": "stop"}.get(leg.get("type"))
            if role is None or role in roles:
                raise ValueError("Unknown or duplicate protective role")
            price_key, expected_price = ("stop_price", stop_loss) if role == "stop" else ("limit_price", take_profit)
            if (leg.get("symbol") != symbol or leg.get("asset_class") != "us_equity"
                    or leg.get("side") != ("sell" if side == "buy" else "buy")
                    or number(leg.get(price_key)) != number(expected_price)
                    or number(expected_price) <= 0
                    or leg.get("group_role") not in (None, "", "stop_loss" if role == "stop" else "take_profit")
                    or leg.get("group_id") != parent.get("group_id")):
                raise ValueError("Protective leg identity or price differs")
            roles[role] = leg
        orders = {"parent": parent, **roles}
        ids = {role: order.get("id") for role, order in orders.items()}
        if any(not isinstance(x, str) or not x.strip() for x in ids.values()) or len(set(ids.values())) != 3:
            raise ValueError("Missing or duplicate order identities")
        if expected_ids is not None and ids != expected_ids:
            raise ValueError("Order identity changed; replacement needs manual reconciliation")
        fills = {}
        for role, order in orders.items():
            total, filled = number(order.get("qty")), number(order.get("filled_qty"))
            state = order.get("status")
            if (state not in KNOWN or order.get("replaced_by") or order.get("replaces")
                    or total <= 0 or total > qty or not 0 <= filled <= total
                    or (state == "filled" and filled != total)
                    or (state == "partially_filled" and not 0 < filled < total)
                    or (state == "new" and filled != 0)):
                raise ValueError("Unknown, replaced or inconsistent order state")
            fills[role] = filled
        closed = number(close_filled)
        remaining = fills["parent"] - fills["stop"] - fills["target"] - closed
        if remaining < 0 or closed < 0:
            raise ValueError("Exit overfill or OCO race requires manual reconciliation")
        if position is None:
            actual = Decimal(0)
        else:
            if (not isinstance(position, dict) or position.get("symbol") != symbol
                    or position.get("asset_class") != "us_equity"
                    or position.get("side") != ("long" if side == "buy" else "short")):
                raise ValueError("Position identity/direction differs")
            signed = number(position.get("qty"))
            if (side == "buy" and signed <= 0) or (side == "sell" and signed >= 0):
                raise ValueError("Position quantity sign differs")
            actual = abs(signed)
        if actual != remaining:
            raise ValueError("Position quantity differs from cumulative bracket/close fills")
        protected = (remaining > 0 and parent.get("status") == "filled"
                     and all(o.get("status") in ACTIVE
                             and number(o["qty"]) - number(o["filled_qty"]) == remaining
                             for o in roles.values()))
        if protected:
            entry = number(position.get("avg_entry_price"))
            stop, target = number(stop_loss), number(take_profit)
            protected = stop < entry < target if side == "buy" else target < entry < stop
        terminal = all(o.get("status") in TERMINAL for o in orders.values())
        classification = ("flat_terminal" if remaining == 0 and terminal else
                          "flat_orders_pending" if remaining == 0 else
                          "partial_entry_unprotected" if parent.get("status") != "filled" else
                          "remaining_position_protected" if protected else
                          "remaining_position_unprotected")
        result.update(manageable=True, protected_at_observation=protected,
                      classification=classification, remaining_quantity=str(remaining),
                      ids=ids, all_orders_terminal=terminal)
    except (ValueError, TypeError, KeyError) as exc:
        result["reason"] = str(exc)
    return result
