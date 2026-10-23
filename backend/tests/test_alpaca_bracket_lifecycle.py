from copy import deepcopy

import pytest

from backend.services.alpaca_bracket_lifecycle import assess_lifecycle
from backend.tests.test_alpaca_bracket_reconcile import _parent, _position


def check(parent=None, position=None, **kwargs):
    return assess_lifecycle(parent=parent or _parent(), position=position or _position(),
                            client_order_id="ta-ap-brkt-check", symbol="SPY", side="buy",
                            quantity=2, stop_loss=99, take_profit=104, **kwargs)


def test_full_and_partial_exit_protection():
    assert check()["protected_at_observation"]
    p, pos = _parent(), _position()
    p["legs"][0].update(status="partially_filled", filled_qty="1")
    p["legs"][1]["qty"] = "1"
    pos["qty"] = "1"
    result = check(p, pos)
    assert result["remaining_quantity"] == "1"
    assert result["protected_at_observation"]
    p["legs"][1]["qty"] = "2"
    assert check(p, pos)["classification"] == "remaining_position_unprotected"


def test_partial_entry_is_never_protected():
    p, pos = _parent(), _position()
    p.update(status="partially_filled", filled_qty="1")
    pos["qty"] = "1"
    assert check(p, pos)["classification"] == "partial_entry_unprotected"


@pytest.mark.parametrize("value", [None, True, "NaN", "Infinity", "-1", "3", "bad"])
def test_bad_fill_fails_closed(value):
    p = _parent()
    p["legs"][0]["filled_qty"] = value
    assert not check(p)["manageable"]


@pytest.mark.parametrize("status", ["held", "pending_cancel", "canceled", "expired", "rejected", "done_for_day"])
def test_inactive_protection(status):
    p = _parent()
    p["legs"][1]["status"] = status
    assert not check(p)["protected_at_observation"]


def test_identity_drift_and_oco_double_fill():
    p = _parent()
    ids = check()["ids"]
    p["legs"][1]["id"] = "replacement"
    assert not check(p, expected_ids=ids)["manageable"]
    p = _parent()
    for leg in p["legs"]:
        leg.update(status="filled", filled_qty="2")
    assert not check(p)["manageable"]


def test_short_position_uses_signed_broker_quantity():
    p, pos = _parent(), _position()
    p["side"] = "sell"
    for leg in p["legs"]:
        leg["side"] = "buy"
    p["legs"][0]["limit_price"] = "99"
    p["legs"][1]["stop_price"] = "104"
    pos.update(side="short", qty="-2")
    result = assess_lifecycle(parent=p, position=pos, client_order_id=p["client_order_id"],
                              symbol="SPY", side="sell", quantity=2, stop_loss=104, take_profit=99)
    assert result["protected_at_observation"]


def test_external_position_and_close_partial_fill():
    pos = _position()
    pos["qty"] = "1"
    assert not check(position=pos)["manageable"]
    p = _parent()
    for leg in p["legs"]:
        leg["status"] = "canceled"
    result = check(p, pos, close_filled="1")
    assert result["manageable"] and result["remaining_quantity"] == "1"
    assert not result["protected_at_observation"]


def test_flat_oco_requires_sibling_terminal():
    p = _parent()
    p["legs"][0].update(status="filled", filled_qty="2")
    args = dict(parent=p, position=None, client_order_id=p["client_order_id"],
                symbol="SPY", side="buy", quantity=2, stop_loss=99, take_profit=104)
    assert assess_lifecycle(**args)["classification"] == "flat_orders_pending"
    p["legs"][1]["status"] = "canceled"
    assert assess_lifecycle(**args)["classification"] == "flat_terminal"
