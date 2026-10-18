from __future__ import annotations

from copy import deepcopy

import pytest

from backend.services import alpaca_paper
from backend.services.alpaca_protection import assess_equity_bracket_truth


def _bracket():
    return {
        "id": "parent-1", "client_order_id": "tradeagent-bracket-1",
        "symbol": "SPY", "asset_class": "us_equity", "side": "buy",
        "type": "market", "time_in_force": "gtc", "order_class": "bracket",
        "group_id": "group-1", "group_role": "entry",
        "qty": "2", "filled_qty": "2", "status": "filled",
        "legs": [
            {
                "id": "tp-1", "symbol": "SPY", "side": "sell", "type": "limit",
                "group_id": "group-1", "group_role": "take_profit",
                "qty": "2", "filled_qty": "0", "status": "new", "limit_price": "104",
                "legs": None,
            },
            {
                "id": "sl-1", "symbol": "SPY", "side": "sell", "type": "stop",
                "group_id": "group-1", "group_role": "stop_loss",
                "qty": "2", "filled_qty": "0", "status": "new", "stop_price": "99",
                "legs": None,
            },
        ],
    }


def _position():
    return {"symbol": "SPY", "asset_class": "us_equity", "side": "long", "qty": "2"}


def _assess(parent=None, position=None, **kw):
    return assess_equity_bracket_truth(
        parent=_bracket() if parent is None else parent,
        position=_position() if position is None else position,
        expected_client_order_id=kw.get("client", "tradeagent-bracket-1"),
        symbol=kw.get("symbol", "SPY"),
        side=kw.get("side", "buy"),
        quantity=kw.get("quantity", 2),
        stop_loss=kw.get("stop", 99),
        take_profit=kw.get("target", 104),
    )


def test_nested_bracket_normalization_preserves_exact_parent_leg_truth():
    normalized = alpaca_paper._normalize_order(_bracket())
    assert normalized["order_class"] == "bracket"
    assert normalized["group_id"] == "group-1"
    assert normalized["group_role"] == "entry"
    assert [x["id"] for x in normalized["legs"]] == ["tp-1", "sl-1"]
    assert normalized["legs"][0]["limit_price"] == "104"
    assert normalized["legs"][1]["stop_price"] == "99"
    assert normalized["legs"][0]["legs"] is None
    assert _assess(parent=normalized).verified


def test_broker_lookup_requests_nested_truth_explicitly(monkeypatch):
    calls = []
    def _lookup(path, *, params=None):
        calls.append((path, params))
        return _bracket()
    monkeypatch.setattr(alpaca_paper, "_request_optional_json", _lookup)
    order_id = alpaca_paper.get_order_by_id("parent-1", nested=True)
    cid = alpaca_paper.get_order_by_client_order_id("tradeagent-bracket-1", nested=True)
    assert calls == [
        ("/v2/orders/parent-1", {"nested": True}),
        ("/v2/orders:by_client_order_id", {"client_order_id": "tradeagent-bracket-1", "nested": True}),
    ]
    assert order_id["legs"][1]["id"] == "sl-1"
    assert cid["legs"][0]["id"] == "tp-1"


def test_legacy_simple_order_lookup_keeps_default_query_shape(monkeypatch):
    calls = []
    def _lookup(path, *, params=None):
        calls.append((path, params))
        return {"id": "old-simple", "type": "market", "status": "filled"}
    monkeypatch.setattr(alpaca_paper, "_request_optional_json", _lookup)
    assert alpaca_paper.get_order_by_id("old-simple")["legs"] is None
    assert alpaca_paper.get_order_by_client_order_id("old-client")["order_class"] is None
    assert calls == [
        ("/v2/orders/old-simple", None),
        ("/v2/orders:by_client_order_id", {"client_order_id": "old-client"}),
    ]


@pytest.mark.parametrize("legs", [{"id": "bad"}, ["invalid"], [{"id": "leg", "legs": [{"id": "nested"}]}]])
def test_malformed_nested_leg_data_is_never_silently_dropped(legs):
    raw = _bracket()
    raw["legs"] = legs
    with pytest.raises(alpaca_paper.AlpacaPaperError, match="unexpected shape"):
        alpaca_paper._normalize_order(raw)


def test_complete_broker_parent_legs_and_position_truth_are_verifiable():
    found = _assess()
    assert found.verified
    assert found.parent_order_id == "parent-1"
    assert found.take_profit_order_id == "tp-1"
    assert found.stop_order_id == "sl-1"


@pytest.mark.parametrize("change", [
    lambda p: p.update(status="partially_filled"),
    lambda p: p.update(order_class="simple"),
    lambda p: p.update(client_order_id="other-client"),
    lambda p: p.update(group_role="stop_loss"),
    lambda p: p.update(filled_qty="1"),
    lambda p: p.update(legs=None),
    lambda p: p.update(legs=[]),
    lambda p: p["legs"][0].update(status="held"),
    lambda p: p["legs"][0].update(status="filled", filled_qty="2"),
    lambda p: p["legs"][1].update(status="pending_cancel"),
    lambda p: p["legs"][1].update(group_id="unrelated"),
    lambda p: p["legs"][1].update(id="tp-1"),
    lambda p: p["legs"][1].update(group_role="take_profit"),
    lambda p: p["legs"][0].update(limit_price="105"),
    lambda p: p["legs"][1].update(stop_price="97"),
    lambda p: p["legs"][0].update(qty="1"),
    lambda p: p["legs"][1].update(side="buy"),
    lambda p: p["legs"][1].update(replaced_by="new-leg-id"),
])
def test_unconfirmed_or_mismatched_bracket_never_counts_as_protected(change):
    parent = _bracket()
    change(parent)
    actual = _assess(parent=parent)
    assert not actual.verified
    assert actual.reason
    assert actual.stop_order_id is None


@pytest.mark.parametrize("position", [
    {}, {"symbol": "QQQ", "side": "long", "asset_class": "us_equity", "qty": "2"},
    {"symbol": "SPY", "side": "short", "asset_class": "us_equity", "qty": "2"},
    {"symbol": "SPY", "side": "long", "asset_class": "us_equity", "qty": "1"},
])
def test_missing_or_mismatched_open_position_cannot_prove_protection(position):
    assert not _assess(position=position).verified


def test_optional_pre_october_group_fields_cannot_silently_fake_missing_legs():
    parent = _bracket()
    parent.pop("group_id")
    parent.pop("group_role")
    for leg in parent["legs"]:
        leg.pop("group_id")
        leg.pop("group_role")
    assert _assess(parent=parent).verified
    parent["legs"] = None
    assert not _assess(parent=parent).verified


def test_truth_assessment_is_strictly_pure_and_never_calls_broker(monkeypatch):
    monkeypatch.setattr(
        alpaca_paper, "_request_mutation_json",
        lambda *a, **k: pytest.fail("read-only proof cannot transmit broker mutation"),
    )
    assert _assess().verified
