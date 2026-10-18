from __future__ import annotations

import pytest

from backend.services.alpaca_paper import build_equity_bracket_market_order


@pytest.mark.parametrize("side,ref,stop,target", [
    ("buy", 101.0, 99.0, 104.0),
    ("sell", 101.0, 104.0, 99.0),
])
def test_equity_bracket_payload_contains_real_broker_protective_legs(side, ref, stop, target):
    body = build_equity_bracket_market_order(
        symbol="SPY", side=side, client_order_id="tradeagent-protected-123",
        qty=2.0, reference_price=ref, stop_loss=stop, take_profit=target,
    )
    assert body == {
        "symbol": "SPY",
        "side": side,
        "type": "market",
        "time_in_force": "gtc",
        "order_class": "bracket",
        "client_order_id": "tradeagent-protected-123",
        "qty": "2.0",
        "take_profit": {"limit_price": str(float(target))},
        "stop_loss": {"stop_price": str(float(stop))},
    }
    assert "notional" not in body


@pytest.mark.parametrize("overrides", [
    {"symbol": "BTC/USD"},
    {"qty": 0.5},
    {"qty": 0},
    {"qty": float("nan")},
    {"qty": float("inf")},
    {"stop_loss": 0},
    {"stop_loss": float("nan")},
    {"stop_loss": 99.12345},
    {"take_profit": 104.12345},
    {"side": "flat"},
    {"client_order_id": ""},
    {"client_order_id": "x" * 49},
    {"reference_price": 97.0},
    {"stop_loss": 105.0},
    {"take_profit": 98.0},
])
def test_equity_bracket_payload_fails_closed_on_unsupported_or_unprotected_requests(overrides):
    data = dict(
        symbol="SPY", side="buy", client_order_id="tradeagent-protected-abc",
        qty=2, reference_price=101, stop_loss=99, take_profit=104,
    )
    data.update(overrides)
    with pytest.raises(ValueError):
        build_equity_bracket_market_order(**data)


def test_subdollar_price_precision_and_bracket_side_ordering():
    body = build_equity_bracket_market_order(
        symbol="PENNY", side="buy", client_order_id="tradeagent-price",
        qty=1, reference_price=0.5, stop_loss=0.4999, take_profit=0.5123,
    )
    assert body["stop_loss"]["stop_price"] == "0.4999"
    with pytest.raises(ValueError):
        build_equity_bracket_market_order(
            symbol="PENNY", side="sell", client_order_id="tradeagent-price",
            qty=1, reference_price=0.5, stop_loss=0.49, take_profit=0.51,
        )


def test_pure_bracket_builder_does_not_transmit_orders(monkeypatch):
    from backend.services import alpaca_paper
    monkeypatch.setattr(
        alpaca_paper, "_request_mutation_json",
        lambda *a, **k: pytest.fail("payload builder must never contact broker"),
    )
    build_equity_bracket_market_order(
        symbol="AAPL", side="buy", client_order_id="tradeagent-pure",
        qty=1, reference_price=250, stop_loss=245, take_profit=260,
    )
