from __future__ import annotations

import httpx
import pytest

from backend import chatgpt_mcp_readonly as gateway


def test_reads_only_local_market_candles(monkeypatch) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"candles": [{"time": 1, "open": 2.0, "high": 3.0, "low": 1.0, "close": 2.5}]})

    original_client = httpx.Client

    def client_factory(*, timeout, trust_env):
        assert trust_env is False
        return original_client(timeout=timeout, trust_env=trust_env, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(gateway.httpx, "Client", client_factory)
    result = gateway.get_ctrader_candles("eurusd", "m5", 20)

    assert result["provider"] == "ctrader_via_local_tradeagent"
    assert result["symbol"] == "EURUSD"
    assert len(result["candles"]) == 1
    assert seen[0].url.host == "127.0.0.1"
    assert seen[0].url.path == "/api/market/candles"
    assert seen[0].url.params["live"] == "true"
    assert seen[0].method == "GET"


@pytest.mark.parametrize("symbol", ["", "../orders", "http://evil.com", "EUR/USD", "EURUSD?foo=1"])
def test_rejects_unsafe_symbol_without_network(symbol, monkeypatch) -> None:
    monkeypatch.setattr(gateway.httpx, "Client", lambda *a, **k: pytest.fail("unexpected network call"))
    with pytest.raises(ValueError):
        gateway.get_ctrader_candles(symbol)


@pytest.mark.parametrize("timeframe,count", [("S1", 10), ("M1", 0), ("M5", 501), ("H4", 1.0), ("M5", True)])
def test_rejects_unsupported_input_without_network(timeframe, count, monkeypatch) -> None:
    monkeypatch.setattr(gateway.httpx, "Client", lambda *a, **k: pytest.fail("unexpected network call"))
    with pytest.raises(ValueError):
        gateway.get_ctrader_candles("EURUSD", timeframe, count)


def test_rejects_malformed_backend_payload(monkeypatch) -> None:
    original_client = httpx.Client
    monkeypatch.setattr(
        gateway.httpx,
        "Client",
        lambda *, timeout, trust_env: original_client(
            timeout=timeout,
            trust_env=trust_env,
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"candles": [{}]})),
        ),
    )
    with pytest.raises(RuntimeError, match="malformed"):
        gateway.get_ctrader_candles("US500")


def test_no_order_tools_exist() -> None:
    assert not hasattr(gateway, "place_order")
    assert not hasattr(gateway, "cancel_order")
    assert not hasattr(gateway, "close_position")
