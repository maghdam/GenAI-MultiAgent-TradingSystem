"""Isolated MCP research facade tests: no real cTrader/HTTP/order calls."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from backend import chatgpt_mcp_readonly as gateway


def client_with_response(monkeypatch, response):
    seen = []
    original = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return response(request) if callable(response) else response

    def factory(*, timeout, trust_env, follow_redirects):
        assert trust_env is False
        assert follow_redirects is False
        return original(timeout=timeout, trust_env=trust_env,
                        follow_redirects=follow_redirects, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(gateway.httpx, "Client", factory)
    return seen


def sample(time=1760000000):
    return {"time": time, "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0}


def test_reads_only_tradeagent_loopback_market_data(monkeypatch):
    rows = [sample(), sample(1760000900)]
    seen = client_with_response(monkeypatch, httpx.Response(200, json={"candles": rows}))
    result = gateway.get_ctrader_candles("us500", "m15", 20)
    assert result["provider_route"] == "ctrader_via_local_tradeagent"
    assert (result["symbol"], result["timeframe"]) == ("US500", "M15")
    assert result["requested_bars"] == 20
    assert result["observed_bars"] == 2  # Honest when broker returns fewer.
    assert result["first_bar_utc"] < result["last_bar_utc"]
    assert "unverified" in result["timestamp_convention"]
    assert "not independently attested" in result["data_provenance"]
    assert result["candles"] == rows
    assert len(seen) == 1
    request = seen[0]
    assert request.method == "GET"
    assert (request.url.host, request.url.port, request.url.path) == ("127.0.0.1", 4000, "/api/market/candles")
    assert request.url.params["live"] == "true"
    assert request.url.params["num_bars"] == "20"
    assert request.url.params["symbol"] == "US500"


@pytest.mark.parametrize("symbol", ["", "../orders", "http://host", "EUR/USD", "USD?x=1", "A" * 33, None])
def test_rejects_unsafe_symbol_without_network(monkeypatch, symbol):
    monkeypatch.setattr(gateway.httpx, "Client", lambda **kw: pytest.fail("network attempted"))
    with pytest.raises(ValueError):
        gateway.get_ctrader_candles(symbol)


@pytest.mark.parametrize("timeframe,count", [
    ("W1", 10), ("S1", 20), ("M5", 0), ("M5", 501),
    ("H4", 1.0), ("M5", True), ("M5", "20"), (None, 20),
])
def test_rejects_invalid_timeframe_or_size_without_network(monkeypatch, timeframe, count):
    monkeypatch.setattr(gateway.httpx, "Client", lambda **kw: pytest.fail("network attempted"))
    with pytest.raises(ValueError):
        gateway.get_ctrader_candles("US30", timeframe, count)


@pytest.mark.parametrize("rows", [
    [],
    [{}],
    [dict(sample(), time=True)],
    [sample(1760000000), sample(1760000000)],
    [sample(1760000900), sample(1760000000)],
    [dict(sample(), close=float("nan"))],
    [dict(sample(), high=98.0)],
    [dict(sample(), open="bad")],
])
def test_rejects_malformed_rows_fail_closed(monkeypatch, rows):
    client_with_response(monkeypatch, httpx.Response(200, json={"candles": rows}))
    with pytest.raises(RuntimeError):
        gateway.get_ctrader_candles("US30")


@pytest.mark.parametrize("body", [None, {"candles": "not a list"}, {"candles": [sample(), sample(), sample()]}])
def test_rejects_invalid_payloads(monkeypatch, body):
    client_with_response(monkeypatch, httpx.Response(200, json=body))
    with pytest.raises(RuntimeError):
        gateway.get_ctrader_candles("US30", num_bars=2)


def test_http_failure_does_not_fallback_to_generated_market_data(monkeypatch):
    client_with_response(monkeypatch, httpx.Response(503, json={"detail": "unavailable"}))
    with pytest.raises(RuntimeError, match="Local TradeAgent"):
        gateway.get_ctrader_candles("XAUUSD")


def test_mcp_sdk_v2_registers_exactly_one_tool():
    from mcp.server.mcpserver import MCPServer

    server = gateway.create_server()
    assert isinstance(server, MCPServer)
    tools = asyncio.run(server.list_tools())
    assert [tool.name for tool in tools] == ["get_ctrader_candles"]


def test_gateway_run_only_on_loopback(monkeypatch):
    observed = []
    monkeypatch.setattr(gateway, "create_server", lambda: SimpleNamespace(run=lambda **kw: observed.append(kw)))
    gateway.main()
    assert observed == [{
        "transport": "streamable-http", "host": "127.0.0.1",
        "port": 8765, "stateless_http": True, "json_response": True,
    }]


def test_no_broker_or_mutation_tools_exposed():
    for forbidden in ("place_order", "close_position", "cancel_order", "switch_account", "save_strategy"):
        assert not hasattr(gateway, forbidden)
