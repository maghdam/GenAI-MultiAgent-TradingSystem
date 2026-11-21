"""Isolated MCP research facade tests: no real cTrader/HTTP/order calls."""
from __future__ import annotations

import asyncio
import hashlib
import json
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
    # httpx.Response(json=...) rejects NaN while constructing the mock,
    # before gateway validation runs. Supply raw upstream text instead to
    # exercise the gateway's own fail-closed non-finite price guard.
    raw_upstream = json.dumps({"candles": rows}, allow_nan=True)
    client_with_response(
        monkeypatch,
        httpx.Response(200, content=raw_upstream, headers={"content-type": "application/json"}),
    )
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


def test_catalog_reads_only_fixed_loopback_get_and_omits_cwd_and_source(monkeypatch):
    seen = client_with_response(monkeypatch, httpx.Response(200, json={
        "files": ["sma.py", "rsi.py", "sma.py"],
        "cwd": "C:\\Users\\operator\\private-folder",
        "source": "secret source not for MCP",
        "account": {"token": "not-for-mcp"},
    }))
    result = gateway.list_studio_saved_strategies()

    assert result == {
        "source_route": "/api/studio/strategy-files",
        "strategies": ["rsi", "sma"],
        "observed_count": 3,
        "returned_count": 2,
        "truncated": False,
        "warning": (
            "Saved-file names only; catalog entries are not validated lifecycle, "
            "strategy performance, backtest results or authorization to trade."
        ),
    }
    assert "private-folder" not in json.dumps(result)
    assert "secret source" not in json.dumps(result)
    assert "token" not in json.dumps(result)
    assert len(seen) == 1
    request = seen[0]
    assert request.method == "GET"
    assert (request.url.scheme, request.url.host, request.url.port, request.url.path) == (
        "http", "127.0.0.1", 4000, "/api/studio/strategy-files",
    )
    assert not request.url.params


def test_catalog_can_be_empty_without_fabricating_names(monkeypatch):
    client_with_response(monkeypatch, httpx.Response(200, json={"files": [], "cwd": "C:\\private"}))
    result = gateway.list_studio_saved_strategies()
    assert result["strategies"] == []
    assert result["observed_count"] == 0
    assert result["returned_count"] == 0
    assert result["truncated"] is False


def test_catalog_bounded_deterministic_sort_and_truncation(monkeypatch):
    filenames = [f"strategy_{i:03d}.py" for i in reversed(range(65))]
    client_with_response(monkeypatch, httpx.Response(200, json={"files": filenames}))
    result = gateway.list_studio_saved_strategies()
    assert result["observed_count"] == 65
    assert result["returned_count"] == 50
    assert result["truncated"] is True
    assert result["strategies"] == [f"strategy_{i:03d}" for i in range(50)]


def test_catalog_duplicate_entries_do_not_fake_truncation(monkeypatch):
    filenames = [f"strategy_{i:03d}.py" for i in range(45)] + ["strategy_000.py"] * 25
    client_with_response(monkeypatch, httpx.Response(200, json={"files": filenames}))
    result = gateway.list_studio_saved_strategies()
    assert result["observed_count"] == 70
    assert result["returned_count"] == 45
    assert result["truncated"] is False


@pytest.mark.parametrize("payload", [
    None, [], {}, {"files": None}, {"files": {}}, {"files": "sma.py"},
    {"files": [0]}, {"files": [True]}, {"files": [None]},
    {"files": ["sma.py", {"name": "rsi.py"}]},
])
def test_catalog_rejects_bad_envelope_and_entries(monkeypatch, payload):
    client_with_response(monkeypatch, httpx.Response(200, json=payload))
    with pytest.raises(RuntimeError, match="catalog|saved-strategy"):
        gateway.list_studio_saved_strategies()


@pytest.mark.parametrize("filename", [
    "../orders.py", "/tmp/secret.py", r"C:\\secret.py", "sample.env",
    "SMA.py", "strange.PY", "script.py/other", ".py", "x" * 65 + ".py",
    "x.py\nsecret", "x/../y.py", "symbol:code.py", "test.py.py",
])
def test_catalog_rejects_unsafe_saved_filename(monkeypatch, filename):
    client_with_response(monkeypatch, httpx.Response(200, json={"files": [filename]}))
    with pytest.raises(RuntimeError, match="unsafe"):
        gateway.list_studio_saved_strategies()


def test_catalog_rejects_too_many_upstream_files(monkeypatch):
    client_with_response(monkeypatch, httpx.Response(200, json={"files": ["sma.py"] * 513}))
    with pytest.raises(RuntimeError, match="observed-file limit"):
        gateway.list_studio_saved_strategies()


def test_catalog_rejects_oversize_response_even_if_valid_json(monkeypatch):
    huge_response = {"files": [], "cwd": "x" * 65_000}
    client_with_response(monkeypatch, httpx.Response(200, json=huge_response))
    with pytest.raises(RuntimeError, match="size limit"):
        gateway.list_studio_saved_strategies()


@pytest.mark.parametrize("response", [
    httpx.Response(503, json={"detail": "backend unavailable"}),
    httpx.Response(302, headers={"Location": "https://example.org/sensitive"}),
    httpx.Response(200, content=b"not valid json"),
])
def test_catalog_http_redirect_and_json_fail_closed(monkeypatch, response):
    seen = client_with_response(monkeypatch, response)
    with pytest.raises(RuntimeError, match="Local TradeAgent strategy catalog"):
        gateway.list_studio_saved_strategies()
    assert len(seen) == 1
    assert seen[0].method == "GET"
    assert seen[0].url.path == "/api/studio/strategy-files"


def test_catalog_does_not_accept_external_urls_or_strategy_names():
    # A zero-argument, fixed-route API; no URL, file path, or SQL can be supplied.
    import inspect

    assert list(inspect.signature(gateway.list_studio_saved_strategies).parameters) == []



def lifecycle_record(*, evidence_count=1, metrics=None, stage="backtested"):
    metrics = metrics if metrics is not None else {
        "Number of Trades": 40.0,
        "Total Return [%]": 8.5,
        "Max Drawdown [%]": -4.0,
        "Fees [bps]": 1.5,
        "Win Rate [%]": 52.5,
        "private_note": "do not expose arbitrary metrics",
    }
    return {
        "strategy": "sma",
        "version": 2,
        "version_hash": "a" * 64,
        "stage": stage,
        "current_source": True,
        "hypothesis": "private research description",
        "blockers": ["private blocker"],
        "gates": {"private_policy": "do not expose"},
        "transitions": [{"operator": "private operator", "reason": "private reason"}],
        "evidence": [{
            "evidence_type": "development_backtest",
            "passed": i % 2 == 0,
            "created_at": "2026-11-18T10:45:00",
            "summary": "private summary not for MCP",
            "context": {"broker_account": "private account", "symbol": "XAUUSD"},
            "metrics": metrics,
            "id": i + 1,
            "lifecycle_id": 2222,
        } for i in range(evidence_count)],
    }


def test_recorded_evidence_reads_only_fixed_loopback_get_and_projects_fields(monkeypatch):
    seen = client_with_response(monkeypatch, httpx.Response(200, json=lifecycle_record()))
    payload = gateway.get_studio_recorded_evidence("sma")
    assert payload == {
        "source_route": "/api/studio/lifecycle/{strategy}",
        "strategy": "sma",
        "version": 2,
        "version_hash": "a" * 64,
        "stage": "backtested",
        "current_source": True,
        "latest_version_only": True,
        "observed_evidence_count": 1,
        "returned_evidence_count": 1,
        "truncated": False,
        "evidence": [{
            "evidence_type": "development_backtest",
            "passed": True,
            "recorded_at_utc": "2026-11-18T10:45:00+00:00",
            "metrics": {
                "Number of Trades": 40.0,
                "Total Return [%]": 8.5,
                "Max Drawdown [%]": -4.0,
                "Fees [bps]": 1.5,
                "Win Rate [%]": 52.5,
            },
        }],
        "warning": (
            "Previously stored lifecycle evidence only, not a complete backtest report, "
            "profit guarantee or authorization to execute or trade."
        ),
    }
    encoded = json.dumps(payload)
    for forbidden in ("private", "context", "operator", "lifecycle_id", "hypothesis", "blockers", "source_code"):
        assert forbidden not in encoded
    assert len(seen) == 1
    request = seen[0]
    assert request.method == "GET"
    assert (request.url.scheme, request.url.host, request.url.port, request.url.path) == (
        "http", "127.0.0.1", 4000, "/api/studio/lifecycle/sma",
    )
    assert not request.url.params


def test_recorded_evidence_empty_is_honestly_empty(monkeypatch):
    client_with_response(monkeypatch, httpx.Response(200, json=lifecycle_record(evidence_count=0, stage="draft")))
    result = gateway.get_studio_recorded_evidence("sma")
    assert result["stage"] == "draft"
    assert result["observed_evidence_count"] == 0
    assert result["returned_evidence_count"] == 0
    assert result["truncated"] is False
    assert result["evidence"] == []


def test_recorded_evidence_latest_only_limit_and_provenance(monkeypatch):
    source = lifecycle_record(evidence_count=4)
    source["evidence"][0]["created_at"] = "2026-11-18T11:45:00+01:00"
    client_with_response(monkeypatch, httpx.Response(200, json=source))
    result = gateway.get_studio_recorded_evidence("sma", limit=2)
    assert result["latest_version_only"] is True
    assert result["version"] == 2
    assert result["observed_evidence_count"] == 4
    assert result["returned_evidence_count"] == 2
    assert result["truncated"] is True
    assert [x["passed"] for x in result["evidence"]] == [True, False]
    assert result["evidence"][0]["recorded_at_utc"] == "2026-11-18T10:45:00+00:00"


@pytest.mark.parametrize("bad", [
    "", "SMA", "sma.py", "x" * 65, "a/b", "../orders",
    "http://evil", "sma?x=1", "a\\b", "a b", None, 33,
])
def test_recorded_evidence_rejects_unsafe_strategy_without_network(monkeypatch, bad):
    monkeypatch.setattr(gateway.httpx, "Client", lambda **kw: pytest.fail("network attempted"))
    with pytest.raises(ValueError):
        gateway.get_studio_recorded_evidence(bad)


@pytest.mark.parametrize("bad", [0, 6, -1, True, 2.0, "2", None])
def test_recorded_evidence_rejects_invalid_limit_without_network(monkeypatch, bad):
    monkeypatch.setattr(gateway.httpx, "Client", lambda **kw: pytest.fail("network attempted"))
    with pytest.raises(ValueError):
        gateway.get_studio_recorded_evidence("sma", bad)


@pytest.mark.parametrize("override", [
    {"strategy": "unrequested"},
    {"version": True},
    {"version": 0},
    {"version_hash": "invalid"},
    {"version_hash": "B" * 64},
    {"stage": "live"},
    {"stage": []},
    {"current_source": 1},
    {"evidence": None},
    {"evidence": {}},
])
def test_recorded_evidence_rejects_bad_lifecycle_fields(monkeypatch, override):
    body = lifecycle_record()
    body.update(override)
    client_with_response(monkeypatch, httpx.Response(200, json=body))
    with pytest.raises(RuntimeError, match="invalid recorded-evidence lifecycle"):
        gateway.get_studio_recorded_evidence("sma")


@pytest.mark.parametrize("entry", [
    None, 1, {},
    {"evidence_type": []},
    {"evidence_type": "unrecognized"},
    {"passed": 1},
    {"metrics": None},
    {"metrics": "a string"},
    {"created_at": "not a timestamp"},
    {"created_at": None},
])
def test_recorded_evidence_rejects_malformed_entry(monkeypatch, entry):
    body = lifecycle_record()
    if isinstance(entry, dict) and entry:
        body["evidence"][0].update(entry)
    else:
        body["evidence"][0] = entry
    client_with_response(monkeypatch, httpx.Response(200, json=body))
    with pytest.raises(RuntimeError, match="recorded-evidence"):
        gateway.get_studio_recorded_evidence("sma")


@pytest.mark.parametrize("bad", [True, "15", None, 1e10, 10**1000, float("inf"), float("nan")])
def test_recorded_evidence_rejects_invalid_numeric_metric(monkeypatch, bad):
    body = lifecycle_record()
    body["evidence"][0]["metrics"]["Total Return [%]"] = bad
    raw = json.dumps(body, allow_nan=True)
    client_with_response(monkeypatch, httpx.Response(200, content=raw))
    with pytest.raises(RuntimeError, match="invalid recorded-evidence numeric metric"):
        gateway.get_studio_recorded_evidence("sma")


def test_recorded_evidence_rejects_oversized_upstream_payload(monkeypatch):
    body = lifecycle_record()
    body["hypothesis"] = "private" * 40000
    client_with_response(monkeypatch, httpx.Response(200, json=body))
    with pytest.raises(RuntimeError, match="size limit"):
        gateway.get_studio_recorded_evidence("sma")


def test_recorded_evidence_rejects_too_many_entries(monkeypatch):
    client_with_response(monkeypatch, httpx.Response(200, json=lifecycle_record(evidence_count=201)))
    with pytest.raises(RuntimeError, match="observed-entry limit"):
        gateway.get_studio_recorded_evidence("sma")


@pytest.mark.parametrize("response", [
    httpx.Response(404, json={"detail": "private missing strategy details"}),
    httpx.Response(503, json={"detail": "backend unavailable"}),
    httpx.Response(302, headers={"Location": "http://evil.test/collect"}),
    httpx.Response(200, content=b"not-json"),
    httpx.Response(200, json={"unexpected": "envelope"}),
])
def test_recorded_evidence_http_redirect_invalid_json_fail_closed(monkeypatch, response):
    seen = client_with_response(monkeypatch, response)
    with pytest.raises(RuntimeError, match="TradeAgent"):
        gateway.get_studio_recorded_evidence("sma")
    assert len(seen) == 1
    assert seen[0].method == "GET"



def strategy_source(source="def signals(df):\n    return df['close'] > 20\n", **overrides):
    return dict({"strategy": "sma", "filename": "sma.py", "source": source}, **overrides)


def test_source_preview_fixed_get_and_never_leaks_literals_comments_or_other_fields(monkeypatch):
    raw_source = (
        "# hidden private account note\n"
        "API_KEY = 'very-secret-token'\n"
        "period = 20\n"
        "def signals(df):\n"
        "    return df['close'] > 20\n"
    )
    payload = strategy_source(raw_source)
    payload["cwd"] = "C:\\\\Users\\\\operator\\\\hidden-folder"
    payload["account_token"] = "hidden account"
    seen = client_with_response(monkeypatch, httpx.Response(200, json=payload))
    result = gateway.inspect_studio_saved_strategy("sma")
    assert set(result) == {
        "source_route", "strategy", "filename", "source_sha256", "source_bytes",
        "source_lines", "signals_entrypoint_detected", "preview_mode",
        "source_preview", "warning",
    }
    assert result["source_route"] == "/api/studio/strategy-file/{strategy}"
    assert result["strategy"] == "sma"
    assert result["source_sha256"] == hashlib.sha256(raw_source.encode()).hexdigest()
    assert result["signals_entrypoint_detected"] is True
    assert "def signals(df):" in result["source_preview"]
    assert "period = 20" in result["source_preview"]
    assert "REDACTED_LITERAL" in result["source_preview"]
    assert "very-secret-token" not in json.dumps(result)
    assert "private account" not in json.dumps(result)
    assert "hidden-folder" not in json.dumps(result)
    assert "'close'" not in json.dumps(result)
    assert "source execution" in result["warning"]
    assert len(seen) == 1
    assert seen[0].method == "GET"
    assert (seen[0].url.scheme, seen[0].url.host, seen[0].url.port, seen[0].url.path) == (
        "http", "127.0.0.1", 4000, "/api/studio/strategy-file/sma"
    )
    assert not seen[0].url.params


@pytest.mark.parametrize("bad", [
    "", "SMA", "sma.py", "../orders", "a/b", "a\\\\b", "abc?next",
    "https://remote", "x" * 65, 0, None, True,
])
def test_source_preview_rejects_unsafe_names_before_network(monkeypatch, bad):
    monkeypatch.setattr(gateway.httpx, "Client", lambda **kw: pytest.fail("network attempted"))
    with pytest.raises(ValueError):
        gateway.inspect_studio_saved_strategy(bad)


@pytest.mark.parametrize("body", [
    {}, [], None, {"strategy": "sma", "filename": "sma.py", "source": None},
    {"strategy": "sma", "filename": "sma.py", "source": 23},
    {"strategy": "wrong", "filename": "sma.py", "source": "x=1"},
    {"strategy": "sma", "filename": "../sma.py", "source": "x=1"},
    {"strategy": "sma", "filename": "sma.py", "source": ""},
])
def test_source_preview_rejects_invalid_envelope(monkeypatch, body):
    client_with_response(monkeypatch, httpx.Response(200, json=body))
    with pytest.raises(RuntimeError):
        gateway.inspect_studio_saved_strategy("sma")


@pytest.mark.parametrize("source", [
    "print('broken'", "x=1\x00\ny=2", "x='" + ("x" * 20000) + "'",
    "x=1\n" * 401,
])
def test_source_preview_rejects_invalid_or_large_source(monkeypatch, source):
    client_with_response(monkeypatch, httpx.Response(200, json=strategy_source(source)))
    with pytest.raises(RuntimeError):
        gateway.inspect_studio_saved_strategy("sma")


@pytest.mark.parametrize("response", [
    httpx.Response(404, json={"detail": "private server directory"}),
    httpx.Response(503, json={"detail": "backend unavailable"}),
    httpx.Response(302, headers={"Location": "https://example.org/exfiltrate"}),
    httpx.Response(200, content=b"not-json"),
    httpx.Response(200, json={"strategy": "sma", "filename": "sma.py", "source": "x=1", "private": "a" * 49000}),
])
def test_source_preview_http_errors_redirects_and_big_envelopes_fail_closed(monkeypatch, response):
    seen = client_with_response(monkeypatch, response)
    with pytest.raises(RuntimeError, match="TradeAgent"):
        gateway.inspect_studio_saved_strategy("sma")
    assert len(seen) == 1
    assert seen[0].method == "GET"
    assert seen[0].url.path == "/api/studio/strategy-file/sma"


def test_source_preview_rejects_invalid_unicode_without_leaking_raw_payload(monkeypatch):
    client_with_response(monkeypatch, httpx.Response(200, json=strategy_source("x='" + chr(0xD800) + "'")))
    with pytest.raises(RuntimeError, match="invalid UTF-8"):
        gateway.inspect_studio_saved_strategy("sma")


def test_source_preview_parses_but_does_not_execute_source(monkeypatch):
    code = "raise RuntimeError('would be executed')\ndef signals(df):\n    return 1\n"
    client_with_response(monkeypatch, httpx.Response(200, json=strategy_source(code)))
    result = gateway.inspect_studio_saved_strategy("sma")
    assert result["signals_entrypoint_detected"] is True
    assert "raise RuntimeError" in result["source_preview"]
    assert "would be executed" not in result["source_preview"]


def test_source_preview_drops_inline_fstring_secret_and_masks_large_numeric_literals(monkeypatch):
    raw = (
        "secret = f'private-credential-{1234}'\n"
        "position_id = 999999999\n"
        "def helper():\n    return secret\n"
    )
    client_with_response(monkeypatch, httpx.Response(200, json=strategy_source(raw)))
    result = gateway.inspect_studio_saved_strategy("sma")
    assert "private-credential" not in json.dumps(result)
    assert "999999999" not in json.dumps(result)
    assert "999999999" not in result["source_preview"]
    assert result["signals_entrypoint_detected"] is False
    assert "source_sha256" in result


def test_mcp_sdk_v2_registers_exactly_four_approved_research_tools():
    from mcp.server.mcpserver import MCPServer

    server = gateway.create_server()
    assert isinstance(server, MCPServer)
    tools = asyncio.run(server.list_tools())
    assert [tool.name for tool in tools] == [
        "get_ctrader_candles",
        "list_studio_saved_strategies",
        "get_studio_recorded_evidence",
        "inspect_studio_saved_strategy",
    ]


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
