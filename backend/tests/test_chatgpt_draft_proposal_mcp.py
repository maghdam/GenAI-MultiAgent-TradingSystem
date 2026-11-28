"""§35 isolated HTTP token gate and proposal-only MCP tests.

All proposal persistence uses backend/tests/conftest.py's temporary SQLite.
Never start a listener or create an actual strategy as part of acceptance.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from backend import chatgpt_draft_proposal_mcp as gateway
from backend.storage.db import get_db


TOKEN = "a" * 64
OTHER_TOKEN = "b" * 64
CODE = (
    "import pandas as pd\n\n"
    "def signals(df):\n"
    "    return pd.Series(0, index=df.index)\n"
)


def _count(table):
    with get_db() as db:
        return db.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]


def _sample_app(calls):
    async def app(scope, receive, send):
        calls.append(scope["path"])
        body = b'{"ok":true}'
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        })
        await send({"type": "http.response.body", "body": body})

    return app


async def _request(
    token=TOKEN, *, origin=None, base_url="http://127.0.0.1:8766",
    client_ip="127.0.0.1", extra_headers=(), path="/mcp"
):
    calls = []
    gate = gateway._TokenGate(_sample_app(calls), TOKEN)
    headers = list(extra_headers)
    if token is not None:
        headers.append(("Authorization", "Bearer " + token))
    if origin is not None:
        headers.append(("Origin", origin))
    transport = httpx.ASGITransport(app=gate, client=(client_ip, 47000))
    async with httpx.AsyncClient(transport=transport, base_url=base_url) as client:
        response = await client.post(path, headers=headers, json={"ignore": True})
    return response, calls


def test_matching_token_on_fixed_loopback_delegates_to_inner_mcp():
    response, calls = asyncio.run(_request())
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert calls == ["/mcp"]


@pytest.mark.parametrize("provided,expected", [
    (None, 401),
    ("", 401),
    (OTHER_TOKEN, 401),
    (TOKEN.upper(), 401),
    (TOKEN + "0", 401),
], ids=["missing", "empty", "wrong", "different-case", "too-long"])
def test_missing_wrong_or_malformed_token_rejected_before_any_tool(provided, expected):
    response, calls = asyncio.run(_request(provided))
    assert response.status_code == expected
    assert response.json() == {"error": "request_denied"}
    assert not calls
    assert TOKEN not in response.text


def test_duplicate_credentials_rejected_even_if_one_is_correct():
    response, calls = asyncio.run(_request(
        extra_headers=[("Authorization", "Bearer " + TOKEN)]
    ))
    assert response.status_code == 401
    assert not calls


def test_origin_is_denied_even_with_correct_token():
    response, calls = asyncio.run(_request(origin="http://evil.example"))
    assert response.status_code == 403
    assert not calls


@pytest.mark.parametrize("url,ip,expected", [
    ("http://localhost:8766", "127.0.0.1", 403),
    ("http://127.0.0.1:8765", "127.0.0.1", 403),
    ("http://127.0.0.1:8766", "192.0.2.50", 403),
], ids=["other-host", "readonly-port", "remote-client"])
def test_alternate_host_readonly_port_or_remote_ip_denied(url, ip, expected):
    response, calls = asyncio.run(_request(base_url=url, client_ip=ip))
    assert response.status_code == expected
    assert not calls


def test_non_mcp_path_denied_even_when_authorized():
    response, calls = asyncio.run(_request(path="/api/studio/tasks"))
    assert response.status_code == 404
    assert not calls


def test_rejection_does_not_store_approval_or_strategy():
    response, calls = asyncio.run(_request(token=OTHER_TOKEN))
    assert response.status_code == 401
    assert not calls
    assert _count("studio_draft_approval_requests") == 0
    assert _count("strategy_lifecycles") == 0


@pytest.mark.parametrize("bad", [
    None, "", "short", "z" * 64, "A" * 64, "a" * 63,
], ids=["none", "empty", "short", "not-hex", "uppercase", "length-63"])
def test_weak_or_malformed_config_token_refused(bad):
    with pytest.raises(ValueError, match="proposal_mcp_token_unavailable"):
        gateway._validate_token(bad)


def test_proposal_writes_only_pending_hash_metadata_not_source_or_lifecycle():
    response = gateway.propose_studio_draft_for_review("us30_mcp_research", CODE)
    assert response["status"] == "pending"
    assert response["strategy"] == "us30_mcp_research"
    assert len(response["request_id"]) == 32
    assert len(response["source_sha256"]) == 64
    assert "source" not in response
    assert CODE not in json.dumps(response)
    assert response["expires_at_epoch"] > 0
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM studio_draft_approval_requests WHERE request_id=?",
            (response["request_id"],),
        ).fetchone()
        assert row is not None
        assert row["state"] == "pending"
        assert row["source_sha256"] == response["source_sha256"]
        assert "source" not in row.keys()
    assert _count("strategy_lifecycles") == 0


@pytest.mark.parametrize("bad_name,bad_source", [
    ("../bad", CODE),
    ("new", "open('/secrets')"),
    ("new", CODE * 400),
], ids=["path-traversal", "disallowed-source", "oversized-source"])
def test_invalid_proposals_cannot_create_ledger_or_lifecycle(bad_name, bad_source):
    with pytest.raises(ValueError, match="invalid_candidate"):
        gateway.propose_studio_draft_for_review(bad_name, bad_source)
    assert _count("studio_draft_approval_requests") == 0
    assert _count("strategy_lifecycles") == 0


def test_registered_tools_are_only_one_pending_proposal_and_five_readonly():
    from backend.chatgpt_mcp_readonly import create_server as read_server

    proposal_tools = {
        t.name for t in asyncio.run(gateway.create_server().list_tools())
    }
    assert proposal_tools == {"propose_studio_draft_for_review"}
    read_tools = {t.name for t in asyncio.run(read_server().list_tools())}
    assert read_tools == {
        "get_ctrader_candles",
        "list_studio_saved_strategies",
        "get_studio_recorded_evidence",
        "inspect_studio_saved_strategy",
        "validate_studio_strategy_draft",
    }
    assert proposal_tools.isdisjoint(read_tools)
    assert "consume_approved_once" not in proposal_tools
    assert "create_from_approved_local_request" not in proposal_tools


def test_main_requires_explicit_opt_in_before_loading_token(monkeypatch):
    monkeypatch.delenv("TRADEAGENT_ENABLE_DRAFT_PROPOSALS", raising=False)
    monkeypatch.setenv("TRADEAGENT_PROPOSAL_MCP_TOKEN", TOKEN)
    with pytest.raises(SystemExit, match="proposal_mcp_not_opted_in"):
        gateway.main()


def test_main_denies_missing_token_when_opted_in(monkeypatch):
    monkeypatch.setenv("TRADEAGENT_ENABLE_DRAFT_PROPOSALS", "1")
    monkeypatch.delenv("TRADEAGENT_PROPOSAL_MCP_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="proposal_mcp_token_unavailable"):
        gateway.main()


def test_main_starts_separate_loopback_server_only_with_opt_in(monkeypatch):
    import uvicorn

    captured = []
    fake = object()
    monkeypatch.setenv("TRADEAGENT_ENABLE_DRAFT_PROPOSALS", "1")
    monkeypatch.setenv("TRADEAGENT_PROPOSAL_MCP_TOKEN", TOKEN)
    monkeypatch.setattr(gateway, "create_app", lambda token: (
        fake if token == TOKEN else pytest.fail("unexpected token")
    ))
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: captured.append((app, kw)))
    gateway.main()
    assert captured == [(fake, {
        "host": "127.0.0.1", "port": 8766,
        "log_level": "warning", "access_log": False,
    })]
