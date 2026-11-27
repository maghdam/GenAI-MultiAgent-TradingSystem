"""Offline §33 approval ledger: no network, strategy save, broker, or backtest."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.services import studio_draft_approval as approval
from backend.storage.db import get_db


CODE = (
    "import pandas as pd\n\n"
    "def signals(df):\n"
    "    return pd.Series(0, index=df.index)\n"
)
OTHER = CODE.replace("Series(0,", "Series(1,")


def _tty(monkeypatch, typed: str):
    io = SimpleNamespace(
        stdin=SimpleNamespace(isatty=lambda: True),
        stdout=SimpleNamespace(isatty=lambda: True),
    )
    monkeypatch.setattr(approval, "sys", io)
    monkeypatch.setattr("builtins.input", lambda prompt: typed)
    return io


def _approved(monkeypatch, name="research_draft", source=CODE):
    req = approval.request_draft_approval(name, source)
    _tty(monkeypatch, name + " " + req["source_sha256"])
    result = approval.approve_from_local_console(req["request_id"], source)
    assert result["status"] == "approved"
    return req, result


def _state(request_id):
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM studio_draft_approval_requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
    return dict(row) if row else None


def test_proposal_stores_only_bounded_metadata_not_raw_source(monkeypatch):
    monkeypatch.setattr(approval, "_now", lambda: 1_000_000)
    req = approval.request_draft_approval("us30_proposal", CODE)
    assert req["status"] == "pending"
    assert req["strategy"] == "us30_proposal"
    assert len(req["request_id"]) == 32
    assert len(req["source_sha256"]) == 64
    assert req["expires_at_epoch"] == 1_000_600
    row = _state(req["request_id"])
    assert row["state"] == "pending"
    assert row["created_at_epoch"] == 1_000_000
    assert row["source_sha256"] == req["source_sha256"]
    assert "source" not in row and "code" not in row and "token" not in row
    assert CODE not in json.dumps(row)
    with get_db() as db:
        assert db.execute(
            "SELECT COUNT(*) AS n FROM strategy_lifecycles"
        ).fetchone()["n"] == 0


@pytest.mark.parametrize("name,source", [
    ("Upper", CODE),
    ("../dir", CODE),
    ("fine", "def signals(df):\n    open('private')\n"),
    ("fine", "x = 1"),
    ("fine", CODE * 400),
    (None, CODE),
    ("fine", None),
])
def test_bad_proposal_never_creates_row(name, source):
    with pytest.raises(approval.DraftApprovalError, match="invalid_candidate"):
        approval.request_draft_approval(name, source)
    with get_db() as db:
        assert db.execute(
            "SELECT COUNT(*) AS n FROM studio_draft_approval_requests"
        ).fetchone()["n"] == 0


def test_pending_cap_fails_closed_without_extra_row(monkeypatch):
    monkeypatch.setattr(approval, "MAX_OUTSTANDING_REQUESTS", 2)
    approval.request_draft_approval("one", CODE)
    approval.request_draft_approval("two", CODE)
    with pytest.raises(approval.DraftApprovalError, match="too_many_pending_requests"):
        approval.request_draft_approval("three", CODE)
    with get_db() as db:
        assert db.execute(
            "SELECT COUNT(*) AS n FROM studio_draft_approval_requests"
        ).fetchone()["n"] == 2


def test_approval_requires_separate_human_tty(monkeypatch):
    req = approval.request_draft_approval("draft", CODE)
    monkeypatch.setattr(
        approval, "sys",
        SimpleNamespace(
            stdin=SimpleNamespace(isatty=lambda: False),
            stdout=SimpleNamespace(isatty=lambda: True),
        ),
    )
    with pytest.raises(approval.DraftApprovalError, match="interactive_operator_required"):
        approval.approve_from_local_console(req["request_id"], CODE)
    assert _state(req["request_id"])["state"] == "pending"


def test_wrong_source_cannot_be_approved(monkeypatch):
    req = approval.request_draft_approval("draft", CODE)
    _tty(monkeypatch, "draft " + req["source_sha256"])
    with pytest.raises(approval.DraftApprovalError, match="source_fingerprint_mismatch"):
        approval.approve_from_local_console(req["request_id"], OTHER)
    assert _state(req["request_id"])["state"] == "pending"


def test_wrong_operator_confirmation_cannot_approve(monkeypatch, capsys):
    req = approval.request_draft_approval("draft", CODE)
    _tty(monkeypatch, "approved by AI")
    with pytest.raises(approval.DraftApprovalError, match="operator_approval_denied"):
        approval.approve_from_local_console(req["request_id"], CODE)
    assert _state(req["request_id"])["state"] == "pending"
    out = capsys.readouterr().out
    assert req["source_sha256"] in out
    assert CODE not in out


def test_pending_expiry_blocks_approval(monkeypatch):
    req = approval.request_draft_approval("draft", CODE)
    later = _state(req["request_id"])["expires_at_epoch"]
    monkeypatch.setattr(approval, "_now", lambda: later)
    _tty(monkeypatch, "draft " + req["source_sha256"])
    with pytest.raises(approval.DraftApprovalError, match="approval_expired"):
        approval.approve_from_local_console(req["request_id"], CODE)
    assert _state(req["request_id"])["state"] == "pending"


def test_successful_approval_is_hash_bound_short_lived_and_not_a_save(monkeypatch, capsys):
    req, result = _approved(monkeypatch)
    assert result["request_id"] == req["request_id"]
    assert result["source_sha256"] == req["source_sha256"]
    assert result["expires_at_epoch"] <= req["expires_at_epoch"]
    assert result["expires_at_epoch"] <= approval._now() + approval.APPROVAL_TTL_SECONDS
    row = _state(req["request_id"])
    assert row["state"] == "approved"
    assert row["approved_at_epoch"] is not None
    assert row["consumed_at_epoch"] is None
    assert CODE not in capsys.readouterr().out
    with get_db() as db:
        assert db.execute(
            "SELECT COUNT(*) AS n FROM strategy_lifecycles"
        ).fetchone()["n"] == 0
    with pytest.raises(approval.DraftApprovalError, match="request_not_pending"):
        approval.approve_from_local_console(req["request_id"], CODE)


def test_exact_approval_can_be_consumed_once_but_does_not_save(monkeypatch):
    req, _ = _approved(monkeypatch, name="draft")
    with pytest.raises(approval.DraftApprovalError, match="approval_tuple_mismatch"):
        approval.consume_approved_once(req["request_id"], "another", CODE)
    with pytest.raises(approval.DraftApprovalError, match="approval_tuple_mismatch"):
        approval.consume_approved_once(req["request_id"], "draft", OTHER)
    assert _state(req["request_id"])["state"] == "approved"
    result = approval.consume_approved_once(req["request_id"], "draft", CODE)
    assert result["approval_consumed"] is True
    assert result["source_sha256"] == req["source_sha256"]
    assert _state(req["request_id"])["state"] == "consumed"
    with pytest.raises(approval.DraftApprovalError, match="approval_not_available"):
        approval.consume_approved_once(req["request_id"], "draft", CODE)
    with get_db() as db:
        assert db.execute(
            "SELECT COUNT(*) AS n FROM strategy_lifecycles"
        ).fetchone()["n"] == 0


def test_approved_expiry_blocks_consume(monkeypatch):
    req, result = _approved(monkeypatch, name="draft")
    monkeypatch.setattr(approval, "_now", lambda: result["expires_at_epoch"])
    with pytest.raises(approval.DraftApprovalError, match="approval_expired"):
        approval.consume_approved_once(req["request_id"], "draft", CODE)
    assert _state(req["request_id"])["state"] == "approved"


@pytest.mark.parametrize("bad", ["", "../file", "A" * 32, "0" * 31, None, 5])
def test_unsafe_request_id_is_never_accepted(bad):
    with pytest.raises(approval.DraftApprovalError, match="invalid_request_id"):
        approval.consume_approved_once(bad, "draft", CODE)


def _thread_consume(args):
    from backend.storage import db as storage

    request_id, name = args
    try:
        try:
            approval.consume_approved_once(request_id, name, CODE)
            return "consumed"
        except approval.DraftApprovalError:
            return "rejected"
    finally:
        conn = getattr(storage._LOCAL, "connection", None)
        if conn is not None:
            conn.close()
            storage._LOCAL.connection = None


def test_simultaneous_consume_allows_one_winner(monkeypatch):
    req, _ = _approved(monkeypatch, name="draft")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            _thread_consume,
            [(req["request_id"], "draft"), (req["request_id"], "draft")],
        ))
    assert sorted(results) == ["consumed", "rejected"]
    assert _state(req["request_id"])["state"] == "consumed"


def test_cli_refuses_noninteractive_invocation(monkeypatch, tmp_path, capsys):
    from scripts import review_studio_draft_request as cli

    source_path = tmp_path / "reviewed.py"
    source_path.write_text(CODE, encoding="utf-8")
    monkeypatch.setattr(
        cli, "sys",
        SimpleNamespace(
            stdin=SimpleNamespace(isatty=lambda: False),
            stdout=SimpleNamespace(isatty=lambda: True),
        ),
    )
    result = cli.main(["--request-id", "0" * 32, "--source-file", str(source_path)])
    assert result == 2
    assert "interactive_operator_required" in capsys.readouterr().out


def test_cli_approves_exact_file_in_isolated_test_db(monkeypatch, tmp_path, capsys):
    from scripts import review_studio_draft_request as cli

    req = approval.request_draft_approval("draft", CODE)
    source_path = tmp_path / "reviewed.py"
    source_path.write_text(CODE, encoding="utf-8")
    io = _tty(monkeypatch, "draft " + req["source_sha256"])
    monkeypatch.setattr(cli, "sys", io)
    assert cli.main(["--request-id", req["request_id"], "--source-file", str(source_path)]) == 0
    output = capsys.readouterr().out
    assert '"status": "approved"' in output
    assert CODE not in output
    assert str(source_path) not in output
    assert _state(req["request_id"])["state"] == "approved"


def test_readonly_mcp_still_has_exactly_five_tools_no_write_gateway():
    from backend.chatgpt_mcp_readonly import create_server

    names = {item.name for item in asyncio.run(create_server().list_tools())}
    assert names == {
        "get_ctrader_candles",
        "get_studio_recorded_evidence",
        "inspect_studio_saved_strategy",
        "list_studio_saved_strategies",
        "validate_studio_strategy_draft",
    }
    assert "request_draft_approval" not in names
    assert "consume_approved_once" not in names
