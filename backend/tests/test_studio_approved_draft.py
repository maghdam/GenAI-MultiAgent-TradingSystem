"""§34 offline exact-human-approval-bound draft commit, isolated from live storage."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.services import studio_draft_approval as approvals
from backend.services import studio_draft_create as drafts
from backend.storage.db import get_db


CODE = (
    "import pandas as pd\n\n"
    "def signals(df):\n"
    "    return pd.Series(0, index=df.index)\n"
)
OTHER = CODE.replace("Series(0,", "Series(1,")


@pytest.fixture
def generated_dir(tmp_path, monkeypatch):
    directory = tmp_path / "isolated_generated_strategies"
    directory.mkdir()
    monkeypatch.setattr(drafts, "STRATEGY_DIR", directory)
    return directory


def _approval_state(request_id):
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM studio_draft_approval_requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
    return dict(row) if row else None


def _lifecycle(name):
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM strategy_lifecycles WHERE strategy = ?",
            (name,),
        ).fetchone()
    return dict(row) if row else None


def _fake_operator(monkeypatch, typed):
    io = SimpleNamespace(
        stdin=SimpleNamespace(isatty=lambda: True),
        stdout=SimpleNamespace(isatty=lambda: True),
    )
    monkeypatch.setattr(approvals, "sys", io)
    monkeypatch.setattr("builtins.input", lambda prompt: typed)
    return io


def _approved(monkeypatch, name="new_research_strategy", source=CODE):
    request = approvals.request_draft_approval(name, source)
    _fake_operator(monkeypatch, name + " " + request["source_sha256"])
    result = approvals.approve_from_local_console(request["request_id"], source)
    assert result["status"] == "approved"
    return request, result


def test_pending_request_cannot_create_without_human_approval(generated_dir):
    proposal = approvals.request_draft_approval("new_research_strategy", CODE)
    assert _approval_state(proposal["request_id"])["state"] == "pending"
    with pytest.raises(drafts.DraftCreateError, match="operator_approval_not_available"):
        drafts.create_from_approved_local_request(
            proposal["request_id"], "new_research_strategy", CODE
        )
    assert list(generated_dir.iterdir()) == []
    assert _lifecycle("new_research_strategy") is None
    assert _approval_state(proposal["request_id"])["state"] == "pending"


def test_exact_approved_commit_creates_only_version_one_draft(monkeypatch, generated_dir):
    proposal, _ = _approved(monkeypatch)
    result = drafts.create_from_approved_local_request(
        proposal["request_id"], "new_research_strategy", CODE
    )
    assert result["saved"] is True
    assert result["strategy"] == "new_research_strategy"
    assert result["source_sha256"] == proposal["source_sha256"]
    assert result["lifecycle_version"] == 1
    assert result["stage"] == "draft"
    assert (generated_dir / "new_research_strategy.py").read_bytes() == CODE.encode()
    row = _lifecycle("new_research_strategy")
    assert row["version"] == 1
    assert row["stage"] == "draft"
    assert row["version_hash"] == proposal["source_sha256"]
    assert _approval_state(proposal["request_id"])["state"] == "consumed"
    with pytest.raises(drafts.DraftCreateError):
        drafts.create_from_approved_local_request(
            proposal["request_id"], "new_research_strategy", CODE
        )
    assert _lifecycle("new_research_strategy") == row
    assert list(generated_dir.iterdir()) == [generated_dir / "new_research_strategy.py"]


@pytest.mark.parametrize("strategy,source", [
    ("another_strategy", CODE),
    ("new_research_strategy", OTHER),
], ids=["different-name", "different-source"])
def test_wrong_name_or_hash_never_creates_and_approval_survives(
    monkeypatch, generated_dir, strategy, source
):
    proposal, _ = _approved(monkeypatch)
    with pytest.raises(drafts.DraftCreateError, match="operator_approval_not_available"):
        drafts.create_from_approved_local_request(proposal["request_id"], strategy, source)
    assert _approval_state(proposal["request_id"])["state"] == "approved"
    assert list(generated_dir.iterdir()) == []
    assert _lifecycle(strategy) is None


def test_consumed_standalone_approval_not_reusable_for_create(monkeypatch, generated_dir):
    proposal, _ = _approved(monkeypatch)
    approvals.consume_approved_once(
        proposal["request_id"], "new_research_strategy", CODE
    )
    with pytest.raises(drafts.DraftCreateError, match="operator_approval_not_available"):
        drafts.create_from_approved_local_request(
            proposal["request_id"], "new_research_strategy", CODE
        )
    assert _approval_state(proposal["request_id"])["state"] == "consumed"
    assert not list(generated_dir.iterdir())


def test_expired_approval_never_creates(monkeypatch, generated_dir):
    proposal, grant = _approved(monkeypatch)
    monkeypatch.setattr(approvals, "_now", lambda: grant["expires_at_epoch"])
    with pytest.raises(drafts.DraftCreateError, match="operator_approval_not_available"):
        drafts.create_from_approved_local_request(
            proposal["request_id"], "new_research_strategy", CODE
        )
    assert _approval_state(proposal["request_id"])["state"] == "approved"
    assert not list(generated_dir.iterdir())


def test_existing_file_preserved_and_approval_rolls_back(monkeypatch, generated_dir):
    proposal, _ = _approved(monkeypatch)
    file = generated_dir / "new_research_strategy.py"
    file.write_bytes(b"EXISTING PRIVATE CODE\n")
    with pytest.raises(drafts.DraftCreateError, match="strategy_name_already_exists"):
        drafts.create_from_approved_local_request(
            proposal["request_id"], "new_research_strategy", CODE
        )
    assert file.read_bytes() == b"EXISTING PRIVATE CODE\n"
    assert _lifecycle("new_research_strategy") is None
    assert _approval_state(proposal["request_id"])["state"] == "approved"


def test_old_lifecycle_blocks_name_even_if_file_removed(monkeypatch, generated_dir):
    proposal, _ = _approved(monkeypatch)
    with get_db() as db:
        db.execute(
            """INSERT INTO strategy_lifecycles(
               strategy, version, version_hash, stage, hypothesis,
               gates_json, created_at, updated_at
            ) VALUES (?, 1, ?, 'draft', '', '{}', '2026-01-01', '2026-01-01')""",
            ("new_research_strategy", proposal["source_sha256"]),
        )
        db.commit()
    before = _lifecycle("new_research_strategy")
    with pytest.raises(drafts.DraftCreateError, match="strategy_lifecycle_already_exists"):
        drafts.create_from_approved_local_request(
            proposal["request_id"], "new_research_strategy", CODE
        )
    assert before == _lifecycle("new_research_strategy")
    assert _approval_state(proposal["request_id"])["state"] == "approved"
    assert not list(generated_dir.iterdir())


def test_owned_file_write_failure_rolls_back_approval_and_lifecycle(
    monkeypatch, generated_dir
):
    proposal, _ = _approved(monkeypatch)
    monkeypatch.setattr(drafts.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("disk I/O")))
    with pytest.raises(drafts.DraftCreateError, match="draft_create_failed"):
        drafts.create_from_approved_local_request(
            proposal["request_id"], "new_research_strategy", CODE
        )
    assert not list(generated_dir.iterdir())
    assert _approval_state(proposal["request_id"])["state"] == "approved"
    assert _lifecycle("new_research_strategy") is None


def test_approval_expiring_after_file_write_rolls_back_atomically(
    monkeypatch, generated_dir
):
    proposal, grant = _approved(monkeypatch)
    times = iter([grant["expires_at_epoch"] - 1, grant["expires_at_epoch"]])
    monkeypatch.setattr(approvals, "_now", lambda: next(times))
    with pytest.raises(drafts.DraftCreateError, match="operator_approval_not_available"):
        drafts.create_from_approved_local_request(
            proposal["request_id"], "new_research_strategy", CODE
        )
    assert not list(generated_dir.iterdir())
    assert _approval_state(proposal["request_id"])["state"] == "approved"
    assert _lifecycle("new_research_strategy") is None


def test_commit_result_ambiguous_never_blindly_removes_file(monkeypatch, generated_dir):
    proposal, _ = _approved(monkeypatch)
    actual_db = drafts.get_db

    @contextmanager
    def uncertain_db():
        with actual_db() as conn:
            class Uncertain:
                def execute(self, sql, *args):
                    return conn.execute(sql, *args)

                def rollback(self):
                    return conn.rollback()

                def commit(self):
                    conn.commit()
                    raise RuntimeError("unknown acknowledgment after durable commit")

            yield Uncertain()

    monkeypatch.setattr(drafts, "get_db", uncertain_db)
    with pytest.raises(drafts.DraftCreateError, match="manual_reconciliation_required"):
        drafts.create_from_approved_local_request(
            proposal["request_id"], "new_research_strategy", CODE
        )
    assert (generated_dir / "new_research_strategy.py").read_bytes() == CODE.encode()
    assert _approval_state(proposal["request_id"])["state"] == "consumed"
    assert _lifecycle("new_research_strategy") is not None
    with pytest.raises(drafts.DraftCreateError):
        drafts.create_from_approved_local_request(
            proposal["request_id"], "new_research_strategy", CODE
        )


def _thread_attempt(request_id):
    from backend.storage import db as storage

    try:
        try:
            drafts.create_from_approved_local_request(
                request_id, "new_research_strategy", CODE
            )
            return "saved"
        except drafts.DraftCreateError:
            return "rejected"
    finally:
        # Thread-local SQLite handles must not hold the Windows temp DB open.
        conn = getattr(storage._LOCAL, "connection", None)
        if conn is not None:
            conn.close()
            storage._LOCAL.connection = None


def test_concurrent_create_single_winner_for_approved_request(
    monkeypatch, generated_dir
):
    proposal, _ = _approved(monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            _thread_attempt, [proposal["request_id"], proposal["request_id"]]
        ))
    assert sorted(results) == ["rejected", "saved"]
    assert _approval_state(proposal["request_id"])["state"] == "consumed"
    assert (generated_dir / "new_research_strategy.py").read_bytes() == CODE.encode()
    assert _lifecycle("new_research_strategy")["stage"] == "draft"


def test_fake_canonical_candidate_rejected_before_database_or_file(
    monkeypatch, generated_dir
):
    proposal, _ = _approved(monkeypatch)
    candidate = drafts.DraftCandidate(
        name="new_research_strategy",
        canonical_bytes=CODE.replace("Series(0,", "Series(1,").encode(),
        source_sha256=proposal["source_sha256"],
        source_lines=4,
    )
    with pytest.raises(drafts.DraftCreateError, match="invalid_source"):
        drafts._create_exclusively(candidate, approved_request_id=proposal["request_id"])
    assert not list(generated_dir.iterdir())
    assert _approval_state(proposal["request_id"])["state"] == "approved"


def test_cli_propose_requires_interactive_console(monkeypatch, generated_dir, tmp_path, capsys):
    from scripts import studio_approved_draft as cli

    source_path = tmp_path / "reviewed.py"
    source_path.write_text(CODE, encoding="utf-8")
    monkeypatch.setattr(
        cli, "sys",
        SimpleNamespace(
            stdin=SimpleNamespace(isatty=lambda: False),
            stdout=SimpleNamespace(isatty=lambda: True),
        ),
    )
    assert cli.main([
        "propose", "--name", "new_research_strategy",
        "--source-file", str(source_path),
    ]) == 2
    assert "interactive_operator_required" in capsys.readouterr().out
    assert not list(generated_dir.iterdir())


def test_cli_operator_propose_approve_commit_isolated_end_to_end(
    monkeypatch, generated_dir, tmp_path, capsys
):
    from scripts import studio_approved_draft as cli

    source_path = tmp_path / "reviewed.py"
    source_path.write_text(CODE, encoding="utf-8")
    io = _fake_operator(monkeypatch, "")
    monkeypatch.setattr(cli, "sys", io)

    assert cli.main([
        "propose", "--name", "new_research_strategy",
        "--source-file", str(source_path),
    ]) == 0
    proposal_text = capsys.readouterr().out
    proposal = json.loads(proposal_text)
    assert proposal["status"] == "pending"

    _fake_operator(
        monkeypatch, "new_research_strategy " + proposal["source_sha256"]
    )
    approved = approvals.approve_from_local_console(
        proposal["request_id"], CODE
    )
    assert approved["status"] == "approved"
    capsys.readouterr()

    assert cli.main([
        "commit", "--request-id", proposal["request_id"],
        "--name", "new_research_strategy", "--source-file", str(source_path),
    ]) == 0
    receipt = capsys.readouterr().out
    result = json.loads(receipt)
    assert result["saved"] is True
    assert result["stage"] == "draft"
    assert CODE not in receipt
    assert str(source_path) not in receipt
    assert (generated_dir / "new_research_strategy.py").read_bytes() == CODE.encode()
    assert _approval_state(proposal["request_id"])["state"] == "consumed"


def test_read_gateway_remains_exactly_five_tools_and_does_not_expose_approval():
    from backend.chatgpt_mcp_readonly import create_server

    tools = {tool.name for tool in asyncio.run(create_server().list_tools())}
    assert tools == {
        "get_ctrader_candles",
        "get_studio_recorded_evidence",
        "inspect_studio_saved_strategy",
        "list_studio_saved_strategies",
        "validate_studio_strategy_draft",
    }
    assert "request_draft_approval" not in tools
    assert "create_from_approved_local_request" not in tools
