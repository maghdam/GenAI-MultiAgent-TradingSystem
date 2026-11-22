"""Create-only local-operator draft writer: no MCP, LLM, broker or backtest."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.services import studio_draft_create as drafts
from backend.services.strategy_lifecycle import source_hash
from backend.storage.db import get_db


CODE = (
    "import pandas as pd\n\n"
    "def signals(df):\n"
    "    return pd.Series(0, index=df.index)\n"
)


@pytest.fixture
def local_drafts(monkeypatch, tmp_path):
    directory = tmp_path / "strategies_generated"
    directory.mkdir()
    monkeypatch.setattr(drafts, "STRATEGY_DIR", directory)
    return directory


def candidate(name="new_research_draft", code=CODE):
    return drafts.prepare_candidate(name, code)


def _row(name):
    with get_db() as db:
        return db.execute(
            "SELECT * FROM strategy_lifecycles WHERE strategy = ?",
            (name,),
        ).fetchone()


def _fake_console(monkeypatch, answer):
    # Patch only the imported module's sys reference; do not change pytest's I/O.
    monkeypatch.setattr(
        drafts,
        "sys",
        SimpleNamespace(
            stdin=SimpleNamespace(isatty=lambda: True),
            stdout=SimpleNamespace(isatty=lambda: True),
        ),
    )
    monkeypatch.setattr("builtins.input", lambda prompt: answer)


def test_canonical_source_policy_hash_and_bounded_metadata():
    result = candidate(code="\r\n" + CODE.replace("\n", "\r\n") + "\r\n")
    assert result.name == "new_research_draft"
    assert result.source_sha256 == hashlib.sha256(CODE.encode()).hexdigest()
    assert result.source_sha256 == source_hash(CODE)
    assert result.canonical_bytes == CODE.encode()
    assert result.source_lines == 4


@pytest.mark.parametrize("bad", ["", "A", "name.py", "../file", "a/b", "a\\b", "x" * 65, 4, None])
def test_invalid_name_never_creates_files_or_lifecycles(local_drafts, bad):
    with pytest.raises(drafts.DraftCreateError, match="invalid_strategy_name"):
        drafts.prepare_candidate(bad, CODE)
    assert list(local_drafts.iterdir()) == []


@pytest.mark.parametrize("bad", [
    None, 42, "", "   \n",
    "x = 1\n",
    "def signals(df):\n    open('private.txt')\n",
    "import os\ndef signals(df):\n    return 0\n",
    "async def signals(df):\n    return 0\n",
    ("# comment\n" * 401) + CODE,
    "x = '" + ("a" * 16_384) + "'\n" + CODE,
    "def signals(df):\n    return 0\0\n",
    "def signals(df):\n    return '" + chr(0xD800) + "'\n",
])
def test_invalid_source_has_no_side_effect(local_drafts, bad):
    with pytest.raises(drafts.DraftCreateError):
        drafts.prepare_candidate("safe_name", bad)
    assert list(local_drafts.iterdir()) == []


def test_missing_tty_denies_before_any_write(monkeypatch, local_drafts):
    monkeypatch.setattr(
        drafts, "sys",
        SimpleNamespace(
            stdin=SimpleNamespace(isatty=lambda: False),
            stdout=SimpleNamespace(isatty=lambda: True),
        ),
    )
    with pytest.raises(drafts.DraftCreateError, match="interactive_console_required"):
        drafts.create_from_interactive_console(candidate())
    assert list(local_drafts.iterdir()) == []
    assert _row("new_research_draft") is None


def test_wrong_operator_hash_denies_without_write(monkeypatch, local_drafts):
    c = candidate()
    _fake_console(monkeypatch, c.name + " " + "0" * 64)
    with pytest.raises(drafts.DraftCreateError, match="operator_approval_denied"):
        drafts.create_from_interactive_console(c)
    assert list(local_drafts.iterdir()) == []
    assert _row(c.name) is None


def test_expired_operator_approval_denies_without_write(monkeypatch, local_drafts):
    c = candidate()
    _fake_console(monkeypatch, c.name + " " + c.source_sha256)
    timestamps = iter([0.0, float(drafts.APPROVAL_SECONDS + 1)])
    monkeypatch.setattr(drafts.time, "monotonic", lambda: next(timestamps))
    with pytest.raises(drafts.DraftCreateError, match="operator_approval_expired"):
        drafts.create_from_interactive_console(c)
    assert list(local_drafts.iterdir()) == []
    assert _row(c.name) is None


def test_operator_exact_name_and_sha_creates_one_draft_only(monkeypatch, local_drafts):
    c = candidate()
    _fake_console(monkeypatch, c.name + " " + c.source_sha256)
    result = drafts.create_from_interactive_console(c)
    assert result == {
        "strategy": c.name,
        "source_sha256": c.source_sha256,
        "lifecycle_version": 1,
        "stage": "draft",
        "saved": True,
        "warning": "Draft saved only. No source execution, backtest, broker connection or trade.",
    }
    assert (local_drafts / (c.name + ".py")).read_bytes() == CODE.encode()
    record = _row(c.name)
    assert record["version"] == 1
    assert record["stage"] == "draft"
    assert record["version_hash"] == c.source_sha256
    assert json.loads(record["gates_json"]) == drafts.DEFAULT_GATES
    assert list(local_drafts.iterdir()) == [local_drafts / (c.name + ".py")]


def test_existing_file_never_overwritten(local_drafts):
    c = candidate()
    dest = local_drafts / (c.name + ".py")
    dest.write_bytes(b"EXISTING PRIVATE STRATEGY\n")
    with pytest.raises(drafts.DraftCreateError, match="strategy_name_already_exists"):
        drafts._create_exclusively(c)
    assert dest.read_bytes() == b"EXISTING PRIVATE STRATEGY\n"
    assert _row(c.name) is None


def test_existing_lifecycle_never_modified_even_without_file(local_drafts):
    c = candidate()
    with get_db() as db:
        db.execute(
            """INSERT INTO strategy_lifecycles(
                strategy, version, version_hash, stage, hypothesis,
                gates_json, created_at, updated_at
            ) VALUES (?, 1, ?, 'draft', '', '{}', '2026-01-01', '2026-01-01')""",
            (c.name, c.source_sha256),
        )
        db.commit()
    before = dict(_row(c.name))
    with pytest.raises(drafts.DraftCreateError, match="strategy_lifecycle_already_exists"):
        drafts._create_exclusively(c)
    assert dict(_row(c.name)) == before
    assert list(local_drafts.iterdir()) == []


def test_second_attempt_cannot_overwrite_or_create_version_two(local_drafts):
    c = candidate()
    drafts._create_exclusively(c)
    with pytest.raises(drafts.DraftCreateError, match="strategy_lifecycle_already_exists"):
        drafts._create_exclusively(c)
    assert (local_drafts / (c.name + ".py")).read_bytes() == CODE.encode()
    assert _row(c.name)["version"] == 1


def test_two_simultaneous_create_attempts_have_single_winner(local_drafts):
    c = candidate()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: _try_commit(c), range(2)))
    assert sorted(results) == ["created", "rejected"]
    assert (local_drafts / (c.name + ".py")).read_bytes() == CODE.encode()
    with get_db() as db:
        row = db.execute(
            "SELECT COUNT(*) AS n FROM strategy_lifecycles WHERE strategy=?",
            (c.name,),
        ).fetchone()
    assert row["n"] == 1


def _try_commit(c):
    from backend.storage import db as db_module

    try:
        try:
            drafts._create_exclusively(c)
            return "created"
        except drafts.DraftCreateError:
            return "rejected"
    finally:
        # Worker-thread SQLite connections must not hold a Windows test DB open.
        conn = getattr(db_module._LOCAL, "connection", None)
        if conn is not None:
            conn.close()
            db_module._LOCAL.connection = None


def test_database_insert_failure_rolls_back_and_removes_owned_file(monkeypatch, local_drafts):
    real_db = drafts.get_db

    @contextmanager
    def broken_db():
        with real_db() as conn:
            class Broken:
                def execute(self, query, *args):
                    if query.strip().startswith("INSERT INTO strategy_lifecycles"):
                        raise sqlite3.IntegrityError("injected DB failure")
                    return conn.execute(query, *args)

                def rollback(self):
                    return conn.rollback()

                def commit(self):
                    return conn.commit()

            yield Broken()

    monkeypatch.setattr(drafts, "get_db", broken_db)
    with pytest.raises(drafts.DraftCreateError, match="strategy_lifecycle_already_exists"):
        drafts._create_exclusively(candidate())
    assert list(local_drafts.iterdir()) == []
    assert _row("new_research_draft") is None


def test_file_write_failure_removes_partial_owned_file(monkeypatch, local_drafts):
    def fail_fsync(fd):
        raise OSError("disk error with sensitive absolute path")

    monkeypatch.setattr(drafts.os, "fsync", fail_fsync)
    with pytest.raises(drafts.DraftCreateError, match="draft_create_failed"):
        drafts._create_exclusively(candidate())
    assert list(local_drafts.iterdir()) == []
    assert _row("new_research_draft") is None


def test_ambiguous_commit_preserves_file_for_manual_reconciliation(monkeypatch, local_drafts):
    real_db = drafts.get_db

    @contextmanager
    def uncertain_db():
        with real_db() as conn:
            class Uncertain:
                def execute(self, query, *args):
                    return conn.execute(query, *args)

                def rollback(self):
                    return conn.rollback()

                def commit(self):
                    conn.commit()
                    raise RuntimeError("ambiguous commit acknowledgment")

            yield Uncertain()

    monkeypatch.setattr(drafts, "get_db", uncertain_db)
    with pytest.raises(drafts.DraftCreateError, match="manual_reconciliation_required"):
        drafts._create_exclusively(candidate())
    assert (local_drafts / "new_research_draft.py").read_bytes() == CODE.encode()
    assert _row("new_research_draft") is not None
    with pytest.raises(drafts.DraftCreateError):
        drafts._create_exclusively(candidate())


def test_missing_directory_fails_closed_and_does_not_create(local_drafts):
    local_drafts.rmdir()
    with pytest.raises(drafts.DraftCreateError, match="strategy_directory_unavailable"):
        drafts._create_exclusively(candidate())
    assert not local_drafts.exists()


def test_cli_refuses_automation_without_tty(monkeypatch, tmp_path, local_drafts, capsys):
    from scripts import approve_studio_draft as cli

    source = tmp_path / "candidate.py"
    source.write_text(CODE, encoding="utf-8")
    monkeypatch.setattr(
        cli, "sys",
        SimpleNamespace(
            stdin=SimpleNamespace(isatty=lambda: False),
            stdout=SimpleNamespace(isatty=lambda: True),
        ),
    )
    assert cli.main(["--name", "new_research_draft", "--source-file", str(source)]) == 2
    assert "interactive_console_required" in capsys.readouterr().out
    assert not list(local_drafts.iterdir())


def test_no_gateway_tool_or_generic_studio_route_is_added():
    from backend.chatgpt_mcp_readonly import create_server
    import asyncio

    names = {item.name for item in asyncio.run(create_server().list_tools())}
    assert names == {
        "get_ctrader_candles",
        "list_studio_saved_strategies",
        "get_studio_recorded_evidence",
        "inspect_studio_saved_strategy",
        "validate_studio_strategy_draft",
    }
    assert "create_studio_draft" not in names
