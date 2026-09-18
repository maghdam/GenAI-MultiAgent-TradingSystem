from __future__ import annotations

import pytest

from backend import config as config_module
from backend.storage.db import get_db


def test_resolve_db_path_honors_runtime_override(monkeypatch, tmp_path) -> None:
    target = tmp_path / "runtime" / "tradeagent.db"
    monkeypatch.setenv("TRADEAGENT_DB_PATH", str(target))

    assert config_module.resolve_db_path() == target.resolve()


def test_resolve_db_path_falls_back_to_worktree_data(monkeypatch) -> None:
    monkeypatch.delenv("TRADEAGENT_DB_PATH", raising=False)

    assert config_module.resolve_db_path() == config_module.DATA_DIR / "tradeagent.db"


def test_runtime_connection_uses_wal_and_busy_timeout() -> None:
    with get_db() as db:
        journal_mode = str(db.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        busy_timeout = int(db.execute("PRAGMA busy_timeout").fetchone()[0])

    assert journal_mode == "wal"
    assert busy_timeout == 5000


def test_get_db_rolls_back_failed_transaction() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with get_db() as db:
            db.execute(
                "INSERT INTO config(key, value, updated_at) VALUES(?, ?, ?)",
                ("rollback-test", "value", "2026-09-18T20:00:00"),
            )
            raise RuntimeError("boom")

    with get_db() as db:
        row = db.execute("SELECT key FROM config WHERE key = ?", ("rollback-test",)).fetchone()

    assert row is None
