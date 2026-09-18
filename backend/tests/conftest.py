from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from backend import config as config_module
from backend.config import AppSettings

_IMPORT_DB_DIR = Path(tempfile.mkdtemp(prefix="tradeagent-tests-"))
config_module.SETTINGS = AppSettings(
    version=config_module.SETTINGS.version,
    db_path=_IMPORT_DB_DIR / "bootstrap.db",
)

from backend.storage import db as db_module  # noqa: E402
from backend.storage.db import init_db  # noqa: E402


def _reset_connection() -> None:
    conn = getattr(db_module._LOCAL, "connection", None)
    if conn is not None:
        conn.close()
        db_module._LOCAL.connection = None


def pytest_runtest_setup(item) -> None:
    _reset_connection()


def pytest_runtest_teardown(item, nextitem) -> None:
    _reset_connection()


def pytest_sessionfinish(session, exitstatus) -> None:
    _reset_connection()
    shutil.rmtree(_IMPORT_DB_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def isolated_v2_db(monkeypatch, tmp_path):
    """Keep test state outside synced/worktree directories."""
    _reset_connection()
    test_dir = tmp_path / "tradeagent"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_settings = AppSettings(version=db_module.SETTINGS.version, db_path=test_dir / "tradeagent_test.db")
    monkeypatch.setattr(db_module, "SETTINGS", test_settings)
    init_db()
    yield
    _reset_connection()
