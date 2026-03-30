from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from backend.config import AppSettings
from backend.storage import db as db_module
from backend.storage.db import init_db


def _reset_connection() -> None:
    conn = getattr(db_module._LOCAL, "connection", None)
    if conn is not None:
        conn.close()
        db_module._LOCAL.connection = None


def pytest_runtest_setup(item) -> None:
    _reset_connection()


def pytest_runtest_teardown(item, nextitem) -> None:
    _reset_connection()


@pytest.fixture(autouse=True)
def isolated_v2_db(monkeypatch):
    _reset_connection()
    test_root = Path(__file__).resolve().parent / "_tmp"
    test_root.mkdir(parents=True, exist_ok=True)
    test_dir = test_root / uuid4().hex
    test_dir.mkdir(parents=True, exist_ok=True)
    test_settings = AppSettings(version=db_module.SETTINGS.version, db_path=test_dir / "tradeagent_test.db")
    monkeypatch.setattr(db_module, "SETTINGS", test_settings)
    init_db()
    yield
    _reset_connection()
    shutil.rmtree(test_dir, ignore_errors=True)
