from __future__ import annotations

from pathlib import Path
import subprocess

from backend import config as config_module
from backend.domain.models import EngineConfig, WatchlistItem
from backend.storage import db as db_module
from backend.storage.repositories import load_engine_config, save_engine_config


REPO_ROOT = Path(__file__).resolve().parents[2]


def _close_storage_connection() -> None:
    conn = getattr(db_module._LOCAL, "connection", None)
    if conn is not None:
        conn.close()
        db_module._LOCAL.connection = None


def test_settings_and_watchlist_survive_storage_restart() -> None:
    original = EngineConfig(
        enabled=True,
        selected_ctrader_account_id=2123962,
        selected_ctrader_account_type="live",
        paper_autotrade=True,
        ctrader_autotrade=False,
        kill_switch=False,
        min_confidence=0.73,
        daily_loss_limit_pct=1.7,
        max_daily_trades=9,
        max_open_positions=4,
        max_positions_per_symbol=2,
        cooldown_minutes=17,
        session_filter_enabled=True,
        session_start_hour_utc=7,
        session_end_hour_utc=19,
        require_stops=True,
        operator_note="persistence acceptance",
        watchlist=[
            WatchlistItem(
                symbol="XAUUSD",
                timeframe="M5",
                strategy="sma_cross",
                enabled=True,
                trading_enabled=True,
                lot_size=0.12,
                params={"fast": 8, "slow": 21},
            ),
            WatchlistItem(
                symbol="EURUSD",
                timeframe="H1",
                strategy="rsi_reversal",
                enabled=False,
                trading_enabled=False,
                lot_size=0.2,
                params={"period": 14},
            ),
        ],
    )

    save_engine_config(original)
    _close_storage_connection()

    loaded = load_engine_config(EngineConfig())

    assert loaded.model_dump() == original.model_dump()
    assert [item.model_dump() for item in loaded.watchlist] == [
        item.model_dump() for item in original.watchlist
    ]


def test_runtime_db_override_takes_precedence(monkeypatch, tmp_path) -> None:
    target = tmp_path / "runtime" / "tradeagent.db"
    monkeypatch.setenv("TRADEAGENT_DB_PATH", str(target))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "LocalAppData"))

    assert config_module.resolve_db_path() == target.resolve()


def test_windows_default_runtime_db_uses_local_appdata(monkeypatch, tmp_path) -> None:
    local_app_data = tmp_path / "LocalAppData"
    monkeypatch.delenv("TRADEAGENT_DB_PATH", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))

    resolved = config_module.resolve_db_path()

    assert resolved == (
        local_app_data / "TradeAgent" / "data" / "tradeagent.db"
    ).resolve()
    assert config_module.BASE_DIR not in resolved.parents


def test_gitignore_covers_sqlite_runtime_state() -> None:
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "backend/data/*.db" in ignored
    assert "backend/data/*.db-wal" in ignored
    assert "backend/data/*.db-shm" in ignored
    assert "backend/data/*.db-journal" in ignored


def test_git_tracks_no_sqlite_runtime_state() -> None:
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    sqlite_state = [
        path
        for path in tracked
        if path.lower().endswith((".db", ".db-wal", ".db-shm", ".db-journal"))
    ]

    assert sqlite_state == []


def test_legacy_runtime_db_is_migrated_once(monkeypatch, tmp_path) -> None:
    import sqlite3

    legacy = tmp_path / "repo" / "backend" / "data" / "tradeagent.db"
    target = tmp_path / "local" / "TradeAgent" / "data" / "tradeagent.db"
    legacy.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(legacy) as db:
        db.execute(
            "CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        db.execute(
            "INSERT INTO config(key, value, updated_at) VALUES(?, ?, ?)",
            ("engine_config", '{"operator_note":"legacy-state"}', "2026-09-27T18:00:00"),
        )
        db.commit()

    _close_storage_connection()
    monkeypatch.setattr(db_module, "LEGACY_DB_PATH", legacy)
    monkeypatch.setattr(db_module, "DB_PATH", target)
    monkeypatch.setattr(
        db_module,
        "SETTINGS",
        config_module.AppSettings(version="1.0.0", db_path=target),
    )

    with db_module.get_db() as db:
        row = db.execute(
            "SELECT value FROM config WHERE key = ?",
            ("engine_config",),
        ).fetchone()

    assert target.exists()
    assert row is not None
    assert "legacy-state" in row["value"]
