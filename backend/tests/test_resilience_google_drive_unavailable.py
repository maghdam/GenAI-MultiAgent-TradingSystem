from __future__ import annotations

import asyncio
from pathlib import Path
import socket

from fastapi.testclient import TestClient

from backend import config as config_module
from backend.domain.models import EngineConfig
from backend.services.engine import V2Engine
from backend.storage.repositories import load_engine_config, load_runtime, save_engine_config


REPO_ROOT = Path(__file__).resolve().parents[2]
_DRIVE_HOST_MARKERS = (
    "drive.google.com",
    "www.googleapis.com",
    "oauth2.googleapis.com",
)


def _disable_external_boot_services(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_EVENT_INTELLIGENCE_ON_BOOT", "0")


def _block_google_drive_network(monkeypatch) -> list[str]:
    attempts: list[str] = []
    real_getaddrinfo = socket.getaddrinfo

    def _guard(host, *args, **kwargs):
        host_text = str(host or "").lower()
        if any(marker in host_text for marker in _DRIVE_HOST_MARKERS):
            attempts.append(host_text)
            raise OSError("simulated Google Drive outage")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", _guard)
    return attempts


def test_runtime_dependency_manifest_has_no_google_drive_sdk() -> None:
    manifest = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()

    forbidden = (
        "google-api-python-client",
        "googleapiclient",
        "pydrive",
        "pydrive2",
    )
    assert all(name not in manifest for name in forbidden)


def test_local_api_and_sqlite_remain_usable_when_google_drive_is_unreachable(monkeypatch) -> None:
    _disable_external_boot_services(monkeypatch)
    attempts = _block_google_drive_network(monkeypatch)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", r"Z:\unavailable-drive\credentials.json")
    monkeypatch.setenv("GOOGLE_DRIVE_ROOT", r"Z:\unavailable-drive")

    save_engine_config(
        EngineConfig(
            enabled=False,
            ctrader_autotrade=False,
                        kill_switch=True,
            operator_note="before simulated Drive outage",
        )
    )

    from backend.app import app

    with TestClient(app) as client:
        response = client.get("/api/config")
        assert response.status_code == 200

        updated = response.json()
        updated["operator_note"] = "local state updated while Drive unavailable"
        save_response = client.post("/api/config", json=updated)
        assert save_response.status_code == 200
        assert save_response.json()["operator_note"] == "local state updated while Drive unavailable"

        incidents_response = client.get("/api/incidents")
        assert incidents_response.status_code == 200
        assert isinstance(incidents_response.json(), list)

    persisted = load_engine_config(EngineConfig())
    assert persisted.operator_note == "local state updated while Drive unavailable"
    assert attempts == []


def test_drive_outage_does_not_interfere_with_live_capable_local_config(monkeypatch) -> None:
    _disable_external_boot_services(monkeypatch)
    attempts = _block_google_drive_network(monkeypatch)
    save_engine_config(
        EngineConfig(
            enabled=False,
            ctrader_autotrade=False,
            kill_switch=True,
        )
    )

    from backend.app import app

    with TestClient(app) as client:
        payload = client.get("/api/config").json()
        payload["ctrader_autotrade"] = True
        payload["selected_ctrader_account_id"] = 47139918
        payload["selected_ctrader_account_type"] = "live"
        response = client.post("/api/config", json=payload)

    assert response.status_code == 200
    persisted = load_engine_config(EngineConfig())
    assert persisted.ctrader_autotrade is True
    assert persisted.selected_ctrader_account_type == "live"
    assert attempts == []


def test_engine_kill_switch_cycle_persists_without_google_drive_access(monkeypatch) -> None:
    attempts = _block_google_drive_network(monkeypatch)
    save_engine_config(
        EngineConfig(
            enabled=True,
            ctrader_autotrade=False,
                        kill_switch=True,
        )
    )

    subject = V2Engine()
    summary = asyncio.run(subject.run_once())

    assert summary == "kill switch active"
    runtime = load_runtime()
    assert runtime.running is True
    assert runtime.loop_active is False
    assert runtime.last_cycle_summary == "kill switch active"
    assert attempts == []


def test_runtime_database_path_is_independent_of_google_drive_environment(monkeypatch, tmp_path) -> None:
    local_app_data = tmp_path / "LocalAppData"
    unavailable_drive = tmp_path / "MissingGoogleDrive"
    monkeypatch.delenv("TRADEAGENT_DB_PATH", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.setenv("GOOGLE_DRIVE_ROOT", str(unavailable_drive))
    monkeypatch.setenv("GOOGLE_DRIVE_PATH", str(unavailable_drive))

    resolved = config_module.resolve_db_path()

    assert resolved == (
        local_app_data / "TradeAgent" / "data" / "tradeagent.db"
    ).resolve()
    assert unavailable_drive.resolve() not in resolved.parents
