from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from backend.domain.models import EngineConfig
from backend.services.engine import V2Engine
from backend.storage.repositories import load_engine_config, save_engine_config


def _disable_boot_services(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_LEGACY_CONTROLLER_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_EVENT_INTELLIGENCE_ON_BOOT", "0")


def test_engine_restart_reuses_normal_stop_start_sequence() -> None:
    subject = V2Engine()
    calls: list[str] = []

    async def fake_stop() -> None:
        calls.append("stop")

    async def fake_start() -> None:
        calls.append("start")

    subject.stop = fake_stop  # type: ignore[method-assign]
    subject.start = fake_start  # type: ignore[method-assign]

    asyncio.run(subject.restart())

    assert calls == ["stop", "start"]


def test_system_engine_lifecycle_endpoints_match_operator_controls(monkeypatch) -> None:
    _disable_boot_services(monkeypatch)
    save_engine_config(EngineConfig(enabled=False, ctrader_autotrade=False))

    wake_calls: list[str] = []
    restart_calls: list[str] = []
    scan_calls: list[str] = []

    async def fake_restart() -> None:
        restart_calls.append("restart")

    async def fake_run_once() -> str:
        scan_calls.append("scan")
        return "processed=1 actions=0 watchlist=1 market_skips=0"

    monkeypatch.setattr("backend.api.router.engine.wake", lambda: wake_calls.append("wake"))
    monkeypatch.setattr("backend.api.router.engine.restart", fake_restart)
    monkeypatch.setattr("backend.api.router.engine.run_once", fake_run_once)
    monkeypatch.setattr(
        "backend.api.router.recover_runtime_state",
        lambda config: {
            "enabled": config.enabled,
            "active_watchlist": ["XAUUSD:M5"],
            "running": config.enabled,
        },
    )
    monkeypatch.setattr(
        "backend.api.router.reconcile_open_positions",
        lambda reason: {
            "checked": 1,
            "closed": 0,
            "skipped": 0,
            "reason": reason,
        },
    )

    from backend.app import app

    with TestClient(app) as client:
        start_response = client.post("/api/engine/start")
        assert start_response.status_code == 200
        assert start_response.json() == {"ok": True, "enabled": True}
        assert load_engine_config(EngineConfig()).enabled is True

        stop_response = client.post("/api/engine/stop")
        assert stop_response.status_code == 200
        assert stop_response.json() == {"ok": True, "enabled": False}
        assert load_engine_config(EngineConfig()).enabled is False

        restart_response = client.post("/api/engine/restart")
        assert restart_response.status_code == 200
        assert restart_response.json() == {"ok": True, "enabled": False, "restarted": True}

        scan_response = client.post("/api/engine/scan")
        assert scan_response.status_code == 200
        assert scan_response.json() == {
            "ok": True,
            "summary": "processed=1 actions=0 watchlist=1 market_skips=0",
        }

        recover_response = client.post("/api/engine/recover")
        assert recover_response.status_code == 200
        assert recover_response.json() == {
            "ok": True,
            "enabled": False,
            "active_watchlist": ["XAUUSD:M5"],
            "running": False,
        }

        reconcile_response = client.post("/api/engine/reconcile")
        assert reconcile_response.status_code == 200
        assert reconcile_response.json() == {
            "ok": True,
            "checked": 1,
            "closed": 0,
            "skipped": 0,
            "reason": "manual",
            "broker_history_recovery": {
                "checked": 0,
                "recovered": 0,
                "already_tracked": 0,
                "still_open": 0,
                "external_ignored": 0,
                "ready": False,
            },
            "closed_history": {
                "checked": 0,
                "reconciled": 0,
                "missing_broker_id": 0,
                "unavailable": 0,
            },
        }

    assert wake_calls == ["wake", "wake"]
    assert restart_calls == ["restart"]
    assert scan_calls == ["scan"]
