from __future__ import annotations

from fastapi.testclient import TestClient


def test_v2_calendar_next_endpoint_uses_service(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_LEGACY_CONTROLLER_ON_BOOT", "0")
    monkeypatch.setattr(
        "backend.api.router.get_next_event",
        lambda: {"ts": 1700000000.0, "title": "CPI", "impact": "high", "source": "test"},
    )
    from backend.app import app

    with TestClient(app) as client:
        response = client.get("/api/calendar/next")

    assert response.status_code == 200
    assert response.json()["title"] == "CPI"


