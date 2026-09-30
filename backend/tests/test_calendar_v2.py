from __future__ import annotations

import time

from fastapi.testclient import TestClient

from backend.calendar import get_next_event


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

def test_calendar_rejects_past_env_event(monkeypatch) -> None:
    monkeypatch.setenv("CALENDAR_NEXT_TS", str(time.time() - 60))
    monkeypatch.setenv("CALENDAR_NEXT_TITLE", "Expired CPI")
    monkeypatch.setenv("CALENDAR_NEXT_IMPACT", "high")

    assert get_next_event() is None


def test_calendar_accepts_future_env_event(monkeypatch) -> None:
    future = time.time() + 3600
    monkeypatch.setenv("CALENDAR_NEXT_TS", str(future))
    monkeypatch.setenv("CALENDAR_NEXT_TITLE", "Upcoming CPI")
    monkeypatch.setenv("CALENDAR_NEXT_IMPACT", "high")

    event = get_next_event()

    assert event is not None
    assert event["title"] == "Upcoming CPI"
    assert event["impact"] == "high"
    assert event["source"] == "env"
    assert event["ts"] > time.time()

