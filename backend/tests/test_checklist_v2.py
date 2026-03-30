from __future__ import annotations

from fastapi.testclient import TestClient


def test_v2_checklist_auto_endpoint_uses_service(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_LEGACY_CONTROLLER_ON_BOOT", "0")
    monkeypatch.setattr(
        "backend.api.router.get_auto_checklist",
        lambda tf, structure_tf: {"ts": 123.0, "scenario": "A", "correlation": "normal", "components": {}},
    )
    from backend.app import app

    with TestClient(app) as client:
        response = client.get("/api/checklist/auto?tf=M15&structure_tf=H1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["scenario"] == "A"
    assert payload["correlation"] == "normal"


def test_v2_calendar_next_endpoint_uses_service(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_LEGACY_CONTROLLER_ON_BOOT", "0")
    monkeypatch.setattr(
        "backend.api.router.get_calendar_next",
        lambda: {"ts": 1700000000.0, "title": "CPI", "impact": "high", "source": "test"},
    )
    from backend.app import app

    with TestClient(app) as client:
        response = client.get("/api/calendar/next")

    assert response.status_code == 200
    assert response.json()["title"] == "CPI"


def test_v2_checklist_service_validates_timeframes() -> None:
    from fastapi import HTTPException

    from backend.services.checklist_feed import get_auto_checklist

    try:
        get_auto_checklist(tf="INVALID", structure_tf="H1")
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "Invalid timeframe" in exc.detail
    else:
        raise AssertionError("Expected HTTPException for invalid timeframe")
