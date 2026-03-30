from __future__ import annotations

import asyncio
from fastapi.testclient import TestClient

from backend.services import model_service


def test_model_service_status_payload_success(monkeypatch) -> None:
    async def _fake_fetch_tags(timeout: float = 10.0):
        return {
            "ok": True,
            "status_code": 200,
            "models": ["phi3:mini", "llama3.2:3b-instruct-fp16"],
            "error": None,
        }

    monkeypatch.setattr(model_service, "fetch_tags", _fake_fetch_tags)

    payload = asyncio.run(model_service.status_payload())

    assert payload["ollama"] == 200
    assert payload["reachable"] is True
    assert payload["model"] == model_service.default_model()


def test_app_llm_status_endpoint_uses_model_service(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_LEGACY_CONTROLLER_ON_BOOT", "0")

    async def _fake_fetch_tags(timeout: float = 10.0):
        return {
            "ok": False,
            "status_code": 503,
            "models": [],
            "error": "ollama /api/tags -> 503",
        }

    monkeypatch.setattr(model_service, "fetch_tags", _fake_fetch_tags)
    from backend.app import app

    with TestClient(app) as client:
        response = client.get("/api/llm_status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ollama"] == 503
    assert payload["reachable"] is False
    assert "error" in payload
