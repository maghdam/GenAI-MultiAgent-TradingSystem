from __future__ import annotations

import asyncio

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.api import router as router_module
from backend.domain.models import BrokerStatus, EngineConfig, EngineRuntime, StudioTaskRequest
from backend.services import model_service, studio_llm, studio_tasks
from backend.storage.repositories import list_order_intents


def _broker_ready() -> BrokerStatus:
    return BrokerStatus(
        connected=True,
        socket_connected=True,
        account_authorized=True,
        symbols_loaded=250,
        open_positions=0,
        pending_orders=0,
        ready=True,
        market_data_ready=True,
        broker_mode="demo",
        account_type="demo",
        demo_account_confirmed=True,
        execution_ready=True,
        notes=[],
    )


def test_model_failure_cache_rechecks_after_five_seconds_and_recovers(monkeypatch) -> None:
    clock = {"now": 104.9}
    failure = {
        "ok": False,
        "status_code": None,
        "models": [],
        "error": "connection refused",
    }

    monkeypatch.setattr(model_service, "_CACHED_TAGS", failure)
    monkeypatch.setattr(model_service, "_LAST_FETCH_TS", 100.0)
    monkeypatch.setattr(model_service.time, "time", lambda: clock["now"])

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "application/json"}

        def json(self):
            return {"models": [{"name": model_service.default_model()}]}

    calls: list[str] = []

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url: str):
            calls.append(url)
            return FakeResponse()

    monkeypatch.setattr(model_service.httpx, "AsyncClient", FakeAsyncClient)

    still_cached = asyncio.run(model_service.fetch_tags())
    assert still_cached["ok"] is False
    assert calls == []

    clock["now"] = 105.1
    recovered = asyncio.run(model_service.fetch_tags())

    assert recovered["ok"] is True
    assert recovered["models"] == [model_service.default_model()]
    assert len(calls) == 1


def test_studio_ollama_unreachable_fails_before_any_generation_attempt(monkeypatch) -> None:
    async def unavailable_tags(timeout: float = 10.0, force_refresh: bool = False):
        return {
            "ok": False,
            "status_code": None,
            "models": [],
            "error": "All connection attempts failed",
        }

    attempts: list[str] = []

    async def generate_should_not_run(**kwargs):
        attempts.append(str(kwargs.get("model")))
        return "unexpected"

    monkeypatch.setattr(studio_llm.model_service, "fetch_tags", unavailable_tags)
    monkeypatch.setattr(studio_llm, "_generate_with_ollama", generate_should_not_run)

    with pytest.raises(RuntimeError, match="Ollama is unavailable"):
        asyncio.run(
            studio_llm.generate_text(
                prompt="test outage",
                provider="ollama",
                model="missing-model",
                timeout=45.0,
            )
        )

    assert attempts == []


def test_studio_reachable_without_models_fails_before_generation(monkeypatch) -> None:
    async def empty_tags(timeout: float = 10.0, force_refresh: bool = False):
        return {
            "ok": True,
            "status_code": 200,
            "models": [],
            "error": None,
        }

    attempts: list[str] = []

    async def generate_should_not_run(**kwargs):
        attempts.append(str(kwargs.get("model")))
        return "unexpected"

    monkeypatch.setattr(studio_llm.model_service, "fetch_tags", empty_tags)
    monkeypatch.setattr(studio_llm, "_generate_with_ollama", generate_should_not_run)

    with pytest.raises(RuntimeError, match="no models are installed"):
        asyncio.run(
            studio_llm.generate_text(
                prompt="test missing models",
                provider="ollama",
                model="missing-model",
            )
        )

    assert attempts == []


def test_studio_missing_requested_model_uses_one_installed_fallback(monkeypatch) -> None:
    async def available_tags(timeout: float = 10.0, force_refresh: bool = False):
        return {
            "ok": True,
            "status_code": 200,
            "models": ["installed-model:latest"],
            "error": None,
        }

    attempts: list[str] = []

    async def generate(**kwargs):
        attempts.append(str(kwargs["model"]))
        return "fallback response"

    monkeypatch.setattr(studio_llm.model_service, "fetch_tags", available_tags)
    monkeypatch.setattr(studio_llm, "_generate_with_ollama", generate)

    result = asyncio.run(
        studio_llm.generate_text(
            prompt="test installed fallback",
            provider="ollama",
            model="missing-model",
        )
    )

    assert result["model"] == "installed-model:latest"
    assert result["text"] == "fallback response"
    assert attempts == ["installed-model:latest"]


def test_model_outage_is_actionable_while_engine_remains_scanning() -> None:
    config = EngineConfig(enabled=True, demo_autotrade=True, kill_switch=False)
    runtime = EngineRuntime(
        running=True,
        loop_active=True,
        ollama_ready=False,
        active_watchlist=["XAUUSD:M5"],
    )

    truth = router_module._status_truth_checks(config, _broker_ready(), runtime)
    truth_by_name = {item.name: item for item in truth}
    incidents = router_module._active_status_incidents(config, _broker_ready(), runtime)
    by_code = {item.code: item for item in incidents}

    assert truth_by_name["engine_scanning"].ok is True
    assert truth_by_name["model_ready"].ok is False
    assert "deterministic trading remains independent" in truth_by_name["model_ready"].detail
    assert "model_not_ready" in by_code
    assert "deterministic trading remains independent" in by_code["model_not_ready"].message
    assert "/api/llm_status" in by_code["model_not_ready"].message


def test_deterministic_analyze_works_without_model_service(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_LEGACY_CONTROLLER_ON_BOOT", "0")

    index = pd.date_range("2026-09-30T10:00:00Z", periods=250, freq="5min")
    close = pd.Series([100.0 + (i * 0.05) for i in range(250)], index=index)
    frame = pd.DataFrame(
        {
            "open": close - 0.02,
            "high": close + 0.10,
            "low": close - 0.10,
            "close": close,
            "volume": 1000.0,
        },
        index=index,
    )

    monkeypatch.setattr(router_module, "get_bars", lambda *args, **kwargs: frame)

    async def model_should_not_be_queried(*args, **kwargs):
        raise AssertionError("deterministic /api/analyze must not query model health")

    monkeypatch.setattr(model_service, "fetch_tags", model_should_not_be_queried)

    from backend.app import app

    with TestClient(app) as client:
        response = client.post(
            "/api/analyze",
            json={
                "symbol": "XAUUSD",
                "timeframe": "M5",
                "strategy": "sma_cross",
                "num_bars": 250,
                "params": {},
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["strategy"] == "sma_cross"
    assert payload["context"]["engine_source"] == "manual_analyze"
    assert list_order_intents(20) == []


def test_strategy_drafting_degrades_to_template_without_order_side_effects(monkeypatch) -> None:
    async def unavailable_generate(**kwargs):
        raise RuntimeError(
            "Ollama is unavailable: connection refused. "
            "Check /api/llm_status and start or reconnect Ollama before retrying."
        )

    monkeypatch.setattr(studio_tasks.studio_llm, "generate_text", unavailable_generate)

    result = asyncio.run(
        studio_tasks.execute_studio_task(
            StudioTaskRequest(
                task_type="chat",
                goal="create an XAUUSD M5 breakout strategy",
                params={"llm_provider": "ollama", "llm_model": "missing-model"},
            )
        )
    )

    assert result.status == "success"
    assert "built-in fallback" in result.message
    assert (result.result or {})["provider"] == "template"
    assert "def signals" in (result.result or {})["stdout"]
    assert list_order_intents(20) == []
