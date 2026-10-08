from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from backend.domain.models import StudioTaskRequest
from backend.services import studio_llm, studio_tasks


VALID_CODE = (
    "import pandas as pd\n\n"
    "def signals(df: pd.DataFrame) -> pd.Series:\n"
    "    return pd.Series(0.0, index=df.index)\n"
)


def test_chat_request_returns_llm_research_reply(monkeypatch) -> None:
    async def fake_generate_text(**kwargs):
        return {
            "provider": "ollama",
            "model": "research-model",
            "text": "Use a trend filter and define the invalidation rule before backtesting.",
        }

    monkeypatch.setattr(studio_tasks.studio_llm, "generate_text", fake_generate_text)

    result = asyncio.run(
        studio_tasks.execute_studio_task(
            StudioTaskRequest(
                task_type="chat",
                goal="How should I make this mean-reversion idea more testable?",
                params={"symbol": "XAUUSD", "timeframe": "M5"},
            )
        )
    )

    assert result.status == "success"
    assert "trend filter" in result.message
    assert result.result is None


def test_strategy_drafting_returns_valid_editable_source(monkeypatch) -> None:
    async def fake_generate_text(**kwargs):
        return {
            "provider": "ollama",
            "model": "draft-model",
            "text": VALID_CODE,
        }

    monkeypatch.setattr(studio_tasks.studio_llm, "generate_text", fake_generate_text)

    result = asyncio.run(
        studio_tasks.execute_studio_task(
            StudioTaskRequest(
                task_type="create_strategy",
                goal="Create a simple XAUUSD M5 trend strategy",
                params={},
            )
        )
    )

    assert result.status == "success"
    assert (result.result or {})["stdout"] == VALID_CODE.strip()
    assert (result.result or {})["provider"] == "ollama"
    assert (result.result or {})["model"] == "draft-model"


def test_selected_provider_and_model_are_forwarded_to_generation(monkeypatch) -> None:
    observed = {}

    async def fake_generate_text(**kwargs):
        observed.update(kwargs)
        return {
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "text": VALID_CODE,
        }

    monkeypatch.setattr(studio_tasks.studio_llm, "generate_text", fake_generate_text)

    result = asyncio.run(
        studio_tasks.execute_studio_task(
            StudioTaskRequest(
                task_type="chat",
                goal="Create a strategy with explicit entry and exit rules",
                params={
                    "llm_provider": "gemini",
                    "llm_model": "gemini-2.5-flash",
                    "symbol": "US100",
                    "timeframe": "M5",
                },
            )
        )
    )

    assert result.status == "success"
    assert observed["provider"] == "gemini"
    assert observed["model"] == "gemini-2.5-flash"
    assert (result.result or {})["provider"] == "gemini"
    assert (result.result or {})["model"] == "gemini-2.5-flash"


def test_ollama_generation_retries_without_thinking_and_keeps_studio_timeout(monkeypatch) -> None:
    calls = []

    def fake_ollama_generate(**kwargs):
        calls.append(dict(kwargs))
        if kwargs.get("think") != False:
            raise RuntimeError(
                'Ollama API error: 400 — {"error":"\\\"phi3:mini\\\" does not support thinking"}'
            )
        return VALID_CODE

    monkeypatch.setattr(studio_llm, "_OLLAMA_STUDIO_THINK", "low")
    monkeypatch.setattr(studio_llm, "_ollama_generate", fake_ollama_generate)

    result = asyncio.run(
        studio_llm._generate_with_ollama(
            prompt="create strategy",
            model="phi3:mini",
            timeout=90.0,
            num_predict=900,
        )
    )

    assert result == VALID_CODE
    assert [call["think"] for call in calls] == ["low", False]
    assert all(call["attempt_timeout_cap"] == 90.0 for call in calls)


def test_strategy_drafting_uses_safe_template_fallback_when_llm_fails(monkeypatch) -> None:
    async def failing_generate_text(**kwargs):
        raise TimeoutError("selected provider timed out")

    async def fallback_generate_code(self, goal: str, task_type: str) -> str:
        assert task_type == "strategy"
        return VALID_CODE

    monkeypatch.setattr(studio_tasks.studio_llm, "generate_text", failing_generate_text)
    monkeypatch.setattr(studio_tasks.ProgrammerAgent, "generate_code", fallback_generate_code)

    result = asyncio.run(
        studio_tasks.execute_studio_task(
            StudioTaskRequest(
                task_type="chat",
                goal="Create an XAUUSD M5 strategy with a trend filter",
                params={"llm_provider": "ollama", "llm_model": "missing-model"},
            )
        )
    )

    assert result.status == "success"
    assert "built-in fallback" in result.message
    assert (result.result or {})["provider"] == "template"
    assert (result.result or {})["model"] == "programmer_agent"
    assert (result.result or {})["stdout"] == VALID_CODE.strip()


def test_save_strategy_persists_validated_source_and_draft_lifecycle(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)

    result = asyncio.run(
        studio_tasks.execute_studio_task(
            StudioTaskRequest(
                task_type="save_strategy",
                goal="save strategy phase41_saved",
                params={
                    "strategy_name": "phase41_saved",
                    "code": VALID_CODE,
                },
            )
        )
    )

    saved = Path("backend/strategies_generated/phase41_saved.py")

    assert result.status == "success"
    assert saved.exists()
    assert saved.read_text(encoding="utf-8").strip() == VALID_CODE.strip()
    lifecycle = (result.result or {})["lifecycle"]
    assert lifecycle["strategy"] == "phase41_saved"
    assert lifecycle["stage"] == "draft"


def test_reload_saved_strategy_returns_source_through_api(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_LEGACY_CONTROLLER_ON_BOOT", "0")

    root = Path("backend/strategies_generated")
    root.mkdir(parents=True, exist_ok=True)
    (root / "phase41_reload.py").write_text(VALID_CODE, encoding="utf-8")

    from backend.app import app

    with TestClient(app) as client:
        response = client.get("/api/studio/strategy-file/phase41_reload")

    assert response.status_code == 200
    payload = response.json()
    assert payload["strategy"] == "phase41_reload"
    assert payload["filename"] == "phase41_reload.py"
    assert payload["source"] == VALID_CODE
