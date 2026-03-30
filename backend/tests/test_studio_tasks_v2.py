from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient


def test_v2_studio_tasks_endpoint_uses_service(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_LEGACY_CONTROLLER_ON_BOOT", "0")

    async def _fake_execute(request):
        assert request.task_type == "chat"
        assert request.goal == "what can you do?"
        from backend.domain.models import StudioTaskResponse

        return StudioTaskResponse(status="success", message="help text")

    monkeypatch.setattr("backend.api.router.execute_studio_task", _fake_execute)
    from backend.app import app

    with TestClient(app) as client:
        response = client.post(
            "/api/studio/tasks",
            json={"task_type": "chat", "goal": "what can you do?", "params": {"symbol": "XAUUSD"}},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["message"] == "help text"


def test_v2_studio_save_strategy_validates_required_fields() -> None:
    from backend.domain.models import StudioTaskRequest
    from backend.services.studio_tasks import execute_studio_task

    result = asyncio.run(execute_studio_task(StudioTaskRequest(task_type="save_strategy", goal="", params={})))

    assert result.status == "error"
    assert "strategy_name is required" in result.message


def test_v2_studio_create_strategy_generates_smc_code_for_smc_requests(monkeypatch) -> None:
    from backend.domain.models import StudioTaskRequest
    from backend.services.studio_tasks import execute_studio_task

    async def _fake_generate_text(**kwargs):
        return {
            "provider": "ollama",
            "model": "llama3.1:8b",
            "text": (
                "import pandas as pd\n\n"
                "# FVG\n"
                "# Market structure\n"
                "def signals(df: pd.DataFrame) -> pd.Series:\n"
                "    return pd.Series(0.0, index=df.index)\n"
            ),
        }

    monkeypatch.setattr("backend.services.studio_tasks.studio_llm.generate_text", _fake_generate_text)

    result = asyncio.run(
        execute_studio_task(
            StudioTaskRequest(
                task_type="create_strategy",
                goal="create a strategy using fvg, market structure, smc",
                params={},
            )
        )
    )

    assert result.status == "success"
    stdout = (result.result or {}).get("stdout", "")
    assert "Default crossover strategy" not in stdout
    assert "FVG" in stdout
    assert "Market structure" in stdout
    assert "def signals" in stdout


def test_v2_studio_chat_creates_draft_with_llm_provider(monkeypatch) -> None:
    from backend.domain.models import StudioTaskRequest
    from backend.services.studio_tasks import execute_studio_task

    async def _fake_generate_text(**kwargs):
        return {
            "provider": "ollama",
            "model": "llama3.1:8b",
            "text": "import pandas as pd\n\ndef signals(df: pd.DataFrame) -> pd.Series:\n    return pd.Series(0.0, index=df.index)\n",
        }

    monkeypatch.setattr("backend.services.studio_tasks.studio_llm.generate_text", _fake_generate_text)

    result = asyncio.run(
        execute_studio_task(
            StudioTaskRequest(
                task_type="chat",
                goal="create a trend strategy",
                params={"llm_provider": "ollama", "llm_model": "llama3.1:8b"},
            )
        )
    )

    assert result.status == "success"
    assert "stdout" in (result.result or {})
    assert "llama3.1:8b" in result.message


def test_v2_studio_chat_falls_back_to_template_when_create_times_out(monkeypatch) -> None:
    from backend.domain.models import StudioTaskRequest
    from backend.services.studio_tasks import execute_studio_task

    async def _fake_generate_text(**kwargs):
        raise TimeoutError("LLM request timed out after 45.0 seconds.")

    monkeypatch.setattr("backend.services.studio_tasks.studio_llm.generate_text", _fake_generate_text)

    result = asyncio.run(
        execute_studio_task(
            StudioTaskRequest(
                task_type="chat",
                goal="create an XAUUSD M5 strategy using FVG and market structure",
                params={"llm_provider": "ollama", "llm_model": "phi3:mini"},
            )
        )
    )

    assert result.status == "success"
    assert "built-in fallback" in result.message
    stdout = (result.result or {}).get("stdout", "")
    assert "def signals" in stdout
    assert "FVG" in stdout


def test_v2_studio_chat_can_save_current_draft(monkeypatch) -> None:
    from backend.domain.models import StudioTaskRequest
    from backend.domain.models import StudioTaskResponse
    from backend.services.studio_tasks import execute_studio_task

    def _fake_save(name: str, code: str) -> StudioTaskResponse:
        assert name == "chat_saved_strategy"
        assert "def signals" in code
        return StudioTaskResponse(status="success", message="Strategy saved as backend/strategies_generated/chat_saved_strategy.py")

    monkeypatch.setattr("backend.services.studio_tasks._save_strategy_code", _fake_save)

    result = asyncio.run(
        execute_studio_task(
            StudioTaskRequest(
                task_type="chat",
                goal="save it as chat_saved_strategy",
                params={
                    "current_code": "import pandas as pd\n\ndef signals(df: pd.DataFrame) -> pd.Series:\n    return pd.Series(0.0, index=df.index)\n",
                },
            )
        )
    )

    assert result.status == "success"
    assert "chat_saved_strategy.py" in result.message
