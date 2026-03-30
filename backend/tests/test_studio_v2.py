from __future__ import annotations

from fastapi.testclient import TestClient


def test_v2_models_endpoint_uses_model_service(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_LEGACY_CONTROLLER_ON_BOOT", "0")
    async def _fake_models_payload():
        return {"provider": "ollama", "models": ["phi3"], "default": "phi3", "fallback": None}

    monkeypatch.setattr("backend.api.router.model_service.models_payload", _fake_models_payload)
    from backend.app import app

    with TestClient(app) as client:
        response = client.get("/api/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "ollama"
    assert payload["models"] == ["phi3"]


def test_v2_studio_models_endpoint_uses_studio_service(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_LEGACY_CONTROLLER_ON_BOOT", "0")

    async def _fake_models_payload(provider=None):
        return {
            "provider": provider or "ollama",
            "providers": [{"key": "ollama", "label": "Ollama", "configured": True}],
            "models": ["llama3.1:8b"],
            "default": "llama3.1:8b",
            "fallback": None,
        }

    monkeypatch.setattr("backend.api.router.studio_llm.models_payload", _fake_models_payload)
    from backend.app import app

    with TestClient(app) as client:
        response = client.get("/api/studio/models?provider=ollama")

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "ollama"
    assert payload["models"] == ["llama3.1:8b"]


def test_v2_studio_strategy_files_endpoint_uses_service(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_LEGACY_CONTROLLER_ON_BOOT", "0")
    monkeypatch.setattr(
        "backend.api.router.list_saved_strategy_files",
        lambda: {"files": ["smc.py", "sma.py"], "cwd": "test"},
    )
    from backend.app import app

    with TestClient(app) as client:
        response = client.get("/api/studio/strategy-files")

    assert response.status_code == 200
    payload = response.json()
    assert payload["files"] == ["smc.py", "sma.py"]


def test_v2_studio_backtest_endpoint_uses_service(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_LEGACY_CONTROLLER_ON_BOOT", "0")
    monkeypatch.setattr(
        "backend.api.router.run_saved_strategy_backtest",
        lambda strategy, symbol, timeframe, num_bars, fee_bps, slippage_bps: {
            "strategy": strategy,
            "symbol": symbol,
            "timeframe": timeframe,
            "Total Return [%]": 12.5,
            "Fees [bps]": fee_bps,
            "Slippage [bps]": slippage_bps,
            "bars": num_bars,
        },
    )
    from backend.app import app

    with TestClient(app) as client:
        response = client.get(
            "/api/studio/backtest?strategy=smc&symbol=XAUUSD&timeframe=M5&num_bars=750&fee_bps=1.5&slippage_bps=0.5"
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["strategy"] == "smc"
    assert payload["symbol"] == "XAUUSD"
    assert payload["Total Return [%]"] == 12.5
    assert payload["Fees [bps]"] == 1.5
    assert payload["Slippage [bps]"] == 0.5
    assert payload["bars"] == 750
