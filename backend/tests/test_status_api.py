from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from backend.domain.models import SymbolLimits


def test_v2_status_exposes_new_risk_and_audit_fields(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_LEGACY_CONTROLLER_ON_BOOT", "0")
    from backend.app import app

    with TestClient(app) as client:
        response = client.get("/api/status")

    assert response.status_code == 200
    payload = response.json()

    assert "recent_order_intents" in payload
    assert "recent_trade_audits" in payload

    config = payload["config"]
    readiness_names = {item["name"] for item in payload["readiness"]}
    assert "max_daily_trades" in config
    assert "max_positions_per_symbol" in config
    assert "cooldown_minutes" in config
    assert "session_filter_enabled" in config
    assert "session_start_hour_utc" in config
    assert "session_end_hour_utc" in config
    assert "market_data_feed" in readiness_names
    assert any("startup disabled" in note for note in payload["broker"]["notes"])


def test_v2_market_status_reports_feed_unavailable_without_booting_services(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_LEGACY_CONTROLLER_ON_BOOT", "0")
    from backend.app import app

    with TestClient(app) as client:
        response = client.get("/api/market/status?symbol=XAUUSD&timeframe=M5")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "XAUUSD"
    assert payload["timeframe"] == "M5"
    assert payload["ok"] is False
    assert "not connected" in payload["reason"]


def test_v2_symbols_and_candles_endpoints(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_LEGACY_CONTROLLER_ON_BOOT", "0")
    monkeypatch.setattr("backend.api.router.list_symbols", lambda: ["EURUSD", "XAUUSD"])

    df = pd.DataFrame(
        [
            {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5},
            {"open": 1.5, "high": 2.2, "low": 1.2, "close": 2.0},
        ],
        index=pd.to_datetime(["2026-03-19T10:00:00Z", "2026-03-19T10:05:00Z"], utc=True),
    )
    monkeypatch.setattr("backend.api.router.get_bars", lambda symbol, timeframe, num_bars: df)

    from backend.app import app

    with TestClient(app) as client:
        symbols_response = client.get("/api/symbols")
        candles_response = client.get("/api/market/candles?symbol=XAUUSD&timeframe=M5&num_bars=2")

    assert symbols_response.status_code == 200
    assert symbols_response.json()["symbols"] == ["EURUSD", "XAUUSD"]

    assert candles_response.status_code == 200
    payload = candles_response.json()
    assert len(payload["candles"]) == 2
    assert payload["candles"][0]["open"] == 1.0
    assert payload["indicators"] == {}


def test_v2_symbol_limits_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_LEGACY_CONTROLLER_ON_BOOT", "0")
    monkeypatch.setattr(
        "backend.api.router.get_symbol_limits",
        lambda symbol: SymbolLimits(
            symbol=symbol.upper(),
            source="broker",
            min_lots=0.01,
            step_lots=0.01,
            max_lots=50.0,
            min_api_units=100,
            step_api_units=100,
            max_api_units=500000,
            hard_min=False,
            hard_step=False,
        ),
    )

    from backend.app import app

    with TestClient(app) as client:
        response = client.get("/api/symbol-limits?symbol=XAUUSD")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "XAUUSD"
    assert payload["step_lots"] == 0.01


def test_v2_manual_order_executes_even_if_paper_autotrade_is_disabled(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_LEGACY_CONTROLLER_ON_BOOT", "0")
    from backend.domain.models import EngineConfig
    from backend.storage.repositories import save_engine_config

    save_engine_config(
        EngineConfig(
            enabled=True,
            paper_autotrade=False,
            kill_switch=False,
            default_symbol="XAUUSD",
            default_timeframe="M5",
        )
    )

    from backend.app import app

    with TestClient(app) as client:
        response = client.post(
            "/api/orders/manual",
            json={
                "symbol": "XAUUSD",
                "timeframe": "M5",
                "strategy": "sma_cross",
                "signal": "long",
                "quantity": 0.5,
                "confidence": 0.9,
                "entry_price": 100.0,
                "stop_loss": 99.0,
                "take_profit": 102.0,
                "reasons": ["manual dashboard trade"],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["status"] == "executed"


def test_v2_manual_order_rejects_invalid_quantity(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_LEGACY_CONTROLLER_ON_BOOT", "0")
    monkeypatch.setattr(
        "backend.services.quantity_rules.get_symbol_limits",
        lambda symbol: SymbolLimits(
            symbol=symbol.upper(),
            source="broker",
            min_lots=0.05,
            step_lots=0.05,
            max_lots=2.0,
            min_api_units=500,
            step_api_units=500,
            max_api_units=20000,
            hard_min=True,
            hard_step=True,
        ),
    )

    from backend.domain.models import EngineConfig
    from backend.storage.repositories import save_engine_config

    save_engine_config(
        EngineConfig(
            enabled=True,
            paper_autotrade=False,
            kill_switch=False,
            default_symbol="XAUUSD",
            default_timeframe="M5",
        )
    )

    from backend.app import app

    with TestClient(app) as client:
        response = client.post(
            "/api/orders/manual",
            json={
                "symbol": "XAUUSD",
                "timeframe": "M5",
                "strategy": "sma_cross",
                "signal": "long",
                "quantity": 0.12,
                "confidence": 0.9,
                "entry_price": 100.0,
                "stop_loss": 99.0,
                "take_profit": 102.0,
                "reasons": ["manual dashboard trade"],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["status"] == "rejected"
    assert "step size of 0.0500 lots" in payload["summary"]
