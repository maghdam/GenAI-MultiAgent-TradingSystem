from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from fastapi.testclient import TestClient
import pytest

from backend.domain.models import EngineConfig, EngineRuntime, WatchlistItem
from backend.services import engine as engine_module
from backend.services.engine import V2Engine
from backend.services.engine_cycle_latency import build_engine_cycle_latency
from backend.storage import db as db_module
from backend.storage.repositories import (
    load_runtime,
    save_engine_config,
    save_runtime,
)


def _clock(monkeypatch, start: float, end: float) -> None:
    values = iter((start, end))
    monkeypatch.setattr(engine_module, "perf_counter", lambda: next(values))


def _restart_storage_connection() -> None:
    conn = getattr(db_module._LOCAL, "connection", None)
    if conn is not None:
        conn.close()
    db_module._LOCAL.connection = None
    db_module.init_db()


@pytest.mark.parametrize(
    ("config", "expected_summary"),
    [
        (EngineConfig(enabled=False, kill_switch=False), "engine disabled"),
        (EngineConfig(enabled=True, kill_switch=True), "kill switch active"),
        (EngineConfig(enabled=True, kill_switch=False, watchlist=[]), "watchlist empty"),
    ],
)
def test_early_return_cycles_record_monotonic_latency(
    monkeypatch,
    config: EngineConfig,
    expected_summary: str,
) -> None:
    save_engine_config(config)
    _clock(monkeypatch, 100.0, 100.025)

    summary = asyncio.run(V2Engine().run_once())

    runtime = load_runtime()
    assert summary == expected_summary
    assert runtime.tick_count == 1
    assert runtime.last_cycle_summary == expected_summary
    assert runtime.last_cycle_at is not None
    assert runtime.last_cycle_completed_at is not None
    assert runtime.last_cycle_completed_at >= runtime.last_cycle_at
    assert runtime.last_cycle_duration_ms == pytest.approx(25.0)


def test_normal_scan_cycle_records_latency_without_changing_scan_result(monkeypatch) -> None:
    save_engine_config(
        EngineConfig(
            enabled=True,
            kill_switch=False,
            demo_autotrade=False,
            watchlist=[
                WatchlistItem(
                    symbol="XAUUSD",
                    timeframe="M5",
                    strategy="sma_cross",
                    enabled=True,
                    trading_enabled=False,
                )
            ],
        )
    )
    subject = V2Engine()
    scan_calls: list[str] = []

    async def _scan(config, item, bar_state):
        scan_calls.append(f"{item.symbol}:{item.timeframe}")
        return True, False

    monkeypatch.setattr(subject, "_scan_item", _scan)
    _clock(monkeypatch, 200.0, 200.123)

    summary = asyncio.run(subject.run_once())

    assert summary == "processed=1 actions=0 watchlist=1 market_skips=0"
    assert scan_calls == ["XAUUSD:M5"]
    runtime = load_runtime()
    assert runtime.loop_active is True
    assert runtime.tick_count == 1
    assert runtime.last_cycle_summary == summary
    assert runtime.last_cycle_duration_ms == pytest.approx(123.0)


def test_erroring_cycle_records_latency_and_preserves_exception(monkeypatch) -> None:
    subject = V2Engine()

    async def _boom() -> str:
        raise RuntimeError("simulated cycle failure")

    monkeypatch.setattr(subject, "_run_once_body", _boom)
    _clock(monkeypatch, 300.0, 300.040)

    with pytest.raises(RuntimeError, match="simulated cycle failure"):
        asyncio.run(subject.run_once())

    runtime = load_runtime()
    assert runtime.last_cycle_at is not None
    assert runtime.last_cycle_completed_at is not None
    assert runtime.last_cycle_duration_ms == pytest.approx(40.0)


def test_latency_persistence_failure_does_not_change_cycle_result(monkeypatch) -> None:
    subject = V2Engine()

    async def _ok() -> str:
        return "cycle result preserved"

    monkeypatch.setattr(subject, "_run_once_body", _ok)
    monkeypatch.setattr(
        engine_module,
        "save_runtime",
        lambda runtime: (_ for _ in ()).throw(RuntimeError("telemetry write failed")),
    )
    _clock(monkeypatch, 400.0, 400.010)

    assert asyncio.run(subject.run_once()) == "cycle result preserved"


def test_cycle_latency_report_is_unavailable_before_first_completed_cycle() -> None:
    report = build_engine_cycle_latency()

    assert report.available is False
    assert report.unit == "ms"
    assert report.duration_ms is None
    assert report.cycle_started_at is None
    assert report.cycle_completed_at is None
    assert report.tick_count == 0
    assert "no completed engine cycle" in report.message.lower()


def test_cycle_latency_survives_storage_restart(monkeypatch) -> None:
    save_engine_config(EngineConfig(enabled=False, kill_switch=False))
    _clock(monkeypatch, 500.0, 500.075)

    asyncio.run(V2Engine().run_once())
    before = build_engine_cycle_latency()

    _restart_storage_connection()
    after = build_engine_cycle_latency()

    assert before.available is True
    assert before.duration_ms == pytest.approx(75.0)
    assert after.model_dump() == before.model_dump()
    assert after.cycle_started_at is not None
    assert after.cycle_completed_at is not None


def test_cycle_latency_api_is_read_only(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_EVENT_INTELLIGENCE_ON_BOOT", "0")

    started_at = datetime(2026, 10, 3, 6, 0, tzinfo=UTC).replace(tzinfo=None)
    completed_at = datetime(2026, 10, 3, 6, 0, 0, 125000, tzinfo=UTC).replace(tzinfo=None)
    save_runtime(
        EngineRuntime(
            running=True,
            loop_active=True,
            last_cycle_at=started_at,
            last_cycle_completed_at=completed_at,
            last_cycle_duration_ms=125.0,
            last_cycle_summary="processed=1 actions=0 watchlist=1 market_skips=0",
            tick_count=7,
            active_watchlist=["XAUUSD:M5"],
        )
    )
    before = load_runtime().model_dump()

    from backend.app import app

    with TestClient(app) as client:
        response = client.get("/api/reports/engine-cycle-latency")

    after = load_runtime().model_dump()
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["unit"] == "ms"
    assert payload["duration_ms"] == pytest.approx(125.0)
    assert payload["tick_count"] == 7
    assert payload["last_cycle_summary"] == "processed=1 actions=0 watchlist=1 market_skips=0"
    assert before == after
