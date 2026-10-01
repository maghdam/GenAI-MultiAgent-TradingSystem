from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from backend.adapters.ctrader import adapter
from backend.api import router as router_module
from backend.domain.models import (
    BrokerStatus,
    EngineConfig,
    EngineRuntime,
    StrategyAnalysis,
    WatchlistItem,
)
from backend.services import engine as engine_module
from backend.services import market_data
from backend.services.engine import V2Engine
from backend.services.execution_engine import ExecutionResult
from backend.services.market_data import MarketDataError
from backend.services.runtime_state import market_data_dependency_state
from backend.storage.repositories import (
    list_incidents,
    list_order_intents,
    load_bar_state,
    save_engine_config,
)


@pytest.fixture(autouse=True)
def reset_market_dependency_state():
    market_data_dependency_state.last_checked_at = None
    market_data_dependency_state.last_symbol = ""
    market_data_dependency_state.last_timeframe = ""
    market_data_dependency_state.last_success = None
    market_data_dependency_state.last_success_at = None
    market_data_dependency_state.last_reason = ""
    market_data_dependency_state.market_data_ready = False
    yield


def _config() -> EngineConfig:
    return EngineConfig(
        enabled=True,
        paper_autotrade=True,
        demo_autotrade=False,
        kill_switch=False,
        min_confidence=0.60,
        watchlist=[_watch_item()],
    )


def _watch_item() -> WatchlistItem:
    return WatchlistItem(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        enabled=True,
        trading_enabled=False,
        params={},
    )


def _analysis() -> StrategyAnalysis:
    return StrategyAnalysis(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        signal="long",
        confidence=0.82,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        reasons=["stale-market resilience test"],
        context={},
    )


def _bars(at: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        [{"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0}],
        index=pd.DatetimeIndex([at]),
    )


class _Strategy:
    def analyze(self, **kwargs) -> StrategyAnalysis:
        return _analysis()


def _broker(*, market_ready: bool, notes: list[str]) -> BrokerStatus:
    return BrokerStatus(
        connected=True,
        socket_connected=True,
        account_authorized=True,
        symbols_loaded=250,
        open_positions=0,
        pending_orders=0,
        ready=True,
        market_data_ready=market_ready,
        broker_mode="demo",
        account_type="demo",
        demo_account_confirmed=True,
        execution_ready=True,
        notes=notes,
    )


def test_shared_market_freshness_classifier_distinguishes_stale_and_fresh() -> None:
    now = datetime(2026, 10, 1, 15, 0, tzinfo=UTC)

    stale_ok, stale_details, stale_reason = market_data.assess_market_bar_freshness(
        "M5",
        now - timedelta(minutes=20),
        now=now,
    )
    fresh_ok, fresh_details, fresh_reason = market_data.assess_market_bar_freshness(
        "M5",
        now - timedelta(minutes=5),
        now=now,
    )

    assert stale_ok is False
    assert stale_reason == "Latest market bar is stale for the configured timeframe."
    assert stale_details["bar_age_seconds"] == pytest.approx(1200.0)
    assert stale_details["max_bar_age_seconds"] == pytest.approx(930.0)

    assert fresh_ok is True
    assert fresh_reason is None
    assert fresh_details["bar_age_seconds"] == pytest.approx(300.0)
    assert fresh_details["max_bar_age_seconds"] == pytest.approx(930.0)


def test_stale_auto_scan_defers_before_same_bar_maintenance_or_intent(monkeypatch) -> None:
    stale_at = datetime.now(UTC) - timedelta(minutes=20)
    stale_bars = _bars(stale_at)
    engine = V2Engine()
    bar_state: dict[str, int] = {}

    monkeypatch.setattr(engine_module, "get_bars", lambda *args, **kwargs: stale_bars)
    monkeypatch.setattr(
        engine_module,
        "get_strategy",
        lambda *args, **kwargs: pytest.fail("strategy analysis must not run on a stale bar"),
    )
    monkeypatch.setattr(
        engine_module,
        "execute_paper_signal",
        lambda **kwargs: pytest.fail("execution must not run on a stale bar"),
    )
    monkeypatch.setattr(
        engine,
        "_mark_positions",
        lambda *args, **kwargs: pytest.fail("local position marking must be suppressed on stale data"),
    )
    monkeypatch.setattr(
        engine,
        "_sync_existing_demo_protection",
        lambda *args, **kwargs: pytest.fail("broker protection maintenance must be suppressed on stale data"),
    )

    for _ in range(2):
        with pytest.raises(MarketDataError, match="stale"):
            asyncio.run(engine._scan_item(_config(), _watch_item(), bar_state))

    assert bar_state == {}
    assert list_order_intents(20) == []
    incidents = [item for item in list_incidents(20) if item.code == "market_data_stale"]
    assert len(incidents) == 1
    assert incidents[0].details["retryable"] is True
    assert incidents[0].details["bar_state_advanced"] is False
    assert incidents[0].details["order_intent_suppressed"] is True
    assert incidents[0].details["broker_mutation_suppressed"] is True
    assert market_data_dependency_state.market_data_ready is False
    assert "stale" in market_data_dependency_state.last_reason.lower()


def test_run_once_counts_stale_market_skip_without_consuming_bar(monkeypatch) -> None:
    stale_at = datetime.now(UTC) - timedelta(minutes=20)
    save_engine_config(_config())
    monkeypatch.setattr(engine_module, "get_bars", lambda *args, **kwargs: _bars(stale_at))

    summary = asyncio.run(V2Engine().run_once())

    assert "processed=0" in summary
    assert "actions=0" in summary
    assert "market_skips=1" in summary
    assert load_bar_state() == {}
    assert list_order_intents(20) == []
    assert market_data_dependency_state.market_data_ready is False
    assert "XAUUSD:M5" in market_data_dependency_state.last_reason


def test_fresh_bar_recovery_processes_once_and_prevents_duplicate_replay(monkeypatch) -> None:
    stale_at = datetime.now(UTC) - timedelta(minutes=20)
    fresh_at = datetime.now(UTC) - timedelta(minutes=1)
    current = {"bars": _bars(stale_at)}
    engine = V2Engine()
    bar_state: dict[str, int] = {}
    executions = 0

    monkeypatch.setattr(engine_module, "get_bars", lambda *args, **kwargs: current["bars"])
    monkeypatch.setattr(engine_module, "get_strategy", lambda name: _Strategy())
    monkeypatch.setattr(engine_module, "add_analysis", lambda analysis: analysis)
    monkeypatch.setattr(engine_module, "record_confluence_shadow", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine, "_mark_positions", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine, "_sync_existing_demo_protection", lambda *args, **kwargs: None)

    def _execute(**kwargs) -> ExecutionResult:
        nonlocal executions
        executions += 1
        return ExecutionResult(
            action_taken=True,
            intent_id=10,
            status="executed",
            summary="fresh recovery executed",
            position_id=20,
            retryable=False,
        )

    monkeypatch.setattr(engine_module, "execute_paper_signal", _execute)

    with pytest.raises(MarketDataError, match="stale"):
        asyncio.run(engine._scan_item(_config(), _watch_item(), bar_state))
    assert bar_state == {}
    assert executions == 0

    current["bars"] = _bars(fresh_at)
    recovered = asyncio.run(engine._scan_item(_config(), _watch_item(), bar_state))
    assert recovered == (True, True)
    assert executions == 1
    assert bar_state["XAUUSD|M5"] == int(fresh_at.timestamp())
    assert market_data_dependency_state.market_data_ready is True

    replay = asyncio.run(engine._scan_item(_config(), _watch_item(), bar_state))
    assert replay == (False, False)
    assert executions == 1


def test_market_status_reports_stale_then_recovers_on_fresh_bar(monkeypatch) -> None:
    stale_at = datetime.now(UTC) - timedelta(minutes=20)
    fresh_at = datetime.now(UTC) - timedelta(minutes=1)
    latest = {"value": stale_at}

    monkeypatch.setattr(
        market_data.adapter,
        "get_market_data_status",
        lambda **kwargs: {
            "checked_at": datetime.now(UTC).isoformat(),
            "symbol": "XAUUSD",
            "timeframe": "M5",
            "ok": True,
            "reason": "Fetched 5 bars",
            "latest_bar_at": latest["value"].isoformat(),
        },
    )

    stale = market_data.get_market_data_status("XAUUSD", "M5")
    assert stale["ok"] is False
    assert stale["market_data_ready"] is False
    assert "stale" in str(stale["reason"]).lower()

    latest["value"] = fresh_at
    recovered = market_data.get_market_data_status("XAUUSD", "M5")
    assert recovered["ok"] is True
    assert recovered["market_data_ready"] is True
    assert market_data_dependency_state.last_success is True


def test_active_stale_market_incident_is_actionable_and_clears_after_recovery() -> None:
    config = EngineConfig(
        enabled=True,
        demo_autotrade=False,
        kill_switch=False,
        watchlist=[_watch_item()],
    )
    runtime = EngineRuntime(
        running=True,
        loop_active=True,
        ollama_ready=True,
        active_watchlist=["XAUUSD:M5"],
    )
    message = (
        "Latest market bar is stale for the configured timeframe. "
        "XAUUSD:M5 (age=1200.0s, max=930.0s)"
    )
    stale_broker = _broker(
        market_ready=False,
        notes=[f"market_data_unavailable: {message}"],
    )

    incidents = router_module._active_status_incidents(config, stale_broker, runtime)
    stale_incident = next(item for item in incidents if item.code == "market_data_stale")

    assert stale_incident.level == "warning"
    assert "XAUUSD:M5" in stale_incident.message
    assert "stale" in stale_incident.message.lower()

    recovered_broker = stale_broker.model_copy(
        update={
            "market_data_ready": True,
            "notes": [],
        }
    )
    recovered = router_module._active_status_incidents(config, recovered_broker, runtime)
    assert all(item.code != "market_data_stale" for item in recovered)
