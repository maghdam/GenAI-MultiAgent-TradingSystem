from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from time import monotonic

import pandas as pd
import pytest

import backend.ctrader_client as ctd
from backend.adapters.ctrader import adapter
from backend.api import router as router_module
from backend.domain.models import BrokerStatus, EngineConfig, EngineRuntime, StrategyAnalysis, WatchlistItem
from backend.services import engine as engine_module
from backend.services import market_data
from backend.services.engine import V2Engine
from backend.services.execution_engine import ExecutionResult
from backend.services.market_bar_validation import assess_market_bar_snapshot
from backend.services.market_data import MarketDataError
from backend.services.runtime_state import market_data_dependency_state
from backend.storage.repositories import list_incidents, list_order_intents


@pytest.fixture(autouse=True)
def reset_market_state(monkeypatch):
    market_data._bars_cache.clear()
    market_data_dependency_state.last_checked_at = None
    market_data_dependency_state.last_symbol = ""
    market_data_dependency_state.last_timeframe = ""
    market_data_dependency_state.last_success = None
    market_data_dependency_state.last_success_at = None
    market_data_dependency_state.last_reason = ""
    market_data_dependency_state.market_data_ready = False

    monkeypatch.setattr(ctd, "is_connected", lambda: True)
    monkeypatch.setattr(ctd, "is_authorized", lambda: True)
    monkeypatch.setattr(ctd, "symbol_name_to_id", {"XAUUSD": 1})


def _config() -> EngineConfig:
    return EngineConfig(
        enabled=True,
        paper_autotrade=True,
        ctrader_autotrade=False,
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
        reasons=["malformed-market resilience test"],
        context={},
    )


def _frame(
    *,
    open_: object = 99.8,
    high: object = 100.3,
    low: object = 99.5,
    close: object = 100.0,
    at: datetime | None = None,
) -> pd.DataFrame:
    timestamp = at or (datetime.now(UTC) - timedelta(minutes=1))
    return pd.DataFrame(
        [{"open": open_, "high": high, "low": low, "close": close, "volume": 1000}],
        index=pd.DatetimeIndex([timestamp]),
    )


class _Strategy:
    def analyze(self, **kwargs) -> StrategyAnalysis:
        return _analysis()


@pytest.mark.parametrize(
    ("snapshot", "reason_fragment"),
    [
        ({"open": 100.0, "high": 101.0, "low": 99.0}, "incomplete or non-numeric"),
        ({"open": "bad", "high": 101.0, "low": 99.0, "close": 100.0}, "incomplete or non-numeric"),
        ({"open": float("nan"), "high": 101.0, "low": 99.0, "close": 100.0}, "incomplete or non-numeric"),
        ({"open": 100.0, "high": 101.0, "low": 0.0, "close": 100.0}, "non-positive"),
        ({"open": 100.0, "high": 99.0, "low": 101.0, "close": 100.0}, "invalid high/low"),
        ({"open": 102.0, "high": 101.0, "low": 99.0, "close": 100.0}, "outside the high-low"),
        ({"open": 100.0, "high": 104.0, "low": 96.0, "close": 100.0}, "range is too wide"),
    ],
)
def test_shared_validator_rejects_malformed_bar_classes(snapshot, reason_fragment) -> None:
    ok, details, reason = assess_market_bar_snapshot("M5", snapshot)

    assert ok is False
    assert reason is not None
    assert reason_fragment in reason
    assert isinstance(details, dict)


def test_ctrader_adapter_rejects_malformed_latest_row_instead_of_dropping_it(monkeypatch) -> None:
    now = datetime.now(UTC)
    rows = [
        {
            "time": now - timedelta(minutes=5),
            "open": 99.8,
            "high": 100.3,
            "low": 99.5,
            "close": 100.0,
            "volume": 1000,
        },
        {
            "time": now,
            "open": 100.0,
            "high": 100.4,
            "low": 99.7,
            "close": "not-a-number",
            "volume": 1000,
        },
    ]
    monkeypatch.setattr(ctd, "get_ohlc_data", lambda **kwargs: rows)

    with pytest.raises(RuntimeError, match="Malformed market data"):
        adapter._get_real_bars("XAUUSD", "M5", 2)


def test_memory_cache_malformed_frame_fails_closed_without_broker_fallback(monkeypatch) -> None:
    malformed = _frame(high=99.0, low=101.0)
    market_data._bars_cache[("XAUUSD", "M5")] = (monotonic(), malformed)

    monkeypatch.setattr(
        market_data.adapter,
        "get_bars",
        lambda **kwargs: pytest.fail("malformed memory cache must fail closed, not silently route elsewhere"),
    )

    with pytest.raises(MarketDataError, match="Malformed market data"):
        market_data.get_bars("XAUUSD", "M5", 1)

    assert market_data_dependency_state.market_data_ready is False
    assert "malformed market data" in market_data_dependency_state.last_reason.lower()


def test_malformed_auto_scan_suppresses_strategy_maintenance_and_intent(monkeypatch) -> None:
    malformed = _frame(close="bad")
    engine = V2Engine()
    state: dict[str, int] = {}

    monkeypatch.setattr(engine_module, "get_bars", lambda *args, **kwargs: malformed)
    monkeypatch.setattr(
        engine_module,
        "get_strategy",
        lambda *args, **kwargs: pytest.fail("strategy analysis must not run on malformed data"),
    )
    monkeypatch.setattr(
        engine_module,
        "execute_paper_signal",
        lambda **kwargs: pytest.fail("execution must not run on malformed data"),
    )
    monkeypatch.setattr(
        engine,
        "_mark_positions",
        lambda *args, **kwargs: pytest.fail("local marking must be suppressed on malformed data"),
    )
    monkeypatch.setattr(
        engine,
        "_sync_existing_ctrader_protection",
        lambda *args, **kwargs: pytest.fail("broker protection mutation must be suppressed on malformed data"),
    )

    for _ in range(2):
        with pytest.raises(MarketDataError, match="Malformed market data"):
            asyncio.run(engine._scan_item(_config(), _watch_item(), state))

    assert state == {}
    assert list_order_intents(20) == []

    incidents = [item for item in list_incidents(20) if item.code == "market_data_malformed"]
    assert len(incidents) == 1
    assert incidents[0].details["retryable"] is True
    assert incidents[0].details["bar_state_advanced"] is False
    assert incidents[0].details["strategy_analysis_suppressed"] is True
    assert incidents[0].details["order_intent_suppressed"] is True
    assert incidents[0].details["broker_mutation_suppressed"] is True


def test_valid_bar_recovery_processes_once_after_malformed_episode(monkeypatch) -> None:
    malformed = _frame(high=99.0, low=101.0)
    valid = _frame(at=datetime.now(UTC) - timedelta(minutes=1))
    current = {"frame": malformed}
    engine = V2Engine()
    state: dict[str, int] = {}
    execution_calls = 0

    monkeypatch.setattr(engine_module, "get_bars", lambda *args, **kwargs: current["frame"])
    monkeypatch.setattr(engine_module, "get_strategy", lambda name: _Strategy())
    monkeypatch.setattr(engine_module, "add_analysis", lambda value: value)
    monkeypatch.setattr(engine_module, "record_confluence_shadow", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine, "_mark_positions", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine, "_sync_existing_ctrader_protection", lambda *args, **kwargs: None)

    def _execute(**kwargs) -> ExecutionResult:
        nonlocal execution_calls
        execution_calls += 1
        return ExecutionResult(
            action_taken=True,
            intent_id=10,
            status="executed",
            summary="valid recovery executed",
            position_id=20,
            retryable=False,
        )

    monkeypatch.setattr(engine_module, "execute_paper_signal", _execute)

    with pytest.raises(MarketDataError, match="Malformed market data"):
        asyncio.run(engine._scan_item(_config(), _watch_item(), state))
    assert state == {}
    assert execution_calls == 0

    current["frame"] = valid
    recovered = asyncio.run(engine._scan_item(_config(), _watch_item(), state))
    assert recovered == (True, True)
    assert execution_calls == 1
    assert state["XAUUSD|M5"] == int(valid.index[-1].timestamp())

    replay = asyncio.run(engine._scan_item(_config(), _watch_item(), state))
    assert replay == (False, False)
    assert execution_calls == 1


def test_active_malformed_market_incident_is_actionable_and_clears() -> None:
    config = _config()
    runtime = EngineRuntime(
        running=True,
        loop_active=True,
        ollama_ready=True,
        active_watchlist=["XAUUSD:M5"],
    )
    malformed_reason = (
        "Malformed market data: Latest market bar has invalid high/low ordering. "
        "XAUUSD:M5 source=broker"
    )
    broker = BrokerStatus(
        connected=True,
        socket_connected=True,
        account_authorized=True,
        symbols_loaded=250,
        open_positions=0,
        pending_orders=0,
        ready=True,
        market_data_ready=False,
        broker_mode="demo",
        account_type="demo",
        account_verified=True,
        demo_account_confirmed=True,
        execution_ready=True,
        notes=[f"market_data_unavailable: {malformed_reason}"],
    )

    incidents = router_module._active_status_incidents(config, broker, runtime)
    malformed = next(item for item in incidents if item.code == "market_data_malformed")

    assert malformed.level == "warning"
    assert "invalid high/low" in malformed.message

    recovered = broker.model_copy(update={"market_data_ready": True, "notes": []})
    recovered_incidents = router_module._active_status_incidents(config, recovered, runtime)
    assert all(item.code != "market_data_malformed" for item in recovered_incidents)
