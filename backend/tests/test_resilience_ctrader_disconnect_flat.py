from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pandas as pd
import pytest

from backend.adapters.ctrader import CTraderBrokerAdapter
from backend.api import router as router_module
from backend.domain.models import BrokerStatus, EngineConfig, EngineRuntime, StrategyAnalysis, WatchlistItem
from backend.services import engine as engine_module
from backend.services.engine import V2Engine
from backend.services.execution_engine import ExecutionResult, execute_paper_signal
from backend.storage.repositories import list_incidents, list_order_intents, list_paper_positions


def _config() -> EngineConfig:
    return EngineConfig(
        enabled=True,
        paper_autotrade=False,
        demo_autotrade=True,
        kill_switch=False,
        require_stops=True,
        min_confidence=0.60,
    )


def _watch_item() -> WatchlistItem:
    return WatchlistItem(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        enabled=True,
        trading_enabled=True,
        lot_size=0.10,
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
        reasons=["disconnect resilience test"],
        context={},
    )


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        [{"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0}],
        index=pd.to_datetime(["2026-09-30T15:45:00Z"], utc=True),
    )


class _Strategy:
    def analyze(self, **kwargs) -> StrategyAnalysis:
        return _analysis()


def test_demo_preflight_requires_live_transport_and_authorization_even_with_cached_contract(monkeypatch) -> None:
    from backend import ctrader_client as ctd

    connected = {"value": False}
    authorized = {"value": True}

    monkeypatch.setattr(ctd, "is_connected", lambda: connected["value"])
    monkeypatch.setattr(ctd, "is_authorized", lambda: authorized["value"])
    monkeypatch.setattr(ctd, "get_auth_error", lambda: "authorization expired")
    monkeypatch.setattr(ctd, "is_demo_account_confirmed", lambda: True)
    monkeypatch.setattr(ctd, "get_account_verification_error", lambda: None)
    monkeypatch.setattr(ctd, "symbol_name_to_id", {"XAUUSD": 7})
    monkeypatch.setattr(ctd, "symbol_lot_size_map", {7: 100.0})
    monkeypatch.setattr(ctd, "symbol_min_volume_map", {7: 100})
    monkeypatch.setattr(ctd, "symbol_step_volume_map", {7: 100})
    monkeypatch.setattr(ctd, "symbol_max_volume_map", {7: 500_000})

    adapter = CTraderBrokerAdapter()

    ready, reason = adapter.demo_symbol_execution_readiness("XAUUSD")
    assert ready is False
    assert reason == "cTrader transport is not connected."

    connected["value"] = True
    authorized["value"] = False
    ready, reason = adapter.demo_symbol_execution_readiness("XAUUSD")
    assert ready is False
    assert reason == "authorization expired"

    authorized["value"] = True
    ready, reason = adapter.demo_symbol_execution_readiness("XAUUSD")
    assert ready is True
    assert "ready" in reason.lower()


def test_demo_order_guard_never_submits_when_transport_is_disconnected(monkeypatch) -> None:
    from backend import ctrader_client as ctd

    monkeypatch.setattr(ctd, "is_connected", lambda: False)
    monkeypatch.setattr(
        ctd,
        "place_order",
        lambda **kwargs: pytest.fail("disconnected demo order must never reach broker submission"),
    )

    with pytest.raises(RuntimeError, match="transport is not connected"):
        CTraderBrokerAdapter().place_demo_market_order(
            symbol="XAUUSD",
            direction="long",
            quantity_lots=0.10,
            stop_loss=99.0,
            take_profit=102.0,
            client_msg_id="disconnect-flat-test",
        )


def test_flat_demo_execution_defers_before_order_submission_when_disconnected(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.services.execution_engine.get_demo_symbol_execution_readiness",
        lambda symbol: (False, "cTrader transport is not connected."),
    )
    monkeypatch.setattr(
        "backend.services.execution_engine.place_demo_market_order",
        lambda **kwargs: pytest.fail("preflight disconnect must block broker order submission"),
    )

    result = execute_paper_signal(
        config=_config(),
        watch_item=_watch_item(),
        analysis=_analysis(),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC),
        bar_snapshot={"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0},
    )

    assert result.action_taken is False
    assert result.status == "deferred"
    assert result.retryable is True
    assert result.intent_id is None
    assert list_paper_positions("open") == []
    assert list_order_intents(10) == []

    incidents = list_incidents(10)
    assert incidents[0].code == "ctrader_demo_symbol_not_ready"
    assert incidents[0].details["reason"] == "cTrader transport is not connected."
    assert incidents[0].details["retryable"] is True


def test_same_bar_is_retried_once_after_reconnect_without_duplicate_submission(monkeypatch) -> None:
    bars = _bars()
    state: dict[str, int] = {}
    execution_calls = 0
    broker_submissions = 0

    monkeypatch.setattr(engine_module, "get_bars", lambda *args, **kwargs: bars)
    monkeypatch.setattr(engine_module, "get_strategy", lambda name: _Strategy())
    monkeypatch.setattr(engine_module, "add_analysis", lambda value: value)
    monkeypatch.setattr(engine_module, "record_confluence_shadow", lambda *args, **kwargs: None)

    def _execute(**kwargs) -> ExecutionResult:
        nonlocal execution_calls, broker_submissions
        execution_calls += 1
        if execution_calls == 1:
            return ExecutionResult(
                action_taken=False,
                intent_id=None,
                status="deferred",
                summary="cTrader transport is not connected.",
                mode="demo_enabled",
                retryable=True,
            )
        broker_submissions += 1
        return ExecutionResult(
            action_taken=True,
            intent_id=10,
            status="executed",
            summary="demo order executed after recovery",
            position_id=20,
            mode="demo_enabled",
            broker_position_id=30,
            retryable=False,
        )

    monkeypatch.setattr(engine_module, "execute_paper_signal", _execute)

    first = asyncio.run(V2Engine()._scan_item(_config(), _watch_item(), state))
    assert first == (True, False)
    assert state == {}
    assert broker_submissions == 0

    second = asyncio.run(V2Engine()._scan_item(_config(), _watch_item(), state))
    assert second == (True, True)
    expected_ts = int(bars.index[-1].timestamp())
    assert state["XAUUSD|M5"] == expected_ts
    assert broker_submissions == 1

    third = asyncio.run(V2Engine()._scan_item(_config(), _watch_item(), state))
    assert third == (False, False)
    assert execution_calls == 2
    assert broker_submissions == 1


def test_flat_disconnect_status_is_actionable_and_clears_after_recovery() -> None:
    config = _config().model_copy(
        update={
            "watchlist": [_watch_item()],
        }
    )
    runtime = EngineRuntime(
        running=True,
        loop_active=True,
        ollama_ready=True,
        active_watchlist=["XAUUSD:M5"],
    )
    disconnected = BrokerStatus(
        connected=False,
        socket_connected=False,
        account_authorized=True,
        symbols_loaded=6460,
        open_positions=0,
        pending_orders=0,
        ready=False,
        market_data_ready=False,
        broker_mode="demo",
        account_type="demo",
        demo_account_confirmed=True,
        execution_ready=False,
        notes=["cTrader transport is not connected."],
    )

    incidents = router_module._active_status_incidents(config, disconnected, runtime)
    truth = {item.name: item for item in router_module._status_truth_checks(config, disconnected, runtime)}

    assert [(item.level, item.code, item.message) for item in incidents] == [
        ("warning", "broker_disconnected", "cTrader socket is currently disconnected.")
    ]
    assert truth["connected"].ok is False
    assert truth["execution_ready"].ok is False

    recovered = disconnected.model_copy(
        update={
            "connected": True,
            "socket_connected": True,
            "account_authorized": True,
            "ready": True,
            "market_data_ready": True,
            "execution_ready": True,
            "notes": [],
        }
    )
    recovered_incidents = router_module._active_status_incidents(config, recovered, runtime)
    recovered_truth = {item.name: item for item in router_module._status_truth_checks(config, recovered, runtime)}

    assert all(item.code != "broker_disconnected" for item in recovered_incidents)
    assert recovered_truth["connected"].ok is True
    assert recovered_truth["execution_ready"].ok is True
