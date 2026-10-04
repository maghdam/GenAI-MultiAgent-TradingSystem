from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

import backend.ctrader_client as ctd
from backend.adapters.ctrader import adapter
from backend.api import router as router_module
from backend.domain.models import (
    BrokerAccountSnapshot,
    EngineConfig,
    EngineRuntime,
    StrategyAnalysis,
    WatchlistItem,
)
from backend.services.execution_engine import execute_paper_signal
from backend.storage.repositories import (
    list_incidents,
    list_order_intents,
    list_paper_positions,
)


@pytest.fixture(autouse=True)
def isolated_ctrader_symbol_state(monkeypatch):
    for name in (
        "symbol_map",
        "symbol_name_to_id",
        "symbol_digits_map",
        "symbol_money_digits_map",
        "symbol_min_volume_map",
        "symbol_step_volume_map",
        "symbol_max_volume_map",
        "symbol_lot_size_map",
        "symbol_min_verified",
        "symbol_step_verified",
    ):
        monkeypatch.setattr(ctd, name, {})

    monkeypatch.setattr(ctd, "CONNECTED", True)
    monkeypatch.setattr(ctd, "AUTHORIZED", True)
    monkeypatch.setattr(ctd, "AUTH_ERROR", None)
    monkeypatch.setattr(ctd, "ACCOUNT_ID", 123)
    monkeypatch.setattr(ctd, "ACTIVE_ACCOUNT_ID", 123)
    monkeypatch.setattr(ctd, "HOST_TYPE", "demo")
    monkeypatch.setattr(ctd, "CLIENT_HOST_TYPE", "demo")
    monkeypatch.setattr(ctd, "ACTIVE_HOST_TYPE", "demo")
    monkeypatch.setattr(ctd, "ACCOUNT_IS_DEMO", True)
    monkeypatch.setattr(ctd, "ACCOUNT_VERIFICATION_ERROR", None)
    monkeypatch.setattr(ctd, "ACCOUNT_SWITCH_IN_PROGRESS", False)
    monkeypatch.setattr(ctd, "ACCOUNT_SWITCH_TARGET_ID", None)
    monkeypatch.setattr(ctd, "ACCOUNT_SWITCH_ERROR", None)
    monkeypatch.setattr(ctd, "SYMBOL_METADATA_READY", False)
    monkeypatch.setattr(ctd, "FALLBACK_SYMBOLS", ["XAUUSD", "EURUSD"])

    monkeypatch.setattr(adapter, "_symbol_cache", [])
    monkeypatch.setattr(adapter, "_symbol_cache_count", 0)


def _verified_account_snapshot() -> BrokerAccountSnapshot:
    return BrokerAccountSnapshot(
        account_id=123,
        currency="USD",
        balance=10_000.0,
        equity=10_000.0,
        source="ctrader",
        verified=True,
        as_of=datetime.now(UTC),
    )


def _install_light_xau_symbol() -> None:
    ctd.symbol_map[1] = "XAUUSD"
    ctd.symbol_name_to_id["XAUUSD"] = 1
    ctd.symbol_digits_map[1] = 2
    ctd.symbol_min_volume_map[1] = 100
    ctd.symbol_step_volume_map[1] = 100
    ctd.symbol_max_volume_map[1] = 500_000


def _load_full_xau_contract(monkeypatch) -> None:
    payload = SimpleNamespace(
        symbol=[
            SimpleNamespace(
                symbolId=1,
                digits=2,
                lotSize=10_000,
                minVolume=100,
                stepVolume=100,
                maxVolume=500_000,
            )
        ]
    )
    monkeypatch.setattr(ctd.Protobuf, "extract", lambda _: payload)
    ctd.symbol_details_response_cb(object())


def test_disconnect_invalidates_stale_symbol_contract_metadata() -> None:
    _install_light_xau_symbol()
    ctd.symbol_lot_size_map[1] = 100.0
    ctd.symbol_min_verified[1] = True
    ctd.symbol_step_verified[1] = True
    ctd.SYMBOL_METADATA_READY = True

    ctd._on_disconnected(ctd.client, "test metadata delay")

    assert ctd.CONNECTED is False
    assert ctd.AUTHORIZED is False
    assert ctd.SYMBOL_METADATA_READY is False
    assert ctd.symbol_map == {}
    assert ctd.symbol_name_to_id == {}
    assert ctd.symbol_lot_size_map == {}
    assert ctd.symbol_min_volume_map == {}
    assert ctd.symbol_step_volume_map == {}
    assert ctd.symbol_max_volume_map == {}


def test_fallback_symbols_are_never_treated_as_verified_broker_contracts(monkeypatch) -> None:
    ctd._install_fallback_symbols("symbol details delayed")

    assert ctd.symbol_name_to_id
    assert ctd.is_symbol_metadata_ready() is False

    ready, reason = adapter.demo_symbol_execution_readiness("XAUUSD")
    limits = adapter.get_symbol_limits("XAUUSD")
    spec = adapter.get_instrument_spec("XAUUSD", "USD")

    assert ready is False
    assert "current cTrader session" in reason
    assert limits.source == "fallback"
    assert limits.hard_min is False
    assert limits.hard_step is False
    assert spec.verified is False
    assert spec.source == "fallback"
    assert "current cTrader session" in spec.notes[0]

    monkeypatch.setattr(
        ctd,
        "place_order",
        lambda **kwargs: pytest.fail("broker submission must not run while metadata is delayed"),
    )
    with pytest.raises(RuntimeError, match="symbol contract metadata"):
        adapter.place_demo_market_order(
            symbol="XAUUSD",
            direction="long",
            quantity_lots=0.1,
        )


def test_status_reports_delayed_metadata_as_not_ready_and_actionable(monkeypatch) -> None:
    ctd._install_fallback_symbols("symbol details delayed")
    monkeypatch.setattr(
        ctd,
        "get_reconcile_snapshot",
        lambda: {"positions": [], "orders": [], "error": None},
    )
    monkeypatch.setattr(adapter, "get_account_snapshot", lambda force=False: _verified_account_snapshot())

    status = adapter.get_status()

    assert status.socket_connected is True
    assert status.account_authorized is True
    assert status.demo_account_confirmed is True
    assert status.symbols_loaded == 0
    assert status.ready is False
    assert status.execution_ready is False
    assert any("current cTrader session" in note for note in status.notes)

    incidents = router_module._active_status_incidents(
        EngineConfig(enabled=True, demo_autotrade=True, kill_switch=False),
        status,
        EngineRuntime(
            running=True,
            loop_active=True,
            ollama_ready=True,
            active_watchlist=["XAUUSD:M5"],
        ),
    )
    metadata_incident = next(item for item in incidents if item.code == "symbol_metadata_unavailable")
    assert "demo execution is blocked" in metadata_incident.message
    assert "contract load completes" in metadata_incident.message


def test_demo_execution_defers_without_intent_or_order_while_metadata_is_delayed(monkeypatch) -> None:
    ctd._install_fallback_symbols("symbol details delayed")
    monkeypatch.setattr(
        "backend.services.execution_engine.place_demo_market_order",
        lambda **kwargs: pytest.fail("order submission must not run while metadata is delayed"),
    )

    result = execute_paper_signal(
        config=EngineConfig(
            enabled=True,
            demo_autotrade=True,
            paper_autotrade=False,
            kill_switch=False,
            default_symbol="XAUUSD",
            default_timeframe="M5",
        ),
        watch_item=WatchlistItem(
            symbol="XAUUSD",
            timeframe="M5",
            strategy="sma_cross",
            enabled=True,
            trading_enabled=True,
            lot_size=0.1,
        ),
        analysis=StrategyAnalysis(
            symbol="XAUUSD",
            timeframe="M5",
            strategy="sma_cross",
            signal="long",
            confidence=0.85,
            entry_price=100.0,
            stop_loss=99.0,
            take_profit=102.0,
            reasons=["metadata-delay resilience test"],
        ),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC),
        bar_snapshot={"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0},
    )

    assert result.action_taken is False
    assert result.status == "deferred"
    assert result.retryable is True
    assert result.intent_id is None
    assert "current cTrader session" in result.summary
    assert list_paper_positions("open") == []
    assert list_order_intents(20) == []

    incidents = list_incidents(10)
    assert incidents[0].code == "ctrader_demo_symbol_not_ready"
    assert incidents[0].details["retryable"] is True


def test_full_contract_response_deterministically_restores_execution_readiness(monkeypatch) -> None:
    _install_light_xau_symbol()
    assert ctd.is_symbol_metadata_ready() is False

    before_ready, _ = adapter.demo_symbol_execution_readiness("XAUUSD")
    assert before_ready is False

    _load_full_xau_contract(monkeypatch)

    assert ctd.is_symbol_metadata_ready() is True
    ready, reason = adapter.demo_symbol_execution_readiness("XAUUSD")
    assert ready is True
    assert reason == "Broker symbol contract metadata is ready."

    limits = adapter.get_symbol_limits("XAUUSD")
    spec = adapter.get_instrument_spec("XAUUSD", "USD")

    assert limits.source == "broker"
    assert limits.hard_min is True
    assert limits.hard_step is True
    assert limits.min_lots == pytest.approx(0.01)
    assert spec.source == "ctrader_contract"
    assert spec.verified is True
    assert spec.lot_size_units == pytest.approx(100.0)

    monkeypatch.setattr(
        ctd,
        "get_reconcile_snapshot",
        lambda: {"positions": [], "orders": [], "error": None},
    )
    monkeypatch.setattr(adapter, "get_account_snapshot", lambda force=False: _verified_account_snapshot())

    status = adapter.get_status()
    assert status.symbols_loaded == 1
    assert status.ready is True
    assert status.execution_ready is True


def test_symbol_list_cache_cannot_leak_across_metadata_sessions(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_symbol_cache", ["STALE_SYMBOL"])
    monkeypatch.setattr(adapter, "_symbol_cache_count", 1)

    ctd._install_fallback_symbols("new session metadata pending")
    assert adapter.list_symbols() == ["EURUSD", "XAUUSD"]
    assert adapter._symbol_cache == []
    assert adapter._symbol_cache_count == 0

    ctd._clear_symbol_metadata()
    _install_light_xau_symbol()
    _load_full_xau_contract(monkeypatch)

    assert adapter.list_symbols() == ["XAUUSD"]
    assert adapter._symbol_cache == ["XAUUSD"]
