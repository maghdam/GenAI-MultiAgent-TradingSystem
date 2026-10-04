from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from backend.api import router as router_module
from backend.domain.models import (
    BrokerStatus,
    EngineConfig,
    EngineRuntime,
    IncidentRecord,
    WatchlistItem,
)
from backend.services.reconciler import recover_runtime_state
from backend.storage.repositories import load_runtime, save_runtime


def _broker(**overrides) -> BrokerStatus:
    payload = BrokerStatus(
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
        account_verified=True,
        demo_account_confirmed=True,
        execution_ready=True,
        notes=[],
    ).model_dump()
    payload.update(overrides)
    return BrokerStatus(**payload)


def _runtime(**overrides) -> EngineRuntime:
    payload = EngineRuntime(
        running=True,
        loop_active=True,
        ollama_ready=True,
        active_watchlist=["XAUUSD:M5"],
    ).model_dump()
    payload.update(overrides)
    return EngineRuntime(**payload)


def test_status_truth_uses_live_sources_for_all_phase_3_4_flags() -> None:
    checks = router_module._status_truth_checks(
        EngineConfig(enabled=True, kill_switch=False),
        _broker(),
        _runtime(),
    )
    values = {item.name: item.ok for item in checks}

    assert values == {
        "connected": True,
        "account_verified": True,
        "execution_ready": True,
        "symbol_metadata_ready": True,
        "engine_scanning": True,
        "model_ready": True,
    }


def test_engine_scanning_truth_is_runtime_state_not_enabled_config() -> None:
    checks = router_module._status_truth_checks(
        EngineConfig(
            enabled=True,
            kill_switch=False,
            watchlist=[WatchlistItem(symbol="XAUUSD", timeframe="M5", enabled=True)],
        ),
        _broker(),
        _runtime(loop_active=False),
    )
    by_name = {item.name: item for item in checks}

    assert by_name["engine_scanning"].ok is False
    assert "not active" in by_name["engine_scanning"].detail


def test_active_incidents_are_derived_from_current_faults() -> None:
    incidents = router_module._active_status_incidents(
        EngineConfig(
            enabled=True,
            ctrader_autotrade=True,
            kill_switch=False,
            watchlist=[WatchlistItem(symbol="XAUUSD", timeframe="M5", enabled=True)],
        ),
        _broker(
            account_authorized=False,
            auth_error="authorization expired",
            symbols_loaded=0,
            ready=False,
            account_verified=False,
            demo_account_confirmed=False,
            execution_ready=False,
        ),
        _runtime(loop_active=False, ollama_ready=False, last_error="latest scan failed"),
    )
    codes = {item.code for item in incidents}

    assert codes == {
        "broker_not_authorized",
        "symbol_metadata_unavailable",
        "ctrader_account_not_verified",
        "engine_not_scanning",
        "model_not_ready",
        "engine_runtime_error",
    }


def test_intentionally_disabled_broker_startup_is_not_current_incident() -> None:
    incidents = router_module._active_status_incidents(
        EngineConfig(enabled=False),
        _broker(
            connected=False,
            socket_connected=False,
            account_authorized=False,
            symbols_loaded=0,
            ready=False,
            market_data_ready=False,
            account_type="unknown",
            demo_account_confirmed=False,
            execution_ready=False,
            notes=["cTrader: startup disabled by APP_START_CTRADER_ON_BOOT"],
        ),
        _runtime(running=False, loop_active=False, ollama_ready=True, active_watchlist=[]),
    )

    assert all(item.code != "broker_disconnected" for item in incidents)


def test_status_payload_keeps_stale_history_out_of_current_incidents(monkeypatch) -> None:
    config = EngineConfig(enabled=False)
    broker = _broker()
    runtime = _runtime(running=False, loop_active=False, ollama_ready=True, active_watchlist=[])
    stale = IncidentRecord(
        id=99,
        level="error",
        code="old_scan_failure",
        message="Historical failure that has already cleared.",
        details={},
        created_at=datetime.now(UTC).replace(tzinfo=None),
    )

    monkeypatch.setattr(router_module, "_current_config", lambda: config)
    monkeypatch.setattr(router_module, "get_broker_status", lambda: broker)
    monkeypatch.setattr(router_module, "build_readiness", lambda cfg: [])
    monkeypatch.setattr(router_module, "load_runtime", lambda: runtime)
    monkeypatch.setattr(router_module, "list_strategies", lambda: [])
    monkeypatch.setattr(router_module, "list_incidents", lambda limit: [stale])
    monkeypatch.setattr(router_module, "list_recent_analyses", lambda limit: [])
    monkeypatch.setattr(router_module, "list_paper_positions", lambda status=None: [])
    monkeypatch.setattr(router_module, "list_paper_events", lambda limit: [])
    monkeypatch.setattr(router_module, "list_order_intents", lambda limit: [])
    monkeypatch.setattr(router_module, "list_trade_audits", lambda limit: [])
    monkeypatch.setattr(router_module, "list_decision_records", lambda limit: [])
    monkeypatch.setattr(router_module, "list_confluence_shadows", lambda limit: [])

    async def model_ready() -> bool:
        return True

    monkeypatch.setattr(router_module, "_get_cached_ollama_ready", model_ready)

    payload = asyncio.run(router_module._status_payload())

    assert payload.active_incidents == []
    assert [item.code for item in payload.recent_incidents] == ["old_scan_failure"]


def test_runtime_recovery_clears_persisted_stale_last_error() -> None:
    save_runtime(
        EngineRuntime(
            running=True,
            loop_active=True,
            last_error="stale error from the previous backend process",
            active_watchlist=["XAUUSD:M5"],
        )
    )

    recover_runtime_state(
        EngineConfig(
            enabled=True,
            watchlist=[WatchlistItem(symbol="XAUUSD", timeframe="M5", enabled=True)],
        )
    )

    runtime = load_runtime()
    assert runtime.last_error is None
    assert runtime.loop_active is False
