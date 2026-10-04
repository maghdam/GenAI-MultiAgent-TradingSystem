from __future__ import annotations

import asyncio
import pytest

from backend import ctrader_client as ctd
from backend.adapters.ctrader import CTraderBrokerAdapter
from backend.api import router as router_module
from backend.domain.models import BrokerAccountSnapshot, BrokerStatus, EngineConfig, EngineRuntime


def _verified_snapshot(account_id: int = 47139918) -> BrokerAccountSnapshot:
    return BrokerAccountSnapshot(
        account_id=account_id,
        currency="CHF",
        balance=1000.0,
        unrealized_pnl=0.0,
        equity=1000.0,
        money_digits=2,
        source="ctrader",
        verified=True,
    )


def test_engine_config_migrates_legacy_demo_autotrade_and_drops_live_bypass() -> None:
    config = EngineConfig.model_validate(
        {
            "demo_autotrade": True,
            "allow_live": True,
            "selected_ctrader_account_id": 47139918,
            "selected_ctrader_account_type": "live",
        }
    )

    payload = config.model_dump()

    assert config.ctrader_autotrade is True
    assert config.demo_autotrade is True
    assert payload["ctrader_autotrade"] is True
    assert "demo_autotrade" not in payload
    assert "allow_live" not in payload


def test_live_account_is_generically_confirmed_without_becoming_demo(monkeypatch) -> None:
    monkeypatch.setattr(ctd, "CONNECTED", True)
    monkeypatch.setattr(ctd, "AUTHORIZED", True)
    monkeypatch.setattr(ctd, "ACCOUNT_ID", 47139918)
    monkeypatch.setattr(ctd, "ACTIVE_ACCOUNT_ID", 47139918)
    monkeypatch.setattr(ctd, "ACTIVE_HOST_TYPE", "live")
    monkeypatch.setattr(ctd, "ACCOUNT_IS_DEMO", False)

    assert ctd.is_account_confirmed() is True
    assert ctd.is_demo_account_confirmed() is False


def test_verified_live_account_can_use_generic_market_order_path(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(ctd, "is_connected", lambda: True)
    monkeypatch.setattr(ctd, "is_authorized", lambda: True)
    monkeypatch.setattr(ctd, "is_account_confirmed", lambda: True)
    monkeypatch.setattr(ctd, "is_symbol_metadata_ready", lambda: True)
    monkeypatch.setattr(ctd, "get_account_verification_error", lambda: None)
    monkeypatch.setattr(ctd, "get_active_host_type", lambda: "live")
    monkeypatch.setattr(ctd, "symbol_name_to_id", {"XAUUSD": 7})
    monkeypatch.setattr(ctd, "symbol_lot_size_map", {7: 100.0})
    monkeypatch.setattr(ctd, "symbol_min_volume_map", {7: 100})
    monkeypatch.setattr(ctd, "symbol_step_volume_map", {7: 100})
    monkeypatch.setattr(ctd, "symbol_max_volume_map", {7: 500_000})
    monkeypatch.setattr(ctd, "ACCOUNT_ID", 47139918)
    monkeypatch.setattr(ctd, "volume_lots_to_units", lambda symbol_id, lots: 250)
    monkeypatch.setattr(
        ctd,
        "place_order",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    monkeypatch.setattr(
        ctd,
        "wait_for_deferred",
        lambda deferred, timeout: {
            "status": "executed",
            "position_id": 900001,
            "ack": {"ok": True},
        },
    )

    result = CTraderBrokerAdapter().place_market_order(
        symbol="XAUUSD",
        direction="long",
        quantity_lots=0.025,
        stop_loss=99.0,
        take_profit=102.0,
    )

    assert captured["account_id"] == 47139918
    assert captured["symbol_id"] == 7
    assert captured["side"] == "BUY"
    assert result["account_type"] == "live"
    assert result["position_id"] == 900001


def test_broker_status_marks_verified_live_account_execution_ready(monkeypatch) -> None:
    adapter = CTraderBrokerAdapter()
    monkeypatch.setattr(adapter, "connected", lambda: True)
    monkeypatch.setattr(ctd, "is_authorized", lambda: True)
    monkeypatch.setattr(ctd, "get_auth_error", lambda: None)
    monkeypatch.setattr(ctd, "is_account_confirmed", lambda: True)
    monkeypatch.setattr(ctd, "is_symbol_metadata_ready", lambda: True)
    monkeypatch.setattr(ctd, "symbol_name_to_id", {"XAUUSD": 7})
    monkeypatch.setattr(ctd, "get_reconcile_snapshot", lambda: {"positions": [], "orders": []})
    monkeypatch.setattr(ctd, "is_account_switch_in_progress", lambda: False)
    monkeypatch.setattr(ctd, "get_account_switch_target_id", lambda: None)
    monkeypatch.setattr(ctd, "get_account_switch_error", lambda: None)
    monkeypatch.setattr(ctd, "get_active_account_id", lambda: 47139918)
    monkeypatch.setattr(ctd, "get_active_host_type", lambda: "live")
    monkeypatch.setattr(ctd, "get_account_verification_error", lambda: None)
    monkeypatch.setattr(ctd, "get_last_auth_attempt", lambda: None)
    monkeypatch.setattr(adapter, "get_account_snapshot", lambda force=False: _verified_snapshot())

    status = adapter.get_status()

    assert status.account_id == 47139918
    assert status.account_type == "live"
    assert status.account_verified is True
    assert status.demo_account_confirmed is False
    assert status.execution_ready is True


def test_status_payload_reports_live_enabled_for_shared_ctrader_execution(monkeypatch) -> None:
    config = EngineConfig(
        ctrader_autotrade=True,
        selected_ctrader_account_id=47139918,
        selected_ctrader_account_type="live",
    )
    broker = BrokerStatus(
        connected=True,
        socket_connected=True,
        account_authorized=True,
        symbols_loaded=1,
        open_positions=0,
        pending_orders=0,
        ready=True,
        market_data_ready=True,
        broker_mode="live",
        account_id=47139918,
        account_type="live",
        active_host_type="live",
        account_verified=True,
        demo_account_confirmed=False,
        execution_ready=True,
        account_snapshot=_verified_snapshot(),
    )

    monkeypatch.setattr(router_module, "_current_config", lambda: config)
    monkeypatch.setattr(router_module, "get_broker_status", lambda: broker)
    monkeypatch.setattr(router_module, "build_readiness", lambda cfg: [])
    monkeypatch.setattr(router_module, "list_strategies", lambda: [])
    monkeypatch.setattr(router_module, "list_paper_positions", lambda status=None: [])
    monkeypatch.setattr(router_module, "attach_broker_truth", lambda positions, rows=None: positions)
    monkeypatch.setattr(router_module, "list_incidents", lambda limit: [])
    monkeypatch.setattr(router_module, "list_recent_analyses", lambda limit: [])
    monkeypatch.setattr(router_module, "list_paper_events", lambda limit: [])
    monkeypatch.setattr(router_module, "list_order_intents", lambda limit: [])
    monkeypatch.setattr(router_module, "list_trade_audits", lambda limit: [])
    monkeypatch.setattr(router_module, "list_decision_records", lambda limit: [])
    monkeypatch.setattr(router_module, "list_confluence_shadows", lambda limit: [])
    runtime = EngineRuntime(
        loop_active=False,
        active_watchlist=[],
        ollama_ready=True,
        last_error=None,
    )
    monkeypatch.setattr(router_module, "load_runtime", lambda: runtime)

    async def _ready():
        return True

    monkeypatch.setattr(router_module, "_get_cached_ollama_ready", _ready)

    status = asyncio.run(router_module._status_payload())

    assert status.mode == "live_enabled"
    assert status.broker.account_verified is True
    assert status.config.ctrader_autotrade is True


def test_config_api_accepts_live_selected_account_without_allow_live_flag(monkeypatch) -> None:
    saved = {}
    monkeypatch.setattr(
        router_module,
        "save_engine_config",
        lambda config: saved.setdefault("config", config) or config,
    )
    monkeypatch.setattr(router_module.engine, "wake", lambda: None)

    config = EngineConfig(
        ctrader_autotrade=True,
        selected_ctrader_account_id=47139918,
        selected_ctrader_account_type="live",
    )
    result = asyncio.run(router_module.v2_set_config(config))

    assert result.ctrader_autotrade is True
    assert result.selected_ctrader_account_type == "live"
    assert "allow_live" not in result.model_dump()
