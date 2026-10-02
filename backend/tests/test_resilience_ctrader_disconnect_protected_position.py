from __future__ import annotations

from datetime import UTC, datetime

import pytest

from backend.domain.models import BrokerStatus, EngineConfig, StrategyAnalysis, WatchlistItem
from backend.services import engine as engine_module
from backend.services.engine import V2Engine
from backend.services.execution_engine import execute_paper_signal
from backend.storage.repositories import (
    list_incidents,
    list_order_intents,
    list_paper_positions,
    open_paper_position,
)


def _config() -> EngineConfig:
    return EngineConfig(
        enabled=True,
        paper_autotrade=False,
        demo_autotrade=True,
        kill_switch=False,
        require_stops=True,
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


def _broker(*, connected: bool) -> BrokerStatus:
    return BrokerStatus(
        connected=connected,
        socket_connected=connected,
        account_authorized=connected,
        symbols_loaded=6460,
        open_positions=1 if connected else 0,
        pending_orders=0,
        ready=connected,
        market_data_ready=connected,
        broker_mode="demo",
        account_type="demo" if connected else "unknown",
        demo_account_confirmed=connected,
        execution_ready=connected,
        notes=[] if connected else ["cTrader transport is not connected."],
    )


def _open_managed_position():
    return open_paper_position(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        direction="long",
        quantity=0.10,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        broker_position_id=111,
    )


def test_same_bar_disconnect_preserves_protected_tracker_and_suppresses_broker_actions(monkeypatch) -> None:
    opened = _open_managed_position()
    engine = V2Engine()

    monkeypatch.setattr(engine_module, "get_broker_status", lambda: _broker(connected=False))
    monkeypatch.setattr(
        engine_module,
        "list_positions",
        lambda: pytest.fail("disconnected maintenance must not query broker positions"),
    )
    monkeypatch.setattr(
        engine_module,
        "sync_demo_position_targets",
        lambda **kwargs: pytest.fail("disconnected maintenance must not amend broker protection"),
    )
    monkeypatch.setattr(
        engine_module,
        "attempt_verified_demo_close",
        lambda *args, **kwargs: pytest.fail("disconnected maintenance must not close broker or local position"),
    )

    # The mark is deliberately beyond the stored take-profit. In demo-managed
    # mode a disconnected broker remains the execution source of truth, so the
    # local tracker must not synthesize a protective exit.
    engine._mark_positions(_config(), _watch_item(), 103.0)
    engine._sync_existing_demo_protection(_config(), _watch_item(), 103.0)
    engine._sync_existing_demo_protection(_config(), _watch_item(), 103.0)

    positions = list_paper_positions("open")
    assert len(positions) == 1
    assert positions[0].id == opened.id
    assert positions[0].broker_position_id == 111

    incidents = [
        item
        for item in list_incidents(20)
        if item.code == "ctrader_demo_protection_verification_deferred"
    ]
    assert len(incidents) == 1
    assert incidents[0].details["position_id"] == opened.id
    assert incidents[0].details["broker_position_id"] == 111
    assert incidents[0].details["broker_protection_state"] == "unverified_broker_unavailable"
    assert incidents[0].details["local_exit_suppressed"] is True
    assert incidents[0].details["broker_mutation_suppressed"] is True
    assert incidents[0].details["reason"] == "cTrader transport is not connected."


def test_recovery_reconciles_canonical_broker_position_id_and_resumes_protection(monkeypatch) -> None:
    opened = _open_managed_position()
    engine = V2Engine()
    connected = {"value": False}
    sync_calls = []
    list_calls = []

    monkeypatch.setattr(
        engine_module,
        "get_broker_status",
        lambda: _broker(connected=connected["value"]),
    )

    def _positions():
        list_calls.append(True)
        return [
            {
                "position_id": 222,
                "symbol": "XAUUSD",
                "direction": "buy",
                "volume_lots": 0.10,
                "entry_price": 100.0,
                "stop_loss": 98.0,
                "take_profit": 104.0,
            },
            {
                "position_id": 111,
                "symbol": "XAUUSD",
                "direction": "buy",
                "volume_lots": 0.10,
                "entry_price": 100.0,
                "stop_loss": 99.0,
                "take_profit": 102.0,
            },
        ]

    monkeypatch.setattr(engine_module, "list_positions", _positions)
    monkeypatch.setattr(
        engine_module,
        "reconcile_open_demo_position_ledger",
        lambda position, broker_row: {"status": "unchanged"},
    )
    monkeypatch.setattr(
        engine_module,
        "sync_demo_position_targets",
        lambda **kwargs: sync_calls.append(kwargs)
        or {
            "status": "already_synced",
            "verified": True,
            "position_id": kwargs["position_id"],
        },
    )
    monkeypatch.setattr(
        engine_module,
        "attempt_verified_demo_close",
        lambda *args, **kwargs: pytest.fail("healthy protected position must not be closed on recovery"),
    )

    engine._sync_existing_demo_protection(_config(), _watch_item(), 100.5)
    assert list_calls == []
    assert sync_calls == []

    connected["value"] = True
    engine._sync_existing_demo_protection(_config(), _watch_item(), 100.5)

    assert len(list_calls) == 1
    assert len(sync_calls) == 1
    assert sync_calls[0]["position_id"] == 111
    assert sync_calls[0]["position_id"] != 222

    positions = list_paper_positions("open")
    assert len(positions) == 1
    assert positions[0].id == opened.id
    assert positions[0].broker_position_id == 111


def test_new_bar_disconnect_keeps_open_tracker_and_returns_retryable_without_competing_order(monkeypatch) -> None:
    opened = _open_managed_position()

    monkeypatch.setattr(
        "backend.services.execution_engine.get_broker_status",
        lambda: _broker(connected=False),
    )
    monkeypatch.setattr(
        "backend.services.execution_engine.list_positions",
        lambda: pytest.fail("disconnected new-bar refresh must not query broker positions"),
    )
    monkeypatch.setattr(
        "backend.services.execution_engine.sync_demo_position_targets",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("Demo target sync blocked: cTrader transport is not connected.")
        ),
    )
    monkeypatch.setattr(
        "backend.services.execution_engine.place_demo_market_order",
        lambda **kwargs: pytest.fail("existing disconnected position must not create a competing order"),
    )
    monkeypatch.setattr(
        "backend.services.execution_engine.close_demo_position",
        lambda **kwargs: pytest.fail("disconnected position must not be closed by a competing action"),
    )

    analysis = StrategyAnalysis(
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

    result = execute_paper_signal(
        config=_config(),
        watch_item=_watch_item(),
        analysis=analysis,
        mark_price=103.0,
        bar_timestamp=datetime.now(UTC),
        bar_snapshot={"open": 102.5, "high": 103.2, "low": 102.4, "close": 103.0},
    )

    assert result.action_taken is False
    assert result.status == "failed"
    assert result.retryable is True
    assert result.position_id == opened.id
    assert result.mode == "demo_enabled"
    assert list_order_intents(10) == []

    positions = list_paper_positions("open")
    assert len(positions) == 1
    assert positions[0].id == opened.id
    assert positions[0].broker_position_id == 111

    incidents = list_incidents(10)
    assert incidents[0].code == "ctrader_demo_protection_sync_failed"
    assert "transport is not connected" in incidents[0].details["error"]
