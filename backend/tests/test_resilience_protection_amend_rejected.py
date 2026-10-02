from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

import backend.ctrader_client as ctd
from backend.adapters.ctrader import CTraderBrokerAdapter, DemoProtectionSyncFailure
from backend.domain.models import (
    BrokerAccountSnapshot,
    BrokerStatus,
    EngineConfig,
    StrategyAnalysis,
    WatchlistItem,
)
from backend.services import engine as engine_module
from backend.services import execution_engine
from backend.services import protection_safety
from backend.services import reconciler as reconciler_module
from backend.services.engine import V2Engine
from backend.services.execution_engine import execute_paper_signal
from backend.services.reconciler import reconcile_open_positions
from backend.storage.repositories import (
    list_incidents,
    list_order_intents,
    list_paper_positions,
    open_paper_position,
    save_engine_config,
)


def _config() -> EngineConfig:
    return EngineConfig(
        enabled=True,
        paper_autotrade=False,
        demo_autotrade=True,
        allow_live=False,
        kill_switch=False,
        require_stops=True,
        cooldown_minutes=0,
        risk_per_trade_pct=0,
        watchlist=[_watch()],
    )


def _watch() -> WatchlistItem:
    return WatchlistItem(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        enabled=True,
        trading_enabled=True,
        lot_size=0.10,
        params={},
    )


def _analysis(
    *,
    signal: str = "no_trade",
    stop_loss: float = 99.0,
    take_profit: float = 102.0,
) -> StrategyAnalysis:
    return StrategyAnalysis(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        signal=signal,
        confidence=0.90,
        entry_price=100.0,
        stop_loss=stop_loss,
        take_profit=take_profit,
        reasons=["protection-amend resilience test"],
    )


def _open_position():
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


def _broker_row(*, stop_loss=None, take_profit=None):
    return {
        "position_id": 111,
        "symbol": "XAUUSD",
        "direction": "buy",
        "volume_lots": 0.10,
        "entry_price": 100.0,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
    }


def _healthy_broker() -> BrokerStatus:
    return BrokerStatus(
        connected=True,
        socket_connected=True,
        account_authorized=True,
        symbols_loaded=200,
        open_positions=1,
        pending_orders=0,
        ready=True,
        market_data_ready=True,
        broker_mode="demo",
        account_type="demo",
        demo_account_confirmed=True,
        execution_ready=True,
        notes=[],
    )


def _sync_failure(kind: str = "amend_rejected") -> DemoProtectionSyncFailure:
    return DemoProtectionSyncFailure(
        "cTrader demo target sync rejected: TRADING_BAD_STOPS",
        failure_kind=kind,
        broker_position_id=111,
        ack={"status": "order_rejected", "reject_reason": "TRADING_BAD_STOPS"},
    )


def _verified_account() -> BrokerAccountSnapshot:
    return BrokerAccountSnapshot(
        account_id=123,
        currency="CHF",
        balance=20_000.0,
        unrealized_pnl=0.0,
        equity=20_000.0,
        money_digits=2,
        deposit_asset_id=7,
        source="ctrader",
        verified=True,
    )


def test_adapter_classifies_rejected_protection_amend(monkeypatch) -> None:
    monkeypatch.setattr(ctd, "is_demo_account_confirmed", lambda: True)
    monkeypatch.setattr(ctd, "get_account_verification_error", lambda: None)
    monkeypatch.setattr(ctd, "symbol_name_to_id", {"XAUUSD": 7})
    monkeypatch.setattr(ctd, "symbol_digits_map", {7: 2})
    monkeypatch.setattr(ctd, "ACCOUNT_ID", 123)
    monkeypatch.setattr(
        ctd,
        "get_open_positions",
        lambda: [
            {
                "symbol_name": "XAUUSD",
                "symbol_id": 7,
                "position_id": 111,
                "direction": "buy",
                "volume_lots": 0.10,
                "entry_price": 100.0,
                "stop_loss": None,
                "take_profit": None,
            }
        ],
    )
    monkeypatch.setattr(ctd, "modify_position_sltp", lambda **kwargs: object())
    monkeypatch.setattr(
        ctd,
        "wait_for_deferred",
        lambda deferred, timeout: {
            "status": "order_rejected",
            "reject_reason": "TRADING_BAD_STOPS",
        },
    )

    with pytest.raises(DemoProtectionSyncFailure) as exc_info:
        CTraderBrokerAdapter().sync_demo_position_targets(
            symbol="XAUUSD",
            direction="long",
            stop_loss=99.0,
            take_profit=102.0,
            position_id=111,
        )

    assert exc_info.value.failure_kind == "amend_rejected"
    assert exc_info.value.broker_position_id == 111
    assert exc_info.value.ack["reject_reason"] == "TRADING_BAD_STOPS"


def test_rejected_existing_protection_amend_closes_canonical_position_once(monkeypatch) -> None:
    opened = _open_position()
    monkeypatch.setattr(execution_engine, "_refresh_open_position", lambda *args, **kwargs: opened)
    monkeypatch.setattr(
        execution_engine,
        "sync_demo_position_targets",
        lambda **kwargs: (_ for _ in ()).throw(_sync_failure()),
    )
    monkeypatch.setattr(execution_engine, "list_positions", lambda: [_broker_row()])
    close_calls = []
    monkeypatch.setattr(
        protection_safety,
        "close_demo_position",
        lambda **kwargs: close_calls.append(kwargs)
        or {"status": "closed", "position_id": 111, "verified": True},
    )
    monkeypatch.setattr(
        execution_engine,
        "place_demo_market_order",
        lambda **kwargs: pytest.fail("protection failure must never create a competing open order"),
    )

    result = execute_paper_signal(
        config=_config(),
        watch_item=_watch(),
        analysis=_analysis(),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC).replace(tzinfo=None),
        bar_snapshot={"open": 99.8, "high": 100.2, "low": 99.6, "close": 100.0},
    )

    assert result.action_taken is True
    assert result.status == "failed"
    assert result.retryable is False
    assert close_calls == [{"symbol": "XAUUSD", "position_id": 111, "quantity_lots": 0.1}]
    assert list_paper_positions("open") == []
    closed = list_paper_positions("closed")
    assert len(closed) == 1
    assert closed[0].id == opened.id
    assert closed[0].stop_loss == 99.0
    assert closed[0].take_profit == 102.0
    assert closed[0].close_reason == "broker_protection_unverified_failsafe"
    assert list_order_intents(10) == []

    codes = [item.code for item in list_incidents(20)]
    assert "ctrader_demo_protection_unverified" in codes
    assert "ctrader_demo_protection_failsafe_closed" in codes


def test_rejected_amend_and_failed_failsafe_close_retains_tracker_nonretryable(monkeypatch) -> None:
    opened = _open_position()
    monkeypatch.setattr(execution_engine, "_refresh_open_position", lambda *args, **kwargs: opened)
    monkeypatch.setattr(
        execution_engine,
        "sync_demo_position_targets",
        lambda **kwargs: (_ for _ in ()).throw(_sync_failure()),
    )
    monkeypatch.setattr(execution_engine, "list_positions", lambda: [_broker_row()])
    monkeypatch.setattr(
        protection_safety,
        "close_demo_position",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("broker close rejected")),
    )

    result = execute_paper_signal(
        config=_config(),
        watch_item=_watch(),
        analysis=_analysis(),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC).replace(tzinfo=None),
        bar_snapshot={"open": 99.8, "high": 100.2, "low": 99.6, "close": 100.0},
    )

    assert result.action_taken is False
    assert result.status == "protection_failsafe_pending"
    assert result.retryable is False
    assert result.position_id == opened.id

    remaining = list_paper_positions("open")
    assert len(remaining) == 1
    assert remaining[0].id == opened.id
    assert remaining[0].broker_position_id == 111
    assert remaining[0].stop_loss == 99.0
    assert remaining[0].take_profit == 102.0

    incidents = list_incidents(20)
    failure = next(item for item in incidents if item.code == "ctrader_demo_protection_failsafe_close_failed")
    assert failure.details["tracking_retained"] is True
    assert "Do not submit a competing open order" in failure.details["action_required"]


def test_rejected_new_targets_never_replace_local_targets(monkeypatch) -> None:
    opened = _open_position()
    monkeypatch.setattr(execution_engine, "_refresh_open_position", lambda *args, **kwargs: opened)
    monkeypatch.setattr(execution_engine, "get_demo_symbol_execution_readiness", lambda symbol: (True, "ready"))
    monkeypatch.setattr(execution_engine, "get_broker_account_snapshot", _verified_account)
    monkeypatch.setattr(execution_engine, "list_positions", lambda: [_broker_row(stop_loss=99.0, take_profit=102.0)])
    sync_calls = []

    def _sync(**kwargs):
        sync_calls.append((kwargs["stop_loss"], kwargs["take_profit"]))
        if len(sync_calls) == 1:
            return {"status": "already_synced", "verified": True, "position_id": 111}
        raise _sync_failure()

    monkeypatch.setattr(execution_engine, "sync_demo_position_targets", _sync)
    monkeypatch.setattr(
        protection_safety,
        "close_demo_position",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fail-safe close rejected")),
    )

    result = execute_paper_signal(
        config=_config(),
        watch_item=_watch(),
        analysis=_analysis(signal="long", stop_loss=98.5, take_profit=103.0),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC).replace(tzinfo=None),
        bar_snapshot={"open": 99.8, "high": 100.2, "low": 99.6, "close": 100.0},
    )

    assert sync_calls == [(99.0, 102.0), (98.5, 103.0)]
    assert result.status == "protection_failsafe_pending"
    assert result.retryable is False

    remaining = list_paper_positions("open")
    assert len(remaining) == 1
    assert remaining[0].stop_loss == 99.0
    assert remaining[0].take_profit == 102.0

    intent = list_order_intents(10)[0]
    assert intent.intent_type == "update"
    assert intent.status == "failed"
    assert intent.details["local_targets_updated"] is False
    assert intent.details["protection_failure_kind"] == "amend_rejected"


def test_same_bar_pending_failure_suppresses_duplicate_amend_and_recovers_from_broker_truth(monkeypatch) -> None:
    opened = _open_position()
    engine = V2Engine()
    engine._protection_failsafe_pending_positions.add(opened.id)
    broker_row = _broker_row()

    monkeypatch.setattr(engine_module, "get_broker_status", _healthy_broker)
    monkeypatch.setattr(engine_module, "list_positions", lambda: [dict(broker_row)])
    monkeypatch.setattr(
        engine_module,
        "reconcile_open_demo_position_ledger",
        lambda position, row: {"status": "unchanged"},
    )
    monkeypatch.setattr(
        engine_module,
        "sync_demo_position_targets",
        lambda **kwargs: pytest.fail("same-bar pending state must not repeat protection amend"),
    )
    monkeypatch.setattr(
        engine_module,
        "close_demo_position",
        lambda **kwargs: pytest.fail("same-bar pending state must not issue another close"),
    )

    engine._sync_existing_demo_protection(_config(), _watch(), 100.0)
    assert opened.id in engine._protection_failsafe_pending_positions

    broker_row["stop_loss"] = 99.0
    broker_row["take_profit"] = 102.0
    engine._sync_existing_demo_protection(_config(), _watch(), 100.0)

    assert opened.id not in engine._protection_failsafe_pending_positions
    recovery = next(
        item
        for item in list_incidents(20)
        if item.code == "ctrader_demo_protection_recovered"
    )
    assert recovery.details["broker_position_id"] == 111
    assert recovery.details["amend_suppressed"] is True


def test_startup_reconciliation_applies_same_failsafe_close_policy(monkeypatch) -> None:
    save_engine_config(_config())
    opened = _open_position()
    row = _broker_row()

    monkeypatch.setattr(reconciler_module, "recover_demo_broker_trackers", lambda config: {"recovered": 0})
    monkeypatch.setattr(
        reconciler_module,
        "get_broker_status",
        lambda: type("Status", (), {"execution_ready": True})(),
    )
    monkeypatch.setattr(reconciler_module, "list_positions", lambda: [row])
    monkeypatch.setattr(
        reconciler_module,
        "get_bars",
        lambda symbol, timeframe, bars: pd.DataFrame(
            [{"close": 100.0}],
            index=pd.to_datetime(["2026-10-02T06:00:00Z"], utc=True),
        ),
    )
    monkeypatch.setattr(
        reconciler_module,
        "reconcile_open_demo_position_ledger",
        lambda position, broker_row: {"status": "unchanged"},
    )
    monkeypatch.setattr(
        reconciler_module,
        "sync_demo_position_targets",
        lambda **kwargs: (_ for _ in ()).throw(_sync_failure()),
    )
    close_calls = []
    monkeypatch.setattr(
        protection_safety,
        "close_demo_position",
        lambda **kwargs: close_calls.append(kwargs)
        or {"status": "closed", "position_id": 111, "verified": True},
    )

    summary = reconcile_open_positions(reason="startup")

    assert summary["checked"] == 1
    assert summary["closed"] == 1
    assert summary["skipped"] == 0
    assert close_calls == [{"symbol": "XAUUSD", "position_id": 111, "quantity_lots": 0.1}]
    assert list_paper_positions("open") == []
    assert list_paper_positions("closed")[0].id == opened.id


def test_generic_precondition_failure_does_not_trigger_failsafe_close(monkeypatch) -> None:
    opened = _open_position()
    monkeypatch.setattr(execution_engine, "_refresh_open_position", lambda *args, **kwargs: opened)
    monkeypatch.setattr(
        execution_engine,
        "sync_demo_position_targets",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("cTrader transport is not connected")),
    )
    monkeypatch.setattr(
        protection_safety,
        "close_demo_position",
        lambda **kwargs: pytest.fail("precondition failure must not trigger protection fail-safe close"),
    )

    result = execute_paper_signal(
        config=_config(),
        watch_item=_watch(),
        analysis=_analysis(),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC).replace(tzinfo=None),
        bar_snapshot={"open": 99.8, "high": 100.2, "low": 99.6, "close": 100.0},
    )

    assert result.action_taken is False
    assert result.status == "failed"
    assert result.retryable is True
    assert list_paper_positions("open")[0].id == opened.id
