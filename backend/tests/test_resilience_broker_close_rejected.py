from __future__ import annotations

from datetime import UTC, datetime

import pytest

import backend.ctrader_client as ctd
from backend.adapters.ctrader import (
    CTraderBrokerAdapter,
    DemoCloseOutcomeAmbiguous,
    DemoCloseRejected,
)
from backend.api import router as router_module
from backend.domain.models import (
    BrokerAccountSnapshot,
    BrokerStatus,
    EngineConfig,
    EngineRuntime,
    StrategyAnalysis,
    WatchlistItem,
)
from backend.services import close_safety
from backend.services import execution_engine
from backend.services.broker_ledger import close_local_position_after_broker_close
from backend.services.execution_engine import execute_paper_signal
from backend.storage.repositories import (
    list_paper_events,
    list_paper_positions,
    open_paper_position,
)


def _open_position():
    return open_paper_position(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        direction="long",
        quantity=0.25,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        broker_position_id=111,
    )


def _broker_row():
    return {
        "position_id": 111,
        "symbol": "XAUUSD",
        "direction": "buy",
        "volume_lots": 0.25,
        "entry_price": 100.0,
        "stop_loss": 99.0,
        "take_profit": 102.0,
    }


def _healthy_broker() -> BrokerStatus:
    return BrokerStatus(
        connected=True,
        socket_connected=True,
        account_authorized=True,
        symbols_loaded=250,
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
    )


def _watch() -> WatchlistItem:
    return WatchlistItem(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        enabled=True,
        trading_enabled=True,
        lot_size=0.25,
    )


def _analysis() -> StrategyAnalysis:
    return StrategyAnalysis(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        signal="long",
        confidence=0.90,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        reasons=["broker-close resilience test"],
    )


def _mock_demo_ready(monkeypatch) -> None:
    monkeypatch.setattr(
        execution_engine,
        "get_demo_symbol_execution_readiness",
        lambda symbol: (True, "ready"),
    )
    monkeypatch.setattr(
        execution_engine,
        "get_broker_account_snapshot",
        lambda: BrokerAccountSnapshot(
            account_id=123,
            currency="CHF",
            balance=20_000.0,
            unrealized_pnl=0.0,
            equity=20_000.0,
            money_digits=2,
            deposit_asset_id=7,
            source="ctrader",
            verified=True,
        ),
    )
    monkeypatch.setattr(execution_engine, "sleep", lambda _: None)


def test_adapter_classifies_explicit_close_rejection(monkeypatch) -> None:
    monkeypatch.setattr(ctd, "is_demo_account_confirmed", lambda: True)
    monkeypatch.setattr(ctd, "get_account_verification_error", lambda: None)
    monkeypatch.setattr(ctd, "symbol_name_to_id", {"XAUUSD": 7})
    monkeypatch.setattr(ctd, "ACCOUNT_ID", 123)
    monkeypatch.setattr(ctd, "close_position", lambda **kwargs: object())
    monkeypatch.setattr(
        ctd,
        "wait_for_deferred",
        lambda deferred, timeout: {
            "status": "order_rejected",
            "reject_reason": "TRADING_BAD_VOLUME",
        },
    )

    with pytest.raises(DemoCloseRejected) as exc_info:
        CTraderBrokerAdapter().close_demo_position(
            symbol="XAUUSD",
            position_id=111,
            quantity_lots=0.25,
        )

    assert exc_info.value.broker_position_id == 111
    assert exc_info.value.submitted is True
    assert exc_info.value.ambiguous is False
    assert exc_info.value.ack["reject_reason"] == "TRADING_BAD_VOLUME"


def test_adapter_timeout_reconciles_to_verified_close_when_position_disappears(monkeypatch) -> None:
    broker = CTraderBrokerAdapter()
    monkeypatch.setattr(ctd, "is_demo_account_confirmed", lambda: True)
    monkeypatch.setattr(ctd, "get_account_verification_error", lambda: None)
    monkeypatch.setattr(ctd, "symbol_name_to_id", {"XAUUSD": 7})
    monkeypatch.setattr(ctd, "ACCOUNT_ID", 123)
    monkeypatch.setattr(ctd, "close_position", lambda **kwargs: object())
    monkeypatch.setattr(
        ctd,
        "wait_for_deferred",
        lambda deferred, timeout: {"status": "failed", "error": "deferred timeout"},
    )
    monkeypatch.setattr(ctd, "get_open_positions", lambda: [])
    monkeypatch.setattr(broker, "get_closed_position_summary", lambda position_id: None)

    result = broker.close_demo_position(
        symbol="XAUUSD",
        position_id=111,
        quantity_lots=0.25,
    )

    assert result["status"] == "closed"
    assert result["verified"] is True
    assert result["position_id"] == 111
    assert result["acknowledgement_ambiguous"] is True
    assert result["reconciled_from_broker"] is True


def test_adapter_timeout_still_open_is_ambiguous_post_submit(monkeypatch) -> None:
    broker = CTraderBrokerAdapter()
    monkeypatch.setattr(ctd, "is_demo_account_confirmed", lambda: True)
    monkeypatch.setattr(ctd, "get_account_verification_error", lambda: None)
    monkeypatch.setattr(ctd, "symbol_name_to_id", {"XAUUSD": 7})
    monkeypatch.setattr(ctd, "ACCOUNT_ID", 123)
    monkeypatch.setattr(ctd, "close_position", lambda **kwargs: object())
    monkeypatch.setattr(
        ctd,
        "wait_for_deferred",
        lambda deferred, timeout: {"status": "failed", "error": "deferred timeout"},
    )
    monkeypatch.setattr(
        ctd,
        "get_open_positions",
        lambda: [{"position_id": 111, "symbol_name": "XAUUSD"}],
    )
    monkeypatch.setattr("backend.adapters.ctrader.time.sleep", lambda _: None)

    with pytest.raises(DemoCloseOutcomeAmbiguous) as exc_info:
        broker.close_demo_position(
            symbol="XAUUSD",
            position_id=111,
            quantity_lots=0.25,
        )

    assert exc_info.value.broker_position_id == 111
    assert exc_info.value.failure_kind == "ack_timeout"
    assert exc_info.value.submitted is True
    assert exc_info.value.ambiguous is True


def test_local_tracker_refuses_unverified_or_identity_mismatched_close() -> None:
    position = _open_position()

    with pytest.raises(RuntimeError, match="not verified"):
        close_local_position_after_broker_close(
            position,
            broker_close={"status": "rejected", "position_id": 111, "verified": False},
            fallback_price=100.0,
            reason="test",
        )

    with pytest.raises(RuntimeError, match="identity does not match"):
        close_local_position_after_broker_close(
            position,
            broker_close={"status": "closed", "position_id": 222, "verified": True},
            fallback_price=100.0,
            reason="test",
        )

    assert list_paper_positions("open")[0].id == position.id


def test_explicit_rejection_retains_tracker_and_surfaces_active_incident(monkeypatch) -> None:
    position = _open_position()
    calls = {"close": 0}

    def _reject(**kwargs):
        calls["close"] += 1
        raise DemoCloseRejected(
            "broker rejected close",
            broker_position_id=111,
            ack={"status": "order_rejected", "reject_reason": "TRADING_BAD_VOLUME"},
        )

    monkeypatch.setattr(close_safety, "close_demo_position", _reject)

    result = close_safety.attempt_verified_demo_close(
        position,
        fallback_price=100.0,
        reason="broker_take_profit",
        phase="scenario_test",
    )

    assert result["status"] == "rejected"
    assert result["closed"] is False
    assert result["retryable"] is False
    assert calls["close"] == 1
    assert list_paper_positions("open")[0].broker_position_id == 111

    monkeypatch.setattr(close_safety, "list_positions", lambda: [_broker_row()])
    second = close_safety.attempt_verified_demo_close(
        position,
        fallback_price=100.0,
        reason="broker_take_profit",
        phase="same_bar_repeat",
    )
    assert second["status"] == "rejected_pending"
    assert second["close_request_sent"] is False
    assert second["broker_position_still_open"] is True
    assert calls["close"] == 1

    events = list_paper_events(20)
    rejected = next(event for event in events if event.event_type == "ctrader_demo_close_rejected")
    assert rejected.details["broker_position_id"] == 111
    assert rejected.details["tracking_retained"] is True
    assert rejected.details["automatic_retry"] is False

    active = router_module._active_status_incidents(
        _config(),
        _healthy_broker(),
        EngineRuntime(
            running=True,
            loop_active=True,
            ollama_ready=True,
            active_watchlist=["XAUUSD:M5"],
        ),
    )
    close_incident = next(item for item in active if item.code == "broker_close_rejected")
    assert "Local tracking remains open" in close_incident.message


def test_ambiguous_close_never_resubmits_and_later_broker_absence_closes_tracker(monkeypatch) -> None:
    position = _open_position()
    calls = {"close": 0}
    broker_rows = {"rows": [_broker_row()]}

    def _ambiguous(**kwargs):
        calls["close"] += 1
        raise DemoCloseOutcomeAmbiguous(
            "close acknowledgement timed out",
            failure_kind="ack_timeout",
            broker_position_id=111,
            ack={"status": "failed", "error": "deferred timeout"},
        )

    monkeypatch.setattr(close_safety, "close_demo_position", _ambiguous)
    monkeypatch.setattr(close_safety, "list_positions", lambda: list(broker_rows["rows"]))
    monkeypatch.setattr(close_safety, "get_closed_position_summary", lambda *args, **kwargs: None)

    first = close_safety.attempt_verified_demo_close(
        position,
        fallback_price=100.0,
        reason="broker_take_profit",
        phase="scenario_test",
    )
    assert first["status"] == "ambiguous_pending"
    assert first["closed"] is False
    assert first["retryable"] is False
    assert calls["close"] == 1
    assert list_paper_positions("open")[0].id == position.id

    second = close_safety.attempt_verified_demo_close(
        position,
        fallback_price=100.0,
        reason="broker_take_profit",
        phase="later_reconcile_still_open",
    )
    assert second["status"] == "ambiguous_pending"
    assert second["close_request_sent"] is False
    assert second["broker_position_still_open"] is True
    assert calls["close"] == 1

    active = router_module._active_status_incidents(
        _config(),
        _healthy_broker(),
        EngineRuntime(
            running=True,
            loop_active=True,
            ollama_ready=True,
            active_watchlist=["XAUUSD:M5"],
        ),
    )
    assert any(item.code == "broker_close_ambiguous" for item in active)

    broker_rows["rows"] = []
    third = close_safety.attempt_verified_demo_close(
        position,
        fallback_price=100.0,
        reason="broker_take_profit",
        phase="later_reconcile_absent",
    )

    assert third["status"] == "closed"
    assert third["closed"] is True
    assert third["close_request_sent"] is False
    assert third["reconciled_from_broker"] is True
    assert calls["close"] == 1
    assert list_paper_positions("open") == []
    assert len(list_paper_positions("closed")) == 1

    events = list_paper_events(20)
    assert any(event.event_type == "ctrader_demo_close_ambiguous" for event in events)
    assert any(event.event_type == "ctrader_demo_close_reconciled" for event in events)

    recovered_active = router_module._active_status_incidents(
        _config(),
        _healthy_broker(),
        EngineRuntime(
            running=True,
            loop_active=True,
            ollama_ready=True,
            active_watchlist=["XAUUSD:M5"],
        ),
    )
    assert all(item.code != "broker_close_ambiguous" for item in recovered_active)


def test_new_order_protective_close_rejection_sends_only_one_close_and_retains_tracker(monkeypatch) -> None:
    _mock_demo_ready(monkeypatch)
    monkeypatch.setattr(execution_engine, "list_positions", lambda: [])
    monkeypatch.setattr(
        execution_engine,
        "place_demo_market_order",
        lambda **kwargs: {
            "status": "executed",
            "account_type": "demo",
            "symbol": "XAUUSD",
            "direction": "long",
            "quantity_lots": 0.25,
            "position_id": 111,
            "entry_price": 100.0,
            "ack": {},
        },
    )
    monkeypatch.setattr(
        execution_engine,
        "sync_demo_position_targets",
        lambda **kwargs: {
            "status": "exit_due_stop_loss",
            "position_id": 111,
            "quantity_lots": 0.25,
            "reference_price": 100.0,
        },
    )
    calls = {"close": 0}

    def _reject_close(**kwargs):
        calls["close"] += 1
        raise DemoCloseRejected(
            "broker rejected close",
            broker_position_id=111,
            ack={"status": "order_rejected", "reject_reason": "TRADING_BAD_VOLUME"},
        )

    monkeypatch.setattr(execution_engine, "close_demo_position", _reject_close)

    result = execute_paper_signal(
        config=_config(),
        watch_item=_watch(),
        analysis=_analysis(),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC).replace(tzinfo=None),
        bar_snapshot={"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0},
    )

    assert calls["close"] == 1
    assert result.action_taken is True
    assert result.status == "close_rejected"
    assert result.retryable is False
    assert result.broker_position_id == 111

    open_positions = list_paper_positions("open")
    assert len(open_positions) == 1
    assert open_positions[0].broker_position_id == 111

    events = list_paper_events(20)
    rejected = next(event for event in events if event.event_type == "ctrader_demo_close_rejected")
    assert rejected.details["position_id"] == open_positions[0].id
    assert rejected.details["broker_position_id"] == 111
