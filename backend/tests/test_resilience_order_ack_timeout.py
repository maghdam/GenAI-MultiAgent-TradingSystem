from __future__ import annotations

from datetime import UTC, datetime

import pytest

import backend.ctrader_client as ctd
from backend.adapters.ctrader import (
    CTraderOrderAcknowledgementTimeout,
    adapter,
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
from backend.services import execution_engine
from backend.services.execution_engine import execute_paper_signal
from backend.storage.repositories import (
    list_incidents,
    list_order_intents,
    list_paper_positions,
)


def _analysis() -> StrategyAnalysis:
    return StrategyAnalysis(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        signal="long",
        confidence=0.9,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        reasons=["ack-timeout resilience test"],
    )


def _config() -> EngineConfig:
    return EngineConfig(
        enabled=True,
        paper_autotrade=False,
        ctrader_autotrade=True,
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


def _mock_demo_ready(monkeypatch) -> None:
    monkeypatch.setattr(
        execution_engine,
        "get_symbol_execution_readiness",
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
    monkeypatch.setattr(
        execution_engine,
        "sync_position_targets",
        lambda **kwargs: {
            "status": "synced",
            "position_id": kwargs.get("position_id"),
            "verified": True,
        },
    )
    monkeypatch.setattr(execution_engine, "sleep", lambda _: None)


def _run_signal():
    return execute_paper_signal(
        config=_config(),
        watch_item=_watch(),
        analysis=_analysis(),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC).replace(tzinfo=None),
        bar_snapshot={"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0},
    )


def _healthy_broker() -> BrokerStatus:
    return BrokerStatus(
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
        demo_account_confirmed=True,
        execution_ready=True,
        notes=[],
    )


def test_adapter_classifies_post_submit_ack_timeout(monkeypatch) -> None:
    monkeypatch.setattr(ctd, "is_connected", lambda: True)
    monkeypatch.setattr(ctd, "is_authorized", lambda: True)
    monkeypatch.setattr(ctd, "is_account_confirmed", lambda: True)
    monkeypatch.setattr(ctd, "is_symbol_metadata_ready", lambda: True)
    monkeypatch.setattr(ctd, "symbol_name_to_id", {"XAUUSD": 7})
    monkeypatch.setattr(ctd, "symbol_lot_size_map", {7: 100.0})
    monkeypatch.setattr(ctd, "symbol_min_volume_map", {7: 100})
    monkeypatch.setattr(ctd, "symbol_step_volume_map", {7: 100})
    monkeypatch.setattr(ctd, "symbol_max_volume_map", {7: 500_000})
    monkeypatch.setattr(ctd, "volume_lots_to_units", lambda symbol_id, lots: 2500)

    captured = {}

    def _place_order(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(ctd, "place_order", _place_order)
    monkeypatch.setattr(
        ctd,
        "wait_for_deferred",
        lambda deferred, timeout: {"status": "failed", "error": "deferred timeout"},
    )

    with pytest.raises(CTraderOrderAcknowledgementTimeout) as exc_info:
        adapter.place_market_order(
            symbol="XAUUSD",
            direction="long",
            quantity_lots=0.25,
            stop_loss=99.0,
            take_profit=102.0,
            client_msg_id="tradeagent-intent-77",
        )

    assert exc_info.value.submitted is True
    assert exc_info.value.client_msg_id == "tradeagent-intent-77"
    assert captured["client_msg_id"] == "tradeagent-intent-77"


def test_unresolved_ack_timeout_is_nonretryable_and_blocks_later_resubmission(monkeypatch) -> None:
    _mock_demo_ready(monkeypatch)

    snapshots = iter([[], [], [], []])
    monkeypatch.setattr(execution_engine, "list_positions", lambda: next(snapshots))

    calls = {"place": 0}

    def _place(**kwargs):
        calls["place"] += 1
        raise CTraderOrderAcknowledgementTimeout(
            "ack timeout after submission",
            client_msg_id=kwargs.get("client_msg_id"),
        )

    monkeypatch.setattr(execution_engine, "place_market_order", _place)

    first = _run_signal()

    assert first.action_taken is False
    assert first.status == "failed"
    assert first.retryable is False
    assert calls["place"] == 1
    assert list_paper_positions("open") == []

    intents = list_order_intents(10)
    assert len(intents) == 1
    intent = intents[0]
    assert intent.status == "failed"
    assert intent.details["outcome_state"] == "ambiguous_post_submit"
    assert intent.details["acknowledgement_timeout"] is True
    assert intent.details["submission_may_have_succeeded"] is True
    assert intent.details["ambiguity_resolved"] is False
    assert intent.details["retryable"] is False
    assert intent.details["automatic_retry"] is False
    assert intent.details["reconciliation"]["status"] == "broker_position_not_observed"

    incidents = list_incidents(20)
    assert incidents[0].code == "ctrader_order_ack_timeout_ambiguous"
    assert incidents[0].details["automatic_retry"] is False

    second = _run_signal()

    assert second.action_taken is False
    assert second.status == "blocked"
    assert second.retryable is False
    assert second.intent_id == intent.id
    assert calls["place"] == 1
    assert len(list_order_intents(10)) == 1

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
    ack_incident = next(item for item in active if item.code == "order_acknowledgement_ambiguous")
    assert "Automatic resubmission is blocked" in ack_incident.message


def test_unique_new_broker_position_reconciles_without_resubmission(monkeypatch) -> None:
    _mock_demo_ready(monkeypatch)

    baseline = {
        "symbol": "XAUUSD",
        "direction": "buy",
        "volume_lots": 0.25,
        "entry_price": 98.0,
        "position_id": 100,
    }
    confirmed = {
        "symbol": "XAUUSD",
        "direction": "buy",
        "volume_lots": 0.25,
        "entry_price": 100.4,
        "position_id": 321,
    }
    snapshots = iter([[baseline], [baseline, confirmed]])
    monkeypatch.setattr(execution_engine, "list_positions", lambda: next(snapshots))

    calls = {"place": 0}

    def _place(**kwargs):
        calls["place"] += 1
        raise CTraderOrderAcknowledgementTimeout(
            "ack timeout after submission",
            client_msg_id=kwargs.get("client_msg_id"),
        )

    monkeypatch.setattr(execution_engine, "place_market_order", _place)

    result = _run_signal()

    assert calls["place"] == 1
    assert result.action_taken is True
    assert result.status == "failed"
    assert result.retryable is False
    assert result.broker_position_id == 321

    positions = list_paper_positions("open")
    assert len(positions) == 1
    assert positions[0].broker_position_id == 321
    assert positions[0].entry_price == pytest.approx(100.4)
    assert positions[0].quantity == pytest.approx(0.25)

    intent = list_order_intents(10)[0]
    assert intent.status == "failed"
    assert intent.details["outcome_state"] == "ack_timeout_reconciled_broker_position"
    assert intent.details["ambiguity_resolved"] is True
    assert intent.details["broker_position_confirmed"] is True
    assert intent.details["tracking_retained"] is True
    assert intent.details["retryable"] is False
    assert intent.details["broker_order"]["position_id"] == 321
    assert intent.details["broker_order"]["reconciled_from_broker"] is True

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
    assert all(item.code != "order_acknowledgement_ambiguous" for item in active)


def test_multiple_new_broker_candidates_remain_ambiguous(monkeypatch) -> None:
    _mock_demo_ready(monkeypatch)

    candidate_a = {
        "symbol": "XAUUSD",
        "direction": "buy",
        "volume_lots": 0.25,
        "entry_price": 100.1,
        "position_id": 501,
    }
    candidate_b = {
        "symbol": "XAUUSD",
        "direction": "buy",
        "volume_lots": 0.25,
        "entry_price": 100.2,
        "position_id": 502,
    }
    snapshots = iter([[], [candidate_a, candidate_b]])
    monkeypatch.setattr(execution_engine, "list_positions", lambda: next(snapshots))
    monkeypatch.setattr(
        execution_engine,
        "place_market_order",
        lambda **kwargs: (_ for _ in ()).throw(
            CTraderOrderAcknowledgementTimeout(
                "ack timeout after submission",
                client_msg_id=kwargs.get("client_msg_id"),
            )
        ),
    )

    result = _run_signal()

    assert result.action_taken is False
    assert result.retryable is False
    assert list_paper_positions("open") == []

    intent = list_order_intents(10)[0]
    assert intent.details["outcome_state"] == "ambiguous_post_submit"
    assert intent.details["reconciliation"]["status"] == "multiple_new_candidates"
    assert intent.details["reconciliation"]["candidate_count"] == 2
    assert intent.details["reconciliation"]["automatic_adoption"] is False


def test_reconciliation_refuses_symbol_only_adoption_without_baseline(monkeypatch) -> None:
    monkeypatch.setattr(
        execution_engine,
        "list_positions",
        lambda: pytest.fail("broker rows must not be auto-adopted without a pre-submit baseline"),
    )

    row, details = execution_engine._reconcile_order_ack_timeout(
        baseline_available=False,
        baseline_ids=set(),
        symbol="XAUUSD",
        direction="long",
        quantity_lots=0.25,
    )

    assert row is None
    assert details["status"] == "baseline_unavailable"
    assert details["automatic_adoption"] is False


def test_unverified_account_remains_blocked_before_order_submission(monkeypatch) -> None:
    monkeypatch.setattr(ctd, "is_connected", lambda: True)
    monkeypatch.setattr(ctd, "is_authorized", lambda: True)
    monkeypatch.setattr(ctd, "is_account_confirmed", lambda: False)
    monkeypatch.setattr(
        ctd,
        "get_account_verification_error",
        lambda: "The selected cTrader account is not the authenticated active account.",
    )
    monkeypatch.setattr(
        ctd,
        "place_order",
        lambda **kwargs: pytest.fail("live-account order submission must remain blocked"),
    )

    with pytest.raises(RuntimeError, match="not the authenticated active account"):
        adapter.place_market_order(
            symbol="XAUUSD",
            direction="long",
            quantity_lots=0.25,
            stop_loss=99.0,
            take_profit=102.0,
            client_msg_id="tradeagent-intent-live-block",
        )
