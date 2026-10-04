from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from backend.domain.models import (
    BrokerAccountSnapshot,
    EngineConfig,
    EngineRuntime,
    StrategyAnalysis,
    WatchlistItem,
)
from backend.services import execution_engine
from backend.services import reconciler as reconciler_module
from backend.services.execution_engine import execute_paper_signal
from backend.services.reconciler import (
    reconcile_open_positions,
    recover_broker_trackers,
)
from backend.storage import db as db_module
from backend.storage.repositories import (
    create_order_intent,
    list_broker_deals,
    list_incidents,
    list_order_intent_transitions,
    list_order_intents,
    list_paper_positions,
    list_trade_audits,
    load_engine_config,
    load_runtime,
    open_paper_position,
    record_broker_deals,
    save_engine_config,
    save_runtime,
    update_order_intent_status,
)


BROKER_POSITION_ID = 910001


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
        watchlist=[
            WatchlistItem(
                symbol="NAS100",
                timeframe="M5",
                strategy="breakout",
                enabled=True,
                trading_enabled=True,
                lot_size=0.10,
            )
        ],
    )


def _watch() -> WatchlistItem:
    return _config().watchlist[0]


def _analysis() -> StrategyAnalysis:
    return StrategyAnalysis(
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        signal="long",
        confidence=0.90,
        entry_price=29486.2,
        stop_loss=29476.4,
        take_profit=29511.7,
        reasons=["db restart resilience test"],
    )


def _snapshot() -> BrokerAccountSnapshot:
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


def _broker_row(position_id: int = BROKER_POSITION_ID) -> dict:
    return {
        "symbol": "NAS100",
        "direction": "buy",
        "volume_lots": 0.10,
        "entry_price": 29486.2,
        "stop_loss": 29476.4,
        "take_profit": 29511.7,
        "position_id": position_id,
    }


def _bars(price: float = 29490.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "open": price - 1.0,
                "high": price + 1.0,
                "low": price - 2.0,
                "close": price,
                "volume": 1.0,
            }
        ],
        index=pd.to_datetime(["2026-10-02T12:00:00Z"], utc=True),
    )


def _restart_storage_connection() -> None:
    """Simulate the SQLite boundary of a backend/database restart."""
    conn = getattr(db_module._LOCAL, "connection", None)
    if conn is not None:
        conn.close()
    db_module._LOCAL.connection = None
    db_module.init_db()


def _create_executed_intent(*, opened_position_id: int | None = None):
    intent = create_order_intent(
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        direction="long",
        intent_type="open",
        status="accepted",
        confidence=0.90,
        entry_price=29486.2,
        stop_loss=29476.4,
        take_profit=29511.7,
        quantity=0.10,
        rationale="db restart resilience fixture",
        details={},
    )
    return update_order_intent_status(
        intent.id,
        "executed",
        {
            "opened_position_id": opened_position_id,
            "execution_mode": "ctrader_demo",
            "broker_order": {
                "position_id": BROKER_POSITION_ID,
                "symbol": "NAS100",
                "direction": "long",
                "quantity_lots": 0.10,
                "entry_price": 29486.2,
            },
        },
        reason="db_restart_fixture_executed",
    )


def _patch_ready_recovery_broker(monkeypatch, *, rows: list[dict] | None = None):
    broker_rows = rows if rows is not None else [_broker_row()]
    sync_calls: list[dict] = []
    monkeypatch.setattr(
        reconciler_module,
        "get_broker_status",
        lambda: type("S", (), {"execution_ready": True})(),
    )
    monkeypatch.setattr(reconciler_module, "get_broker_account_snapshot", _snapshot)
    monkeypatch.setattr(reconciler_module, "list_positions", lambda: broker_rows)
    monkeypatch.setattr(
        reconciler_module,
        "get_instrument_spec",
        lambda symbol, currency: type(
            "Spec",
            (),
            {
                "cash_per_price_unit_per_lot": 1.0,
                "source": "test",
                "valuation_ready": True,
            },
        )(),
    )
    monkeypatch.setattr(reconciler_module, "get_bars", lambda *args, **kwargs: _bars())
    monkeypatch.setattr(
        reconciler_module,
        "sync_position_targets",
        lambda **kwargs: sync_calls.append(kwargs)
        or {
            "status": "already_synced",
            "verified": True,
            "position_id": kwargs["position_id"],
        },
    )
    monkeypatch.setattr(
        reconciler_module,
        "close_position",
        lambda **kwargs: pytest.fail("healthy DB restart recovery must not close the broker position"),
    )
    return sync_calls


def _run_signal():
    return execute_paper_signal(
        config=_config(),
        watch_item=_watch(),
        analysis=_analysis(),
        mark_price=29486.2,
        bar_timestamp=datetime.now(UTC).replace(tzinfo=None),
        bar_snapshot={
            "open": 29485.0,
            "high": 29490.0,
            "low": 29480.0,
            "close": 29486.2,
        },
    )


def test_durable_runtime_trade_and_ledger_state_survive_storage_restart() -> None:
    config = _config()
    save_engine_config(config)
    runtime = EngineRuntime(
        running=True,
        loop_active=True,
        ollama_ready=True,
        last_error="persisted-before-restart",
        tick_count=17,
        active_watchlist=["NAS100:M5"],
    )
    save_runtime(runtime)

    position = open_paper_position(
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        direction="long",
        quantity=0.10,
        entry_price=29486.2,
        stop_loss=29476.4,
        take_profit=29511.7,
        account_currency="CHF",
        broker_position_id=BROKER_POSITION_ID,
    )
    intent = _create_executed_intent(opened_position_id=position.id)
    recorded = record_broker_deals(
        local_position_id=position.id,
        broker_position_id=BROKER_POSITION_ID,
        symbol="NAS100",
        account_currency="CHF",
        deals=[
            {
                "deal_id": 910101,
                "execution_price": 29490.0,
                "execution_at": datetime.now(UTC).replace(tzinfo=None),
                "closed_volume_api": 100.0,
                "closed_volume_lots": 0.01,
                "gross_profit": 4.0,
                "swap": 0.0,
                "commission": -0.2,
                "pnl_conversion_fee": 0.0,
                "net_profit": 3.8,
            }
        ],
    )
    assert recorded["inserted"] == 1

    _restart_storage_connection()

    loaded_config = load_engine_config(EngineConfig())
    loaded_runtime = load_runtime()
    positions = list_paper_positions("open")
    intents = list_order_intents(20)
    deals = list_broker_deals(local_position_id=position.id)

    assert loaded_config.model_dump() == config.model_dump()
    assert loaded_runtime.tick_count == 17
    assert loaded_runtime.last_error == "persisted-before-restart"
    assert loaded_runtime.active_watchlist == ["NAS100:M5"]
    assert len(positions) == 1
    assert positions[0].id == position.id
    assert positions[0].broker_position_id == BROKER_POSITION_ID
    assert len(intents) == 1
    assert intents[0].id == intent.id
    assert intents[0].status == "executed"
    assert intents[0].details["broker_order"]["position_id"] == BROKER_POSITION_ID
    assert len(list_order_intent_transitions(intent.id)) == 2
    assert [row["deal_id"] for row in deals] == [910101]

    replay = record_broker_deals(
        local_position_id=position.id,
        broker_position_id=BROKER_POSITION_ID,
        symbol="NAS100",
        account_currency="CHF",
        deals=[
            {
                "deal_id": 910101,
                "execution_price": 29490.0,
                "execution_at": datetime.now(UTC).replace(tzinfo=None),
                "closed_volume_api": 100.0,
                "closed_volume_lots": 0.01,
                "gross_profit": 4.0,
                "swap": 0.0,
                "commission": -0.2,
                "pnl_conversion_fee": 0.0,
                "net_profit": 3.8,
            }
        ],
    )
    assert replay["inserted"] == 0
    assert len(list_broker_deals(local_position_id=position.id)) == 1


def test_restart_reconciliation_keeps_one_tracker_and_resumes_canonical_protection(monkeypatch) -> None:
    save_engine_config(_config())
    position = open_paper_position(
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        direction="long",
        quantity=0.10,
        entry_price=29486.2,
        stop_loss=29476.4,
        take_profit=29511.7,
        account_currency="CHF",
        broker_position_id=BROKER_POSITION_ID,
    )
    intent = _create_executed_intent(opened_position_id=position.id)
    _restart_storage_connection()

    sync_calls = _patch_ready_recovery_broker(monkeypatch)
    first = reconcile_open_positions(reason="db_restart")

    assert first["checked"] == 1
    assert first["closed"] == 0
    assert len(list_paper_positions("open")) == 1
    assert len(list_order_intents(20)) == 1
    assert list_order_intents(20)[0].id == intent.id
    assert len(sync_calls) == 1
    assert sync_calls[0]["position_id"] == BROKER_POSITION_ID

    _restart_storage_connection()
    second = reconcile_open_positions(reason="db_restart_repeat")

    positions = list_paper_positions("open")
    assert second["checked"] == 1
    assert second["closed"] == 0
    assert len(positions) == 1
    assert positions[0].id == position.id
    assert positions[0].broker_position_id == BROKER_POSITION_ID
    assert len(list_order_intents(20)) == 1
    assert len(sync_calls) == 2
    assert all(call["position_id"] == BROKER_POSITION_ID for call in sync_calls)


def test_broker_confirmed_handoff_survives_db_restart_and_recovers_exactly_once(monkeypatch) -> None:
    save_engine_config(_config())
    intent = create_order_intent(
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        direction="long",
        intent_type="open",
        status="accepted",
        confidence=0.90,
        entry_price=29486.2,
        stop_loss=29476.4,
        take_profit=29511.7,
        quantity=0.10,
        rationale="broker-confirmed restart handoff",
        details={},
    )
    update_order_intent_status(
        intent.id,
        "accepted",
        {
            "outcome_state": "broker_confirmed_tracking_pending",
            "tracking_retained": True,
            "broker_position_confirmed": True,
            "automatic_retry": False,
            "broker_order": {
                "position_id": BROKER_POSITION_ID,
                "symbol": "NAS100",
                "direction": "long",
                "quantity_lots": 0.10,
                "entry_price": 29486.2,
            },
        },
        reason="broker_confirmed_tracking_pending",
    )

    _restart_storage_connection()
    _patch_ready_recovery_broker(monkeypatch)

    first = recover_broker_trackers(_config())
    assert first["recovered"] == 1
    assert first["untracked"] == 0

    positions = list_paper_positions("open")
    assert len(positions) == 1
    assert positions[0].broker_position_id == BROKER_POSITION_ID

    _restart_storage_connection()
    second = recover_broker_trackers(_config())

    assert second["recovered"] == 0
    positions = list_paper_positions("open")
    assert len(positions) == 1
    assert positions[0].broker_position_id == BROKER_POSITION_ID
    recovered_audits = [
        row
        for row in list_trade_audits(50)
        if row.event_type == "ctrader_demo_tracker_recovered"
    ]
    assert len(recovered_audits) == 1
    assert recovered_audits[0].intent_id == intent.id


def test_unresolved_submission_reservation_survives_restart_and_blocks_duplicate_demo_order(monkeypatch) -> None:
    save_engine_config(_config())
    intent = create_order_intent(
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        direction="long",
        intent_type="open",
        status="accepted",
        confidence=0.90,
        entry_price=29486.2,
        stop_loss=29476.4,
        take_profit=29511.7,
        quantity=0.10,
        rationale="reserved before DB restart",
        details={},
    )
    update_order_intent_status(
        intent.id,
        "accepted",
        {
            "outcome_state": "submission_reserved",
            "submission_may_have_succeeded": True,
            "ambiguity_resolved": False,
            "automatic_retry": False,
            "retryable": False,
            "client_msg_id": f"tradeagent-intent-{intent.id}",
            "baseline_snapshot_available": True,
            "baseline_position_ids": [],
        },
        reason="submission_reserved",
    )
    _restart_storage_connection()

    monkeypatch.setattr(
        execution_engine,
        "get_symbol_execution_readiness",
        lambda symbol: (True, "ready"),
    )
    monkeypatch.setattr(execution_engine, "get_broker_account_snapshot", _snapshot)
    monkeypatch.setattr(execution_engine, "sleep", lambda _: None)
    candidates = [
        _broker_row(BROKER_POSITION_ID),
        _broker_row(BROKER_POSITION_ID + 1),
    ]
    monkeypatch.setattr(execution_engine, "list_positions", lambda: candidates)
    monkeypatch.setattr(
        execution_engine,
        "place_market_order",
        lambda **kwargs: pytest.fail("unresolved durable reservation must block duplicate demo submission"),
    )
    monkeypatch.setattr(
        execution_engine,
        "sync_position_targets",
        lambda **kwargs: pytest.fail("ambiguous restart recovery must not mutate broker protection"),
    )

    first = _run_signal()
    assert first.action_taken is False
    assert first.status == "blocked"
    assert first.intent_id == intent.id
    assert first.retryable is False
    assert len(list_order_intents(20)) == 1
    assert list_paper_positions("open") == []

    _restart_storage_connection()
    second = _run_signal()

    assert second.action_taken is False
    assert second.status == "blocked"
    assert second.intent_id == intent.id
    assert len(list_order_intents(20)) == 1
    persisted = list_order_intents(20)[0]
    assert persisted.status == "accepted"
    assert persisted.details["outcome_state"] == "submission_reserved"
    assert persisted.details["automatic_retry"] is False


def test_restart_tracker_recovery_failure_is_actionable_and_persists_after_reopen(monkeypatch) -> None:
    save_engine_config(_config())
    intent = create_order_intent(
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        direction="long",
        intent_type="open",
        status="accepted",
        confidence=0.90,
        entry_price=29486.2,
        stop_loss=29476.4,
        take_profit=29511.7,
        quantity=0.10,
        rationale="recovery fault fixture",
        details={},
    )
    update_order_intent_status(
        intent.id,
        "executed",
        {
            "broker_order": {
                "position_id": BROKER_POSITION_ID,
                "symbol": "NAS100",
                "quantity_lots": 0.10,
            }
        },
        reason="recovery_fault_fixture",
    )
    _restart_storage_connection()

    monkeypatch.setattr(
        reconciler_module,
        "get_broker_status",
        lambda: type("S", (), {"execution_ready": True})(),
    )
    monkeypatch.setattr(reconciler_module, "get_broker_account_snapshot", _snapshot)
    monkeypatch.setattr(reconciler_module, "list_positions", lambda: [_broker_row()])
    monkeypatch.setattr(
        reconciler_module,
        "get_instrument_spec",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("instrument metadata unavailable during restart recovery")
        ),
    )
    monkeypatch.setattr(
        reconciler_module,
        "sync_position_targets",
        lambda **kwargs: pytest.fail("failed tracker recovery must not mutate broker protection"),
    )
    monkeypatch.setattr(
        reconciler_module,
        "close_position",
        lambda **kwargs: pytest.fail("failed tracker recovery must not close the broker position"),
    )

    summary = reconcile_open_positions(reason="db_restart")

    assert summary["checked"] == 0
    assert list_paper_positions("open") == []
    incidents = list_incidents(20)
    incident = next(
        row for row in incidents
        if row.code == "ctrader_demo_tracker_recovery_failed"
    )
    assert incident.details["automatic_adoption"] is False
    assert incident.details["broker_mutation_suppressed"] is True
    assert "rerun reconciliation" in incident.details["action_required"].lower()

    _restart_storage_connection()
    persisted = list_incidents(20)
    recovered_incident = next(
        row for row in persisted
        if row.code == "ctrader_demo_tracker_recovery_failed"
    )
    assert recovered_incident.details["action_required"] == incident.details["action_required"]
