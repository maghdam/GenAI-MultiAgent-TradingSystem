from __future__ import annotations

from datetime import UTC, datetime
import sqlite3

import pytest

from backend.adapters.ctrader import CTraderCloseRejected
from backend.domain.models import (
    BrokerAccountSnapshot,
    EngineConfig,
    StrategyAnalysis,
    WatchlistItem,
)
from backend.services import execution_engine
from backend.services import reconciler as reconciler_module
from backend.services.execution_engine import execute_paper_signal
from backend.services.reconciler import recover_broker_trackers
from backend.storage import db as db_module
from backend.storage.db import SQLiteBusyError
from backend.storage.repositories import (
    list_broker_deals,
    list_incidents,
    list_order_intents,
    list_paper_positions,
    log_incident,
    open_paper_position,
    record_broker_deals,
)


BROKER_POSITION_ID = 880001


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
        reasons=["sqlite resilience test"],
    )


def _config(*, with_watchlist: bool = False) -> EngineConfig:
    watchlist = [_watch()] if with_watchlist else []
    return EngineConfig(
        enabled=True,
        paper_autotrade=False,
        ctrader_autotrade=True,
                kill_switch=False,
        require_stops=True,
        cooldown_minutes=0,
        risk_per_trade_pct=0,
        watchlist=watchlist,
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
        "symbol": "XAUUSD",
        "direction": "buy",
        "volume_lots": 0.25,
        "entry_price": 100.0,
        "stop_loss": 99.0,
        "take_profit": 102.0,
        "position_id": position_id,
    }


def _broker_order(position_id: int = BROKER_POSITION_ID) -> dict:
    return {
        "status": "executed",
        "account_type": "demo",
        "symbol": "XAUUSD",
        "direction": "long",
        "quantity_lots": 0.25,
        "position_id": position_id,
        "entry_price": 100.0,
        "ack": {},
    }


def _mock_demo_ready(monkeypatch) -> None:
    monkeypatch.setattr(
        execution_engine,
        "get_symbol_execution_readiness",
        lambda symbol: (True, "ready"),
    )
    monkeypatch.setattr(execution_engine, "get_broker_account_snapshot", _snapshot)
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


def _patch_recovery_broker(monkeypatch, rows: list[dict]) -> None:
    monkeypatch.setattr(
        reconciler_module,
        "get_broker_status",
        lambda: type("S", (), {"execution_ready": True})(),
    )
    monkeypatch.setattr(reconciler_module, "get_broker_account_snapshot", _snapshot)
    monkeypatch.setattr(reconciler_module, "list_positions", lambda: rows)
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


def _deal(deal_id: int = 990001) -> dict:
    return {
        "deal_id": deal_id,
        "execution_price": 101.0,
        "execution_at": datetime.now(UTC).replace(tzinfo=None),
        "closed_volume_api": 1000.0,
        "closed_volume_lots": 0.10,
        "gross_profit": 5.0,
        "swap": 0.0,
        "commission": 0.0,
        "pnl_conversion_fee": 0.0,
        "net_profit": 5.0,
    }


def test_real_sqlite_lock_rolls_back_surfaces_incident_and_recovers_idempotently(
    monkeypatch,
) -> None:
    position = open_paper_position(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        direction="long",
        quantity=0.25,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        broker_position_id=BROKER_POSITION_ID,
    )
    monkeypatch.setattr(db_module, "SQLITE_BUSY_TIMEOUT_MS", 25)
    current = getattr(db_module._LOCAL, "connection", None)
    if current is not None:
        current.execute("PRAGMA busy_timeout = 25")

    locker = sqlite3.connect(db_module.SETTINGS.db_path, timeout=0)
    locker.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(SQLiteBusyError, match="busy/locked"):
            record_broker_deals(
                local_position_id=position.id,
                broker_position_id=BROKER_POSITION_ID,
                symbol="XAUUSD",
                account_currency="CHF",
                deals=[_deal()],
            )

        log_incident(
            "error",
            "sqlite_test_lock_visible",
            "SQLite lock is visible while durable incident persistence is unavailable.",
            {
                "action_required": "Release the writer lock and retry local persistence only.",
                "broker_submission_suppressed": True,
            },
        )
        volatile = list_incidents(10)
        assert volatile[0].code == "sqlite_test_lock_visible"
        assert volatile[0].details["broker_submission_suppressed"] is True
    finally:
        locker.rollback()
        locker.close()

    assert list_broker_deals(local_position_id=position.id) == []

    first = record_broker_deals(
        local_position_id=position.id,
        broker_position_id=BROKER_POSITION_ID,
        symbol="XAUUSD",
        account_currency="CHF",
        deals=[_deal()],
    )
    second = record_broker_deals(
        local_position_id=position.id,
        broker_position_id=BROKER_POSITION_ID,
        symbol="XAUUSD",
        account_currency="CHF",
        deals=[_deal()],
    )
    log_incident(
        "info",
        "sqlite_test_recovered",
        "SQLite became writable again.",
        {"recovered": True},
    )

    assert first["inserted"] == 1
    assert second["inserted"] == 0
    rows = list_broker_deals(local_position_id=position.id)
    assert [row["deal_id"] for row in rows] == [990001]
    codes = [row.code for row in list_incidents(10)]
    assert "sqlite_test_lock_visible" in codes
    assert "sqlite_test_recovered" in codes


def test_pre_submit_sqlite_busy_blocks_broker_order_and_surfaces_actionable_incident(
    monkeypatch,
) -> None:
    _mock_demo_ready(monkeypatch)
    monkeypatch.setattr(
        execution_engine,
        "create_decision_record",
        lambda **kwargs: (_ for _ in ()).throw(
            SQLiteBusyError("SQLite persistence is busy/locked")
        ),
    )
    monkeypatch.setattr(
        execution_engine,
        "place_market_order",
        lambda **kwargs: pytest.fail("broker order must not run before durable persistence"),
    )

    result = _run_signal()

    assert result.action_taken is False
    assert result.status == "deferred"
    assert result.retryable is True
    assert list_order_intents(10) == []
    assert list_paper_positions("open") == []

    incident = next(
        row for row in list_incidents(20)
        if row.code == "sqlite_persistence_busy_pre_submit"
    )
    assert incident.details["broker_submission_suppressed"] is True
    assert "no broker order was submitted" in incident.details["action_required"].lower()


def test_post_submit_sqlite_busy_verified_failsafe_close_blocks_duplicate_until_resolved(
    monkeypatch,
) -> None:
    _mock_demo_ready(monkeypatch)
    monkeypatch.setattr(execution_engine, "list_positions", lambda: [])

    calls = {"place": 0, "close": 0}
    monkeypatch.setattr(
        execution_engine,
        "place_market_order",
        lambda **kwargs: calls.__setitem__("place", calls["place"] + 1)
        or _broker_order(),
    )
    monkeypatch.setattr(
        execution_engine,
        "close_position",
        lambda **kwargs: calls.__setitem__("close", calls["close"] + 1)
        or {
            "status": "closed",
            "verified": True,
            "position_id": kwargs["position_id"],
            "close_summary": None,
        },
    )

    real_update = execution_engine.update_order_intent_status
    fail_handoff = {"value": True}

    def _update(intent_id, status, details=None, reason=""):
        payload = details if isinstance(details, dict) else {}
        if (
            fail_handoff["value"]
            and payload.get("outcome_state") == "broker_confirmed_tracking_pending"
        ):
            fail_handoff["value"] = False
            raise SQLiteBusyError("SQLite persistence is busy/locked after submit")
        return real_update(intent_id, status, details, reason)

    monkeypatch.setattr(execution_engine, "update_order_intent_status", _update)

    first = _run_signal()

    assert first.action_taken is True
    assert first.status == "persistence_pending"
    assert first.retryable is False
    assert calls == {"place": 1, "close": 1}
    assert list_paper_positions("open") == []

    reserved = list_order_intents(10)[0]
    assert reserved.status == "accepted"
    assert reserved.details["outcome_state"] == "submission_reserved"
    assert reserved.details["automatic_retry"] is False

    second = _run_signal()

    assert second.action_taken is False
    assert second.status == "deferred"
    assert second.retryable is True
    assert calls == {"place": 1, "close": 1}

    resolved = list_order_intents(10)[0]
    assert resolved.status == "failed"
    assert resolved.details["outcome_state"] == "submission_resolved_no_position"
    assert resolved.details["ambiguity_resolved"] is True
    assert resolved.details["broker_position_confirmed"] is False

    codes = [row.code for row in list_incidents(30)]
    assert "sqlite_persistence_busy_post_submit" in codes
    assert "sqlite_persistence_submission_resolved" in codes


def test_post_submit_sqlite_busy_close_rejection_reconciles_broker_identity_without_resubmit(
    monkeypatch,
) -> None:
    _mock_demo_ready(monkeypatch)
    broker_rows = {"rows": []}
    monkeypatch.setattr(execution_engine, "list_positions", lambda: broker_rows["rows"])

    calls = {"place": 0, "close": 0}
    monkeypatch.setattr(
        execution_engine,
        "place_market_order",
        lambda **kwargs: calls.__setitem__("place", calls["place"] + 1)
        or _broker_order(),
    )

    def _reject_close(**kwargs):
        calls["close"] += 1
        raise CTraderCloseRejected(
            "broker rejected persistence fail-safe close",
            broker_position_id=kwargs["position_id"],
            ack={"status": "order_rejected"},
        )

    monkeypatch.setattr(execution_engine, "close_position", _reject_close)

    real_update = execution_engine.update_order_intent_status
    fail_handoff = {"value": True}

    def _update(intent_id, status, details=None, reason=""):
        payload = details if isinstance(details, dict) else {}
        if (
            fail_handoff["value"]
            and payload.get("outcome_state") == "broker_confirmed_tracking_pending"
        ):
            fail_handoff["value"] = False
            raise SQLiteBusyError("SQLite persistence is busy/locked after submit")
        return real_update(intent_id, status, details, reason)

    monkeypatch.setattr(execution_engine, "update_order_intent_status", _update)

    first = _run_signal()
    assert first.status == "persistence_pending"
    assert calls == {"place": 1, "close": 1}

    broker_rows["rows"] = [_broker_row()]
    second = _run_signal()

    assert second.status == "blocked"
    assert second.broker_position_id == BROKER_POSITION_ID
    assert calls == {"place": 1, "close": 1}

    intent = list_order_intents(10)[0]
    assert intent.status == "failed"
    assert intent.details["outcome_state"] == "broker_confirmed_tracking_pending"
    assert intent.details["tracking_retained"] is True
    assert intent.details["broker_order"]["position_id"] == BROKER_POSITION_ID

    _patch_recovery_broker(monkeypatch, [_broker_row()])
    recovery = recover_broker_trackers(_config(with_watchlist=True))

    assert recovery["recovered"] == 1
    positions = list_paper_positions("open")
    assert len(positions) == 1
    assert positions[0].broker_position_id == BROKER_POSITION_ID
    assert positions[0].quantity == pytest.approx(0.25)


def test_broker_confirmed_handoff_recovers_when_local_tracker_write_is_busy(
    monkeypatch,
) -> None:
    _mock_demo_ready(monkeypatch)
    monkeypatch.setattr(execution_engine, "list_positions", lambda: [])

    calls = {"place": 0}
    monkeypatch.setattr(
        execution_engine,
        "place_market_order",
        lambda **kwargs: calls.__setitem__("place", calls["place"] + 1)
        or _broker_order(),
    )
    monkeypatch.setattr(
        execution_engine,
        "open_paper_position",
        lambda **kwargs: (_ for _ in ()).throw(
            SQLiteBusyError("SQLite persistence is busy/locked while creating tracker")
        ),
    )

    result = _run_signal()

    assert result.action_taken is True
    assert result.status == "persistence_pending"
    assert result.retryable is False
    assert result.broker_position_id == BROKER_POSITION_ID
    assert calls["place"] == 1
    assert list_paper_positions("open") == []

    intent = list_order_intents(10)[0]
    assert intent.status == "accepted"
    assert intent.details["outcome_state"] == "broker_confirmed_tracking_pending"
    assert intent.details["tracking_retained"] is True
    assert intent.details["broker_order"]["position_id"] == BROKER_POSITION_ID

    monkeypatch.setattr(
        execution_engine,
        "open_paper_position",
        open_paper_position,
    )
    _patch_recovery_broker(monkeypatch, [_broker_row()])
    recovery = recover_broker_trackers(_config(with_watchlist=True))

    assert recovery["recovered"] == 1
    positions = list_paper_positions("open")
    assert len(positions) == 1
    assert positions[0].broker_position_id == BROKER_POSITION_ID
    assert positions[0].entry_price == pytest.approx(100.0)
    assert calls["place"] == 1
