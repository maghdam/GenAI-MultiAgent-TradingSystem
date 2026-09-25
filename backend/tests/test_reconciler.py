from __future__ import annotations

from backend.domain.models import EngineConfig, EngineRuntime, WatchlistItem
from backend.services.market_data import MarketDataError
from backend.services.reconciler import reconcile_open_positions, recover_demo_broker_trackers, recover_runtime_state
from backend.storage.repositories import (
    create_order_intent,
    list_incidents,
    list_paper_positions,
    load_runtime,
    open_paper_position,
    save_engine_config,
    save_runtime,
)


def test_recover_runtime_state_rebuilds_active_watchlist() -> None:
    config = EngineConfig(
        enabled=True,
        watchlist=[
            WatchlistItem(symbol="XAUUSD", timeframe="M5", strategy="sma_cross", enabled=True, params={}),
            WatchlistItem(symbol="EURUSD", timeframe="H1", strategy="sma_cross", enabled=False, params={}),
        ],
    )
    save_runtime(EngineRuntime(running=False, loop_active=True, active_watchlist=[]))

    result = recover_runtime_state(config)
    runtime = load_runtime()

    assert result["enabled"] is True
    assert result["active_watchlist"] == ["XAUUSD:M5"]
    assert runtime.running is True
    assert runtime.loop_active is False
    assert runtime.active_watchlist == ["XAUUSD:M5"]
    assert "runtime_recovered" in (runtime.last_reconcile_summary or "")


def test_reconcile_open_positions_closes_take_profit(monkeypatch) -> None:
    open_paper_position(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        direction="long",
        quantity=1.0,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
    )

    def _fake_bars(symbol: str, timeframe: str, bars: int):
        import pandas as pd

        return pd.DataFrame(
            [{"close": 102.5}],
            index=pd.to_datetime(["2026-03-19T10:00:00Z"], utc=True),
        )

    monkeypatch.setattr("backend.services.reconciler.get_bars", _fake_bars)

    summary = reconcile_open_positions(reason="test_take_profit")

    assert summary == {
        "checked": 1,
        "closed": 1,
        "skipped": 0,
        "reason": "test_take_profit",
    }

    open_positions = list_paper_positions("open")
    closed_positions = list_paper_positions("closed")
    assert open_positions == []
    assert len(closed_positions) == 1
    assert closed_positions[0].close_reason == "take_profit"
    assert closed_positions[0].exit_price == 102.0


def test_reconcile_open_positions_logs_skip_incident_when_market_data_fails(monkeypatch) -> None:
    open_paper_position(
        symbol="EURUSD",
        timeframe="H1",
        strategy="sma_cross",
        direction="short",
        quantity=1.0,
        entry_price=1.1,
        stop_loss=1.2,
        take_profit=1.0,
    )

    def _raise_market_data(symbol: str, timeframe: str, bars: int):
        raise MarketDataError("feed unavailable")

    monkeypatch.setattr("backend.services.reconciler.get_bars", _raise_market_data)

    summary = reconcile_open_positions(reason="test_skip")

    assert summary == {
        "checked": 0,
        "closed": 0,
        "skipped": 1,
        "reason": "test_skip",
    }

    incidents = list_incidents(5)
    assert len(incidents) == 1
    assert incidents[0].code == "reconcile_market_data_unavailable"
    assert incidents[0].details["reason"] == "test_skip"


def test_recover_demo_broker_tracker_from_tradeagent_intent(monkeypatch) -> None:
    config = EngineConfig(
        enabled=True,
        demo_autotrade=True,
        watchlist=[
            WatchlistItem(
                symbol="NAS100",
                timeframe="M5",
                strategy="breakout",
                enabled=True,
                trading_enabled=True,
                lot_size=0.1,
                params={},
            )
        ],
    )
    save_engine_config(config)
    intent = create_order_intent(
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        direction="long",
        intent_type="open",
        status="accepted",
        confidence=0.8,
        entry_price=29487.5,
        stop_loss=29476.4,
        take_profit=29511.7,
        quantity=0.1,
        rationale="test",
        details={},
    )
    from backend.storage.repositories import update_order_intent_status
    update_order_intent_status(
        intent.id,
        "executed",
        {
            "broker_order": {
                "position_id": 56980461,
                "symbol": "NAS100",
                "quantity_lots": 0.1,
            }
        },
        reason="ctrader_demo_order_executed",
    )

    monkeypatch.setattr(
        "backend.services.reconciler.get_broker_status",
        lambda: type("S", (), {"execution_ready": True})(),
    )
    monkeypatch.setattr(
        "backend.services.reconciler.list_positions",
        lambda: [
            {
                "symbol": "NAS100",
                "direction": "buy",
                "volume_lots": 0.1,
                "entry_price": 29486.2,
                "stop_loss": None,
                "take_profit": None,
                "position_id": 56980461,
            }
        ],
    )
    monkeypatch.setattr(
        "backend.services.reconciler.get_instrument_spec",
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
    sync_calls = []
    monkeypatch.setattr(
        "backend.services.reconciler.sync_demo_position_targets",
        lambda **kwargs: sync_calls.append(kwargs) or {"status": "synced", "verified": True},
    )

    result = recover_demo_broker_trackers(config)

    assert result["recovered"] == 1
    positions = list_paper_positions("open")
    assert len(positions) == 1
    assert positions[0].symbol == "NAS100"
    assert positions[0].quantity == 0.1
    assert positions[0].entry_price == 29486.2
    assert positions[0].stop_loss == 29476.4
    assert positions[0].take_profit == 29511.7
    assert positions[0].broker_position_id == 56980461
    # Recovery only reconstructs the tracker. Protection is repaired later
    # during market-aware reconciliation so stale targets are never moved.
    assert sync_calls == []


def test_recover_demo_broker_tracker_attaches_id_to_legacy_local_tracker(monkeypatch) -> None:
    config = EngineConfig(
        enabled=True,
        demo_autotrade=True,
        watchlist=[
            WatchlistItem(
                symbol="NAS100",
                timeframe="M5",
                strategy="breakout",
                enabled=True,
                trading_enabled=True,
                lot_size=0.1,
                params={},
            )
        ],
    )
    save_engine_config(config)
    intent = create_order_intent(
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        direction="long",
        intent_type="open",
        status="accepted",
        confidence=0.8,
        entry_price=29487.5,
        stop_loss=29476.4,
        take_profit=29511.7,
        quantity=0.1,
        rationale="test",
        details={},
    )
    from backend.storage.repositories import update_order_intent_status

    update_order_intent_status(
        intent.id,
        "executed",
        {"broker_order": {"position_id": 56980461, "symbol": "NAS100", "quantity_lots": 0.1}},
        reason="ctrader_demo_order_executed",
    )
    legacy = open_paper_position(
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        direction="long",
        quantity=0.1,
        entry_price=29486.2,
        stop_loss=29476.4,
        take_profit=29511.7,
    )

    monkeypatch.setattr(
        "backend.services.reconciler.get_broker_status",
        lambda: type("S", (), {"execution_ready": True})(),
    )
    monkeypatch.setattr(
        "backend.services.reconciler.list_positions",
        lambda: [
            {
                "symbol": "NAS100",
                "direction": "buy",
                "volume_lots": 0.1,
                "entry_price": 29486.2,
                "position_id": 56980461,
            }
        ],
    )

    result = recover_demo_broker_trackers(config)

    assert result["recovered"] == 0
    assert result["attached"] == 1
    positions = list_paper_positions("open")
    assert len(positions) == 1
    assert positions[0].id == legacy.id
    assert positions[0].broker_position_id == 56980461


def test_demo_reconcile_does_not_replace_missing_persisted_id_with_same_side_position(monkeypatch) -> None:
    config = EngineConfig(
        enabled=True,
        demo_autotrade=True,
        watchlist=[
            WatchlistItem(
                symbol="XAUUSD",
                timeframe="M5",
                strategy="sma_cross",
                enabled=True,
                trading_enabled=True,
                lot_size=0.1,
                params={},
            )
        ],
    )
    save_engine_config(config)
    opened = open_paper_position(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        direction="long",
        quantity=0.1,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        broker_position_id=111,
    )

    def _fake_bars(symbol: str, timeframe: str, bars: int):
        import pandas as pd

        return pd.DataFrame(
            [{"close": 100.5}],
            index=pd.to_datetime(["2026-09-18T18:55:00Z"], utc=True),
        )

    monkeypatch.setattr("backend.services.reconciler.get_bars", _fake_bars)
    monkeypatch.setattr(
        "backend.services.reconciler.get_broker_status",
        lambda: type("S", (), {"execution_ready": True})(),
    )
    monkeypatch.setattr(
        "backend.services.reconciler.list_positions",
        lambda: [
            {
                "symbol": "XAUUSD",
                "direction": "buy",
                "volume_lots": 0.1,
                "entry_price": 100.0,
                "position_id": 222,
            }
        ],
    )
    closed = []
    monkeypatch.setattr(
        "backend.services.reconciler.close_local_position_from_broker",
        lambda position, **kwargs: closed.append(position.id),
    )
    monkeypatch.setattr(
        "backend.services.reconciler.sync_demo_position_targets",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("same-side broker position with a different id must not be adopted")
        ),
    )

    summary = reconcile_open_positions(reason="identity_test")

    assert summary["closed"] == 1
    assert closed == [opened.id]


def test_demo_reconcile_does_not_locally_close_while_broker_position_is_open(monkeypatch) -> None:
    config = EngineConfig(
        enabled=True,
        demo_autotrade=True,
        watchlist=[
            WatchlistItem(
                symbol="XAUUSD",
                timeframe="M5",
                strategy="sma_cross",
                enabled=True,
                trading_enabled=True,
                lot_size=0.1,
                params={},
            )
        ],
    )
    save_engine_config(config)
    open_paper_position(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        direction="long",
        quantity=0.1,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
    )

    def _fake_bars(symbol: str, timeframe: str, bars: int):
        import pandas as pd
        return pd.DataFrame(
            [{"close": 102.5}],
            index=pd.to_datetime(["2026-09-18T18:55:00Z"], utc=True),
        )

    monkeypatch.setattr("backend.services.reconciler.get_bars", _fake_bars)
    monkeypatch.setattr(
        "backend.services.reconciler.get_broker_status",
        lambda: type("S", (), {"execution_ready": True})(),
    )
    monkeypatch.setattr(
        "backend.services.reconciler.list_positions",
        lambda: [
            {
                "symbol": "XAUUSD",
                "direction": "buy",
                "volume_lots": 0.1,
                "entry_price": 100.0,
                "stop_loss": 99.0,
                "take_profit": 102.0,
                "position_id": 456,
            }
        ],
    )
    monkeypatch.setattr(
        "backend.services.reconciler.sync_demo_position_targets",
        lambda **kwargs: {"status": "already_synced", "verified": True},
    )

    summary = reconcile_open_positions(reason="demo_test")

    assert summary["closed"] == 0
    assert len(list_paper_positions("open")) == 1


def test_demo_reconcile_closes_broker_when_take_profit_was_already_crossed(monkeypatch) -> None:
    config = EngineConfig(
        enabled=True,
        demo_autotrade=True,
        watchlist=[
            WatchlistItem(
                symbol="NAS100",
                timeframe="M5",
                strategy="breakout",
                enabled=True,
                trading_enabled=True,
                lot_size=0.1,
                params={},
            )
        ],
    )
    save_engine_config(config)
    opened = open_paper_position(
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        direction="long",
        quantity=0.1,
        entry_price=29486.2,
        stop_loss=29476.4,
        take_profit=29511.7,
    )

    def _fake_bars(symbol: str, timeframe: str, bars: int):
        import pandas as pd
        return pd.DataFrame(
            [{"close": 29549.0}],
            index=pd.to_datetime(["2026-09-18T19:05:00Z"], utc=True),
        )

    broker_row = {
        "symbol": "NAS100",
        "direction": "buy",
        "volume_lots": 0.1,
        "entry_price": 29486.2,
        "stop_loss": None,
        "take_profit": None,
        "position_id": 56980461,
    }
    monkeypatch.setattr("backend.services.reconciler.get_bars", _fake_bars)
    monkeypatch.setattr(
        "backend.services.reconciler.get_broker_status",
        lambda: type("S", (), {"execution_ready": True})(),
    )
    monkeypatch.setattr("backend.services.reconciler.list_positions", lambda: [broker_row])
    monkeypatch.setattr(
        "backend.services.reconciler.sync_demo_position_targets",
        lambda **kwargs: {
            "status": "exit_due_take_profit",
            "position_id": 56980461,
            "quantity_lots": 0.1,
            "reference_price": 29549.0,
        },
    )
    close_calls = []
    monkeypatch.setattr(
        "backend.services.reconciler.close_demo_position",
        lambda **kwargs: close_calls.append(kwargs) or {
            "status": "closed",
            "position_id": 56980461,
            "verified": True,
        },
    )

    summary = reconcile_open_positions(reason="crossed_tp_test")

    assert summary["closed"] == 1
    assert close_calls[0]["position_id"] == 56980461
    assert list_paper_positions("open") == []
    closed = list_paper_positions("closed")
    assert len(closed) == 1
    assert closed[0].id == opened.id
    assert closed[0].close_reason == "broker_take_profit"
