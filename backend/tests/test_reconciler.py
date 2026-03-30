from __future__ import annotations

from backend.domain.models import EngineConfig, EngineRuntime, WatchlistItem
from backend.services.market_data import MarketDataError
from backend.services.reconciler import reconcile_open_positions, recover_runtime_state
from backend.storage.repositories import (
    list_incidents,
    list_paper_positions,
    load_runtime,
    open_paper_position,
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
