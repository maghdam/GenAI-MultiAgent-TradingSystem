from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd

from backend.domain.models import EngineConfig, StrategyAnalysis, WatchlistItem
from backend.services import engine as engine_module
from backend.services.engine import V2Engine
from backend.services.execution_engine import execute_paper_signal
from backend.storage.repositories import (
    list_order_intents,
    list_paper_positions,
    load_engine_config,
    save_engine_config,
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
        reasons=["test signal"],
    )


def _config() -> EngineConfig:
    return EngineConfig(
        enabled=True,
        paper_autotrade=False,
        demo_autotrade=True,
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
        "backend.services.execution_engine.get_demo_symbol_execution_readiness",
        lambda symbol: (True, "ready"),
    )
    monkeypatch.setattr(
        "backend.services.execution_engine.sync_demo_position_targets",
        lambda **kwargs: {
            "status": "synced",
            "position_id": kwargs.get("position_id"),
            "quantity_lots": kwargs.get("quantity_lots"),
        },
    )


def test_per_symbol_lot_routes_verified_demo_order_and_tracks_position(monkeypatch) -> None:
    _mock_demo_ready(monkeypatch)
    captured = {}

    def _place(**kwargs):
        captured.update(kwargs)
        return {"status": "executed", "position_id": 321, "account_type": "demo"}

    monkeypatch.setattr("backend.services.execution_engine.place_demo_market_order", _place)

    result = execute_paper_signal(
        config=_config(),
        watch_item=_watch(),
        analysis=_analysis(),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC).replace(tzinfo=None),
        bar_snapshot={"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0},
    )

    assert result.action_taken is True
    assert result.status == "executed"
    assert result.mode == "demo_enabled"
    assert result.broker_position_id == 321
    assert captured["quantity_lots"] == 0.25
    positions = list_paper_positions("open")
    assert len(positions) == 1
    assert positions[0].quantity == 0.25
    intents = list_order_intents(5)
    assert intents[0].details["execution_mode"] == "ctrader_demo"


def test_unprotected_demo_order_is_closed_by_failsafe(monkeypatch) -> None:
    _mock_demo_ready(monkeypatch)
    captured_close = {}

    monkeypatch.setattr(
        "backend.services.execution_engine.place_demo_market_order",
        lambda **kwargs: {
            "status": "executed",
            "position_id": 321,
            "account_type": "demo",
            "quantity_lots": 0.25,
        },
    )
    monkeypatch.setattr(
        "backend.services.execution_engine.sync_demo_position_targets",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("protection verification failed")),
    )

    def _close(**kwargs):
        captured_close.update(kwargs)
        return {
            "status": "closed",
            "verified": True,
            "position_id": kwargs["position_id"],
            "quantity_lots": kwargs["quantity_lots"],
        }

    monkeypatch.setattr("backend.services.execution_engine.close_demo_position", _close)

    result = execute_paper_signal(
        config=_config(),
        watch_item=_watch(),
        analysis=_analysis(),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC).replace(tzinfo=None),
        bar_snapshot={"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0},
    )

    assert result.action_taken is True
    assert result.status == "failed"
    assert result.retryable is False
    assert result.broker_position_id == 321
    assert captured_close["position_id"] == 321
    assert captured_close["quantity_lots"] == 0.25
    assert list_paper_positions("open") == []

    intents = list_order_intents(5)
    assert intents[0].status == "failed"
    assert intents[0].details["failsafe_closed"] is True
    assert intents[0].details["failsafe_close"]["verified"] is True


def test_unprotected_demo_order_close_failure_retains_tracker_for_retry(monkeypatch) -> None:
    _mock_demo_ready(monkeypatch)

    monkeypatch.setattr(
        "backend.services.execution_engine.place_demo_market_order",
        lambda **kwargs: {
            "status": "executed",
            "position_id": 654,
            "account_type": "demo",
            "quantity_lots": 0.25,
        },
    )
    monkeypatch.setattr(
        "backend.services.execution_engine.sync_demo_position_targets",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("protection verification failed")),
    )
    monkeypatch.setattr(
        "backend.services.execution_engine.close_demo_position",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("broker close verification failed")),
    )

    result = execute_paper_signal(
        config=_config(),
        watch_item=_watch(),
        analysis=_analysis(),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC).replace(tzinfo=None),
        bar_snapshot={"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0},
    )

    assert result.action_taken is True
    assert result.status == "failed"
    assert result.retryable is True
    assert result.broker_position_id == 654

    positions = list_paper_positions("open")
    assert len(positions) == 1
    assert positions[0].broker_position_id == 654
    assert positions[0].quantity == 0.25

    intents = list_order_intents(5)
    assert intents[0].status == "failed"
    assert intents[0].details["tracking_retained"] is True
    assert "broker close verification failed" in intents[0].details["failsafe_close_error"]


def test_demo_order_failure_does_not_create_local_position(monkeypatch) -> None:
    _mock_demo_ready(monkeypatch)
    monkeypatch.setattr(
        "backend.services.execution_engine.place_demo_market_order",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("demo account not confirmed")),
    )

    result = execute_paper_signal(
        config=_config(),
        watch_item=_watch(),
        analysis=_analysis(),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC).replace(tzinfo=None),
        bar_snapshot={"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0},
    )

    assert result.action_taken is False
    assert result.status == "failed"
    assert "not confirmed" in result.summary
    assert list_paper_positions("open") == []
    intents = list_order_intents(5)
    assert intents[0].status == "failed"


def test_engine_uses_enabled_row_strategy_and_preserves_per_symbol_lot(monkeypatch) -> None:
    watch = _watch().model_copy(update={"strategy": "rsi_reversal", "lot_size": 0.4})
    config = _config().model_copy(update={"watchlist": [watch]})
    save_engine_config(config)
    bars = pd.DataFrame(
        [{"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0, "volume": 12.0}],
        index=pd.to_datetime(["2026-09-16T10:05:00Z"], utc=True),
    )
    captured = {}

    class _Strategy:
        def analyze(self, **kwargs):
            captured["analysis_kwargs"] = kwargs
            return _analysis().model_copy(update={"strategy": "rsi_reversal"})

    def _execute(**kwargs):
        captured["watch_item"] = kwargs["watch_item"]
        return SimpleNamespace(action_taken=True, retryable=False)

    monkeypatch.setattr(engine_module, "get_bars", lambda *args, **kwargs: bars)
    monkeypatch.setattr(
        engine_module,
        "get_strategy",
        lambda name: captured.update(strategy=name) or _Strategy(),
    )
    monkeypatch.setattr(engine_module, "execute_paper_signal", _execute)
    monkeypatch.setattr(engine_module, "record_confluence_shadow", lambda *args, **kwargs: None)

    summary = asyncio.run(V2Engine().run_once())

    assert "processed=1" in summary
    assert captured["strategy"] == "rsi_reversal"
    assert captured["analysis_kwargs"]["symbol"] == "XAUUSD"
    assert captured["watch_item"].lot_size == 0.4
    assert captured["watch_item"].trading_enabled is True


def test_demo_symbol_not_ready_defers_without_creating_intent(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.services.execution_engine.get_demo_symbol_execution_readiness",
        lambda symbol: (False, "Connected cTrader account is not confirmed as demo."),
    )
    monkeypatch.setattr(
        "backend.services.execution_engine.place_demo_market_order",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("order must not be routed before demo readiness")),
    )

    result = execute_paper_signal(
        config=_config(),
        watch_item=_watch(),
        analysis=_analysis(),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC).replace(tzinfo=None),
        bar_snapshot={"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0},
    )

    assert result.action_taken is False
    assert result.status == "deferred"
    assert result.retryable is True
    assert result.intent_id is None
    assert "not confirmed as demo" in result.summary
    assert list_paper_positions("open") == []
    assert list_order_intents(5) == []


def test_disabled_watchlist_row_does_not_generate_signal(monkeypatch) -> None:
    watch = _watch().model_copy(update={"enabled": False})
    save_engine_config(_config().model_copy(update={"watchlist": [watch]}))
    monkeypatch.setattr(
        engine_module,
        "get_bars",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("disabled row must not fetch bars")),
    )

    summary = asyncio.run(V2Engine().run_once())

    assert summary == "watchlist empty"


def test_legacy_watchlist_row_inherits_original_trade_size_as_lot_size() -> None:
    legacy = WatchlistItem(
        symbol="EURUSD",
        timeframe="M15",
        strategy="sma_cross",
        enabled=True,
        lot_size=None,
    )
    save_engine_config(EngineConfig(paper_trade_size=0.17, watchlist=[legacy]))

    loaded = load_engine_config(EngineConfig())

    assert loaded.watchlist[0].lot_size == 0.17
