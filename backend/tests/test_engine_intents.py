from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from backend.domain.models import EngineConfig, StrategyAnalysis, SymbolLimits, WatchlistItem
from backend.services import engine as engine_module
from backend.services.engine import V2Engine
from backend.services.market_data import MarketDataError
from backend.services.execution_engine import execute_paper_signal
from backend.storage.repositories import (
    list_incidents,
    list_order_intents,
    list_paper_positions,
    list_trade_audits,
    load_bar_state,
    open_paper_position,
    save_engine_config,
)


def _config(**overrides) -> EngineConfig:
    payload = EngineConfig(
        enabled=True,
        paper_autotrade=True,
        kill_switch=False,
        require_stops=True,
        max_open_positions=3,
        max_positions_per_symbol=1,
        cooldown_minutes=0,
    ).model_dump()
    payload.update(overrides)
    return EngineConfig(**payload)


def _watch_item() -> WatchlistItem:
    return WatchlistItem(symbol="XAUUSD", timeframe="M5", strategy="sma_cross", enabled=True, params={})


def _analysis(signal: str = "long", confidence: float = 0.82) -> StrategyAnalysis:
    stop_loss = 99.0 if signal != "short" else 101.0
    take_profit = 102.0 if signal != "short" else 98.0
    return StrategyAnalysis(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        signal=signal,
        confidence=confidence,
        entry_price=100.0,
        stop_loss=stop_loss,
        take_profit=take_profit,
        reasons=["deterministic test signal"],
        context={},
    )


def test_apply_paper_logic_opens_position_and_records_execution() -> None:
    result = execute_paper_signal(
        config=_config(),
        watch_item=_watch_item(),
        analysis=_analysis(),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC).replace(tzinfo=None),
        bar_snapshot={"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0},
    )

    assert result.action_taken is True
    assert result.status == "executed"

    positions = list_paper_positions("open")
    assert len(positions) == 1
    assert positions[0].symbol == "XAUUSD"
    assert positions[0].direction == "long"
    assert positions[0].quantity == 0.5

    intents = list_order_intents(5)
    assert len(intents) == 1
    assert intents[0].intent_type == "open"
    assert intents[0].status == "executed"

    audits = list_trade_audits(10)
    assert any(record.event_type == "paper_signal_open" for record in audits)
    assert any(record.event_type == "paper_position_opened" for record in audits)


def test_apply_paper_logic_rejects_signal_and_records_reason() -> None:
    result = execute_paper_signal(
        config=_config(paper_autotrade=False),
        watch_item=_watch_item(),
        analysis=_analysis(),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC).replace(tzinfo=None),
        bar_snapshot={"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0},
    )

    assert result.action_taken is False
    assert result.status == "rejected"
    assert list_paper_positions("open") == []

    intents = list_order_intents(5)
    assert len(intents) == 1
    assert intents[0].status == "rejected"
    assert "Paper autotrade is disabled." in intents[0].rationale

    audits = list_trade_audits(10)
    assert len(audits) == 1
    assert audits[0].event_type == "paper_signal_rejected"

    incidents = list_incidents(5)
    assert len(incidents) == 1
    assert incidents[0].code == "signal_rejected"


def test_execute_paper_signal_rejects_invalid_long_protective_levels() -> None:
    result = execute_paper_signal(
        config=_config(),
        watch_item=_watch_item(),
        analysis=StrategyAnalysis(
            symbol="XAUUSD",
            timeframe="M5",
            strategy="sma_cross",
            signal="long",
            confidence=0.9,
            entry_price=100.0,
            stop_loss=101.0,
            take_profit=99.0,
            reasons=["invalid geometry"],
            context={},
        ),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC).replace(tzinfo=None),
        bar_snapshot={"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0},
    )

    assert result.action_taken is False
    assert result.status == "rejected"
    assert list_paper_positions("open") == []

    intents = list_order_intents(5)
    assert len(intents) == 1
    assert intents[0].status == "rejected"
    assert "Invalid protective levels for a long signal." in intents[0].rationale
    assert intents[0].details["entry_reference"] == 100.0

    incidents = list_incidents(5)
    assert len(incidents) == 1
    assert incidents[0].code == "signal_rejected"


def test_execute_paper_signal_flips_and_reopens_new_direction() -> None:
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

    result = execute_paper_signal(
        config=_config(),
        watch_item=_watch_item(),
        analysis=_analysis(signal="short"),
        mark_price=100.5,
        bar_timestamp=datetime.now(UTC).replace(tzinfo=None),
        bar_snapshot={"open": 100.4, "high": 100.8, "low": 100.1, "close": 100.5},
    )

    assert result.action_taken is True
    assert result.summary == "position flipped and opened"

    open_positions = list_paper_positions("open")
    closed_positions = list_paper_positions("closed")
    assert len(open_positions) == 1
    assert open_positions[0].direction == "short"
    assert len(closed_positions) == 1
    assert closed_positions[0].close_reason == "signal_flip"

    intents = list_order_intents(5)
    assert intents[0].status == "executed"
    assert intents[0].details["flip"] is True


def test_execute_paper_signal_updates_existing_same_direction_position() -> None:
    open_paper_position(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        direction="long",
        quantity=1.0,
        entry_price=100.0,
        stop_loss=98.5,
        take_profit=101.0,
    )

    result = execute_paper_signal(
        config=_config(),
        watch_item=_watch_item(),
        analysis=_analysis(signal="long"),
        mark_price=100.5,
        bar_timestamp=datetime.now(UTC).replace(tzinfo=None),
        bar_snapshot={"open": 100.2, "high": 100.7, "low": 99.9, "close": 100.5},
    )

    assert result.action_taken is True
    assert result.summary == "position updated"

    open_positions = list_paper_positions("open")
    assert len(open_positions) == 1
    assert open_positions[0].direction == "long"
    assert open_positions[0].stop_loss == 99.0
    assert open_positions[0].take_profit == 102.0

    intents = list_order_intents(5)
    assert intents[0].status == "executed"
    assert intents[0].details["updated_position_id"] == open_positions[0].id

    audits = list_trade_audits(10)
    assert any(record.event_type == "paper_signal_update" for record in audits)


def test_run_once_counts_market_data_skip_without_logging_incident(monkeypatch) -> None:
    save_engine_config(
        EngineConfig(
            enabled=True,
            paper_autotrade=True,
            kill_switch=False,
            default_symbol="XAUUSD",
            default_timeframe="M5",
            watchlist=[_watch_item()],
        )
    )

    def _raise_market_data(*args, **kwargs):
        raise MarketDataError("cTrader feed not connected")

    monkeypatch.setattr(engine_module, "get_bars", _raise_market_data)

    summary = asyncio.run(V2Engine().run_once())

    assert "market_skips=1" in summary
    assert list_incidents(5) == []


def test_run_once_does_not_advance_bar_state_when_execution_raises(monkeypatch) -> None:
    import pandas as pd

    save_engine_config(
        EngineConfig(
            enabled=True,
            paper_autotrade=True,
            kill_switch=False,
            default_symbol="XAUUSD",
            default_timeframe="M5",
            watchlist=[_watch_item()],
        )
    )

    bars = pd.DataFrame(
        [{"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0}],
        index=pd.to_datetime(["2026-03-19T10:05:00Z"], utc=True),
    )

    class _FakeStrategy:
        def analyze(self, **kwargs):
            return _analysis(signal="long")

    def _raise_execution(**kwargs):
        raise RuntimeError("paper execution exploded")

    monkeypatch.setattr(engine_module, "get_bars", lambda *args, **kwargs: bars)
    monkeypatch.setattr(engine_module, "get_strategy", lambda name: _FakeStrategy())
    monkeypatch.setattr(engine_module, "execute_paper_signal", _raise_execution)

    summary = asyncio.run(V2Engine().run_once())

    assert "processed=0" in summary
    assert load_bar_state() == {}

    incidents = list_incidents(5)
    assert len(incidents) == 1
    assert incidents[0].code == "scan_failure"


def test_execute_paper_signal_rejects_stale_market_bar() -> None:
    stale_bar = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=20)

    result = execute_paper_signal(
        config=_config(),
        watch_item=_watch_item(),
        analysis=_analysis(signal="long"),
        mark_price=100.0,
        bar_timestamp=stale_bar,
        bar_snapshot={"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0},
    )

    assert result.action_taken is False
    assert result.status == "rejected"

    intents = list_order_intents(5)
    assert len(intents) == 1
    assert "Latest market bar is stale for the configured timeframe." in intents[0].rationale
    assert intents[0].details["bar_age_seconds"] > intents[0].details["max_bar_age_seconds"]


def test_execute_paper_signal_rejects_extreme_bar_range() -> None:
    result = execute_paper_signal(
        config=_config(),
        watch_item=_watch_item(),
        analysis=_analysis(signal="long"),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC).replace(tzinfo=None),
        bar_snapshot={"open": 100.0, "high": 104.0, "low": 96.0, "close": 100.0},
    )

    assert result.action_taken is False
    assert result.status == "rejected"

    intents = list_order_intents(5)
    assert len(intents) == 1
    assert "Latest market bar range is too wide for the configured timeframe." in intents[0].rationale
    assert intents[0].details["bar_range_pct"] > intents[0].details["max_bar_range_pct"]


def test_execute_paper_signal_normalizes_auto_quantity_to_symbol_step(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.services.quantity_rules.get_symbol_limits",
        lambda symbol: SymbolLimits(
            symbol=symbol.upper(),
            source="broker",
            min_lots=0.05,
            step_lots=0.05,
            max_lots=2.0,
            min_api_units=500,
            step_api_units=500,
            max_api_units=20000,
            hard_min=True,
            hard_step=True,
        ),
    )

    result = execute_paper_signal(
        config=_config(paper_trade_size=0.12, risk_per_trade_pct=0),
        watch_item=_watch_item(),
        analysis=_analysis(signal="long"),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC).replace(tzinfo=None),
        bar_snapshot={"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0},
    )

    assert result.action_taken is True
    assert result.status == "executed"

    positions = list_paper_positions("open")
    assert len(positions) == 1
    assert positions[0].quantity == 0.15

    intents = list_order_intents(5)
    assert len(intents) == 1
    assert intents[0].quantity == 0.15
    assert intents[0].details["quantity_normalized"] is True
    assert intents[0].details["requested_quantity"] == 0.12
    assert intents[0].details["final_quantity"] == 0.15


def test_execute_paper_signal_uses_risk_based_auto_quantity(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.services.quantity_rules.get_symbol_limits",
        lambda symbol: SymbolLimits(
            symbol=symbol.upper(),
            source="broker",
            min_lots=0.01,
            step_lots=0.01,
            max_lots=5.0,
            min_api_units=100,
            step_api_units=100,
            max_api_units=50000,
            hard_min=False,
            hard_step=False,
        ),
    )

    result = execute_paper_signal(
        config=_config(risk_per_trade_pct=0.5),
        watch_item=_watch_item(),
        analysis=_analysis(signal="long"),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC).replace(tzinfo=None),
        bar_snapshot={"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0},
    )

    assert result.action_taken is True
    positions = list_paper_positions("open")
    assert len(positions) == 1
    assert positions[0].quantity == 0.5

    intents = list_order_intents(5)
    assert len(intents) == 1
    assert "Auto quantity derived from 0.50% risk" in intents[0].rationale
    assert intents[0].details["sizing_mode"] == "risk_based"
    assert intents[0].details["raw_risk_quantity"] == 0.5
    assert intents[0].details["stop_pct"] == 1.0


def test_execute_paper_signal_rejects_manual_quantity_outside_symbol_step(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.services.quantity_rules.get_symbol_limits",
        lambda symbol: SymbolLimits(
            symbol=symbol.upper(),
            source="broker",
            min_lots=0.05,
            step_lots=0.05,
            max_lots=2.0,
            min_api_units=500,
            step_api_units=500,
            max_api_units=20000,
            hard_min=True,
            hard_step=True,
        ),
    )

    result = execute_paper_signal(
        config=_config(),
        watch_item=_watch_item(),
        analysis=_analysis(signal="long"),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC).replace(tzinfo=None),
        bar_snapshot={"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0},
        quantity=0.12,
        source="manual",
    )

    assert result.action_taken is False
    assert result.status == "rejected"
    assert "step size of 0.0500 lots" in result.summary

    positions = list_paper_positions("open")
    assert positions == []

    intents = list_order_intents(5)
    assert len(intents) == 1
    assert intents[0].status == "rejected"
    assert "step size of 0.0500 lots" in intents[0].rationale
    assert intents[0].details["requested_quantity"] == 0.12
