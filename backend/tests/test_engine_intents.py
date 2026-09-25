from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from backend.domain.models import EngineConfig, InstrumentSpec, StrategyAnalysis, SymbolLimits, WatchlistItem
from backend.services import engine as engine_module
from backend.services.engine import V2Engine
from backend.services.market_data import MarketDataError
from backend.services.execution_engine import execute_paper_signal
from backend.storage.repositories import (
    list_incidents,
    list_decision_records,
    list_order_intents,
    list_order_intent_transitions,
    list_paper_positions,
    list_trade_audits,
    load_bar_state,
    open_paper_position,
    save_bar_state,
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


@pytest.fixture(autouse=True)
def valued_xau_contract(monkeypatch):
    spec = InstrumentSpec(
        symbol="XAUUSD",
        source="test_contract",
        account_currency="USD",
        quote_currency="USD",
        lot_size_units=100.0,
        tick_size=0.01,
        tick_value_per_lot=1.0,
        cash_per_price_unit_per_lot=100.0,
        conversion_rate_to_account=1.0,
        valuation_ready=True,
        verified=True,
    )
    monkeypatch.setattr("backend.services.quantity_rules.get_instrument_spec", lambda symbol, currency: spec)
    monkeypatch.setattr("backend.services.execution_engine.get_instrument_spec", lambda symbol, currency: spec)


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
    assert positions[0].quantity == 5.0

    intents = list_order_intents(5)
    assert len(intents) == 1
    assert intents[0].intent_type == "open"
    assert intents[0].status == "executed"
    assert intents[0].decision_id is not None

    decisions = list_decision_records(5)
    assert len(decisions) == 1
    assert decisions[0].outcome == "accepted_open"
    assert decisions[0].evidence["risk_amount"] == 500.0

    transitions = list_order_intent_transitions(intents[0].id)
    assert [transition.to_status for transition in transitions] == ["accepted", "executed"]
    assert transitions[-1].reason == "paper_position_opened"

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


def test_run_once_repairs_demo_protection_on_already_processed_bar(monkeypatch) -> None:
    import pandas as pd

    item = _watch_item().model_copy(update={"trading_enabled": True, "lot_size": 0.1})
    save_engine_config(
        EngineConfig(
            enabled=True,
            paper_autotrade=False,
            demo_autotrade=True,
            kill_switch=False,
            default_symbol="XAUUSD",
            default_timeframe="M5",
            watchlist=[item],
        )
    )
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
    bars = pd.DataFrame(
        [{"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0}],
        index=pd.to_datetime(["2026-09-18T18:50:00Z"], utc=True),
    )
    last_ts = int(bars.index[-1].timestamp())
    save_bar_state({"XAUUSD|M5": last_ts})

    calls = []
    monkeypatch.setattr(engine_module, "get_bars", lambda *args, **kwargs: bars)
    monkeypatch.setattr(engine_module, "recover_demo_broker_trackers", lambda config: {"ready": True})
    monkeypatch.setattr(
        engine_module,
        "get_broker_status",
        lambda: type("S", (), {"execution_ready": True})(),
    )
    monkeypatch.setattr(
        engine_module,
        "list_positions",
        lambda: [
            {
                "symbol": "XAUUSD",
                "direction": "buy",
                "volume_lots": 0.1,
                "entry_price": 100.0,
                "stop_loss": None,
                "take_profit": None,
                "position_id": 456,
            }
        ],
    )
    monkeypatch.setattr(
        engine_module,
        "sync_demo_position_targets",
        lambda **kwargs: calls.append(kwargs) or {"status": "synced", "verified": True, "position_id": 456},
    )

    summary = asyncio.run(V2Engine().run_once())

    assert "processed=0" in summary
    assert len(calls) == 1
    assert calls[0]["stop_loss"] == 99.0
    assert calls[0]["take_profit"] == 102.0
    audits = list_trade_audits(10)
    assert any(record.event_type == "ctrader_demo_protection_repaired" for record in audits)


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


def test_execute_paper_signal_accepts_timezone_aware_fresh_bar() -> None:
    result = execute_paper_signal(
        config=_config(),
        watch_item=_watch_item(),
        analysis=_analysis(signal="long"),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC),
        bar_snapshot={"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0},
    )

    assert result.action_taken is True
    assert result.status == "executed"
    intents = list_order_intents(5)
    assert len(intents) == 1
    assert intents[0].details["bar_age_seconds"] >= 0


def test_demo_execution_defers_until_symbol_metadata_is_ready(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.services.execution_engine.get_demo_symbol_execution_readiness",
        lambda symbol: (False, "Broker lotSize metadata is not loaded yet for XAUUSD."),
    )
    result = execute_paper_signal(
        config=_config(paper_autotrade=False, demo_autotrade=True),
        watch_item=_watch_item().model_copy(update={"trading_enabled": True}),
        analysis=_analysis(signal="long"),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC),
        bar_snapshot={"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0},
    )

    assert result.action_taken is False
    assert result.status == "deferred"
    assert result.retryable is True
    assert result.intent_id is None
    assert list_paper_positions("open") == []
    assert list_order_intents(5) == []

    incidents = list_incidents(5)
    assert incidents[0].code == "ctrader_demo_symbol_not_ready"


def test_demo_existing_position_sync_uses_persisted_broker_position_id(monkeypatch) -> None:
    open_paper_position(
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
    calls = []
    monkeypatch.setattr(
        "backend.services.execution_engine.get_broker_status",
        lambda: type("S", (), {"execution_ready": True})(),
    )
    monkeypatch.setattr(
        "backend.services.execution_engine.list_positions",
        lambda: [
            {"position_id": 222, "symbol": "XAUUSD", "direction": "buy", "volume_lots": 0.1},
            {"position_id": 111, "symbol": "XAUUSD", "direction": "buy", "volume_lots": 0.1},
        ],
    )
    monkeypatch.setattr(
        "backend.services.execution_engine.sync_demo_position_targets",
        lambda **kwargs: calls.append(kwargs) or {"status": "already_synced", "verified": True, "position_id": 111},
    )

    result = execute_paper_signal(
        config=_config(paper_autotrade=False, demo_autotrade=True),
        watch_item=_watch_item().model_copy(update={"trading_enabled": True, "lot_size": 0.1}),
        analysis=StrategyAnalysis(
            symbol="XAUUSD",
            timeframe="M5",
            strategy="sma_cross",
            signal="no_trade",
            confidence=0.0,
            entry_price=100.0,
            reasons=["no trade"],
        ),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC),
        bar_snapshot={"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0},
    )

    assert result.status == "rejected"
    assert len(calls) == 1
    assert calls[0]["position_id"] == 111


def test_demo_existing_position_does_not_hijack_different_same_side_broker_position(monkeypatch) -> None:
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
    closed = []
    monkeypatch.setattr(
        "backend.services.execution_engine.get_broker_status",
        lambda: type("S", (), {"execution_ready": True})(),
    )
    monkeypatch.setattr(
        "backend.services.execution_engine.list_positions",
        lambda: [
            {"position_id": 222, "symbol": "XAUUSD", "direction": "buy", "volume_lots": 0.1},
        ],
    )
    monkeypatch.setattr(
        "backend.services.execution_engine.close_local_position_from_broker",
        lambda position, **kwargs: closed.append(position.id),
    )
    monkeypatch.setattr(
        "backend.services.execution_engine.sync_demo_position_targets",
        lambda **kwargs: pytest.fail("different broker position id must not be used for protection"),
    )

    result = execute_paper_signal(
        config=_config(paper_autotrade=False, demo_autotrade=True),
        watch_item=_watch_item().model_copy(update={"trading_enabled": True, "lot_size": 0.1}),
        analysis=StrategyAnalysis(
            symbol="XAUUSD",
            timeframe="M5",
            strategy="sma_cross",
            signal="no_trade",
            confidence=0.0,
            entry_price=100.0,
            reasons=["no trade"],
        ),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC),
        bar_snapshot={"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0},
    )

    assert result.status == "rejected"
    assert closed == [opened.id]


def test_demo_legacy_position_refuses_ambiguous_symbol_direction_identity(monkeypatch) -> None:
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
    monkeypatch.setattr(
        "backend.services.execution_engine.get_broker_status",
        lambda: type("S", (), {"execution_ready": True})(),
    )
    monkeypatch.setattr(
        "backend.services.execution_engine.list_positions",
        lambda: [
            {"position_id": 111, "symbol": "XAUUSD", "direction": "buy", "volume_lots": 0.1},
            {"position_id": 222, "symbol": "XAUUSD", "direction": "buy", "volume_lots": 0.1},
        ],
    )
    monkeypatch.setattr(
        "backend.services.execution_engine.sync_demo_position_targets",
        lambda **kwargs: pytest.fail("ambiguous legacy identity must block broker protection"),
    )

    result = execute_paper_signal(
        config=_config(paper_autotrade=False, demo_autotrade=True),
        watch_item=_watch_item().model_copy(update={"trading_enabled": True, "lot_size": 0.1}),
        analysis=StrategyAnalysis(
            symbol="XAUUSD",
            timeframe="M5",
            strategy="sma_cross",
            signal="no_trade",
            confidence=0.0,
            entry_price=100.0,
            reasons=["no trade"],
        ),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC),
        bar_snapshot={"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0},
    )

    assert result.action_taken is False
    assert result.status == "failed"
    assert result.retryable is True
    incidents = list_incidents(5)
    assert incidents[0].code == "ctrader_demo_position_identity_ambiguous"


def test_demo_existing_position_repairs_broker_protection_even_on_no_trade(monkeypatch) -> None:
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
    calls = []
    monkeypatch.setattr(
        "backend.services.execution_engine.sync_demo_position_targets",
        lambda **kwargs: calls.append(kwargs) or {"status": "synced", "verified": True, "position_id": 456},
    )

    result = execute_paper_signal(
        config=_config(paper_autotrade=False, demo_autotrade=True),
        watch_item=_watch_item().model_copy(update={"trading_enabled": True, "lot_size": 0.1}),
        analysis=StrategyAnalysis(
            symbol="XAUUSD",
            timeframe="M5",
            strategy="sma_cross",
            signal="no_trade",
            confidence=0.0,
            entry_price=100.0,
            reasons=["no trade"],
        ),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC),
        bar_snapshot={"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0},
    )

    assert result.status == "rejected"
    assert len(calls) == 1
    assert calls[0]["stop_loss"] == 99.0
    assert calls[0]["take_profit"] == 102.0

    audits = list_trade_audits(10)
    assert any(record.event_type == "ctrader_demo_protection_repaired" for record in audits)


def test_demo_target_update_keeps_local_targets_when_broker_sync_fails(monkeypatch) -> None:
    opened = open_paper_position(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        direction="long",
        quantity=0.1,
        entry_price=100.0,
        stop_loss=98.5,
        take_profit=101.0,
    )
    calls = []

    def _sync(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {"status": "already_synced", "verified": True, "position_id": 456}
        raise RuntimeError("broker amend rejected")

    monkeypatch.setattr("backend.services.execution_engine.sync_demo_position_targets", _sync)
    monkeypatch.setattr(
        "backend.services.execution_engine.get_demo_symbol_execution_readiness",
        lambda symbol: (True, "Broker symbol contract metadata is ready."),
    )

    result = execute_paper_signal(
        config=_config(paper_autotrade=False, demo_autotrade=True),
        watch_item=_watch_item().model_copy(update={"trading_enabled": True, "lot_size": 0.1}),
        analysis=_analysis(signal="long"),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC),
        bar_snapshot={"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0},
    )

    assert result.status == "failed"
    refreshed = list_paper_positions("open")[0]
    assert refreshed.id == opened.id
    assert refreshed.stop_loss == 98.5
    assert refreshed.take_profit == 101.0

    intents = list_order_intents(5)
    assert intents[0].status == "failed"
    incidents = list_incidents(5)
    assert any(item.code == "ctrader_demo_target_update_failed" for item in incidents)


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
    assert positions[0].quantity == 5.0

    intents = list_order_intents(5)
    assert len(intents) == 1
    assert "Auto quantity derived from 0.50% risk" in intents[0].rationale
    assert intents[0].details["sizing_mode"] == "risk_based"
    assert intents[0].details["raw_risk_quantity"] == 5.0
    assert intents[0].details["risk_amount"] == 500.0
    assert intents[0].details["loss_per_lot_at_stop"] == 100.0
    assert intents[0].details["stop_pct"] == 1.0


def test_execute_paper_signal_blocks_auto_trade_when_contract_cannot_be_valued(monkeypatch) -> None:
    unvalued = InstrumentSpec(
        symbol="XAUUSD",
        source="fallback",
        account_currency="USD",
        quote_currency="USD",
        valuation_ready=False,
        verified=False,
        notes=["lotSize missing"],
    )
    monkeypatch.setattr("backend.services.quantity_rules.get_instrument_spec", lambda symbol, currency: unvalued)
    monkeypatch.setattr("backend.services.execution_engine.get_instrument_spec", lambda symbol, currency: unvalued)

    result = execute_paper_signal(
        config=_config(risk_per_trade_pct=0.5),
        watch_item=_watch_item(),
        analysis=_analysis(signal="long"),
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC).replace(tzinfo=None),
        bar_snapshot={"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0},
    )

    assert result.action_taken is False
    assert result.status == "rejected"
    assert "contract valuation is unavailable" in result.summary
    assert list_paper_positions("open") == []
    decisions = list_decision_records(5)
    assert decisions[0].outcome == "rejected_sizing"


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
