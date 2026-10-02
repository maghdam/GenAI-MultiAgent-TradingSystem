from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from backend.domain.models import EngineConfig, MarketEventInput, StrategyAnalysis, WatchlistItem
from backend.services import engine as engine_module
from backend.services import market_data, risk_engine
from backend.services.engine import V2Engine
from backend.services.event_calibration import evaluate_event_against_bars
from backend.services.event_intelligence import ingest_events
from backend.services.execution_engine import ExecutionResult
from backend.storage import repositories
from backend.storage.db import get_db
from backend.storage.repositories import (
    close_paper_position,
    daily_realized_pnl,
    daily_trade_count,
    get_position_by_id,
    open_paper_position,
    reconcile_closed_paper_position_from_broker,
    record_broker_deals,
)


ZURICH = ZoneInfo("Europe/Zurich")


def _session_config(start: int, end: int) -> EngineConfig:
    return EngineConfig(
        session_filter_enabled=True,
        session_start_hour_utc=start,
        session_end_hour_utc=end,
    )


def _bars(at: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1.0}],
        index=pd.DatetimeIndex([at]),
    )


def test_session_filter_uses_utc_across_zurich_dst_transitions() -> None:
    config = _session_config(1, 2)

    # Spring-forward: 03:30 CEST is 01:30 UTC and must be inside 01:00-02:00 UTC.
    spring_local = datetime(2026, 3, 29, 3, 30, tzinfo=ZURICH)
    assert spring_local.utcoffset() == timedelta(hours=2)
    assert risk_engine._within_session(config, spring_local) is True

    # Fall-back repeated 02:30 occurs twice. Only the second occurrence is 01:30 UTC.
    autumn_first = datetime(2026, 10, 25, 2, 30, tzinfo=ZURICH, fold=0)
    autumn_second = datetime(2026, 10, 25, 2, 30, tzinfo=ZURICH, fold=1)
    assert autumn_first.utcoffset() == timedelta(hours=2)
    assert autumn_second.utcoffset() == timedelta(hours=1)
    assert risk_engine._within_session(config, autumn_first) is False
    assert risk_engine._within_session(config, autumn_second) is True


def test_market_freshness_normalizes_offsets_and_rejects_future_clock_skew() -> None:
    now = datetime(2026, 3, 29, 1, 35, tzinfo=UTC)
    same_clock_bar = datetime(2026, 3, 29, 3, 30, tzinfo=ZURICH)

    ok, details, reason = market_data.assess_market_bar_freshness(
        "M5",
        same_clock_bar,
        now=now,
    )

    assert ok is True
    assert reason is None
    assert details["bar_age_seconds"] == pytest.approx(300.0)
    assert details["bar_clock_skew_seconds"] == pytest.approx(0.0)

    future_bar = datetime(2026, 3, 29, 3, 36, tzinfo=ZURICH)
    future_ok, future_details, future_reason = market_data.assess_market_bar_freshness(
        "M5",
        future_bar,
        now=now,
    )

    assert future_ok is False
    assert "ahead of the evaluation clock" in str(future_reason)
    assert future_details["bar_clock_skew_seconds"] == pytest.approx(60.0)


def test_persistent_cache_freshness_converts_offset_before_age_comparison() -> None:
    now = datetime(2026, 3, 29, 1, 35, tzinfo=UTC)

    # 03:00 CEST == 01:00 UTC, so this M5 cache is 35 minutes old and stale.
    stale_fetched = datetime(2026, 3, 29, 3, 0, tzinfo=ZURICH)
    assert market_data._persistent_cache_fresh_enough(
        stale_fetched,
        "M5",
        now=now,
    ) is False

    fresh_fetched = datetime(2026, 3, 29, 3, 34, tzinfo=ZURICH)
    assert market_data._persistent_cache_fresh_enough(
        fresh_fetched,
        "M5",
        now=now,
    ) is True


def test_daily_trade_and_broker_pnl_use_utc_date_across_offset_midnight(monkeypatch) -> None:
    monkeypatch.setattr(
        repositories,
        "_utcnow",
        lambda: datetime(2026, 10, 24, 23, 45),
    )
    local_offset = timezone(timedelta(hours=2))

    position = open_paper_position(
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        direction="long",
        quantity=0.10,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        account_currency="CHF",
        broker_position_id=920001,
    )

    # Local calendar date is Oct 25, but the instant is Oct 24 23:30 UTC.
    with get_db() as db:
        db.execute(
            "UPDATE paper_positions SET opened_at = ? WHERE id = ?",
            ("2026-10-25T01:30:00+02:00", position.id),
        )
        db.commit()

    assert daily_trade_count() == 1

    close_paper_position(
        position.id,
        101.0,
        "broker_position_closed",
        closed_at_override=datetime(2026, 10, 25, 1, 35, tzinfo=local_offset),
    )
    recorded = record_broker_deals(
        local_position_id=position.id,
        broker_position_id=920001,
        symbol="NAS100",
        account_currency="CHF",
        deals=[
            {
                "deal_id": 920101,
                "execution_price": 101.0,
                "execution_at": datetime(2026, 10, 25, 1, 35, tzinfo=local_offset),
                "closed_volume_api": 10.0,
                "closed_volume_lots": 0.10,
                "gross_profit": -2.50,
                "swap": 0.0,
                "commission": 0.0,
                "pnl_conversion_fee": 0.0,
                "net_profit": -2.50,
            }
        ],
    )

    assert recorded["inserted"] == 1
    assert daily_realized_pnl() == pytest.approx(-2.50)


def test_broker_reconciliation_preserves_instant_and_counts_utc_day(monkeypatch) -> None:
    monkeypatch.setattr(
        repositories,
        "_utcnow",
        lambda: datetime(2026, 10, 24, 23, 50),
    )
    position = open_paper_position(
        symbol="US30",
        timeframe="M5",
        strategy="breakout",
        direction="long",
        quantity=0.10,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        broker_position_id=930001,
    )
    close_paper_position(position.id, 100.5, "broker_position_closed")

    broker_closed_at = datetime(2026, 10, 25, 1, 40, tzinfo=timezone(timedelta(hours=2)))
    reconciled = reconcile_closed_paper_position_from_broker(
        position.id,
        exit_price=99.5,
        realized_pnl=-3.25,
        closed_at=broker_closed_at,
        broker_position_id=930001,
        broker_details={"source": "dst-reconciliation-test"},
    )

    assert reconciled.closed_at is not None
    assert reconciled.closed_at.astimezone(UTC) == datetime(2026, 10, 24, 23, 40, tzinfo=UTC)
    assert daily_realized_pnl() == pytest.approx(-3.25)


def test_event_timing_uses_exact_utc_instant_through_repeated_autumn_hour() -> None:
    published_local = datetime(2026, 10, 25, 2, 30, tzinfo=ZURICH, fold=1)
    inserted, _ = ingest_events(
        [
            MarketEventInput(
                source="Reuters",
                title="Nasdaq 100 rises after policy update",
                url="https://www.reuters.com/markets/dst-event-test",
                published_at=published_local,
            )
        ]
    )
    event = inserted[0]

    index = pd.date_range("2026-10-25T01:25:00Z", periods=4, freq="5min")
    bars = pd.DataFrame(
        {
            "open": [100.0, 100.1, 100.2, 100.3],
            "high": [100.2, 100.3, 100.4, 100.5],
            "low": [99.9, 100.0, 100.1, 100.2],
            "close": [100.0, 100.1, 100.2, 100.3],
            "volume": [1.0, 1.0, 1.0, 1.0],
        },
        index=index,
    )

    outcome = evaluate_event_against_bars(event, "NAS100", "5m", bars)

    assert published_local.astimezone(UTC) == datetime(2026, 10, 25, 1, 30, tzinfo=UTC)
    assert outcome.reference_at == datetime(2026, 10, 25, 1, 30, tzinfo=UTC)
    assert outcome.target_at == datetime(2026, 10, 25, 1, 35, tzinfo=UTC)


def test_same_bar_in_different_timezone_representation_does_not_duplicate_action(monkeypatch) -> None:
    instant_utc = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(minutes=1)
    instant_local = instant_utc.astimezone(ZURICH)
    current = {"bars": _bars(instant_utc)}
    engine = V2Engine()
    bar_state: dict[str, int] = {}
    executions = 0

    config = EngineConfig(
        enabled=True,
        paper_autotrade=True,
        demo_autotrade=False,
        kill_switch=False,
        min_confidence=0.60,
    )
    item = WatchlistItem(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        enabled=True,
        trading_enabled=False,
    )

    class _Strategy:
        def analyze(self, **kwargs) -> StrategyAnalysis:
            return StrategyAnalysis(
                symbol="XAUUSD",
                timeframe="M5",
                strategy="sma_cross",
                signal="long",
                confidence=0.90,
                entry_price=100.5,
                stop_loss=99.0,
                take_profit=102.0,
                reasons=["timezone replay test"],
            )

    monkeypatch.setattr(engine_module, "get_bars", lambda *args, **kwargs: current["bars"])
    monkeypatch.setattr(
        engine_module,
        "record_market_bar_freshness",
        lambda *args, **kwargs: (True, {}, None),
    )
    monkeypatch.setattr(engine_module, "get_strategy", lambda name: _Strategy())
    monkeypatch.setattr(engine_module, "add_analysis", lambda analysis: analysis)
    monkeypatch.setattr(engine_module, "record_confluence_shadow", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine, "_mark_positions", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine, "_sync_existing_demo_protection", lambda *args, **kwargs: None)

    def _execute(**kwargs) -> ExecutionResult:
        nonlocal executions
        executions += 1
        return ExecutionResult(
            action_taken=True,
            intent_id=10,
            status="executed",
            summary="timezone action executed",
            position_id=20,
            retryable=False,
        )

    monkeypatch.setattr(engine_module, "execute_paper_signal", _execute)

    first = asyncio.run(engine._scan_item(config, item, bar_state))
    assert first == (True, True)
    assert executions == 1

    current["bars"] = _bars(instant_local)
    second = asyncio.run(engine._scan_item(config, item, bar_state))

    assert int(instant_utc.timestamp()) == int(instant_local.timestamp())
    assert second == (False, False)
    assert executions == 1
