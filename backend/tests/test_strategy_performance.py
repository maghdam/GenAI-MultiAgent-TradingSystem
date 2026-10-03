from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

from fastapi.testclient import TestClient
import pytest

from backend.services.strategy_performance import build_strategy_performance_report
from backend.storage import repositories
from backend.storage.db import get_db
from backend.storage.repositories import (
    close_paper_position,
    open_paper_position,
    record_broker_deals,
)


NOW = datetime(2026, 10, 3, 8, 0)


def _clock(monkeypatch, initial: datetime):
    state = {"now": initial}
    monkeypatch.setattr(repositories, "_utcnow", lambda: state["now"])
    return state


def _closed_trade(
    *,
    symbol: str,
    timeframe: str,
    strategy: str,
    currency: str,
    closed_at: datetime,
    realized_pnl: float,
    broker_position_id: int | None = None,
):
    position = open_paper_position(
        symbol=symbol,
        timeframe=timeframe,
        strategy=strategy,
        direction="long",
        quantity=1.0,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        account_currency=currency,
        broker_position_id=broker_position_id,
    )
    return close_paper_position(
        position.id,
        100.0,
        "fixture_close",
        realized_pnl_override=realized_pnl,
        closed_at_override=closed_at,
    )


def _trade_table_counts() -> dict[str, int]:
    tables = ("paper_positions", "broker_deals", "trade_audit", "paper_events")
    with get_db() as db:
        return {
            table: int(db.execute(f"SELECT COUNT(*) AS total FROM {table}").fetchone()["total"])
            for table in tables
        }


def test_empty_strategy_performance_report_has_explicit_zero_sample() -> None:
    report = build_strategy_performance_report(now=NOW)

    assert report.window_days == 30
    assert report.sample_count == 0
    assert report.slice_count == 0
    assert report.account_currencies == []
    assert report.slices == []
    assert report.pnl_basis == "broker_deals_preferred_per_closed_trade"
    assert report.outcome_basis == "one_closed_position_one_trade"


def test_strategy_performance_groups_symbol_timeframe_strategy_and_currency(monkeypatch) -> None:
    _clock(monkeypatch, NOW - timedelta(days=2))
    _closed_trade(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        currency="CHF",
        closed_at=NOW - timedelta(days=2),
        realized_pnl=10.0,
    )
    _closed_trade(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        currency="CHF",
        closed_at=NOW - timedelta(days=1),
        realized_pnl=-5.0,
    )
    _closed_trade(
        symbol="XAUUSD",
        timeframe="H1",
        strategy="sma_cross",
        currency="CHF",
        closed_at=NOW - timedelta(hours=20),
        realized_pnl=0.0,
    )
    _closed_trade(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        currency="USD",
        closed_at=NOW - timedelta(hours=10),
        realized_pnl=7.0,
    )
    _closed_trade(
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        currency="CHF",
        closed_at=NOW - timedelta(hours=5),
        realized_pnl=3.0,
    )

    report = build_strategy_performance_report(now=NOW)

    assert report.sample_count == 5
    assert report.slice_count == 4
    assert report.account_currencies == ["CHF", "USD"]

    first = report.slices[0]
    assert (first.symbol, first.timeframe, first.strategy, first.account_currency) == (
        "XAUUSD",
        "M5",
        "sma_cross",
        "CHF",
    )
    assert first.trades == 2
    assert first.realized_pnl == pytest.approx(5.0)
    assert first.average_realized_pnl == pytest.approx(2.5)
    assert first.wins == 1
    assert first.losses == 1
    assert first.breakeven == 0
    assert first.win_rate_pct == pytest.approx(50.0)
    assert first.pure_paper_trades == 2
    assert first.broker_backed_trades == 0


def test_broker_deals_override_paper_estimate_and_partial_closes_are_one_trade(monkeypatch) -> None:
    _clock(monkeypatch, NOW - timedelta(days=40))
    position = _closed_trade(
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        currency="CHF",
        closed_at=NOW - timedelta(hours=1),
        realized_pnl=50.0,
        broker_position_id=991001,
    )

    deals = [
        {
            "deal_id": 991101,
            "execution_price": 99.0,
            "execution_at": NOW - timedelta(days=35),
            "closed_volume_api": 50.0,
            "closed_volume_lots": 0.5,
            "gross_profit": -7.0,
            "swap": 0.0,
            "commission": 0.0,
            "pnl_conversion_fee": 0.0,
            "net_profit": -7.0,
        },
        {
            "deal_id": 991102,
            "execution_price": 101.0,
            "execution_at": NOW - timedelta(hours=1),
            "closed_volume_api": 50.0,
            "closed_volume_lots": 0.5,
            "gross_profit": 2.0,
            "swap": 0.0,
            "commission": 0.0,
            "pnl_conversion_fee": 0.0,
            "net_profit": 2.0,
        },
    ]
    first = record_broker_deals(
        local_position_id=position.id,
        broker_position_id=991001,
        symbol="NAS100",
        account_currency="CHF",
        deals=deals,
    )
    second = record_broker_deals(
        local_position_id=position.id,
        broker_position_id=991001,
        symbol="NAS100",
        account_currency="CHF",
        deals=deals,
    )
    assert first["inserted"] == 2
    assert second["inserted"] == 0

    report = build_strategy_performance_report(window_days=30, now=NOW)

    assert report.sample_count == 1
    item = report.slices[0]
    assert item.trades == 1
    assert item.realized_pnl == pytest.approx(-5.0)
    assert item.average_realized_pnl == pytest.approx(-5.0)
    assert item.wins == 0
    assert item.losses == 1
    assert item.broker_backed_trades == 1
    assert item.pure_paper_trades == 0
    assert item.broker_deal_count == 2


def test_open_positions_are_not_performance_samples(monkeypatch) -> None:
    _clock(monkeypatch, NOW - timedelta(hours=2))
    open_paper_position(
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        direction="long",
        quantity=0.1,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        account_currency="CHF",
    )
    _closed_trade(
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        currency="CHF",
        closed_at=NOW - timedelta(hours=1),
        realized_pnl=4.0,
    )

    report = build_strategy_performance_report(now=NOW)

    assert report.sample_count == 1
    assert report.slices[0].trades == 1
    assert report.slices[0].realized_pnl == pytest.approx(4.0)


def test_strategy_performance_uses_utc_window_for_offset_timestamps(monkeypatch) -> None:
    plus_two = timezone(timedelta(hours=2))
    _clock(monkeypatch, NOW - timedelta(days=2))

    inside = _closed_trade(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        currency="CHF",
        closed_at=NOW - timedelta(hours=5),
        realized_pnl=4.0,
    )
    outside = _closed_trade(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        currency="CHF",
        closed_at=NOW - timedelta(hours=6),
        realized_pnl=9.0,
    )

    with get_db() as db:
        db.execute(
            "UPDATE paper_positions SET closed_at = ? WHERE id = ?",
            ("2026-10-02T10:30:00+02:00", inside.id),
        )
        db.execute(
            "UPDATE paper_positions SET closed_at = ? WHERE id = ?",
            ("2026-10-02T09:30:00+02:00", outside.id),
        )
        db.commit()

    report = build_strategy_performance_report(window_days=1, now=NOW)

    assert report.window_start_utc == datetime(2026, 10, 2, 8, 0)
    assert report.window_end_utc == NOW
    assert report.sample_count == 1
    assert report.slices[0].realized_pnl == pytest.approx(4.0)
    assert report.slices[0].first_closed_at == datetime(2026, 10, 2, 8, 30)


def test_strategy_performance_filters_symbol_timeframe_and_strategy(monkeypatch) -> None:
    _clock(monkeypatch, NOW - timedelta(hours=6))
    _closed_trade(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        currency="CHF",
        closed_at=NOW - timedelta(hours=5),
        realized_pnl=6.0,
    )
    _closed_trade(
        symbol="XAUUSD",
        timeframe="H1",
        strategy="sma_cross",
        currency="CHF",
        closed_at=NOW - timedelta(hours=4),
        realized_pnl=7.0,
    )
    _closed_trade(
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        currency="CHF",
        closed_at=NOW - timedelta(hours=3),
        realized_pnl=8.0,
    )

    report = build_strategy_performance_report(
        now=NOW,
        symbol="xauusd",
        timeframe="m5",
        strategy="sma_cross",
    )

    assert report.symbol_filter == "XAUUSD"
    assert report.timeframe_filter == "M5"
    assert report.strategy_filter == "sma_cross"
    assert report.sample_count == 1
    assert report.slice_count == 1
    assert report.slices[0].realized_pnl == pytest.approx(6.0)


def test_group_limit_is_deterministic_and_reports_truncation(monkeypatch) -> None:
    _clock(monkeypatch, NOW - timedelta(days=3))
    for index in range(3):
        _closed_trade(
            symbol="XAUUSD",
            timeframe="M5",
            strategy="sma_cross",
            currency="CHF",
            closed_at=NOW - timedelta(days=2, hours=index),
            realized_pnl=1.0,
        )
    for index in range(2):
        _closed_trade(
            symbol="NAS100",
            timeframe="M5",
            strategy="breakout",
            currency="CHF",
            closed_at=NOW - timedelta(days=1, hours=index),
            realized_pnl=1.0,
        )
    _closed_trade(
        symbol="US30",
        timeframe="M5",
        strategy="breakout",
        currency="CHF",
        closed_at=NOW - timedelta(hours=2),
        realized_pnl=1.0,
    )

    report = build_strategy_performance_report(now=NOW, group_limit=2)

    assert report.sample_count == 6
    assert report.slice_count == 3
    assert report.group_limit == 2
    assert report.groups_truncated is True
    assert [(item.symbol, item.trades) for item in report.slices] == [
        ("XAUUSD", 3),
        ("NAS100", 2),
    ]


def test_strategy_performance_api_is_read_only_and_enforces_bounds(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_EVENT_INTELLIGENCE_ON_BOOT", "0")

    _clock(monkeypatch, datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1))
    _closed_trade(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        currency="CHF",
        closed_at=datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=30),
        realized_pnl=3.0,
    )
    before = _trade_table_counts()

    from backend.app import app

    with TestClient(app) as client:
        response = client.get(
            "/api/reports/strategy-performance"
            "?days=30&symbol=xauusd&timeframe=m5&strategy=sma_cross&group_limit=10"
        )
        too_wide = client.get("/api/reports/strategy-performance?days=366")
        too_many = client.get("/api/reports/strategy-performance?group_limit=201")

    after = _trade_table_counts()
    assert response.status_code == 200
    payload = response.json()
    assert payload["sample_count"] == 1
    assert payload["slices"][0]["realized_pnl"] == pytest.approx(3.0)
    assert too_wide.status_code == 422
    assert too_many.status_code == 422
    assert before == after
