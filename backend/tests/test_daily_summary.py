from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from backend.domain.models import EngineConfig
from backend.services import daily_summary
from backend.services.daily_summary import build_daily_summary
from backend.storage import repositories
from backend.storage.db import get_db
from backend.storage.repositories import (
    close_paper_position,
    create_decision_record,
    open_paper_position,
    record_broker_deals,
    save_engine_config,
)


REPORT_DAY = date(2026, 10, 2)


def _clock(monkeypatch, initial: datetime):
    state = {"now": initial}
    monkeypatch.setattr(repositories, "_utcnow", lambda: state["now"])
    return state


def _trade_table_counts() -> dict[str, int]:
    tables = ("paper_positions", "broker_deals", "decision_records")
    with get_db() as db:
        return {
            table: int(db.execute(f"SELECT COUNT(*) AS total FROM {table}").fetchone()["total"])
            for table in tables
        }


def test_daily_summary_prefers_broker_deals_and_computes_realized_drawdown(monkeypatch) -> None:
    save_engine_config(EngineConfig(account_currency="CHF"))
    clock = _clock(monkeypatch, datetime(2026, 10, 2, 8, 0))

    first = open_paper_position(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        direction="long",
        quantity=1.0,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=112.0,
        account_currency="CHF",
    )
    clock["now"] = datetime(2026, 10, 2, 8, 30)
    second = open_paper_position(
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        direction="long",
        quantity=1.0,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=112.0,
        account_currency="CHF",
        broker_position_id=940002,
    )
    clock["now"] = datetime(2026, 10, 2, 8, 45)
    third = open_paper_position(
        symbol="US30",
        timeframe="M5",
        strategy="breakout",
        direction="long",
        quantity=1.0,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=106.0,
        account_currency="CHF",
    )

    close_paper_position(
        first.id,
        110.0,
        "take_profit",
        closed_at_override=datetime(2026, 10, 2, 9, 0),
    )
    # This local estimate must be replaced by broker deal truth in the report.
    close_paper_position(
        second.id,
        110.0,
        "broker_position_closed",
        closed_at_override=datetime(2026, 10, 2, 11, 0),
    )
    close_paper_position(
        third.id,
        104.0,
        "take_profit",
        closed_at_override=datetime(2026, 10, 2, 12, 0),
    )

    broker_result = record_broker_deals(
        local_position_id=second.id,
        broker_position_id=940002,
        symbol="NAS100",
        account_currency="CHF",
        deals=[
            {
                "deal_id": 940101,
                "execution_price": 99.5,
                "execution_at": datetime(2026, 10, 2, 10, 0),
                "closed_volume_api": 50.0,
                "closed_volume_lots": 0.5,
                "gross_profit": -7.0,
                "swap": 0.0,
                "commission": 0.0,
                "pnl_conversion_fee": 0.0,
                "net_profit": -7.0,
            },
            {
                "deal_id": 940102,
                "execution_price": 100.5,
                "execution_at": datetime(2026, 10, 2, 11, 0),
                "closed_volume_api": 50.0,
                "closed_volume_lots": 0.5,
                "gross_profit": 2.0,
                "swap": 0.0,
                "commission": 0.0,
                "pnl_conversion_fee": 0.0,
                "net_profit": 2.0,
            },
        ],
    )
    assert broker_result["inserted"] == 2

    summary = build_daily_summary(REPORT_DAY)

    assert summary.date_utc == REPORT_DAY
    assert summary.account_currency == "CHF"
    assert summary.trades == 3
    assert summary.trades_opened == 3
    assert summary.trades_closed == 3
    assert summary.realized_pnl == pytest.approx(9.0)
    assert summary.max_realized_drawdown == pytest.approx(7.0)
    assert summary.wins == 2
    assert summary.losses == 1
    assert summary.breakeven == 0
    assert summary.win_rate_pct == pytest.approx(66.6666666667)
    assert summary.broker_deal_count == 2
    assert summary.pure_paper_close_count == 2
    assert summary.pnl_basis == "broker_deals_preferred"
    assert summary.drawdown_basis == "realized_pnl_sequence"


def test_daily_summary_groups_persisted_rejections_by_reason(monkeypatch) -> None:
    clock = _clock(monkeypatch, datetime(2026, 10, 2, 10, 0))

    for suffix in ("a", "b"):
        create_decision_record(
            correlation_id=f"risk-{suffix}",
            decision_type="paper_execution_gate",
            symbol="XAUUSD",
            timeframe="M5",
            strategy="sma_cross",
            outcome="rejected_risk",
            summary="Signal strength is below the configured minimum.",
            evidence={},
        )

    clock["now"] = datetime(2026, 10, 2, 10, 5)
    create_decision_record(
        correlation_id="sizing-a",
        decision_type="paper_execution_gate",
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        outcome="rejected_sizing",
        summary="Risk budget is below the minimum tradable quantity.",
        evidence={},
    )

    clock["now"] = datetime(2026, 10, 1, 23, 59)
    create_decision_record(
        correlation_id="previous-day",
        decision_type="paper_execution_gate",
        symbol="US30",
        timeframe="M5",
        strategy="breakout",
        outcome="rejected_risk",
        summary="Previous-day rejection.",
        evidence={},
    )

    summary = build_daily_summary(REPORT_DAY)

    assert summary.rejected_signals == 3
    assert [(item.reason, item.count) for item in summary.rejected_by_reason] == [
        ("Signal strength is below the configured minimum.", 2),
        ("Risk budget is below the minimum tradable quantity.", 1),
    ]


def test_daily_summary_uses_utc_calendar_date_for_offset_timestamps(monkeypatch) -> None:
    clock = _clock(monkeypatch, datetime(2026, 10, 24, 23, 45))
    position = open_paper_position(
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

    with get_db() as db:
        db.execute(
            "UPDATE paper_positions SET opened_at = ? WHERE id = ?",
            ("2026-10-25T01:30:00+02:00", position.id),
        )
        db.commit()

    day_24 = build_daily_summary(date(2026, 10, 24))
    day_25 = build_daily_summary(date(2026, 10, 25))

    assert day_24.trades_opened == 1
    assert day_25.trades_opened == 0


def test_daily_summary_drawdown_orders_offset_timestamps_by_utc_instant(monkeypatch) -> None:
    clock = _clock(monkeypatch, datetime(2026, 10, 25, 0, 0))
    position = open_paper_position(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="breakout",
        direction="long",
        quantity=1.0,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        account_currency="CHF",
        broker_position_id=950001,
    )

    offset = timezone(timedelta(hours=2))
    result = record_broker_deals(
        local_position_id=position.id,
        broker_position_id=950001,
        symbol="XAUUSD",
        account_currency="CHF",
        deals=[
            {
                "deal_id": 950101,
                "execution_price": 101.0,
                # 00:30 UTC
                "execution_at": datetime(2026, 10, 25, 2, 30, tzinfo=offset),
                "closed_volume_api": 50.0,
                "closed_volume_lots": 0.5,
                "gross_profit": 10.0,
                "swap": 0.0,
                "commission": 0.0,
                "pnl_conversion_fee": 0.0,
                "net_profit": 10.0,
            },
            {
                "deal_id": 950102,
                "execution_price": 99.0,
                # 00:45 UTC
                "execution_at": datetime(2026, 10, 25, 0, 45, tzinfo=UTC),
                "closed_volume_api": 50.0,
                "closed_volume_lots": 0.5,
                "gross_profit": -6.0,
                "swap": 0.0,
                "commission": 0.0,
                "pnl_conversion_fee": 0.0,
                "net_profit": -6.0,
            },
        ],
    )
    assert result["inserted"] == 2

    summary = build_daily_summary(date(2026, 10, 25))

    assert summary.realized_pnl == pytest.approx(4.0)
    assert summary.max_realized_drawdown == pytest.approx(6.0)


def test_daily_summary_empty_day_is_deterministic(monkeypatch) -> None:
    save_engine_config(EngineConfig(account_currency="CHF"))
    monkeypatch.setattr(daily_summary, "_utc_day", lambda: REPORT_DAY)

    summary = build_daily_summary()

    assert summary.date_utc == REPORT_DAY
    assert summary.account_currency == "CHF"
    assert summary.trades == 0
    assert summary.trades_closed == 0
    assert summary.realized_pnl == 0.0
    assert summary.max_realized_drawdown == 0.0
    assert summary.wins == 0
    assert summary.losses == 0
    assert summary.breakeven == 0
    assert summary.win_rate_pct is None
    assert summary.rejected_signals == 0
    assert summary.rejected_by_reason == []


def test_daily_summary_api_is_read_only_and_supports_explicit_date(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_EVENT_INTELLIGENCE_ON_BOOT", "0")
    clock = _clock(monkeypatch, datetime(2026, 10, 2, 8, 0))
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
    before = _trade_table_counts()

    from backend.app import app

    with TestClient(app) as client:
        response = client.get("/api/reports/daily-summary?date=2026-10-02")

    after = _trade_table_counts()
    assert response.status_code == 200
    payload = response.json()
    assert payload["date_utc"] == "2026-10-02"
    assert payload["trades"] == 1
    assert payload["trades_opened"] == 1
    assert before == after


def test_daily_summary_api_rejects_invalid_date(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_EVENT_INTELLIGENCE_ON_BOOT", "0")

    from backend.app import app

    with TestClient(app) as client:
        response = client.get("/api/reports/daily-summary?date=2026-13-99")

    assert response.status_code == 400
    assert "YYYY-MM-DD" in response.json()["detail"]
