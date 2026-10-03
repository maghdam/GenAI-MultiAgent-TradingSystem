from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

from fastapi.testclient import TestClient
import pytest

from backend.domain.models import BrokerStatementRowInput, StatementComparisonRequest
from backend.services.journal_statement_comparison import (
    build_journal_export,
    compare_external_statement,
    render_journal_export_csv,
)
from backend.storage import repositories
from backend.storage.db import get_db
from backend.storage.repositories import (
    add_trade_audit,
    close_paper_position,
    list_trade_audits,
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
    symbol: str = "NAS100",
    timeframe: str = "M5",
    strategy: str = "breakout",
    direction: str = "long",
    currency: str = "CHF",
    closed_at: datetime | None = None,
    realized_pnl: float = 0.0,
    broker_position_id: int | None = None,
):
    position = open_paper_position(
        symbol=symbol,
        timeframe=timeframe,
        strategy=strategy,
        direction=direction,
        quantity=1.0,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        account_currency=currency,
        broker_position_id=broker_position_id,
    )
    return close_paper_position(
        position.id,
        101.0,
        "fixture_close",
        realized_pnl_override=realized_pnl,
        closed_at_override=closed_at or (NOW - timedelta(hours=1)),
    )


def _deal(
    *,
    deal_id: int,
    execution_at: datetime,
    net_profit: float,
    execution_price: float = 101.0,
    closed_volume_lots: float = 0.5,
) -> dict:
    return {
        "deal_id": deal_id,
        "execution_price": execution_price,
        "execution_at": execution_at,
        "closed_volume_api": closed_volume_lots * 100.0,
        "closed_volume_lots": closed_volume_lots,
        "gross_profit": net_profit,
        "swap": 0.0,
        "commission": 0.0,
        "pnl_conversion_fee": 0.0,
        "net_profit": net_profit,
    }


def _counts() -> dict[str, int]:
    tables = ("paper_positions", "broker_deals", "trade_audit", "paper_events")
    with get_db() as db:
        return {
            table: int(db.execute(f"SELECT COUNT(*) AS total FROM {table}").fetchone()["total"])
            for table in tables
        }


def test_journal_export_prefers_broker_deals_and_preserves_audit_trace(monkeypatch) -> None:
    clock = _clock(monkeypatch, NOW - timedelta(hours=3))
    position = _closed_trade(
        broker_position_id=992001,
        realized_pnl=99.0,
        closed_at=NOW - timedelta(hours=1),
    )
    add_trade_audit(
        event_type="fixture_open",
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        position_id=position.id,
        summary="fixture audit one",
        details={},
    )
    clock["now"] = NOW - timedelta(hours=2)
    add_trade_audit(
        event_type="fixture_close",
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        position_id=position.id,
        summary="fixture audit two",
        details={},
    )
    expected_audits = sorted(
        row.id for row in list_trade_audits(20) if row.position_id == position.id
    )

    recorded = record_broker_deals(
        local_position_id=position.id,
        broker_position_id=992001,
        symbol="NAS100",
        account_currency="CHF",
        deals=[
            _deal(
                deal_id=992101,
                execution_at=NOW - timedelta(hours=2),
                net_profit=-4.0,
            ),
            _deal(
                deal_id=992102,
                execution_at=NOW - timedelta(hours=1),
                net_profit=6.5,
            ),
        ],
    )
    assert recorded["inserted"] == 2

    report = build_journal_export(now=NOW)

    assert report.row_count == 1
    assert report.account_currencies == ["CHF"]
    row = report.rows[0]
    assert row.row_id == f"local_position:{position.id}"
    assert row.local_position_id == position.id
    assert row.broker_position_id == 992001
    assert row.broker_identity_status == "resolved"
    assert row.broker_deal_ids == [992101, 992102]
    assert row.audit_event_ids == expected_audits
    assert row.realized_pnl == pytest.approx(2.5)
    assert row.realized_pnl_basis == "broker_deals"
    assert row.broker_deal_count == 2
    assert row.opened_at_utc.tzinfo == UTC
    assert row.closed_at_utc.tzinfo == UTC


def test_pure_paper_export_is_explicit_and_csv_is_stable(monkeypatch) -> None:
    _clock(monkeypatch, NOW - timedelta(hours=2))
    position = _closed_trade(
        symbol="XAUUSD",
        strategy="sma_cross",
        currency="USD",
        realized_pnl=7.25,
        broker_position_id=None,
    )

    report = build_journal_export(now=NOW)
    csv_text = render_journal_export_csv(report)

    assert report.row_count == 1
    row = report.rows[0]
    assert row.local_position_id == position.id
    assert row.broker_position_id is None
    assert row.broker_identity_status == "not_applicable"
    assert row.broker_deal_ids == []
    assert row.realized_pnl == pytest.approx(7.25)
    assert row.realized_pnl_basis == "paper_estimate"
    assert "row_id,local_position_id,broker_position_id,broker_identity_status" in csv_text
    assert f"local_position:{position.id}" in csv_text
    assert "paper_estimate" in csv_text


def test_journal_export_detects_conflicting_broker_identity(monkeypatch) -> None:
    _clock(monkeypatch, NOW - timedelta(hours=2))
    position = _closed_trade(
        broker_position_id=993001,
        realized_pnl=5.0,
    )
    record_broker_deals(
        local_position_id=position.id,
        broker_position_id=993999,
        symbol="NAS100",
        account_currency="CHF",
        deals=[
            _deal(
                deal_id=993101,
                execution_at=NOW - timedelta(hours=1),
                net_profit=5.0,
                closed_volume_lots=1.0,
            )
        ],
    )

    row = build_journal_export(now=NOW).rows[0]

    assert row.broker_position_id == 993001
    assert row.broker_identity_status == "conflict"
    assert "conflicts" in row.broker_identity_detail.lower()


def test_statement_comparison_aggregates_partial_close_rows_into_one_match(monkeypatch) -> None:
    _clock(monkeypatch, NOW - timedelta(days=40))
    position = _closed_trade(
        broker_position_id=994001,
        realized_pnl=100.0,
        closed_at=NOW - timedelta(hours=1),
    )
    record_broker_deals(
        local_position_id=position.id,
        broker_position_id=994001,
        symbol="NAS100",
        account_currency="CHF",
        deals=[
            _deal(
                deal_id=994101,
                execution_at=NOW - timedelta(days=35),
                net_profit=-3.0,
            ),
            _deal(
                deal_id=994102,
                execution_at=NOW - timedelta(hours=1),
                net_profit=8.0,
            ),
        ],
    )

    request = StatementComparisonRequest(
        window_days=30,
        rows=[
            BrokerStatementRowInput(
                row_id="statement-1",
                broker_position_id=994001,
                deal_id=994101,
                symbol="NAS100",
                direction="long",
                closed_at=NOW - timedelta(days=35),
                account_currency="CHF",
                realized_pnl=-3.0,
            ),
            BrokerStatementRowInput(
                row_id="statement-2",
                broker_position_id=994001,
                deal_id=994102,
                symbol="NAS100",
                direction="long",
                closed_at=NOW - timedelta(hours=1),
                account_currency="CHF",
                realized_pnl=8.0,
            ),
        ],
    )

    report = compare_external_statement(request, now=NOW)

    assert report.local_row_count == 1
    assert report.statement_row_count == 2
    assert report.matched == 1
    assert report.mismatched == 0
    assert report.local_only == 0
    assert report.statement_only == 0
    item = report.items[0]
    assert item.status == "matched"
    assert item.local_position_id == position.id
    assert item.statement_row_ids == ["statement-1", "statement-2"]
    assert item.local_deal_ids == [994101, 994102]
    assert item.statement_deal_ids == [994101, 994102]
    assert item.local_realized_pnl == pytest.approx(5.0)
    assert item.statement_realized_pnl == pytest.approx(5.0)
    assert item.pnl_difference == pytest.approx(0.0)


def test_statement_comparison_reports_explicit_mismatch_reasons(monkeypatch) -> None:
    _clock(monkeypatch, NOW - timedelta(hours=3))
    position = _closed_trade(
        symbol="NAS100",
        direction="long",
        currency="CHF",
        broker_position_id=995001,
        realized_pnl=4.0,
        closed_at=NOW - timedelta(hours=1),
    )
    record_broker_deals(
        local_position_id=position.id,
        broker_position_id=995001,
        symbol="NAS100",
        account_currency="CHF",
        deals=[
            _deal(
                deal_id=995101,
                execution_at=NOW - timedelta(hours=1),
                net_profit=4.0,
                closed_volume_lots=1.0,
            )
        ],
    )

    request = StatementComparisonRequest(
        pnl_tolerance=0.01,
        close_time_tolerance_seconds=60,
        rows=[
            BrokerStatementRowInput(
                row_id="bad-row",
                broker_position_id=995001,
                deal_id=995999,
                symbol="XAUUSD",
                direction="short",
                closed_at=NOW - timedelta(hours=2),
                account_currency="USD",
                realized_pnl=10.0,
            )
        ],
    )

    report = compare_external_statement(request, now=NOW)

    assert report.mismatched == 1
    item = report.items[0]
    assert item.status == "mismatch"
    assert set(item.mismatch_reasons) == {
        "symbol_mismatch",
        "direction_mismatch",
        "currency_mismatch",
        "realized_pnl_mismatch",
        "close_time_mismatch",
        "deal_id_set_mismatch",
    }
    assert item.pnl_difference == pytest.approx(6.0)


def test_statement_comparison_distinguishes_local_only_and_statement_only(monkeypatch) -> None:
    _clock(monkeypatch, NOW - timedelta(hours=4))
    pure_paper = _closed_trade(
        symbol="XAUUSD",
        strategy="sma_cross",
        currency="CHF",
        realized_pnl=2.0,
        broker_position_id=None,
    )
    broker_local = _closed_trade(
        symbol="NAS100",
        strategy="breakout",
        currency="CHF",
        realized_pnl=3.0,
        broker_position_id=996001,
    )

    request = StatementComparisonRequest(
        rows=[
            BrokerStatementRowInput(
                row_id="unknown-broker-row",
                broker_position_id=996999,
                symbol="US30",
                direction="long",
                closed_at=NOW - timedelta(minutes=30),
                account_currency="CHF",
                realized_pnl=1.0,
            )
        ]
    )

    report = compare_external_statement(request, now=NOW)

    assert report.local_only == 2
    assert report.statement_only == 1
    local_by_id = {
        item.local_position_id: item for item in report.items if item.local_position_id is not None
    }
    assert local_by_id[pure_paper.id].mismatch_reasons == ["no_broker_identity"]
    assert local_by_id[broker_local.id].mismatch_reasons == ["missing_from_statement"]
    statement_only = next(item for item in report.items if item.status == "statement_only")
    assert statement_only.statement_row_ids == ["unknown-broker-row"]
    assert statement_only.mismatch_reasons == ["missing_from_tradeagent_journal"]


def test_statement_identity_failures_are_unresolved_without_double_counting_local(monkeypatch) -> None:
    _clock(monkeypatch, NOW - timedelta(hours=4))
    first = _closed_trade(
        symbol="NAS100",
        broker_position_id=997001,
        realized_pnl=2.0,
    )
    record_broker_deals(
        local_position_id=first.id,
        broker_position_id=997001,
        symbol="NAS100",
        account_currency="CHF",
        deals=[
            _deal(
                deal_id=997101,
                execution_at=NOW - timedelta(hours=1),
                net_profit=2.0,
                closed_volume_lots=1.0,
            )
        ],
    )
    second = _closed_trade(
        symbol="XAUUSD",
        strategy="sma_cross",
        broker_position_id=997002,
        realized_pnl=3.0,
    )
    record_broker_deals(
        local_position_id=second.id,
        broker_position_id=997002,
        symbol="XAUUSD",
        account_currency="CHF",
        deals=[
            _deal(
                deal_id=997201,
                execution_at=NOW - timedelta(hours=1),
                net_profit=3.0,
                closed_volume_lots=1.0,
            )
        ],
    )

    request = StatementComparisonRequest(
        rows=[
            BrokerStatementRowInput(
                row_id="duplicate-a",
                broker_position_id=997001,
                deal_id=997101,
                symbol="NAS100",
                direction="long",
                closed_at=NOW - timedelta(hours=1),
                account_currency="CHF",
                realized_pnl=2.0,
            ),
            BrokerStatementRowInput(
                row_id="duplicate-b",
                broker_position_id=997001,
                deal_id=997101,
                symbol="NAS100",
                direction="long",
                closed_at=NOW - timedelta(hours=1),
                account_currency="CHF",
                realized_pnl=2.0,
            ),
            BrokerStatementRowInput(
                row_id="conflict",
                broker_position_id=997001,
                deal_id=997201,
                symbol="NAS100",
                direction="long",
                closed_at=NOW - timedelta(hours=1),
                account_currency="CHF",
                realized_pnl=2.0,
            ),
            BrokerStatementRowInput(
                row_id="missing-identity",
                symbol="US30",
                direction="long",
                closed_at=NOW - timedelta(minutes=30),
                account_currency="CHF",
                realized_pnl=1.0,
            ),
        ]
    )

    report = compare_external_statement(request, now=NOW)

    assert report.identity_unresolved == 4
    # The first local trade is referenced by unresolved statement identity, so it
    # must not also be emitted as local_only.
    assert not any(
        item.status == "local_only" and item.local_position_id == first.id
        for item in report.items
    )
    reasons = [set(item.mismatch_reasons) for item in report.items if item.status == "identity_unresolved"]
    assert any("duplicate_statement_deal_id" in item for item in reasons)
    assert any("statement_identity_conflict" in item for item in reasons)
    assert any("missing_statement_identity" in item for item in reasons)


def test_journal_export_uses_utc_window_for_offset_timestamps(monkeypatch) -> None:
    plus_two = timezone(timedelta(hours=2))
    _clock(monkeypatch, NOW - timedelta(days=2))
    inside = _closed_trade(realized_pnl=4.0)
    outside = _closed_trade(realized_pnl=9.0)

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

    report = build_journal_export(window_days=1, now=NOW)

    assert report.window_start_utc == datetime(2026, 10, 2, 8, 0, tzinfo=UTC)
    assert report.window_end_utc == datetime(2026, 10, 3, 8, 0, tzinfo=UTC)
    assert report.row_count == 1
    assert report.rows[0].local_position_id == inside.id
    assert report.rows[0].closed_at_utc == datetime(2026, 10, 2, 8, 30, tzinfo=UTC)


def test_journal_export_and_statement_comparison_api_are_read_only(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_EVENT_INTELLIGENCE_ON_BOOT", "0")

    _clock(monkeypatch, datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=2))
    position = _closed_trade(
        broker_position_id=998001,
        realized_pnl=3.0,
        closed_at=datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=30),
    )
    record_broker_deals(
        local_position_id=position.id,
        broker_position_id=998001,
        symbol="NAS100",
        account_currency="CHF",
        deals=[
            _deal(
                deal_id=998101,
                execution_at=datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=30),
                net_profit=3.0,
                closed_volume_lots=1.0,
            )
        ],
    )
    before = _counts()

    from backend.app import app

    statement_payload = {
        "window_days": 30,
        "rows": [
            {
                "row_id": "api-statement",
                "broker_position_id": 998001,
                "deal_id": 998101,
                "symbol": "NAS100",
                "direction": "long",
                "closed_at": (datetime.now(UTC) - timedelta(minutes=30)).isoformat(),
                "account_currency": "CHF",
                "realized_pnl": 3.0,
            }
        ],
    }

    with TestClient(app) as client:
        export_response = client.get("/api/reports/journal-export?days=30")
        csv_response = client.get("/api/reports/journal-export.csv?days=30")
        compare_response = client.post(
            "/api/reports/statement-comparison",
            json=statement_payload,
        )
        invalid_days = client.get("/api/reports/journal-export?days=366")
        empty_statement = client.post(
            "/api/reports/statement-comparison",
            json={"rows": []},
        )

    after = _counts()
    assert export_response.status_code == 200
    assert export_response.json()["row_count"] == 1
    assert csv_response.status_code == 200
    assert csv_response.headers["content-type"].startswith("text/csv")
    assert "tradeagent-journal.csv" in csv_response.headers["content-disposition"]
    assert f"local_position:{position.id}" in csv_response.text
    assert compare_response.status_code == 200
    assert compare_response.json()["matched"] == 1
    assert invalid_days.status_code == 422
    assert empty_statement.status_code == 422
    assert before == after
