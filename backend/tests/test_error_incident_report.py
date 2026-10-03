from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

from fastapi.testclient import TestClient
import pytest

from backend.services.error_incident_report import build_error_incident_report
from backend.storage import repositories
from backend.storage.db import get_db
from backend.storage.repositories import (
    add_event_source_run,
    create_order_intent,
    log_incident,
    update_order_intent_status,
)


NOW = datetime(2026, 10, 3, 8, 0)


def _clock(monkeypatch, initial: datetime):
    state = {"now": initial}
    monkeypatch.setattr(repositories, "_utcnow", lambda: state["now"])
    return state


def _intent(*, status: str = "pending"):
    return create_order_intent(
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        direction="long",
        intent_type="open",
        status=status,
        confidence=0.90,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        quantity=0.1,
        rationale="error incident grouping fixture",
        details={},
    )


def _table_counts() -> dict[str, int]:
    tables = ("incidents", "order_intent_transitions", "event_source_runs")
    with get_db() as db:
        return {
            table: int(db.execute(f"SELECT COUNT(*) AS total FROM {table}").fetchone()["total"])
            for table in tables
        }


def test_empty_window_has_explicit_zero_denominator() -> None:
    report = build_error_incident_report(now=NOW)

    assert report.window_hours == 24
    assert report.denominator_name == "terminal_order_intent_transitions_plus_event_source_runs"
    assert report.denominator_count == 0
    assert report.failure_evidence_count == 0
    assert report.failure_evidence_rate_pct is None
    assert report.terminal_intent_outcomes == 0
    assert report.event_source_runs == 0
    assert report.incident_occurrences == 0
    assert report.distinct_incident_groups == 0
    assert report.distinct_group_count == 0
    assert report.groups == []


def test_repeated_incidents_group_by_stable_code(monkeypatch) -> None:
    clock = _clock(monkeypatch, NOW - timedelta(hours=3))
    log_incident("error", "scan_failure", "First scan failure", {"attempt": 1})
    clock["now"] = NOW - timedelta(hours=2)
    log_incident("error", "scan_failure", "Second scan failure", {"attempt": 2})
    clock["now"] = NOW - timedelta(hours=1)
    log_incident("warning", "market_data_stale", "Latest market bar is stale.", {})

    report = build_error_incident_report(now=NOW)

    assert report.incident_occurrences == 3
    assert report.distinct_incident_groups == 2
    assert report.denominator_count == 0
    assert report.failure_evidence_rate_pct is None
    assert [group.group_id for group in report.groups] == [
        "incident:scan_failure",
        "incident:market_data_stale",
    ]
    scan = report.groups[0]
    assert scan.occurrences == 2
    assert scan.level == "error"
    assert scan.sample_message == "Second scan failure"
    assert scan.first_seen_at == NOW - timedelta(hours=3)
    assert scan.last_seen_at == NOW - timedelta(hours=2)


def test_failure_rate_uses_terminal_intents_and_event_source_runs_only(monkeypatch) -> None:
    clock = _clock(monkeypatch, NOW - timedelta(hours=5))

    executed = _intent()
    update_order_intent_status(executed.id, "accepted", {}, reason="risk_accepted")
    clock["now"] = NOW - timedelta(hours=4, minutes=30)
    update_order_intent_status(executed.id, "executed", {}, reason="broker_executed")

    clock["now"] = NOW - timedelta(hours=4)
    failed = _intent()
    update_order_intent_status(failed.id, "accepted", {}, reason="risk_accepted")
    clock["now"] = NOW - timedelta(hours=3, minutes=30)
    update_order_intent_status(failed.id, "failed", {}, reason="broker_order_rejected")

    # Non-terminal intent transitions are deliberately outside the denominator.
    clock["now"] = NOW - timedelta(hours=3)
    pending = _intent()
    update_order_intent_status(pending.id, "accepted", {}, reason="risk_accepted")

    add_event_source_run(
        started_at=NOW - timedelta(hours=2, minutes=30),
        completed_at=NOW - timedelta(hours=2),
        source="calendar_feed",
        fetched_items=10,
        inserted_events=5,
        error="",
    )
    add_event_source_run(
        started_at=NOW - timedelta(hours=90),
        completed_at=NOW - timedelta(hours=1),
        source="news_feed",
        fetched_items=0,
        inserted_events=0,
        error="upstream timeout",
    )

    # Incident occurrences are grouped, but do not distort the real outcome denominator.
    clock["now"] = NOW - timedelta(minutes=30)
    log_incident("error", "scan_failure", "Independent scan failure", {})

    report = build_error_incident_report(now=NOW)

    assert report.terminal_intent_outcomes == 2
    assert report.failed_intent_outcomes == 1
    assert report.event_source_runs == 2
    assert report.failed_event_source_runs == 1
    assert report.denominator_count == 4
    assert report.failure_evidence_count == 2
    assert report.failure_evidence_rate_pct == pytest.approx(50.0)
    assert report.incident_occurrences == 1

    by_id = {group.group_id: group for group in report.groups}
    assert by_id["order_intent_failure:broker_order_rejected"].occurrences == 1
    assert by_id["event_source_failure:news_feed"].occurrences == 1
    assert by_id["event_source_failure:news_feed"].sample_message == "upstream timeout"
    assert by_id["incident:scan_failure"].occurrences == 1


def test_latest_terminal_transition_per_intent_is_counted_once(monkeypatch) -> None:
    clock = _clock(monkeypatch, NOW - timedelta(hours=3))
    intent = _intent()
    update_order_intent_status(intent.id, "accepted", {}, reason="accepted")
    clock["now"] = NOW - timedelta(hours=2)
    update_order_intent_status(intent.id, "failed", {}, reason="first_failure")

    # Simulate legacy/corrupt duplicate terminal history without changing production transition rules.
    with get_db() as db:
        db.execute(
            """
            INSERT INTO order_intent_transitions(
                intent_id, created_at, from_status, to_status, reason, details_json
            ) VALUES(?, ?, ?, ?, ?, ?)
            """,
            (
                intent.id,
                (NOW - timedelta(hours=1)).isoformat(),
                "failed",
                "failed",
                "latest_failure",
                "{}",
            ),
        )
        db.commit()

    report = build_error_incident_report(now=NOW)

    assert report.terminal_intent_outcomes == 1
    assert report.failed_intent_outcomes == 1
    assert report.denominator_count == 1
    assert report.failure_evidence_rate_pct == pytest.approx(100.0)
    assert [group.group_id for group in report.groups] == [
        "order_intent_failure:latest_failure"
    ]


def test_utc_window_normalizes_offset_bearing_persisted_timestamps(monkeypatch) -> None:
    plus_two = timezone(timedelta(hours=2))
    clock = _clock(monkeypatch, datetime(2026, 10, 3, 9, 30, tzinfo=plus_two))
    # 09:30 +02 == 07:30 UTC, inside the 1h window ending 08:00 UTC.
    log_incident("warning", "inside_offset", "Inside UTC window", {})

    clock["now"] = datetime(2026, 10, 3, 8, 30, tzinfo=plus_two)
    # 08:30 +02 == 06:30 UTC, outside the 1h window.
    log_incident("warning", "outside_offset", "Outside UTC window", {})

    report = build_error_incident_report(window_hours=1, now=NOW)

    assert report.window_start_utc == datetime(2026, 10, 3, 7, 0)
    assert report.window_end_utc == NOW
    assert report.incident_occurrences == 1
    assert [group.code for group in report.groups] == ["inside_offset"]
    assert report.groups[0].first_seen_at == datetime(2026, 10, 3, 7, 30)


def test_group_limit_is_bounded_and_ordering_is_deterministic(monkeypatch) -> None:
    clock = _clock(monkeypatch, NOW - timedelta(hours=4))
    for index in range(3):
        clock["now"] = NOW - timedelta(hours=4) + timedelta(minutes=index)
        log_incident("error", "alpha", f"alpha {index}", {})
    for index in range(2):
        clock["now"] = NOW - timedelta(hours=3) + timedelta(minutes=index)
        log_incident("error", "beta", f"beta {index}", {})
    clock["now"] = NOW - timedelta(hours=2)
    log_incident("warning", "gamma", "gamma", {})

    report = build_error_incident_report(now=NOW, group_limit=2)

    assert report.distinct_group_count == 3
    assert report.group_limit == 2
    assert report.groups_truncated is True
    assert [(group.code, group.occurrences) for group in report.groups] == [
        ("alpha", 3),
        ("beta", 2),
    ]


def test_error_incident_api_is_read_only(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_EVENT_INTELLIGENCE_ON_BOOT", "0")

    clock = _clock(monkeypatch, datetime.now(UTC).replace(tzinfo=None))
    log_incident("warning", "api_fixture", "Read-only API fixture", {})
    before = _table_counts()

    from backend.app import app

    with TestClient(app) as client:
        response = client.get("/api/reports/error-incidents?hours=24&group_limit=10")

    after = _table_counts()
    assert response.status_code == 200
    payload = response.json()
    assert payload["window_hours"] == 24
    assert payload["group_limit"] == 10
    assert any(group["group_id"] == "incident:api_fixture" for group in payload["groups"])
    assert before == after


def test_error_incident_api_enforces_bounded_window_and_group_limit(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_EVENT_INTELLIGENCE_ON_BOOT", "0")

    from backend.app import app

    with TestClient(app) as client:
        too_wide = client.get("/api/reports/error-incidents?hours=169")
        too_many_groups = client.get("/api/reports/error-incidents?group_limit=101")

    assert too_wide.status_code == 422
    assert too_many_groups.status_code == 422
