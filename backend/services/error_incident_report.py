from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from backend.domain.models import ErrorIncidentGroup, ErrorIncidentReportResponse
from backend.storage.db import get_db


def _utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _now_utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _parse_instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    return _utc_naive(parsed)


def _group_key(source: str, code: str) -> str:
    return f"{source}:{code}"


def _add_group(
    groups: dict[str, dict[str, Any]],
    *,
    source: str,
    code: str,
    level: str,
    observed_at: datetime,
    message: str,
) -> None:
    key = _group_key(source, code)
    current = groups.get(key)
    if current is None:
        groups[key] = {
            "group_id": key,
            "source": source,
            "code": code,
            "level": level,
            "occurrences": 1,
            "first_seen_at": observed_at,
            "last_seen_at": observed_at,
            "sample_message": message,
        }
        return

    current["occurrences"] += 1
    if observed_at < current["first_seen_at"]:
        current["first_seen_at"] = observed_at
    if observed_at >= current["last_seen_at"]:
        current["last_seen_at"] = observed_at
        current["sample_message"] = message


def build_error_incident_report(
    *,
    window_hours: int = 24,
    group_limit: int = 50,
    now: datetime | None = None,
) -> ErrorIncidentReportResponse:
    """Aggregate persisted incidents and real persisted success/failure outcomes.

    Incidents are grouped and counted but are intentionally excluded from the
    rate denominator because the incident log has no corresponding success rows.
    The failure-evidence rate uses only persisted outcome sources that carry a
    real denominator: terminal order-intent transitions and event-source runs.
    """

    bounded_hours = max(1, min(168, int(window_hours)))
    bounded_limit = max(1, min(100, int(group_limit)))
    window_end = _utc_naive(now or _now_utc())
    window_start = window_end - timedelta(hours=bounded_hours)
    start_iso = window_start.isoformat()
    end_iso = window_end.isoformat()

    with get_db() as db:
        incident_rows = db.execute(
            """
            SELECT id, created_at, level, code, message
            FROM incidents
            WHERE julianday(created_at) >= julianday(?)
              AND julianday(created_at) < julianday(?)
            ORDER BY created_at, id
            """,
            (start_iso, end_iso),
        ).fetchall()

        transition_rows = db.execute(
            """
            SELECT id, intent_id, created_at, to_status, reason
            FROM order_intent_transitions
            WHERE to_status IN ('executed', 'failed')
              AND julianday(created_at) >= julianday(?)
              AND julianday(created_at) < julianday(?)
            ORDER BY intent_id, id
            """,
            (start_iso, end_iso),
        ).fetchall()

        source_rows = db.execute(
            """
            SELECT id, completed_at, source, error
            FROM event_source_runs
            WHERE julianday(completed_at) >= julianday(?)
              AND julianday(completed_at) < julianday(?)
            ORDER BY completed_at, id
            """,
            (start_iso, end_iso),
        ).fetchall()

    groups: dict[str, dict[str, Any]] = {}

    for row in incident_rows:
        code = str(row["code"] or "unspecified_incident").strip() or "unspecified_incident"
        _add_group(
            groups,
            source="incident",
            code=code,
            level=str(row["level"] or "error"),
            observed_at=_parse_instant(str(row["created_at"])),
            message=str(row["message"] or code),
        )

    latest_terminal_by_intent: dict[int, Any] = {}
    for row in transition_rows:
        latest_terminal_by_intent[int(row["intent_id"])] = row

    failed_intents = 0
    for row in latest_terminal_by_intent.values():
        if str(row["to_status"]) != "failed":
            continue
        failed_intents += 1
        reason = str(row["reason"] or "").strip() or "unspecified_failure"
        _add_group(
            groups,
            source="order_intent_failure",
            code=reason,
            level="error",
            observed_at=_parse_instant(str(row["created_at"])),
            message=f"Order intent failed: {reason}",
        )

    failed_source_runs = 0
    for row in source_rows:
        error = str(row["error"] or "").strip()
        if not error:
            continue
        failed_source_runs += 1
        source = str(row["source"] or "").strip() or "unknown_source"
        _add_group(
            groups,
            source="event_source_failure",
            code=source,
            level="error",
            observed_at=_parse_instant(str(row["completed_at"])),
            message=error,
        )

    terminal_intents = len(latest_terminal_by_intent)
    event_runs = len(source_rows)
    denominator = terminal_intents + event_runs
    failure_evidence = failed_intents + failed_source_runs
    rate = (failure_evidence / denominator * 100.0) if denominator else None

    ordered = sorted(
        groups.values(),
        key=lambda item: (
            -int(item["occurrences"]),
            -item["last_seen_at"].timestamp(),
            str(item["group_id"]),
        ),
    )
    distinct_group_count = len(ordered)
    distinct_incident_groups = sum(1 for item in ordered if item["source"] == "incident")
    selected = ordered[:bounded_limit]

    return ErrorIncidentReportResponse(
        window_hours=bounded_hours,
        window_start_utc=window_start,
        window_end_utc=window_end,
        denominator_count=denominator,
        failure_evidence_count=failure_evidence,
        failure_evidence_rate_pct=rate,
        terminal_intent_outcomes=terminal_intents,
        failed_intent_outcomes=failed_intents,
        event_source_runs=event_runs,
        failed_event_source_runs=failed_source_runs,
        incident_occurrences=len(incident_rows),
        distinct_incident_groups=distinct_incident_groups,
        distinct_group_count=distinct_group_count,
        group_limit=bounded_limit,
        groups_truncated=distinct_group_count > bounded_limit,
        groups=[ErrorIncidentGroup(**item) for item in selected],
        message=(
            "Failure-evidence rate uses only terminal order-intent outcomes and event-source runs; "
            "persisted incidents are grouped separately because the incident log has no success denominator."
        ),
    )
