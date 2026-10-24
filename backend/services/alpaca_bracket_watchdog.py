"""Independent opt-in watchdog for the Paper bracket monitor heartbeat.

Local SQLite inspection ONLY. No Alpaca client import, broker API, engine route,
trading permission, process auto-start, alert delivery or automatic recovery.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Iterator

from backend.storage.db import get_db


_OK_STATUSES = frozenset({
    "snapshots_recorded_not_continuous_protection",
    "unverified_no_dispatched_brackets",
})
_UNSAFE_STATUSES = frozenset({
    "attention_required",
    "scope_exceeded_unverified",
})


def check_monitor_heartbeat(
    *, max_age_seconds: int = 120, observed_at: datetime | None = None,
) -> dict[str, object]:
    """Compare the last *completed* sweep with the wall clock and log transitions.

    Even a recent heartbeat proves only that one previous sweep finished.
    It says nothing about continuous protection, broker positions or whether
    another monitor/writer is running right now.
    """
    if (isinstance(max_age_seconds, bool) or not isinstance(max_age_seconds, int)
            or not 30 <= max_age_seconds <= 86400):
        raise ValueError("Watchdog stale threshold must be 30..86400 seconds")
    now = observed_at if observed_at is not None else datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Watchdog timestamp must have an explicit timezone")
    now = now.astimezone(timezone.utc)
    when = now.isoformat()

    # One SQLite transaction prevents concurrent watchdogs from emitting
    # duplicate transition alerts. Missing/locked DB is never considered OK.
    with get_db() as db:
        db.execute("BEGIN IMMEDIATE")
        try:
            last = db.execute(
                "SELECT completed_at, status FROM alpaca_bracket_monitor_heartbeat WHERE id = 1"
            ).fetchone()
            seconds: int | None = None
            if last is None:
                classification = "monitor_never_observed"
                severity = "critical"
            else:
                try:
                    observed = datetime.fromisoformat(last["completed_at"])
                    if observed.tzinfo is None or observed.utcoffset() is None:
                        raise ValueError("Naive monitor heartbeat")
                    lag = (now - observed.astimezone(timezone.utc)).total_seconds()
                    if lag < -5:
                        classification, severity = "monitor_clock_in_future", "critical"
                    elif lag > max_age_seconds:
                        classification, severity = "monitor_heartbeat_stale", "critical"
                    else:
                        seconds = max(0, int(lag))
                        if last["status"] in _OK_STATUSES:
                            classification, severity = "recent_completed_sweep_only", "info"
                        elif last["status"] in _UNSAFE_STATUSES:
                            classification, severity = "recent_sweep_flagged_unsafe", "critical"
                        else:
                            classification, severity = "monitor_status_unknown", "critical"
                    if seconds is None and lag >= 0:
                        seconds = int(lag)
                except (ValueError, TypeError, OverflowError):
                    classification, severity = "monitor_heartbeat_invalid", "critical"

            prior = db.execute(
                "SELECT classification, severity, first_seen_at "
                "FROM alpaca_bracket_monitor_watchdog_state WHERE id = 1"
            ).fetchone()
            changed = (prior is None or prior["classification"] != classification
                       or prior["severity"] != severity)
            first_seen = when if changed else prior["first_seen_at"]
            db.execute(
                """
                INSERT INTO alpaca_bracket_monitor_watchdog_state
                    (id, classification, severity, first_seen_at, last_seen_at)
                VALUES(1, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    classification=excluded.classification,
                    severity=excluded.severity,
                    first_seen_at=excluded.first_seen_at,
                    last_seen_at=excluded.last_seen_at
                """, (classification, severity, first_seen, when),
            )
            incident_created = False
            if changed and severity == "critical":
                db.execute(
                    """
                    INSERT INTO incidents(created_at, level, code, message, details_json)
                    VALUES (?, 'error', ?, ?, ?)
                    """,
                    (when, "ALPACA_PAPER_MONITOR_WATCHDOG_ALERT",
                     "Alpaca Paper bracket monitor heartbeat requires operator attention.",
                     json.dumps({
                         "classification": classification, "monitor_severity": severity,
                         "seconds_since_last_completed_sweep": seconds,
                         "watchdog_only": True, "broker_mutation_permitted": False,
                         "prior_alert_not_acknowledged": True,
                     }, sort_keys=True)),
                )
                incident_created = True
            elif changed and prior is not None and prior["severity"] == "critical":
                # A later healthy-looking heartbeat cannot close an older
                # incident, declare protection or re-enable trading.
                db.execute(
                    """
                    INSERT INTO incidents(created_at, level, code, message, details_json)
                    VALUES (?, 'warning', ?, ?, ?)
                    """,
                    (when, "ALPACA_PAPER_MONITOR_WATCHDOG_CHANGED",
                     "Monitor heartbeat changed; prior watchdog alert requires operator review.",
                     json.dumps({
                         "classification": classification,
                         "previous_classification": prior["classification"],
                         "prior_alert_not_acknowledged": True,
                         "watchdog_only": True, "broker_mutation_permitted": False,
                     }, sort_keys=True)),
                )
                incident_created = True
            db.commit()
        except BaseException:
            db.rollback()
            raise

    return {
        "classification": classification,
        "severity": severity,
        "seconds_since_last_completed_sweep": seconds,
        "incident_created": incident_created,
        "recent_completed_sweep_only": classification == "recent_completed_sweep_only",
        "continuous_monitor_liveness_verified": False,
        "current_protection_verified": False,
        "automatic_trading_allowed": False,
        "broker_mutation_permitted": False,
    }


def explicit_watchdog_cycles(
    *, interval_seconds: int = 30, max_age_seconds: int = 120,
    cycles: int | None = None,
) -> Iterator[dict[str, object]]:
    """Explicit, separate operator-owned process; never registered with app."""
    if (isinstance(interval_seconds, bool) or not isinstance(interval_seconds, int)
            or not 10 <= interval_seconds <= 3600):
        raise ValueError("Watchdog interval must be 10..3600 seconds")
    if cycles is not None and (
        isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 1
    ):
        raise ValueError("Cycles must be a positive integer or None")
    count = 0
    while cycles is None or count < cycles:
        yield check_monitor_heartbeat(max_age_seconds=max_age_seconds)
        count += 1
        if cycles is None or count < cycles:
            time.sleep(interval_seconds)
