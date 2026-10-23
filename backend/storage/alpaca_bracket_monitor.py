"""Durable, restart-aware incident transitions for explicit Paper bracket observations.

No broker access or mutations live here. Alerts persist in the existing System
incidents table; observing later apparent recovery never acknowledges an alert.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from backend.storage.db import get_db


def candidate_paper_equity_brackets(*, limit: int) -> list[dict[str, object]]:
    """List accepted bracket intents with no rolling 200-row blind spot."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("Positive candidate bound required")
    with get_db() as db:
        rows = db.execute(
            """
            SELECT id, symbol, json_extract(details_json, '$.outcome_state') AS outcome
            FROM order_intents
            WHERE status = 'accepted' AND intent_type = 'open'
              AND json_extract(details_json, '$.broker_provider') = 'alpaca'
              AND json_extract(details_json, '$.broker_environment') = 'paper'
              AND json_extract(details_json, '$.asset_class') = 'us_equity'
              AND json_extract(details_json, '$.protection_mode') = 'equity_bracket'
            ORDER BY id ASC LIMIT ?
            """, (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def _incident(db, *, when: str, severity: str, code: str,
              message: str, details: dict[str, object]) -> None:
    # IncidentRecord supports only info/warning/error. Preserve critical
    # monitor severity in details, while displaying a critical alert as error.
    level = "error" if severity == "critical" else severity
    payload = {**details, "monitor_severity": severity}
    db.execute(
        """
        INSERT INTO incidents(created_at, level, code, message, details_json)
        VALUES(?, ?, ?, ?, ?)
        """, (when, level, code, message, json.dumps(payload, sort_keys=True)),
    )


def record_monitor_observation(*, intent_id: int, classification: str,
                               severity: str, observed_at: datetime) -> bool:
    """Atomically persist latest classification and a deduplicated incident.

    Repeated identical unsafe polls do not flood the incident list. Severity
    reductions create an operator-review notice rather than closing any alert.
    """
    if severity not in {"info", "warning", "critical"} or not classification:
        raise ValueError("Valid monitoring severity and classification required")
    when = observed_at.astimezone(timezone.utc).isoformat()
    with get_db() as db:
        db.execute("BEGIN IMMEDIATE")
        try:
            previous = db.execute(
                "SELECT classification, severity, first_seen_at "
                "FROM alpaca_bracket_monitor_states WHERE intent_id = ?", (intent_id,),
            ).fetchone()
            changed = (previous is None
                       or previous["classification"] != classification
                       or previous["severity"] != severity)
            first_seen = when if changed or previous is None else previous["first_seen_at"]
            db.execute(
                """
                INSERT INTO alpaca_bracket_monitor_states
                    (intent_id, classification, severity, first_seen_at, last_seen_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(intent_id) DO UPDATE SET
                    classification=excluded.classification,
                    severity=excluded.severity,
                    first_seen_at=excluded.first_seen_at,
                    last_seen_at=excluded.last_seen_at
                """, (intent_id, classification, severity, first_seen, when),
            )
            escalated = False
            if changed and severity != "info":
                _incident(
                    db, when=when, severity=severity,
                    code="ALPACA_PAPER_BRACKET_MONITOR_ALERT",
                    message="Paper bracket requires operator attention; no automatic recovery.",
                    details={"intent_id": intent_id, "classification": classification,
                             "monitor_only": True, "automatic_trading_allowed": False},
                )
                escalated = True
            elif changed and previous is not None and previous["severity"] != "info":
                _incident(
                    db, when=when, severity="warning",
                    code="ALPACA_PAPER_BRACKET_MONITOR_CHANGED",
                    message="Later broker snapshot changed; review prior incident manually.",
                    details={"intent_id": intent_id, "classification": classification,
                             "prior_classification": previous["classification"],
                             "prior_alert_not_acknowledged": True},
                )
                escalated = True
            db.commit()
            return escalated
        except BaseException:
            db.rollback()
            raise


def finish_monitor_cycle(*, completed_at: datetime, status: str,
                         gap_seconds: int) -> bool:
    """Persist a completed sweep and report a missed restart/watch interval.

    No daemon or external liveness guarantee is implied. A dead monitor cannot
    raise a fresh incident until a subsequent sweep is actually started.
    """
    when = completed_at.astimezone(timezone.utc).isoformat()
    with get_db() as db:
        db.execute("BEGIN IMMEDIATE")
        try:
            old = db.execute(
                "SELECT completed_at FROM alpaca_bracket_monitor_heartbeat WHERE id = 1"
            ).fetchone()
            gap = None
            if old is not None:
                then = datetime.fromisoformat(old["completed_at"])
                gap = max(0, (completed_at.astimezone(timezone.utc) - then).total_seconds())
            missed = old is None or (gap is not None and gap > gap_seconds)
            if missed:
                _incident(
                    db, when=when, severity="warning",
                    code="ALPACA_PAPER_BRACKET_MONITOR_GAP",
                    message="Paper protection monitor has an unobserved startup/restart gap.",
                    details={"seconds_since_previous_poll": gap, "startup": old is None,
                             "monitor_only": True, "no_liveness_guarantee": True},
                )
            db.execute(
                """
                INSERT INTO alpaca_bracket_monitor_heartbeat(id, completed_at, status)
                VALUES(1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    completed_at=excluded.completed_at, status=excluded.status
                """, (when, status),
            )
            db.commit()
            return missed
        except BaseException:
            db.rollback()
            raise
