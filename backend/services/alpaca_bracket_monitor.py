"""Operator-started read-only Alpaca Paper bracket monitoring (Phase 10.21).

The loop is NEVER started by an import, backend boot, strategy, scheduler or UI.
Every classification is historical snapshot evidence, not a current risk gate.
No order submit, cancel, replace, flatten, retry or intent completion occurs.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Iterator

from backend.services.alpaca_bracket_exit import inspect_paper_bracket_exit
from backend.services.alpaca_bracket_reconcile import _DISPATCHED_STATES
from backend.storage.alpaca_bracket_monitor import (
    candidate_paper_equity_brackets,
    finish_monitor_cycle,
    record_monitor_observation,
)

_UNSUBMITTED = "bracket_reserved_not_submitted"


def poll_paper_bracket_protection_once(
    *, max_intents: int = 25, max_read_seconds: float = 45.0,
    gap_seconds: int = 120,
) -> dict[str, object]:
    """One bounded-scope sweep; fail closed when broker truth cannot be read.

    HTTP client timeouts provide the underlying per-request time bound; this
    detects late total reads but cannot forcibly interrupt a hung library call.
    """
    if isinstance(max_intents, bool) or not isinstance(max_intents, int) or not 1 <= max_intents <= 100:
        raise ValueError("Monitoring must be limited to 1..100 intents per sweep")
    if not 5 <= max_read_seconds <= 300 or not 15 <= gap_seconds <= 86400:
        raise ValueError("Invalid monitoring read/gap threshold")

    candidates = candidate_paper_equity_brackets(limit=max_intents + 1)
    counts = {"observed": 0, "attention": 0, "new_incidents": 0,
              "unsubmitted": 0, "unknown": 0}
    results: list[dict[str, object]] = []
    if len(candidates) > max_intents:
        now = datetime.now(timezone.utc)
        raised = record_monitor_observation(
            intent_id=0, classification="scope_exceeded_unverified",
            severity="critical", observed_at=now,
        )
        gap = finish_monitor_cycle(completed_at=now, status="scope_exceeded_unverified",
                                   gap_seconds=gap_seconds)
        return {
            "status": "scope_exceeded_unverified", "counts": counts,
            "incident_created": raised or gap, "snapshot_only": True,
            "current_protection_verified": False, "automatic_trading_allowed": False,
            "results": [],
        }

    for candidate in candidates:
        intent_id = int(candidate["id"])
        outcome = candidate.get("outcome")
        if outcome == _UNSUBMITTED:
            counts["unsubmitted"] += 1
            continue
        observed = time.monotonic()
        if outcome not in _DISPATCHED_STATES:
            severity, classification = "critical", "unknown_dispatch_state_unverified"
            counts["unknown"] += 1
        else:
            try:
                inspected = inspect_paper_bracket_exit(intent_id=intent_id)
            except Exception:
                # Never include sensitive exception details in the public
                # incident. Failed persistence/read is NOT a healthy result.
                inspected = {"status": "blocked"}
            assessment = inspected.get("assessment") if isinstance(inspected, dict) else None
            elapsed = time.monotonic() - observed
            if elapsed > max_read_seconds:
                severity, classification = "critical", "late_broker_snapshot_unverified"
            elif not isinstance(inspected, dict) or inspected.get("status") == "blocked":
                severity, classification = "critical", "broker_truth_unavailable_unverified"
            elif inspected.get("status") == "flat_terminal_at_observation":
                severity, classification = "info", "flat_terminal_snapshot_only"
            elif isinstance(assessment, dict) and assessment.get("protected_at_observation") is True:
                severity, classification = "info", "protected_snapshot_only"
            elif isinstance(assessment, dict):
                classification = str(assessment.get("classification") or "unknown") + "_unverified"
                severity = ("warning" if assessment.get("classification") == "flat_orders_pending"
                            else "critical")
            else:
                severity, classification = "critical", "incomplete_broker_snapshot_unverified"

        created = record_monitor_observation(
            intent_id=intent_id, classification=classification, severity=severity,
            observed_at=datetime.now(timezone.utc),
        )
        counts["observed"] += 1
        counts["attention"] += int(severity != "info")
        counts["new_incidents"] += int(created)
        results.append({"intent_id": intent_id, "symbol": candidate["symbol"],
                        "classification": classification, "severity": severity,
                        "incident_created": created, "snapshot_only": True})

    status = ("attention_required" if counts["attention"] else
              "unverified_no_dispatched_brackets" if counts["observed"] == 0 else
              "snapshots_recorded_not_continuous_protection")
    missed = finish_monitor_cycle(
        completed_at=datetime.now(timezone.utc), status=status,
        gap_seconds=gap_seconds,
    )
    counts["new_incidents"] += int(missed)
    return {
        "status": status, "counts": counts, "startup_or_restart_gap": missed,
        "results": results, "snapshot_only": True,
        "current_protection_verified": False, "automatic_trading_allowed": False,
        "broker_mutation_permitted": False,
    }


def explicit_monitor_cycles(
    *, interval_seconds: int = 30, max_intents: int = 25,
    cycles: int | None = None,
) -> Iterator[dict[str, object]]:
    """Yield opt-in operator-owned sweeps; never spawn workers automatically."""
    if (isinstance(interval_seconds, bool) or not isinstance(interval_seconds, int)
            or not 10 <= interval_seconds <= 3600):
        raise ValueError("Monitor interval must be 10..3600 seconds")
    if cycles is not None and (isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 1):
        raise ValueError("Cycles must be positive or None")
    count = 0
    while cycles is None or count < cycles:
        yield poll_paper_bracket_protection_once(
            max_intents=max_intents, gap_seconds=max(120, 3 * interval_seconds),
        )
        count += 1
        if cycles is None or count < cycles:
            time.sleep(interval_seconds)
