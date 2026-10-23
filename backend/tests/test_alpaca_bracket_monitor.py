"""Offline-only monitoring tests. All broker reads and mutations are mocked."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from backend.services import alpaca_bracket_monitor as monitor
from backend.services import alpaca_paper
from backend.storage.alpaca_bracket_monitor import (
    finish_monitor_cycle, record_monitor_observation,
)
from backend.storage.db import get_db, init_db
from backend.storage.repositories import list_incidents
from backend.tests.test_alpaca_bracket_reconcile import _intent


@pytest.fixture
def no_broker_mutations(monkeypatch):
    def prohibit(*args, **kwargs):
        pytest.fail("Monitor must not send broker mutation requests")
    monkeypatch.setattr(alpaca_paper, "_request_mutation_json", prohibit)
    monkeypatch.setattr(alpaca_paper, "cancel_paper_bracket_order", prohibit)
    monkeypatch.setattr(alpaca_paper, "submit_paper_bracket_close", prohibit)


def _alerts():
    return [i for i in list_incidents(100)
            if i.code == "ALPACA_PAPER_BRACKET_MONITOR_ALERT"]


def test_empty_scope_never_claims_broker_is_flat(monkeypatch, no_broker_mutations):
    monkeypatch.setattr(monitor, "inspect_paper_bracket_exit",
                        lambda **kw: pytest.fail("No dispatched orders to inspect"))
    r = monitor.poll_paper_bracket_protection_once()
    assert r["status"] == "unverified_no_dispatched_brackets"
    assert not r["current_protection_verified"]
    assert r["startup_or_restart_gap"] is True
    assert r["counts"]["observed"] == 0
    assert any(i.code == "ALPACA_PAPER_BRACKET_MONITOR_GAP" for i in list_incidents(20))


def test_reserved_bracket_does_not_count_as_broker_exposure(monkeypatch, no_broker_mutations):
    _intent(state="bracket_reserved_not_submitted")
    monkeypatch.setattr(monitor, "inspect_paper_bracket_exit",
                        lambda **kw: pytest.fail("Unsubmitted reservation must not read broker"))
    r = monitor.poll_paper_bracket_protection_once()
    assert r["counts"]["unsubmitted"] == 1
    assert r["counts"]["observed"] == 0
    assert not _alerts()


def test_healthy_snapshot_is_not_current_protection_authority(monkeypatch, no_broker_mutations):
    intent = _intent()
    monkeypatch.setattr(monitor, "inspect_paper_bracket_exit",
                        lambda **kw: {"status": "reconciliation_required",
                                      "assessment": {"protected_at_observation": True,
                                                     "classification": "remaining_position_protected"}})
    r = monitor.poll_paper_bracket_protection_once()
    assert r["results"][0]["intent_id"] == intent.id
    assert r["status"] == "snapshots_recorded_not_continuous_protection"
    assert r["results"][0]["classification"] == "protected_snapshot_only"
    assert not r["current_protection_verified"]
    assert not r["automatic_trading_allowed"]
    assert not _alerts()


def test_missing_protection_escalates_once_across_restart_and_recovery(monkeypatch, no_broker_mutations):
    intent = _intent()
    snapshot = {"status": "reconciliation_required",
                "assessment": {"protected_at_observation": False,
                               "classification": "remaining_position_unprotected",
                               "remaining_quantity": "2"}}
    monkeypatch.setattr(monitor, "inspect_paper_bracket_exit", lambda **kw: dict(snapshot))
    first = monitor.poll_paper_bracket_protection_once()
    assert first["status"] == "attention_required"
    assert first["counts"]["attention"] == 1
    assert first["results"][0]["severity"] == "critical"
    assert len(_alerts()) == 1
    init_db()  # idempotent restart; no alert duplication after persisted state
    second = monitor.poll_paper_bracket_protection_once()
    assert second["counts"]["new_incidents"] == 0
    assert len(_alerts()) == 1
    snapshot["assessment"] = {"protected_at_observation": True,
                              "classification": "remaining_position_protected"}
    third = monitor.poll_paper_bracket_protection_once()
    assert third["results"][0]["severity"] == "info"
    assert len(_alerts()) == 1  # prior issue is NOT auto-acknowledged
    assert any(i.code == "ALPACA_PAPER_BRACKET_MONITOR_CHANGED" for i in list_incidents(20))
    with get_db() as db:
        row = db.execute("SELECT severity FROM alpaca_bracket_monitor_states WHERE intent_id=?",
                         (intent.id,)).fetchone()
    assert row["severity"] == "info"


@pytest.mark.parametrize("response", [{"status": "blocked"}, None, RuntimeError("offline mock")])
def test_unavailable_truth_is_critical(monkeypatch, no_broker_mutations, response):
    _intent()
    def lookup(**kw):
        if isinstance(response, Exception):
            raise response
        return response
    monkeypatch.setattr(monitor, "inspect_paper_bracket_exit", lookup)
    r = monitor.poll_paper_bracket_protection_once()
    assert r["counts"]["attention"] == 1
    assert r["results"][0]["classification"] == "broker_truth_unavailable_unverified"
    assert len(_alerts()) == 1


def test_unknown_outcome_is_escalated_without_broker_reads(monkeypatch, no_broker_mutations):
    _intent(state="unrecognized_future_dispatched_state")
    monkeypatch.setattr(monitor, "inspect_paper_bracket_exit",
                        lambda **kw: pytest.fail("Unknown intent must not read broker"))
    r = monitor.poll_paper_bracket_protection_once()
    assert r["counts"]["unknown"] == 1
    assert r["results"][0]["classification"] == "unknown_dispatch_state_unverified"


def test_scope_overflow_fails_without_skipping_old_orders(monkeypatch, no_broker_mutations):
    _intent()
    _intent()
    monkeypatch.setattr(monitor, "inspect_paper_bracket_exit",
                        lambda **kw: pytest.fail("Never silently poll a partial scope"))
    r = monitor.poll_paper_bracket_protection_once(max_intents=1)
    assert r["status"] == "scope_exceeded_unverified"
    assert r["current_protection_verified"] is False
    assert len(_alerts()) == 1
    assert _alerts()[0].details["intent_id"] == 0



def test_scope_recovery_rearms_future_critical_incident(monkeypatch, no_broker_mutations):
    _intent()
    _intent()
    monkeypatch.setattr(monitor, "inspect_paper_bracket_exit",
                        lambda **kw: pytest.fail("Scope overflow never reads a partial set"))
    assert monitor.poll_paper_bracket_protection_once(max_intents=1)["status"] == "scope_exceeded_unverified"
    assert len(_alerts()) == 1
    with get_db() as db:
        db.execute("UPDATE order_intents SET status = 'failed'")
        db.commit()
    assert monitor.poll_paper_bracket_protection_once(max_intents=1)["status"] == "unverified_no_dispatched_brackets"
    with get_db() as db:
        db.execute("UPDATE order_intents SET status = 'accepted'")
        db.commit()
    assert monitor.poll_paper_bracket_protection_once(max_intents=1)["status"] == "scope_exceeded_unverified"
    assert len(_alerts()) == 2  # separate operator incident for renewed overflow

def test_monitor_heartbeat_survives_restart_and_detects_gap():
    now = datetime(2026, 10, 23, 11, 0, tzinfo=timezone.utc)
    assert finish_monitor_cycle(completed_at=now, status="idle", gap_seconds=120)
    init_db()
    assert not finish_monitor_cycle(completed_at=now + timedelta(seconds=50),
                                    status="idle", gap_seconds=120)
    assert finish_monitor_cycle(completed_at=now + timedelta(seconds=240),
                                status="idle", gap_seconds=120)
    assert [i.code for i in list_incidents(20)].count("ALPACA_PAPER_BRACKET_MONITOR_GAP") == 2


def test_incident_transition_is_atomic_and_deduplicated_after_restart():
    now = datetime.now(timezone.utc)
    assert record_monitor_observation(intent_id=7, classification="missing_sl",
                                      severity="critical", observed_at=now)
    init_db()
    assert not record_monitor_observation(intent_id=7, classification="missing_sl",
                                          severity="critical", observed_at=now + timedelta(seconds=30))
    assert len(_alerts()) == 1


def test_late_snapshot_is_unverified(monkeypatch, no_broker_mutations):
    _intent()
    ticks = iter([0.0, 120.0])
    monkeypatch.setattr(monitor.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(monitor, "inspect_paper_bracket_exit",
                        lambda **kw: {"status": "flat_terminal_at_observation"})
    r = monitor.poll_paper_bracket_protection_once(max_read_seconds=10)
    assert r["results"][0]["classification"] == "late_broker_snapshot_unverified"
    assert r["results"][0]["severity"] == "critical"


def test_opt_in_loop_bounded_cycles_and_input_validation(monkeypatch, no_broker_mutations):
    calls = []
    monkeypatch.setattr(monitor, "poll_paper_bracket_protection_once",
                        lambda **kw: calls.append(kw) or {"snapshot_only": True})
    monkeypatch.setattr(monitor.time, "sleep", lambda seconds: calls.append(("sleep", seconds)))
    assert len(list(monitor.explicit_monitor_cycles(cycles=2, interval_seconds=10))) == 2
    assert len(calls) == 3
    with pytest.raises(ValueError):
        list(monitor.explicit_monitor_cycles(cycles=0))
    with pytest.raises(ValueError):
        list(monitor.explicit_monitor_cycles(interval_seconds=1, cycles=1))


def test_cli_needs_read_authorization_even_for_one_sweep(monkeypatch):
    from scripts import monitor_alpaca_paper_brackets as cli
    with pytest.raises(SystemExit) as error:
        cli.main(["--once"])
    assert error.value.code == 2
    monkeypatch.setattr(cli, "explicit_monitor_cycles",
                        lambda **kw: iter([{"snapshot_only": True}]))
    assert cli.main(["--once", "--acknowledge-paper-reads"]) == 0
