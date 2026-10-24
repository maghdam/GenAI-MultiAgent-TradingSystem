"""Offline watchdog regression; no Alpaca broker reads or mutations."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from backend.services import alpaca_bracket_watchdog as watchdog
from backend.services import alpaca_paper
from backend.storage.alpaca_bracket_monitor import finish_monitor_cycle
from backend.storage.db import get_db, init_db
from backend.storage.repositories import list_incidents

NOW = datetime(2026, 10, 24, 10, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def forbid_broker_calls(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Watchdog must never query or mutate Alpaca broker")
    for name in ("get_account", "get_open_position", "list_orders",
                 "get_order_by_client_order_id", "_request_mutation_json",
                 "submit_paper_bracket_close", "cancel_paper_bracket_order"):
        monkeypatch.setattr(alpaca_paper, name, fail)


def _watchdog_events():
    return [i for i in list_incidents(100)
            if i.code.startswith("ALPACA_PAPER_MONITOR_WATCHDOG_")]


def _heartbeat(completed_at: datetime, status="snapshots_recorded_not_continuous_protection"):
    finish_monitor_cycle(completed_at=completed_at, status=status, gap_seconds=120)


def test_missing_heartbeat_is_critical_incident_without_broker_calls():
    r = watchdog.check_monitor_heartbeat(observed_at=NOW)
    assert r["classification"] == "monitor_never_observed"
    assert r["severity"] == "critical"
    assert not r["current_protection_verified"]
    assert not r["continuous_monitor_liveness_verified"]
    assert not r["automatic_trading_allowed"]
    assert r["incident_created"]
    alert = _watchdog_events()[0]
    assert alert.level == "error"
    assert alert.details["monitor_severity"] == "critical"


def test_repeated_critical_is_deduplicated_across_restarts():
    assert watchdog.check_monitor_heartbeat(observed_at=NOW)["incident_created"]
    init_db()
    later = watchdog.check_monitor_heartbeat(observed_at=NOW + timedelta(seconds=50))
    assert not later["incident_created"]
    assert len(_watchdog_events()) == 1


def test_healthy_recent_sweep_does_not_verify_protection():
    _heartbeat(NOW - timedelta(seconds=20))
    r = watchdog.check_monitor_heartbeat(observed_at=NOW)
    assert r["classification"] == "recent_completed_sweep_only"
    assert r["severity"] == "info"
    assert r["seconds_since_last_completed_sweep"] == 20
    assert r["recent_completed_sweep_only"]
    assert not r["continuous_monitor_liveness_verified"]
    assert not r["current_protection_verified"]
    assert not _watchdog_events()


def test_stale_then_recent_preserves_old_alert_and_rearms():
    _heartbeat(NOW - timedelta(minutes=7))
    first = watchdog.check_monitor_heartbeat(observed_at=NOW, max_age_seconds=90)
    assert first["classification"] == "monitor_heartbeat_stale"
    assert first["incident_created"]
    assert first["seconds_since_last_completed_sweep"] == 420
    _heartbeat(NOW + timedelta(seconds=20))
    recovered = watchdog.check_monitor_heartbeat(observed_at=NOW + timedelta(seconds=30))
    assert recovered["severity"] == "info"
    assert recovered["incident_created"]  # review warning, not auto-clear
    assert [i.code for i in _watchdog_events()] == [
        "ALPACA_PAPER_MONITOR_WATCHDOG_ALERT",
        "ALPACA_PAPER_MONITOR_WATCHDOG_CHANGED",
    ]
    assert _watchdog_events()[-1].details["prior_alert_not_acknowledged"]
    subsequent = watchdog.check_monitor_heartbeat(observed_at=NOW + timedelta(seconds=40))
    assert not subsequent["incident_created"]
    again = watchdog.check_monitor_heartbeat(observed_at=NOW + timedelta(seconds=200),
                                             max_age_seconds=90)
    assert again["classification"] == "monitor_heartbeat_stale"
    assert again["incident_created"]
    assert len(_watchdog_events()) == 3


@pytest.mark.parametrize("status", ["attention_required", "scope_exceeded_unverified", "impossible"])
def test_recent_but_unsafe_sweep_still_alerts(status):
    _heartbeat(NOW - timedelta(seconds=1), status)
    r = watchdog.check_monitor_heartbeat(observed_at=NOW)
    assert r["severity"] == "critical"
    assert r["classification"] in {"recent_sweep_flagged_unsafe", "monitor_status_unknown"}
    assert r["seconds_since_last_completed_sweep"] == 1
    assert len(_watchdog_events()) == 1


def test_future_timestamp_fails_closed():
    _heartbeat(NOW + timedelta(hours=2))
    r = watchdog.check_monitor_heartbeat(observed_at=NOW)
    assert r["classification"] == "monitor_clock_in_future"
    assert r["severity"] == "critical"


@pytest.mark.parametrize("bad_time", ["not-a-date", "2026-10-24T10:00:00"])
def test_corrupt_or_naive_timestamp_fails_closed(bad_time):
    with get_db() as db:
        db.execute("INSERT INTO alpaca_bracket_monitor_heartbeat VALUES(1, ?, ?)",
                   (bad_time, "snapshots_recorded_not_continuous_protection"))
        db.commit()
    r = watchdog.check_monitor_heartbeat(observed_at=NOW)
    assert r["classification"] == "monitor_heartbeat_invalid"
    assert r["severity"] == "critical"


def test_watchdog_does_not_accept_naive_clock_or_out_of_range_threshold():
    with pytest.raises(ValueError):
        watchdog.check_monitor_heartbeat(observed_at=datetime(2026, 10, 24))
    with pytest.raises(ValueError):
        watchdog.check_monitor_heartbeat(max_age_seconds=0)
    with pytest.raises(ValueError):
        watchdog.check_monitor_heartbeat(max_age_seconds=True)


def test_independent_opt_in_loop_does_not_spawn_or_self_start(monkeypatch):
    calls = []
    monkeypatch.setattr(
        watchdog, "check_monitor_heartbeat",
        lambda **kw: calls.append(kw) or {"severity": "info"},
    )
    monkeypatch.setattr(watchdog.time, "sleep", lambda seconds: calls.append(("sleep", seconds)))
    assert len(list(watchdog.explicit_watchdog_cycles(cycles=2, interval_seconds=10))) == 2
    assert len(calls) == 3
    with pytest.raises(ValueError):
        list(watchdog.explicit_watchdog_cycles(cycles=0))
    with pytest.raises(ValueError):
        list(watchdog.explicit_watchdog_cycles(interval_seconds=1, cycles=1))


def test_cli_once_returns_nonzero_for_unsafe_heartbeat(monkeypatch, capsys):
    from scripts import watch_alpaca_monitor as cli

    monkeypatch.setattr(cli, "explicit_watchdog_cycles",
                        lambda **kw: iter([{"severity": "critical", "classification": "stale"}]))
    assert cli.main(["--once"]) == 2
    assert "ACTION REQUIRED" in capsys.readouterr().err
    monkeypatch.setattr(cli, "explicit_watchdog_cycles",
                        lambda **kw: iter([{"severity": "info", "classification": "recent"}]))
    assert cli.main(["--once"]) == 0


def test_cli_fails_closed_when_local_database_unavailable(monkeypatch, capsys):
    from scripts import watch_alpaca_monitor as cli

    def fail():
        raise RuntimeError("No DB")
    monkeypatch.setattr(cli, "init_db", fail)
    assert cli.main(["--once"]) == 3
    assert "UNVERIFIED" in capsys.readouterr().err
