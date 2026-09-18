from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from backend.domain.models import ConfluenceShadowRecord
from backend.services.confluence_replay import evaluate_confluence_replay


BASE = datetime(2026, 7, 19, 10, 0, 0)


def _record(
    record_id: int,
    *,
    at: datetime = BASE,
    signal: str = "long",
    original_pass: bool = True,
    shadow_pass: bool = True,
) -> ConfluenceShadowRecord:
    return ConfluenceShadowRecord(
        id=record_id,
        created_at=at,
        analysis_created_at=at,
        symbol="US100",
        timeframe="M5",
        strategy="smc",
        original_signal=signal,
        original_confidence=0.65,
        shadow_signal=signal,
        shadow_confidence=0.65 if shadow_pass else 0.45,
        confidence_adjustment=0.0 if shadow_pass else -0.20,
        action="confirm" if shadow_pass else "reduce",
        target_horizon="30m",
        event_score=0.5 if shadow_pass else -0.8,
        eligible_event_count=1,
        event_ids=[1],
        original_would_pass=original_pass,
        shadow_would_pass=shadow_pass,
        rationale="test",
        execution_unchanged=True,
    )


def _bars() -> pd.DataFrame:
    index = pd.date_range("2026-07-19T09:55:00Z", periods=20, freq="5min")
    values = [100, 999, *range(101, 119)]
    return pd.DataFrame(
        {
            "open": values,
            "close": values,
        },
        index=index,
    )


def test_replay_enters_on_next_bar_and_compares_threshold_policies() -> None:
    result = evaluate_confluence_replay(
        [_record(1, shadow_pass=False)], {("US100", "M5"): _bars()}, fee_bps_per_side=0.0
    )

    assert result.priced_records == 1
    assert result.original.trades == 1
    assert result.shadow.trades == 0
    # Signal candle at 10:00 has the deliberately impossible value 999. Entry must use 10:05 open=101.
    assert result.original.expectancy_pct == pytest.approx((107 / 101 - 1) * 100, abs=0.0001)
    assert result.research_only is True


def test_replay_applies_short_direction_and_round_trip_costs() -> None:
    result = evaluate_confluence_replay(
        [_record(1, signal="short")], {("US100", "M5"): _bars()}, fee_bps_per_side=5.0
    )

    expected = ((101 - 107) / 101) - 0.001
    assert result.original.expectancy_pct == pytest.approx(expected * 100, abs=0.0001)
    assert result.original.losses == 1


def test_replay_suppresses_overlapping_repeated_scans_per_policy() -> None:
    records = [
        _record(1, at=BASE),
        _record(2, at=BASE + timedelta(minutes=5)),
        _record(3, at=BASE + timedelta(minutes=35)),
    ]
    result = evaluate_confluence_replay(records, {("US100", "M5"): _bars()})

    assert result.original.candidate_decisions == 3
    assert result.original.trades == 2


def test_replay_reports_pending_and_unavailable_records() -> None:
    result = evaluate_confluence_replay(
        [_record(1), _record(2, at=BASE + timedelta(days=1))],
        {("US100", "M5"): _bars()},
    )

    assert result.priced_records == 1
    assert result.pending_records == 1
    assert result.verdict == "insufficient_data"
    assert any("Future exit bars" in warning for warning in result.warnings)


def test_no_trade_records_are_not_counted_as_pending() -> None:
    result = evaluate_confluence_replay(
        [_record(1, signal="no_trade", original_pass=False, shadow_pass=False)],
        {("US100", "M5"): _bars()},
    )

    assert result.total_records == 1
    assert result.priced_records == 0
    assert result.pending_records == 0
