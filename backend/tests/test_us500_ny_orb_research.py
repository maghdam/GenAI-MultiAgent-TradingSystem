from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backend.services.strategy_policy import validate_strategy_source
from backend.services.strategy_sandbox import run_strategy_source


SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "strategies_generated"
    / "us500_ny_orb_m15_v1.py"
)
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")


def _session(day: str, *, breakout_close: float = 105.0) -> pd.DataFrame:
    local = pd.date_range(
        f"{day} 09:30", f"{day} 15:45", freq="15min",
        tz="America/New_York",
    )
    utc = local.tz_convert("UTC")
    frame = pd.DataFrame(
        {"open": 100.0, "high": 108.0, "low": 96.0, "close": 100.0, "volume": 1.0},
        index=utc,
    )
    frame.iloc[0, frame.columns.get_loc("high")] = 102.0
    frame.iloc[1, frame.columns.get_loc("high")] = 103.0
    frame.iloc[0, frame.columns.get_loc("low")] = 99.0
    frame.iloc[1, frame.columns.get_loc("low")] = 98.0
    frame.iloc[2, frame.columns.get_loc("close")] = breakout_close
    return frame


@pytest.mark.parametrize(
    ("day", "expected_first_utc"),
    [("2026-01-15", "2026-01-15 14:30+00:00"),
     ("2026-07-15", "2026-07-15 13:30+00:00")],
)
def test_orb_follows_ny_open_across_dst_and_exits_on_next_open(
    day: str, expected_first_utc: str,
) -> None:
    validate_strategy_source(SOURCE)
    frame = _session(day)
    assert frame.index[0] == pd.Timestamp(expected_first_utc)

    target = run_strategy_source(SOURCE, frame)

    assert target.index.equals(frame.index)
    assert target.iloc[:2].eq(0.0).all()
    assert target.iloc[2] == 1.0  # 10:00 candle closes at 10:15
    assert target.iloc[2:23].eq(1.0).all()
    assert target.iloc[23:].eq(0.0).all()  # 15:15 closes 15:30
    assert set(target.unique()).issubset({-1.0, 0.0, 1.0})


@pytest.mark.parametrize("missing", [0, 1])
def test_orb_requires_both_completed_opening_range_candles(missing: int) -> None:
    frame = _session("2026-07-15").drop(_session("2026-07-15").index[missing])
    target = run_strategy_source(SOURCE, frame)
    assert target.eq(0.0).all()


def test_orb_can_short_and_reject_late_entries() -> None:
    short = _session("2026-01-15", breakout_close=97.0)
    short_target = run_strategy_source(SOURCE, short)
    assert short_target.iloc[2] == -1.0
    assert short_target.iloc[22] == -1.0

    late = _session("2026-01-15", breakout_close=100.0)
    late.iloc[10, late.columns.get_loc("close")] = 105.0  # 12:00 bar is too late
    late_target = run_strategy_source(SOURCE, late)
    assert late_target.eq(0.0).all()


def test_orb_one_entry_per_day_even_after_opposite_break_and_new_day_resets() -> None:
    day1 = _session("2026-01-15")
    day1.iloc[3, day1.columns.get_loc("close")] = 97.0  # should not reverse
    day2 = _session("2026-01-16", breakout_close=97.0)
    frame = pd.concat([day1, day2])
    target = run_strategy_source(SOURCE, frame)

    assert target.iloc[3] == 1.0
    assert target.iloc[23] == 0.0
    assert target.iloc[len(day1)] == 0.0
    assert target.iloc[len(day1) + 2] == -1.0


def test_orb_prefixes_are_identical_without_future_bars() -> None:
    frame = pd.concat([_session("2026-01-15"), _session("2026-01-16")])
    complete = run_strategy_source(SOURCE, frame)
    for size in (1, 2, 3, 11, 23, 25, 26, 28, len(frame)):
        prefix = run_strategy_source(SOURCE, frame.iloc[:size])
        pd.testing.assert_series_equal(prefix, complete.iloc[:size])
