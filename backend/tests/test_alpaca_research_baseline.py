from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest
from fastapi import HTTPException

from scripts import research_alpaca_strategy_baseline as research


def _bars(count: int = 260) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=count, freq="B", tz="America/New_York").tz_convert("UTC")
    close = pd.Series([100.0 + 0.07 * i for i in range(count)], index=idx)
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000,
        },
        index=idx,
    )


@pytest.fixture(autouse=True)
def _deterministic_read_only_calendar(monkeypatch):
    # Isolate research tests from external trading APIs and account state.
    # A mocked weekday calendar is not used to establish real NYSE closures.
    def calendar(*, start_date: str, end_date: str):
        return [
            {"date": day.date().isoformat(), "open": "09:30", "close": "16:00"}
            for day in pd.date_range(start_date, end_date, freq="B")
        ]

    monkeypatch.setattr(research, "get_stock_trading_calendar", calendar)


def test_baseline_uses_alpaca_bars_only_and_existing_auditor(monkeypatch):
    requested = []
    audited = []

    def fake_bars(**kwargs):
        requested.append(kwargs)
        return _bars()

    def fake_audit(**kwargs):
        audited.append(kwargs)
        return {
            "symbol": kwargs["symbol"],
            "strategy": kwargs["strategy_key"],
            "Full Backtest": {"Total Return [%]": 1.0},
            "Out of Sample": {"Total Return [%]": 0.5},
        }

    monkeypatch.setattr(research, "get_bars", fake_bars)
    monkeypatch.setattr(research, "_audit_on_frame", fake_audit)
    monkeypatch.setattr(research, "_stock_feed", lambda: "iex")

    report = research.run_baseline()
    assert requested == [
        {"symbol": "SPY", "timeframe": "D1", "asset_class": "us_equity", "num_bars": 500},
        {"symbol": "QQQ", "timeframe": "D1", "asset_class": "us_equity", "num_bars": 500},
    ]
    assert [(row["symbol"], row["strategy_key"]) for row in audited] == [
        (symbol, strategy)
        for symbol in ("SPY", "QQQ")
        for strategy in ("sma_cross", "rsi_reversal", "breakout")
    ]
    assert all(row["fee_bps"] == 1.0 and row["slippage_bps"] == 1.0 for row in audited)
    assert all(row["spread_bps"] == 2.0 and row["position_size_pct"] == 100.0 for row in audited)
    assert report["read_only"] and report["broker_order_requests"] == 0
    assert report["data_source"]["stock_feed"] == "iex"
    assert report["data_source"]["session_calendar"] == "alpaca_paper_trading_day_calendar"
    assert all(v["verified_observed_trading_dates"] == 260 for v in report["data_source"]["calendar_validation_by_symbol"].values())
    assert len(report["results"]) == 6
    assert "$500" in " ".join(report["limitations"])


def test_stock_daily_baseline_accepts_new_york_midnight_across_dst(monkeypatch):
    captured = []
    raw = _bars()
    assert {int(hour) for hour in raw.index.hour} == {4, 5}

    monkeypatch.setattr(research, "get_bars", lambda **kw: raw.copy())
    monkeypatch.setattr(
        research,
        "_audit_on_frame",
        lambda **kw: captured.append(kw["df"].copy()) or {"strategy": kw["strategy_key"]},
    )
    monkeypatch.setattr(research, "_stock_feed", lambda: "iex")

    result = research.run_baseline(symbols=("SPY",))

    assert len(captured) == 3
    assert all(frame.index.equals(raw.index) for frame in captured)
    assert result["data_source"]["daily_bar_label"] == "00:00 America/New_York"
    assert result["data_source"]["session_calendar"] == "alpaca_paper_trading_day_calendar"


@pytest.mark.parametrize(
    ("invalid_label", "reason"),
    [
        ("2024-01-03T00:00:00Z", "midnight America/New_York"),
        ("2024-01-06T05:00:00Z", "weekend New York session label"),
    ],
)
def test_stock_daily_baseline_rejects_bad_provider_session_labels_before_audit(
    monkeypatch, invalid_label: str, reason: str
):
    frame = _bars()
    updated_index = frame.index.to_list()
    updated_index[2] = pd.Timestamp(invalid_label)
    frame.index = pd.DatetimeIndex(updated_index)
    monkeypatch.setattr(research, "get_bars", lambda **kw: frame.copy())
    monkeypatch.setattr(
        research, "_audit_on_frame",
        lambda **kw: pytest.fail("Untrusted daily labels must fail before strategy audit"),
    )
    monkeypatch.setattr(research, "_stock_feed", lambda: "iex")

    with pytest.raises(ValueError, match=reason):
        research.run_baseline(symbols=("SPY",))


def test_d1_calendar_rejects_weekday_exchange_holiday_before_audit(monkeypatch):
    frame = _bars()
    excluded_date = frame.index[6].tz_convert("America/New_York").date().isoformat()

    def calendar(*, start_date: str, end_date: str):
        return [
            {"date": day.date().isoformat(), "open": "09:30", "close": "16:00"}
            for day in pd.date_range(start_date, end_date, freq="B")
            if day.date().isoformat() != excluded_date
        ]

    monkeypatch.setattr(research, "get_stock_trading_calendar", calendar)
    monkeypatch.setattr(research, "get_bars", lambda **kw: frame.copy())
    monkeypatch.setattr(research, "_stock_feed", lambda: "iex")
    monkeypatch.setattr(
        research, "_audit_on_frame",
        lambda **kw: pytest.fail("A market-closed bar must never enter the strategy audit"),
    )
    with pytest.raises(ValueError, match="non-trading exchange date"):
        research.run_baseline(symbols=("SPY",))


def test_d1_calendar_reports_early_closes_without_filling(monkeypatch):
    frame = _bars()
    short_day = frame.index[12].tz_convert("America/New_York").date().isoformat()

    def calendar(*, start_date: str, end_date: str):
        return [
            {
                "date": day.date().isoformat(),
                "open": "09:30",
                "close": "13:00" if day.date().isoformat() == short_day else "16:00",
            }
            for day in pd.date_range(start_date, end_date, freq="B")
        ]

    monkeypatch.setattr(research, "get_stock_trading_calendar", calendar)
    monkeypatch.setattr(research, "get_bars", lambda **kw: frame.copy())
    audited = []
    monkeypatch.setattr(
        research, "_audit_on_frame",
        lambda **kw: audited.append(kw["df"].copy()) or {"strategy": kw["strategy_key"]},
    )
    monkeypatch.setattr(research, "_stock_feed", lambda: "iex")

    report = research.run_baseline(symbols=("SPY",))
    summary = report["data_source"]["calendar_validation_by_symbol"]["SPY"]
    assert summary["early_close_observations"] == 1
    assert summary["verified_observed_trading_dates"] == 260
    assert summary["unobserved_exchange_sessions"] == 0
    assert summary["exchange_session_coverage_checked"] is True
    assert not summary["observed_bar_completeness_verified"]
    assert len(audited) == 3 and all(bar.index.equals(frame.index) for bar in audited)


@pytest.mark.parametrize("missing_count", [1, 4])
def test_d1_missing_exchange_sessions_fail_closed_before_strategy_audit(
    monkeypatch, missing_count
):
    raw = _bars()
    removed = raw.index[20 : 20 + missing_count]
    missing_labels = [
        timestamp.tz_convert("America/New_York").date().isoformat()
        for timestamp in removed
    ]
    frame = raw.drop(index=removed)
    assert len(frame) >= research.MIN_COMPLETE_BARS

    def calendar(*, start_date: str, end_date: str):
        return [
            {"date": day.date().isoformat(), "open": "09:30", "close": "16:00"}
            for day in pd.date_range(start_date, end_date, freq="B")
        ]

    monkeypatch.setattr(research, "get_stock_trading_calendar", calendar)
    monkeypatch.setattr(research, "get_bars", lambda **kw: frame.copy())
    monkeypatch.setattr(research, "_stock_feed", lambda: "iex")
    monkeypatch.setattr(
        research,
        "_audit_on_frame",
        lambda **kw: pytest.fail("Missing exchange dates must reject before any strategy audit"),
    )
    with pytest.raises(ValueError, match=f"SPY: {missing_count} missing exchange trading") as exc:
        research.run_baseline(symbols=("SPY",))
    for label in missing_labels[:3]:
        assert label in str(exc.value)
    assert ("..." in str(exc.value)) == (missing_count > 3)


def test_d1_missing_exchange_sessions_cli_fails_without_report(monkeypatch, tmp_path):
    raw = _bars()
    frame = raw.drop(index=raw.index[20])
    monkeypatch.setattr(research, "get_bars", lambda **kw: frame.copy())
    monkeypatch.setattr(research, "_stock_feed", lambda: "iex")
    monkeypatch.setattr(
        research,
        "_audit_on_frame",
        lambda **kw: pytest.fail("CLI must reject before strategy audit"),
    )
    target = tmp_path / "not-created.json"
    assert research.main(["--symbol", "SPY", "--output", str(target)]) == 2
    assert not target.exists()


@pytest.mark.parametrize(
    "field, value",
    [("open", "bad"), ("close", "09:00"), ("date", "2024-99-01")],
)
def test_d1_calendar_fails_closed_on_malformed_session_details(monkeypatch, field, value):
    frame = _bars()
    def calendar(*, start_date: str, end_date: str):
        result = [
            {"date": day.date().isoformat(), "open": "09:30", "close": "16:00"}
            for day in pd.date_range(start_date, end_date, freq="B")
        ]
        result[0][field] = value
        return result
    monkeypatch.setattr(research, "get_stock_trading_calendar", calendar)
    monkeypatch.setattr(research, "get_bars", lambda **kw: frame.copy())
    monkeypatch.setattr(research, "_stock_feed", lambda: "iex")
    monkeypatch.setattr(
        research, "_audit_on_frame",
        lambda **kw: pytest.fail("Malformed exchange calendar must prevent auditing"),
    )
    with pytest.raises(research.AlpacaMarketDataError, match="calendar"):
        research.run_baseline(symbols=("SPY",))


def test_rejects_bad_scope_or_cost_before_any_provider_call(monkeypatch):
    monkeypatch.setattr(research, "get_bars", lambda **kw: pytest.fail("No Alpaca reads allowed for bad input"))
    for kwargs in (
        {"symbols": ("TSLA",)},
        {"symbols": ("SPY", "SPY")},
        {"symbols": ()},
        {"num_bars": 249},
        {"num_bars": 1001},
        {"num_bars": True},
        {"fee_bps": -1},
        {"slippage_bps": float("nan")},
        {"spread_bps": float("inf")},
    ):
        with pytest.raises((ValueError, HTTPException)) as error:
            research.run_baseline(**kwargs)
        assert "No Alpaca reads allowed" not in str(error.value)


def test_requires_sufficient_completed_bars_and_never_audits_partial_data(monkeypatch):
    audits = []
    monkeypatch.setattr(research, "get_bars", lambda **kw: _bars(249))
    monkeypatch.setattr(research, "_audit_on_frame", lambda **kw: audits.append(kw))
    with pytest.raises(ValueError, match="completed daily bars"):
        research.run_baseline(symbols=("SPY",))
    assert audits == []


def test_discards_current_utc_day_as_potentially_incomplete(monkeypatch):
    historical = _bars()
    today = pd.Timestamp(datetime.now(UTC).date(), tz="UTC")
    current = pd.DataFrame(
        {"open": [999.0], "high": [999.0], "low": [999.0], "close": [999.0], "volume": [1.0]},
        index=pd.DatetimeIndex([today]),
    )
    monkeypatch.setattr(research, "get_bars", lambda **kw: pd.concat([historical, current]))
    inspected = []
    monkeypatch.setattr(
        research,
        "_audit_on_frame",
        lambda **kw: inspected.append(kw["df"].copy()) or {"strategy": kw["strategy_key"]},
    )
    research.run_baseline(symbols=("SPY",))
    assert len(inspected) == 3
    assert all(len(frame) == 260 and today not in frame.index for frame in inspected)


def test_cli_writes_explicit_local_json_report_without_orders(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(research, "run_baseline", lambda **kw: {"mode": "research_only", "symbols": ["SPY"]})
    path = tmp_path / "study" / "report.json"
    assert research.main(["--symbol", "SPY", "--output", str(path)]) == 0
    assert '"mode": "research_only"' in path.read_text(encoding="utf-8")
    assert str(path) in capsys.readouterr().out


def test_cli_broker_error_fails_closed_without_writing_report(monkeypatch, tmp_path):
    def error(**kw):
        raise research.AlpacaMarketDataError("provider unavailable")
    monkeypatch.setattr(research, "get_bars", error)
    path = tmp_path / "report.json"
    assert research.main(["--symbol", "SPY", "--output", str(path)]) == 2
    assert not path.exists()
