from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from scripts import research_alpaca_strategy_baseline as research


def _bars(count: int = 260) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=count, freq="D", tz="UTC")
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
    assert len(report["results"]) == 6
    assert "$500" in " ".join(report["limitations"])


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
        with pytest.raises((ValueError, Exception)) as error:
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
