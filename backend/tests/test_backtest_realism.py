from __future__ import annotations

import math
import statistics

import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.services import studio_backtests


def _frame(
    prices: list[float],
    *,
    opens: list[float] | None = None,
    index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    open_values = opens or prices
    timestamps = index or pd.date_range("2026-01-01", periods=len(prices), freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": open_values,
            "high": [max(o, c) + 1.0 for o, c in zip(open_values, prices)],
            "low": [min(o, c) - 1.0 for o, c in zip(open_values, prices)],
            "close": prices,
            "volume": [100.0] * len(prices),
        },
        index=timestamps,
    )


def _source(values: list[float]) -> str:
    return (
        "import pandas as pd\n\n"
        "def signals(df: pd.DataFrame) -> pd.Series:\n"
        f"    values = {values!r}\n"
        "    return pd.Series(values[:len(df)], index=df.index)\n"
    )


def _run(
    monkeypatch,
    frame: pd.DataFrame,
    signals: list[float],
    **kwargs,
) -> dict:
    def fake_fetch_data(symbol: str, timeframe: str, num_bars: int):
        return frame.copy(), float(frame["close"].iloc[-1])

    monkeypatch.setattr(studio_backtests.data_fetcher, "fetch_data", fake_fetch_data)
    return studio_backtests.run_strategy_code_backtest(
        code=_source(signals),
        symbol="TEST",
        timeframe="M5",
        num_bars=len(frame),
        validation_kind="regime",
        **kwargs,
    )


def test_signal_executes_next_bar_open_without_same_bar_lookahead(monkeypatch) -> None:
    frame = _frame(
        [100.0, 200.0, 200.0, 200.0],
        opens=[100.0, 100.0, 200.0, 200.0],
    )

    result = _run(monkeypatch, frame, [0.0, 1.0, 0.0, 0.0])

    assert result["Execution Timing"] == "Signal on bar close; execute at next bar open."
    assert result["Number of Trades"] == 1.0
    assert result["Total Return [%]"] == pytest.approx(0.0)
    assert result["Avg Trade [%]"] == pytest.approx(0.0)


def test_fee_slippage_spread_and_position_size_are_applied_to_turnover(monkeypatch) -> None:
    frame = _frame([100.0] * 5)

    result = _run(
        monkeypatch,
        frame,
        [1.0, 1.0, 0.0, 0.0, 0.0],
        fee_bps=1.0,
        slippage_bps=2.0,
        spread_bps=4.0,
        position_size_pct=50.0,
    )

    assert result["Fees [bps]"] == 1.0
    assert result["Slippage [bps]"] == 2.0
    assert result["Spread [bps]"] == 4.0
    assert result["Position Size [%]"] == 50.0
    assert result["Total Return [%]"] == pytest.approx(-0.05, abs=1e-4)
    assert result["Avg Trade [%]"] == pytest.approx(-0.05, abs=1e-4)


def test_position_size_scales_portfolio_return(monkeypatch) -> None:
    frame = _frame([100.0, 100.0, 110.0, 110.0])
    signals = [1.0, 1.0, 0.0, 0.0]

    full = _run(monkeypatch, frame, signals, position_size_pct=100.0)
    half = _run(monkeypatch, frame, signals, position_size_pct=50.0)

    assert full["Total Return [%]"] == pytest.approx(10.0)
    assert half["Total Return [%]"] == pytest.approx(5.0)


def test_short_trade_uses_linear_cfd_return_math(monkeypatch) -> None:
    frame = _frame([100.0, 100.0, 50.0, 50.0])

    result = _run(monkeypatch, frame, [-1.0, -1.0, 0.0, 0.0])

    assert result["Number of Trades"] == 1.0
    assert result["Total Return [%]"] == pytest.approx(50.0)
    assert result["Avg Trade [%]"] == pytest.approx(50.0)


def test_market_gaps_are_reported_without_synthetic_filling(monkeypatch) -> None:
    index = pd.DatetimeIndex(
        [
            "2026-01-01 00:00:00+01:00",
            "2026-01-01 00:05:00+01:00",
            "2026-01-01 00:15:00+01:00",
            "2026-01-01 00:20:00+01:00",
        ]
    )
    frame = _frame([100.0] * 4, index=index)

    result = _run(monkeypatch, frame, [0.0, 0.0, 0.0, 0.0])

    assert result["Data Timezone"] == "UTC"
    assert result["Observed Gaps"] == 1
    assert result["Largest Gap [min]"] == 10.0
    assert "never forward-filled" in result["Session Assumption"]
    assert result["Selected Bars"] == 4
    assert result["Data Start"].endswith("+00:00")


def test_daily_sharpe_uses_observed_utc_daily_equity() -> None:
    equity = pd.Series(
        [1.01, 1.00495, 1.025049],
        index=pd.date_range("2026-01-01", periods=3, freq="1D", tz="UTC"),
        dtype=float,
    )
    returns = [0.01, -0.005, 0.02]
    expected = statistics.mean(returns) / statistics.pstdev(returns) * math.sqrt(252)

    sharpe, avg_daily_pct = studio_backtests._daily_metrics(equity)

    assert sharpe == pytest.approx(expected)
    assert avg_daily_pct == pytest.approx(statistics.mean(returns) * 100.0)


def test_drawdown_and_hold_duration_follow_mark_to_market_equity(monkeypatch) -> None:
    frame = _frame([100.0, 100.0, 120.0, 90.0, 90.0])

    result = _run(monkeypatch, frame, [1.0, 1.0, 1.0, 0.0, 0.0])

    assert result["Total Return [%]"] == pytest.approx(-10.0)
    assert result["Max Drawdown [%]"] == pytest.approx(-25.0)
    assert result["Avg Hold [bars]"] == 3.0
    assert result["Avg Hold [min]"] == 15.0
    assert result["Closed Trades"] == 1.0
    assert result["Marked Open Trades"] == 0.0


def test_open_trade_is_marked_to_final_close_without_exit_cost(monkeypatch) -> None:
    frame = _frame([100.0, 100.0, 110.0])

    result = _run(
        monkeypatch,
        frame,
        [1.0, 1.0, 1.0],
        fee_bps=10.0,
        position_size_pct=100.0,
    )

    # One entry fee is paid. The still-open position is marked at the final
    # close without inventing an exit fee.
    assert result["Marked Open Trades"] == 1.0
    assert result["Closed Trades"] == 0.0
    assert result["Total Return [%]"] == pytest.approx(9.9)
    assert result["Avg Hold [bars]"] == 2.0
    assert result["Avg Hold [min]"] == 10.0


@pytest.mark.parametrize("position_size_pct", [0.0, -1.0, 100.1])
def test_position_size_must_be_positive_and_at_most_full_allocation(monkeypatch, position_size_pct: float) -> None:
    frame = _frame([100.0, 100.0, 100.0])

    with pytest.raises(HTTPException, match="position_size_pct"):
        _run(
            monkeypatch,
            frame,
            [1.0, 0.0, 0.0],
            position_size_pct=position_size_pct,
        )


def test_studio_backtest_api_forwards_realism_assumptions(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_LEGACY_CONTROLLER_ON_BOOT", "0")

    captured: dict = {}

    def fake_backtest(**kwargs):
        captured.update(kwargs)
        return {"Total Return [%]": 0.0}

    monkeypatch.setattr("backend.api.router.run_saved_strategy_backtest", fake_backtest)

    from backend.app import app

    with TestClient(app) as client:
        response = client.get(
            "/api/studio/backtest",
            params={
                "strategy": "sample",
                "symbol": "XAUUSD",
                "timeframe": "M5",
                "num_bars": 500,
                "fee_bps": 1.0,
                "slippage_bps": 2.0,
                "spread_bps": 3.0,
                "position_size_pct": 40.0,
                "validation_kind": "regime",
            },
        )

    assert response.status_code == 200
    assert captured["fee_bps"] == 1.0
    assert captured["slippage_bps"] == 2.0
    assert captured["spread_bps"] == 3.0
    assert captured["position_size_pct"] == 40.0
    assert captured["validation_kind"] == "regime"
