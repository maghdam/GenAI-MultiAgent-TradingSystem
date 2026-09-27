from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backend.services import studio_backtests


STRATEGY_SOURCE = """
import pandas as pd

def signals(df: pd.DataFrame) -> pd.Series:
    values = [0.0, 1.0, 1.0, -1.0, -1.0, 0.0, 1.0, 1.0]
    return pd.Series(values[:len(df)], index=df.index)
"""


def _bars() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=8, freq="5min", tz="UTC")
    close = [100.0, 100.0, 105.0, 110.0, 105.0, 100.0, 100.0, 90.0]
    return pd.DataFrame(
        {
            "open": close,
            "high": [value + 1.0 for value in close],
            "low": [value - 1.0 for value in close],
            "close": close,
            "volume": [100.0] * len(close),
        },
        index=index,
    )


def _mock_market_data(monkeypatch) -> None:
    bars = _bars()

    def fake_fetch_data(symbol: str, timeframe: str, num_bars: int):
        assert symbol == "TEST"
        assert timeframe == "M5"
        assert num_bars == 8
        return bars.copy(), {"source": "test"}

    monkeypatch.setattr(studio_backtests.data_fetcher, "fetch_data", fake_fetch_data)


def test_draft_backtest_computes_trade_win_rate_for_flip_exit_and_final_open(monkeypatch) -> None:
    _mock_market_data(monkeypatch)

    result = studio_backtests.run_strategy_code_backtest(
        code=STRATEGY_SOURCE,
        symbol="TEST",
        timeframe="M5",
        num_bars=8,
        strategy_name="draft_case",
        validation_kind="regime",
    )

    # t1 long -> t3 flip, t3 short -> t5 flat, t6 long -> final bar.
    assert result["Number of Trades"] == 3.0
    assert result["Win Rate [%]"] == 66.67
    assert result["Avg Hold [bars]"] == 1.67
    assert result["draft"] is True


def test_draft_and_saved_backtests_use_identical_accounting(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    _mock_market_data(monkeypatch)

    strategy_dir = Path("backend/strategies_generated")
    strategy_dir.mkdir(parents=True, exist_ok=True)
    (strategy_dir / "parity_case.py").write_text(STRATEGY_SOURCE, encoding="utf-8")

    draft = studio_backtests.run_strategy_code_backtest(
        code=STRATEGY_SOURCE,
        symbol="TEST",
        timeframe="M5",
        num_bars=8,
        fee_bps=1.5,
        slippage_bps=0.5,
        strategy_name="parity_case",
        validation_kind="regime",
    )
    saved = studio_backtests.run_saved_strategy_backtest(
        strategy="parity_case",
        symbol="TEST",
        timeframe="M5",
        num_bars=8,
        fee_bps=1.5,
        slippage_bps=0.5,
        validation_kind="regime",
    )

    shared_metric_keys = {
        "Total Return [%]",
        "Number of Trades",
        "Win Rate [%]",
        "Avg Trade [%]",
        "Max Drawdown [%]",
        "Sharpe",
        "Daily Sharpe",
        "Avg Daily Return [%]",
        "Fees [bps]",
        "Slippage [bps]",
        "Trade Sharpe",
        "SQN",
        "Trades/Day",
        "Avg Hold [bars]",
        "Validation Kind",
        "Data Start",
        "Data End",
        "Selected Bars",
        "Fetched Bars",
    }

    assert shared_metric_keys.issubset(draft)
    assert shared_metric_keys.issubset(saved)
    assert {key: draft[key] for key in shared_metric_keys} == {
        key: saved[key] for key in shared_metric_keys
    }
    assert draft["Number of Trades"] == 3.0
    assert draft["Win Rate [%]"] == pytest.approx(66.67)


def test_final_bar_entry_without_holding_period_is_not_counted(monkeypatch) -> None:
    bars = _bars()

    def fake_fetch_data(symbol: str, timeframe: str, num_bars: int):
        return bars.copy(), {"source": "test"}

    monkeypatch.setattr(studio_backtests.data_fetcher, "fetch_data", fake_fetch_data)

    source = """
import pandas as pd

def signals(df: pd.DataFrame) -> pd.Series:
    values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    return pd.Series(values[:len(df)], index=df.index)
"""

    result = studio_backtests.run_strategy_code_backtest(
        code=source,
        symbol="TEST",
        timeframe="M5",
        num_bars=8,
        strategy_name="last_bar_entry",
        validation_kind="regime",
    )

    assert result["Number of Trades"] == 0.0
    assert result["Win Rate [%]"] == 0.0
