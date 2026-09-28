from __future__ import annotations

import pandas as pd
import pytest
from fastapi import HTTPException

from backend.services import studio_backtests


PARAMETERIZED_SOURCE = """
import pandas as pd

def signals(df: pd.DataFrame, direction: int = 1) -> pd.Series:
    return pd.Series(float(direction), index=df.index)
"""

FLAT_SOURCE = """
import pandas as pd

def signals(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=df.index)
"""


def _frame(prices: list[float]) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=len(prices), freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": prices,
            "high": [value + 0.5 for value in prices],
            "low": [max(0.01, value - 0.5) for value in prices],
            "close": prices,
            "volume": [100.0] * len(prices),
        },
        index=index,
    )


def test_minimum_holdout_sample_is_enforced() -> None:
    frame = _frame([100.0] * 150)

    with pytest.raises(HTTPException, match="minimum is 50"):
        studio_backtests._validation_window(frame, "out_of_sample")


def test_walk_forward_uses_expanding_train_and_strictly_forward_test_folds(monkeypatch) -> None:
    frame = _frame([100.0] * 300)

    def fake_fetch_data(symbol: str, timeframe: str, num_bars: int):
        return frame.copy(), 100.0

    monkeypatch.setattr(studio_backtests.data_fetcher, "fetch_data", fake_fetch_data)

    result = studio_backtests.run_strategy_code_backtest(
        code=FLAT_SOURCE,
        symbol="TEST",
        timeframe="M5",
        num_bars=300,
        validation_kind="walk_forward",
        walk_forward_folds=3,
    )

    folds = result["Fold Results"]
    assert result["Validation Kind"] == "walk_forward"
    assert result["Walk Forward Folds"] == 3
    assert [item["Train Bars"] for item in folds] == [150, 200, 250]
    assert [item["Test Bars"] for item in folds] == [50, 50, 50]

    for item in folds:
        assert pd.Timestamp(item["Train End"]) < pd.Timestamp(item["Test Start"])

    assert pd.Timestamp(folds[0]["Test End"]) < pd.Timestamp(folds[1]["Test Start"])
    assert pd.Timestamp(folds[1]["Test End"]) < pd.Timestamp(folds[2]["Test Start"])


def test_walk_forward_minimum_fold_samples_are_enforced(monkeypatch) -> None:
    frame = _frame([100.0] * 180)

    def fake_fetch_data(symbol: str, timeframe: str, num_bars: int):
        return frame.copy(), 100.0

    monkeypatch.setattr(studio_backtests.data_fetcher, "fetch_data", fake_fetch_data)

    with pytest.raises(HTTPException, match="at least 190 bars"):
        studio_backtests.run_strategy_code_backtest(
            code=FLAT_SOURCE,
            symbol="TEST",
            timeframe="M5",
            num_bars=180,
            validation_kind="walk_forward",
            walk_forward_folds=3,
        )


def test_parameter_optimization_never_uses_holdout_for_selection(monkeypatch) -> None:
    development = [100.0 + (100.0 * i / 139.0) for i in range(140)]
    holdout = [200.0 - (120.0 * i / 59.0) for i in range(60)]
    frame = _frame(development + holdout)

    def fake_fetch_data(symbol: str, timeframe: str, num_bars: int):
        return frame.copy(), float(frame["close"].iloc[-1])

    monkeypatch.setattr(studio_backtests.data_fetcher, "fetch_data", fake_fetch_data)

    result = studio_backtests.optimize_strategy_source(
        source=PARAMETERIZED_SOURCE,
        strategy="direction_test",
        symbol="TEST",
        timeframe="M5",
        num_bars=200,
        param_grid={"direction": [1, -1]},
        objective="return",
    )

    assert result["Optimization Dataset"] == "development_backtest (first 70%) only"
    assert result["Holdout Used For Selection"] is False
    assert result["Optimization Combinations"] == 2
    assert result["Best Parameters"] == {"direction": 1}
    assert result["development_result"]["Total Return [%]"] > 0
    assert result["holdout_result"]["Total Return [%]"] < 0
    assert result["development_result"]["Data End"] < result["holdout_result"]["Data Start"]


def test_parameter_optimization_grid_is_bounded(monkeypatch) -> None:
    frame = _frame([100.0] * 200)

    def fake_fetch_data(symbol: str, timeframe: str, num_bars: int):
        return frame.copy(), 100.0

    monkeypatch.setattr(studio_backtests.data_fetcher, "fetch_data", fake_fetch_data)

    with pytest.raises(HTTPException, match="maximum is 100"):
        studio_backtests.optimize_strategy_source(
            source=PARAMETERIZED_SOURCE,
            strategy="too_many",
            symbol="TEST",
            timeframe="M5",
            num_bars=200,
            param_grid={"direction": list(range(101))},
        )


def test_regime_window_is_operator_supplied_alternate_context() -> None:
    frame = _frame([100.0] * 200)

    regime, kind, metadata = studio_backtests._validation_window(frame, "regime")

    assert kind == "regime"
    assert len(regime) == 200
    assert metadata["Minimum Bars Required"] == 100
    assert "Alternate market or non-overlapping period" in metadata["Validation Selection"]
