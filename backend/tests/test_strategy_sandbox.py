from __future__ import annotations

import pandas as pd
import pytest

from backend.services.strategy_policy import StrategyPolicyError, validate_strategy_source
from backend.services.strategy_sandbox import StrategySandboxError, run_strategy_source


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 99.0],
            "high": [102.0, 102.0, 101.0],
            "low": [99.0, 98.0, 98.0],
            "close": [101.0, 99.0, 100.0],
            "volume": [10.0, 11.0, 12.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True),
    )


def test_generated_strategy_is_validated_without_executing_top_level_code():
    source = """
import pandas as pd

raise RuntimeError('must not run during validation')

def signals(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=df.index)
"""
    assert "must not run" in validate_strategy_source(source)


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef signals(df): return df.close",
        "def signals(df):\n    open('secret.txt').read()\n    return df.close",
        "def signals(df):\n    return pd.read_csv('secret.csv')",
        "def signals(df):\n    return df.__class__",
    ],
)
def test_generated_strategy_policy_rejects_dangerous_capabilities(source: str):
    with pytest.raises(StrategyPolicyError):
        validate_strategy_source(source)


def test_strategy_runs_in_isolated_process_and_returns_aligned_series():
    source = """
import pandas as pd
import numpy as np

def signals(df: pd.DataFrame, period=2) -> pd.Series:
    average = df['close'].rolling(period).mean()
    return pd.Series(np.where(df['close'] >= average, 1.0, -1.0), index=df.index)
"""
    result = run_strategy_source(source, _bars(), {"period": 2})
    assert list(result.index) == list(_bars().index)
    assert result.tolist() == [-1.0, -1.0, 1.0]


def test_strategy_sandbox_enforces_timeout():
    source = """
import pandas as pd

def signals(df: pd.DataFrame) -> pd.Series:
    while True:
        pass
"""
    with pytest.raises(StrategySandboxError, match="execution limit"):
        run_strategy_source(source, _bars(), timeout_seconds=0.5)
