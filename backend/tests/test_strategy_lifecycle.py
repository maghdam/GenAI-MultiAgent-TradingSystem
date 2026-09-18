from __future__ import annotations

import pandas as pd
import pytest

from backend.services import strategy_lifecycle as lifecycle_service
from backend.services.strategy_lifecycle import StrategyLifecycleError
from backend.services.studio_backtests import _validation_window


SOURCE_V1 = "import pandas as pd\n\ndef signals(df: pd.DataFrame) -> pd.Series:\n    return pd.Series(0.0, index=df.index)\n"
SOURCE_V2 = SOURCE_V1.replace("0.0", "1.0")


def passing_metrics(trades: int = 40) -> dict:
    return {
        "Number of Trades": trades,
        "Total Return [%]": 8.5,
        "Max Drawdown [%]": -4.0,
        "Fees [bps]": 1.0,
        "Slippage [bps]": 0.5,
    }


def test_lifecycle_creates_immutable_source_versions(monkeypatch) -> None:
    active_source = {"value": SOURCE_V1}
    monkeypatch.setattr(lifecycle_service, "_strategy_source", lambda strategy: active_source["value"])

    first = lifecycle_service.ensure_lifecycle("gold_test", SOURCE_V1, "Trend continuation should persist after a pullback.")
    same = lifecycle_service.ensure_lifecycle("gold_test", SOURCE_V1)
    active_source["value"] = SOURCE_V2
    second = lifecycle_service.ensure_lifecycle("gold_test", SOURCE_V2)

    assert first.id == same.id
    assert first.version == 1
    assert second.version == 2
    assert second.version_hash != first.version_hash
    assert second.stage == "draft"
    assert second.hypothesis == first.hypothesis


def test_lifecycle_enforces_sequential_evidence_gates(monkeypatch) -> None:
    monkeypatch.setattr(lifecycle_service, "_strategy_source", lambda strategy: SOURCE_V1)
    record = lifecycle_service.ensure_lifecycle(
        "gated_strategy",
        SOURCE_V1,
        "A cost-adjusted trend signal should have positive expectancy across holdout and alternate regimes.",
    )
    assert record.stage == "draft"
    allowed, details, reason = lifecycle_service.paper_execution_gate("gated_strategy")
    assert not allowed and details["stage"] == "draft"
    assert "not approved" in str(reason)
    assert not record.promotion_ready

    record = lifecycle_service.record_evidence("gated_strategy", "development_backtest", passing_metrics())
    assert record.promotion_ready
    record = lifecycle_service.promote("gated_strategy", "test operator", "development gate passed")
    assert record.stage == "backtested"

    with pytest.raises(StrategyLifecycleError, match="holdout"):
        lifecycle_service.promote("gated_strategy", "test operator")

    lifecycle_service.record_evidence("gated_strategy", "out_of_sample", passing_metrics(25))
    record = lifecycle_service.record_evidence("gated_strategy", "regime", passing_metrics(15))
    assert record.promotion_ready
    record = lifecycle_service.promote("gated_strategy", "test operator", "independent validation passed")
    assert record.stage == "validated"

    record = lifecycle_service.promote("gated_strategy", "test operator", "approved for supervised paper observation")
    assert record.stage == "paper"
    assert not record.promotion_ready
    allowed, details, reason = lifecycle_service.paper_execution_gate("gated_strategy")
    assert allowed and details["stage"] == "paper"
    assert reason is None

    record = lifecycle_service.record_evidence("gated_strategy", "paper", passing_metrics(24))
    assert record.promotion_ready
    record = lifecycle_service.promote("gated_strategy", "test operator", "paper sample passed")
    assert record.stage == "eligible"
    assert len(record.transitions) == 4


def test_zero_cost_backtest_cannot_pass_gate(monkeypatch) -> None:
    monkeypatch.setattr(lifecycle_service, "_strategy_source", lambda strategy: SOURCE_V1)
    lifecycle_service.ensure_lifecycle("zero_cost", SOURCE_V1, "Test cost gate.")
    metrics = passing_metrics()
    metrics["Fees [bps]"] = 0
    metrics["Slippage [bps]"] = 0

    record = lifecycle_service.record_evidence("zero_cost", "development_backtest", metrics)

    assert record.evidence[0].passed is False
    assert "modeled costs 0.00 bps" in record.evidence[0].summary
    assert not record.promotion_ready


def test_validation_windows_are_chronological() -> None:
    frame = pd.DataFrame({"close": range(200)}, index=pd.date_range("2026-01-01", periods=200, freq="5min"))

    development, development_kind = _validation_window(frame, "development_backtest")
    holdout, holdout_kind = _validation_window(frame, "out_of_sample")
    regime, regime_kind = _validation_window(frame, "regime")

    assert development_kind == "development_backtest"
    assert holdout_kind == "out_of_sample"
    assert regime_kind == "regime"
    assert len(development) == 140
    assert len(holdout) == 60
    assert development.index.max() < holdout.index.min()
    assert len(regime) == 200
