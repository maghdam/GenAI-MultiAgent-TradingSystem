from __future__ import annotations

from pathlib import Path

import pytest

from backend import strategy as legacy_strategy
from backend.services import strategy_lifecycle as lifecycle_service
from backend.services.strategy_lifecycle import StrategyLifecycleError
from backend.storage.repositories import close_paper_position, open_paper_position
from backend.strategies.registry import get_strategy as get_trusted_strategy


SOURCE_V1 = "import pandas as pd\n\ndef signals(df: pd.DataFrame) -> pd.Series:\n    return pd.Series(1.0, index=df.index)\n"
SOURCE_V2 = SOURCE_V1.replace("1.0", "-1.0")


def _passing_metrics(trades: int = 40) -> dict:
    return {
        "Number of Trades": trades,
        "Total Return [%]": 8.0,
        "Max Drawdown [%]": -4.0,
        "Fees [bps]": 1.0,
        "Slippage [bps]": 0.5,
        "Spread [bps]": 0.5,
    }


def _development_context() -> dict:
    return {
        "symbol": "XAUUSD",
        "timeframe": "M5",
        "data_start": "2026-01-01T00:00:00+00:00",
        "data_end": "2026-01-02T00:00:00+00:00",
    }


def _advance_to_paper(strategy: str) -> object:
    lifecycle_service.record_evidence(
        strategy,
        "development_backtest",
        _passing_metrics(),
        context=_development_context(),
    )
    lifecycle_service.promote(strategy, "qa operator", "development evidence passed")
    lifecycle_service.record_evidence(
        strategy,
        "out_of_sample",
        _passing_metrics(25),
        context={
            "symbol": "XAUUSD",
            "timeframe": "M5",
            "data_start": "2026-01-02T00:05:00+00:00",
            "data_end": "2026-01-03T00:00:00+00:00",
        },
    )
    lifecycle_service.record_evidence(
        strategy,
        "regime",
        _passing_metrics(15),
        context={
            "symbol": "US30",
            "timeframe": "M5",
            "data_start": "2026-01-01T00:00:00+00:00",
            "data_end": "2026-01-02T00:00:00+00:00",
        },
    )
    lifecycle_service.promote(strategy, "qa operator", "independent validation passed")
    return lifecycle_service.promote(strategy, "qa operator", "approved for supervised paper observation")


def _close_profitable_trade(strategy: str, version_hash: str, sequence: int) -> None:
    position = open_paper_position(
        symbol=f"T{sequence:03d}",
        timeframe="M5",
        strategy=strategy,
        direction="long",
        quantity=1.0,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        lifecycle_version_hash=version_hash,
    )
    close_paper_position(position.id, 101.0, "test_exit")


def test_full_lifecycle_requires_hypothesis_evidence_and_audit_metadata(monkeypatch) -> None:
    monkeypatch.setattr(lifecycle_service, "_strategy_source", lambda strategy: SOURCE_V1)
    lifecycle_service.ensure_lifecycle("lifecycle_acceptance", SOURCE_V1)

    lifecycle_service.record_evidence(
        "lifecycle_acceptance",
        "development_backtest",
        _passing_metrics(),
        context=_development_context(),
    )
    with pytest.raises(StrategyLifecycleError, match="hypothesis"):
        lifecycle_service.promote("lifecycle_acceptance", "qa operator", "development passed")

    lifecycle_service.update_hypothesis(
        "lifecycle_acceptance",
        "The exact saved source should retain positive cost-adjusted expectancy out of sample.",
    )
    with pytest.raises(StrategyLifecycleError, match="operator name"):
        lifecycle_service.promote("lifecycle_acceptance", "", "development passed")
    with pytest.raises(StrategyLifecycleError, match="audit reason"):
        lifecycle_service.promote("lifecycle_acceptance", "qa operator", "")

    record = lifecycle_service.promote(
        "lifecycle_acceptance",
        "qa operator",
        "development passed",
    )
    assert record.stage == "backtested"

    lifecycle_service.record_evidence(
        "lifecycle_acceptance",
        "out_of_sample",
        _passing_metrics(25),
        context={
            "symbol": "XAUUSD",
            "timeframe": "M5",
            "data_start": "2026-01-02T00:05:00+00:00",
            "data_end": "2026-01-03T00:00:00+00:00",
        },
    )
    lifecycle_service.record_evidence(
        "lifecycle_acceptance",
        "regime",
        _passing_metrics(15),
        context={
            "symbol": "US30",
            "timeframe": "M5",
            "data_start": "2026-01-01T00:00:00+00:00",
            "data_end": "2026-01-02T00:00:00+00:00",
        },
    )
    record = lifecycle_service.promote(
        "lifecycle_acceptance",
        "qa operator",
        "holdout and regime evidence passed",
    )
    assert record.stage == "validated"

    record = lifecycle_service.promote(
        "lifecycle_acceptance",
        "qa operator",
        "approve supervised paper observation",
    )
    assert record.stage == "paper"

    record = lifecycle_service.record_evidence(
        "lifecycle_acceptance",
        "paper",
        _passing_metrics(24),
    )
    assert record.promotion_ready
    record = lifecycle_service.promote(
        "lifecycle_acceptance",
        "qa operator",
        "paper sample passed",
    )

    assert record.stage == "eligible"
    assert [item.to_stage for item in reversed(record.transitions)] == [
        "backtested",
        "validated",
        "paper",
        "eligible",
    ]
    assert all(item.operator.strip() for item in record.transitions)
    assert all(item.reason.strip() for item in record.transitions)


def test_promotion_remains_sequential_even_when_validation_evidence_is_precollected(monkeypatch) -> None:
    monkeypatch.setattr(lifecycle_service, "_strategy_source", lambda strategy: SOURCE_V1)
    lifecycle_service.ensure_lifecycle(
        "stage_guard",
        SOURCE_V1,
        "Promotion must advance one lifecycle stage at a time.",
    )

    lifecycle_service.record_evidence(
        "stage_guard",
        "development_backtest",
        _passing_metrics(),
        context=_development_context(),
    )
    lifecycle_service.record_evidence(
        "stage_guard",
        "out_of_sample",
        _passing_metrics(25),
        context={
            "symbol": "XAUUSD",
            "timeframe": "M5",
            "data_start": "2026-01-02T00:05:00+00:00",
            "data_end": "2026-01-03T00:00:00+00:00",
        },
    )
    lifecycle_service.record_evidence(
        "stage_guard",
        "regime",
        _passing_metrics(15),
        context={
            "symbol": "US30",
            "timeframe": "M5",
            "data_start": "2026-01-01T00:00:00+00:00",
            "data_end": "2026-01-02T00:00:00+00:00",
        },
    )

    record = lifecycle_service.promote(
        "stage_guard",
        "qa operator",
        "development gate passed",
    )
    assert record.stage == "backtested"
    assert record.next_stage == "validated"

    record = lifecycle_service.promote(
        "stage_guard",
        "qa operator",
        "independent validation gates passed",
    )
    assert record.stage == "validated"
    assert record.next_stage == "paper"

    with pytest.raises(StrategyLifecycleError, match="Paper evidence can only be collected"):
        lifecycle_service.record_paper_evidence("stage_guard")

    record = lifecycle_service.promote(
        "stage_guard",
        "qa operator",
        "approved for supervised paper observation",
    )
    assert record.stage == "paper"
    assert record.next_stage == "eligible"


def test_source_change_creates_fresh_draft_and_invalidates_old_evidence(monkeypatch) -> None:
    active_source = {"value": SOURCE_V1}
    monkeypatch.setattr(lifecycle_service, "_strategy_source", lambda strategy: active_source["value"])

    first = lifecycle_service.ensure_lifecycle(
        "version_guard",
        SOURCE_V1,
        "The saved source should be revalidated after any source change.",
    )
    lifecycle_service.record_evidence(
        "version_guard",
        "development_backtest",
        _passing_metrics(),
        context=_development_context(),
    )
    first = lifecycle_service.promote(
        "version_guard",
        "qa operator",
        "version one development passed",
    )
    assert first.stage == "backtested"

    active_source["value"] = SOURCE_V2
    second = lifecycle_service.ensure_lifecycle("version_guard", SOURCE_V2)
    old = lifecycle_service.get_lifecycle("version_guard", first.version_hash)

    assert second.version == first.version + 1
    assert second.stage == "draft"
    assert second.evidence == []
    assert second.transitions == []
    assert old.stage == "backtested"
    assert len(old.evidence) == 1
    assert old.current_source is False

    allowed, details, reason = lifecycle_service.paper_execution_gate("version_guard")
    assert allowed is False
    assert details["version_hash"] == second.version_hash
    assert details["stage"] == "draft"
    assert "not approved" in str(reason)


def test_paper_evidence_counts_only_closed_trades_from_exact_source_version(monkeypatch) -> None:
    active_source = {"value": SOURCE_V1}
    monkeypatch.setattr(lifecycle_service, "_strategy_source", lambda strategy: active_source["value"])

    first = lifecycle_service.ensure_lifecycle(
        "paper_version_guard",
        SOURCE_V1,
        "Paper evidence should be tied to the exact saved source version.",
    )
    first = _advance_to_paper("paper_version_guard")
    assert first.stage == "paper"

    for index in range(20):
        _close_profitable_trade("paper_version_guard", first.version_hash, index)

    active_source["value"] = SOURCE_V2
    second = lifecycle_service.ensure_lifecycle("paper_version_guard", SOURCE_V2)
    second = _advance_to_paper("paper_version_guard")
    assert second.stage == "paper"
    assert second.version_hash != first.version_hash

    _close_profitable_trade("paper_version_guard", second.version_hash, 100)
    second = lifecycle_service.record_paper_evidence("paper_version_guard")

    assert second.evidence[0].evidence_type == "paper"
    assert second.evidence[0].metrics["Number of Trades"] == 1
    assert second.evidence[0].context["lifecycle_version_hash"] == second.version_hash
    assert second.evidence[0].passed is False
    assert second.promotion_ready is False

    for index in range(101, 120):
        _close_profitable_trade("paper_version_guard", second.version_hash, index)

    second = lifecycle_service.record_paper_evidence("paper_version_guard")
    assert second.evidence[0].metrics["Number of Trades"] == 20
    assert second.evidence[0].passed is True
    assert second.promotion_ready is True


def test_generated_research_file_never_enters_trusted_runtime_registry(monkeypatch, tmp_path: Path) -> None:
    generated = tmp_path / "generated_candidate.py"
    generated.write_text(SOURCE_V1, encoding="utf-8")
    before = legacy_strategy.available_strategies()

    loaded = legacy_strategy.load_generated_strategies(tmp_path)

    assert loaded == 0
    assert legacy_strategy.available_strategies() == before
    with pytest.raises(KeyError, match="Unknown V2 strategy"):
        get_trusted_strategy("generated_candidate")

    monkeypatch.setattr(lifecycle_service, "_strategy_source", lambda strategy: SOURCE_V1)
    lifecycle_service.ensure_lifecycle(
        "generated_candidate",
        SOURCE_V1,
        "Generated research code must not bypass lifecycle governance.",
    )
    allowed, details, reason = lifecycle_service.paper_execution_gate("generated_candidate")
    assert allowed is False
    assert details["governed"] is True
    assert details["stage"] == "draft"
    assert "not approved" in str(reason)
