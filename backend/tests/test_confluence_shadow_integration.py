from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pandas as pd

from backend.domain.models import ConfluenceShadowRecord, EngineConfig, StrategyAnalysis, WatchlistItem
from backend.services import engine as engine_module
from backend.services.engine import V2Engine
from backend.services.execution_engine import ExecutionResult


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        [{"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0}],
        index=pd.to_datetime(["2026-09-30T14:00:00Z"], utc=True),
    )


def _analysis() -> StrategyAnalysis:
    return StrategyAnalysis(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        signal="long",
        confidence=0.65,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        reasons=["forward observation test"],
        context={},
        created_at=datetime(2026, 9, 30, 14, 0, tzinfo=UTC),
    )


def _config() -> EngineConfig:
    return EngineConfig(
        enabled=True,
        paper_autotrade=True,
        kill_switch=False,
        min_confidence=0.60,
    )


def _watch_item() -> WatchlistItem:
    return WatchlistItem(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        enabled=True,
        params={},
    )


class _Strategy:
    def __init__(self, analysis: StrategyAnalysis) -> None:
        self.analysis = analysis

    def analyze(self, **kwargs) -> StrategyAnalysis:
        return self.analysis


def test_forward_scan_observes_shadow_but_executes_original_analysis(monkeypatch) -> None:
    analysis = _analysis()
    captured: dict[str, object] = {}

    monkeypatch.setattr(engine_module, "get_bars", lambda *args, **kwargs: _bars())
    monkeypatch.setattr(engine_module, "get_strategy", lambda name: _Strategy(analysis))
    monkeypatch.setattr(engine_module, "add_analysis", lambda value: value)

    def _record_shadow(value: StrategyAnalysis, min_confidence: float) -> ConfluenceShadowRecord:
        captured["shadow_input"] = value
        captured["shadow_min_confidence"] = min_confidence
        return ConfluenceShadowRecord(
            id=1,
            created_at=value.created_at,
            analysis_created_at=value.created_at,
            symbol=value.symbol,
            timeframe=value.timeframe,
            strategy=value.strategy,
            original_signal=value.signal,
            original_confidence=value.confidence,
            shadow_signal=value.signal,
            shadow_confidence=0.77,
            confidence_adjustment=0.12,
            action="confirm",
            target_horizon="30m",
            event_score=1.0,
            eligible_event_count=1,
            event_ids=[7],
            original_would_pass=True,
            shadow_would_pass=True,
            rationale="test shadow confirmation",
            evidence={
                "promotion_policy": "manual_review_only",
                "automatic_promotion": False,
                "execution_source": "original_strategy_analysis",
            },
            execution_unchanged=True,
        )

    def _execute(**kwargs) -> ExecutionResult:
        captured["execution_analysis"] = kwargs["analysis"]
        captured["execution_confidence"] = kwargs["analysis"].confidence
        captured["execution_context"] = dict(kwargs["analysis"].context)
        return ExecutionResult(
            action_taken=False,
            intent_id=None,
            status="rejected",
            summary="test execution",
            retryable=False,
        )

    monkeypatch.setattr(engine_module, "record_confluence_shadow", _record_shadow)
    monkeypatch.setattr(engine_module, "execute_paper_signal", _execute)

    bar_state: dict[str, int] = {}
    processed, acted = asyncio.run(V2Engine()._scan_item(_config(), _watch_item(), bar_state))

    assert processed is True
    assert acted is False
    assert captured["shadow_input"] is analysis
    assert captured["shadow_min_confidence"] == 0.60
    assert captured["execution_analysis"] is analysis
    assert captured["execution_confidence"] == 0.65
    assert captured["execution_context"]["engine_source"] == "auto_loop"
    assert analysis.confidence == 0.65
    assert bar_state["XAUUSD|M5"] == int(_bars().index[-1].timestamp())


def test_shadow_failure_cannot_block_or_modify_original_execution(monkeypatch) -> None:
    analysis = _analysis()
    captured: dict[str, object] = {}
    incidents: list[tuple] = []

    monkeypatch.setattr(engine_module, "get_bars", lambda *args, **kwargs: _bars())
    monkeypatch.setattr(engine_module, "get_strategy", lambda name: _Strategy(analysis))
    monkeypatch.setattr(engine_module, "add_analysis", lambda value: value)

    def _shadow_failure(*args, **kwargs):
        raise RuntimeError("shadow subsystem unavailable")

    def _execute(**kwargs) -> ExecutionResult:
        captured["execution_analysis"] = kwargs["analysis"]
        captured["execution_confidence"] = kwargs["analysis"].confidence
        return ExecutionResult(
            action_taken=True,
            intent_id=10,
            status="executed",
            summary="original path executed",
            position_id=20,
            retryable=False,
        )

    monkeypatch.setattr(engine_module, "record_confluence_shadow", _shadow_failure)
    monkeypatch.setattr(engine_module, "execute_paper_signal", _execute)
    monkeypatch.setattr(
        engine_module,
        "log_incident",
        lambda *args, **kwargs: incidents.append((args, kwargs)),
    )

    processed, acted = asyncio.run(V2Engine()._scan_item(_config(), _watch_item(), {}))

    assert processed is True
    assert acted is True
    assert captured["execution_analysis"] is analysis
    assert captured["execution_confidence"] == 0.65
    assert len(incidents) == 1
    args, _ = incidents[0]
    assert args[1] == "confluence_shadow_failed"
    assert args[3]["execution_unchanged"] is True
