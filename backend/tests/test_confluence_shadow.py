from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from backend.domain.models import EventCalibrationGroup, MarketEventRecord, StrategyAnalysis
from backend.services.confluence_shadow import build_confluence_shadow, canonical_event_symbol, target_horizon
from backend.storage.repositories import create_confluence_shadow, list_confluence_shadows


NOW = datetime(2026, 7, 19, 12, 0, 0)


def _analysis(signal: str = "long", confidence: float = 0.65) -> StrategyAnalysis:
    return StrategyAnalysis(
        symbol="US100",
        timeframe="M5",
        strategy="smc",
        signal=signal,
        confidence=confidence,
        created_at=NOW,
    )


def _group(gate: str = "eligible") -> EventCalibrationGroup:
    return EventCalibrationGroup(
        event_type="earnings",
        horizon="30m",
        samples=45,
        hit_rate=0.64,
        average_brier_score=0.19,
        gate=gate,
        gate_reason="test calibration",
    )


def _event(sentiment_score: float) -> MarketEventRecord:
    return MarketEventRecord(
        id=7,
        content_hash="event-7",
        source="test",
        title="Nasdaq driver",
        published_at=NOW - timedelta(minutes=10),
        ingested_at=NOW - timedelta(minutes=9),
        symbols=["NAS100"],
        event_type="earnings",
        sentiment="bullish" if sentiment_score > 0 else "bearish",
        sentiment_score=sentiment_score,
        impact="high",
        credibility_score=0.9,
    )


def test_symbol_aliases_and_timeframes_are_canonical() -> None:
    assert canonical_event_symbol("ustec") == "NAS100"
    assert canonical_event_symbol("NDX") == "NAS100"
    assert target_horizon("M1") == "5m"
    assert target_horizon("M5") == "30m"
    assert target_horizon("H1") == "1d"


def test_shadow_requires_calibrated_eligible_evidence() -> None:
    shadow = build_confluence_shadow(
        _analysis(), 0.6, groups=[_group("observe")], events=[_event(0.8)], now=NOW
    )

    assert shadow.action == "insufficient_evidence"
    assert shadow.shadow_confidence == shadow.original_confidence
    assert shadow.execution_unchanged is True


def test_aligned_event_evidence_confirms_but_is_bounded() -> None:
    analysis = _analysis(confidence=0.65)
    shadow = build_confluence_shadow(analysis, 0.6, groups=[_group()], events=[_event(1.0)], now=NOW)

    assert shadow.action == "confirm"
    assert shadow.confidence_adjustment == pytest.approx(0.12)
    assert shadow.shadow_confidence == pytest.approx(0.77)
    assert analysis.confidence == 0.65
    assert shadow.execution_unchanged is True


def test_conflicting_evidence_can_change_shadow_threshold_only() -> None:
    shadow = build_confluence_shadow(
        _analysis(confidence=0.65), 0.6, groups=[_group()], events=[_event(-1.0)], now=NOW
    )

    assert shadow.action == "reduce"
    assert shadow.original_would_pass is True
    assert shadow.shadow_would_pass is False
    assert shadow.shadow_confidence == pytest.approx(0.45)
    assert shadow.execution_unchanged is True


def test_no_trade_cannot_be_promoted_by_events() -> None:
    shadow = build_confluence_shadow(
        _analysis(signal="no_trade", confidence=0.2), 0.6, groups=[_group()], events=[_event(1.0)], now=NOW
    )

    assert shadow.action == "context_only"
    assert shadow.shadow_signal == "no_trade"
    assert shadow.confidence_adjustment == 0.0


def test_shadow_record_round_trips_through_repository() -> None:
    saved = create_confluence_shadow(
        build_confluence_shadow(_analysis(), 0.6, groups=[_group()], events=[_event(0.8)], now=NOW)
    )

    records = list_confluence_shadows(symbol="US100")
    assert records[0].id == saved.id
    assert records[0].event_ids == [7]
    assert records[0].execution_unchanged is True
