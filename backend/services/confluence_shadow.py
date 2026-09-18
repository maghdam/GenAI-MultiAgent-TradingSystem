from __future__ import annotations

from datetime import UTC, datetime, timedelta

from backend.domain.models import ConfluenceShadowRecord, EventCalibrationGroup, MarketEventRecord, StrategyAnalysis
from backend.services.event_calibration import build_event_calibration
from backend.storage.repositories import create_confluence_shadow, list_market_events


_TIMEFRAME_HORIZON = {
    "M1": "5m",
    "M5": "30m",
    "M15": "4h",
    "M30": "4h",
    "H1": "1d",
    "H4": "1d",
    "D1": "1d",
}
_EVENT_MAX_AGE = {
    "5m": timedelta(minutes=30),
    "30m": timedelta(hours=2),
    "4h": timedelta(hours=12),
    "1d": timedelta(days=3),
}
_IMPACT_WEIGHT = {"high": 1.0, "medium": 0.65, "low": 0.35, "unknown": 0.25}
_NASDAQ_ALIASES = {"NAS100", "US100", "USTEC", "NDX", "USTECH"}


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def canonical_event_symbol(symbol: str) -> str:
    upper = symbol.upper()
    return "NAS100" if upper in _NASDAQ_ALIASES else upper


def target_horizon(timeframe: str) -> str:
    return _TIMEFRAME_HORIZON.get(timeframe.upper(), "4h")


def _quality(group: EventCalibrationGroup) -> float:
    hit_rate = float(group.hit_rate or 0.5)
    brier = float(group.average_brier_score if group.average_brier_score is not None else 0.25)
    return max(0.5, min(1.0, 0.5 + (hit_rate - 0.55) * 2.0 + (0.24 - brier)))


def _eligible_events(
    analysis: StrategyAnalysis,
    groups: list[EventCalibrationGroup],
    events: list[MarketEventRecord],
    now: datetime,
) -> tuple[str, list[tuple[MarketEventRecord, EventCalibrationGroup, float]]]:
    horizon = target_horizon(analysis.timeframe)
    eligible_groups = {
        (group.event_type, group.horizon): group
        for group in groups
        if group.gate == "eligible" and group.horizon == horizon
    }
    canonical = canonical_event_symbol(analysis.symbol)
    maximum_age = _EVENT_MAX_AGE[horizon]
    selected: list[tuple[MarketEventRecord, EventCalibrationGroup, float]] = []
    for event in events:
        if canonical not in {canonical_event_symbol(symbol) for symbol in event.symbols}:
            continue
        group = eligible_groups.get((event.event_type, horizon))
        if not group:
            continue
        occurred_at = _naive_utc(event.published_at or event.ingested_at)
        age = now - occurred_at
        if age < timedelta(0) or age > maximum_age:
            continue
        recency_weight = max(0.25, 1.0 - age.total_seconds() / maximum_age.total_seconds())
        selected.append((event, group, recency_weight))
    return horizon, selected


def build_confluence_shadow(
    analysis: StrategyAnalysis,
    min_confidence: float,
    *,
    groups: list[EventCalibrationGroup] | None = None,
    events: list[MarketEventRecord] | None = None,
    now: datetime | None = None,
) -> ConfluenceShadowRecord:
    evaluated_at = now or _now()
    calibration_groups = groups if groups is not None else build_event_calibration().groups
    market_events = events if events is not None else list_market_events(200)
    horizon, selected = _eligible_events(analysis, calibration_groups, market_events, evaluated_at)
    original_confidence = max(0.0, min(1.0, float(analysis.confidence)))
    evidence_rows: list[dict[str, object]] = []
    weighted_sum = 0.0
    weight_sum = 0.0
    for event, group, recency_weight in selected:
        weight = (
            float(event.credibility_score)
            * _IMPACT_WEIGHT.get(event.impact, 0.25)
            * _quality(group)
            * recency_weight
        )
        contribution = float(event.sentiment_score) * weight
        weighted_sum += contribution
        weight_sum += weight
        evidence_rows.append(
            {
                "event_id": event.id,
                "title": event.title,
                "event_type": event.event_type,
                "sentiment": event.sentiment,
                "sentiment_score": event.sentiment_score,
                "credibility_score": event.credibility_score,
                "impact": event.impact,
                "calibration_samples": group.samples,
                "calibration_hit_rate": group.hit_rate,
                "calibration_brier_score": group.average_brier_score,
                "recency_weight": recency_weight,
                "contribution": contribution,
            }
        )
    event_score = max(-1.0, min(1.0, weighted_sum / weight_sum)) if weight_sum else 0.0

    adjustment = 0.0
    if not selected:
        action = "insufficient_evidence"
        rationale = f"No eligible {horizon} event evidence is available for {canonical_event_symbol(analysis.symbol)}."
    elif analysis.signal == "no_trade":
        action = "context_only"
        rationale = f"Eligible event score is {event_score:+.2f}, but shadow mode never creates a trade from no_trade."
    else:
        signal_sign = 1.0 if analysis.signal == "long" else -1.0
        alignment = signal_sign * event_score
        if alignment >= 0.10:
            adjustment = min(0.12, abs(event_score) * 0.12)
            action = "confirm"
            rationale = f"Eligible event evidence confirms the {analysis.signal} signal with score {event_score:+.2f}."
        elif alignment <= -0.10:
            adjustment = -min(0.20, abs(event_score) * 0.20)
            action = "reduce"
            rationale = f"Eligible event evidence conflicts with the {analysis.signal} signal with score {event_score:+.2f}."
        else:
            action = "neutral"
            rationale = f"Eligible event evidence is balanced with score {event_score:+.2f}."

    shadow_confidence = max(0.0, min(1.0, original_confidence + adjustment))
    return ConfluenceShadowRecord(
        id=0,
        created_at=evaluated_at,
        analysis_created_at=analysis.created_at,
        symbol=analysis.symbol.upper(),
        timeframe=analysis.timeframe.upper(),
        strategy=analysis.strategy,
        original_signal=analysis.signal,
        original_confidence=original_confidence,
        shadow_signal=analysis.signal,
        shadow_confidence=shadow_confidence,
        confidence_adjustment=adjustment,
        action=action,  # type: ignore[arg-type]
        target_horizon=horizon,  # type: ignore[arg-type]
        event_score=event_score,
        eligible_event_count=len(selected),
        event_ids=[event.id for event, _, _ in selected],
        original_would_pass=analysis.signal != "no_trade" and original_confidence >= min_confidence,
        shadow_would_pass=analysis.signal != "no_trade" and shadow_confidence >= min_confidence,
        rationale=rationale,
        evidence={
            "canonical_symbol": canonical_event_symbol(analysis.symbol),
            "minimum_confidence": min_confidence,
            "events": evidence_rows,
            "bounded_adjustment": {"confirm_max": 0.12, "conflict_max": -0.20},
        },
        execution_unchanged=True,
    )


def record_confluence_shadow(analysis: StrategyAnalysis, min_confidence: float) -> ConfluenceShadowRecord:
    return create_confluence_shadow(build_confluence_shadow(analysis, min_confidence))

