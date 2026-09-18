from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta
import os
from statistics import fmean, median

import pandas as pd

from backend.domain.models import (
    EventCalibrationGroup,
    EventCalibrationResponse,
    EventCalibrationRunResponse,
    EventOutcomeRecord,
    MarketEventRecord,
)
from backend.services.market_data import MarketDataError, get_bars
from backend.services.broker import list_symbols
from backend.storage.repositories import list_event_outcomes, list_market_events, upsert_event_outcome


_HORIZONS: dict[str, tuple[timedelta, float]] = {
    "5m": (timedelta(minutes=5), 0.05),
    "30m": (timedelta(minutes=30), 0.10),
    "4h": (timedelta(hours=4), 0.25),
    "1d": (timedelta(days=1), 0.50),
}

_MARKET_SYMBOL_ALIASES: dict[str, tuple[str, ...]] = {
    "NAS100": ("NAS100", "US100", "USTEC", "NDX", "USTECH"),
    "US500": ("US500", "SPX500", "SP500", "SPX"),
    "US30": ("US30", "DJ30", "WS30", "DJI"),
    "BTCUSD": ("BTCUSD", "BTC/USD"),
}


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _utc_timestamp(value: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize(UTC) if ts.tzinfo is None else ts.tz_convert(UTC)


def _empty_outcome(
    event: MarketEventRecord,
    symbol: str,
    horizon: str,
    status: str,
    reason: str,
    market_symbol: str | None = None,
) -> EventOutcomeRecord:
    return EventOutcomeRecord(
        id=0,
        event_id=event.id,
        symbol=symbol,
        market_symbol=market_symbol or symbol,
        horizon=horizon,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        predicted_direction=event.sentiment,
        realized_direction="unknown",
        threshold_pct=_HORIZONS[horizon][1],
        reason=reason,
        computed_at=_now(),
    )


def evaluate_event_against_bars(
    event: MarketEventRecord,
    symbol: str,
    horizon: str,
    bars: pd.DataFrame,
    *,
    market_symbol: str | None = None,
) -> EventOutcomeRecord:
    if horizon not in _HORIZONS:
        raise ValueError(f"Unsupported event horizon: {horizon}")
    if bars is None or bars.empty:
        return _empty_outcome(event, symbol, horizon, "unavailable", "No market bars are available.", market_symbol)

    frame = bars.copy().sort_index()
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame[~frame.index.isna()]
    required = {"close", "high", "low"}
    if frame.empty or not required.issubset(frame.columns):
        return _empty_outcome(event, symbol, horizon, "unavailable", "Market bars are incomplete.", market_symbol)

    event_at = _utc_timestamp(event.published_at or event.ingested_at)
    reference_candidates = frame.loc[frame.index >= event_at]
    if reference_candidates.empty:
        latest = frame.index[-1]
        status = "pending" if latest < event_at else "unavailable"
        return _empty_outcome(event, symbol, horizon, status, "No closed reference bar exists after the event.", market_symbol)

    reference_at = reference_candidates.index[0]
    reference_price = float(reference_candidates.iloc[0]["close"])
    if reference_price <= 0:
        return _empty_outcome(event, symbol, horizon, "unavailable", "Reference price is invalid.", market_symbol)

    duration, threshold_pct = _HORIZONS[horizon]
    desired_target_at = reference_at + duration
    target_candidates = frame.loc[frame.index >= desired_target_at]
    if target_candidates.empty:
        outcome = _empty_outcome(event, symbol, horizon, "pending", "Target horizon has not closed yet.", market_symbol)
        outcome.reference_at = reference_at.to_pydatetime()
        outcome.reference_price = reference_price
        return outcome

    target_at = target_candidates.index[0]
    target_price = float(target_candidates.iloc[0]["close"])
    window = frame.loc[(frame.index >= reference_at) & (frame.index <= target_at)]
    forward_return = (target_price / reference_price - 1.0) * 100.0
    high_excursion = (float(window["high"].max()) / reference_price - 1.0) * 100.0
    low_excursion = (float(window["low"].min()) / reference_price - 1.0) * 100.0

    prediction_sign = 1 if event.sentiment == "bullish" else -1 if event.sentiment == "bearish" else 0
    if event.sentiment == "mixed":
        prediction_sign = 1 if event.sentiment_score > 0 else -1 if event.sentiment_score < 0 else 0
    if prediction_sign > 0:
        favorable = max(0.0, high_excursion)
        adverse = min(0.0, low_excursion)
    elif prediction_sign < 0:
        favorable = max(0.0, -low_excursion)
        adverse = min(0.0, -high_excursion)
    else:
        favorable = max(abs(high_excursion), abs(low_excursion))
        adverse = 0.0

    if forward_return > threshold_pct:
        realized_direction = "up"
    elif forward_return < -threshold_pct:
        realized_direction = "down"
    else:
        realized_direction = "flat"

    if prediction_sign > 0:
        hit = realized_direction == "up"
    elif prediction_sign < 0:
        hit = realized_direction == "down"
    else:
        hit = realized_direction == "flat"
    probability_up = max(0.0, min(1.0, (float(event.sentiment_score) + 1.0) / 2.0))
    actual_up = 1.0 if forward_return > 0 else 0.0
    brier = (probability_up - actual_up) ** 2

    return EventOutcomeRecord(
        id=0,
        event_id=event.id,
        symbol=symbol,
        market_symbol=market_symbol or symbol,
        horizon=horizon,  # type: ignore[arg-type]
        status="evaluated",
        reference_at=reference_at.to_pydatetime(),
        target_at=target_at.to_pydatetime(),
        reference_price=reference_price,
        target_price=target_price,
        forward_return_pct=forward_return,
        max_favorable_excursion_pct=favorable,
        max_adverse_excursion_pct=adverse,
        predicted_direction=event.sentiment,
        realized_direction=realized_direction,  # type: ignore[arg-type]
        direction_hit=hit,
        brier_score=brier,
        threshold_pct=threshold_pct,
        reason="Forward outcome evaluated from closed bars.",
        computed_at=_now(),
    )


def resolve_market_symbol(symbol: str, available_symbols: set[str] | None = None) -> str:
    canonical = symbol.upper()
    available = {item.upper() for item in (available_symbols or set())}
    candidates = _MARKET_SYMBOL_ALIASES.get(canonical, (canonical,))
    return next((candidate for candidate in candidates if candidate.upper() in available), candidates[0])


def calibrate_pending_event_outcomes(event_limit: int = 200) -> EventCalibrationRunResponse:
    events = list_market_events(max(1, min(2000, event_limit)))
    result = EventCalibrationRunResponse(ok=True, events_checked=len(events))
    existing = {
        (outcome.event_id, outcome.symbol, outcome.horizon): outcome
        for outcome in list_event_outcomes(limit=20_000)
    }
    events_by_symbol: dict[str, list[MarketEventRecord]] = defaultdict(list)
    for event in events:
        for symbol in event.symbols:
            events_by_symbol[symbol.upper()].append(event)

    available_symbols = {symbol.upper() for symbol in list_symbols()}

    for symbol, symbol_events in events_by_symbol.items():
        market_symbol = resolve_market_symbol(symbol, available_symbols)
        try:
            bars = get_bars(market_symbol, "M5", 5000, prefer_live=True)
            fetch_error = ""
        except MarketDataError as exc:
            bars = pd.DataFrame()
            fetch_error = str(exc)
            result.errors.append(f"{symbol} ({market_symbol}): {fetch_error}")
            result.ok = False
        for event in symbol_events:
            for horizon in _HORIZONS:
                prior = existing.get((event.id, symbol, horizon))
                if prior and prior.status == "evaluated":
                    continue
                if fetch_error:
                    outcome = _empty_outcome(event, symbol, horizon, "unavailable", fetch_error, market_symbol)
                else:
                    outcome = evaluate_event_against_bars(
                        event,
                        symbol,
                        horizon,
                        bars,
                        market_symbol=market_symbol,
                    )
                upsert_event_outcome(outcome)
                if outcome.status == "evaluated":
                    result.outcomes_evaluated += 1
                elif outcome.status == "pending":
                    result.outcomes_pending += 1
                else:
                    result.outcomes_unavailable += 1
    return result


def _gate(samples: int, hit_rate: float | None, brier: float | None, minimum_samples: int) -> tuple[str, str]:
    if samples < minimum_samples:
        return "insufficient_samples", f"Needs {minimum_samples - samples} more evaluated outcomes."
    if hit_rate is None or brier is None:
        return "observe", "Metrics are incomplete."
    if hit_rate >= 0.55 and brier <= 0.24:
        return "eligible", "Historical accuracy and calibration clear the research gate."
    if hit_rate < 0.45 or brier > 0.30:
        return "degraded", "Observed performance is below the evidence gate."
    return "observe", "Sample is sufficient, but performance does not clear the evidence gate."


def build_event_calibration() -> EventCalibrationResponse:
    outcomes = list_event_outcomes(limit=20_000)
    events = {event.id: event for event in list_market_events(20_000)}
    try:
        minimum_samples = int(os.getenv("EVENT_CALIBRATION_MIN_SAMPLES", "30"))
    except ValueError:
        minimum_samples = 30
    minimum_samples = max(10, min(1000, minimum_samples))
    evaluated = [outcome for outcome in outcomes if outcome.status == "evaluated" and outcome.event_id in events]
    grouped: dict[tuple[str, str], list[EventOutcomeRecord]] = defaultdict(list)
    for outcome in evaluated:
        grouped[(events[outcome.event_id].event_type, outcome.horizon)].append(outcome)

    groups: list[EventCalibrationGroup] = []
    for (event_type, horizon), rows in sorted(grouped.items()):
        returns = [float(row.forward_return_pct) for row in rows if row.forward_return_pct is not None]
        hits = [1.0 if row.direction_hit else 0.0 for row in rows if row.direction_hit is not None]
        briers = [float(row.brier_score) for row in rows if row.brier_score is not None]
        favorable = [float(row.max_favorable_excursion_pct) for row in rows if row.max_favorable_excursion_pct is not None]
        adverse = [float(row.max_adverse_excursion_pct) for row in rows if row.max_adverse_excursion_pct is not None]
        hit_rate = fmean(hits) if hits else None
        average_brier = fmean(briers) if briers else None
        gate, reason = _gate(len(rows), hit_rate, average_brier, minimum_samples)
        groups.append(
            EventCalibrationGroup(
                event_type=event_type,
                horizon=horizon,  # type: ignore[arg-type]
                samples=len(rows),
                hit_rate=hit_rate,
                average_return_pct=fmean(returns) if returns else None,
                median_return_pct=median(returns) if returns else None,
                average_brier_score=average_brier,
                average_favorable_excursion_pct=fmean(favorable) if favorable else None,
                average_adverse_excursion_pct=fmean(adverse) if adverse else None,
                gate=gate,  # type: ignore[arg-type]
                gate_reason=reason,
            )
        )
    return EventCalibrationResponse(
        evaluated_outcomes=len(evaluated),
        pending_outcomes=sum(1 for outcome in outcomes if outcome.status == "pending"),
        unavailable_outcomes=sum(1 for outcome in outcomes if outcome.status == "unavailable"),
        minimum_samples=minimum_samples,
        groups=groups,
    )
