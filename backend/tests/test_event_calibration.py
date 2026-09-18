from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from backend.domain.models import MarketEventInput
from backend.services.event_calibration import (
    _gate,
    calibrate_pending_event_outcomes,
    evaluate_event_against_bars,
    resolve_market_symbol,
)
from backend.services.event_intelligence import ingest_events
from backend.storage.repositories import list_event_outcomes


def _event():
    inserted, _ = ingest_events(
        [
            MarketEventInput(
                source="Reuters",
                title="Nasdaq 100 surges on strong demand",
                url="https://www.reuters.com/markets/nasdaq-test",
                published_at=datetime(2026, 7, 19, 10, 2, tzinfo=UTC),
            )
        ]
    )
    return inserted[0]


def _bars(periods: int = 300) -> pd.DataFrame:
    index = pd.date_range("2026-07-19T10:00:00Z", periods=periods, freq="5min")
    closes = [100.0 + index_value * 0.1 for index_value in range(periods)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [value + 0.2 for value in closes],
            "low": [value - 0.1 for value in closes],
            "close": closes,
            "volume": [100.0] * periods,
        },
        index=index,
    )


def test_evaluate_event_uses_first_closed_bar_after_event_without_lookahead() -> None:
    outcome = evaluate_event_against_bars(_event(), "NAS100", "5m", _bars(5))

    assert outcome.status == "evaluated"
    assert outcome.reference_at == datetime(2026, 7, 19, 10, 5, tzinfo=UTC)
    assert outcome.target_at == datetime(2026, 7, 19, 10, 10, tzinfo=UTC)
    assert outcome.reference_price == 100.1
    assert outcome.target_price == 100.2
    assert outcome.forward_return_pct is not None and outcome.forward_return_pct > 0
    assert outcome.direction_hit is True
    assert outcome.brier_score == 0.0


def test_evaluate_event_stays_pending_until_horizon_closes() -> None:
    outcome = evaluate_event_against_bars(_event(), "NAS100", "4h", _bars(10))

    assert outcome.status == "pending"
    assert outcome.reference_price == 100.1
    assert outcome.target_price is None


def test_calibration_run_persists_all_horizons(monkeypatch) -> None:
    _event()
    bars = _bars(310)
    monkeypatch.setattr(
        "backend.services.event_calibration.get_bars",
        lambda symbol, timeframe, num_bars, prefer_live: bars,
    )

    result = calibrate_pending_event_outcomes()
    outcomes = list_event_outcomes(20)

    assert result.ok is True
    assert result.events_checked == 1
    assert result.outcomes_evaluated == 4
    assert {outcome.horizon for outcome in outcomes} == {"5m", "30m", "4h", "1d"}
    assert all(outcome.status == "evaluated" for outcome in outcomes)


def test_evidence_gate_requires_sample_size_and_calibration() -> None:
    assert _gate(12, 0.70, 0.10, 30)[0] == "insufficient_samples"
    assert _gate(30, 0.60, 0.20, 30)[0] == "eligible"
    assert _gate(30, 0.40, 0.35, 30)[0] == "degraded"


def test_nasdaq_canonical_symbol_resolves_to_broker_alias() -> None:
    assert resolve_market_symbol("NAS100", {"EURUSD", "US100", "XAUUSD"}) == "US100"
    assert resolve_market_symbol("NAS100", {"USTEC"}) == "USTEC"
