from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from backend.services import market_data
from backend.storage.db import get_db
from backend.storage.repositories import load_cached_market_bars, upsert_market_bars


def _sample_bars(base_close: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"open": base_close - 1.0, "high": base_close + 1.0, "low": base_close - 2.0, "close": base_close, "volume": 10},
            {"open": base_close, "high": base_close + 2.0, "low": base_close - 1.0, "close": base_close + 1.0, "volume": 12},
        ],
        index=pd.to_datetime(["2026-03-29T10:00:00Z", "2026-03-29T10:05:00Z"], utc=True),
    )


def test_get_bars_uses_persisted_cache_when_fresh(monkeypatch) -> None:
    market_data._bars_cache.clear()
    cached = _sample_bars(100.0)
    upsert_market_bars("XAUUSD", "M5", cached)

    def _unexpected_fetch(*args, **kwargs):
        raise AssertionError("broker fetch should not run when persisted cache is fresh")

    monkeypatch.setattr("backend.services.market_data.adapter.get_bars", _unexpected_fetch)

    bars = market_data.get_bars("XAUUSD", "M5", 2)

    assert len(bars) == 2
    assert float(bars.iloc[-1]["close"]) == 101.0


def test_get_bars_refreshes_stale_persisted_cache(monkeypatch) -> None:
    market_data._bars_cache.clear()
    stale = _sample_bars(100.0)
    fresh = _sample_bars(200.0)
    upsert_market_bars("XAUUSD", "M5", stale)

    stale_fetched_at = (datetime.now(UTC) - timedelta(minutes=20)).replace(tzinfo=None).isoformat()
    with get_db() as db:
        db.execute("UPDATE market_bars SET fetched_at = ?", (stale_fetched_at,))
        db.commit()

    monkeypatch.setattr(
        "backend.services.market_data.adapter.get_bars",
        lambda symbol, timeframe, num_bars: (fresh, float(fresh["close"].iloc[-1])),
    )

    bars = market_data.get_bars("XAUUSD", "M5", 2)
    persisted, fetched_at = load_cached_market_bars("XAUUSD", "M5", 2)

    assert len(bars) == 2
    assert float(bars.iloc[-1]["close"]) == 201.0
    assert len(persisted) == 2
    assert float(persisted.iloc[-1]["close"]) == 201.0
    assert fetched_at is not None
