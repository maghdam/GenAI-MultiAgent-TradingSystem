from __future__ import annotations

from datetime import UTC, datetime
from threading import Lock
from time import monotonic

import pandas as pd

from backend.adapters.ctrader import adapter
from backend.storage.repositories import load_cached_market_bars, upsert_market_bars
from backend.services.runtime_state import market_data_dependency_state


class MarketDataError(RuntimeError):
    pass


_BARS_CACHE_TTL_SEC = 8.0
_bars_cache_lock = Lock()
_bars_cache: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}
_TIMEFRAME_SECONDS = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1_800,
    "H1": 3_600,
    "H4": 14_400,
    "D1": 86_400,
}


def _get_cached_bars(symbol: str, timeframe: str, num_bars: int) -> pd.DataFrame | None:
    key = (symbol.upper(), timeframe.upper())
    now = monotonic()
    with _bars_cache_lock:
        cached = _bars_cache.get(key)
        if not cached:
            return None
        expires_at, df = cached
        if now >= expires_at:
            _bars_cache.pop(key, None)
            return None
        if len(df) < num_bars:
            return None
        return df.tail(num_bars).copy()


def _store_cached_bars(symbol: str, timeframe: str, df: pd.DataFrame) -> None:
    key = (symbol.upper(), timeframe.upper())
    with _bars_cache_lock:
        _bars_cache[key] = (monotonic() + _BARS_CACHE_TTL_SEC, df.copy())


def _persistent_cache_fresh_enough(fetched_at: datetime | None, timeframe: str) -> bool:
    if fetched_at is None:
        return False
    tf_seconds = _TIMEFRAME_SECONDS.get((timeframe or "").upper(), 300)
    max_age = max(30, tf_seconds * 2)
    age_seconds = (datetime.now(UTC).replace(tzinfo=None) - fetched_at.replace(tzinfo=None)).total_seconds()
    return age_seconds <= max_age


def get_bars(symbol: str, timeframe: str, num_bars: int) -> pd.DataFrame:
    cached = _get_cached_bars(symbol, timeframe, num_bars)
    market_data_dependency_state.last_symbol = symbol.upper()
    market_data_dependency_state.last_timeframe = timeframe.upper()
    market_data_dependency_state.last_checked_at = datetime.now(UTC).replace(tzinfo=None)

    if cached is not None:
        market_data_dependency_state.last_success = True
        market_data_dependency_state.last_success_at = datetime.now(UTC).replace(tzinfo=None)
        market_data_dependency_state.market_data_ready = True
        market_data_dependency_state.last_reason = f"Served {len(cached)} cached bars for {symbol.upper()}:{timeframe.upper()}"
        return cached

    persisted, fetched_at = load_cached_market_bars(symbol, timeframe, num_bars)
    if len(persisted) >= num_bars and _persistent_cache_fresh_enough(fetched_at, timeframe):
        _store_cached_bars(symbol, timeframe, persisted)
        market_data_dependency_state.last_success = True
        market_data_dependency_state.last_success_at = datetime.now(UTC).replace(tzinfo=None)
        market_data_dependency_state.market_data_ready = True
        market_data_dependency_state.last_reason = f"Served {len(persisted)} persisted bars for {symbol.upper()}:{timeframe.upper()}"
        return persisted.tail(num_bars).copy()

    try:
        df, _ = adapter.get_bars(symbol=symbol, timeframe=timeframe, num_bars=num_bars)
    except MarketDataError:
        raise
    except Exception as exc:
        market_data_dependency_state.last_success = False
        market_data_dependency_state.market_data_ready = False
        market_data_dependency_state.last_reason = str(exc)
        raise MarketDataError(str(exc)) from exc
    if df is None or df.empty:
        market_data_dependency_state.last_success = False
        market_data_dependency_state.market_data_ready = False
        market_data_dependency_state.last_reason = f"No market data available for {symbol}:{timeframe}"
        raise MarketDataError(market_data_dependency_state.last_reason)
    _store_cached_bars(symbol, timeframe, df)
    upsert_market_bars(symbol, timeframe, df)
    market_data_dependency_state.last_success = True
    market_data_dependency_state.last_success_at = datetime.now(UTC).replace(tzinfo=None)
    market_data_dependency_state.market_data_ready = True
    market_data_dependency_state.last_reason = f"Fetched {len(df)} bars for {symbol.upper()}:{timeframe.upper()}"
    return df


def get_market_data_status(symbol: str, timeframe: str) -> dict[str, object]:
    status = adapter.get_market_data_status(symbol=symbol, timeframe=timeframe)
    market_data_dependency_state.last_checked_at = datetime.now(UTC).replace(tzinfo=None)
    market_data_dependency_state.last_symbol = symbol.upper()
    market_data_dependency_state.last_timeframe = timeframe.upper()
    market_data_dependency_state.last_success = bool(status.get("ok"))
    market_data_dependency_state.market_data_ready = bool(status.get("ok"))
    if status.get("ok"):
        market_data_dependency_state.last_success_at = datetime.now(UTC).replace(tzinfo=None)
    market_data_dependency_state.last_reason = str(status.get("reason") or "")
    return {
        **status,
        **market_data_dependency_state.snapshot(),
    }
