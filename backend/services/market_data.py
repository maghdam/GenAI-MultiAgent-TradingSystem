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


_DEFAULT_BARS_CACHE_TTL_SEC = 8.0
_LIVE_BARS_CACHE_TTL_SEC = 1.0
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
    "W1": 604_800,
}


def assess_market_bar_freshness(
    timeframe: str,
    bar_timestamp: datetime | None,
    *,
    now: datetime | None = None,
) -> tuple[bool, dict[str, object], str | None]:
    evaluated_at = now or datetime.now(UTC)
    details: dict[str, object] = {
        "bar_timestamp": bar_timestamp.isoformat() if bar_timestamp else None,
        "evaluated_at": evaluated_at.isoformat(),
        "timeframe": (timeframe or "M5").upper(),
    }
    if bar_timestamp is None:
        return False, details, "No market-data timestamp was available for the signal."

    normalized_now = (
        evaluated_at.astimezone(UTC).replace(tzinfo=None)
        if evaluated_at.tzinfo is not None
        else evaluated_at
    )
    normalized_bar = (
        bar_timestamp.astimezone(UTC).replace(tzinfo=None)
        if bar_timestamp.tzinfo is not None
        else bar_timestamp
    )
    timeframe_key = (timeframe or "M5").upper()
    max_age_seconds = (_TIMEFRAME_SECONDS.get(timeframe_key, 300) * 3) + 30
    age_seconds = (normalized_now - normalized_bar).total_seconds()

    details["bar_timestamp"] = normalized_bar.isoformat()
    details["evaluated_at"] = normalized_now.isoformat()
    details["bar_age_seconds"] = max(0.0, age_seconds)
    details["max_bar_age_seconds"] = float(max_age_seconds)

    if age_seconds > max_age_seconds:
        return False, details, "Latest market bar is stale for the configured timeframe."
    return True, details, None


def record_market_bar_freshness(
    symbol: str,
    timeframe: str,
    bar_timestamp: datetime | None,
    *,
    now: datetime | None = None,
) -> tuple[bool, dict[str, object], str | None]:
    ok, details, reason = assess_market_bar_freshness(
        timeframe,
        bar_timestamp,
        now=now,
    )
    checked_at = now or datetime.now(UTC)
    normalized_checked = (
        checked_at.astimezone(UTC).replace(tzinfo=None)
        if checked_at.tzinfo is not None
        else checked_at
    )
    market_data_dependency_state.last_checked_at = normalized_checked
    market_data_dependency_state.last_symbol = symbol.upper()
    market_data_dependency_state.last_timeframe = timeframe.upper()
    market_data_dependency_state.last_success = ok
    market_data_dependency_state.market_data_ready = ok
    if ok:
        market_data_dependency_state.last_success_at = normalized_checked
        market_data_dependency_state.last_reason = (
            f"Fresh market bar available for {symbol.upper()}:{timeframe.upper()}."
        )
    else:
        market_data_dependency_state.last_reason = (
            f"{reason or 'Market data is not ready.'} "
            f"{symbol.upper()}:{timeframe.upper()} "
            f"(age={details.get('bar_age_seconds')}s, "
            f"max={details.get('max_bar_age_seconds')}s)"
        )
    return ok, details, reason


def _cache_ttl_for_request(prefer_live: bool) -> float:
    return _LIVE_BARS_CACHE_TTL_SEC if prefer_live else _DEFAULT_BARS_CACHE_TTL_SEC


def _get_cached_bars(symbol: str, timeframe: str, num_bars: int, *, ttl_sec: float) -> pd.DataFrame | None:
    key = (symbol.upper(), timeframe.upper())
    now = monotonic()
    with _bars_cache_lock:
        cached = _bars_cache.get(key)
        if not cached:
            return None
        stored_at, df = cached
        if now - stored_at >= ttl_sec:
            return None
        if len(df) < num_bars:
            return None
        return df.tail(num_bars).copy()


def _store_cached_bars(symbol: str, timeframe: str, df: pd.DataFrame) -> None:
    key = (symbol.upper(), timeframe.upper())
    with _bars_cache_lock:
        _bars_cache[key] = (monotonic(), df.copy())


def _persistent_cache_fresh_enough(fetched_at: datetime | None, timeframe: str) -> bool:
    if fetched_at is None:
        return False
    tf_seconds = _TIMEFRAME_SECONDS.get((timeframe or "").upper(), 300)
    max_age = max(30, tf_seconds * 2)
    age_seconds = (datetime.now(UTC).replace(tzinfo=None) - fetched_at.replace(tzinfo=None)).total_seconds()
    return age_seconds <= max_age


def get_bars(symbol: str, timeframe: str, num_bars: int, *, prefer_live: bool = False) -> pd.DataFrame:
    cached = _get_cached_bars(symbol, timeframe, num_bars, ttl_sec=_cache_ttl_for_request(prefer_live))
    market_data_dependency_state.last_symbol = symbol.upper()
    market_data_dependency_state.last_timeframe = timeframe.upper()
    market_data_dependency_state.last_checked_at = datetime.now(UTC).replace(tzinfo=None)

    if cached is not None:
        market_data_dependency_state.last_success = True
        market_data_dependency_state.last_success_at = datetime.now(UTC).replace(tzinfo=None)
        market_data_dependency_state.market_data_ready = True
        market_data_dependency_state.last_reason = f"Served {len(cached)} cached bars for {symbol.upper()}:{timeframe.upper()}"
        return cached

    if not prefer_live:
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
    checked_at = datetime.now(UTC).replace(tzinfo=None)
    market_data_dependency_state.last_checked_at = checked_at
    market_data_dependency_state.last_symbol = symbol.upper()
    market_data_dependency_state.last_timeframe = timeframe.upper()

    if status.get("ok") and status.get("latest_bar_at"):
        try:
            latest_bar_at = datetime.fromisoformat(str(status["latest_bar_at"]).replace("Z", "+00:00"))
        except ValueError:
            latest_bar_at = None
        fresh, freshness, stale_reason = record_market_bar_freshness(
            symbol,
            timeframe,
            latest_bar_at,
            now=checked_at,
        )
        status["ok"] = fresh
        status["freshness"] = freshness
        if not fresh:
            status["reason"] = stale_reason or "Latest market bar is stale."
    else:
        market_data_dependency_state.last_success = bool(status.get("ok"))
        market_data_dependency_state.market_data_ready = bool(status.get("ok"))
        if status.get("ok"):
            market_data_dependency_state.last_success_at = checked_at
        market_data_dependency_state.last_reason = str(status.get("reason") or "")

    return {
        **status,
        **market_data_dependency_state.snapshot(),
    }
