from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pandas as pd

from backend.security.credential_store import get_alpaca_secret


_DEFAULT_DATA_BASE_URL = "https://data.alpaca.markets"
_DEFAULT_STOCK_FEED = "iex"
_TIMEFRAME_MAP = {
    "M1": "1Min",
    "M5": "5Min",
    "M15": "15Min",
    "M30": "30Min",
    "H1": "1Hour",
    "H4": "4Hour",
    "D1": "1Day",
    "W1": "1Week",
}
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


class AlpacaMarketDataError(RuntimeError):
    """Raised when Alpaca market data cannot be returned safely."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _base_url() -> str:
    return (os.getenv("ALPACA_DATA_BASE_URL") or _DEFAULT_DATA_BASE_URL).strip().rstrip("/")


def _stock_feed() -> str:
    value = (os.getenv("ALPACA_STOCK_DATA_FEED") or _DEFAULT_STOCK_FEED).strip().lower()
    allowed = {"iex", "sip", "otc", "boats"}
    if value not in allowed:
        raise AlpacaMarketDataError(
            "ALPACA_STOCK_DATA_FEED must be one of iex, sip, otc, or boats."
        )
    return value


def _credentials() -> tuple[str, str]:
    key_id = (get_alpaca_secret("ALPACA_API_KEY_ID") or "").strip()
    secret_key = (get_alpaca_secret("ALPACA_API_SECRET_KEY") or "").strip()
    if not key_id or not secret_key:
        raise AlpacaMarketDataError(
            "Alpaca market-data credentials are not configured. "
            "Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY."
        )
    return key_id, secret_key


def _request_json(path: str, *, params: dict[str, object]) -> Any:
    key_id, secret_key = _credentials()
    headers = {
        "APCA-API-KEY-ID": key_id,
        "APCA-API-SECRET-KEY": secret_key,
        "Accept": "application/json",
    }
    try:
        with httpx.Client(base_url=_base_url(), timeout=15.0, headers=headers) as client:
            response = client.get(path, params=params)
    except httpx.HTTPError as exc:
        raise AlpacaMarketDataError(
            f"Alpaca market-data request failed before a response was received for {path}."
        ) from exc

    if response.status_code >= 400:
        raise AlpacaMarketDataError(
            f"Alpaca market-data request returned HTTP {response.status_code} for {path}.",
            status_code=int(response.status_code),
        )

    try:
        return response.json()
    except ValueError as exc:
        raise AlpacaMarketDataError(
            f"Alpaca market-data request returned a non-JSON response for {path}."
        ) from exc


def _lookback_start(
    *,
    timeframe: str,
    num_bars: int,
    asset_class: str,
    now: datetime | None = None,
) -> str:
    timeframe_key = timeframe.upper()
    seconds_per_bar = _TIMEFRAME_SECONDS[timeframe_key]
    horizon_seconds = seconds_per_bar * max(1, int(num_bars))
    if asset_class == "us_equity":
        multiplier = 4.0 if seconds_per_bar < 86_400 else 1.7
        minimum_seconds = 7 * 86_400 if seconds_per_bar < 86_400 else 30 * 86_400
    else:
        multiplier = 1.5
        minimum_seconds = 2 * 86_400
    lookback_seconds = max(minimum_seconds, int(horizon_seconds * multiplier))
    evaluated_at = now or datetime.now(UTC)
    start = evaluated_at - timedelta(seconds=lookback_seconds)
    return start.isoformat().replace("+00:00", "Z")


def _extract_symbol_bars(payload: Any, symbol: str) -> tuple[list[dict[str, object]], str | None]:
    if not isinstance(payload, dict):
        raise AlpacaMarketDataError("Alpaca market-data response had an unexpected shape.")

    raw_bars = payload.get("bars")
    if isinstance(raw_bars, dict):
        candidates = (
            raw_bars.get(symbol),
            raw_bars.get(symbol.upper()),
            raw_bars.get(symbol.replace("/", "")),
        )
        bars = next((value for value in candidates if isinstance(value, list)), None)
    elif isinstance(raw_bars, list):
        bars = raw_bars
    else:
        bars = None

    if bars is None:
        bars = []
    for item in bars:
        if not isinstance(item, dict):
            raise AlpacaMarketDataError("Alpaca market-data bars contained an invalid row.")

    token = payload.get("next_page_token")
    return list(bars), str(token) if token else None


def _bars_frame(rows: list[dict[str, object]], symbol: str, timeframe: str) -> pd.DataFrame:
    normalized: list[dict[str, object]] = []
    for item in rows:
        timestamp = item.get("t")
        if timestamp is None:
            raise AlpacaMarketDataError(
                f"Alpaca market-data bar for {symbol}:{timeframe} is missing a timestamp."
            )
        try:
            normalized.append(
                {
                    "time": pd.to_datetime(timestamp, utc=True),
                    "open": float(item["o"]),
                    "high": float(item["h"]),
                    "low": float(item["l"]),
                    "close": float(item["c"]),
                    "volume": float(item.get("v") or 0.0),
                }
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise AlpacaMarketDataError(
                f"Alpaca market-data bar for {symbol}:{timeframe} is malformed."
            ) from exc

    if not normalized:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    frame = pd.DataFrame(normalized).set_index("time").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame.index.name = None
    return frame


def get_bars(
    *,
    symbol: str,
    timeframe: str,
    num_bars: int,
    asset_class: str,
) -> pd.DataFrame:
    normalized_symbol = str(symbol or "").strip().upper()
    if not normalized_symbol:
        raise ValueError("Alpaca market-data symbol must not be empty.")

    normalized_timeframe = str(timeframe or "").strip().upper()
    if normalized_timeframe not in _TIMEFRAME_MAP:
        raise ValueError(
            "Unsupported Alpaca timeframe. Use M1, M5, M15, M30, H1, H4, D1, or W1."
        )

    normalized_asset_class = str(asset_class or "").strip().lower()
    if normalized_asset_class not in {"us_equity", "crypto"}:
        raise ValueError("Alpaca asset_class must be us_equity or crypto.")

    normalized_limit = int(num_bars)
    if normalized_limit < 1 or normalized_limit > 10_000:
        raise ValueError("Alpaca market-data num_bars must be between 1 and 10000.")

    if normalized_asset_class == "us_equity":
        path = "/v2/stocks/bars"
        params: dict[str, object] = {
            "symbols": normalized_symbol,
            "timeframe": _TIMEFRAME_MAP[normalized_timeframe],
            "start": _lookback_start(
                timeframe=normalized_timeframe,
                num_bars=normalized_limit,
                asset_class=normalized_asset_class,
            ),
            "limit": normalized_limit,
            "sort": "desc",
            "adjustment": "raw",
            "feed": _stock_feed(),
        }
    else:
        path = "/v1beta3/crypto/us/bars"
        params = {
            "symbols": normalized_symbol,
            "timeframe": _TIMEFRAME_MAP[normalized_timeframe],
            "start": _lookback_start(
                timeframe=normalized_timeframe,
                num_bars=normalized_limit,
                asset_class=normalized_asset_class,
            ),
            "limit": normalized_limit,
            "sort": "desc",
        }

    rows: list[dict[str, object]] = []
    page_token: str | None = None
    while len(rows) < normalized_limit:
        request_params = dict(params)
        request_params["limit"] = min(10_000, normalized_limit - len(rows))
        if page_token:
            request_params["page_token"] = page_token

        payload = _request_json(path, params=request_params)
        page_rows, next_page_token = _extract_symbol_bars(payload, normalized_symbol)
        rows.extend(page_rows)
        if not next_page_token or next_page_token == page_token:
            break
        page_token = next_page_token

    frame = _bars_frame(rows, normalized_symbol, normalized_timeframe)
    if frame.empty:
        raise AlpacaMarketDataError(
            f"No Alpaca market data available for {normalized_symbol}:{normalized_timeframe}."
        )
    return frame.tail(normalized_limit).copy()
