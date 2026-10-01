from __future__ import annotations

from math import isfinite
from typing import Mapping

import pandas as pd


_REQUIRED_OHLC = ("open", "high", "low", "close")


def max_bar_range_pct(timeframe: str) -> float:
    return {
        "M1": 1.5,
        "M5": 2.0,
        "M15": 3.0,
        "M30": 4.0,
        "H1": 5.0,
        "H4": 8.0,
        "D1": 15.0,
        "W1": 25.0,
    }.get((timeframe or "M5").upper(), 3.0)


def assess_market_bar_snapshot(
    timeframe: str,
    bar_snapshot: Mapping[str, object] | None,
    *,
    check_extreme_range: bool = True,
) -> tuple[bool, dict[str, object], str | None]:
    if not bar_snapshot:
        return False, {}, "Latest market bar is incomplete or non-numeric."

    details: dict[str, object] = {
        "bar_open": bar_snapshot.get("open"),
        "bar_high": bar_snapshot.get("high"),
        "bar_low": bar_snapshot.get("low"),
        "bar_close": bar_snapshot.get("close"),
    }
    try:
        open_price = float(bar_snapshot["open"])
        high_price = float(bar_snapshot["high"])
        low_price = float(bar_snapshot["low"])
        close_price = float(bar_snapshot["close"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False, details, "Latest market bar is incomplete or non-numeric."

    prices = (open_price, high_price, low_price, close_price)
    if not all(isfinite(value) for value in prices):
        return False, details, "Latest market bar is incomplete or non-numeric."

    if min(prices) <= 0:
        return False, details, "Latest market bar contains non-positive prices."
    if high_price < low_price:
        return False, details, "Latest market bar has invalid high/low ordering."
    if not (low_price <= open_price <= high_price and low_price <= close_price <= high_price):
        return False, details, "Latest market bar has open/close outside the high-low range."

    range_pct = ((high_price - low_price) / close_price) * 100.0
    details["bar_range_pct"] = range_pct
    details["max_bar_range_pct"] = max_bar_range_pct(timeframe)
    if check_extreme_range and range_pct > float(details["max_bar_range_pct"]):
        return False, details, "Latest market bar range is too wide for the configured timeframe."

    return True, details, None


def assess_market_frame(
    timeframe: str,
    df: pd.DataFrame | None,
) -> tuple[bool, dict[str, object], str | None]:
    if df is None or df.empty:
        return False, {"bars_checked": 0}, "Market data contains no bars."

    missing = [column for column in _REQUIRED_OHLC if column not in df.columns]
    if missing:
        return (
            False,
            {"bars_checked": len(df), "missing_columns": missing},
            f"Market data is incomplete; missing OHLC columns: {', '.join(missing)}.",
        )

    if isinstance(df.index, pd.DatetimeIndex) and bool(df.index.isna().any()):
        return (
            False,
            {"bars_checked": len(df), "invalid_timestamp": True},
            "Market data contains an invalid bar timestamp.",
        )

    last_position = len(df) - 1
    for position, (bar_index, row) in enumerate(df.iterrows()):
        ok, details, reason = assess_market_bar_snapshot(
            timeframe,
            {column: row[column] for column in _REQUIRED_OHLC},
            check_extreme_range=(position == last_position),
        )
        if not ok:
            return (
                False,
                {
                    **details,
                    "bars_checked": len(df),
                    "invalid_bar_position": position,
                    "invalid_bar_timestamp": (
                        bar_index.isoformat()
                        if hasattr(bar_index, "isoformat")
                        else str(bar_index)
                    ),
                },
                f"Malformed market data: {reason}",
            )

    return True, {"bars_checked": len(df)}, None
