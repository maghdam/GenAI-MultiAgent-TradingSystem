"""Opt-in localhost-only MCP entry point for *TradeAgent* research reads.

This is a thin, read-only facade over vetted TradeAgent GET APIs. It is NOT
an execution surface and does not start from the normal TradeAgent application.
Never expose either loopback listener through a tunnel or public reverse proxy.
"""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime

import httpx

_BACKEND_URL = "http://127.0.0.1:4000/api/market/candles"
_STUDIO_FILES_URL = "http://127.0.0.1:4000/api/studio/strategy-files"
_STUDIO_ROUTE = "/api/studio/strategy-files"
_MAX_STUDIO_RESPONSE_BYTES = 64_000
_MAX_STUDIO_OBSERVED_FILES = 512
_MAX_STUDIO_RETURNED_FILES = 50
_STUDIO_FILE_RE = re.compile(r"[a-z0-9_-]{1,64}\.py\Z")
_ALLOWED_SYMBOL = re.compile(r"[A-Za-z0-9_.-]{2,32}\Z")
# Mirrors the existing cTrader adapter, which currently rejects W1.
_ALLOWED_TIMEFRAMES = frozenset({"M1", "M5", "M15", "M30", "H1", "H4", "D1"})
_PRICE_KEYS = ("open", "high", "low", "close")


def _check_rows(payload: object, requested: int) -> tuple[list[dict], datetime, datetime]:
    if not isinstance(payload, dict):
        raise RuntimeError("TradeAgent returned an invalid market-data payload")
    rows = payload.get("candles")
    if not isinstance(rows, list) or not 0 < len(rows) <= requested:
        raise RuntimeError("TradeAgent returned an empty or invalid candle set")

    previous: int | None = None
    first: datetime | None = None
    last: datetime | None = None
    for row in rows:
        if not isinstance(row, dict) or not set(_PRICE_KEYS).issubset(row):
            raise RuntimeError("TradeAgent returned malformed candle rows")
        timestamp = row.get("time")
        if type(timestamp) is not int or timestamp <= 0 or (previous is not None and timestamp <= previous):
            raise RuntimeError("TradeAgent returned invalid or out-of-order candle timestamps")
        try:
            bar_time = datetime.fromtimestamp(timestamp, tz=UTC)
        except (OverflowError, OSError, ValueError) as exc:
            raise RuntimeError("TradeAgent returned an invalid candle timestamp") from exc
        prices = [row[key] for key in _PRICE_KEYS]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in prices):
            raise RuntimeError("TradeAgent returned non-finite or nonnumeric candle prices")
        opn, high, low, close = prices
        if low > high or high < max(opn, close) or low > min(opn, close):
            raise RuntimeError("TradeAgent returned inconsistent candle OHLC prices")
        previous = timestamp
        if first is None:
            first = bar_time
        last = bar_time
    assert first is not None and last is not None
    return rows, first, last


def get_ctrader_candles(symbol: str, timeframe: str = "M5", num_bars: int = 100) -> dict:
    """Read existing TradeAgent cTrader-route M1-D1 candles, never place orders.

    Returned seconds are interpreted as UTC; a completed bar/open-label
    convention or independent broker feed provenance is NOT attested here.
    Requested live data may still be served from TradeAgent's memory cache.
    """
    if not isinstance(symbol, str) or not _ALLOWED_SYMBOL.fullmatch(symbol.strip()):
        raise ValueError("symbol must be 2-32 broker ticker characters (A-Z, 0-9, dot, underscore, hyphen)")
    if not isinstance(timeframe, str) or timeframe.strip().upper() not in _ALLOWED_TIMEFRAMES:
        raise ValueError("timeframe must be M1, M5, M15, M30, H1, H4 or D1")
    if type(num_bars) is not int or not 2 <= num_bars <= 500:
        raise ValueError("num_bars must be an integer between 2 and 500")

    sym = symbol.strip().upper()
    tf = timeframe.strip().upper()
    try:
        # Hard-coded loopback target: no caller-chosen URL, no inherited proxy.
        with httpx.Client(timeout=20.0, trust_env=False, follow_redirects=False) as client:
            response = client.get(_BACKEND_URL, params={
                "symbol": sym,
                "timeframe": tf,
                "num_bars": num_bars,
                "live": "true",
            })
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError("Local TradeAgent candle read failed or returned invalid JSON") from exc

    candles, first, last = _check_rows(payload, num_bars)
    return {
        "provider_route": "ctrader_via_local_tradeagent",
        "symbol": sym,
        "timeframe": tf,
        "requested_bars": num_bars,
        "observed_bars": len(candles),
        "requested_live": True,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "first_bar_utc": first.isoformat(),
        "last_bar_utc": last.isoformat(),
        "data_provenance": "TradeAgent cTrader route; cached-versus-broker origin not independently attested",
        "timestamp_convention": "Unix seconds interpreted as UTC; bar completion and open-label convention unverified",
        "candles": candles,
        "warning": "Research observations only. No signal, fills, strategy profit or broker order is asserted.",
    }


def list_studio_saved_strategies() -> dict:
    """List bounded saved Strategy Studio names without source or execution.

    This only reads the existing TradeAgent strategy-files GET route. The
    upstream response also includes a local cwd; never return or log it.
    File names are research catalog entries, not verified strategy lifecycles.
    """
    try:
        # Fixed loopback route, no caller-controlled URL, proxies or redirects.
        with httpx.Client(timeout=10.0, trust_env=False, follow_redirects=False) as client:
            response = client.get(_STUDIO_FILES_URL)
            response.raise_for_status()
            if len(response.content) > _MAX_STUDIO_RESPONSE_BYTES:
                raise RuntimeError("Local TradeAgent strategy catalog payload exceeds the size limit")
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError("Local TradeAgent strategy catalog read failed or returned invalid JSON") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
        raise RuntimeError("Local TradeAgent returned an invalid strategy catalog envelope")

    files = payload["files"]
    if len(files) > _MAX_STUDIO_OBSERVED_FILES:
        raise RuntimeError("Local TradeAgent strategy catalog exceeds the observed-file limit")
    # Strictly permit existing lower-case saved-file names (the Studio source
    # loader itself normalizes strategy IDs to lower case). Reject all paths,
    # extensions, oversized names and nonstrings instead of guessing.
    if any(type(name) is not str or not _STUDIO_FILE_RE.fullmatch(name) for name in files):
        raise RuntimeError("Local TradeAgent returned an unsafe saved-strategy filename")

    unique_names = sorted({name[:-3] for name in files})
    selected = unique_names[:_MAX_STUDIO_RETURNED_FILES]
    return {
        "source_route": _STUDIO_ROUTE,
        "strategies": selected,
        "observed_count": len(files),
        "returned_count": len(selected),
        "truncated": len(unique_names) > _MAX_STUDIO_RETURNED_FILES,
        "warning": (
            "Saved-file names only; catalog entries are not validated lifecycle, "
            "strategy performance, backtest results or authorization to trade."
        ),
    }


def create_server():
    """Register only the approved read-only tools with MCP Python SDK v2."""
    # Import lazily: the SDK is an opt-in dependency, never needed to boot TradeAgent.
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("TradeAgent Local Read-Only Research")
    server.tool()(get_ctrader_candles)
    server.tool()(list_studio_saved_strategies)
    return server


def main() -> None:
    """Run independently; do not add this process to TradeAgent startup."""
    create_server().run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8765,
        stateless_http=True,
        json_response=True,
    )


if __name__ == "__main__":
    main()
