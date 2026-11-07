"""Explicitly started localhost-only MCP research gateway for TradeAgent.

The tool surface intentionally exposes market-data reads only. It talks to the
existing TradeAgent process on loopback, not directly to a broker. There is no
order, account-switching, mutation, or live-execution tool in this module.

Do not put this service behind a public reverse proxy or tunnel before adding
server-side authentication/authorization and TLS to the MCP deployment.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

import httpx

_BACKEND_CANDLES_URL = "http://127.0.0.1:4000/api/market/candles"
_VALID_SYMBOL = re.compile(r"[A-Za-z0-9_.-]{2,32}\Z")
_ALLOWED_TIMEFRAMES = frozenset({"M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"})


def get_ctrader_candles(symbol: str, timeframe: str = "M5", num_bars: int = 100) -> dict[str, object]:
    """Read OHLC candles from the *running local* TradeAgent cTrader service.

    This is market-data research, not a trading signal. Bars can be stale or
    unavailable. Chart source/timezone must be independently verified.
    """
    if not isinstance(symbol, str) or not _VALID_SYMBOL.fullmatch(symbol.strip()):
        raise ValueError("symbol must be a 2–32 character broker ticker (letters, digits, ., _, -)")
    normalized_symbol = symbol.strip().upper()
    normalized_timeframe = str(timeframe).strip().upper()
    if normalized_timeframe not in _ALLOWED_TIMEFRAMES:
        raise ValueError("timeframe must be M1, M5, M15, M30, H1, H4, D1 or W1")
    if type(num_bars) is not int or not 2 <= num_bars <= 500:
        raise ValueError("num_bars must be an integer between 2 and 500")

    try:
        # No caller-controlled URL; no environment-provided HTTP proxy.
        with httpx.Client(timeout=20.0, trust_env=False) as client:
            response = client.get(
                _BACKEND_CANDLES_URL,
                params={
                    "symbol": normalized_symbol,
                    "timeframe": normalized_timeframe,
                    "num_bars": num_bars,
                    "live": "true",
                },
            )
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError("Local TradeAgent candle retrieval failed; verify the backend and cTrader connection") from exc

    candles = payload.get("candles") if isinstance(payload, dict) else None
    if not isinstance(candles, list) or len(candles) > num_bars:
        raise RuntimeError("Local TradeAgent returned an invalid candles payload")
    required = {"time", "open", "high", "low", "close"}
    if any(not isinstance(row, dict) or not required.issubset(row) for row in candles):
        raise RuntimeError("Local TradeAgent returned malformed candle rows")

    return {
        "provider": "ctrader_via_local_tradeagent",
        "symbol": normalized_symbol,
        "timeframe": normalized_timeframe,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "candles": candles,
        "warning": "Market-data observations only. No trade recommendation or broker order was generated.",
    }


def create_server():
    """Register only read operations; MCP remains opt-in and separate from app.py."""
    from mcp.server import MCPServer

    server = MCPServer("TradeAgent Read-Only Research")
    server.tool()(get_ctrader_candles)
    return server


def main() -> None:
    # Intentionally hardcoded to loopback. Do not publish this unauthenticated POC.
    create_server().run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8765,
        stateless_http=True,
        json_response=True,
    )


if __name__ == "__main__":
    main()
