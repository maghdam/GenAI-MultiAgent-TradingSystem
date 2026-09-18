from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from backend.domain.models import EngineConfig
from backend.services.market_data import get_market_data_status
from backend.services.runtime_state import market_data_dependency_state
from backend.storage.repositories import load_engine_config


def resolve_market_data_probe_target(config: EngineConfig) -> tuple[str, str]:
    """Use the first enabled watch item, then fall back to the configured defaults."""
    first_enabled = next((item for item in config.watchlist if item.enabled), None)
    symbol = first_enabled.symbol if first_enabled else config.default_symbol
    timeframe = first_enabled.timeframe if first_enabled else config.default_timeframe
    return symbol.strip().upper(), timeframe.strip().upper()


async def probe_market_data_once() -> dict[str, object]:
    """Probe cTrader without blocking FastAPI's event loop."""
    config = await asyncio.to_thread(load_engine_config, EngineConfig())
    symbol, timeframe = resolve_market_data_probe_target(config)
    try:
        return await asyncio.to_thread(get_market_data_status, symbol, timeframe)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        market_data_dependency_state.last_checked_at = datetime.now(UTC).replace(tzinfo=None)
        market_data_dependency_state.last_symbol = symbol
        market_data_dependency_state.last_timeframe = timeframe
        market_data_dependency_state.last_success = False
        market_data_dependency_state.market_data_ready = False
        market_data_dependency_state.last_reason = f"Market-data probe failed: {exc}"
        return {
            "ok": False,
            "symbol": symbol,
            "timeframe": timeframe,
            "reason": market_data_dependency_state.last_reason,
            **market_data_dependency_state.snapshot(),
        }


async def market_data_probe_loop(interval_sec: int = 60, retry_sec: int = 5) -> None:
    """Keep readiness current even while the trading engine is disabled."""
    while True:
        status = await probe_market_data_once()
        delay = interval_sec if status.get("ok") else retry_sec
        await asyncio.sleep(delay)
