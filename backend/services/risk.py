from __future__ import annotations

from typing import List

from backend.domain.models import EngineConfig, ReadinessCheck
from backend.services.broker import get_broker_status
from backend.services.runtime_state import market_data_dependency_state
from backend.storage.repositories import daily_realized_pnl


def _daily_loss_cap(config: EngineConfig) -> float:
    return abs(float(config.daily_loss_limit_pct or 0.0))


def build_readiness(config: EngineConfig) -> List[ReadinessCheck]:
    broker = get_broker_status()
    realized_today = daily_realized_pnl()
    probe_symbol = ""
    probe_timeframe = ""
    if config.watchlist:
        first_enabled = next((item for item in config.watchlist if item.enabled), None)
        if first_enabled:
            probe_symbol = first_enabled.symbol.upper()
            probe_timeframe = first_enabled.timeframe.upper()
    probe_symbol = probe_symbol or config.default_symbol.upper()
    probe_timeframe = probe_timeframe or config.default_timeframe.upper()

    market_ok = market_data_dependency_state.last_success
    if market_ok is None:
        market_detail = f"Market data has not been probed yet for {probe_symbol}:{probe_timeframe}."
    else:
        market_detail = market_data_dependency_state.last_reason or f"Last market-data probe for {probe_symbol}:{probe_timeframe}."
    daily_loss_cap = _daily_loss_cap(config)
    checks = [
        ReadinessCheck(
            name="engine_enabled",
            ok=config.enabled,
            detail="Paper engine enabled." if config.enabled else "Paper engine disabled.",
        ),
        ReadinessCheck(
            name="kill_switch",
            ok=not config.kill_switch,
            detail="Kill switch is off." if not config.kill_switch else "Kill switch is active; live execution must remain disabled.",
        ),
        ReadinessCheck(
            name="broker_transport",
            ok=broker.connected,
            detail="cTrader transport connected." if broker.connected else "cTrader transport disconnected.",
        ),
        ReadinessCheck(
            name="symbol_metadata",
            ok=broker.symbols_loaded > 0,
            detail=f"{broker.symbols_loaded} symbols loaded." if broker.symbols_loaded > 0 else "No symbols loaded from broker.",
        ),
        ReadinessCheck(
            name="market_data_feed",
            ok=bool(market_ok),
            detail=market_detail,
        ),
        ReadinessCheck(
            name="live_permission",
            ok=not config.allow_live,
            detail="Paper-only mode enforced." if not config.allow_live else "Live mode requested; execution engine is not enabled in V2 yet.",
        ),
        ReadinessCheck(
            name="daily_loss_limit",
            ok=realized_today > -daily_loss_cap,
            detail=f"Daily realized PnL = {realized_today:.2f} / cap {daily_loss_cap:.2f}.",
        ),
    ]
    return checks
