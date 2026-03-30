from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Dict

from backend.domain.models import EngineConfig
from backend.services.market_data import MarketDataError, get_bars
from backend.services.paper_book import reconcile_position
from backend.storage.repositories import (
    add_paper_event,
    list_paper_positions,
    load_engine_config,
    load_runtime,
    log_incident,
    save_runtime,
)


def recover_runtime_state(config: EngineConfig | None = None) -> Dict[str, Any]:
    cfg = config or load_engine_config(EngineConfig())
    runtime = load_runtime()
    runtime.running = cfg.enabled
    runtime.loop_active = False
    runtime.active_watchlist = [f"{item.symbol.upper()}:{item.timeframe.upper()}" for item in cfg.watchlist if item.enabled]
    runtime.last_reconcile_at = datetime.now(UTC).replace(tzinfo=None)
    runtime.last_reconcile_summary = f"runtime_recovered watchlist={len(runtime.active_watchlist)} enabled={cfg.enabled}"
    save_runtime(runtime)
    add_paper_event(
        "runtime_recovered",
        "Recovered V2 runtime state from persisted config.",
        {"enabled": cfg.enabled, "watchlist": runtime.active_watchlist},
    )
    return {
        "active_watchlist": runtime.active_watchlist,
        "enabled": cfg.enabled,
    }


def reconcile_open_positions(reason: str = "manual") -> Dict[str, Any]:
    positions = list_paper_positions("open")
    checked = 0
    closed = 0
    skipped = 0

    for position in positions:
        try:
            df = get_bars(position.symbol, position.timeframe, 5)
            last_price = float(df["close"].iloc[-1])
        except MarketDataError as exc:
            skipped += 1
            log_incident(
                "warning",
                "reconcile_market_data_unavailable",
                f"Could not reconcile {position.symbol}:{position.timeframe}",
                {"reason": reason, "error": str(exc), "position_id": position.id},
            )
            continue

        checked += 1
        closed_position = reconcile_position(position, last_price)
        if closed_position is not None:
            closed += 1

    runtime = load_runtime()
    runtime.last_reconcile_at = datetime.now(UTC).replace(tzinfo=None)
    runtime.last_reconcile_summary = f"checked={checked} closed={closed} skipped={skipped} reason={reason}"
    save_runtime(runtime)
    add_paper_event(
        "positions_reconciled",
        "Reconciled open paper positions against current market prices.",
        {"checked": checked, "closed": closed, "skipped": skipped, "reason": reason},
    )
    return {
        "checked": checked,
        "closed": closed,
        "skipped": skipped,
        "reason": reason,
    }
