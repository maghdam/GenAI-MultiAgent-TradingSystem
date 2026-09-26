from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Dict

from backend.domain.models import EngineConfig
from backend.services.broker import (
    close_demo_position,
    get_broker_account_snapshot,
    get_broker_status,
    get_instrument_spec,
    list_positions,
    sync_demo_position_targets,
)
from backend.services.market_data import MarketDataError, get_bars
from backend.services.financial_units import resolve_monetary_basis
from backend.services.broker_ledger import (
    close_local_position_after_broker_close,
    close_local_position_from_broker,
)
from backend.services.broker_position_match import match_broker_position
from backend.services.paper_book import apply_mark, reconcile_position
from backend.storage.repositories import (
    add_paper_event,
    add_trade_audit,
    close_paper_position,
    list_order_intents,
    list_paper_positions,
    load_engine_config,
    load_runtime,
    log_incident,
    open_paper_position,
    save_runtime,
    set_paper_position_broker_id,
)


def recover_demo_broker_trackers(config: EngineConfig | None = None) -> Dict[str, Any]:
    """Recover local tracking rows for cTrader demo positions opened by TradeAgent.

    Recovery is limited to broker positions whose position id can be traced to
    an executed TradeAgent order intent. Manually opened broker positions are
    never silently adopted.
    """
    cfg = config or load_engine_config(EngineConfig())
    if not cfg.demo_autotrade:
        return {"checked": 0, "recovered": 0, "untracked": 0, "ready": False}

    status = get_broker_status()
    if not status.execution_ready:
        return {"checked": 0, "recovered": 0, "untracked": 0, "ready": False}

    monetary_basis = resolve_monetary_basis(
        cfg,
        demo_execution=True,
        account_snapshot=get_broker_account_snapshot(),
    )
    if not monetary_basis.verified:
        return {
            "checked": 0,
            "recovered": 0,
            "attached": 0,
            "untracked": 0,
            "ready": False,
            "reason": monetary_basis.reason,
        }

    broker_rows = list_positions()
    local_rows = list_paper_positions("open")
    intents = list_order_intents(200)
    watch_by_symbol = {
        item.symbol.upper(): item
        for item in cfg.watchlist
        if item.enabled and item.trading_enabled
    }
    local_broker_ids = {
        int(p.broker_position_id)
        for p in local_rows
        if p.broker_position_id is not None and int(p.broker_position_id) > 0
    }

    recovered = 0
    attached = 0
    untracked = 0
    for row in broker_rows:
        symbol = str(row.get("symbol") or "").upper()
        direction = "long" if str(row.get("direction") or "").lower() == "buy" else "short"
        if symbol not in watch_by_symbol:
            continue
        broker_position_id = int(row.get("position_id") or 0)
        if broker_position_id <= 0:
            untracked += 1
            log_incident(
                "error",
                "ctrader_demo_broker_position_missing_id",
                f"Broker demo position for {symbol} has no valid position id.",
                {"broker_position": row, "automatic_adoption": False},
            )
            continue
        if broker_position_id in local_broker_ids:
            continue

        matching_intent = None
        for intent in intents:
            broker_order = intent.details.get("broker_order") if isinstance(intent.details, dict) else None
            if not isinstance(broker_order, dict):
                continue
            try:
                intent_position_id = int(broker_order.get("position_id") or 0)
            except (TypeError, ValueError):
                intent_position_id = 0
            if intent_position_id == broker_position_id:
                matching_intent = intent
                break

        if matching_intent is None:
            untracked += 1
            log_incident(
                "error",
                "ctrader_demo_untracked_broker_position",
                f"Broker demo position {broker_position_id} for {symbol} has no local tracker.",
                {"broker_position": row, "automatic_adoption": False},
            )
            continue

        legacy_matches = [
            p
            for p in local_rows
            if p.broker_position_id is None
            and p.symbol.upper() == symbol
            and p.direction == direction
            and p.timeframe.upper() == matching_intent.timeframe.upper()
            and p.strategy == matching_intent.strategy
        ]
        if len(legacy_matches) == 1:
            legacy = legacy_matches[0]
            updated = set_paper_position_broker_id(legacy.id, broker_position_id)
            local_rows = [updated if p.id == legacy.id else p for p in local_rows]
            local_broker_ids.add(broker_position_id)
            attached += 1
            add_trade_audit(
                event_type="ctrader_demo_tracker_identity_attached",
                symbol=symbol,
                timeframe=updated.timeframe,
                strategy=updated.strategy,
                position_id=updated.id,
                intent_id=matching_intent.id,
                summary="Attached persisted cTrader position id to a legacy local tracker.",
                details={"broker_position_id": broker_position_id},
            )
            continue

        conflicting_local = next(
            (
                p
                for p in local_rows
                if p.symbol.upper() == symbol
                and p.timeframe.upper() == matching_intent.timeframe.upper()
            ),
            None,
        )
        if conflicting_local is not None:
            untracked += 1
            log_incident(
                "error",
                "ctrader_demo_broker_position_identity_conflict",
                f"Broker demo position {broker_position_id} conflicts with an existing local tracker.",
                {
                    "broker_position": row,
                    "local_position_id": conflicting_local.id,
                    "local_broker_position_id": conflicting_local.broker_position_id,
                    "automatic_adoption": False,
                },
            )
            continue

        watch = watch_by_symbol[symbol]
        instrument = get_instrument_spec(symbol, monetary_basis.currency)
        created = open_paper_position(
            symbol=symbol,
            timeframe=matching_intent.timeframe or watch.timeframe,
            strategy=matching_intent.strategy or watch.strategy,
            direction=direction,
            quantity=float(row.get("volume_lots") or matching_intent.quantity or 0.0),
            entry_price=float(row.get("entry_price") or matching_intent.entry_price or 0.0),
            stop_loss=matching_intent.stop_loss,
            take_profit=matching_intent.take_profit,
            account_currency=monetary_basis.currency,
            cash_per_price_unit_per_lot=float(instrument.cash_per_price_unit_per_lot or 1.0),
            instrument_spec_source=instrument.source if instrument.valuation_ready else "unvalued_fallback",
            broker_position_id=broker_position_id,
        )
        local_rows.append(created)
        local_broker_ids.add(broker_position_id)
        recovered += 1
        add_trade_audit(
            event_type="ctrader_demo_tracker_recovered",
            symbol=symbol,
            timeframe=created.timeframe,
            strategy=created.strategy,
            position_id=created.id,
            intent_id=matching_intent.id,
            summary="Recovered local tracking position from an existing cTrader demo position.",
            details={"broker_position_id": broker_position_id},
        )
        # Market-aware protection repair happens in normal reconciliation,
        # where a fresh reference price is available. Recovery only restores
        # the missing local tracker and never moves stale protective targets.

    return {
        "checked": len(broker_rows),
        "recovered": recovered,
        "attached": attached,
        "untracked": untracked,
        "ready": True,
    }


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
    cfg = load_engine_config(EngineConfig())
    if cfg.demo_autotrade:
        try:
            recover_demo_broker_trackers(cfg)
        except Exception as exc:
            log_incident(
                "error",
                "ctrader_demo_tracker_recovery_failed",
                "Could not reconcile cTrader demo positions with local trackers.",
                {"reason": reason, "error": str(exc)},
            )
    positions = list_paper_positions("open")
    checked = 0
    closed = 0
    skipped = 0
    demo_ready = bool(cfg.demo_autotrade and get_broker_status().execution_ready)
    broker_rows = list_positions() if demo_ready else []

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
        watch = next(
            (
                item
                for item in cfg.watchlist
                if item.enabled
                and item.trading_enabled
                and item.symbol.upper() == position.symbol.upper()
                and item.timeframe.upper() == position.timeframe.upper()
            ),
            None,
        )
        demo_managed = bool(cfg.demo_autotrade and watch is not None)

        if demo_managed:
            apply_mark(position, last_price)
            if not demo_ready:
                skipped += 1
                continue
            match = match_broker_position(position, broker_rows)
            if match.status in {"id_mismatch", "legacy_ambiguous"}:
                skipped += 1
                log_incident(
                    "error",
                    "ctrader_demo_position_identity_ambiguous",
                    f"Could not safely identify broker position for {position.symbol}:{position.timeframe}.",
                    {
                        "reason": reason,
                        "position_id": position.id,
                        "broker_position_id": position.broker_position_id,
                        "match_status": match.status,
                        "candidates": match.candidates,
                    },
                )
                continue
            if not match.matched:
                close_local_position_from_broker(
                    position,
                    fallback_price=last_price,
                    fallback_reason="broker_position_closed",
                )
                closed += 1
                continue
            if match.status == "legacy_match":
                position = set_paper_position_broker_id(
                    position.id,
                    int(match.row.get("position_id") or 0),
                )
            broker_match = match.row
            try:
                protection = sync_demo_position_targets(
                    symbol=position.symbol,
                    direction=position.direction,
                    stop_loss=position.stop_loss,
                    take_profit=position.take_profit,
                    position_id=position.broker_position_id,
                    reference_price=last_price,
                )
                if protection.get("status") in {"exit_due_stop_loss", "exit_due_take_profit"}:
                    broker_close = close_demo_position(
                        symbol=position.symbol,
                        position_id=int(broker_match.get("position_id") or 0),
                        quantity_lots=float(broker_match.get("volume_lots") or position.quantity),
                    )
                    close_reason = (
                        "broker_stop_loss"
                        if protection.get("status") == "exit_due_stop_loss"
                        else "broker_take_profit"
                    )
                    close_local_position_after_broker_close(
                        position,
                        broker_close=broker_close,
                        fallback_price=last_price,
                        reason=close_reason,
                    )
                    closed += 1
                    add_trade_audit(
                        event_type="ctrader_demo_protective_exit",
                        symbol=position.symbol,
                        timeframe=position.timeframe,
                        strategy=position.strategy,
                        position_id=position.id,
                        summary="Closed cTrader demo position because the intended protective target was already crossed.",
                        details={
                            "protection": protection,
                            "broker_close": broker_close,
                            "reference_price": last_price,
                        },
                    )
            except Exception as exc:
                skipped += 1
                log_incident(
                    "error",
                    "ctrader_demo_protection_sync_failed",
                    f"Could not synchronize broker protection for {position.symbol}:{position.timeframe}",
                    {"reason": reason, "position_id": position.id, "error": str(exc)},
                )
            continue

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
