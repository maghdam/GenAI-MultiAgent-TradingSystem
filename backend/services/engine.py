from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Dict

from backend.domain.models import EngineConfig, EngineRuntime, WatchlistItem
from backend.services.execution_engine import execute_paper_signal
from backend.services.broker import close_demo_position, get_broker_status, list_positions, sync_demo_position_targets
from backend.services.confluence_shadow import record_confluence_shadow
from backend.services.market_data import MarketDataError, get_bars
from backend.services.paper_book import apply_mark, reconcile_position
from backend.services.broker_ledger import (
    close_local_position_after_broker_close,
    close_local_position_from_broker,
)
from backend.services.broker_position_match import match_broker_position
from backend.services.reconciler import reconcile_open_positions, recover_demo_broker_trackers, recover_runtime_state
from backend.storage.repositories import (
    add_analysis,
    add_trade_audit,
    close_paper_position,
    get_open_position,
    list_paper_positions,
    load_bar_state,
    load_engine_config,
    load_runtime,
    log_incident,
    save_bar_state,
    save_runtime,
    set_paper_position_broker_id,
)
from backend.strategies.registry import get_strategy


class V2Engine:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._wake = asyncio.Event()

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        config = load_engine_config(EngineConfig())
        recover_runtime_state(config)
        reconcile_open_positions(reason="startup")
        self._stop = asyncio.Event()
        self._wake = asyncio.Event()
        self._task = asyncio.create_task(self._run_forever(), name="tradeagent-engine")

    async def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._task:
            await self._task
        runtime = load_runtime()
        runtime.loop_active = False
        runtime.running = False
        save_runtime(runtime)

    def wake(self) -> None:
        self._wake.set()

    async def run_once(self) -> str:
        config = load_engine_config(EngineConfig())
        runtime = load_runtime()
        runtime.tick_count += 1
        runtime.last_cycle_at = datetime.now(UTC).replace(tzinfo=None)
        runtime.running = config.enabled
        runtime.active_watchlist = [f"{item.symbol.upper()}:{item.timeframe.upper()}" for item in config.watchlist if item.enabled]
        runtime.last_error = None
        summary = "idle"

        if not config.enabled:
            runtime.loop_active = False
            runtime.last_cycle_summary = "engine disabled"
            save_runtime(runtime)
            return runtime.last_cycle_summary

        if config.kill_switch:
            runtime.loop_active = False
            runtime.last_cycle_summary = "kill switch active"
            save_runtime(runtime)
            return runtime.last_cycle_summary

        watchlist = [item for item in config.watchlist if item.enabled]
        if not watchlist:
            runtime.loop_active = False
            runtime.last_cycle_summary = "watchlist empty"
            save_runtime(runtime)
            return runtime.last_cycle_summary

        runtime.loop_active = True
        if config.demo_autotrade:
            try:
                recover_demo_broker_trackers(config)
            except Exception as exc:
                log_incident(
                    "error",
                    "ctrader_demo_tracker_recovery_failed",
                    "Could not reconcile cTrader demo positions with local trackers.",
                    {"error": str(exc)},
                )
        bar_state = load_bar_state()
        processed = 0
        actions = 0
        skipped_market_data = 0

        for item in watchlist:
            try:
                did_process, did_act = await self._scan_item(config, item, bar_state)
                processed += 1 if did_process else 0
                actions += 1 if did_act else 0
            except MarketDataError:
                skipped_market_data += 1
            except Exception as exc:
                runtime.last_error = f"{item.symbol}:{item.timeframe} {exc}"
                log_incident(
                    "error",
                    "scan_failure",
                    f"V2 scan failed for {item.symbol}:{item.timeframe}",
                    {"error": str(exc)},
                )

        save_bar_state(bar_state)
        summary = f"processed={processed} actions={actions} watchlist={len(watchlist)} market_skips={skipped_market_data}"
        runtime.last_cycle_summary = summary
        save_runtime(runtime)
        return summary

    async def _run_forever(self) -> None:
        while not self._stop.is_set():
            try:
                await self.run_once()
            except Exception as exc:
                runtime = load_runtime()
                runtime.last_error = str(exc)
                runtime.last_cycle_at = datetime.now(UTC).replace(tzinfo=None)
                runtime.last_cycle_summary = "loop error"
                save_runtime(runtime)
                log_incident("error", "engine_loop_crash", "V2 engine loop crashed", {"error": str(exc)})

            config = load_engine_config(EngineConfig())
            timeout = max(2, int(config.scan_interval_sec or 10))
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass

    async def _scan_item(self, config: EngineConfig, item: WatchlistItem, bar_state: Dict[str, int]) -> tuple[bool, bool]:
        df = get_bars(item.symbol.upper(), item.timeframe.upper(), 600)
        last_dt = df.index[-1].to_pydatetime()
        last_ts = int(last_dt.timestamp())
        key = f"{item.symbol.upper()}|{item.timeframe.upper()}"
        if bar_state.get(key) == last_ts:
            last_price = float(df["close"].iloc[-1])
            self._mark_positions(config, item, last_price)
            self._sync_existing_demo_protection(config, item, last_price)
            return False, False
        strategy = get_strategy(item.strategy)
        analysis = strategy.analyze(
            df=df,
            symbol=item.symbol.upper(),
            timeframe=item.timeframe.upper(),
            params=item.params,
        )
        analysis.context["engine_source"] = "auto_loop"
        add_analysis(analysis)
        try:
            record_confluence_shadow(analysis, config.min_confidence)
        except Exception as exc:
            log_incident(
                "warning",
                "confluence_shadow_failed",
                f"Shadow confluence failed for {analysis.symbol}:{analysis.timeframe}",
                {"error": str(exc), "execution_unchanged": True},
            )
        result = execute_paper_signal(
            config=config,
            watch_item=item,
            analysis=analysis,
            mark_price=float(df["close"].iloc[-1]),
            bar_timestamp=last_dt,
            bar_snapshot={
                "open": float(df["open"].iloc[-1]),
                "high": float(df["high"].iloc[-1]),
                "low": float(df["low"].iloc[-1]),
                "close": float(df["close"].iloc[-1]),
            },
        )
        if not result.retryable:
            bar_state[key] = last_ts
        return True, result.action_taken

    def _sync_existing_demo_protection(
        self,
        config: EngineConfig,
        item: WatchlistItem,
        last_price: float,
    ) -> None:
        if not (config.demo_autotrade and item.trading_enabled):
            return
        position = get_open_position(item.symbol.upper(), item.timeframe.upper())
        if not position:
            return

        match = match_broker_position(position, list_positions())
        if match.status in {"id_mismatch", "legacy_ambiguous"}:
            log_incident(
                "error",
                "ctrader_demo_position_identity_ambiguous",
                f"Could not safely identify broker position for {position.symbol}:{position.timeframe}.",
                {
                    "position_id": position.id,
                    "broker_position_id": position.broker_position_id,
                    "match_status": match.status,
                    "candidates": match.candidates,
                    "phase": "same_bar_maintenance",
                },
            )
            return
        if not match.matched:
            return
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
                reason = (
                    "broker_stop_loss"
                    if protection.get("status") == "exit_due_stop_loss"
                    else "broker_take_profit"
                )
                close_local_position_after_broker_close(
                    position,
                    broker_close=broker_close,
                    fallback_price=last_price,
                    reason=reason,
                )
                add_trade_audit(
                    event_type="ctrader_demo_protective_exit",
                    symbol=position.symbol,
                    timeframe=position.timeframe,
                    strategy=position.strategy,
                    position_id=position.id,
                    summary="Closed cTrader demo position after its intended protective target was crossed.",
                    details={"protection": protection, "broker_close": broker_close},
                )
                return
            if protection.get("status") == "synced":
                add_trade_audit(
                    event_type="ctrader_demo_protection_repaired",
                    symbol=position.symbol,
                    timeframe=position.timeframe,
                    strategy=position.strategy,
                    position_id=position.id,
                    summary="Repaired broker SL/TP during same-bar engine maintenance.",
                    details=protection,
                )
        except Exception as exc:
            log_incident(
                "error",
                "ctrader_demo_protection_sync_failed",
                f"Could not synchronize broker protection for {position.symbol}:{position.timeframe}",
                {"position_id": position.id, "error": str(exc), "phase": "same_bar_maintenance"},
            )

    def _mark_positions(self, config: EngineConfig, item: WatchlistItem, last_price: float) -> None:
        position = get_open_position(item.symbol.upper(), item.timeframe.upper())
        if not position:
            return

        demo_managed = bool(config.demo_autotrade and item.trading_enabled)
        if not demo_managed:
            reconcile_position(position, last_price)
            return

        apply_mark(position, last_price)
        status = get_broker_status()
        if not status.execution_ready:
            return
        match = match_broker_position(position, list_positions())
        if match.status in {"id_mismatch", "legacy_ambiguous"}:
            log_incident(
                "error",
                "ctrader_demo_position_identity_ambiguous",
                f"Could not safely identify broker position for {position.symbol}:{position.timeframe}.",
                {
                    "position_id": position.id,
                    "broker_position_id": position.broker_position_id,
                    "match_status": match.status,
                    "candidates": match.candidates,
                    "phase": "same_bar_mark",
                },
            )
            return
        if not match.matched:
            close_local_position_from_broker(
                position,
                fallback_price=last_price,
                fallback_reason="broker_position_closed",
            )
            return
        if match.status == "legacy_match":
            set_paper_position_broker_id(
                position.id,
                int(match.row.get("position_id") or 0),
            )

engine = V2Engine()
