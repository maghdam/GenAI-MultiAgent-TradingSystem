from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from time import perf_counter
from typing import Dict

from backend.domain.models import EngineConfig, EngineRuntime, WatchlistItem
from backend.services.execution_engine import execute_paper_signal
from backend.services.broker import (
    DemoProtectionSyncFailure,
    get_broker_status,
    list_positions,
    sync_demo_position_targets,
)
from backend.services.confluence_shadow import record_confluence_shadow
from backend.services.market_data import MarketDataError, get_bars, record_market_bar_freshness
from backend.services.market_bar_validation import assess_market_frame
from backend.services.close_safety import attempt_verified_demo_close
from backend.services.protection_safety import (
    broker_protection_matches,
    fail_safe_close_unverified_demo_position,
)
from backend.services.paper_book import apply_mark, reconcile_position
from backend.services.runtime_state import market_data_dependency_state
from backend.services.broker_ledger import (
    close_local_position_from_broker,
    reconcile_open_demo_position_ledger,
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
        self._deferred_protection_positions: set[int] = set()
        self._protection_failsafe_pending_positions: set[int] = set()
        self._stale_market_items: set[str] = set()
        self._malformed_market_items: set[str] = set()

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

    async def restart(self) -> None:
        """Restart the runtime loop using the normal startup recovery path."""
        await self.stop()
        await self.start()

    def wake(self) -> None:
        self._wake.set()

    async def run_once(self) -> str:
        cycle_started = perf_counter()
        try:
            return await self._run_once_body()
        finally:
            self._record_cycle_latency(cycle_started)

    def _record_cycle_latency(self, cycle_started: float) -> None:
        """Persist non-fatal observability for the completed cycle attempt."""
        duration_ms = max(0.0, (perf_counter() - cycle_started) * 1000.0)
        completed_at = datetime.now(UTC).replace(tzinfo=None)
        try:
            runtime = load_runtime()
            runtime.last_cycle_duration_ms = duration_ms
            runtime.last_cycle_completed_at = completed_at
            save_runtime(runtime)
        except Exception:
            # Observability must never change trading/control flow.
            return

    async def _run_once_body(self) -> str:
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
        market_data_errors: list[str] = []

        for item in watchlist:
            try:
                did_process, did_act = await self._scan_item(config, item, bar_state)
                processed += 1 if did_process else 0
                actions += 1 if did_act else 0
            except MarketDataError as exc:
                skipped_market_data += 1
                market_data_errors.append(
                    f"{item.symbol.upper()}:{item.timeframe.upper()} {exc}"
                )
            except Exception as exc:
                runtime.last_error = f"{item.symbol}:{item.timeframe} {exc}"
                log_incident(
                    "error",
                    "scan_failure",
                    f"V2 scan failed for {item.symbol}:{item.timeframe}",
                    {"error": str(exc)},
                )

        if market_data_errors:
            market_data_dependency_state.last_success = False
            market_data_dependency_state.market_data_ready = False
            market_data_dependency_state.last_reason = "; ".join(market_data_errors)

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
        symbol = item.symbol.upper()
        timeframe = item.timeframe.upper()
        key = f"{symbol}|{timeframe}"
        try:
            df = get_bars(symbol, timeframe, 600)
        except MarketDataError as exc:
            if "malformed market data" in str(exc).lower():
                if key not in self._malformed_market_items:
                    log_incident(
                        "warning",
                        "market_data_malformed",
                        f"Deferred {symbol}:{timeframe} because market data is malformed.",
                        {
                            "reason": str(exc),
                            "retryable": True,
                            "bar_state_advanced": False,
                            "strategy_analysis_suppressed": True,
                            "order_intent_suppressed": True,
                            "position_mark_suppressed": True,
                            "broker_mutation_suppressed": True,
                        },
                    )
                    self._malformed_market_items.add(key)
            raise

        valid_frame, malformed_details, malformed_reason = assess_market_frame(
            timeframe,
            df,
        )
        if not valid_frame:
            reason_text = malformed_reason or "Malformed market data."
            if not reason_text.lower().startswith("malformed market data"):
                reason_text = f"Malformed market data: {reason_text}"
            market_data_dependency_state.last_success = False
            market_data_dependency_state.market_data_ready = False
            market_data_dependency_state.last_symbol = symbol
            market_data_dependency_state.last_timeframe = timeframe
            market_data_dependency_state.last_checked_at = datetime.now(UTC).replace(tzinfo=None)
            market_data_dependency_state.last_reason = f"{reason_text} {symbol}:{timeframe}"
            if key not in self._malformed_market_items:
                log_incident(
                    "warning",
                    "market_data_malformed",
                    f"Deferred {symbol}:{timeframe} because market data is malformed.",
                    {
                        **malformed_details,
                        "reason": reason_text,
                        "retryable": True,
                        "bar_state_advanced": False,
                        "strategy_analysis_suppressed": True,
                        "order_intent_suppressed": True,
                        "position_mark_suppressed": True,
                        "broker_mutation_suppressed": True,
                    },
                )
                self._malformed_market_items.add(key)
            raise MarketDataError(reason_text)

        self._malformed_market_items.discard(key)
        last_dt = df.index[-1].to_pydatetime()
        fresh, freshness, stale_reason = record_market_bar_freshness(
            symbol,
            timeframe,
            last_dt,
        )
        if not fresh:
            if key not in self._stale_market_items:
                log_incident(
                    "warning",
                    "market_data_stale",
                    f"Deferred {symbol}:{timeframe} because the latest market bar is stale.",
                    {
                        **freshness,
                        "reason": stale_reason,
                        "retryable": True,
                        "bar_state_advanced": False,
                        "strategy_analysis_suppressed": True,
                        "order_intent_suppressed": True,
                        "position_mark_suppressed": True,
                        "broker_mutation_suppressed": True,
                    },
                )
                self._stale_market_items.add(key)
            raise MarketDataError(
                stale_reason or "Latest market bar is not fresh."
            )

        self._stale_market_items.discard(key)
        last_ts = int(last_dt.timestamp())
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
        result_position_id = getattr(result, "position_id", None)
        if result_position_id is not None:
            if getattr(result, "status", None) == "protection_failsafe_pending":
                self._protection_failsafe_pending_positions.add(int(result_position_id))
            else:
                self._protection_failsafe_pending_positions.discard(int(result_position_id))
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

        broker = get_broker_status()
        broker_available = bool(
            broker.socket_connected
            and broker.account_authorized
            and broker.demo_account_confirmed
        )
        if not broker_available:
            position_key = int(position.id)
            if position_key not in self._deferred_protection_positions:
                if not broker.socket_connected:
                    reason = "cTrader transport is not connected."
                elif not broker.account_authorized:
                    reason = broker.auth_error or "cTrader account is not authorized."
                else:
                    reason = "Connected cTrader account is not confirmed as demo."
                log_incident(
                    "warning",
                    "ctrader_demo_protection_verification_deferred",
                    f"Deferred broker protection verification for {position.symbol}:{position.timeframe}.",
                    {
                        "position_id": position.id,
                        "broker_position_id": position.broker_position_id,
                        "phase": "same_bar_maintenance",
                        "reason": reason,
                        "socket_connected": broker.socket_connected,
                        "account_authorized": broker.account_authorized,
                        "demo_account_confirmed": broker.demo_account_confirmed,
                        "broker_protection_state": "unverified_broker_unavailable",
                        "local_exit_suppressed": True,
                        "broker_mutation_suppressed": True,
                    },
                )
                self._deferred_protection_positions.add(position_key)
            return

        self._deferred_protection_positions.discard(int(position.id))
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
        ledger_sync = reconcile_open_demo_position_ledger(position, broker_match)
        if ledger_sync.get("status") == "partial_close_synced":
            position = get_open_position(item.symbol.upper(), item.timeframe.upper()) or position
            add_trade_audit(
                event_type="ctrader_demo_partial_close_synced",
                symbol=position.symbol,
                timeframe=position.timeframe,
                strategy=position.strategy,
                position_id=position.id,
                summary="Synchronized broker partial close during same-bar maintenance.",
                details=ledger_sync,
            )
        elif ledger_sync.get("status") == "pending_deal_history":
            log_incident(
                "warning",
                "ctrader_demo_partial_close_history_pending",
                f"Broker volume decreased for {position.symbol}:{position.timeframe}, but close deal history is not complete yet.",
                {"position_id": position.id, **ledger_sync, "phase": "same_bar_maintenance"},
            )
        elif ledger_sync.get("status") in {"identity_mismatch", "volume_increase_mismatch", "unavailable"}:
            log_incident(
                "error",
                "ctrader_demo_volume_reconciliation_failed",
                f"Could not reconcile broker volume for {position.symbol}:{position.timeframe}.",
                {"position_id": position.id, **ledger_sync, "phase": "same_bar_maintenance"},
            )
            return

        position_key = int(position.id)
        if position_key in self._protection_failsafe_pending_positions:
            if broker_protection_matches(position, broker_match):
                self._protection_failsafe_pending_positions.discard(position_key)
                log_incident(
                    "warning",
                    "ctrader_demo_protection_recovered",
                    f"Broker protection is verified again for {position.symbol}:{position.timeframe}.",
                    {
                        "position_id": position.id,
                        "broker_position_id": position.broker_position_id,
                        "phase": "same_bar_maintenance",
                        "recovery": "broker_truth_verified",
                        "amend_suppressed": True,
                    },
                )
                add_trade_audit(
                    event_type="ctrader_demo_protection_recovered",
                    symbol=position.symbol,
                    timeframe=position.timeframe,
                    strategy=position.strategy,
                    position_id=position.id,
                    summary="Verified broker SL/TP after a prior protection fail-safe close failure.",
                    details={
                        "broker_position_id": position.broker_position_id,
                        "stop_loss": broker_match.get("stop_loss"),
                        "take_profit": broker_match.get("take_profit"),
                        "amend_suppressed": True,
                    },
                )
            return

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
                reason = (
                    "broker_stop_loss"
                    if protection.get("status") == "exit_due_stop_loss"
                    else "broker_take_profit"
                )
                close_result = attempt_verified_demo_close(
                    position,
                    fallback_price=last_price,
                    reason=reason,
                    phase="same_bar_protective_exit",
                    quantity_lots=float(broker_match.get("volume_lots") or position.quantity),
                )
                if close_result.get("closed"):
                    add_trade_audit(
                        event_type="ctrader_demo_protective_exit",
                        symbol=position.symbol,
                        timeframe=position.timeframe,
                        strategy=position.strategy,
                        position_id=position.id,
                        summary="Closed cTrader demo position after its intended protective target was crossed.",
                        details={"protection": protection, "close_result": close_result},
                    )
                return
            if protection.get("status") in {"synced", "already_synced"}:
                self._protection_failsafe_pending_positions.discard(int(position.id))
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
        except DemoProtectionSyncFailure as exc:
            failsafe = fail_safe_close_unverified_demo_position(
                position,
                broker_row=broker_match,
                fallback_price=last_price,
                protection_error=exc,
                phase="same_bar_maintenance",
            )
            if failsafe.get("closed"):
                self._protection_failsafe_pending_positions.discard(int(position.id))
            else:
                self._protection_failsafe_pending_positions.add(int(position.id))
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
            position = set_paper_position_broker_id(
                position.id,
                int(match.row.get("position_id") or 0),
            )

        ledger_sync = reconcile_open_demo_position_ledger(position, match.row)
        if ledger_sync.get("status") == "partial_close_synced":
            add_trade_audit(
                event_type="ctrader_demo_partial_close_synced",
                symbol=position.symbol,
                timeframe=position.timeframe,
                strategy=position.strategy,
                position_id=position.id,
                summary="Synchronized broker partial close during mark reconciliation.",
                details=ledger_sync,
            )
        elif ledger_sync.get("status") == "pending_deal_history":
            log_incident(
                "warning",
                "ctrader_demo_partial_close_history_pending",
                f"Broker volume decreased for {position.symbol}:{position.timeframe}, but close deal history is not complete yet.",
                {"position_id": position.id, **ledger_sync, "phase": "same_bar_mark"},
            )
        elif ledger_sync.get("status") in {"identity_mismatch", "volume_increase_mismatch", "unavailable"}:
            log_incident(
                "error",
                "ctrader_demo_volume_reconciliation_failed",
                f"Could not reconcile broker volume for {position.symbol}:{position.timeframe}.",
                {"position_id": position.id, **ledger_sync, "phase": "same_bar_mark"},
            )

engine = V2Engine()
