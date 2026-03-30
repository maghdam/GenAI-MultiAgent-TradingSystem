from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Dict

from backend.domain.models import EngineConfig, EngineRuntime, WatchlistItem
from backend.services.execution_engine import execute_paper_signal
from backend.services.market_data import MarketDataError, get_bars
from backend.services.paper_book import reconcile_position
from backend.services.reconciler import reconcile_open_positions, recover_runtime_state
from backend.storage.repositories import (
    add_analysis,
    get_open_position,
    list_paper_positions,
    load_bar_state,
    load_engine_config,
    load_runtime,
    log_incident,
    save_bar_state,
    save_runtime,
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
            self._mark_positions(item, float(df["close"].iloc[-1]))
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
        bar_state[key] = last_ts
        return True, result.action_taken

    def _mark_positions(self, item: WatchlistItem, last_price: float) -> None:
        position = get_open_position(item.symbol.upper(), item.timeframe.upper())
        if not position:
            return
        reconcile_position(position, last_price)

engine = V2Engine()
