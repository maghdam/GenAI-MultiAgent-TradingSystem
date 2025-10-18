# backend/agent_controller.py
import asyncio
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple
from backend.agents.runner import run_symbol
from backend.agent_state import clear_task_status, update_task_status, clear_last_bar_ts

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "agent_config.json")

@dataclass
class AgentConfig:
    enabled: bool = False
    watchlist: List[Tuple[str, str]] = field(default_factory=list)  # [("EURUSD","M15"), ...]
    interval_sec: int = 60
    min_confidence: float = 0.65
    trading_mode: str = "paper"   # "paper" | "live"
    autotrade: bool = False       # only if trading_mode == "live"
    lot_size_lots: float = 0.10
    strategy: str = "smc"         # "smc" | "rsi"

class AgentController:
    def __init__(self):
        self.config = self._load_config()
        self._tasks: Dict[Tuple[str, str], asyncio.Task] = {}
        self._stops: Dict[Tuple[str, str], asyncio.Event] = {}
        self._lock = asyncio.Lock()

    def _load_config(self) -> AgentConfig:
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                return AgentConfig(**data)
        except (FileNotFoundError, json.JSONDecodeError):
            return AgentConfig()

    def _save_config(self, config: AgentConfig):
        with open(CONFIG_FILE, "w") as f:
            json.dump(asdict(config), f, indent=2)

    async def start_from_config(self):
        if self.config.enabled:
            await self.apply_config(self.config)

    def _auto_trade(self) -> bool:
        return self.config.trading_mode.lower() == "live" and self.config.autotrade

    async def _start_pair(self, pair: Tuple[str, str]):
        if pair in self._tasks: return
        stop = asyncio.Event()
        self._stops[pair] = stop
        sym, tf = pair
        clear_last_bar_ts(sym, tf)
        update_task_status(sym, tf, state="starting")
        t = asyncio.create_task(run_symbol(
            sym, tf,
            self.config.interval_sec,
            self.config.min_confidence,
            self._auto_trade(),
            self.config.lot_size_lots,
            self.config.strategy,
            stop
        ))
        self._tasks[pair] = t

    async def _stop_pair(self, pair: Tuple[str, str]):
        task = self._tasks.pop(pair, None)
        stop = self._stops.pop(pair, None)
        sym, tf = pair
        update_task_status(sym, tf, state="stopping")
        if stop: stop.set()
        if task:
            task.cancel()
            try: await task
            except Exception: pass
        clear_task_status(sym, tf)

    async def apply_config(self, new_cfg: AgentConfig):
        async with self._lock:
            old_cfg = self.config
            self.config = new_cfg
            self._save_config(new_cfg)

            if not new_cfg.enabled:
                await self.stop_all()
                return

            restart_needed = (
                new_cfg.trading_mode != old_cfg.trading_mode
                or new_cfg.autotrade != old_cfg.autotrade
                or new_cfg.interval_sec != old_cfg.interval_sec
                or new_cfg.min_confidence != old_cfg.min_confidence
                or new_cfg.lot_size_lots != old_cfg.lot_size_lots
                or new_cfg.strategy != old_cfg.strategy
            )

            desired = set(new_cfg.watchlist)
            running = set(self._tasks.keys())

            to_stop = running - desired
            to_start = desired - running
            to_restart = running & desired if restart_needed else set()

            tasks_to_stop = to_stop | to_restart
            tasks_to_start = to_start | to_restart

            if tasks_to_stop:
                await asyncio.gather(*(self._stop_pair(pair) for pair in tasks_to_stop))

            if tasks_to_start:
                for pair in tasks_to_start:
                    await self._start_pair(pair)

    async def stop_all(self):
        for pair in list(self._tasks.keys()):
            await self._stop_pair(pair)

    async def snapshot(self) -> Dict[str, object]:
        async with self._lock:
            cfg = self.config
            running = list(self._tasks.keys())
        return {
            "config": asdict(cfg),
            "running_pairs": running,
        }

controller = AgentController()
