import asyncio
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Tuple

from backend.agents.runner import run_symbol
from backend.data_fetcher import ALLOWED_TF
from backend.agent_state import clear_task_status, update_task_status, clear_last_bar_ts

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "agent_config.json")


@dataclass
class WatchlistItem:
    symbol: str
    timeframe: str
    lot_size: float

    def key(self) -> Tuple[str, str]:
        return (self.symbol.upper(), self.timeframe.upper())


def normalize_watchlist(raw: Iterable[Any] | None, default_lot: float) -> List[WatchlistItem]:
    normalized: List[WatchlistItem] = []
    seen: set[Tuple[str, str]] = set()
    fallback = default_lot if isinstance(default_lot, (int, float)) and default_lot > 0 else 0.01

    for item in raw or []:
        sym = tf = None
        lot: Any = None

        if isinstance(item, WatchlistItem):
            sym = item.symbol
            tf = item.timeframe
            lot = item.lot_size
        elif isinstance(item, dict):
            sym = item.get("symbol") or item.get("pair") or item.get("name")
            tf = item.get("timeframe") or item.get("tf")
            lot = item.get("lot_size") or item.get("lotSize") or item.get("volume")
        elif isinstance(item, (list, tuple)):
            if len(item) >= 2:
                sym, tf = item[0], item[1]
            if len(item) >= 3:
                lot = item[2]
        elif isinstance(item, str):
            if ":" in item:
                sym, tf = item.split(":", 1)

        sym_u = (sym or "").strip().upper()
        tf_u = (tf or "").strip().upper()

        try:
            lot_val = float(lot) if lot is not None else float(fallback)
        except (TypeError, ValueError):
            lot_val = float(fallback)

        if lot_val <= 0:
            lot_val = float(fallback)
        lot_val = max(lot_val, 0.01)

        if not sym_u or not tf_u:
            continue

        # Validate timeframe against allowed set to avoid typos like 'M%'
        if tf_u not in ALLOWED_TF:
            continue

        key = (sym_u, tf_u)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(WatchlistItem(symbol=sym_u, timeframe=tf_u, lot_size=lot_val))

    return normalized


@dataclass
class AgentConfig:
    enabled: bool = False
    watchlist: List[WatchlistItem] = field(default_factory=list)
    interval_sec: int = 60
    min_confidence: float = 0.65
    trading_mode: str = "paper"
    autotrade: bool = False
    lot_size_lots: float = 0.01
    strategy: str = "smc"
    order_type: str = "MARKET"  # MARKET|LIMIT|STOP|AUTO

    def __post_init__(self) -> None:
        fallback = self.lot_size_lots if self.lot_size_lots > 0 else 0.01
        self.lot_size_lots = fallback
        self.watchlist = normalize_watchlist(self.watchlist, fallback)


class AgentController:
    def __init__(self):
        self.config = self._load_config()
        self._tasks: Dict[Tuple[str, str], asyncio.Task] = {}
        self._stops: Dict[Tuple[str, str], asyncio.Event] = {}
        self._lock = asyncio.Lock()
        self._pair_defaults: Dict[Tuple[str, str], float] = {
            item.key(): item.lot_size for item in (self.config.watchlist or [])
        }

    def _load_config(self) -> AgentConfig:
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                return AgentConfig(**data)
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return AgentConfig()

    def _save_config(self, config: AgentConfig) -> None:
        with open(CONFIG_FILE, "w") as f:
            json.dump(asdict(config), f, indent=2)

    async def start_from_config(self) -> None:
        if self.config.enabled:
            await self.apply_config(self.config)

    def _auto_trade(self) -> bool:
        return self.config.trading_mode.lower() == "live" and self.config.autotrade

    def _pair_lot_size(self, sym: str, tf: str) -> float:
        return self._pair_defaults.get((sym.upper(), tf.upper()), self.config.lot_size_lots)

    async def _start_pair(self, pair: Tuple[str, str]) -> None:
        if pair in self._tasks:
            return
        stop = asyncio.Event()
        self._stops[pair] = stop
        sym, tf = pair
        lot_size = self._pair_lot_size(sym, tf)
        print(f"[AGENT] starting {sym}/{tf} lot_size={lot_size}")
        clear_last_bar_ts(sym, tf)
        update_task_status(sym, tf, state="starting")
        task = asyncio.create_task(
            run_symbol(
                sym,
                tf,
                self.config.interval_sec,
                self.config.min_confidence,
                self._auto_trade(),
                lot_size,
                self.config.strategy,
                self.config.order_type,
                stop,
            )
        )
        self._tasks[pair] = task

    async def _stop_pair(self, pair: Tuple[str, str]) -> None:
        task = self._tasks.pop(pair, None)
        stop = self._stops.pop(pair, None)
        sym, tf = pair
        update_task_status(sym, tf, state="stopping")
        if stop:
            stop.set()
        if task:
            task.cancel()
            try:
                await task
            except Exception:
                pass
        clear_task_status(sym, tf)

    async def apply_config(self, new_cfg: AgentConfig) -> None:
        async with self._lock:
            old_cfg = self.config

            new_payload = asdict(new_cfg)
            normalized_cfg = AgentConfig(**new_payload)

            self.config = AgentConfig(
                enabled=normalized_cfg.enabled,
                watchlist=normalized_cfg.watchlist,
                interval_sec=normalized_cfg.interval_sec,
                min_confidence=normalized_cfg.min_confidence,
                trading_mode=normalized_cfg.trading_mode,
                autotrade=normalized_cfg.autotrade,
                lot_size_lots=normalized_cfg.lot_size_lots,
                strategy=normalized_cfg.strategy,
                order_type=(getattr(normalized_cfg, 'order_type', 'MARKET') or 'MARKET').upper(),
            )
            print("[AGENT] applied config:", self.config)
            self._pair_defaults = {item.key(): item.lot_size for item in self.config.watchlist}
            self._save_config(self.config)

            if not new_cfg.enabled:
                await self.stop_all()
                return

            restart_needed = (
                normalized_cfg.trading_mode != old_cfg.trading_mode
                or normalized_cfg.autotrade != old_cfg.autotrade
                or normalized_cfg.interval_sec != old_cfg.interval_sec
                or normalized_cfg.min_confidence != old_cfg.min_confidence
                or normalized_cfg.lot_size_lots != old_cfg.lot_size_lots
                or normalized_cfg.strategy != old_cfg.strategy
            )

            desired = {(item.symbol, item.timeframe) for item in self.config.watchlist}
            running = set(self._tasks.keys())

            old_lot_map = {(item.symbol, item.timeframe): item.lot_size for item in old_cfg.watchlist or []}
            new_lot_map = {(item.symbol, item.timeframe): item.lot_size for item in self.config.watchlist}
            lot_change_pairs = {
                pair
                for pair, lot in new_lot_map.items()
                if pair in old_lot_map and abs(lot - old_lot_map.get(pair, lot)) > 1e-9
            }

            to_stop = running - desired
            to_start = desired - running
            common_pairs = running & desired
            to_restart = common_pairs if restart_needed else (common_pairs & lot_change_pairs)

            tasks_to_stop = to_stop | to_restart
            tasks_to_start = to_start | to_restart

            if tasks_to_stop:
                await asyncio.gather(*(self._stop_pair(pair) for pair in tasks_to_stop))

            if tasks_to_start:
                for pair in tasks_to_start:
                    await self._start_pair(pair)

    async def stop_all(self) -> None:
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
