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
    strategy: str | None = None
    risk_enabled: bool | None = None
    risk_mode: str | None = None
    atr_len: int | None = None
    atr_mult: float | None = None
    rr: float | None = None
    swing_lookback: int | None = None
    tick_pct: float | None = None

    def key(self) -> Tuple[str, str]:
        return (self.symbol.upper(), self.timeframe.upper())


def normalize_watchlist(raw: Iterable[Any] | None, default_lot: float) -> List[WatchlistItem]:
    normalized: List[WatchlistItem] = []
    seen: set[Tuple[str, str]] = set()
    fallback = default_lot if isinstance(default_lot, (int, float)) and default_lot > 0 else 0.01

    for item in raw or []:
        sym = tf = strat = None
        lot: Any = None
        risk_enabled = None
        risk_mode = None
        atr_len = None
        atr_mult = None
        rr = None
        swing_lookback = None
        tick_pct = None

        if isinstance(item, WatchlistItem):
            sym = item.symbol
            tf = item.timeframe
            lot = item.lot_size
            strat = item.strategy
            risk_enabled = getattr(item, "risk_enabled", None)
            risk_mode = getattr(item, "risk_mode", None)
            atr_len = getattr(item, "atr_len", None)
            atr_mult = getattr(item, "atr_mult", None)
            rr = getattr(item, "rr", None)
            swing_lookback = getattr(item, "swing_lookback", None)
            tick_pct = getattr(item, "tick_pct", None)
        elif isinstance(item, dict):
            sym = item.get("symbol") or item.get("pair") or item.get("name")
            tf = item.get("timeframe") or item.get("tf")
            lot = item.get("lot_size") or item.get("lotSize") or item.get("volume")
            strat = item.get("strategy") or item.get("strat")
            risk_enabled = item.get("risk_enabled")
            risk_mode = item.get("risk_mode")
            atr_len = item.get("atr_len")
            atr_mult = item.get("atr_mult")
            rr = item.get("rr")
            swing_lookback = item.get("swing_lookback")
            tick_pct = item.get("tick_pct")
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
        strat_u = (strat or "").strip().lower() or None
        risk_mode_u = (risk_mode or "").strip().lower() or None

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
        normalized.append(
            WatchlistItem(
                symbol=sym_u,
                timeframe=tf_u,
                lot_size=lot_val,
                strategy=strat_u,
                risk_enabled=bool(risk_enabled) if risk_enabled is not None else None,
                risk_mode=risk_mode_u,
                atr_len=int(atr_len) if atr_len is not None else None,
                atr_mult=float(atr_mult) if atr_mult is not None else None,
                rr=float(rr) if rr is not None else None,
                swing_lookback=int(swing_lookback) if swing_lookback is not None else None,
                tick_pct=float(tick_pct) if tick_pct is not None else None,
            )
        )

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
    llm_gate_enabled: bool = True
    llm_gate_threshold: int = 3
    # Risk defaults (global; per-symbol env overrides are also supported)
    risk_mode: str = "atr"      # atr|swing
    atr_len: int = 14
    atr_mult: float = 1.0
    rr: float = 2.0
    swing_lookback: int = 10
    tick_pct: float = 0.0005

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
        self._pair_strategies: Dict[Tuple[str, str], str | None] = {
            item.key(): item.strategy for item in (self.config.watchlist or [])
        }
        self._pair_risk: Dict[Tuple[str, str], Dict[str, object]] = {
            item.key(): {
                "enabled": item.risk_enabled,
                "mode": item.risk_mode,
                "atr_len": item.atr_len,
                "atr_mult": item.atr_mult,
                "rr": item.rr,
                "swing_lookback": item.swing_lookback,
                "tick_pct": item.tick_pct,
            }
            for item in (self.config.watchlist or [])
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

    def _pair_strategy(self, sym: str, tf: str) -> str:
        return (self._pair_strategies.get((sym.upper(), tf.upper())) or self.config.strategy)

    def _pair_risk_cfg(self, sym: str, tf: str) -> Dict[str, object]:
        return self._pair_risk.get((sym.upper(), tf.upper()), {})

    async def _start_pair(self, pair: Tuple[str, str]) -> None:
        if pair in self._tasks:
            return
        stop = asyncio.Event()
        self._stops[pair] = stop
        sym, tf = pair
        lot_size = self._pair_lot_size(sym, tf)
        strategy_name = self._pair_strategy(sym, tf)
        risk_cfg = self._pair_risk_cfg(sym, tf)
        print(f"[AGENT] starting {sym}/{tf} lot_size={lot_size} strategy={strategy_name}")
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
                strategy_name,
                self.config.order_type,
                risk_cfg,
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
                llm_gate_enabled=bool(getattr(normalized_cfg, 'llm_gate_enabled', True)),
                llm_gate_threshold=int(getattr(normalized_cfg, 'llm_gate_threshold', 3) or 3),
                risk_mode=str(getattr(normalized_cfg, 'risk_mode', 'atr') or 'atr'),
                atr_len=int(getattr(normalized_cfg, 'atr_len', 14) or 14),
                atr_mult=float(getattr(normalized_cfg, 'atr_mult', 1.0) or 1.0),
                rr=float(getattr(normalized_cfg, 'rr', 2.0) or 2.0),
                swing_lookback=int(getattr(normalized_cfg, 'swing_lookback', 10) or 10),
                tick_pct=float(getattr(normalized_cfg, 'tick_pct', 0.0005) or 0.0005),
            )
            print("[AGENT] applied config:", self.config)
            self._pair_defaults = {item.key(): item.lot_size for item in self.config.watchlist}
            self._pair_strategies = {item.key(): item.strategy for item in self.config.watchlist}
            self._pair_risk = {
                item.key(): {
                    "enabled": item.risk_enabled,
                    "mode": item.risk_mode,
                    "atr_len": item.atr_len,
                    "atr_mult": item.atr_mult,
                    "rr": item.rr,
                    "swing_lookback": item.swing_lookback,
                    "tick_pct": item.tick_pct,
                }
                for item in self.config.watchlist
            }
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
            old_strat_map = {(item.symbol, item.timeframe): (item.strategy or old_cfg.strategy) for item in old_cfg.watchlist or []}
            new_strat_map = {(item.symbol, item.timeframe): (item.strategy or self.config.strategy) for item in self.config.watchlist}
            old_risk_map = {
                (item.symbol, item.timeframe): {
                    "enabled": item.risk_enabled,
                    "mode": item.risk_mode,
                    "atr_len": item.atr_len,
                    "atr_mult": item.atr_mult,
                    "rr": item.rr,
                    "swing_lookback": item.swing_lookback,
                    "tick_pct": item.tick_pct,
                }
                for item in old_cfg.watchlist or []
            }
            new_risk_map = {
                (item.symbol, item.timeframe): {
                    "enabled": item.risk_enabled,
                    "mode": item.risk_mode,
                    "atr_len": item.atr_len,
                    "atr_mult": item.atr_mult,
                    "rr": item.rr,
                    "swing_lookback": item.swing_lookback,
                    "tick_pct": item.tick_pct,
                }
                for item in self.config.watchlist
            }
            lot_change_pairs = {
                pair
                for pair, lot in new_lot_map.items()
                if pair in old_lot_map and abs(lot - old_lot_map.get(pair, lot)) > 1e-9
            }
            strat_change_pairs = {
                pair
                for pair, strat in new_strat_map.items()
                if pair in old_strat_map and (strat or "") != (old_strat_map.get(pair, "") or "")
            }
            risk_change_pairs = {
                pair
                for pair, risk in new_risk_map.items()
                if pair in old_risk_map and risk != old_risk_map.get(pair, {})
            }

            to_stop = running - desired
            to_start = desired - running
            common_pairs = running & desired
            to_restart = common_pairs if restart_needed else (common_pairs & (lot_change_pairs | strat_change_pairs | risk_change_pairs))

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
            running_detail = [
                {"symbol": sym, "timeframe": tf, "strategy": self._pair_strategy(sym, tf)}
                for sym, tf in running
            ]
        return {
            "config": asdict(cfg),
            "running_pairs": running,
            "running_pairs_detail": running_detail,
        }


controller = AgentController()
