"""Strategy registry and base implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type
import inspect
import importlib.util
from pathlib import Path
import types
import textwrap

import pandas as pd
from backend.llm_analyzer import TradeDecision

_STRATEGY_REGISTRY: Dict[str, Type["Strategy"]] = {}
_GENERATED_LOADED: bool = False
_LAST_LOAD_ERRORS: list[dict[str, str]] = []


def register_strategy(name: str):
    """Decorator to register a strategy class by name."""

    def decorator(cls: Type["Strategy"]):
        key = name.lower()
        cls.strategy_name = key
        _STRATEGY_REGISTRY[key] = cls
        return cls

    return decorator


def available_strategies() -> List[str]:
    return sorted(_STRATEGY_REGISTRY.keys())


def get_strategy(name: str | None, **params: Any) -> "Strategy":
    key = (name or "smc").lower()
    cls = _STRATEGY_REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"Unknown strategy '{name}'")
    return cls(**params)


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    strategy_name: str = "base"
    requires_figure: bool = False

    def __init__(self, **params: Any):
        self.params = params

    @property
    def name(self) -> str:
        return self.strategy_name

    def build_figure(self, df: pd.DataFrame):
        """Optional candlestick figure for strategies that rely on chart context."""
        return None

    @abstractmethod
    async def analyze(
        self,
        *,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        indicators: List[str] | None = None,
        fig=None,
        **kwargs: Any,
    ) -> TradeDecision:
        ...


def _register_generated_module(name: str, module: types.ModuleType) -> None:
    """Wrap a simple module (with a `signals(df, ...)` function) as a Strategy.

    This allows saved files under `backend/strategies_generated/*.py` to appear in
    the dashboard's strategy selector without requiring users to subclass
    Strategy manually.
    """
    # Extended contract (backward compatible):
    # - Prefer `trade_decision(df, symbol=None, timeframe=None)` (can be async)
    #   which returns either a TradeDecision, a dict like TradeDecision.dict(),
    #   or a tuple (signal, sl, tp, confidence, reasons?).
    # - Fallback to classic `signals(df)` returning a numeric Series.
    td_fn = getattr(module, "trade_decision", None) or getattr(module, "analyze", None)
    signals_fn = getattr(module, "signals", None)
    if not callable(td_fn) and not callable(signals_fn):
        return

    class GeneratedStrategy(Strategy):
        strategy_name = name
        requires_figure = False

        async def analyze(self, *, df: pd.DataFrame, symbol: str, timeframe: str, indicators: List[str] | None = None, fig=None, **kwargs: Any) -> TradeDecision:
            if df is None or df.empty:
                raise ValueError("Generated strategy requires historical bars.")
            # Path 1: full trade decision function available
            if callable(td_fn):
                try:
                    if inspect.iscoroutinefunction(td_fn):
                        res = await td_fn(df, symbol=symbol, timeframe=timeframe, **kwargs)
                    else:
                        res = td_fn(df, symbol=symbol, timeframe=timeframe, **kwargs)
                except Exception as e:
                    raise ValueError(f"generated strategy error: {e}") from e

                # Normalize into TradeDecision
                if isinstance(res, TradeDecision):
                    return res
                if isinstance(res, dict):
                    return TradeDecision(
                        signal=str(res.get("signal", "no_trade")),
                        sl=res.get("sl"),
                        tp=res.get("tp"),
                        confidence=res.get("confidence"),
                        reasons=res.get("reasons") or [],
                        rationale=res.get("rationale") or "",
                    )
                if isinstance(res, (list, tuple)):
                    # Accept (signal, sl, tp, confidence, reasons)
                    sig = str(res[0]) if len(res) > 0 else "no_trade"
                    sl = res[1] if len(res) > 1 else None
                    tp = res[2] if len(res) > 2 else None
                    conf = res[3] if len(res) > 3 else None
                    reasons = res[4] if len(res) > 4 else []
                    return TradeDecision(signal=sig, sl=sl, tp=tp, confidence=conf, reasons=list(reasons) if reasons else [])
                raise ValueError("generated trade_decision must return TradeDecision, dict, or tuple")

            # Path 2: legacy numeric signals series
            try:
                # Allow strategies to accept optional kwargs as well
                try:
                    sig_series = signals_fn(df, **kwargs)
                except TypeError:
                    sig_series = signals_fn(df)
            except Exception as e:
                raise ValueError(f"generated strategy error: {e}") from e
            last = None
            try:
                last = float(sig_series.iloc[-1])
            except Exception:
                pass
            signal = "no_trade"
            if last is not None:
                if last > 0:
                    signal = "long"
                elif last < 0:
                    signal = "short"
            reasons = [f"generated: {name}", f"last={last}"]
            return TradeDecision(signal=signal, sl=None, tp=None, confidence=0.5 if signal != "no_trade" else 0.0, reasons=reasons)

    _STRATEGY_REGISTRY[name] = GeneratedStrategy


def load_generated_strategies(root: str | Path = "backend/strategies_generated") -> int:
    """Load user-saved strategy modules from disk and register them.

    A file is considered a strategy if it defines a callable `signals(df, ...)`.
    Returns the number of strategies registered.
    """
    global _GENERATED_LOADED
    count = 0
    errors: list[dict[str, str]] = []
    try:
        p = Path(root)
        if not p.exists():
            return 0
        for path in p.glob("*.py"):
            name = path.stem
            key = name.lower()
            if key in _STRATEGY_REGISTRY:
                continue
            # Normalize + compile from source, then register directly without importlib
            txt: str = ""
            norm: str = ""
            try:
                txt = path.read_text(encoding="utf-8")
                norm = textwrap.dedent(txt).lstrip("\n").replace("\r\n", "\n")
                # If still not compilable, aggressively strip leading spaces on all lines
                try:
                    compile(norm, str(path.name), 'exec')
                except SyntaxError:
                    norm2 = "\n".join([ln.lstrip() for ln in norm.splitlines()]) + "\n"
                    compile(norm2, str(path.name), 'exec')
                    norm = norm2
                # Persist normalized source back to disk for transparency
                if norm and norm != txt:
                    try:
                        path.write_text(norm, encoding="utf-8")
                    except Exception:
                        pass
                # Create a module from normalized source and register
                mod = types.ModuleType(f"strategies_generated.{name}")
                exec(norm, mod.__dict__)
                _register_generated_module(key, mod)
                count += 1
            except Exception as e:
                errors.append({"file": str(path), "error": str(e)})
                try:
                    print(f"[strategy-load] Failed to load {path}: {e}")
                except Exception:
                    pass
                continue
    finally:
        _GENERATED_LOADED = True
        # expose last errors for diagnostics
        try:
            global _LAST_LOAD_ERRORS
            _LAST_LOAD_ERRORS = errors
        except Exception:
            pass
    return count


def get_last_strategy_load_errors() -> list[dict[str, str]]:
    """Return errors captured during the last load_generated_strategies() call."""
    return list(_LAST_LOAD_ERRORS)


    # No built-in base strategies here; everything comes from generated modules.
