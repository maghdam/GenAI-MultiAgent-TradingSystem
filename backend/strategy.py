"""Strategy registry and base implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type
import importlib.util
from pathlib import Path
import types
import textwrap

import pandas as pd
import plotly.graph_objects as go

from backend.llm_analyzer import TradeDecision, analyze_chart_with_llm

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
    signals_fn = getattr(module, "signals", None)
    if not callable(signals_fn):
        return

    class GeneratedStrategy(Strategy):
        strategy_name = name
        requires_figure = False

        async def analyze(self, *, df: pd.DataFrame, symbol: str, timeframe: str, indicators: List[str] | None = None, fig=None, **kwargs: Any) -> TradeDecision:
            if df is None or df.empty:
                raise ValueError("Generated strategy requires historical bars.")
            try:
                sig_series = signals_fn(df)
            except Exception as e:
                raise ValueError(f"generated strategy error: {e}") from e
            # Expect a numeric series; derive last signal
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


def _build_candlestick(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Candles",
        )
    )
    return fig


@register_strategy("smc")
class SMCStrategy(Strategy):
    """Smart Money Concepts strategy using the LLM analyzer pipeline."""

    strategy_name = "smc"
    requires_figure = True

    def build_figure(self, df: pd.DataFrame):
        return _build_candlestick(df)

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
        model = kwargs.get("model")
        options = kwargs.get("options")
        ollama_url = kwargs.get("ollama_url")
        fig_to_use = fig or self.build_figure(df)
        return await analyze_chart_with_llm(
            fig=fig_to_use,
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            indicators=indicators or [],
            model=model,
            options=options,
            ollama_url=ollama_url,
        )


@register_strategy("rsi")
class RSIStrategy(Strategy):
    """Simple RSI divergence strategy (rule-based)."""

    strategy_name = "rsi"
    requires_figure = False

    def __init__(self, **params: Any):
        super().__init__(**params)
        self.length = int(self.params.get("length", 14))
        self.overbought = float(self.params.get("overbought", 70.0))
        self.oversold = float(self.params.get("oversold", 30.0))

    @staticmethod
    def _compute_rsi(series: pd.Series, length: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / length, min_periods=length).mean()
        avg_loss = loss.ewm(alpha=1 / length, min_periods=length).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(method="bfill")

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
        if df.empty:
            raise ValueError("RSI strategy requires historical bars.")

        closes = df["close"].astype(float)
        rsi_series = self._compute_rsi(closes, self.length)
        if rsi_series.empty or pd.isna(rsi_series.iloc[-1]):
            raise ValueError("Not enough data to compute RSI.")

        latest_rsi = float(rsi_series.iloc[-1])
        signal = "no_trade"
        reasons: List[str] = [f"RSI={latest_rsi:.2f}"]

        if latest_rsi >= self.overbought:
            signal = "short"
            reasons.append(f"RSI above overbought ({self.overbought})")
        elif latest_rsi <= self.oversold:
            signal = "long"
            reasons.append(f"RSI below oversold ({self.oversold})")
        else:
            reasons.append("RSI within neutral range")

        confidence = max(0.0, min(1.0, abs(latest_rsi - 50.0) / 50.0))
        if signal == "no_trade":
            confidence = 0.0

        return TradeDecision(
            signal=signal,
            sl=None,
            tp=None,
            confidence=confidence,
            reasons=reasons,
        )
