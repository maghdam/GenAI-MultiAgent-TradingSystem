# backend/strategy.py
from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd

# --- Base interface ---------------------------------------------------------
class Strategy(ABC):
    def __init__(self, **params: Any):
        self.params = params

    @abstractmethod
    def extract_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        ...

    @abstractmethod
    def analyze(self, symbol: str, timeframe: str, df: pd.DataFrame, features: Dict[str, Any]) -> Dict[str, Any]:
        ...

# --- SMC adapter (reuses your existing code) --------------------------------
from .smc_features import build_feature_snapshot
from .llm_analyzer import analyze_chart_with_llm

class SMCStrategy(Strategy):
    def extract_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        return build_feature_snapshot(df)

    def analyze(self, symbol: str, timeframe: str, df: pd.DataFrame, features: Dict[str, Any]) -> Dict[str, Any]:
        # We reuse your LLM pipeline (fig building happens in app.py).
        # Here we just return a dict-compatible payload.
        # app.py will still call analyze_chart_with_llm to produce TradeDecision.
        # This method remains for future non-LLM/rule-based use if needed.
        return {"ok": True, "features": features}

def load_strategy(name: str, **params) -> Strategy:
    name = (name or "smc").lower()
    if name == "smc":
        return SMCStrategy(**params)
    # Add simple stubs later, e.g., RSI/MACD without new files.
    raise ValueError(f"Unknown strategy '{name}'")
