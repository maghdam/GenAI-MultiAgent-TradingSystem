from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd

from backend.domain.models import StrategyAnalysis, StrategyInfo


class StrategyV2(ABC):
    key: str = "base"
    label: str = "Base"
    description: str = ""
    parameters: Dict[str, Any] = {}

    def info(self) -> StrategyInfo:
        return StrategyInfo(
            key=self.key,
            label=self.label,
            description=self.description,
            parameters=self.parameters,
        )

    @abstractmethod
    def analyze(self, df: pd.DataFrame, symbol: str, timeframe: str, params: Dict[str, Any]) -> StrategyAnalysis:
        raise NotImplementedError
