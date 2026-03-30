from __future__ import annotations

from typing import Dict, List

from backend.strategies.base import StrategyV2
from backend.strategies.deterministic import BreakoutStrategy, RsiReversalStrategy, SmaCrossStrategy


_STRATEGIES: Dict[str, StrategyV2] = {
    strategy.key: strategy
    for strategy in (
        SmaCrossStrategy(),
        RsiReversalStrategy(),
        BreakoutStrategy(),
    )
}


def list_strategies() -> List[StrategyV2]:
    return list(_STRATEGIES.values())


def get_strategy(key: str) -> StrategyV2:
    normalized = (key or "").strip().lower()
    if normalized not in _STRATEGIES:
        raise KeyError(f"Unknown V2 strategy '{key}'")
    return _STRATEGIES[normalized]
