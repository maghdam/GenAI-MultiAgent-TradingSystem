from __future__ import annotations

from typing import Any, Dict, List

from backend.adapters.ctrader import adapter
from backend.domain.models import BrokerStatus, SymbolLimits


def get_broker_status() -> BrokerStatus:
    return adapter.get_status()


def list_positions() -> List[Dict[str, Any]]:
    return adapter.list_positions()


def list_symbols() -> List[str]:
    return adapter.list_symbols()


def get_symbol_limits(symbol: str) -> SymbolLimits:
    return adapter.get_symbol_limits(symbol)
