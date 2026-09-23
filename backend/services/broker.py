from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from backend.adapters.ctrader import adapter
from backend.domain.models import BrokerStatus, InstrumentSpec, SymbolLimits


def get_broker_status() -> BrokerStatus:
    return adapter.get_status()


def list_positions() -> List[Dict[str, Any]]:
    return adapter.list_positions()


def list_symbols() -> List[str]:
    return adapter.list_symbols()


def get_symbol_limits(symbol: str) -> SymbolLimits:
    return adapter.get_symbol_limits(symbol)


def get_instrument_spec(symbol: str, account_currency: str = "USD") -> InstrumentSpec:
    return adapter.get_instrument_spec(symbol, account_currency)


def get_demo_symbol_execution_readiness(symbol: str) -> tuple[bool, str]:
    return adapter.demo_symbol_execution_readiness(symbol)


def sync_demo_position_targets(**kwargs) -> Dict[str, Any]:
    return adapter.sync_demo_position_targets(**kwargs)


def close_demo_position(**kwargs) -> Dict[str, Any]:
    return adapter.close_demo_position(**kwargs)


def get_closed_position_summary(
    position_id: int,
    *,
    closed_at_hint: datetime | None = None,
) -> Dict[str, Any] | None:
    return adapter.get_closed_position_summary(position_id, closed_at_hint=closed_at_hint)


def place_demo_market_order(**kwargs) -> Dict[str, Any]:
    return adapter.place_demo_market_order(**kwargs)
