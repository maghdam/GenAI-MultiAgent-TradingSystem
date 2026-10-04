from __future__ import annotations

from datetime import datetime
from time import perf_counter
from typing import Any, Callable, Dict, List, TypeVar

import backend.ctrader_client as ctd

from backend.adapters.ctrader import (
    DemoCloseOutcomeAmbiguous,
    DemoCloseRejected,
    DemoOrderAcknowledgementTimeout,
    DemoProtectionSyncFailure,
    adapter,
)
from backend.domain.models import BrokerAccountSnapshot, BrokerStatus, CTraderAccount, InstrumentSpec, SymbolLimits
from backend.services.latency_observability import (
    record_broker_latency,
    record_broker_unavailable,
)


_T = TypeVar("_T")


def _execution_ready() -> bool:
    """Classify availability from local connection/auth/demo flags only."""
    try:
        return bool(
            ctd.is_connected()
            and ctd.is_authorized()
            and ctd.is_demo_account_confirmed()
        )
    except Exception:
        return False


def _timed_broker_call(
    operation: str,
    call: Callable[[], _T],
    *,
    available: Callable[[_T], bool] | None = None,
) -> _T:
    started = perf_counter()
    try:
        result = call()
    except Exception as exc:
        try:
            record_broker_unavailable(
                operation,
                f"{type(exc).__name__}: {exc}",
            )
        except Exception:
            pass
        raise

    duration_ms = max(0.0, (perf_counter() - started) * 1000.0)
    try:
        is_available = available(result) if available is not None else _execution_ready()
        if is_available:
            record_broker_latency(operation, duration_ms)
        else:
            record_broker_unavailable(
                operation,
                "Broker-facing service call completed without confirmed demo execution availability.",
            )
    except Exception:
        # Observability must never change broker-service behavior.
        pass
    return result


def get_broker_status() -> BrokerStatus:
    return adapter.get_status()


def list_accounts() -> List[CTraderAccount]:
    return adapter.list_accounts()


def switch_account(account_id: int, account_type: str) -> Dict[str, Any]:
    return adapter.switch_account(account_id, account_type)


def get_broker_account_snapshot(*, force: bool = False) -> BrokerAccountSnapshot:
    return _timed_broker_call(
        "get_broker_account_snapshot",
        lambda: adapter.get_account_snapshot(force=force),
        available=lambda snapshot: bool(snapshot.verified),
    )


def list_positions() -> List[Dict[str, Any]]:
    return _timed_broker_call(
        "list_positions",
        adapter.list_positions,
    )


def list_symbols() -> List[str]:
    return adapter.list_symbols()


def get_symbol_limits(symbol: str) -> SymbolLimits:
    return adapter.get_symbol_limits(symbol)


def get_instrument_spec(symbol: str, account_currency: str = "USD") -> InstrumentSpec:
    return adapter.get_instrument_spec(symbol, account_currency)


def get_demo_symbol_execution_readiness(symbol: str) -> tuple[bool, str]:
    return adapter.demo_symbol_execution_readiness(symbol)


def sync_demo_position_targets(**kwargs) -> Dict[str, Any]:
    return _timed_broker_call(
        "sync_demo_position_targets",
        lambda: adapter.sync_demo_position_targets(**kwargs),
    )


def close_demo_position(**kwargs) -> Dict[str, Any]:
    return _timed_broker_call(
        "close_demo_position",
        lambda: adapter.close_demo_position(**kwargs),
    )


def get_position_close_deals(
    position_id: int,
    *,
    symbol: str | None = None,
    opened_at_hint: datetime | None = None,
) -> List[Dict[str, Any]]:
    return _timed_broker_call(
        "get_position_close_deals",
        lambda: adapter.get_position_close_deals(
            position_id,
            symbol=symbol,
            opened_at_hint=opened_at_hint,
        ),
    )


def get_closed_position_summary(
    position_id: int,
    *,
    closed_at_hint: datetime | None = None,
    symbol: str | None = None,
) -> Dict[str, Any] | None:
    return _timed_broker_call(
        "get_closed_position_summary",
        lambda: adapter.get_closed_position_summary(
            position_id,
            closed_at_hint=closed_at_hint,
            symbol=symbol,
        ),
    )


def place_demo_market_order(**kwargs) -> Dict[str, Any]:
    return _timed_broker_call(
        "place_demo_market_order",
        lambda: adapter.place_demo_market_order(**kwargs),
    )
