from __future__ import annotations

from threading import Lock


_lock = Lock()
_armed_account_id: int | None = None


def arm_live_trading(account_id: int) -> int:
    """Arm real-money entry submission for exactly one active Live account."""
    normalized = int(account_id)
    if normalized <= 0:
        raise ValueError("Live trading requires a valid cTrader account ID.")
    global _armed_account_id
    with _lock:
        _armed_account_id = normalized
    return normalized


def disarm_live_trading() -> None:
    """Fail closed for Live entry submission."""
    global _armed_account_id
    with _lock:
        _armed_account_id = None


def get_live_trading_armed_account_id() -> int | None:
    with _lock:
        return _armed_account_id


def is_live_trading_armed(account_id: int | None = None) -> bool:
    armed = get_live_trading_armed_account_id()
    if armed is None:
        return False
    if account_id is None:
        return True
    try:
        return armed == int(account_id)
    except (TypeError, ValueError):
        return False


def live_entry_block_reason(account_type: str | None, account_id: int | None) -> str | None:
    """Return a fail-closed reason only for new Live entries.

    Existing broker-position protection/reconciliation is intentionally not
    controlled by this gate; once a position exists, TradeAgent must remain
    able to protect, reconcile, and close it safely.
    """
    if str(account_type or "").lower() != "live":
        return None
    if account_id is None:
        return "Live cTrader entry submission is blocked because no active Live account is identified."
    if not is_live_trading_armed(account_id):
        return (
            "Live cTrader entry submission is disarmed. "
            "Arm Live Trading explicitly for the currently active Live account in System before placing real-money entries."
        )
    return None
