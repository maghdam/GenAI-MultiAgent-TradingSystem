"""Durable reservation-only foundation for Alpaca Paper equity bracket entries.

There is NO broker mutation, API/engine route, or automatic submission in
this module. A future manually supervised sender must atomically consume a
reservation once and reconcile ambiguous broker outcomes before retrying.
"""

from __future__ import annotations

from uuid import uuid4

from backend.services.alpaca_execution import _execution_ready_account
from backend.services.alpaca_paper import (
    AlpacaPaperError,
    build_equity_bracket_market_order,
    get_open_position,
    list_orders,
)
from backend.storage.repositories import reserve_alpaca_bracket_order_intent


def reserve_paper_equity_bracket(
    *,
    symbol: str,
    side: str,
    qty: int,
    reference_price: float,
    stop_loss: float,
    take_profit: float,
    timeframe: str,
    strategy: str,
    rationale: str,
) -> dict[str, object]:
    """Reserve a new bracket intent; return no action without confirmed safety.

    The stored broker request is prospective evidence only. This does not
    bypass Phase 10.14's autonomous Alpaca entry lockout or claim that the
    stop/target legs exist at the broker.
    """
    client_id = "ta-ap-brkt-" + uuid4().hex
    payload = build_equity_bracket_market_order(
        symbol=symbol, side=side, client_order_id=client_id, qty=qty,
        reference_price=reference_price, stop_loss=stop_loss,
        take_profit=take_profit,
    )
    account = _execution_ready_account()
    if payload["side"] == "sell" and not bool(account.get("shorting_enabled")):
        raise AlpacaPaperError("Alpaca Paper account does not permit short equity entries.")

    normalized = str(payload["symbol"])
    if get_open_position(normalized) is not None:
        return {
            "status": "blocked", "intent_id": None, "client_order_id": None,
            "summary": "An existing broker position blocks bracket reservation for this symbol.",
            "broker_mutation_permitted": False,
        }
    # Open broker orders that are not in this TradeAgent DB must also block
    # a reservation; no new order is submitted by this module.
    open_orders = list_orders(status="open", limit=500)
    if any(
        str(item.get("symbol") or "").upper() == normalized
        for item in open_orders
    ):
        return {
            "status": "blocked", "intent_id": None, "client_order_id": None,
            "summary": "An existing broker order blocks bracket reservation for this symbol.",
            "broker_mutation_permitted": False,
        }

    record = reserve_alpaca_bracket_order_intent(
        symbol=normalized,
        timeframe=timeframe,
        strategy=strategy,
        direction="long" if payload["side"] == "buy" else "short",
        quantity=int(payload["qty"]),
        entry_price=reference_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        rationale=rationale or "Prospective supervised Alpaca Paper equity bracket.",
        client_order_id=client_id,
        broker_request=payload,
    )
    if record is None:
        return {
            "status": "blocked", "intent_id": None, "client_order_id": None,
            "summary": "An existing accepted Alpaca Paper intent already reserves this symbol.",
            "broker_mutation_permitted": False,
        }
    return {
        "status": "reserved_not_submitted",
        "intent_id": record.id,
        "client_order_id": client_id,
        "summary": "Durable Paper equity bracket identity reserved; no broker order submitted.",
        "broker_mutation_permitted": False,
    }
