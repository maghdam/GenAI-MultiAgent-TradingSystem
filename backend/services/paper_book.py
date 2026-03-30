from __future__ import annotations

from typing import Optional

from backend.domain.models import PaperPosition
from backend.storage.repositories import close_paper_position, update_paper_position_mark


def unrealized_pnl(position: PaperPosition, last_price: float) -> float:
    move = (last_price - position.entry_price) if position.direction == "long" else (position.entry_price - last_price)
    return move * position.quantity


def apply_mark(position: PaperPosition, last_price: float) -> None:
    update_paper_position_mark(position.id, last_price, unrealized_pnl(position, last_price))


def exit_reason(direction: str, last_price: float, stop_loss: float | None, take_profit: float | None) -> tuple[Optional[str], Optional[float]]:
    if direction == "long":
        if stop_loss is not None and last_price <= stop_loss:
            return "stop_loss", stop_loss
        if take_profit is not None and last_price >= take_profit:
            return "take_profit", take_profit
    else:
        if stop_loss is not None and last_price >= stop_loss:
            return "stop_loss", stop_loss
        if take_profit is not None and last_price <= take_profit:
            return "take_profit", take_profit
    return None, None


def reconcile_position(position: PaperPosition, last_price: float) -> PaperPosition | None:
    apply_mark(position, last_price)
    reason, fill_price = exit_reason(position.direction, last_price, position.stop_loss, position.take_profit)
    if reason is not None and fill_price is not None:
        return close_paper_position(position.id, fill_price, reason)
    return None
