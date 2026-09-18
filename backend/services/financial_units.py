from __future__ import annotations

from dataclasses import dataclass

from backend.domain.models import EngineConfig


@dataclass(frozen=True)
class DailyLossBudget:
    currency: str
    starting_equity_amount: float
    limit_percent: float
    limit_amount: float
    realized_pnl_amount: float
    realized_loss_percent: float
    breached: bool

    def as_details(self) -> dict[str, object]:
        return {
            "account_currency": self.currency,
            "starting_equity_amount": self.starting_equity_amount,
            "daily_loss_limit_percent": self.limit_percent,
            "daily_loss_limit_amount": self.limit_amount,
            "daily_realized_pnl_amount": self.realized_pnl_amount,
            "daily_realized_loss_percent": self.realized_loss_percent,
            "daily_loss_limit_breached": self.breached,
        }


def daily_loss_budget(config: EngineConfig, realized_pnl_amount: float) -> DailyLossBudget:
    """Convert the configured percentage into an account-currency loss budget.

    P&L values in the paper ledger are account-currency amounts. This function is
    the only supported conversion between the percentage setting and that ledger.
    """
    equity = float(config.paper_starting_equity_amount)
    limit_percent = abs(float(config.daily_loss_limit_pct))
    limit_amount = equity * (limit_percent / 100.0)
    realized = float(realized_pnl_amount)
    loss_percent = max(0.0, -realized / equity * 100.0)
    return DailyLossBudget(
        currency=config.account_currency.upper(),
        starting_equity_amount=equity,
        limit_percent=limit_percent,
        limit_amount=limit_amount,
        realized_pnl_amount=realized,
        realized_loss_percent=loss_percent,
        breached=realized <= -limit_amount,
    )


def risk_budget_amount(config: EngineConfig) -> float:
    return float(config.paper_starting_equity_amount) * (float(config.risk_per_trade_pct) / 100.0)
