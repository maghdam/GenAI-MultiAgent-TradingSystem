from __future__ import annotations

from dataclasses import dataclass

from backend.domain.models import BrokerAccountSnapshot, EngineConfig


@dataclass(frozen=True)
class MonetaryBasis:
    currency: str
    equity_amount: float
    source: str
    verified: bool
    reason: str = ""

    def as_details(self) -> dict[str, object]:
        return {
            "monetary_basis_currency": self.currency,
            "monetary_basis_equity_amount": self.equity_amount,
            "monetary_basis_source": self.source,
            "monetary_basis_verified": self.verified,
            "monetary_basis_reason": self.reason,
        }


def resolve_monetary_basis(
    config: EngineConfig,
    *,
    demo_execution: bool = False,
    account_snapshot: BrokerAccountSnapshot | None = None,
) -> MonetaryBasis:
    if not demo_execution:
        return MonetaryBasis(
            currency=config.account_currency.upper(),
            equity_amount=float(config.paper_starting_equity_amount),
            source="paper_config",
            verified=True,
        )

    snapshot = account_snapshot
    if snapshot is None or not snapshot.verified:
        reason = (
            "; ".join(snapshot.notes)
            if snapshot is not None and snapshot.notes
            else "Verified cTrader account monetary snapshot is unavailable."
        )
        return MonetaryBasis(
            currency=(snapshot.currency or "").upper() if snapshot is not None else "",
            equity_amount=float(snapshot.equity or 0.0) if snapshot is not None else 0.0,
            source="ctrader",
            verified=False,
            reason=reason,
        )

    currency = str(snapshot.currency or "").strip().upper()
    equity = float(snapshot.equity or 0.0)
    if len(currency) != 3 or equity <= 0:
        return MonetaryBasis(
            currency=currency,
            equity_amount=equity,
            source="ctrader",
            verified=False,
            reason="cTrader monetary snapshot did not include a valid currency and positive equity.",
        )

    return MonetaryBasis(
        currency=currency,
        equity_amount=equity,
        source="ctrader",
        verified=True,
    )


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


def daily_loss_budget(
    config: EngineConfig,
    realized_pnl_amount: float,
    monetary_basis: MonetaryBasis | None = None,
) -> DailyLossBudget:
    """Convert the configured percentage into an account-currency loss budget.

    P&L values in the paper ledger are account-currency amounts. This function is
    the only supported conversion between the percentage setting and that ledger.
    """
    basis = monetary_basis or resolve_monetary_basis(config)
    equity = float(basis.equity_amount)
    if equity <= 0:
        raise ValueError("Daily loss budget requires positive account equity.")
    limit_percent = abs(float(config.daily_loss_limit_pct))
    limit_amount = equity * (limit_percent / 100.0)
    realized = float(realized_pnl_amount)
    loss_percent = max(0.0, -realized / equity * 100.0)
    return DailyLossBudget(
        currency=basis.currency.upper(),
        starting_equity_amount=equity,
        limit_percent=limit_percent,
        limit_amount=limit_amount,
        realized_pnl_amount=realized,
        realized_loss_percent=loss_percent,
        breached=realized <= -limit_amount,
    )


def risk_budget_amount(config: EngineConfig, monetary_basis: MonetaryBasis | None = None) -> float:
    basis = monetary_basis or resolve_monetary_basis(config)
    return float(basis.equity_amount) * (float(config.risk_per_trade_pct) / 100.0)
