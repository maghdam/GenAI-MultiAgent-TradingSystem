from backend.domain.models import BrokerAccountSnapshot, EngineConfig
from backend.services.financial_units import (
    daily_loss_budget,
    resolve_monetary_basis,
    risk_budget_amount,
)


def test_daily_loss_percentage_is_converted_to_currency_amount():
    config = EngineConfig(
        paper_starting_equity_amount=100_000,
        account_currency="USD",
        daily_loss_limit_pct=2.0,
    )

    before_limit = daily_loss_budget(config, -1_999.99)
    at_limit = daily_loss_budget(config, -2_000.00)

    assert before_limit.limit_amount == 2_000.0
    assert before_limit.breached is False
    assert at_limit.breached is True
    assert at_limit.realized_loss_percent == 2.0
    assert at_limit.as_details()["daily_realized_pnl_amount"] == -2_000.0


def test_risk_budget_has_explicit_account_currency_amount():
    config = EngineConfig(paper_starting_equity_amount=50_000, risk_per_trade_pct=0.5)
    assert risk_budget_amount(config) == 250.0


def test_demo_monetary_basis_uses_verified_broker_equity_and_currency():
    config = EngineConfig(
        paper_starting_equity_amount=100_000,
        account_currency="USD",
        risk_per_trade_pct=0.5,
        daily_loss_limit_pct=2.0,
    )
    snapshot = BrokerAccountSnapshot(
        account_id=123,
        currency="CHF",
        balance=20_000,
        unrealized_pnl=-250,
        equity=19_750,
        verified=True,
    )

    basis = resolve_monetary_basis(
        config,
        demo_execution=True,
        account_snapshot=snapshot,
    )

    assert basis.verified is True
    assert basis.source == "ctrader"
    assert basis.currency == "CHF"
    assert basis.equity_amount == 19_750
    assert risk_budget_amount(config, basis) == 98.75

    budget = daily_loss_budget(config, -395.0, basis)
    assert budget.currency == "CHF"
    assert budget.limit_amount == 395.0
    assert budget.breached is True


def test_demo_monetary_basis_fails_closed_without_verified_snapshot():
    config = EngineConfig()
    snapshot = BrokerAccountSnapshot(
        account_id=123,
        verified=False,
        notes=["snapshot unavailable"],
    )

    basis = resolve_monetary_basis(
        config,
        demo_execution=True,
        account_snapshot=snapshot,
    )

    assert basis.verified is False
    assert basis.source == "ctrader"
    assert "snapshot unavailable" in basis.reason
