from backend.domain.models import EngineConfig
from backend.services.financial_units import daily_loss_budget, risk_budget_amount


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
