from __future__ import annotations

from typing import List

from backend.domain.models import EngineConfig, ReadinessCheck
from backend.services.broker import get_broker_status
from backend.services.runtime_state import market_data_dependency_state
from backend.storage.repositories import daily_realized_pnl
from backend.services.financial_units import daily_loss_budget, resolve_monetary_basis
from backend.services.live_trading_guard import is_live_trading_armed


def build_readiness(config: EngineConfig) -> List[ReadinessCheck]:
    broker = get_broker_status()
    realized_today = daily_realized_pnl()
    probe_symbol = ""
    probe_timeframe = ""
    if config.watchlist:
        first_enabled = next((item for item in config.watchlist if item.enabled), None)
        if first_enabled:
            probe_symbol = first_enabled.symbol.upper()
            probe_timeframe = first_enabled.timeframe.upper()
    probe_symbol = probe_symbol or config.default_symbol.upper()
    probe_timeframe = probe_timeframe or config.default_timeframe.upper()

    market_ok = market_data_dependency_state.last_success
    if market_ok is None:
        market_detail = f"Market data has not been probed yet for {probe_symbol}:{probe_timeframe}."
    else:
        market_detail = market_data_dependency_state.last_reason or f"Last market-data probe for {probe_symbol}:{probe_timeframe}."
    monetary_basis = resolve_monetary_basis(
        config,
        ctrader_execution=bool(config.ctrader_autotrade),
        account_snapshot=broker.account_snapshot,
    )
    loss_budget = daily_loss_budget(config, realized_today, monetary_basis) if monetary_basis.verified else None
    live_arm_required = bool(config.ctrader_autotrade and broker.account_type == "live")
    live_arm_ok = (
        not live_arm_required
        or (
            broker.account_id is not None
            and is_live_trading_armed(broker.account_id)
        )
    )
    checks = [
        ReadinessCheck(
            name="engine_enabled",
            ok=config.enabled,
            detail="Paper engine enabled." if config.enabled else "Paper engine disabled.",
        ),
        ReadinessCheck(
            name="kill_switch",
            ok=not config.kill_switch,
            detail="Kill switch is off." if not config.kill_switch else "Kill switch is active; cTrader execution is blocked.",
        ),
        ReadinessCheck(
            name="broker_transport",
            ok=broker.connected,
            detail="cTrader transport connected." if broker.connected else "cTrader transport disconnected.",
        ),
        ReadinessCheck(
            name="ctrader_execution",
            ok=not config.ctrader_autotrade or broker.execution_ready,
            detail=(
                f"cTrader {broker.account_type} account and monetary risk basis are verified for execution."
                if broker.execution_ready
                else (
                    "cTrader execution is off."
                    if not config.ctrader_autotrade
                    else (
                        "cTrader execution is blocked until the selected account is authenticated and verified."
                        if not broker.account_verified
                        else "cTrader execution is blocked until cTrader provides a verified account currency and positive equity."
                    )
                )
            ),
        ),
        ReadinessCheck(
            name="live_trading_arm",
            ok=live_arm_ok,
            detail=(
                f"Live Trading is armed for account {broker.account_id}."
                if live_arm_required and live_arm_ok
                else (
                    "Live Trading is disarmed; new Live entries are blocked."
                    if live_arm_required
                    else "Live Trading arming is not required for the active Demo account."
                )
            ),
        ),
        ReadinessCheck(
            name="symbol_metadata",
            ok=broker.symbols_loaded > 0,
            detail=f"{broker.symbols_loaded} symbols loaded." if broker.symbols_loaded > 0 else "No symbols loaded from broker.",
        ),
        ReadinessCheck(
            name="account_monetary_basis",
            ok=monetary_basis.verified,
            detail=(
                f"Risk basis = {monetary_basis.equity_amount:.2f} {monetary_basis.currency} from {monetary_basis.source}."
                if monetary_basis.verified
                else (monetary_basis.reason or "Account monetary basis is unavailable.")
            ),
        ),
        ReadinessCheck(
            name="market_data_feed",
            ok=bool(market_ok),
            detail=market_detail,
        ),
        ReadinessCheck(
            name="daily_loss_limit",
            ok=bool(loss_budget is not None and not loss_budget.breached),
            detail=(
                (
                    f"Daily realized P&L = {loss_budget.realized_pnl_amount:.2f} {loss_budget.currency}; "
                    f"loss limit = {loss_budget.limit_amount:.2f} {loss_budget.currency} "
                    f"({loss_budget.limit_percent:.2f}% of {loss_budget.starting_equity_amount:.2f})."
                )
                if loss_budget is not None
                else "Daily loss limit cannot be verified until the account monetary basis is available."
            ),
        ),
    ]
    return checks
