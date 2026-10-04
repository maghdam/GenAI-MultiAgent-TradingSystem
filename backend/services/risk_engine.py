from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Dict, List

from backend.domain.models import EngineConfig, PaperPosition, StrategyAnalysis, WatchlistItem
from backend.storage.repositories import daily_realized_pnl, daily_trade_count, list_paper_positions
from backend.services.financial_units import MonetaryBasis, daily_loss_budget, resolve_monetary_basis
from backend.services.market_data import assess_market_bar_freshness
from backend.services.market_bar_validation import assess_market_bar_snapshot
from backend.services.strategy_lifecycle import paper_execution_gate


@dataclass
class RiskDecision:
    accepted: bool
    reasons: List[str] = field(default_factory=list)
    details: Dict[str, object] = field(default_factory=dict)
    intent_type: str = "skip"


def _validate_protective_levels(
    analysis: StrategyAnalysis,
    entry_reference: float | None,
    require_stops: bool,
) -> tuple[bool, Dict[str, object], str | None]:
    details: Dict[str, object] = {
        "entry_reference": entry_reference,
        "stop_loss": analysis.stop_loss,
        "take_profit": analysis.take_profit,
    }
    if entry_reference is None:
        return False, details, "No valid entry reference price was available for the signal."

    stop_loss = analysis.stop_loss
    take_profit = analysis.take_profit
    if stop_loss is None or take_profit is None:
        if require_stops:
            return False, details, "Stops are required and the strategy did not produce both stop and target."
        return True, details, None

    risk_distance = abs(float(entry_reference) - float(stop_loss))
    reward_distance = abs(float(take_profit) - float(entry_reference))
    details["risk_distance"] = risk_distance
    details["reward_distance"] = reward_distance
    details["implied_rr"] = (reward_distance / risk_distance) if risk_distance > 0 else None

    signal = analysis.signal
    if signal == "long":
        if not (float(stop_loss) < float(entry_reference) < float(take_profit)):
            return False, details, "Invalid protective levels for a long signal."
    elif signal == "short":
        if not (float(take_profit) < float(entry_reference) < float(stop_loss)):
            return False, details, "Invalid protective levels for a short signal."

    if risk_distance <= 0 or reward_distance <= 0:
        return False, details, "Protective levels must imply positive risk and reward distances."

    return True, details, None


def _utc_naive(value: datetime) -> datetime:
    return (
        value.astimezone(UTC).replace(tzinfo=None)
        if value.tzinfo is not None
        else value
    )


def _within_session(config: EngineConfig, now: datetime) -> bool:
    if not config.session_filter_enabled:
        return True
    start = int(config.session_start_hour_utc)
    end = int(config.session_end_hour_utc)
    hour = _utc_naive(now).hour
    if start == end:
        return True
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def _same_symbol_positions(open_positions: List[PaperPosition], symbol: str) -> List[PaperPosition]:
    return [position for position in open_positions if position.symbol.upper() == symbol.upper()]



def _in_symbol_cooldown(config: EngineConfig, open_positions: List[PaperPosition], symbol: str, timeframe: str) -> bool:
    if config.cooldown_minutes <= 0:
        return False
    cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=int(config.cooldown_minutes))
    for position in list_paper_positions():
        if position.symbol.upper() != symbol.upper() or position.timeframe.upper() != timeframe.upper():
            continue
        if position.closed_at and _utc_naive(position.closed_at) >= cutoff:
            return True
    return False


def evaluate_risk(
    *,
    config: EngineConfig,
    watch_item: WatchlistItem,
    analysis: StrategyAnalysis,
    existing_position: PaperPosition | None,
    mark_price: float | None,
    bar_timestamp: datetime | None,
    bar_snapshot: Dict[str, object] | None,
    source: str = "auto",
    monetary_basis: MonetaryBasis | None = None,
) -> RiskDecision:
    now = datetime.now(UTC).replace(tzinfo=None)
    open_positions = list_paper_positions("open")
    same_symbol_positions = _same_symbol_positions(open_positions, analysis.symbol)
    decision = RiskDecision(accepted=False, reasons=[], details={}, intent_type="skip")

    if analysis.signal == "no_trade":
        decision.reasons.append("Strategy returned no_trade.")
        return decision

    lifecycle_ok, lifecycle_details, lifecycle_error = paper_execution_gate(analysis.strategy)
    if lifecycle_details.get("governed"):
        decision.details["strategy_lifecycle"] = lifecycle_details
    if not lifecycle_ok:
        decision.reasons.append(lifecycle_error or "Strategy lifecycle gate rejected execution.")
        return decision

    if config.kill_switch:
        decision.reasons.append("Kill switch is active.")
        return decision

    if not _within_session(config, now):
        decision.reasons.append("Signal is outside the configured trading session.")
        decision.details["session"] = {
            "enabled": config.session_filter_enabled,
            "start_hour_utc": config.session_start_hour_utc,
            "end_hour_utc": config.session_end_hour_utc,
        }
        return decision

    if analysis.confidence < config.min_confidence:
        decision.reasons.append("Signal strength is below the configured minimum.")
        decision.details["confidence"] = analysis.confidence
        decision.details["min_confidence"] = config.min_confidence
        return decision

    if source != "manual" or bar_timestamp is not None:
        market_ok, market_details, market_error = assess_market_bar_freshness(
            analysis.timeframe,
            bar_timestamp,
            now=now,
        )
        decision.details.update(market_details)
        if not market_ok and market_error:
            decision.reasons.append(market_error)
            return decision

    if source != "manual" or bar_snapshot:
        bar_ok, bar_details, bar_error = assess_market_bar_snapshot(analysis.timeframe, bar_snapshot)
        decision.details.update(bar_details)
        if not bar_ok and bar_error:
            decision.reasons.append(bar_error)
            return decision

    levels_ok, level_details, level_error = _validate_protective_levels(
        analysis,
        analysis.entry_price if analysis.entry_price is not None else mark_price,
        config.require_stops,
    )
    decision.details.update(level_details)
    if not levels_ok and level_error:
        decision.reasons.append(level_error)
        return decision

    basis = monetary_basis or resolve_monetary_basis(config)
    decision.details.update(basis.as_details())
    if not basis.verified:
        decision.reasons.append("Account monetary basis is not verified.")
        return decision

    pnl_today = daily_realized_pnl()
    loss_budget = daily_loss_budget(config, pnl_today, basis)
    decision.details.update(loss_budget.as_details())
    if loss_budget.breached:
        decision.reasons.append("Daily loss cap reached.")
        return decision

    trades_today = daily_trade_count()
    if trades_today >= int(config.max_daily_trades):
        decision.reasons.append("Max daily trade count reached.")
        decision.details["daily_trade_count"] = trades_today
        return decision

    if existing_position and existing_position.direction == analysis.signal:
        decision.accepted = True
        decision.intent_type = "update"
        decision.reasons.append("Existing same-direction paper position will be updated.")
        decision.details["position_id"] = existing_position.id
        return decision

    if existing_position and existing_position.direction != analysis.signal:
        decision.accepted = True
        decision.intent_type = "close"
        decision.reasons.append("Existing position conflicts with the new signal and will be flipped.")
        decision.details["position_id"] = existing_position.id
        return decision

    if len(open_positions) >= int(config.max_open_positions):
        decision.reasons.append("Max open positions reached.")
        decision.details["open_positions"] = len(open_positions)
        return decision

    if len(same_symbol_positions) >= int(config.max_positions_per_symbol):
        decision.reasons.append("Max open positions per symbol reached.")
        decision.details["same_symbol_positions"] = len(same_symbol_positions)
        return decision

    if _in_symbol_cooldown(config, open_positions, analysis.symbol, analysis.timeframe):
        decision.reasons.append("Symbol is still inside cooldown from the last closed trade.")
        decision.details["cooldown_minutes"] = config.cooldown_minutes
        return decision

    if source != "manual":
        paper_enabled = bool(config.paper_autotrade)
        ctrader_enabled = bool(config.ctrader_autotrade and watch_item.trading_enabled)
        if not paper_enabled and not ctrader_enabled:
            decision.reasons.append(
                "Paper and cTrader autotrade are disabled."
                if not config.ctrader_autotrade
                else "Automatic cTrader execution is disabled for this symbol."
            )
            return decision

    decision.accepted = True
    decision.intent_type = "open"
    decision.reasons.append("Signal passed risk checks.")
    decision.details["watch_item"] = watch_item.model_dump(mode="json")
    return decision
