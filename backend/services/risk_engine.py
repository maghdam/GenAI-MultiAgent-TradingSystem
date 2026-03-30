from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Dict, List

from backend.domain.models import EngineConfig, PaperPosition, StrategyAnalysis, WatchlistItem
from backend.storage.repositories import daily_realized_pnl, daily_trade_count, list_paper_positions


@dataclass
class RiskDecision:
    accepted: bool
    reasons: List[str] = field(default_factory=list)
    details: Dict[str, object] = field(default_factory=dict)
    intent_type: str = "skip"


def _timeframe_delta(timeframe: str) -> timedelta:
    return {
        "M1": timedelta(minutes=1),
        "M5": timedelta(minutes=5),
        "M15": timedelta(minutes=15),
        "M30": timedelta(minutes=30),
        "H1": timedelta(hours=1),
        "H4": timedelta(hours=4),
        "D1": timedelta(days=1),
        "W1": timedelta(days=7),
    }.get((timeframe or "M5").upper(), timedelta(minutes=5))


def _validate_market_freshness(
    timeframe: str,
    bar_timestamp: datetime | None,
    now: datetime,
) -> tuple[bool, Dict[str, object], str | None]:
    details: Dict[str, object] = {
        "bar_timestamp": bar_timestamp.isoformat() if bar_timestamp else None,
        "evaluated_at": now.isoformat(),
        "timeframe": timeframe.upper(),
    }
    if bar_timestamp is None:
        return False, details, "No market-data timestamp was available for the signal."

    age = now - bar_timestamp
    max_age = (_timeframe_delta(timeframe) * 3) + timedelta(seconds=30)
    details["bar_age_seconds"] = max(0.0, age.total_seconds())
    details["max_bar_age_seconds"] = max_age.total_seconds()
    if age > max_age:
        return False, details, "Latest market bar is stale for the configured timeframe."
    return True, details, None


def _max_bar_range_pct(timeframe: str) -> float:
    return {
        "M1": 1.5,
        "M5": 2.0,
        "M15": 3.0,
        "M30": 4.0,
        "H1": 5.0,
        "H4": 8.0,
        "D1": 15.0,
        "W1": 25.0,
    }.get((timeframe or "M5").upper(), 3.0)


def _validate_bar_snapshot(
    timeframe: str,
    bar_snapshot: Dict[str, object] | None,
) -> tuple[bool, Dict[str, object], str | None]:
    if not bar_snapshot:
        return True, {}, None

    details: Dict[str, object] = {
        "bar_open": bar_snapshot.get("open"),
        "bar_high": bar_snapshot.get("high"),
        "bar_low": bar_snapshot.get("low"),
        "bar_close": bar_snapshot.get("close"),
    }
    try:
        open_price = float(bar_snapshot["open"])
        high_price = float(bar_snapshot["high"])
        low_price = float(bar_snapshot["low"])
        close_price = float(bar_snapshot["close"])
    except (KeyError, TypeError, ValueError):
        return False, details, "Latest market bar is incomplete or non-numeric."

    if min(open_price, high_price, low_price, close_price) <= 0:
        return False, details, "Latest market bar contains non-positive prices."
    if high_price < low_price:
        return False, details, "Latest market bar has invalid high/low ordering."
    if not (low_price <= open_price <= high_price and low_price <= close_price <= high_price):
        return False, details, "Latest market bar has open/close outside the high-low range."

    range_pct = ((high_price - low_price) / close_price) * 100.0 if close_price > 0 else 0.0
    details["bar_range_pct"] = range_pct
    details["max_bar_range_pct"] = _max_bar_range_pct(timeframe)
    if range_pct > details["max_bar_range_pct"]:
        return False, details, "Latest market bar range is too wide for the configured timeframe."

    return True, details, None


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


def _within_session(config: EngineConfig, now: datetime) -> bool:
    if not config.session_filter_enabled:
        return True
    start = int(config.session_start_hour_utc)
    end = int(config.session_end_hour_utc)
    hour = now.hour
    if start == end:
        return True
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def _same_symbol_positions(open_positions: List[PaperPosition], symbol: str) -> List[PaperPosition]:
    return [position for position in open_positions if position.symbol.upper() == symbol.upper()]


def _daily_loss_cap(config: EngineConfig) -> float:
    return abs(float(config.daily_loss_limit_pct or 0.0))


def _in_symbol_cooldown(config: EngineConfig, open_positions: List[PaperPosition], symbol: str, timeframe: str) -> bool:
    if config.cooldown_minutes <= 0:
        return False
    cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=int(config.cooldown_minutes))
    for position in list_paper_positions():
        if position.symbol.upper() != symbol.upper() or position.timeframe.upper() != timeframe.upper():
            continue
        if position.closed_at and position.closed_at >= cutoff:
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
) -> RiskDecision:
    now = datetime.now(UTC).replace(tzinfo=None)
    open_positions = list_paper_positions("open")
    same_symbol_positions = _same_symbol_positions(open_positions, analysis.symbol)
    decision = RiskDecision(accepted=False, reasons=[], details={}, intent_type="skip")

    if analysis.signal == "no_trade":
        decision.reasons.append("Strategy returned no_trade.")
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
        decision.reasons.append("Confidence is below the configured minimum.")
        decision.details["confidence"] = analysis.confidence
        decision.details["min_confidence"] = config.min_confidence
        return decision

    if source != "manual" or bar_timestamp is not None:
        market_ok, market_details, market_error = _validate_market_freshness(
            analysis.timeframe,
            bar_timestamp,
            now,
        )
        decision.details.update(market_details)
        if not market_ok and market_error:
            decision.reasons.append(market_error)
            return decision

    if source != "manual" or bar_snapshot:
        bar_ok, bar_details, bar_error = _validate_bar_snapshot(analysis.timeframe, bar_snapshot)
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

    pnl_today = daily_realized_pnl()
    daily_loss_cap = _daily_loss_cap(config)
    if pnl_today <= -daily_loss_cap:
        decision.reasons.append("Daily loss cap reached.")
        decision.details["daily_realized_pnl"] = pnl_today
        decision.details["daily_loss_cap"] = daily_loss_cap
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

    if source != "manual" and not config.paper_autotrade:
        decision.reasons.append("Paper autotrade is disabled.")
        return decision

    decision.accepted = True
    decision.intent_type = "open"
    decision.reasons.append("Signal passed risk checks.")
    decision.details["watch_item"] = watch_item.model_dump(mode="json")
    return decision
