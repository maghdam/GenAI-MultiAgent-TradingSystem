from __future__ import annotations

from datetime import UTC, datetime, timedelta

from backend.domain.models import EngineConfig, PaperPosition, StrategyAnalysis, WatchlistItem
from backend.services.alpaca_execution import close_paper_position, submit_paper_market_order
from backend.services.alpaca_bracket_readiness import paper_bracket_automation_readiness
from backend.services.alpaca_paper import AlpacaPaperError, get_account, get_open_position, list_positions
from backend.services.execution_engine import ExecutionResult
from backend.services.financial_units import MonetaryBasis
from backend.services.risk_engine import RiskPortfolioSnapshot, evaluate_risk
from backend.storage.repositories import (
    add_trade_audit,
    daily_realized_pnl,
    daily_trade_count,
    list_order_intents,
    list_paper_positions,
    list_trade_audits,
)


def _symbol_key(symbol: str) -> str:
    return str(symbol or "").strip().upper().replace("/", "")


def _broker_direction(position: dict[str, object] | None) -> str | None:
    if not position:
        return None
    side = str(position.get("side") or "").strip().lower()
    if side == "long":
        return "long"
    if side == "short":
        return "short"
    return None


def _paper_position_view(
    position: dict[str, object] | None,
    analysis: StrategyAnalysis,
    mark_price: float | None,
) -> PaperPosition | None:
    direction = _broker_direction(position)
    if position is None or direction is None:
        return None
    try:
        quantity = abs(float(position.get("qty") or position.get("qty_available") or 0.0))
    except (TypeError, ValueError):
        quantity = 0.0
    try:
        entry_price = float(position.get("avg_entry_price") or mark_price or 0.0)
    except (TypeError, ValueError):
        entry_price = float(mark_price or 0.0)
    if quantity <= 0 or entry_price <= 0:
        return None
    return PaperPosition(
        id=-1,
        symbol=analysis.symbol.upper(),
        timeframe=analysis.timeframe.upper(),
        strategy=analysis.strategy,
        direction=direction,
        quantity=quantity,
        status="open",
        entry_price=entry_price,
        current_price=float(mark_price) if mark_price is not None else None,
        stop_loss=None,
        take_profit=None,
        opened_at=datetime.now(UTC).replace(tzinfo=None),
        account_currency="USD",
        cash_per_price_unit_per_lot=1.0,
        instrument_spec_source="alpaca_broker_truth",
        realized_pnl_source="alpaca_broker_truth",
        broker_sync_status="alpaca_paper",
    )


def _alpaca_monetary_basis(account: dict[str, object]) -> MonetaryBasis:
    currency = str(account.get("currency") or "").strip().upper()
    try:
        equity = float(account.get("equity") or 0.0)
    except (TypeError, ValueError):
        equity = 0.0

    ready = (
        str(account.get("environment") or "") == "paper"
        and str(account.get("status") or "").upper() == "ACTIVE"
        and not bool(account.get("trading_blocked"))
        and not bool(account.get("account_blocked"))
        and not bool(account.get("trade_suspended_by_user"))
        and len(currency) == 3
        and equity > 0
    )
    reason = ""
    if not ready:
        reason = "Alpaca Paper account monetary/execution truth is not verified."
    return MonetaryBasis(
        currency=currency,
        equity_amount=equity,
        source="alpaca_paper",
        verified=ready,
        reason=reason,
    )


def _alpaca_realized_pnl_today(currency: str) -> float:
    today = datetime.now(UTC).date()
    total = 0.0
    for audit in list_trade_audits(500):
        if audit.created_at.date() != today:
            continue
        details = audit.details if isinstance(audit.details, dict) else {}
        if details.get("broker_provider") != "alpaca":
            continue
        value = details.get("realized_pnl_estimate")
        if value is None:
            continue
        pnl_currency = str(details.get("realized_pnl_currency") or "USD").upper()
        if pnl_currency != currency.upper():
            continue
        try:
            total += float(value)
        except (TypeError, ValueError):
            continue
    return total


def _alpaca_executed_entries_today() -> int:
    today = datetime.now(UTC).date()
    total = 0
    for intent in list_order_intents(500):
        details = intent.details if isinstance(intent.details, dict) else {}
        if (
            intent.created_at.date() == today
            and intent.intent_type == "open"
            and intent.status == "executed"
            and details.get("broker_provider") == "alpaca"
            and details.get("broker_environment") == "paper"
        ):
            total += 1
    return total


def _alpaca_symbol_cooldown_active(
    config: EngineConfig,
    symbol: str,
    timeframe: str,
) -> bool:
    if config.cooldown_minutes <= 0:
        return False
    cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(
        minutes=int(config.cooldown_minutes)
    )
    symbol_key = _symbol_key(symbol)
    timeframe_key = timeframe.upper()
    for audit in list_trade_audits(500):
        if audit.created_at < cutoff:
            continue
        if _symbol_key(audit.symbol) != symbol_key or audit.timeframe.upper() != timeframe_key:
            continue
        if audit.event_type in {
            "alpaca_paper_position_close_filled",
            "alpaca_paper_position_flat_confirmed",
        }:
            return True
    return False


def _portfolio_snapshot(
    *,
    config: EngineConfig,
    analysis: StrategyAnalysis,
    alpaca_positions: list[dict[str, object]],
    currency: str,
) -> RiskPortfolioSnapshot:
    local_positions = list_paper_positions("open")
    symbol_key = _symbol_key(analysis.symbol)
    local_same = sum(
        1 for position in local_positions if _symbol_key(position.symbol) == symbol_key
    )
    alpaca_same = sum(
        1
        for position in alpaca_positions
        if _symbol_key(str(position.get("symbol") or "")) == symbol_key
    )
    return RiskPortfolioSnapshot(
        open_positions_count=len(local_positions) + len(alpaca_positions),
        same_symbol_positions_count=local_same + alpaca_same,
        daily_realized_pnl_amount=(
            float(daily_realized_pnl()) + _alpaca_realized_pnl_today(currency)
        ),
        daily_trade_count_value=(
            int(daily_trade_count()) + _alpaca_executed_entries_today()
        ),
        symbol_cooldown_active=_alpaca_symbol_cooldown_active(
            config,
            analysis.symbol,
            analysis.timeframe,
        ),
        source="combined_local_ctrader_alpaca_paper",
    )


def _runtime_details(
    *,
    watch_item: WatchlistItem,
    analysis: StrategyAnalysis,
    risk_details: dict[str, object],
    risk_reasons: list[str],
) -> dict[str, object]:
    return {
        "runtime_execution": {
            "source": "trusted_strategy_auto_loop",
            "broker_venue": "alpaca",
            "broker_environment": "paper",
            "asset_class": watch_item.alpaca_asset_class,
            "size_mode": watch_item.alpaca_size_mode,
            "size_value": watch_item.alpaca_size_value,
            "signal": analysis.signal,
            "confidence": analysis.confidence,
            "entry_price": analysis.entry_price,
            "stop_loss": analysis.stop_loss,
            "take_profit": analysis.take_profit,
            "risk_reasons": risk_reasons,
            "risk_details": risk_details,
            "broker_protection_delivery": "signal_metadata_only",
            "automatic_resubmission": False,
        }
    }


def _audit_runtime_block(
    *,
    analysis: StrategyAnalysis,
    summary: str,
    details: dict[str, object],
) -> None:
    add_trade_audit(
        event_type="alpaca_runtime_signal_blocked",
        symbol=analysis.symbol,
        timeframe=analysis.timeframe,
        strategy=analysis.strategy,
        summary=summary,
        details={
            "broker_provider": "alpaca",
            "broker_environment": "paper",
            **details,
        },
    )


def _result_from_broker(result: dict[str, object]) -> ExecutionResult:
    return ExecutionResult(
        action_taken=bool(result.get("action_taken")),
        intent_id=(
            int(result["intent_id"])
            if result.get("intent_id") is not None
            else None
        ),
        status=str(result.get("status") or "unknown"),
        summary=str(result.get("summary") or "Alpaca Paper runtime result."),
        mode="alpaca_paper",
        retryable=False,
    )


def execute_alpaca_runtime_signal(
    *,
    config: EngineConfig,
    watch_item: WatchlistItem,
    analysis: StrategyAnalysis,
    mark_price: float | None,
    bar_timestamp: datetime | None,
    bar_snapshot: dict[str, object] | None,
) -> ExecutionResult:
    if watch_item.venue != "alpaca" or watch_item.alpaca_asset_class is None:
        return ExecutionResult(
            action_taken=False,
            intent_id=None,
            status="blocked",
            summary="Alpaca runtime execution requires an explicit Alpaca venue and asset class.",
            mode="alpaca_paper",
        )

    try:
        account = get_account()
        alpaca_positions = list_positions()
        broker_position = get_open_position(analysis.symbol)
    except AlpacaPaperError as exc:
        _audit_runtime_block(
            analysis=analysis,
            summary="Alpaca Paper broker truth was unavailable before runtime execution.",
            details={"reason": str(exc), "preflight_failed": True},
        )
        return ExecutionResult(
            action_taken=False,
            intent_id=None,
            status="blocked",
            summary="Alpaca Paper broker truth is unavailable; runtime execution failed closed.",
            mode="alpaca_paper",
            retryable=True,
        )

    monetary_basis = _alpaca_monetary_basis(account)
    position_view = _paper_position_view(broker_position, analysis, mark_price)
    portfolio = _portfolio_snapshot(
        config=config,
        analysis=analysis,
        alpaca_positions=alpaca_positions,
        currency=monetary_basis.currency or "USD",
    )
    risk = evaluate_risk(
        config=config,
        watch_item=watch_item,
        analysis=analysis,
        existing_position=position_view,
        mark_price=mark_price,
        bar_timestamp=bar_timestamp,
        bar_snapshot=bar_snapshot,
        source="auto",
        monetary_basis=monetary_basis,
        portfolio_snapshot=portfolio,
    )
    execution_details = _runtime_details(
        watch_item=watch_item,
        analysis=analysis,
        risk_details=risk.details,
        risk_reasons=list(risk.reasons),
    )

    if not risk.accepted:
        _audit_runtime_block(
            analysis=analysis,
            summary=risk.reasons[0] if risk.reasons else "Alpaca runtime risk gate rejected the signal.",
            details={
                **execution_details,
                "risk_accepted": False,
                "risk_intent_type": risk.intent_type,
            },
        )
        return ExecutionResult(
            action_taken=False,
            intent_id=None,
            status="rejected",
            summary=risk.reasons[0] if risk.reasons else "Alpaca runtime risk gate rejected the signal.",
            mode="alpaca_paper",
        )

    if risk.intent_type == "update":
        add_trade_audit(
            event_type="alpaca_runtime_same_direction_hold",
            symbol=analysis.symbol,
            timeframe=analysis.timeframe,
            strategy=analysis.strategy,
            summary="Existing Alpaca Paper position already matches the strategy direction; no add-on order was submitted.",
            details={
                "broker_provider": "alpaca",
                "broker_environment": "paper",
                **execution_details,
            },
        )
        return ExecutionResult(
            action_taken=False,
            intent_id=None,
            status="held",
            summary="Existing Alpaca Paper position already matches the strategy direction.",
            mode="alpaca_paper",
        )

    if risk.intent_type == "close":
        result = close_paper_position(
            symbol_or_asset_id=analysis.symbol,
            timeframe=analysis.timeframe,
            strategy=analysis.strategy,
            rationale="; ".join(analysis.reasons) or "Trusted runtime strategy requested a risk-reducing Alpaca Paper close.",
            confidence=analysis.confidence,
            execution_details=execution_details,
        )
        return _result_from_broker(result)

    if risk.intent_type != "open":
        return ExecutionResult(
            action_taken=False,
            intent_id=None,
            status="skipped",
            summary=risk.reasons[0] if risk.reasons else "No Alpaca Paper mutation was required.",
            mode="alpaca_paper",
        )

    if watch_item.alpaca_size_mode is None or watch_item.alpaca_size_value is None:
        summary = "Alpaca runtime sizing is not explicitly configured."
        _audit_runtime_block(
            analysis=analysis,
            summary=summary,
            details={**execution_details, "sizing_missing": True},
        )
        return ExecutionResult(
            action_taken=False,
            intent_id=None,
            status="rejected",
            summary=summary,
            mode="alpaca_paper",
        )

    side = "buy" if analysis.signal == "long" else "sell"
    if watch_item.alpaca_asset_class == "crypto" and side == "sell":
        summary = "Alpaca crypto runtime does not open short positions; no order was submitted."
        _audit_runtime_block(
            analysis=analysis,
            summary=summary,
            details={**execution_details, "crypto_short_blocked": True},
        )
        return ExecutionResult(
            action_taken=False,
            intent_id=None,
            status="rejected",
            summary=summary,
            mode="alpaca_paper",
        )

    if side == "sell" and watch_item.alpaca_size_mode == "notional":
        summary = "Alpaca short equity runtime requires quantity sizing; notional short entries are blocked."
        _audit_runtime_block(
            analysis=analysis,
            summary=summary,
            details={**execution_details, "notional_short_blocked": True},
        )
        return ExecutionResult(
            action_taken=False,
            intent_id=None,
            status="rejected",
            summary=summary,
            mode="alpaca_paper",
        )

    summary = (
        "Automatic Alpaca Paper entry is blocked until broker-native protective-order "
        "lifecycle validation and explicit activation approval are complete. Existing positions "
        "can still receive trusted close signals; operator-only simple Paper orders "
        "remain a separate supervised workflow."
    )
    _audit_runtime_block(
        analysis=analysis,
        summary=summary,
        details={
            **execution_details,
            "automatic_entry_protection_required": True,
            "bracket_automation_readiness": paper_bracket_automation_readiness(),
            "broker_native_protection_verified": False,
            "risk_accepted": True,
            "risk_intent_type": "open",
        },
    )
    return ExecutionResult(
        action_taken=False,
        intent_id=None,
        status="rejected",
        summary=summary,
        mode="alpaca_paper",
    )

    qty = (
        float(watch_item.alpaca_size_value)
        if watch_item.alpaca_size_mode == "qty"
        else None
    )
    notional = (
        float(watch_item.alpaca_size_value)
        if watch_item.alpaca_size_mode == "notional"
        else None
    )
    result = submit_paper_market_order(
        symbol=analysis.symbol,
        asset_class=watch_item.alpaca_asset_class,
        side=side,
        qty=qty,
        notional=notional,
        timeframe=analysis.timeframe,
        strategy=analysis.strategy,
        rationale="; ".join(analysis.reasons) or "Trusted runtime strategy signal.",
        confidence=analysis.confidence,
        entry_price=analysis.entry_price or mark_price,
        stop_loss=analysis.stop_loss,
        take_profit=analysis.take_profit,
        execution_details=execution_details,
    )
    return _result_from_broker(result)
