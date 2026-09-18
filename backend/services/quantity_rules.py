from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from backend.domain.models import EngineConfig, StrategyAnalysis
from backend.domain.models import SymbolLimits
from backend.services.broker import get_instrument_spec, get_symbol_limits

def _protocol_volume_per_lot(limits: SymbolLimits) -> float:
    """Infer protocol-volume units per lot from broker-derived limits.

    SymbolLimits stores raw cTrader protocol volumes alongside their lot
    equivalents. Their ratio is contract-specific (e.g. XAU vs FX).
    """
    candidates = (
        (limits.min_api_units, limits.min_lots),
        (limits.step_api_units, limits.step_lots),
        (limits.max_api_units, limits.max_lots),
    )
    for protocol_volume, lots in candidates:
        try:
            protocol_value = float(protocol_volume)
            lots_value = float(lots)
        except (TypeError, ValueError):
            continue
        if protocol_value > 0 and lots_value > 0:
            return protocol_value / lots_value
    raise ValueError(f"Unable to infer protocol volume per lot for {limits.symbol}")


@dataclass
class QuantityDecision:
    accepted: bool
    requested_quantity: float
    final_quantity: float | None
    reasons: List[str] = field(default_factory=list)
    details: Dict[str, object] = field(default_factory=dict)


@dataclass
class AutoSizingDecision:
    accepted: bool
    requested_quantity: float
    reasons: List[str] = field(default_factory=list)
    details: Dict[str, object] = field(default_factory=dict)


def _decision_details(
    *,
    limits: SymbolLimits,
    requested_quantity: float,
    requested_api_units: int,
    final_api_units: int | None,
    mode: str,
    protocol_per_lot: float,
) -> Dict[str, object]:
    return {
        "quantity_mode": mode,
        "requested_quantity": requested_quantity,
        "requested_api_units": requested_api_units,
        "final_quantity": (final_api_units / protocol_per_lot) if final_api_units is not None else None,
        "final_api_units": final_api_units,
        "protocol_volume_per_lot": protocol_per_lot,
        "quantity_normalized": final_api_units is not None and final_api_units != requested_api_units,
        "symbol_limits": limits.model_dump(mode="json"),
    }


def derive_auto_quantity(config: EngineConfig, analysis: StrategyAnalysis, mark_price: float | None) -> AutoSizingDecision:
    fallback_quantity = float(config.paper_trade_size or 1.0)
    entry_reference = float(analysis.entry_price if analysis.entry_price is not None else (mark_price or 0.0))
    stop_loss = analysis.stop_loss
    risk_pct = float(config.risk_per_trade_pct or 0.0)
    details: Dict[str, object] = {
        "sizing_mode": "fixed_fallback",
        "fallback_quantity": fallback_quantity,
        "risk_per_trade_pct": risk_pct,
        "entry_reference": entry_reference or None,
        "stop_loss": stop_loss,
    }
    instrument = get_instrument_spec(analysis.symbol, config.account_currency)
    details["instrument_spec"] = instrument.model_dump(mode="json")

    if entry_reference <= 0 or stop_loss is None:
        return AutoSizingDecision(
            accepted=risk_pct <= 0,
            requested_quantity=fallback_quantity,
            reasons=[
                "Risk-based auto quantity requires a valid entry and stop reference."
                if risk_pct > 0
                else "Auto quantity uses the configured fixed size because risk sizing is disabled."
            ],
            details=details,
        )

    stop_distance = abs(entry_reference - float(stop_loss))
    stop_pct = (stop_distance / abs(entry_reference)) if entry_reference else 0.0
    details["stop_distance"] = stop_distance
    details["stop_pct"] = stop_pct * 100.0

    if stop_distance <= 0 or stop_pct <= 0:
        return AutoSizingDecision(
            accepted=risk_pct <= 0,
            requested_quantity=fallback_quantity,
            reasons=[
                "Risk-based auto quantity requires a positive stop distance."
                if risk_pct > 0
                else "Auto quantity uses the configured fixed size because risk sizing is disabled."
            ],
            details=details,
        )

    if risk_pct <= 0:
        return AutoSizingDecision(
            accepted=True,
            requested_quantity=fallback_quantity,
            reasons=["Auto quantity fell back to the configured fixed size because risk sizing is disabled."],
            details=details,
        )

    cash_per_price_unit = instrument.cash_per_price_unit_per_lot
    if not instrument.valuation_ready or not cash_per_price_unit or cash_per_price_unit <= 0:
        details["sizing_mode"] = "fixed_fallback_unvalued_contract"
        return AutoSizingDecision(
            accepted=False,
            requested_quantity=fallback_quantity,
            reasons=[
                "Automatic execution is blocked because broker contract valuation is unavailable."
            ],
            details=details,
        )

    equity_amount = float(config.paper_starting_equity_amount)
    risk_amount = equity_amount * (risk_pct / 100.0)
    loss_per_lot = stop_distance * float(cash_per_price_unit)
    requested_quantity = risk_amount / loss_per_lot
    details["sizing_mode"] = "risk_based"
    details["raw_risk_quantity"] = requested_quantity
    details["equity_amount"] = equity_amount
    details["risk_amount"] = risk_amount
    details["loss_per_lot_at_stop"] = loss_per_lot
    return AutoSizingDecision(
        accepted=True,
        requested_quantity=requested_quantity,
        reasons=[
            f"Auto quantity derived from {risk_pct:.2f}% risk ({config.account_currency} {risk_amount:.2f}) "
            f"and {config.account_currency} {loss_per_lot:.2f} loss per lot at the stop."
        ],
        details=details,
    )


def evaluate_order_quantity(symbol: str, quantity: float, source: str) -> QuantityDecision:
    try:
        requested_quantity = float(quantity)
    except (TypeError, ValueError):
        return QuantityDecision(
            accepted=False,
            requested_quantity=0.0,
            final_quantity=None,
            reasons=["Requested quantity must be numeric."],
            details={"quantity_mode": "invalid"},
        )

    if requested_quantity <= 0:
        return QuantityDecision(
            accepted=False,
            requested_quantity=requested_quantity,
            final_quantity=None,
            reasons=["Requested quantity must be greater than zero."],
            details={"quantity_mode": "invalid"},
        )

    limits = get_symbol_limits(symbol)
    protocol_per_lot = _protocol_volume_per_lot(limits)
    requested_api_units = int(round(requested_quantity * protocol_per_lot))
    min_api_units = max(int(limits.min_api_units), 1)
    step_api_units = max(int(limits.step_api_units), 1)
    max_api_units = max(int(limits.max_api_units), min_api_units)
    mode = "strict" if source == "manual" else "coerce"

    if source == "manual":
        details = _decision_details(
            limits=limits,
            requested_quantity=requested_quantity,
            requested_api_units=requested_api_units,
            final_api_units=requested_api_units,
            mode=mode,
            protocol_per_lot=protocol_per_lot,
        )
        if requested_api_units < min_api_units:
            return QuantityDecision(
                accepted=False,
                requested_quantity=requested_quantity,
                final_quantity=None,
                reasons=[f"Requested quantity is below the symbol minimum of {limits.min_lots:.4f} lots."],
                details=details,
            )
        if requested_api_units > max_api_units:
            return QuantityDecision(
                accepted=False,
                requested_quantity=requested_quantity,
                final_quantity=None,
                reasons=[f"Requested quantity exceeds the symbol maximum of {limits.max_lots:.4f} lots."],
                details=details,
            )
        if requested_api_units % step_api_units:
            return QuantityDecision(
                accepted=False,
                requested_quantity=requested_quantity,
                final_quantity=None,
                reasons=[f"Requested quantity must align to the symbol step size of {limits.step_lots:.4f} lots."],
                details=details,
            )
        return QuantityDecision(
            accepted=True,
            requested_quantity=requested_quantity,
            final_quantity=requested_quantity,
            reasons=["Requested quantity matches the symbol limits."],
            details=details,
        )

    final_api_units = requested_api_units
    if final_api_units < min_api_units:
        final_api_units = min_api_units
    if final_api_units % step_api_units:
        final_api_units += step_api_units - (final_api_units % step_api_units)
    if final_api_units > max_api_units:
        final_api_units = max_api_units
        if final_api_units % step_api_units:
            final_api_units -= final_api_units % step_api_units
            final_api_units = max(final_api_units, min_api_units)
    final_quantity = final_api_units / protocol_per_lot
    details = _decision_details(
        limits=limits,
        requested_quantity=requested_quantity,
        requested_api_units=requested_api_units,
        final_api_units=final_api_units,
        mode=mode,
    )
    reasons = ["Quantity fits the symbol limits."]
    if final_api_units != requested_api_units:
        reasons = [f"Quantity was normalized to {final_quantity:.4f} lots to fit symbol limits."]
    return QuantityDecision(
        accepted=True,
        requested_quantity=requested_quantity,
        final_quantity=final_quantity,
        reasons=reasons,
        details=details,
    )
