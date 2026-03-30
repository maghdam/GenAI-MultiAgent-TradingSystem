from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from backend.domain.models import EngineConfig, StrategyAnalysis
from backend.domain.models import SymbolLimits
from backend.services.broker import get_symbol_limits

_API_VOLUME_PRECISION = 10_000


@dataclass
class QuantityDecision:
    accepted: bool
    requested_quantity: float
    final_quantity: float | None
    reasons: List[str] = field(default_factory=list)
    details: Dict[str, object] = field(default_factory=dict)


@dataclass
class AutoSizingDecision:
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
) -> Dict[str, object]:
    return {
        "quantity_mode": mode,
        "requested_quantity": requested_quantity,
        "requested_api_units": requested_api_units,
        "final_quantity": (final_api_units / _API_VOLUME_PRECISION) if final_api_units is not None else None,
        "final_api_units": final_api_units,
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

    if entry_reference <= 0 or stop_loss is None:
        return AutoSizingDecision(
            requested_quantity=fallback_quantity,
            reasons=["Auto quantity fell back to the configured fixed size because no valid stop reference was available."],
            details=details,
        )

    stop_distance = abs(entry_reference - float(stop_loss))
    stop_pct = (stop_distance / abs(entry_reference)) if entry_reference else 0.0
    details["stop_distance"] = stop_distance
    details["stop_pct"] = stop_pct * 100.0

    if stop_distance <= 0 or stop_pct <= 0:
        return AutoSizingDecision(
            requested_quantity=fallback_quantity,
            reasons=["Auto quantity fell back to the configured fixed size because the stop distance was invalid."],
            details=details,
        )

    if risk_pct <= 0:
        return AutoSizingDecision(
            requested_quantity=fallback_quantity,
            reasons=["Auto quantity fell back to the configured fixed size because risk sizing is disabled."],
            details=details,
        )

    requested_quantity = (risk_pct / 100.0) / stop_pct
    details["sizing_mode"] = "risk_based"
    details["raw_risk_quantity"] = requested_quantity
    return AutoSizingDecision(
        requested_quantity=requested_quantity,
        reasons=[f"Auto quantity derived from {risk_pct:.2f}% risk and a {stop_pct * 100.0:.2f}% stop distance."],
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
    requested_api_units = int(round(requested_quantity * _API_VOLUME_PRECISION))
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
    final_quantity = final_api_units / _API_VOLUME_PRECISION
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
