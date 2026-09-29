from __future__ import annotations

from collections import Counter, defaultdict
from statistics import fmean
from typing import Any

from backend.storage.repositories import list_confidence_calibration_outcomes


_BUCKETS: tuple[tuple[str, float, float | None], ...] = (
    ("<60%", 0.0, 0.60),
    ("60-65%", 0.60, 0.65),
    ("65-70%", 0.65, 0.70),
    ("70-75%", 0.70, 0.75),
    ("75-80%", 0.75, 0.80),
    ("80%+", 0.80, None),
)


def _bucket_for(confidence: float) -> tuple[str, float, float | None]:
    value = max(0.0, min(1.0, float(confidence)))
    for label, lower, upper in _BUCKETS:
        if upper is None:
            if value >= lower:
                return label, lower, upper
        elif lower <= value < upper:
            return label, lower, upper
    return _BUCKETS[-1]


def _risk_amount(row: dict[str, Any]) -> float | None:
    stop = row.get("signal_stop_loss")
    if stop is None:
        return None
    entry = float(row.get("entry_price") or 0.0)
    quantity = float(row.get("quantity") or 0.0)
    cash_per_unit = float(row.get("cash_per_price_unit_per_lot") or 0.0)
    risk = abs(entry - float(stop)) * quantity * cash_per_unit
    if risk <= 0.0:
        return None
    return risk


def _expectancy_by_currency(rows: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        currency = str(row.get("account_currency") or "UNKNOWN").upper()
        values[currency].append(float(row.get("realized_pnl") or 0.0))
    return {
        currency: round(fmean(pnls), 6)
        for currency, pnls in sorted(values.items())
        if pnls
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trade_count = len(rows)
    wins = sum(1 for row in rows if float(row.get("realized_pnl") or 0.0) > 0.0)
    losses = sum(1 for row in rows if float(row.get("realized_pnl") or 0.0) < 0.0)
    breakeven = trade_count - wins - losses
    r_values = [
        float(row["r_multiple"])
        for row in rows
        if row.get("r_multiple") is not None
    ]
    pnl_sources = Counter(str(row.get("realized_pnl_source") or "unknown") for row in rows)
    broker_trades = sum(1 for row in rows if row.get("broker_position_id") is not None)

    return {
        "trade_count": trade_count,
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "win_rate_pct": round((wins / trade_count) * 100.0, 4) if trade_count else None,
        "expectancy": _expectancy_by_currency(rows),
        "average_r": round(fmean(r_values), 6) if r_values else None,
        "r_trade_count": len(r_values),
        "broker_trade_count": broker_trades,
        "paper_trade_count": trade_count - broker_trades,
        "realized_pnl_sources": dict(sorted(pnl_sources.items())),
    }


def build_confidence_calibration(
    *,
    strategy: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    include_manual: bool = False,
) -> dict[str, Any]:
    """Build descriptive signal-strength calibration from closed executed trades.

    Signal strength remains a deterministic heuristic. This report is evidence
    for calibration work only and never turns the score into a probability or
    changes execution thresholds.
    """
    source_rows = list_confidence_calibration_outcomes(
        strategy=strategy,
        symbol=symbol,
        timeframe=timeframe,
    )

    rows: list[dict[str, Any]] = []
    excluded_manual = 0
    for source_row in source_rows:
        details = source_row.get("intent_details")
        source = (
            str(details.get("source") or "auto").strip().lower()
            if isinstance(details, dict)
            else "auto"
        )
        if source == "manual" and not include_manual:
            excluded_manual += 1
            continue

        row = dict(source_row)
        label, lower, upper = _bucket_for(float(row.get("confidence") or 0.0))
        row["bucket"] = label
        row["bucket_lower"] = lower
        row["bucket_upper"] = upper
        risk_amount = _risk_amount(row)
        row["initial_risk_amount"] = risk_amount
        row["r_multiple"] = (
            float(row.get("realized_pnl") or 0.0) / risk_amount
            if risk_amount is not None
            else None
        )
        rows.append(row)

    drawdown_contribution: dict[str, float] = defaultdict(float)
    cumulative_r = 0.0
    peak_r = 0.0
    previous_drawdown = 0.0
    max_drawdown_r = 0.0
    for row in rows:
        r_multiple = row.get("r_multiple")
        if r_multiple is None:
            continue
        cumulative_r += float(r_multiple)
        peak_r = max(peak_r, cumulative_r)
        drawdown = max(0.0, peak_r - cumulative_r)
        increase = max(0.0, drawdown - previous_drawdown)
        drawdown_contribution[str(row["bucket"])] += increase
        previous_drawdown = drawdown
        max_drawdown_r = max(max_drawdown_r, drawdown)

    bucket_rows: dict[str, list[dict[str, Any]]] = {label: [] for label, _, _ in _BUCKETS}
    for row in rows:
        bucket_rows[str(row["bucket"])].append(row)

    buckets: list[dict[str, Any]] = []
    for label, lower, upper in _BUCKETS:
        bucket = bucket_rows[label]
        summary = _summarize(bucket)
        summary["drawdown_contribution_r"] = round(drawdown_contribution.get(label, 0.0), 6)
        buckets.append(
            {
                "bucket": label,
                "lower_bound": lower,
                "upper_bound": upper,
                **summary,
            }
        )

    overall = _summarize(rows)
    overall["drawdown_contribution_r"] = round(sum(drawdown_contribution.values()), 6)
    overall["max_drawdown_r"] = round(max_drawdown_r, 6) if any(
        row.get("r_multiple") is not None for row in rows
    ) else None

    return {
        "status": "descriptive_only" if rows else "no_data",
        "semantics": {
            "score_name": "signal_strength",
            "probability_label_supported": False,
            "threshold_changes_automatic": False,
            "description": (
                "Signal strength is a deterministic strategy-strength heuristic, not a calibrated "
                "win probability. This report is descriptive evidence only."
            ),
        },
        "filters": {
            "strategy": strategy,
            "symbol": symbol.upper() if symbol else None,
            "timeframe": timeframe.upper() if timeframe else None,
            "include_manual": include_manual,
        },
        "methodology": {
            "bucket_rule": "Lower bound inclusive; upper bound exclusive; 80%+ has no upper bound.",
            "win_rate": "Winning closed trades divided by all closed trades in the bucket.",
            "expectancy": (
                "Mean realized P&L per closed trade, reported separately by account currency "
                "so unlike currencies are never averaged together."
            ),
            "average_r": (
                "Mean realized P&L divided by initial stop-defined monetary risk. Trades without "
                "a valid opening stop/risk amount remain in win-rate/expectancy counts but are "
                "excluded from R metrics."
            ),
            "drawdown_contribution_r": (
                "Sum of chronological increases in cumulative-R drawdown depth attributed to "
                "trades in the bucket; this is descriptive attribution, not causal impact."
            ),
        },
        "sample": {
            "linked_closed_trades": len(rows),
            "r_eligible_trades": sum(1 for row in rows if row.get("r_multiple") is not None),
            "r_ineligible_trades": sum(1 for row in rows if row.get("r_multiple") is None),
            "excluded_manual_trades": excluded_manual,
        },
        "overall": overall,
        "buckets": buckets,
    }
