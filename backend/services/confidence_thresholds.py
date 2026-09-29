from __future__ import annotations

from collections import Counter
from datetime import datetime
from math import sqrt
from typing import Any

from backend.storage.repositories import list_confidence_calibration_outcomes


_BASELINE_THRESHOLD = 0.60
_EXECUTION_BUCKETS: tuple[tuple[str, float, float | None], ...] = (
    ("60-65%", 0.60, 0.65),
    ("65-70%", 0.65, 0.70),
    ("70-75%", 0.70, 0.75),
    ("75-80%", 0.75, 0.80),
    ("80%+", 0.80, None),
)

# Conservative screening floors for deciding whether it is worth beginning a
# threshold-comparison study for one exact strategy × symbol × timeframe cell.
# These are operational evidence gates, not a claim of statistical proof.
_MIN_BASELINE_TRADES = 100
_MAX_WORST_CASE_WIN_RATE_MOE95_PCT = 10.0
_MIN_R_COVERAGE_RATIO = 0.80
_MIN_WINS = 10
_MIN_LOSSES = 10
_MIN_POPULATED_BUCKETS = 3
_MIN_TRADES_PER_POPULATED_BUCKET = 10


def _bucket_label(confidence: float) -> str | None:
    value = float(confidence)
    if value < _BASELINE_THRESHOLD:
        return None
    for label, lower, upper in _EXECUTION_BUCKETS:
        if upper is None:
            if value >= lower:
                return label
        elif lower <= value < upper:
            return label
    return _EXECUTION_BUCKETS[-1][0]


def _initial_risk_amount(row: dict[str, Any]) -> float | None:
    stop = row.get("signal_stop_loss")
    if stop is None:
        return None
    entry = float(row.get("entry_price") or 0.0)
    quantity = float(row.get("quantity") or 0.0)
    cash_per_unit = float(row.get("cash_per_price_unit_per_lot") or 0.0)
    risk = abs(entry - float(stop)) * quantity * cash_per_unit
    return risk if risk > 0.0 else None


def _is_manual(row: dict[str, Any]) -> bool:
    details = row.get("intent_details")
    source = (
        str(details.get("source") or "auto").strip().lower()
        if isinstance(details, dict)
        else "auto"
    )
    return source == "manual"


def _worst_case_win_rate_moe95_pct(trade_count: int) -> float | None:
    if trade_count <= 0:
        return None
    # Worst-case binomial standard error occurs at p=0.5.
    return round(100.0 * 1.96 * sqrt(0.25 / float(trade_count)), 4)


def _cell_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("strategy") or ""),
        str(row.get("symbol") or "").upper(),
        str(row.get("timeframe") or "").upper(),
    )


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _assess_cell(
    key: tuple[str, str, str],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_rows = [
        row for row in rows
        if float(row.get("confidence") or 0.0) >= _BASELINE_THRESHOLD
    ]
    below_baseline = len(rows) - len(baseline_rows)
    trade_count = len(baseline_rows)
    wins = sum(1 for row in baseline_rows if float(row.get("realized_pnl") or 0.0) > 0.0)
    losses = sum(1 for row in baseline_rows if float(row.get("realized_pnl") or 0.0) < 0.0)
    breakeven = trade_count - wins - losses

    r_eligible = sum(1 for row in baseline_rows if _initial_risk_amount(row) is not None)
    r_coverage_ratio = (r_eligible / trade_count) if trade_count else 0.0

    bucket_counts = Counter(
        label
        for row in baseline_rows
        if (label := _bucket_label(float(row.get("confidence") or 0.0))) is not None
    )
    populated_buckets = [
        label
        for label, _, _ in _EXECUTION_BUCKETS
        if bucket_counts.get(label, 0) >= _MIN_TRADES_PER_POPULATED_BUCKET
    ]

    timestamps = [
        parsed
        for row in baseline_rows
        if (parsed := _parse_time(row.get("closed_at"))) is not None
    ]
    first_closed_at = min(timestamps).isoformat() if timestamps else None
    last_closed_at = max(timestamps).isoformat() if timestamps else None

    broker_trades = sum(1 for row in baseline_rows if row.get("broker_position_id") is not None)
    pnl_sources = Counter(
        str(row.get("realized_pnl_source") or "unknown")
        for row in baseline_rows
    )
    moe95 = _worst_case_win_rate_moe95_pct(trade_count)

    checks = {
        "minimum_baseline_trades": {
            "passed": trade_count >= _MIN_BASELINE_TRADES,
            "observed": trade_count,
            "required": _MIN_BASELINE_TRADES,
        },
        "worst_case_win_rate_moe95": {
            "passed": moe95 is not None and moe95 <= _MAX_WORST_CASE_WIN_RATE_MOE95_PCT,
            "observed_pct_points": moe95,
            "maximum_pct_points": _MAX_WORST_CASE_WIN_RATE_MOE95_PCT,
        },
        "r_coverage": {
            "passed": r_coverage_ratio >= _MIN_R_COVERAGE_RATIO,
            "observed_ratio": round(r_coverage_ratio, 4),
            "required_ratio": _MIN_R_COVERAGE_RATIO,
            "eligible_trades": r_eligible,
        },
        "outcome_diversity": {
            "passed": wins >= _MIN_WINS and losses >= _MIN_LOSSES,
            "wins": wins,
            "losses": losses,
            "minimum_each": _MIN_WINS,
        },
        "strength_bucket_coverage": {
            "passed": len(populated_buckets) >= _MIN_POPULATED_BUCKETS,
            "populated_buckets": populated_buckets,
            "minimum_buckets": _MIN_POPULATED_BUCKETS,
            "minimum_trades_per_bucket": _MIN_TRADES_PER_POPULATED_BUCKET,
        },
    }
    failed_checks = [name for name, check in checks.items() if not bool(check["passed"])]
    sufficient = not failed_checks

    strategy, symbol, timeframe = key
    return {
        "strategy": strategy,
        "symbol": symbol,
        "timeframe": timeframe,
        "status": "sufficient_for_threshold_study" if sufficient else "insufficient_evidence",
        "sufficient_for_threshold_study": sufficient,
        "failed_checks": failed_checks,
        "sample": {
            "linked_closed_auto_trades": len(rows),
            "baseline_60pct_trade_count": trade_count,
            "below_baseline_trade_count": below_baseline,
            "wins": wins,
            "losses": losses,
            "breakeven": breakeven,
            "r_eligible_trades": r_eligible,
            "r_coverage_ratio": round(r_coverage_ratio, 4),
            "broker_trade_count": broker_trades,
            "paper_trade_count": trade_count - broker_trades,
            "realized_pnl_sources": dict(sorted(pnl_sources.items())),
            "first_closed_at": first_closed_at,
            "last_closed_at": last_closed_at,
        },
        "bucket_counts": {
            label: int(bucket_counts.get(label, 0))
            for label, _, _ in _EXECUTION_BUCKETS
        },
        "checks": checks,
    }


def build_threshold_sufficiency_assessment(
    *,
    strategy: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> dict[str, Any]:
    """Assess whether each calibration cell has enough evidence to study thresholds.

    This is a fail-closed screening gate only. It does not rank candidate
    thresholds, recommend a replacement for 60%, or change execution settings.
    """
    source_rows = list_confidence_calibration_outcomes(
        strategy=strategy,
        symbol=symbol,
        timeframe=timeframe,
    )
    automatic_rows = [row for row in source_rows if not _is_manual(row)]

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in automatic_rows:
        grouped.setdefault(_cell_key(row), []).append(row)

    assessments = [
        _assess_cell(key, rows)
        for key, rows in sorted(grouped.items())
    ]
    sufficient_cells = sum(
        1 for item in assessments if item["sufficient_for_threshold_study"]
    )

    return {
        "status": (
            "no_data"
            if not assessments
            else (
                "eligible_cells_present"
                if sufficient_cells
                else "insufficient_evidence"
            )
        ),
        "baseline_threshold": _BASELINE_THRESHOLD,
        "semantics": {
            "score_name": "signal_strength",
            "probability_label_supported": False,
            "automatic_threshold_change": False,
            "selected_threshold": None,
            "description": (
                "This report only decides whether an exact strategy × symbol × timeframe "
                "cell has enough closed automatic-trade evidence to begin a separate "
                "leakage-safe threshold comparison study."
            ),
        },
        "screening_policy": {
            "minimum_baseline_trades": _MIN_BASELINE_TRADES,
            "maximum_worst_case_win_rate_moe95_pct_points": _MAX_WORST_CASE_WIN_RATE_MOE95_PCT,
            "minimum_r_coverage_ratio": _MIN_R_COVERAGE_RATIO,
            "minimum_wins": _MIN_WINS,
            "minimum_losses": _MIN_LOSSES,
            "minimum_populated_strength_buckets": _MIN_POPULATED_BUCKETS,
            "minimum_trades_per_populated_bucket": _MIN_TRADES_PER_POPULATED_BUCKET,
            "note": (
                "These are conservative operational screening floors, not proof that a "
                "threshold is optimal. Passing only permits a later development/holdout "
                "threshold study; it does not authorize an execution-setting change."
            ),
        },
        "filters": {
            "strategy": strategy,
            "symbol": symbol.upper() if symbol else None,
            "timeframe": timeframe.upper() if timeframe else None,
        },
        "summary": {
            "cell_count": len(assessments),
            "sufficient_cell_count": sufficient_cells,
            "insufficient_cell_count": len(assessments) - sufficient_cells,
            "excluded_manual_trade_count": len(source_rows) - len(automatic_rows),
        },
        "cells": assessments,
    }
