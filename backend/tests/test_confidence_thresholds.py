from __future__ import annotations

from fastapi.testclient import TestClient

from backend.services import confidence_thresholds as thresholds


def _row(
    *,
    confidence: float,
    realized_pnl: float,
    strategy: str = "sma_cross",
    symbol: str = "NAS100",
    timeframe: str = "M5",
    source: str = "auto",
    stop_loss: float | None = 95.0,
    broker: bool = False,
    index: int = 0,
) -> dict:
    return {
        "position_id": index + 1,
        "intent_id": index + 1001,
        "strategy": strategy,
        "symbol": symbol,
        "timeframe": timeframe,
        "confidence": confidence,
        "realized_pnl": realized_pnl,
        "signal_stop_loss": stop_loss,
        "entry_price": 100.0,
        "quantity": 1.0,
        "cash_per_price_unit_per_lot": 1.0,
        "broker_position_id": index + 5000 if broker else None,
        "realized_pnl_source": "ctrader_deal" if broker else "paper_estimate",
        "closed_at": f"2026-09-{(index % 28) + 1:02d}T12:00:00",
        "intent_details": {"source": source},
    }


def _sufficient_rows(
    *,
    strategy: str = "sma_cross",
    symbol: str = "NAS100",
    timeframe: str = "M5",
) -> list[dict]:
    confidences = [0.62, 0.67, 0.72, 0.77, 0.82]
    rows: list[dict] = []
    for index in range(100):
        rows.append(
            _row(
                confidence=confidences[index // 20],
                realized_pnl=5.0 if index % 2 == 0 else -5.0,
                strategy=strategy,
                symbol=symbol,
                timeframe=timeframe,
                broker=index % 4 == 0,
                index=index,
            )
        )
    return rows


def test_small_sample_fails_closed_without_threshold_selection(monkeypatch) -> None:
    rows = [
        _row(
            confidence=0.62,
            realized_pnl=5.0 if index % 2 == 0 else -5.0,
            index=index,
        )
        for index in range(20)
    ]
    monkeypatch.setattr(
        thresholds,
        "list_confidence_calibration_outcomes",
        lambda **kwargs: rows,
    )

    result = thresholds.build_threshold_sufficiency_assessment()

    assert result["status"] == "insufficient_evidence"
    assert result["semantics"]["automatic_threshold_change"] is False
    assert result["semantics"]["selected_threshold"] is None
    assert result["summary"]["sufficient_cell_count"] == 0
    cell = result["cells"][0]
    assert cell["sufficient_for_threshold_study"] is False
    assert "minimum_baseline_trades" in cell["failed_checks"]
    assert "worst_case_win_rate_moe95" in cell["failed_checks"]
    assert "strength_bucket_coverage" in cell["failed_checks"]


def test_well_distributed_100_trade_sample_passes_screening_gate(monkeypatch) -> None:
    rows = _sufficient_rows()
    monkeypatch.setattr(
        thresholds,
        "list_confidence_calibration_outcomes",
        lambda **kwargs: rows,
    )

    result = thresholds.build_threshold_sufficiency_assessment()

    assert result["status"] == "eligible_cells_present"
    assert result["summary"] == {
        "cell_count": 1,
        "sufficient_cell_count": 1,
        "insufficient_cell_count": 0,
        "excluded_manual_trade_count": 0,
    }
    cell = result["cells"][0]
    assert cell["status"] == "sufficient_for_threshold_study"
    assert cell["failed_checks"] == []
    assert cell["sample"]["baseline_60pct_trade_count"] == 100
    assert cell["sample"]["wins"] == 50
    assert cell["sample"]["losses"] == 50
    assert cell["sample"]["r_eligible_trades"] == 100
    assert cell["sample"]["r_coverage_ratio"] == 1.0
    assert cell["sample"]["broker_trade_count"] == 25
    assert cell["bucket_counts"] == {
        "60-65%": 20,
        "65-70%": 20,
        "70-75%": 20,
        "75-80%": 20,
        "80%+": 20,
    }
    assert cell["checks"]["worst_case_win_rate_moe95"]["observed_pct_points"] <= 10.0


def test_r_coverage_gate_blocks_threshold_study(monkeypatch) -> None:
    rows = _sufficient_rows()
    for index, row in enumerate(rows):
        if index >= 79:
            row["signal_stop_loss"] = None
    monkeypatch.setattr(
        thresholds,
        "list_confidence_calibration_outcomes",
        lambda **kwargs: rows,
    )

    result = thresholds.build_threshold_sufficiency_assessment()
    cell = result["cells"][0]

    assert cell["sample"]["r_eligible_trades"] == 79
    assert cell["sample"]["r_coverage_ratio"] == 0.79
    assert cell["sufficient_for_threshold_study"] is False
    assert cell["failed_checks"] == ["r_coverage"]


def test_manual_trades_do_not_inflate_sufficiency(monkeypatch) -> None:
    rows = _sufficient_rows()
    manual_rows = [
        _row(
            confidence=0.90,
            realized_pnl=20.0,
            source="manual",
            index=200 + index,
        )
        for index in range(50)
    ]
    monkeypatch.setattr(
        thresholds,
        "list_confidence_calibration_outcomes",
        lambda **kwargs: [*rows, *manual_rows],
    )

    result = thresholds.build_threshold_sufficiency_assessment()

    assert result["summary"]["excluded_manual_trade_count"] == 50
    assert result["cells"][0]["sample"]["linked_closed_auto_trades"] == 100
    assert result["cells"][0]["sample"]["baseline_60pct_trade_count"] == 100


def test_assessment_is_separate_for_each_strategy_symbol_timeframe(monkeypatch) -> None:
    sufficient = _sufficient_rows(
        strategy="sma_cross",
        symbol="NAS100",
        timeframe="M5",
    )
    insufficient = [
        _row(
            confidence=0.72,
            realized_pnl=4.0 if index % 2 == 0 else -4.0,
            strategy="breakout",
            symbol="XAUUSD",
            timeframe="M15",
            index=300 + index,
        )
        for index in range(20)
    ]
    monkeypatch.setattr(
        thresholds,
        "list_confidence_calibration_outcomes",
        lambda **kwargs: [*sufficient, *insufficient],
    )

    result = thresholds.build_threshold_sufficiency_assessment()

    assert result["summary"]["cell_count"] == 2
    assert result["summary"]["sufficient_cell_count"] == 1
    by_key = {
        (item["strategy"], item["symbol"], item["timeframe"]): item
        for item in result["cells"]
    }
    assert by_key[("sma_cross", "NAS100", "M5")]["sufficient_for_threshold_study"] is True
    assert by_key[("breakout", "XAUUSD", "M15")]["sufficient_for_threshold_study"] is False


def test_no_data_returns_fail_closed_report(monkeypatch) -> None:
    monkeypatch.setattr(
        thresholds,
        "list_confidence_calibration_outcomes",
        lambda **kwargs: [],
    )

    result = thresholds.build_threshold_sufficiency_assessment()

    assert result["status"] == "no_data"
    assert result["summary"]["cell_count"] == 0
    assert result["cells"] == []
    assert result["semantics"]["selected_threshold"] is None


def test_threshold_sufficiency_endpoint_forwards_filters(monkeypatch) -> None:
    captured: dict = {}

    def _fake_build(**kwargs):
        captured.update(kwargs)
        return {
            "status": "insufficient_evidence",
            "semantics": {
                "automatic_threshold_change": False,
                "selected_threshold": None,
            },
            "summary": {"cell_count": 1},
            "cells": [],
        }

    monkeypatch.setattr(
        "backend.api.router.build_threshold_sufficiency_assessment",
        _fake_build,
    )

    from backend.app import app

    with TestClient(app) as client:
        response = client.get(
            "/api/studio/confidence-threshold-sufficiency",
            params={
                "strategy": "breakout",
                "symbol": "XAUUSD",
                "timeframe": "M15",
            },
        )

    assert response.status_code == 200
    assert response.json()["semantics"]["automatic_threshold_change"] is False
    assert response.json()["semantics"]["selected_threshold"] is None
    assert captured == {
        "strategy": "breakout",
        "symbol": "XAUUSD",
        "timeframe": "M15",
    }
