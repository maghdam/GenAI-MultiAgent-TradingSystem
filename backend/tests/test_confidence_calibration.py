from __future__ import annotations

from fastapi.testclient import TestClient

from backend.services.confidence_calibration import build_confidence_calibration
from backend.storage.repositories import (
    add_trade_audit,
    close_paper_position,
    create_order_intent,
    open_paper_position,
    update_order_intent_status,
)


def _record_closed_trade(
    *,
    confidence: float,
    realized_pnl: float,
    strategy: str = "sma_cross",
    symbol: str = "NAS100",
    timeframe: str = "M5",
    source: str = "auto",
    account_currency: str = "USD",
    entry_price: float = 100.0,
    stop_loss: float | None = 95.0,
    quantity: float = 1.0,
    cash_per_price_unit_per_lot: float = 1.0,
    broker_position_id: int | None = None,
) -> int:
    intent = create_order_intent(
        symbol=symbol,
        timeframe=timeframe,
        strategy=strategy,
        direction="long",
        intent_type="open",
        status="accepted",
        confidence=confidence,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=120.0,
        quantity=quantity,
        rationale="confidence calibration fixture",
        details={"source": source},
    )
    position = open_paper_position(
        symbol=symbol,
        timeframe=timeframe,
        strategy=strategy,
        direction="long",
        quantity=quantity,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=120.0,
        account_currency=account_currency,
        cash_per_price_unit_per_lot=cash_per_price_unit_per_lot,
        instrument_spec_source="test_fixture",
        broker_position_id=broker_position_id,
    )
    update_order_intent_status(
        intent.id,
        "executed",
        {"opened_position_id": position.id},
        reason="paper_position_opened",
    )
    add_trade_audit(
        event_type="ctrader_demo_order_executed" if broker_position_id is not None else "paper_signal_open",
        symbol=symbol,
        timeframe=timeframe,
        strategy=strategy,
        position_id=position.id,
        intent_id=intent.id,
        summary="Opened calibration fixture position.",
        details={},
    )
    close_paper_position(
        position.id,
        exit_price=entry_price + 1.0,
        reason="fixture_close",
        realized_pnl_override=realized_pnl,
        realized_pnl_source="ctrader_deal" if broker_position_id is not None else "paper_estimate",
    )
    return position.id


def _bucket(result: dict, label: str) -> dict:
    return next(item for item in result["buckets"] if item["bucket"] == label)


def test_calibration_reports_bucket_metrics_without_probability_claim() -> None:
    _record_closed_trade(confidence=0.62, realized_pnl=10.0)
    _record_closed_trade(confidence=0.64, realized_pnl=-5.0)
    _record_closed_trade(confidence=0.72, realized_pnl=5.0)
    _record_closed_trade(confidence=0.82, realized_pnl=-10.0)

    result = build_confidence_calibration()

    assert result["status"] == "descriptive_only"
    assert result["semantics"]["score_name"] == "signal_strength"
    assert result["semantics"]["probability_label_supported"] is False
    assert result["semantics"]["threshold_changes_automatic"] is False
    assert result["sample"] == {
        "linked_closed_trades": 4,
        "r_eligible_trades": 4,
        "r_ineligible_trades": 0,
        "excluded_manual_trades": 0,
    }

    low = _bucket(result, "60-65%")
    assert low["trade_count"] == 2
    assert low["wins"] == 1
    assert low["losses"] == 1
    assert low["win_rate_pct"] == 50.0
    assert low["expectancy"] == {"USD": 2.5}
    assert low["average_r"] == 0.5
    assert low["drawdown_contribution_r"] == 1.0

    mid = _bucket(result, "70-75%")
    assert mid["trade_count"] == 1
    assert mid["win_rate_pct"] == 100.0
    assert mid["expectancy"] == {"USD": 5.0}
    assert mid["average_r"] == 1.0

    high = _bucket(result, "80%+")
    assert high["trade_count"] == 1
    assert high["win_rate_pct"] == 0.0
    assert high["expectancy"] == {"USD": -10.0}
    assert high["average_r"] == -2.0
    assert high["drawdown_contribution_r"] == 2.0

    assert result["overall"]["trade_count"] == 4
    assert result["overall"]["win_rate_pct"] == 50.0
    assert result["overall"]["expectancy"] == {"USD": 0.0}
    assert result["overall"]["average_r"] == 0.0
    assert result["overall"]["max_drawdown_r"] == 2.0
    assert result["overall"]["drawdown_contribution_r"] == 3.0


def test_bucket_boundaries_are_lower_inclusive_and_upper_exclusive() -> None:
    _record_closed_trade(confidence=0.599, realized_pnl=1.0)
    _record_closed_trade(confidence=0.60, realized_pnl=1.0)
    _record_closed_trade(confidence=0.65, realized_pnl=1.0)
    _record_closed_trade(confidence=0.70, realized_pnl=1.0)
    _record_closed_trade(confidence=0.75, realized_pnl=1.0)
    _record_closed_trade(confidence=0.80, realized_pnl=1.0)

    result = build_confidence_calibration()

    assert _bucket(result, "<60%")["trade_count"] == 1
    assert _bucket(result, "60-65%")["trade_count"] == 1
    assert _bucket(result, "65-70%")["trade_count"] == 1
    assert _bucket(result, "70-75%")["trade_count"] == 1
    assert _bucket(result, "75-80%")["trade_count"] == 1
    assert _bucket(result, "80%+")["trade_count"] == 1


def test_manual_trades_are_excluded_by_default_but_can_be_requested() -> None:
    _record_closed_trade(confidence=0.66, realized_pnl=5.0, source="auto")
    _record_closed_trade(confidence=0.90, realized_pnl=7.0, source="manual")

    default_result = build_confidence_calibration()
    assert default_result["sample"]["linked_closed_trades"] == 1
    assert default_result["sample"]["excluded_manual_trades"] == 1
    assert _bucket(default_result, "80%+")["trade_count"] == 0

    included_result = build_confidence_calibration(include_manual=True)
    assert included_result["sample"]["linked_closed_trades"] == 2
    assert included_result["sample"]["excluded_manual_trades"] == 0
    assert _bucket(included_result, "80%+")["trade_count"] == 1


def test_expectancy_keeps_account_currencies_separate_and_r_skips_missing_stop() -> None:
    _record_closed_trade(
        confidence=0.71,
        realized_pnl=10.0,
        account_currency="CHF",
        stop_loss=None,
    )
    _record_closed_trade(
        confidence=0.72,
        realized_pnl=5.0,
        account_currency="USD",
        stop_loss=95.0,
    )

    result = build_confidence_calibration()
    bucket = _bucket(result, "70-75%")

    assert bucket["trade_count"] == 2
    assert bucket["expectancy"] == {"CHF": 10.0, "USD": 5.0}
    assert bucket["r_trade_count"] == 1
    assert bucket["average_r"] == 1.0
    assert result["sample"]["r_eligible_trades"] == 1
    assert result["sample"]["r_ineligible_trades"] == 1


def test_calibration_filters_strategy_symbol_and_timeframe() -> None:
    _record_closed_trade(
        confidence=0.67,
        realized_pnl=4.0,
        strategy="sma_cross",
        symbol="NAS100",
        timeframe="M5",
    )
    _record_closed_trade(
        confidence=0.76,
        realized_pnl=-3.0,
        strategy="breakout",
        symbol="XAUUSD",
        timeframe="M15",
    )

    result = build_confidence_calibration(
        strategy="breakout",
        symbol="xauusd",
        timeframe="m15",
    )

    assert result["filters"]["strategy"] == "breakout"
    assert result["filters"]["symbol"] == "XAUUSD"
    assert result["filters"]["timeframe"] == "M15"
    assert result["sample"]["linked_closed_trades"] == 1
    assert _bucket(result, "75-80%")["trade_count"] == 1
    assert result["overall"]["expectancy"] == {"USD": -3.0}


def test_unlinked_closed_position_is_not_guessed_into_calibration() -> None:
    position = open_paper_position(
        symbol="US30",
        timeframe="M5",
        strategy="sma_cross",
        direction="long",
        quantity=1.0,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=120.0,
        account_currency="USD",
        cash_per_price_unit_per_lot=1.0,
        instrument_spec_source="test_fixture",
    )
    close_paper_position(
        position.id,
        exit_price=101.0,
        reason="unlinked_fixture",
        realized_pnl_override=4.0,
    )

    result = build_confidence_calibration()

    assert result["status"] == "no_data"
    assert result["sample"]["linked_closed_trades"] == 0
    assert result["overall"]["trade_count"] == 0


def test_confidence_calibration_endpoint_forwards_filters(monkeypatch) -> None:
    captured: dict = {}

    def _fake_build(**kwargs):
        captured.update(kwargs)
        return {
            "status": "descriptive_only",
            "semantics": {"probability_label_supported": False},
            "sample": {"linked_closed_trades": 3},
            "buckets": [],
        }

    monkeypatch.setattr("backend.api.router.build_confidence_calibration", _fake_build)

    from backend.app import app

    with TestClient(app) as client:
        response = client.get(
            "/api/studio/confidence-calibration",
            params={
                "strategy": "breakout",
                "symbol": "XAUUSD",
                "timeframe": "M5",
                "include_manual": "true",
            },
        )

    assert response.status_code == 200
    assert response.json()["semantics"]["probability_label_supported"] is False
    assert captured == {
        "strategy": "breakout",
        "symbol": "XAUUSD",
        "timeframe": "M5",
        "include_manual": True,
    }
