from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.domain.models import StrategyAnalysis
from backend.services import runtime_strategy_validation as validation


def _frame(prices: list[float], timeframe: str = "5min") -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=len(prices), freq=timeframe, tz="UTC")
    close = pd.Series(prices, index=index, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.5
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.5
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100.0,
        },
        index=index,
    )


def _analysis(
    signal: str,
    *,
    stop_loss: float | None = None,
    take_profit: float | None = None,
) -> StrategyAnalysis:
    return StrategyAnalysis(
        symbol="TEST",
        timeframe="M5",
        strategy="sma_cross",
        signal=signal,
        confidence=0.8 if signal != "no_trade" else 0.0,
        entry_price=100.0,
        stop_loss=stop_loss,
        take_profit=take_profit,
        reasons=["runtime validation fixture"],
        context={},
    )


def _synthetic_market(bars: int = 180) -> pd.DataFrame:
    x = np.arange(bars, dtype=float)
    close = 100.0 + 0.025 * x + 5.0 * np.sin(x / 5.0) + 1.5 * np.sin(x / 17.0)
    # Add deterministic jumps so the breakout strategy sees both directions.
    close = close.copy()
    for idx in range(30, bars, 45):
        close[idx : min(idx + 4, bars)] += 4.0
    for idx in range(52, bars, 55):
        close[idx : min(idx + 4, bars)] -= 4.0
    return _frame(close.tolist())


def test_runtime_registry_contains_exact_deterministic_strategies() -> None:
    assert validation.runtime_strategy_keys() == ["breakout", "rsi_reversal", "sma_cross"]


def test_runtime_no_trade_holds_existing_position_until_later_action() -> None:
    frame = _frame([100.0, 100.0, 105.0, 110.0])
    analyses = [
        _analysis("long", stop_loss=50.0, take_profit=200.0),
        _analysis("no_trade"),
        _analysis("no_trade"),
        _analysis("no_trade"),
    ]

    result = validation._runtime_accounting(
        df=frame,
        analyses=analyses,
        start_bar=0,
        end_bar=len(frame),
        timeframe="M5",
        fee_bps=0.0,
        slippage_bps=0.0,
        spread_bps=0.0,
        position_size_pct=100.0,
    )

    assert result["Number of Trades"] == 1.0
    assert result["Marked Open Trades"] == 1.0
    assert result["Protective Exits"] == 0.0
    assert result["Total Return [%]"] == pytest.approx(10.0)
    assert "Hold existing position" in result["Runtime no_trade Semantics"]


def test_completed_bar_signal_cannot_capture_same_bar_jump() -> None:
    index = pd.date_range("2026-01-01", periods=2, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [100.0, 200.0],
            "high": [200.0, 200.5],
            "low": [99.5, 199.5],
            "close": [200.0, 200.0],
            "volume": [100.0, 100.0],
        },
        index=index,
    )
    analyses = [
        _analysis("long", stop_loss=50.0, take_profit=400.0),
        _analysis("no_trade"),
    ]

    result = validation._runtime_accounting(
        df=frame,
        analyses=analyses,
        start_bar=0,
        end_bar=2,
        timeframe="M5",
        fee_bps=0.0,
        slippage_bps=0.0,
        spread_bps=0.0,
        position_size_pct=100.0,
    )

    assert result["Total Return [%]"] == pytest.approx(0.0)
    assert result["Execution Timing"] == "Completed-bar decision; execute at next bar open."


def test_protective_target_exit_uses_runtime_fill_and_costs() -> None:
    frame = _frame([100.0, 104.0, 106.0])
    analyses = [
        _analysis("long", stop_loss=95.0, take_profit=105.0),
        _analysis("no_trade"),
        _analysis("no_trade"),
    ]

    gross = validation._runtime_accounting(
        df=frame,
        analyses=analyses,
        start_bar=0,
        end_bar=3,
        timeframe="M5",
        fee_bps=0.0,
        slippage_bps=0.0,
        spread_bps=0.0,
        position_size_pct=100.0,
    )
    net = validation._runtime_accounting(
        df=frame,
        analyses=analyses,
        start_bar=0,
        end_bar=3,
        timeframe="M5",
        fee_bps=1.0,
        slippage_bps=2.0,
        spread_bps=4.0,
        position_size_pct=100.0,
    )

    assert gross["Closed Trades"] == 1.0
    assert gross["Take-Profit Exits"] == 1.0
    assert gross["Total Return [%]"] == pytest.approx(5.0)
    assert net["Total Return [%]"] < gross["Total Return [%]"]


@pytest.mark.parametrize("strategy_key", ["sma_cross", "rsi_reversal", "breakout"])
def test_runtime_strategy_audit_covers_phase_51_dimensions(monkeypatch, strategy_key: str) -> None:
    frame = _synthetic_market()

    monkeypatch.setattr(
        validation.data_fetcher,
        "fetch_data",
        lambda symbol, timeframe, num_bars: (frame.copy(), float(frame["close"].iloc[-1])),
    )

    result = validation.run_runtime_strategy_audit(
        strategy=strategy_key,
        symbol="TEST",
        timeframe="M5",
        num_bars=len(frame),
        fee_bps=1.0,
        slippage_bps=0.5,
        spread_bps=1.0,
        position_size_pct=75.0,
    )

    assert result["strategy"] == strategy_key
    assert result["Development"]["Development Bars"] == 125
    assert result["Out of Sample"]["Holdout Bars"] == 55
    assert result["Development"]["Data End"] < result["Out of Sample"]["Data Start"]
    assert len(result["Regime Analysis"]) == 3
    assert all(item["Bars"] == 60 for item in result["Regime Analysis"])
    assert result["Costs"]["Gross Return [%]"] >= result["Costs"]["Net Return [%]"] - 1e-9
    assert "Expectancy [%]" in result["Full Backtest"]
    assert "Trades/Day" in result["Full Backtest"]
    assert "Max Drawdown [%]" in result["Full Backtest"]
    assert result["Parameter Sensitivity"]["Dataset"].startswith("Development window")
    cases = result["Parameter Sensitivity"]["Cases"]
    assert len(cases) >= 5
    assert cases[0]["Case"] == "baseline"
    assert all("Expectancy [%]" in item for item in cases)
    assert math.isfinite(float(result["Full Backtest"]["Total Return [%]"]))


def test_runtime_strategy_matrix_evaluates_every_strategy_target_pair(monkeypatch) -> None:
    frame = _synthetic_market()

    calls: list[tuple[str, str]] = []

    def fake_fetch(symbol: str, timeframe: str, num_bars: int):
        calls.append((symbol, timeframe))
        return frame.copy(), float(frame["close"].iloc[-1])

    def fake_audit(**kwargs):
        return {
            "strategy": kwargs["strategy_key"],
            "symbol": kwargs["symbol"].upper(),
            "timeframe": kwargs["timeframe"].upper(),
            "Full Backtest": {"Number of Trades": 1.0},
        }

    monkeypatch.setattr(validation.data_fetcher, "fetch_data", fake_fetch)
    monkeypatch.setattr(validation, "_audit_on_frame", fake_audit)

    result = validation.run_runtime_strategy_matrix(
        targets=[("XAUUSD", "M5"), ("US100", "H1")],
        strategies=["sma_cross", "rsi_reversal", "breakout"],
        num_bars=180,
    )

    assert calls == [("XAUUSD", "M5"), ("US100", "H1")]
    assert len(result["Results"]) == 6
    assert all(item["status"] == "ok" for item in result["Results"])
    assert {(item["strategy"], item["symbol"], item["timeframe"]) for item in result["Results"]} == {
        ("sma_cross", "XAUUSD", "M5"),
        ("rsi_reversal", "XAUUSD", "M5"),
        ("breakout", "XAUUSD", "M5"),
        ("sma_cross", "US100", "H1"),
        ("rsi_reversal", "US100", "H1"),
        ("breakout", "US100", "H1"),
    }


def test_runtime_strategy_audit_endpoint_forwards_assumptions(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_LEGACY_CONTROLLER_ON_BOOT", "0")

    captured: dict = {}

    def fake_audit(**kwargs):
        captured.update(kwargs)
        return {"strategy": kwargs["strategy"], "Full Backtest": {"Number of Trades": 3.0}}

    monkeypatch.setattr("backend.api.router.run_runtime_strategy_audit", fake_audit)

    from backend.app import app

    with TestClient(app) as client:
        response = client.get(
            "/api/studio/runtime-strategy-audit",
            params={
                "strategy": "breakout",
                "symbol": "XAUUSD",
                "timeframe": "H1",
                "num_bars": 900,
                "fee_bps": 1.5,
                "slippage_bps": 0.75,
                "spread_bps": 2.0,
                "position_size_pct": 60.0,
            },
        )

    assert response.status_code == 200
    assert captured == {
        "strategy": "breakout",
        "symbol": "XAUUSD",
        "timeframe": "H1",
        "num_bars": 900,
        "fee_bps": 1.5,
        "slippage_bps": 0.75,
        "spread_bps": 2.0,
        "position_size_pct": 60.0,
    }
    assert response.json()["Full Backtest"]["Number of Trades"] == 3.0


def test_unknown_runtime_strategy_is_rejected() -> None:
    frame = _synthetic_market()

    with pytest.raises(Exception, match="Runtime strategy must be one of"):
        validation._audit_on_frame(
            strategy_key="generated_candidate",
            symbol="TEST",
            timeframe="M5",
            df=frame,
            params=None,
            fee_bps=0.0,
            slippage_bps=0.0,
            spread_bps=0.0,
            position_size_pct=100.0,
        )
