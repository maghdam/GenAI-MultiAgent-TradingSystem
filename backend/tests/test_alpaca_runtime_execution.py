from __future__ import annotations

from datetime import UTC, datetime

import pytest

from backend.domain.models import EngineConfig, StrategyAnalysis, WatchlistItem
from backend.services import alpaca_runtime
from backend.services.risk_engine import RiskDecision
from backend.storage.repositories import list_trade_audits


def _config(**overrides) -> EngineConfig:
    payload = EngineConfig(
        enabled=True,
        paper_autotrade=False,
        ctrader_autotrade=False,
        kill_switch=False,
        min_confidence=0.60,
        daily_loss_limit_pct=2.0,
        max_daily_trades=12,
        max_open_positions=3,
        max_positions_per_symbol=1,
        cooldown_minutes=0,
        session_filter_enabled=False,
        require_stops=True,
    ).model_dump()
    payload.update(overrides)
    return EngineConfig(**payload)


def _watch(**overrides) -> WatchlistItem:
    payload = WatchlistItem(
        symbol="SPY",
        timeframe="M5",
        strategy="sma_cross",
        enabled=True,
        trading_enabled=True,
        venue="alpaca",
        alpaca_asset_class="us_equity",
        alpaca_size_mode="notional",
        alpaca_size_value=25.0,
        lot_size=0.1,
    ).model_dump()
    payload.update(overrides)
    return WatchlistItem(**payload)


def _analysis(**overrides) -> StrategyAnalysis:
    payload = StrategyAnalysis(
        symbol="SPY",
        timeframe="M5",
        strategy="sma_cross",
        signal="long",
        confidence=0.82,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        reasons=["trusted runtime test"],
        context={},
    ).model_dump()
    payload.update(overrides)
    return StrategyAnalysis(**payload)


def _account() -> dict[str, object]:
    return {
        "provider": "alpaca",
        "environment": "paper",
        "status": "ACTIVE",
        "currency": "USD",
        "equity": "100000",
        "trading_blocked": False,
        "account_blocked": False,
        "trade_suspended_by_user": False,
        "shorting_enabled": True,
    }


def _position(*, side: str = "long", symbol: str = "SPY") -> dict[str, object]:
    return {
        "asset_id": "asset-spy",
        "symbol": symbol,
        "asset_class": "us_equity",
        "exchange": "NYSE",
        "side": side,
        "qty": "1",
        "qty_available": "1",
        "avg_entry_price": "100",
        "current_price": "101",
        "market_value": "101",
        "cost_basis": "100",
        "unrealized_pl": "1",
        "unrealized_plpc": "0.01",
    }


@pytest.fixture
def broker_truth(monkeypatch):
    monkeypatch.setattr(alpaca_runtime, "get_account", _account)
    monkeypatch.setattr(alpaca_runtime, "list_positions", lambda: [])
    monkeypatch.setattr(alpaca_runtime, "get_open_position", lambda symbol: None)
    monkeypatch.setattr(alpaca_runtime, "list_paper_positions", lambda status=None: [])
    monkeypatch.setattr(alpaca_runtime, "daily_realized_pnl", lambda: 0.0)
    monkeypatch.setattr(alpaca_runtime, "daily_trade_count", lambda: 0)
    monkeypatch.setattr(alpaca_runtime, "list_order_intents", lambda limit=500: [])
    monkeypatch.setattr(alpaca_runtime, "list_trade_audits", lambda limit=500: [])


def _run(watch: WatchlistItem, analysis: StrategyAnalysis):
    return alpaca_runtime.execute_alpaca_runtime_signal(
        config=_config(),
        watch_item=watch,
        analysis=analysis,
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC).replace(tzinfo=None),
        bar_snapshot={"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0},
    )


def test_runtime_long_notional_routes_once_to_phase_10_9_adapter(monkeypatch, broker_truth) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        alpaca_runtime,
        "evaluate_risk",
        lambda **kwargs: RiskDecision(
            accepted=True,
            reasons=["Signal passed risk checks."],
            details={"execution_environment": "alpaca_paper"},
            intent_type="open",
        ),
    )

    def _submit(**kwargs):
        captured.update(kwargs)
        return {
            "action_taken": True,
            "intent_id": 44,
            "status": "accepted",
            "summary": "acknowledged",
            "broker_order": {"id": "order-44", "status": "pending_new"},
        }

    monkeypatch.setattr(alpaca_runtime, "submit_paper_market_order", _submit)
    monkeypatch.setattr(
        alpaca_runtime,
        "close_paper_position",
        lambda **kwargs: pytest.fail("entry signal must not close"),
    )

    result = _run(_watch(), _analysis())

    assert result.action_taken is True
    assert result.intent_id == 44
    assert result.status == "accepted"
    assert result.mode == "alpaca_paper"
    assert captured["symbol"] == "SPY"
    assert captured["asset_class"] == "us_equity"
    assert captured["side"] == "buy"
    assert captured["qty"] is None
    assert captured["notional"] == 25.0
    assert captured["confidence"] == pytest.approx(0.82)
    assert captured["stop_loss"] == 99.0
    assert captured["take_profit"] == 102.0
    runtime = captured["execution_details"]["runtime_execution"]
    assert runtime["source"] == "trusted_strategy_auto_loop"
    assert runtime["size_mode"] == "notional"
    assert runtime["automatic_resubmission"] is False


def test_runtime_quantity_short_equity_uses_sell_without_ctrader_lots(monkeypatch, broker_truth) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        alpaca_runtime,
        "evaluate_risk",
        lambda **kwargs: RiskDecision(
            accepted=True,
            reasons=["Signal passed risk checks."],
            details={},
            intent_type="open",
        ),
    )
    monkeypatch.setattr(
        alpaca_runtime,
        "submit_paper_market_order",
        lambda **kwargs: captured.update(kwargs) or {
            "action_taken": True,
            "intent_id": 45,
            "status": "executed",
            "summary": "filled",
        },
    )

    result = _run(
        _watch(alpaca_size_mode="qty", alpaca_size_value=0.25),
        _analysis(
            signal="short",
            stop_loss=101.0,
            take_profit=98.0,
        ),
    )

    assert result.status == "executed"
    assert captured["side"] == "sell"
    assert captured["qty"] == 0.25
    assert captured["notional"] is None


def test_runtime_conflicting_position_closes_only_and_never_same_cycle_reverses(monkeypatch, broker_truth) -> None:
    monkeypatch.setattr(alpaca_runtime, "list_positions", lambda: [_position(side="long")])
    monkeypatch.setattr(alpaca_runtime, "get_open_position", lambda symbol: _position(side="long"))
    monkeypatch.setattr(
        alpaca_runtime,
        "evaluate_risk",
        lambda **kwargs: RiskDecision(
            accepted=True,
            reasons=["Existing position conflicts with the new signal and will be flipped."],
            details={"position_id": -1},
            intent_type="close",
        ),
    )
    calls = {"close": 0}

    def _close(**kwargs):
        calls["close"] += 1
        return {
            "action_taken": True,
            "intent_id": 46,
            "status": "accepted",
            "summary": "close acknowledged",
        }

    monkeypatch.setattr(alpaca_runtime, "close_paper_position", _close)
    monkeypatch.setattr(
        alpaca_runtime,
        "submit_paper_market_order",
        lambda **kwargs: pytest.fail("opposing signal must flatten only, never reverse in the same cycle"),
    )

    result = _run(
        _watch(alpaca_size_mode="qty", alpaca_size_value=1.0),
        _analysis(
            signal="short",
            stop_loss=101.0,
            take_profit=98.0,
        ),
    )

    assert result.status == "accepted"
    assert calls["close"] == 1


def test_runtime_same_direction_position_holds_without_add_on(monkeypatch, broker_truth) -> None:
    monkeypatch.setattr(alpaca_runtime, "list_positions", lambda: [_position(side="long")])
    monkeypatch.setattr(alpaca_runtime, "get_open_position", lambda symbol: _position(side="long"))
    monkeypatch.setattr(
        alpaca_runtime,
        "evaluate_risk",
        lambda **kwargs: RiskDecision(
            accepted=True,
            reasons=["Existing same-direction paper position will be updated."],
            details={"position_id": -1},
            intent_type="update",
        ),
    )
    monkeypatch.setattr(
        alpaca_runtime,
        "submit_paper_market_order",
        lambda **kwargs: pytest.fail("same-direction signal must not add to the Alpaca position"),
    )
    monkeypatch.setattr(
        alpaca_runtime,
        "close_paper_position",
        lambda **kwargs: pytest.fail("same-direction signal must not close"),
    )

    result = _run(_watch(), _analysis())

    assert result.action_taken is False
    assert result.status == "held"
    assert "already matches" in result.summary


def test_runtime_crypto_short_and_notional_equity_short_fail_closed(monkeypatch, broker_truth) -> None:
    monkeypatch.setattr(
        alpaca_runtime,
        "evaluate_risk",
        lambda **kwargs: RiskDecision(
            accepted=True,
            reasons=["Signal passed risk checks."],
            details={},
            intent_type="open",
        ),
    )
    monkeypatch.setattr(
        alpaca_runtime,
        "submit_paper_market_order",
        lambda **kwargs: pytest.fail("blocked sizing/direction must never reach the broker adapter"),
    )

    crypto = _run(
        _watch(
            symbol="BTC/USD",
            alpaca_asset_class="crypto",
            alpaca_size_mode="notional",
            alpaca_size_value=20.0,
        ),
        _analysis(
            symbol="BTC/USD",
            signal="short",
            stop_loss=101.0,
            take_profit=98.0,
        ),
    )
    equity = _run(
        _watch(alpaca_size_mode="notional", alpaca_size_value=25.0),
        _analysis(
            signal="short",
            stop_loss=101.0,
            take_profit=98.0,
        ),
    )

    assert crypto.status == "rejected"
    assert "does not open short" in crypto.summary
    assert equity.status == "rejected"
    assert "requires quantity sizing" in equity.summary


def test_runtime_risk_rejection_never_calls_mutation_adapter(monkeypatch, broker_truth) -> None:
    monkeypatch.setattr(
        alpaca_runtime,
        "evaluate_risk",
        lambda **kwargs: RiskDecision(
            accepted=False,
            reasons=["Daily loss cap reached."],
            details={"daily_loss_limit_breached": True},
            intent_type="skip",
        ),
    )
    monkeypatch.setattr(
        alpaca_runtime,
        "submit_paper_market_order",
        lambda **kwargs: pytest.fail("risk rejection must not submit"),
    )
    monkeypatch.setattr(
        alpaca_runtime,
        "close_paper_position",
        lambda **kwargs: pytest.fail("risk rejection must not close"),
    )

    result = _run(_watch(), _analysis())

    assert result.status == "rejected"
    assert result.action_taken is False
    assert result.intent_id is None
    assert "Daily loss cap" in result.summary


def test_runtime_broker_truth_failure_is_retryable_without_order_intent(monkeypatch) -> None:
    monkeypatch.setattr(
        alpaca_runtime,
        "get_account",
        lambda: (_ for _ in ()).throw(
            alpaca_runtime.AlpacaPaperError("account timeout")
        ),
    )
    monkeypatch.setattr(
        alpaca_runtime,
        "submit_paper_market_order",
        lambda **kwargs: pytest.fail("preflight failure must not submit"),
    )

    result = _run(_watch(), _analysis())

    assert result.status == "blocked"
    assert result.retryable is True
    assert result.intent_id is None
    audits = list_trade_audits(10)
    assert audits[0].event_type == "alpaca_runtime_signal_blocked"
    assert audits[0].details["preflight_failed"] is True
