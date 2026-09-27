from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from backend.domain.models import EngineConfig, PaperPosition, StrategyAnalysis, WatchlistItem
from backend.services import risk_engine
from backend.services.financial_units import MonetaryBasis
from backend.services.risk_engine import evaluate_risk


def _config(**overrides) -> EngineConfig:
    payload = EngineConfig(
        enabled=True,
        paper_autotrade=True,
        demo_autotrade=False,
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
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        enabled=True,
        trading_enabled=False,
        lot_size=0.1,
    ).model_dump()
    payload.update(overrides)
    return WatchlistItem(**payload)


def _analysis(**overrides) -> StrategyAnalysis:
    payload = StrategyAnalysis(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        signal="long",
        confidence=0.82,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        reasons=["safety-control acceptance test"],
        context={},
    ).model_dump()
    payload.update(overrides)
    return StrategyAnalysis(**payload)


def _position(
    *,
    position_id: int,
    symbol: str,
    timeframe: str = "M5",
    status: str = "open",
    closed_at: datetime | None = None,
) -> PaperPosition:
    opened_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1)
    return PaperPosition(
        id=position_id,
        symbol=symbol,
        timeframe=timeframe,
        strategy="sma_cross",
        direction="long",
        quantity=0.1,
        status=status,
        entry_price=100.0,
        current_price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        opened_at=opened_at,
        closed_at=closed_at,
        exit_price=100.0 if status == "closed" else None,
    )


@pytest.fixture(autouse=True)
def safe_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(
        risk_engine,
        "paper_execution_gate",
        lambda strategy: (True, {"governed": False}, None),
    )
    monkeypatch.setattr(risk_engine, "list_paper_positions", lambda status=None: [])
    monkeypatch.setattr(risk_engine, "daily_realized_pnl", lambda: 0.0)
    monkeypatch.setattr(risk_engine, "daily_trade_count", lambda: 0)


def _evaluate(
    *,
    config: EngineConfig | None = None,
    watch: WatchlistItem | None = None,
    analysis: StrategyAnalysis | None = None,
):
    return evaluate_risk(
        config=config or _config(),
        watch_item=watch or _watch(),
        analysis=analysis or _analysis(),
        existing_position=None,
        mark_price=100.0,
        bar_timestamp=datetime.now(UTC).replace(tzinfo=None),
        bar_snapshot={"open": 99.8, "high": 100.3, "low": 99.5, "close": 100.0},
        monetary_basis=MonetaryBasis(
            currency="USD",
            equity_amount=100_000.0,
            source="test",
            verified=True,
        ),
    )


def test_kill_switch_blocks_execution() -> None:
    decision = _evaluate(config=_config(kill_switch=True))

    assert decision.accepted is False
    assert decision.reasons == ["Kill switch is active."]


def test_minimum_signal_strength_blocks_weak_signal() -> None:
    decision = _evaluate(
        config=_config(min_confidence=0.80),
        analysis=_analysis(confidence=0.70),
    )

    assert decision.accepted is False
    assert decision.reasons == ["Signal strength is below the configured minimum."]
    assert decision.details["min_confidence"] == 0.80


def test_cooldown_blocks_recently_closed_symbol(monkeypatch) -> None:
    closed = _position(
        position_id=1,
        symbol="XAUUSD",
        status="closed",
        closed_at=datetime.now(UTC).replace(tzinfo=None),
    )
    monkeypatch.setattr(
        risk_engine,
        "list_paper_positions",
        lambda status=None: [] if status == "open" else [closed],
    )

    decision = _evaluate(config=_config(cooldown_minutes=30))

    assert decision.accepted is False
    assert decision.reasons == ["Symbol is still inside cooldown from the last closed trade."]
    assert decision.details["cooldown_minutes"] == 30


def test_max_daily_trades_blocks_new_position(monkeypatch) -> None:
    monkeypatch.setattr(risk_engine, "daily_trade_count", lambda: 4)

    decision = _evaluate(config=_config(max_daily_trades=4))

    assert decision.accepted is False
    assert decision.reasons == ["Max daily trade count reached."]
    assert decision.details["daily_trade_count"] == 4


def test_max_open_positions_blocks_new_position(monkeypatch) -> None:
    open_positions = [
        _position(position_id=1, symbol="EURUSD"),
        _position(position_id=2, symbol="GBPUSD"),
    ]
    monkeypatch.setattr(
        risk_engine,
        "list_paper_positions",
        lambda status=None: open_positions if status == "open" else open_positions,
    )

    decision = _evaluate(config=_config(max_open_positions=2, max_positions_per_symbol=2))

    assert decision.accepted is False
    assert decision.reasons == ["Max open positions reached."]
    assert decision.details["open_positions"] == 2


def test_max_positions_per_symbol_blocks_second_symbol_position(monkeypatch) -> None:
    open_positions = [_position(position_id=1, symbol="XAUUSD", timeframe="H1")]
    monkeypatch.setattr(
        risk_engine,
        "list_paper_positions",
        lambda status=None: open_positions if status == "open" else open_positions,
    )

    decision = _evaluate(config=_config(max_open_positions=5, max_positions_per_symbol=1))

    assert decision.accepted is False
    assert decision.reasons == ["Max open positions per symbol reached."]
    assert decision.details["same_symbol_positions"] == 1


def test_daily_loss_limit_blocks_execution(monkeypatch) -> None:
    monkeypatch.setattr(risk_engine, "daily_realized_pnl", lambda: -2_000.0)

    decision = _evaluate(config=_config(daily_loss_limit_pct=2.0))

    assert decision.accepted is False
    assert decision.reasons == ["Daily loss cap reached."]
    assert decision.details["daily_loss_limit_breached"] is True


def test_require_protective_stops_blocks_missing_levels() -> None:
    decision = _evaluate(
        config=_config(require_stops=True),
        analysis=_analysis(stop_loss=None, take_profit=None),
    )

    assert decision.accepted is False
    assert decision.reasons == [
        "Stops are required and the strategy did not produce both stop and target."
    ]


def test_session_filter_blocks_signal_outside_window() -> None:
    hour = datetime.now(UTC).hour
    start = (hour + 1) % 24
    end = (hour + 2) % 24

    decision = _evaluate(
        config=_config(
            session_filter_enabled=True,
            session_start_hour_utc=start,
            session_end_hour_utc=end,
        )
    )

    assert decision.accepted is False
    assert decision.reasons == ["Signal is outside the configured trading session."]
    assert decision.details["session"]["enabled"] is True


def test_per_symbol_trading_permission_blocks_demo_auto_execution() -> None:
    decision = _evaluate(
        config=_config(paper_autotrade=False, demo_autotrade=True),
        watch=_watch(trading_enabled=False),
    )

    assert decision.accepted is False
    assert decision.reasons == ["Automatic execution is disabled for this symbol."]


def test_per_symbol_trading_permission_allows_demo_path_when_enabled() -> None:
    decision = _evaluate(
        config=_config(paper_autotrade=False, demo_autotrade=True),
        watch=_watch(trading_enabled=True),
    )

    assert decision.accepted is True
    assert decision.intent_type == "open"
    assert decision.reasons == ["Signal passed risk checks."]
