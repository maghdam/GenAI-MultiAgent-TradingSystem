from __future__ import annotations

from datetime import UTC, datetime

import pytest

from backend.domain.models import BrokerAccountSnapshot, EngineConfig, InstrumentSpec, StrategyAnalysis, WatchlistItem
from backend.services import execution_engine
from backend.services.execution_engine import _ctrader_new_entry_inventory_gate, execute_paper_signal
from backend.services.financial_units import MonetaryBasis
from backend.services.quantity_rules import QuantityDecision
from backend.services.risk_engine import RiskDecision
from backend.storage.repositories import (
    list_incidents,
    list_order_intents,
    list_paper_positions,
    open_paper_position,
)


UNTRACKED_ID = 57868693
TRACKED_ID = 57871070


def _config() -> EngineConfig:
    return EngineConfig(
        enabled=True,
        paper_autotrade=False,
        ctrader_autotrade=True,
        kill_switch=False,
        require_stops=True,
        min_confidence=0.60,
        selected_ctrader_account_id=44089601,
        selected_ctrader_account_type="demo",
    )


def _watch_item() -> WatchlistItem:
    return WatchlistItem(
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        enabled=True,
        trading_enabled=True,
        lot_size=0.10,
        params={},
    )


def _analysis(signal: str = "long") -> StrategyAnalysis:
    return StrategyAnalysis(
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        signal=signal,
        confidence=0.82 if signal != "no_trade" else 0.0,
        entry_price=31_300.0,
        stop_loss=31_250.0 if signal != "no_trade" else None,
        take_profit=31_400.0 if signal != "no_trade" else None,
        reasons=["untracked broker exposure resilience test"],
        context={},
    )


def _broker_row(position_id: int, *, symbol: str = "XAUUSD", direction: str = "sell") -> dict[str, object]:
    return {
        "position_id": position_id,
        "symbol": symbol,
        "direction": direction,
        "volume_lots": 0.10,
        "entry_price": 4_134.34,
        "stop_loss": 4_135.84,
        "take_profit": 4_126.04,
    }


def _patch_new_entry_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(
        execution_engine,
        "get_symbol_execution_readiness",
        lambda symbol: (True, "Broker symbol contract metadata is ready."),
    )
    monkeypatch.setattr(
        execution_engine,
        "get_broker_account_snapshot",
        lambda: BrokerAccountSnapshot(
            account_id=44089601,
            currency="CHF",
            balance=20_000.0,
            unrealized_pnl=0.0,
            equity=20_000.0,
            verified=True,
        ),
    )
    monkeypatch.setattr(
        execution_engine,
        "resolve_monetary_basis",
        lambda *args, **kwargs: MonetaryBasis(
            currency="CHF",
            equity_amount=20_000.0,
            source="ctrader_account_snapshot",
            verified=True,
        ),
    )
    monkeypatch.setattr(
        execution_engine,
        "get_instrument_spec",
        lambda symbol, currency: InstrumentSpec(
            symbol=symbol,
            source="test",
            account_currency=currency,
            cash_per_price_unit_per_lot=1.0,
            valuation_ready=True,
            verified=True,
        ),
    )
    monkeypatch.setattr(
        execution_engine,
        "evaluate_order_quantity",
        lambda *args, **kwargs: QuantityDecision(
            accepted=True,
            requested_quantity=0.10,
            final_quantity=0.10,
            reasons=[],
            details={},
        ),
    )
    monkeypatch.setattr(
        execution_engine,
        "evaluate_risk",
        lambda **kwargs: RiskDecision(
            accepted=True,
            reasons=["Risk controls accepted the new position."],
            details={},
            intent_type="open",
        ),
    )
    monkeypatch.setattr(
        execution_engine,
        "get_broker_status",
        lambda: type(
            "Broker",
            (),
            {
                "execution_ready": True,
                "account_type": "demo",
                "account_id": 44089601,
            },
        )(),
    )
    monkeypatch.setattr(execution_engine, "live_entry_block_reason", lambda *args, **kwargs: None)


def test_inventory_gate_detects_untracked_broker_position(monkeypatch) -> None:
    monkeypatch.setattr(execution_engine, "list_positions", lambda: [_broker_row(UNTRACKED_ID)])

    accepted, details, reason = _ctrader_new_entry_inventory_gate()

    assert accepted is False
    assert details["broker_inventory_available"] is True
    assert details["broker_open_position_ids"] == [UNTRACKED_ID]
    assert details["local_tracked_broker_position_ids"] == []
    assert details["untracked_broker_position_ids"] == [UNTRACKED_ID]
    assert details["broker_submission_suppressed"] is True
    assert reason is not None
    assert "Untracked cTrader broker exposure" in reason


def test_inventory_gate_accepts_when_every_broker_position_has_canonical_tracker(monkeypatch) -> None:
    open_paper_position(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        direction="short",
        quantity=0.01,
        entry_price=4_132.57,
        stop_loss=4_135.84,
        take_profit=4_126.04,
        broker_position_id=TRACKED_ID,
    )
    monkeypatch.setattr(execution_engine, "list_positions", lambda: [_broker_row(TRACKED_ID)])

    accepted, details, reason = _ctrader_new_entry_inventory_gate()

    assert accepted is True
    assert reason is None
    assert details["untracked_broker_position_ids"] == []
    assert details["broker_submission_suppressed"] is False


def test_inventory_gate_fails_closed_when_broker_inventory_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        execution_engine,
        "list_positions",
        lambda: (_ for _ in ()).throw(RuntimeError("broker snapshot unavailable")),
    )

    accepted, details, reason = _ctrader_new_entry_inventory_gate()

    assert accepted is False
    assert details["broker_inventory_available"] is False
    assert details["broker_submission_suppressed"] is True
    assert reason is not None
    assert "inventory is unavailable" in reason


def test_untracked_broker_exposure_blocks_new_ctrader_entry_before_submission(monkeypatch) -> None:
    _patch_new_entry_dependencies(monkeypatch)
    monkeypatch.setattr(execution_engine, "list_positions", lambda: [_broker_row(UNTRACKED_ID)])
    monkeypatch.setattr(
        execution_engine,
        "place_market_order",
        lambda **kwargs: pytest.fail("untracked broker exposure must block broker submission"),
    )

    result = execute_paper_signal(
        config=_config(),
        watch_item=_watch_item(),
        analysis=_analysis(),
        mark_price=31_300.0,
        bar_timestamp=datetime.now(UTC),
        bar_snapshot={"open": 31_290.0, "high": 31_320.0, "low": 31_280.0, "close": 31_300.0},
    )

    assert result.action_taken is False
    assert result.status == "rejected"
    assert result.retryable is False
    assert list_paper_positions("open") == []

    intents = list_order_intents(10)
    assert len(intents) == 1
    assert intents[0].status == "rejected"
    inventory = intents[0].details["broker_entry_inventory"]
    assert inventory["untracked_broker_position_ids"] == [UNTRACKED_ID]
    assert inventory["broker_submission_suppressed"] is True
    assert "Untracked cTrader broker exposure" in intents[0].rationale

    incidents = list_incidents(20)
    blocking = next(item for item in incidents if item.code == "ctrader_untracked_broker_exposure_blocks_entry")
    assert blocking.details["untracked_broker_position_ids"] == [UNTRACKED_ID]
    assert blocking.details["broker_submission_suppressed"] is True
    assert blocking.details["automatic_adoption"] is False


def test_existing_managed_position_maintenance_is_not_blocked_by_untracked_exposure(monkeypatch) -> None:
    managed = open_paper_position(
        symbol="XAUUSD",
        timeframe="M5",
        strategy="sma_cross",
        direction="short",
        quantity=0.01,
        entry_price=4_132.57,
        stop_loss=4_135.84,
        take_profit=4_126.04,
        broker_position_id=TRACKED_ID,
    )
    rows = [
        _broker_row(TRACKED_ID),
        _broker_row(UNTRACKED_ID),
    ]
    sync_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        execution_engine,
        "get_broker_status",
        lambda: type(
            "Broker",
            (),
            {
                "execution_ready": True,
                "account_type": "demo",
                "account_id": 44089601,
            },
        )(),
    )
    monkeypatch.setattr(execution_engine, "list_positions", lambda: rows)
    monkeypatch.setattr(
        execution_engine,
        "reconcile_open_position_ledger",
        lambda position, broker_row: {"status": "unchanged"},
    )
    monkeypatch.setattr(
        execution_engine,
        "sync_position_targets",
        lambda **kwargs: sync_calls.append(kwargs)
        or {"status": "already_synced", "verified": True, "position_id": TRACKED_ID},
    )
    monkeypatch.setattr(
        execution_engine,
        "_ctrader_new_entry_inventory_gate",
        lambda: pytest.fail("new-entry inventory gate must not block maintenance of an existing tracker"),
    )
    monkeypatch.setattr(
        execution_engine,
        "resolve_monetary_basis",
        lambda *args, **kwargs: MonetaryBasis(
            currency="CHF",
            equity_amount=20_000.0,
            source="ctrader_account_snapshot",
            verified=True,
        ),
    )
    monkeypatch.setattr(
        execution_engine,
        "get_instrument_spec",
        lambda symbol, currency: InstrumentSpec(
            symbol=symbol,
            source="test",
            account_currency=currency,
            cash_per_price_unit_per_lot=1.0,
            valuation_ready=True,
            verified=True,
        ),
    )
    monkeypatch.setattr(
        execution_engine,
        "evaluate_order_quantity",
        lambda *args, **kwargs: QuantityDecision(
            accepted=True,
            requested_quantity=0.01,
            final_quantity=0.01,
            reasons=[],
            details={},
        ),
    )
    monkeypatch.setattr(
        execution_engine,
        "evaluate_risk",
        lambda **kwargs: RiskDecision(
            accepted=False,
            reasons=["Strategy returned no_trade."],
            details={},
            intent_type="skip",
        ),
    )

    result = execute_paper_signal(
        config=_config(),
        watch_item=WatchlistItem(
            symbol="XAUUSD",
            timeframe="M5",
            strategy="sma_cross",
            enabled=True,
            trading_enabled=True,
            lot_size=0.01,
            params={},
        ),
        analysis=StrategyAnalysis(
            symbol="XAUUSD",
            timeframe="M5",
            strategy="sma_cross",
            signal="no_trade",
            confidence=0.0,
            entry_price=4_133.0,
            reasons=["maintenance path"],
            context={},
        ),
        mark_price=4_133.0,
        bar_timestamp=datetime.now(UTC),
        bar_snapshot={"open": 4_134.0, "high": 4_135.0, "low": 4_132.0, "close": 4_133.0},
    )

    assert result.status == "rejected"
    assert list_paper_positions("open")[0].id == managed.id
    assert len(sync_calls) == 1
    assert sync_calls[0]["position_id"] == TRACKED_ID
