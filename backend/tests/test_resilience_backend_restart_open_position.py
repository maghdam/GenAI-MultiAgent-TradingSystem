from __future__ import annotations

import pandas as pd
import pytest

from backend.domain.models import BrokerAccountSnapshot, EngineConfig, WatchlistItem
from backend.services import reconciler as reconciler_module
from backend.services.reconciler import reconcile_open_positions, recover_demo_broker_trackers
from backend.storage.repositories import (
    create_order_intent,
    list_incidents,
    list_order_intents,
    list_paper_positions,
    open_paper_position,
    save_engine_config,
    update_order_intent_status,
)


BROKER_POSITION_ID = 56980461


def _config() -> EngineConfig:
    return EngineConfig(
        enabled=True,
        paper_autotrade=False,
        demo_autotrade=True,
        kill_switch=False,
        watchlist=[
            WatchlistItem(
                symbol="NAS100",
                timeframe="M5",
                strategy="breakout",
                enabled=True,
                trading_enabled=True,
                lot_size=0.10,
                params={},
            )
        ],
    )


def _snapshot() -> BrokerAccountSnapshot:
    return BrokerAccountSnapshot(
        account_id=123,
        currency="CHF",
        balance=20_000.0,
        unrealized_pnl=0.0,
        equity=20_000.0,
        verified=True,
    )


def _broker_row() -> dict:
    return {
        "symbol": "NAS100",
        "direction": "buy",
        "volume_lots": 0.10,
        "entry_price": 29486.2,
        "stop_loss": 29476.4,
        "take_profit": 29511.7,
        "position_id": BROKER_POSITION_ID,
    }


def _bars(price: float = 29490.0) -> pd.DataFrame:
    return pd.DataFrame(
        [{"open": price - 1.0, "high": price + 1.0, "low": price - 2.0, "close": price}],
        index=pd.to_datetime(["2026-09-30T17:20:00Z"], utc=True),
    )


def _create_tradeagent_open_intent(
    *,
    symbol: str = "NAS100",
    direction: str = "long",
    status: str = "executed",
    tracking_retained: bool = False,
    failsafe_closed: bool = False,
):
    intent = create_order_intent(
        symbol=symbol,
        timeframe="M5",
        strategy="breakout",
        direction=direction,
        intent_type="open",
        status="accepted",
        confidence=0.80,
        entry_price=29487.5,
        stop_loss=29476.4,
        take_profit=29511.7,
        quantity=0.10,
        rationale="restart resilience test",
        details={},
    )
    details = {
        "execution_mode": "ctrader_demo",
        "broker_order": {
            "position_id": BROKER_POSITION_ID,
            "symbol": symbol,
            "quantity_lots": 0.10,
        },
    }
    if status == "failed":
        details.update(
            {
                "tracking_retained": tracking_retained,
                "failsafe_closed": failsafe_closed,
            }
        )
    return update_order_intent_status(
        intent.id,
        status,
        details,
        reason="restart_resilience_fixture",
    )


def _patch_ready_broker(monkeypatch) -> list[dict]:
    sync_calls: list[dict] = []
    monkeypatch.setattr(
        reconciler_module,
        "get_broker_status",
        lambda: type("S", (), {"execution_ready": True})(),
    )
    monkeypatch.setattr(reconciler_module, "get_broker_account_snapshot", _snapshot)
    monkeypatch.setattr(reconciler_module, "list_positions", lambda: [_broker_row()])
    monkeypatch.setattr(
        reconciler_module,
        "get_instrument_spec",
        lambda symbol, currency: type(
            "Spec",
            (),
            {
                "cash_per_price_unit_per_lot": 1.0,
                "source": "test",
                "valuation_ready": True,
            },
        )(),
    )
    monkeypatch.setattr(reconciler_module, "get_bars", lambda *args, **kwargs: _bars())
    monkeypatch.setattr(
        reconciler_module,
        "reconcile_open_demo_position_ledger",
        lambda position, broker_row: {"status": "unchanged"},
    )
    monkeypatch.setattr(
        reconciler_module,
        "sync_demo_position_targets",
        lambda **kwargs: sync_calls.append(kwargs)
        or {
            "status": "already_synced",
            "verified": True,
            "position_id": kwargs["position_id"],
        },
    )
    monkeypatch.setattr(
        reconciler_module,
        "close_demo_position",
        lambda **kwargs: pytest.fail("healthy restart recovery must not close the broker position"),
    )
    return sync_calls


def test_startup_recovery_is_idempotent_and_keeps_canonical_broker_position(monkeypatch) -> None:
    save_engine_config(_config())
    opening_intent = _create_tradeagent_open_intent()
    sync_calls = _patch_ready_broker(monkeypatch)

    first = reconcile_open_positions(reason="startup")
    positions_after_first = list_paper_positions("open")

    assert first["checked"] == 1
    assert first["closed"] == 0
    assert len(positions_after_first) == 1
    assert positions_after_first[0].broker_position_id == BROKER_POSITION_ID
    assert positions_after_first[0].entry_price == _broker_row()["entry_price"]
    assert positions_after_first[0].quantity == _broker_row()["volume_lots"]
    assert len(sync_calls) == 1
    assert sync_calls[0]["position_id"] == BROKER_POSITION_ID

    second = reconcile_open_positions(reason="startup")
    positions_after_second = list_paper_positions("open")

    assert second["checked"] == 1
    assert second["closed"] == 0
    assert len(positions_after_second) == 1
    assert positions_after_second[0].id == positions_after_first[0].id
    assert positions_after_second[0].broker_position_id == BROKER_POSITION_ID
    assert len(sync_calls) == 2
    assert all(call["position_id"] == BROKER_POSITION_ID for call in sync_calls)

    intents = list_order_intents(20)
    assert len(intents) == 1
    assert intents[0].id == opening_intent.id
    assert intents[0].intent_type == "open"
    assert intents[0].status == "executed"


@pytest.mark.parametrize(
    ("intent_symbol", "intent_direction"),
    [
        ("XAUUSD", "long"),
        ("NAS100", "short"),
    ],
)
def test_startup_recovery_rejects_broker_id_with_wrong_tradeagent_identity(
    monkeypatch,
    intent_symbol: str,
    intent_direction: str,
) -> None:
    save_engine_config(_config())
    _create_tradeagent_open_intent(
        symbol=intent_symbol,
        direction=intent_direction,
    )

    monkeypatch.setattr(
        reconciler_module,
        "get_broker_status",
        lambda: type("S", (), {"execution_ready": True})(),
    )
    monkeypatch.setattr(reconciler_module, "get_broker_account_snapshot", _snapshot)
    monkeypatch.setattr(reconciler_module, "list_positions", lambda: [_broker_row()])
    monkeypatch.setattr(
        reconciler_module,
        "get_instrument_spec",
        lambda *args, **kwargs: pytest.fail("identity mismatch must not create a local tracker"),
    )

    result = recover_demo_broker_trackers(_config())

    assert result["recovered"] == 0
    assert result["attached"] == 0
    assert result["untracked"] == 1
    assert list_paper_positions("open") == []

    incidents = list_incidents(10)
    assert incidents[0].code == "ctrader_demo_untracked_broker_position"
    assert incidents[0].details["automatic_adoption"] is False
    assert (
        incidents[0].details["required_identity"]
        == "tradeagent_open_intent+broker_position_id+symbol+direction"
    )


def test_startup_recovery_keeps_failed_but_explicitly_retained_tradeagent_position(monkeypatch) -> None:
    save_engine_config(_config())
    retained = _create_tradeagent_open_intent(
        status="failed",
        tracking_retained=True,
        failsafe_closed=False,
    )

    monkeypatch.setattr(
        reconciler_module,
        "get_broker_status",
        lambda: type("S", (), {"execution_ready": True})(),
    )
    monkeypatch.setattr(reconciler_module, "get_broker_account_snapshot", _snapshot)
    monkeypatch.setattr(reconciler_module, "list_positions", lambda: [_broker_row()])
    monkeypatch.setattr(
        reconciler_module,
        "get_instrument_spec",
        lambda symbol, currency: type(
            "Spec",
            (),
            {
                "cash_per_price_unit_per_lot": 1.0,
                "source": "test",
                "valuation_ready": True,
            },
        )(),
    )

    result = recover_demo_broker_trackers(_config())

    assert result["recovered"] == 1
    positions = list_paper_positions("open")
    assert len(positions) == 1
    assert positions[0].broker_position_id == BROKER_POSITION_ID
    assert retained.status == "failed"
    assert retained.details["tracking_retained"] is True
    assert retained.details["failsafe_closed"] is False


def test_startup_recovery_rejects_failed_intent_when_tracking_was_not_retained(monkeypatch) -> None:
    save_engine_config(_config())
    _create_tradeagent_open_intent(
        status="failed",
        tracking_retained=False,
        failsafe_closed=True,
    )

    monkeypatch.setattr(
        reconciler_module,
        "get_broker_status",
        lambda: type("S", (), {"execution_ready": True})(),
    )
    monkeypatch.setattr(reconciler_module, "get_broker_account_snapshot", _snapshot)
    monkeypatch.setattr(reconciler_module, "list_positions", lambda: [_broker_row()])

    result = recover_demo_broker_trackers(_config())

    assert result["recovered"] == 0
    assert result["untracked"] == 1
    assert list_paper_positions("open") == []


def test_startup_before_broker_readiness_preserves_existing_demo_tracker(monkeypatch) -> None:
    save_engine_config(_config())
    opened = open_paper_position(
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        direction="long",
        quantity=0.10,
        entry_price=29486.2,
        stop_loss=29476.4,
        take_profit=29511.7,
        broker_position_id=BROKER_POSITION_ID,
    )

    monkeypatch.setattr(
        reconciler_module,
        "get_broker_status",
        lambda: type("S", (), {"execution_ready": False})(),
    )
    monkeypatch.setattr(reconciler_module, "get_bars", lambda *args, **kwargs: _bars(29550.0))
    monkeypatch.setattr(
        reconciler_module,
        "list_positions",
        lambda: pytest.fail("broker positions must not be queried before broker readiness"),
    )
    monkeypatch.setattr(
        reconciler_module,
        "sync_demo_position_targets",
        lambda **kwargs: pytest.fail("startup must not mutate protection before broker readiness"),
    )
    monkeypatch.setattr(
        reconciler_module,
        "close_demo_position",
        lambda **kwargs: pytest.fail("startup must not close broker position before broker readiness"),
    )
    monkeypatch.setattr(
        reconciler_module,
        "close_local_position_from_broker",
        lambda *args, **kwargs: pytest.fail("startup must not synthesize broker truth while unavailable"),
    )

    summary = reconcile_open_positions(reason="startup")

    assert summary == {
        "checked": 1,
        "closed": 0,
        "skipped": 1,
        "reason": "startup",
    }
    positions = list_paper_positions("open")
    assert len(positions) == 1
    assert positions[0].id == opened.id
    assert positions[0].broker_position_id == BROKER_POSITION_ID
