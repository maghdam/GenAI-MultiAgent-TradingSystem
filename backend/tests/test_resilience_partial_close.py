from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from backend.domain.models import EngineConfig, WatchlistItem
from backend.services import broker_ledger, reconciler as reconciler_module
from backend.services.broker_ledger import reconcile_open_position_ledger
from backend.services.reconciler import reconcile_open_positions
from backend.storage.repositories import (
    list_broker_deals,
    list_incidents,
    list_paper_positions,
    open_paper_position,
    record_broker_deals,
    save_engine_config,
)


BROKER_POSITION_ID = 771001


def _deal(
    *,
    deal_id: int,
    closed_lots: float,
    net_profit: float,
    price: float = 101.0,
) -> dict:
    return {
        "deal_id": deal_id,
        "execution_price": price,
        "execution_at": datetime.now(UTC).replace(tzinfo=None),
        "closed_volume_api": closed_lots * 10_000,
        "closed_volume_lots": closed_lots,
        "gross_profit": net_profit,
        "swap": 0.0,
        "commission": 0.0,
        "pnl_conversion_fee": 0.0,
        "net_profit": net_profit,
    }


def _open_position(quantity: float = 0.30):
    return open_paper_position(
        symbol="NAS100",
        timeframe="M5",
        strategy="breakout",
        direction="long",
        quantity=quantity,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        account_currency="CHF",
        cash_per_price_unit_per_lot=1.0,
        instrument_spec_source="ctrader_contract",
        broker_position_id=BROKER_POSITION_ID,
    )


def _broker_row(quantity: float) -> dict:
    return {
        "position_id": BROKER_POSITION_ID,
        "symbol": "NAS100",
        "direction": "buy",
        "volume_lots": quantity,
        "entry_price": 100.0,
        "stop_loss": 99.0,
        "take_profit": 102.0,
    }


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
                lot_size=0.30,
                params={},
            )
        ],
    )


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        [{"open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5}],
        index=pd.to_datetime(["2026-10-02T12:00:00Z"], utc=True),
    )


def test_partial_close_preserves_canonical_identity_and_books_once(monkeypatch) -> None:
    position = _open_position()
    calls: list[int] = []
    deal = _deal(deal_id=94001, closed_lots=0.10, net_profit=7.25)
    monkeypatch.setattr(
        broker_ledger,
        "get_position_close_deals",
        lambda broker_position_id, **kwargs: calls.append(broker_position_id) or [deal],
    )

    first = reconcile_open_position_ledger(position, _broker_row(0.20))
    current = list_paper_positions("open")[0]

    assert first["status"] == "partial_close_synced"
    assert current.status == "open"
    assert current.quantity == pytest.approx(0.20)
    assert current.broker_position_id == BROKER_POSITION_ID
    assert current.realized_pnl == pytest.approx(7.25)
    assert [row["deal_id"] for row in list_broker_deals(local_position_id=position.id)] == [94001]

    second = reconcile_open_position_ledger(current, _broker_row(0.20))
    assert second["status"] == "in_sync"
    assert calls == [BROKER_POSITION_ID]
    assert len(list_broker_deals(local_position_id=position.id)) == 1


def test_partial_close_refuses_broker_identity_change_without_mutation(monkeypatch) -> None:
    position = _open_position()
    monkeypatch.setattr(
        broker_ledger,
        "get_position_close_deals",
        lambda *args, **kwargs: pytest.fail("identity mismatch must fail before deal-history lookup"),
    )

    result = reconcile_open_position_ledger(
        position,
        {
            **_broker_row(0.20),
            "position_id": BROKER_POSITION_ID + 1,
        },
    )

    assert result["status"] == "identity_mismatch"
    current = list_paper_positions("open")[0]
    assert current.quantity == pytest.approx(0.30)
    assert current.broker_position_id == BROKER_POSITION_ID
    assert list_broker_deals(local_position_id=position.id) == []


def test_second_partial_close_recovers_when_deal_was_persisted_before_quantity_update(
    monkeypatch,
) -> None:
    position = _open_position()
    first_deal = _deal(deal_id=94011, closed_lots=0.10, net_profit=5.0, price=101.0)
    monkeypatch.setattr(
        broker_ledger,
        "get_position_close_deals",
        lambda broker_position_id, **kwargs: [first_deal],
    )
    first = reconcile_open_position_ledger(position, _broker_row(0.20))
    assert first["status"] == "partial_close_synced"

    current = list_paper_positions("open")[0]
    second_deal = _deal(deal_id=94012, closed_lots=0.10, net_profit=6.0, price=101.5)

    # Simulate a process interruption after the second broker deal reached the
    # immutable ledger but before local remaining quantity was updated.
    recorded = record_broker_deals(
        local_position_id=current.id,
        broker_position_id=BROKER_POSITION_ID,
        symbol=current.symbol,
        account_currency=current.account_currency,
        deals=[second_deal],
    )
    assert recorded["inserted"] == 1

    monkeypatch.setattr(
        broker_ledger,
        "get_position_close_deals",
        lambda broker_position_id, **kwargs: [first_deal, second_deal],
    )
    recovered = reconcile_open_position_ledger(current, _broker_row(0.10))

    assert recovered["status"] == "partial_close_synced"
    assert recovered["inserted_deals"] == 0
    assert recovered["required_closed_lots"] == pytest.approx(0.20)
    updated = list_paper_positions("open")[0]
    assert updated.quantity == pytest.approx(0.10)
    assert updated.broker_position_id == BROKER_POSITION_ID
    assert updated.realized_pnl == pytest.approx(11.0)
    assert [row["deal_id"] for row in list_broker_deals(local_position_id=position.id)] == [
        94011,
        94012,
    ]


def test_missing_second_deal_stays_pending_instead_of_reusing_old_volume(monkeypatch) -> None:
    position = _open_position()
    first_deal = _deal(deal_id=94021, closed_lots=0.10, net_profit=4.0)
    monkeypatch.setattr(
        broker_ledger,
        "get_position_close_deals",
        lambda broker_position_id, **kwargs: [first_deal],
    )
    first = reconcile_open_position_ledger(position, _broker_row(0.20))
    assert first["status"] == "partial_close_synced"

    current = list_paper_positions("open")[0]
    pending = reconcile_open_position_ledger(current, _broker_row(0.10))

    assert pending["status"] == "pending_deal_history"
    assert pending["total_closed_lots"] == pytest.approx(0.10)
    assert pending["required_closed_lots"] == pytest.approx(0.20)
    assert "do not synthesize" in pending["action_required"].lower()
    unchanged = list_paper_positions("open")[0]
    assert unchanged.quantity == pytest.approx(0.20)
    assert unchanged.realized_pnl == pytest.approx(4.0)


def test_reconciler_waits_for_deal_history_then_converges_without_close_submission(
    monkeypatch,
) -> None:
    save_engine_config(_config())
    position = _open_position()
    broker_rows = {"rows": [_broker_row(0.20)]}
    deal_visible = {"value": False}
    history_calls: list[int] = []
    protection_calls: list[dict] = []

    monkeypatch.setattr(reconciler_module, "recover_broker_trackers", lambda cfg: {})
    monkeypatch.setattr(
        reconciler_module,
        "get_broker_status",
        lambda: type("S", (), {"execution_ready": True})(),
    )
    monkeypatch.setattr(reconciler_module, "list_positions", lambda: broker_rows["rows"])
    monkeypatch.setattr(reconciler_module, "get_bars", lambda *args, **kwargs: _bars())
    monkeypatch.setattr(
        reconciler_module,
        "sync_position_targets",
        lambda **kwargs: protection_calls.append(kwargs)
        or {
            "status": "already_synced",
            "verified": True,
            "position_id": kwargs["position_id"],
        },
    )
    monkeypatch.setattr(
        reconciler_module,
        "close_local_position_from_broker",
        lambda *args, **kwargs: pytest.fail("residual broker position must never be treated as fully closed"),
    )
    monkeypatch.setattr(
        reconciler_module,
        "attempt_verified_close",
        lambda *args, **kwargs: pytest.fail("partial-close reconciliation must not submit a broker close"),
    )
    partial_deal = _deal(deal_id=94031, closed_lots=0.10, net_profit=8.0)
    monkeypatch.setattr(
        broker_ledger,
        "get_position_close_deals",
        lambda broker_position_id, **kwargs: history_calls.append(broker_position_id)
        or ([partial_deal] if deal_visible["value"] else []),
    )

    first = reconcile_open_positions(reason="partial_close_pending")
    after_first = list_paper_positions("open")[0]
    assert first["closed"] == 0
    assert after_first.id == position.id
    assert after_first.quantity == pytest.approx(0.30)
    pending_incident = next(
        row
        for row in list_incidents(20)
        if row.code == "ctrader_partial_close_history_pending"
    )
    assert "do not synthesize" in pending_incident.details["action_required"].lower()

    deal_visible["value"] = True
    second = reconcile_open_positions(reason="partial_close_recovered")
    after_second = list_paper_positions("open")[0]
    assert second["closed"] == 0
    assert after_second.id == position.id
    assert after_second.status == "open"
    assert after_second.quantity == pytest.approx(0.20)
    assert after_second.broker_position_id == BROKER_POSITION_ID
    assert after_second.realized_pnl == pytest.approx(8.0)

    third = reconcile_open_positions(reason="partial_close_idempotent")
    after_third = list_paper_positions("open")[0]
    assert third["closed"] == 0
    assert after_third.quantity == pytest.approx(0.20)
    assert len(list_broker_deals(local_position_id=position.id)) == 1
    assert history_calls == [BROKER_POSITION_ID, BROKER_POSITION_ID]
    assert len(protection_calls) == 3
    assert all(call["position_id"] == BROKER_POSITION_ID for call in protection_calls)
