from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from backend.api import router as router_module
from backend.domain.models import (
    BrokerStatus,
    CTraderAccount,
    CTraderAccountSelectionRequest,
    EngineConfig,
    LiveTradingArmRequest,
)
from backend.services.live_trading_guard import (
    arm_live_trading,
    disarm_live_trading,
    get_live_trading_armed_account_id,
    is_live_trading_armed,
    live_entry_block_reason,
)


@pytest.fixture(autouse=True)
def _reset_live_arm() -> None:
    disarm_live_trading()
    yield
    disarm_live_trading()


def _live_broker(account_id: int = 48922568) -> BrokerStatus:
    return BrokerStatus(
        connected=True,
        socket_connected=True,
        account_authorized=True,
        symbols_loaded=10,
        open_positions=0,
        pending_orders=0,
        ready=True,
        market_data_ready=True,
        broker_mode="live",
        account_id=account_id,
        account_type="live",
        active_host_type="live",
        account_verified=True,
        demo_account_confirmed=False,
        execution_ready=True,
    )


def test_live_arm_endpoint_arms_only_active_verified_live_account(monkeypatch) -> None:
    account_id = 48922568
    config = EngineConfig(
        enabled=False,
        ctrader_autotrade=True,
        selected_ctrader_account_id=account_id,
        selected_ctrader_account_type="live",
    )
    monkeypatch.setattr(router_module, "_current_config", lambda: config)
    monkeypatch.setattr(router_module, "get_broker_status", lambda: _live_broker(account_id))

    response = asyncio.run(
        router_module.v2_live_trading_arm(LiveTradingArmRequest(armed=True))
    )

    assert response.armed is True
    assert response.armed_account_id == account_id
    assert response.active_account_id == account_id
    assert response.account_type == "live"
    assert is_live_trading_armed(account_id) is True


def test_live_arm_endpoint_requires_engine_stopped(monkeypatch) -> None:
    account_id = 48922568
    config = EngineConfig(
        enabled=True,
        ctrader_autotrade=True,
        selected_ctrader_account_id=account_id,
        selected_ctrader_account_type="live",
    )
    monkeypatch.setattr(router_module, "_current_config", lambda: config)
    monkeypatch.setattr(router_module, "get_broker_status", lambda: _live_broker(account_id))

    with pytest.raises(HTTPException, match="Stop the engine") as exc:
        asyncio.run(
            router_module.v2_live_trading_arm(LiveTradingArmRequest(armed=True))
        )

    assert exc.value.status_code == 409
    assert get_live_trading_armed_account_id() is None


def test_engine_start_blocks_unarmed_live_and_allows_matching_arm(monkeypatch) -> None:
    account_id = 48922568
    config = EngineConfig(
        enabled=False,
        ctrader_autotrade=True,
        selected_ctrader_account_id=account_id,
        selected_ctrader_account_type="live",
    )
    saved: list[EngineConfig] = []
    monkeypatch.setattr(router_module, "_current_config", lambda: config)
    monkeypatch.setattr(router_module, "get_broker_status", lambda: _live_broker(account_id))
    monkeypatch.setattr(
        router_module,
        "save_engine_config",
        lambda value: saved.append(value.model_copy(deep=True)) or value,
    )
    monkeypatch.setattr(router_module.engine, "wake", lambda: None)

    with pytest.raises(HTTPException, match="Live Trading is disarmed") as exc:
        asyncio.run(router_module.v2_engine_start())
    assert exc.value.status_code == 409
    assert config.enabled is False

    arm_live_trading(account_id)
    result = asyncio.run(router_module.v2_engine_start())

    assert result == {"ok": True, "enabled": True}
    assert saved[-1].enabled is True


def test_account_switch_disarms_live_arm(monkeypatch) -> None:
    active_live_id = 48922568
    target_demo_id = 44089601
    arm_live_trading(active_live_id)

    rows = [
        CTraderAccount(
            account_id=active_live_id,
            account_type="live",
            is_live=True,
            trader_login=2123962,
            active=True,
        ),
        CTraderAccount(
            account_id=target_demo_id,
            account_type="demo",
            is_live=False,
            trader_login=4206993,
            active=False,
        ),
    ]
    config = EngineConfig(
        enabled=False,
        selected_ctrader_account_id=active_live_id,
        selected_ctrader_account_type="live",
    )
    monkeypatch.setattr(router_module, "list_accounts", lambda: rows)
    monkeypatch.setattr(router_module, "list_paper_positions", lambda status=None: [])
    monkeypatch.setattr(router_module, "_current_config", lambda: config)
    monkeypatch.setattr(router_module, "save_engine_config", lambda value: value)
    monkeypatch.setattr(
        router_module,
        "switch_account",
        lambda account_id, account_type: {"switch_started": True},
    )

    response = asyncio.run(
        router_module.v2_select_broker_account(
            CTraderAccountSelectionRequest(account_id=target_demo_id)
        )
    )

    assert response.switch_started is True
    assert get_live_trading_armed_account_id() is None


def test_engine_restart_disarms_live_trading(monkeypatch) -> None:
    arm_live_trading(48922568)

    async def _restart() -> None:
        return None

    monkeypatch.setattr(router_module.engine, "restart", _restart)
    monkeypatch.setattr(
        router_module,
        "_current_config",
        lambda: EngineConfig(enabled=False),
    )

    result = asyncio.run(router_module.v2_engine_restart())

    assert result == {"ok": True, "enabled": False, "restarted": True}
    assert get_live_trading_armed_account_id() is None


def test_live_entry_gate_is_account_bound_and_demo_needs_no_arm() -> None:
    live_id = 48922568

    assert live_entry_block_reason("demo", 44089601) is None
    assert "disarmed" in str(live_entry_block_reason("live", live_id)).lower()

    arm_live_trading(live_id)

    assert live_entry_block_reason("live", live_id) is None
    assert live_entry_block_reason("live", live_id + 1) is not None


def test_config_rejects_ctrader_autotrade_without_selected_account(monkeypatch) -> None:
    monkeypatch.setattr(
        router_module,
        "_current_config",
        lambda: EngineConfig(),
    )

    with pytest.raises(HTTPException, match="Select a cTrader account") as exc:
        asyncio.run(
            router_module.v2_set_config(
                EngineConfig(ctrader_autotrade=True)
            )
        )

    assert exc.value.status_code == 409

def test_config_save_cannot_bypass_live_arm_when_engine_enabled(monkeypatch) -> None:
    account_id = 48922568
    current = EngineConfig(
        enabled=False,
        ctrader_autotrade=True,
        selected_ctrader_account_id=account_id,
        selected_ctrader_account_type="live",
    )
    target = current.model_copy(update={"enabled": True})
    saved: list[EngineConfig] = []

    monkeypatch.setattr(router_module, "_current_config", lambda: current)
    monkeypatch.setattr(router_module, "get_broker_status", lambda: _live_broker(account_id))
    monkeypatch.setattr(
        router_module,
        "save_engine_config",
        lambda value: saved.append(value.model_copy(deep=True)) or value,
    )
    monkeypatch.setattr(router_module.engine, "wake", lambda: None)

    with pytest.raises(HTTPException, match="Live Trading is disarmed") as exc:
        asyncio.run(router_module.v2_set_config(target))

    assert exc.value.status_code == 409
    assert saved == []

    arm_live_trading(account_id)
    result = asyncio.run(router_module.v2_set_config(target))

    assert result.enabled is True
    assert saved[-1].enabled is True
