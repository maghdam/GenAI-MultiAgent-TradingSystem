from __future__ import annotations

import asyncio
from types import SimpleNamespace

from backend import ctrader_client as ctd
from fastapi import HTTPException
import pytest

from backend.api import router as router_module
from backend.domain.models import (
    CTraderAccount,
    CTraderAccountSelectionRequest,
    EngineConfig,
)


class _Deferred:
    def addCallbacks(self, success, error):
        self.callbacks = (success, error)
        return self


def test_account_list_retains_demo_and_live_accounts(monkeypatch) -> None:
    sent = []
    monkeypatch.setattr(ctd, "ACCOUNT_ID", 47140414)
    monkeypatch.setattr(ctd, "HOST_TYPE", "demo")
    monkeypatch.setattr(
        ctd.Protobuf,
        "extract",
        lambda _: SimpleNamespace(
            ctidTraderAccount=[
                SimpleNamespace(
                    ctidTraderAccountId=47139918,
                    isLive=True,
                    traderLogin=2123962,
                    brokerTitleShort="FP Trading",
                ),
                SimpleNamespace(
                    ctidTraderAccountId=47140414,
                    isLive=False,
                    traderLogin=1105460,
                    brokerTitleShort="FP Trading",
                ),
                SimpleNamespace(
                    ctidTraderAccountId=47140449,
                    isLive=False,
                    traderLogin=1105462,
                    brokerTitleShort="FP Trading",
                ),
            ]
        ),
    )
    monkeypatch.setattr(ctd.client, "send", lambda request: sent.append(request) or _Deferred())
    ctd.AVAILABLE_ACCOUNTS.clear()

    ctd.account_list_response_cb(object())

    assert ctd.get_available_accounts() == [
        {
            "account_id": 47139918,
            "account_type": "live",
            "is_live": True,
            "trader_login": 2123962,
            "broker_title": "FP Trading",
            "selected": False,
            "active": False,
        },
        {
            "account_id": 47140414,
            "account_type": "demo",
            "is_live": False,
            "trader_login": 1105460,
            "broker_title": "FP Trading",
            "selected": True,
            "active": True,
        },
        {
            "account_id": 47140449,
            "account_type": "demo",
            "is_live": False,
            "trader_login": 1105462,
            "broker_title": "FP Trading",
            "selected": False,
            "active": False,
        },
    ]
    assert len(sent) == 1
    assert sent[0].ctidTraderAccountId == 47140414


def test_broker_accounts_endpoint_overlays_persisted_selection(monkeypatch) -> None:
    rows = [
        CTraderAccount(
            account_id=2123962,
            account_type="live",
            is_live=True,
            selected=False,
            active=False,
        ),
        CTraderAccount(
            account_id=1105460,
            account_type="demo",
            is_live=False,
            selected=True,
            active=True,
        ),
    ]
    monkeypatch.setattr(router_module, "list_accounts", lambda: rows)
    monkeypatch.setattr(
        router_module,
        "_current_config",
        lambda: EngineConfig(selected_ctrader_account_id=2123962),
    )

    result = asyncio.run(router_module.v2_broker_accounts())

    assert [row.account_id for row in result] == [2123962, 1105460]
    assert [row.account_type for row in result] == ["live", "demo"]
    assert [row.selected for row in result] == [True, False]
    assert [row.active for row in result] == [False, True]


def test_select_broker_account_persists_choice_without_switching_transport(monkeypatch) -> None:
    saved = {}
    rows = [
        CTraderAccount(
            account_id=2123962,
            account_type="live",
            is_live=True,
            selected=False,
            active=False,
        ),
        CTraderAccount(
            account_id=1105460,
            account_type="demo",
            is_live=False,
            selected=True,
            active=True,
        ),
    ]
    monkeypatch.setattr(router_module, "list_accounts", lambda: rows)
    monkeypatch.setattr(router_module, "_current_config", lambda: EngineConfig())
    monkeypatch.setattr(
        router_module,
        "save_engine_config",
        lambda config: saved.setdefault("config", config) or config,
    )

    response = asyncio.run(
        router_module.v2_select_broker_account(
            CTraderAccountSelectionRequest(account_id=2123962)
        )
    )

    assert saved["config"].selected_ctrader_account_id == 2123962
    assert response.selected_account.account_id == 2123962
    assert response.selected_account.selected is True
    assert response.selected_account.active is False
    assert response.active_account_id == 1105460
    assert response.transport_switch_required is True


def test_select_broker_account_rejects_account_not_in_authorized_directory(monkeypatch) -> None:
    monkeypatch.setattr(router_module, "list_accounts", lambda: [])

    with pytest.raises(HTTPException, match="not available") as exc:
        asyncio.run(
            router_module.v2_select_broker_account(
                CTraderAccountSelectionRequest(account_id=999999)
            )
        )

    assert exc.value.status_code == 404
