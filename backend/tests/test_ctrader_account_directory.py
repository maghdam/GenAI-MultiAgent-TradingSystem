from __future__ import annotations

import asyncio
from types import SimpleNamespace

from backend import ctrader_client as ctd
from backend.api import router as router_module
from backend.domain.models import CTraderAccount


class _Deferred:
    def addCallbacks(self, success, error):
        self.callbacks = (success, error)
        return self


def test_account_list_retains_demo_and_live_accounts(monkeypatch) -> None:
    sent = []
    monkeypatch.setattr(ctd, "ACCOUNT_ID", 1105460)
    monkeypatch.setattr(ctd, "HOST_TYPE", "demo")
    monkeypatch.setattr(
        ctd.Protobuf,
        "extract",
        lambda _: SimpleNamespace(
            ctidTraderAccount=[
                SimpleNamespace(ctidTraderAccountId=2123962, isLive=True),
                SimpleNamespace(ctidTraderAccountId=1105460, isLive=False),
                SimpleNamespace(ctidTraderAccountId=1105462, isLive=False),
            ]
        ),
    )
    monkeypatch.setattr(ctd.client, "send", lambda request: sent.append(request) or _Deferred())
    ctd.AVAILABLE_ACCOUNTS.clear()

    ctd.account_list_response_cb(object())

    assert ctd.get_available_accounts() == [
        {
            "account_id": 2123962,
            "account_type": "live",
            "is_live": True,
            "selected": False,
        },
        {
            "account_id": 1105460,
            "account_type": "demo",
            "is_live": False,
            "selected": True,
        },
        {
            "account_id": 1105462,
            "account_type": "demo",
            "is_live": False,
            "selected": False,
        },
    ]
    assert len(sent) == 1
    assert sent[0].ctidTraderAccountId == 1105460


def test_broker_accounts_endpoint_returns_normalized_directory(monkeypatch) -> None:
    rows = [
        CTraderAccount(
            account_id=2123962,
            account_type="live",
            is_live=True,
            selected=False,
        ),
        CTraderAccount(
            account_id=1105460,
            account_type="demo",
            is_live=False,
            selected=True,
        ),
    ]
    monkeypatch.setattr(router_module, "list_accounts", lambda: rows)

    result = asyncio.run(router_module.v2_broker_accounts())

    assert [row.account_id for row in result] == [2123962, 1105460]
    assert [row.account_type for row in result] == ["live", "demo"]
    assert [row.selected for row in result] == [False, True]
