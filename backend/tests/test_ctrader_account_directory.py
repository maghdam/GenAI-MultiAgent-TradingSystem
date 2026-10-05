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


def test_live_app_auth_defers_demo_directory_probe_until_account_auth(monkeypatch) -> None:
    sent = []
    probes = []
    active_client = SimpleNamespace(
        send=lambda request, **kwargs: sent.append((request, kwargs)) or _Deferred(),
    )
    monkeypatch.setattr(ctd, "client", active_client)
    monkeypatch.setattr(ctd, "CLIENT_HOST_TYPE", "live")
    monkeypatch.setattr(ctd, "ACCOUNT_ID", 47139918)
    monkeypatch.setattr(ctd, "ACCESS_TOKEN", "test-token")
    monkeypatch.setattr(ctd, "ACCOUNT_IS_DEMO", None)
    monkeypatch.setattr(ctd, "AUTHORIZED", False)
    monkeypatch.setattr(ctd, "AUTH_ERROR", None)
    monkeypatch.setattr(ctd, "ACTIVE_ACCOUNT_ID", None)
    monkeypatch.setattr(ctd, "ACTIVE_HOST_TYPE", None)
    monkeypatch.setattr(ctd, "ACCOUNT_SWITCH_IN_PROGRESS", True)
    monkeypatch.setattr(ctd, "ACCOUNT_SWITCH_TARGET_ID", 47139918)
    monkeypatch.setattr(ctd, "ACCOUNT_SWITCH_ERROR", None)
    monkeypatch.setattr(ctd, "ACCOUNT_VERIFICATION_ERROR", None)
    monkeypatch.setattr(ctd, "LAST_AUTH_ATTEMPT_AT", None)
    monkeypatch.setattr(ctd, "AVAILABLE_ACCOUNTS", [])
    monkeypatch.setattr(ctd, "_start_demo_directory_probe", lambda: probes.append(True) or True)

    deferred = ctd.app_auth_cb(object(), active_client)

    assert probes == []
    assert ctd.ACCOUNT_IS_DEMO is False
    assert len(sent) == 1
    request, kwargs = sent[0]
    assert request.ctidTraderAccountId == 47139918
    assert request.accessToken == ctd.ACCESS_TOKEN
    assert kwargs["responseTimeoutInSeconds"] == 15
    assert kwargs["clientMsgId"].startswith("tradeagent:live-account-auth:47139918:")
    assert deferred is not None

    success, _ = deferred.callbacks
    success(object())

    assert probes == [True]


def test_demo_directory_probe_populates_mixed_token_accounts(monkeypatch) -> None:
    probe = SimpleNamespace(_tradeagent_directory_stage="account_list")
    monkeypatch.setattr(ctd, "ACCOUNT_ID", 47139918)
    monkeypatch.setattr(ctd, "ACTIVE_ACCOUNT_ID", 47139918)
    monkeypatch.setattr(ctd, "AVAILABLE_ACCOUNTS", [])
    monkeypatch.setattr(ctd, "_DEMO_DIRECTORY_PROBE_CLIENT", probe)
    monkeypatch.setattr(ctd, "_stop_client_service", lambda _: None)

    class ProtoOAGetAccountListByAccessTokenRes:
        ctidTraderAccount = [
            SimpleNamespace(
                ctidTraderAccountId=47140414,
                isLive=False,
                traderLogin=1105460,
                brokerTitleShort="FP Trading",
            ),
            SimpleNamespace(
                ctidTraderAccountId=47139918,
                isLive=True,
                traderLogin=2123962,
                brokerTitleShort="FP Trading",
            ),
        ]

    monkeypatch.setattr(
        ctd.Protobuf,
        "extract",
        lambda _: ProtoOAGetAccountListByAccessTokenRes(),
    )

    ctd._demo_directory_probe_message_received(probe, object(), probe)

    rows = ctd.get_available_accounts()
    assert {row["trader_login"] for row in rows} == {1105460, 2123962}
    live = next(row for row in rows if row["trader_login"] == 2123962)
    assert live["account_type"] == "live"
    assert live["active"] is True
    assert ctd._DEMO_DIRECTORY_PROBE_CLIENT is None


def test_demo_directory_probe_only_starts_when_live_transport_is_active(monkeypatch) -> None:
    monkeypatch.setattr(ctd.reactor, "running", True, raising=False)
    monkeypatch.setattr(ctd, "CLIENT_HOST_TYPE", "demo")
    monkeypatch.setattr(
        ctd,
        "_new_client",
        lambda host_type: pytest.fail(f"unexpected directory probe for {host_type}"),
    )

    assert ctd._start_demo_directory_probe() is False


def test_account_list_retains_demo_and_live_accounts(monkeypatch) -> None:
    sent = []
    monkeypatch.setattr(ctd, "ACCOUNT_ID", 47140414)
    monkeypatch.setattr(ctd, "HOST_TYPE", "demo")
    monkeypatch.setattr(ctd, "ACTIVE_ACCOUNT_ID", None)
    monkeypatch.setattr(ctd, "ACTIVE_HOST_TYPE", None)
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
    monkeypatch.setattr(ctd.client, "send", lambda request, **kwargs: sent.append((request, kwargs)) or _Deferred())
    monkeypatch.setattr(ctd, "AVAILABLE_ACCOUNTS", [])

    deferred = ctd.account_list_response_cb(object())

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
            "active": False,
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
    assert sent[0][0].ctidTraderAccountId == 47140414
    assert sent[0][1]["responseTimeoutInSeconds"] == 15

    success, _ = deferred.callbacks
    success(object())

    active_rows = ctd.get_available_accounts()
    active = next(row for row in active_rows if row["account_id"] == 47140414)
    assert active["active"] is True
    assert ctd.ACTIVE_ACCOUNT_ID == 47140414
    assert ctd.ACTIVE_HOST_TYPE == "demo"


def test_broker_accounts_endpoint_does_not_invent_selection_from_active_account(monkeypatch) -> None:
    rows = [
        CTraderAccount(
            account_id=1105460,
            account_type="demo",
            is_live=False,
            selected=True,
            active=True,
        ),
        CTraderAccount(
            account_id=2123962,
            account_type="live",
            is_live=True,
            selected=False,
            active=False,
        ),
    ]
    monkeypatch.setattr(router_module, "list_accounts", lambda: rows)
    monkeypatch.setattr(router_module, "_current_config", lambda: EngineConfig())

    result = asyncio.run(router_module.v2_broker_accounts())

    assert [row.account_id for row in result] == [1105460, 2123962]
    assert [row.active for row in result] == [True, False]
    assert [row.selected for row in result] == [False, False]


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


def test_select_broker_account_persists_choice_and_starts_transport_switch(monkeypatch) -> None:
    saved = {}
    switches = []
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
    monkeypatch.setattr(router_module, "list_paper_positions", lambda status=None: [])
    monkeypatch.setattr(router_module, "_current_config", lambda: EngineConfig())
    monkeypatch.setattr(
        router_module,
        "save_engine_config",
        lambda config: saved.setdefault("config", config) or config,
    )
    monkeypatch.setattr(
        router_module,
        "switch_account",
        lambda account_id, account_type: switches.append((account_id, account_type))
        or {"switch_started": True},
    )

    response = asyncio.run(
        router_module.v2_select_broker_account(
            CTraderAccountSelectionRequest(account_id=2123962)
        )
    )

    assert saved["config"].selected_ctrader_account_id == 2123962
    assert saved["config"].selected_ctrader_account_type == "live"
    assert switches == [(2123962, "live")]
    assert response.selected_account.account_id == 2123962
    assert response.selected_account.selected is True
    assert response.selected_account.active is False
    assert response.active_account_id == 1105460
    assert response.transport_switch_required is True
    assert response.switch_started is True


def test_select_broker_account_requires_engine_safe_state(monkeypatch) -> None:
    rows = [
        CTraderAccount(
            account_id=2123962,
            account_type="live",
            is_live=True,
            active=False,
        ),
        CTraderAccount(
            account_id=1105460,
            account_type="demo",
            is_live=False,
            active=True,
        ),
    ]
    monkeypatch.setattr(router_module, "list_accounts", lambda: rows)
    monkeypatch.setattr(
        router_module,
        "_current_config",
        lambda: EngineConfig(enabled=True, kill_switch=False),
    )

    with pytest.raises(HTTPException, match="kill switch") as exc:
        asyncio.run(
            router_module.v2_select_broker_account(
                CTraderAccountSelectionRequest(account_id=2123962)
            )
        )

    assert exc.value.status_code == 409


def test_select_broker_account_blocks_managed_open_broker_position(monkeypatch) -> None:
    rows = [
        CTraderAccount(
            account_id=2123962,
            account_type="live",
            is_live=True,
            active=False,
        ),
        CTraderAccount(
            account_id=1105460,
            account_type="demo",
            is_live=False,
            active=True,
        ),
    ]
    monkeypatch.setattr(router_module, "list_accounts", lambda: rows)
    monkeypatch.setattr(router_module, "_current_config", lambda: EngineConfig())
    monkeypatch.setattr(
        router_module,
        "list_paper_positions",
        lambda status=None: [SimpleNamespace(broker_position_id=555)],
    )

    with pytest.raises(HTTPException, match="open broker-backed position") as exc:
        asyncio.run(
            router_module.v2_select_broker_account(
                CTraderAccountSelectionRequest(account_id=2123962)
            )
        )

    assert exc.value.status_code == 409


def test_select_broker_account_rejects_account_not_in_authorized_directory(monkeypatch) -> None:
    monkeypatch.setattr(router_module, "list_accounts", lambda: [])
    monkeypatch.setattr(router_module, "list_paper_positions", lambda status=None: [])

    with pytest.raises(HTTPException, match="not available") as exc:
        asyncio.run(
            router_module.v2_select_broker_account(
                CTraderAccountSelectionRequest(account_id=999999)
            )
        )

    assert exc.value.status_code == 404

def test_demo_directory_probe_ignores_uncorrelated_error(monkeypatch) -> None:
    sent = []
    stopped = []
    probe = SimpleNamespace(
        _tradeagent_directory_stage="app_auth",
        _tradeagent_directory_client_msg_id="tradeagent:demo-directory-app-auth:none:1",
        send=lambda request, **kwargs: sent.append((request, kwargs)) or _Deferred(),
    )
    monkeypatch.setattr(ctd, "_DEMO_DIRECTORY_PROBE_CLIENT", probe)
    monkeypatch.setattr(ctd, "_stop_client_service", lambda target: stopped.append(target))

    class ProtoOAErrorRes:
        errorCode = "INVALID_REQUEST"
        description = "Trading account is not authorized"

    monkeypatch.setattr(ctd.Protobuf, "extract", lambda _: ProtoOAErrorRes())

    unrelated = SimpleNamespace(clientMsgId="2113629339616")
    ctd._demo_directory_probe_message_received(probe, unrelated, probe)

    assert ctd._DEMO_DIRECTORY_PROBE_CLIENT is probe
    assert stopped == []
    assert sent == []


def test_demo_directory_probe_matching_error_fails_probe(monkeypatch) -> None:
    stopped = []
    request_id = "tradeagent:demo-directory-app-auth:none:1"
    probe = SimpleNamespace(
        _tradeagent_directory_stage="app_auth",
        _tradeagent_directory_client_msg_id=request_id,
    )
    monkeypatch.setattr(ctd, "_DEMO_DIRECTORY_PROBE_CLIENT", probe)
    monkeypatch.setattr(ctd, "_stop_client_service", lambda target: stopped.append(target))

    class ProtoOAErrorRes:
        errorCode = "INVALID_REQUEST"
        description = "Bad request"

    monkeypatch.setattr(ctd.Protobuf, "extract", lambda _: ProtoOAErrorRes())

    matching = SimpleNamespace(clientMsgId=request_id)
    ctd._demo_directory_probe_message_received(probe, matching, probe)

    assert ctd._DEMO_DIRECTORY_PROBE_CLIENT is None
    assert stopped == [probe]

def test_main_client_routes_matching_probe_error_to_demo_directory_probe(monkeypatch) -> None:
    sent = []
    stopped = []
    request_id = "tradeagent:demo-directory-app-auth:none:1"
    probe = SimpleNamespace(
        _tradeagent_directory_stage="app_auth",
        _tradeagent_directory_client_msg_id=request_id,
        send=lambda request, **kwargs: sent.append((request, kwargs)) or _Deferred(),
    )
    monkeypatch.setattr(ctd, "_DEMO_DIRECTORY_PROBE_CLIENT", probe)
    monkeypatch.setattr(ctd, "_stop_client_service", lambda target: stopped.append(target))

    class ProtoOAErrorRes:
        errorCode = "ALREADY_LOGGED_IN"
        description = "Open API application is already authorized"

    monkeypatch.setattr(ctd.Protobuf, "extract", lambda _: ProtoOAErrorRes())

    matching = SimpleNamespace(clientMsgId=request_id)

    assert ctd._route_main_message_to_demo_directory_probe(matching) is True
    assert ctd._DEMO_DIRECTORY_PROBE_CLIENT is probe
    assert stopped == []
    assert len(sent) == 1
    request, kwargs = sent[0]
    assert request.__class__.__name__ == "ProtoOAGetAccountListByAccessTokenReq"
    assert kwargs["clientMsgId"].startswith(
        "tradeagent:demo-directory-account-list:none:"
    )
    assert probe._tradeagent_directory_stage == "account_list"


def test_main_client_does_not_route_unrelated_message_to_demo_directory_probe(monkeypatch) -> None:
    probe = SimpleNamespace(
        _tradeagent_directory_stage="app_auth",
        _tradeagent_directory_client_msg_id="tradeagent:demo-directory-app-auth:none:1",
    )
    monkeypatch.setattr(ctd, "_DEMO_DIRECTORY_PROBE_CLIENT", probe)

    unrelated = SimpleNamespace(clientMsgId="different-request")

    assert ctd._route_main_message_to_demo_directory_probe(unrelated) is False

