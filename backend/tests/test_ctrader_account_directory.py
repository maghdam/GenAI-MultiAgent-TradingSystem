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


def test_probe_application_auth_message_advances_to_account_list(monkeypatch) -> None:
    sent = []
    probe = SimpleNamespace(
        _tradeagent_directory_stage="app_auth",
        send=lambda request, **kwargs: sent.append((request, kwargs)) or _Deferred(),
    )
    monkeypatch.setattr(ctd, "_ACCOUNT_DIRECTORY_PROBE_CLIENTS", {"live": probe})

    class ProtoOAApplicationAuthRes:
        pass

    monkeypatch.setattr(ctd.Protobuf, "extract", lambda _: ProtoOAApplicationAuthRes())

    ctd._account_directory_probe_message_received(probe, object(), "live", probe)

    assert probe._tradeagent_directory_stage == "account_list"
    assert len(sent) == 1
    assert sent[0][1]["responseTimeoutInSeconds"] == 15


def test_probe_account_list_message_merges_and_finishes(monkeypatch) -> None:
    probe = SimpleNamespace(_tradeagent_directory_stage="account_list")
    monkeypatch.setattr(ctd, "ACCOUNT_ID", 47140414)
    monkeypatch.setattr(ctd, "ACTIVE_ACCOUNT_ID", 47140414)
    monkeypatch.setattr(ctd, "_ACCOUNT_DIRECTORY_BY_HOST", {"demo": [], "live": []})
    monkeypatch.setattr(ctd, "AVAILABLE_ACCOUNTS", [])
    monkeypatch.setattr(ctd, "_ACCOUNT_DIRECTORY_PROBE_CLIENTS", {"live": probe})
    monkeypatch.setattr(
        ctd,
        "_ACCOUNT_DIRECTORY_PROBE_ERRORS",
        {"demo": None, "live": None},
    )
    monkeypatch.setattr(ctd, "_stop_client_service", lambda _: None)
    ctd._replace_account_directory_host(
        "demo",
        [
            SimpleNamespace(
                ctidTraderAccountId=47140414,
                isLive=False,
                traderLogin=1105460,
                brokerTitleShort="FP Trading",
            )
        ],
    )

    class ProtoOAGetAccountListByAccessTokenRes:
        ctidTraderAccount = [
            SimpleNamespace(
                ctidTraderAccountId=47139918,
                isLive=True,
                traderLogin=2123962,
                brokerTitleShort="FP Trading",
            )
        ]

    monkeypatch.setattr(
        ctd.Protobuf,
        "extract",
        lambda _: ProtoOAGetAccountListByAccessTokenRes(),
    )

    ctd._account_directory_probe_message_received(probe, object(), "live", probe)

    rows = ctd.get_available_accounts()
    assert {row["trader_login"] for row in rows} == {1105460, 2123962}
    assert ctd._ACCOUNT_DIRECTORY_PROBE_CLIENTS == {}


def test_probe_already_logged_in_event_advances_to_account_list(monkeypatch) -> None:
    sent = []
    probe = SimpleNamespace(
        _tradeagent_directory_stage="app_auth",
        send=lambda request, **kwargs: sent.append((request, kwargs)) or _Deferred(),
    )
    monkeypatch.setattr(
        ctd,
        "_ACCOUNT_DIRECTORY_PROBE_CLIENTS",
        {"live": probe},
    )
    monkeypatch.setattr(
        ctd.Protobuf,
        "extract",
        lambda _: SimpleNamespace(
            __class__=SimpleNamespace(__name__="ProtoOAErrorRes"),
            errorCode=14,
            description="Open API application is already authorized",
        ),
    )

    class _ErrorPayload:
        errorCode = 14
        description = "Open API application is already authorized"

    monkeypatch.setattr(
        ctd.Protobuf,
        "extract",
        lambda _: _ErrorPayload(),
    )
    _ErrorPayload.__name__ = "ProtoOAErrorRes"

    ctd._account_directory_probe_message_received(
        probe,
        object(),
        "live",
        probe,
    )

    assert probe._tradeagent_directory_stage == "account_list"
    assert len(sent) == 1
    assert sent[0][1]["responseTimeoutInSeconds"] == 15


def test_probe_app_auth_timeout_falls_back_to_account_list(monkeypatch) -> None:
    sent = []
    probe = SimpleNamespace(
        _tradeagent_directory_stage="app_auth",
        send=lambda request, **kwargs: sent.append((request, kwargs)) or _Deferred(),
    )
    monkeypatch.setattr(
        ctd,
        "_ACCOUNT_DIRECTORY_PROBE_CLIENTS",
        {"live": probe},
    )

    from twisted.python.failure import Failure

    result = ctd._account_directory_probe_error_cb(
        Failure(ctd.DeferredTimeoutError(15, "Deferred")),
        "live",
        probe,
        stage="app_auth",
    )

    assert result is None
    assert probe._tradeagent_directory_stage == "account_list"
    assert len(sent) == 1
    assert sent[0][1]["responseTimeoutInSeconds"] == 15


def test_probe_string_already_logged_in_event_advances_to_account_list(monkeypatch) -> None:
    sent = []
    probe = SimpleNamespace(
        _tradeagent_directory_stage="app_auth",
        send=lambda request, **kwargs: sent.append((request, kwargs)) or _Deferred(),
    )
    monkeypatch.setattr(ctd, "_ACCOUNT_DIRECTORY_PROBE_CLIENTS", {"live": probe})

    class ProtoOAErrorRes:
        errorCode = "ALREADY_LOGGED_IN"
        description = "Open API application is already authorized"

    monkeypatch.setattr(ctd.Protobuf, "extract", lambda _: ProtoOAErrorRes())

    ctd._account_directory_probe_message_received(probe, object(), "live", probe)

    assert probe._tradeagent_directory_stage == "account_list"
    assert len(sent) == 1


def test_stale_probe_auth_timeout_is_consumed_after_account_list_started(monkeypatch) -> None:
    probe = SimpleNamespace(_tradeagent_directory_stage="account_list")
    monkeypatch.setattr(
        ctd,
        "_ACCOUNT_DIRECTORY_PROBE_CLIENTS",
        {"live": probe},
    )
    finished = []
    monkeypatch.setattr(
        ctd,
        "_finish_account_directory_probe",
        lambda *args, **kwargs: finished.append((args, kwargs)),
    )

    result = ctd._account_directory_probe_error_cb(
        RuntimeError("late auth timeout"),
        "live",
        probe,
        stage="app_auth",
    )

    assert result is None
    assert finished == []


def test_account_directory_probe_waits_for_running_reactor(monkeypatch) -> None:
    monkeypatch.setattr(ctd, "CLIENT_HOST_TYPE", "demo")
    monkeypatch.setattr(ctd.reactor, "running", False, raising=False)
    monkeypatch.setattr(
        ctd,
        "_new_client",
        lambda host_type: pytest.fail(f"unexpected probe client for {host_type}"),
    )

    assert ctd._start_account_directory_probe("live") is False


def test_account_directory_merges_demo_and_live_host_results(monkeypatch) -> None:
    monkeypatch.setattr(ctd, "ACCOUNT_ID", 47140414)
    monkeypatch.setattr(ctd, "ACTIVE_ACCOUNT_ID", 47140414)
    monkeypatch.setattr(ctd, "_ACCOUNT_DIRECTORY_BY_HOST", {"demo": [], "live": []})
    monkeypatch.setattr(ctd, "AVAILABLE_ACCOUNTS", [])

    ctd._replace_account_directory_host(
        "demo",
        [
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
        ],
    )
    ctd._replace_account_directory_host(
        "live",
        [
            SimpleNamespace(
                ctidTraderAccountId=47139918,
                isLive=True,
                traderLogin=2123962,
                brokerTitleShort="FP Trading",
            ),
        ],
    )

    rows = ctd.get_available_accounts()
    assert {row["trader_login"] for row in rows} == {1105460, 1105462, 2123962}
    live = next(row for row in rows if row["trader_login"] == 2123962)
    active = next(row for row in rows if row["trader_login"] == 1105460)
    assert live["account_type"] == "live"
    assert live["is_live"] is True
    assert active["active"] is True
    assert active["selected"] is True


def test_active_account_list_starts_opposite_environment_probe(monkeypatch) -> None:
    sent = []
    probes = []
    monkeypatch.setattr(ctd, "ACCOUNT_ID", 47140414)
    monkeypatch.setattr(ctd, "CLIENT_HOST_TYPE", "demo")
    monkeypatch.setattr(ctd, "ACTIVE_ACCOUNT_ID", None)
    monkeypatch.setattr(ctd, "_ACCOUNT_DIRECTORY_BY_HOST", {"demo": [], "live": []})
    monkeypatch.setattr(ctd, "AVAILABLE_ACCOUNTS", [])
    monkeypatch.setattr(
        ctd.Protobuf,
        "extract",
        lambda _: SimpleNamespace(
            ctidTraderAccount=[
                SimpleNamespace(
                    ctidTraderAccountId=47140414,
                    isLive=False,
                    traderLogin=1105460,
                    brokerTitleShort="FP Trading",
                )
            ]
        ),
    )
    monkeypatch.setattr(ctd.client, "send", lambda request: sent.append(request) or _Deferred())
    monkeypatch.setattr(
        ctd,
        "_start_account_directory_probe",
        lambda host_type: probes.append(host_type) or True,
    )

    ctd.account_list_response_cb(object())

    assert probes == ["live"]
    assert [row["trader_login"] for row in ctd.get_available_accounts()] == [1105460]
    assert len(sent) == 1
    assert sent[0].ctidTraderAccountId == 47140414


def test_probe_account_list_merges_without_replacing_active_host_accounts(monkeypatch) -> None:
    probe = SimpleNamespace()
    monkeypatch.setattr(ctd, "ACCOUNT_ID", 47140414)
    monkeypatch.setattr(ctd, "ACTIVE_ACCOUNT_ID", 47140414)
    monkeypatch.setattr(ctd, "_ACCOUNT_DIRECTORY_BY_HOST", {"demo": [], "live": []})
    monkeypatch.setattr(ctd, "AVAILABLE_ACCOUNTS", [])
    monkeypatch.setattr(
        ctd,
        "_ACCOUNT_DIRECTORY_PROBE_CLIENTS",
        {"live": probe},
    )
    monkeypatch.setattr(
        ctd,
        "_ACCOUNT_DIRECTORY_PROBE_ERRORS",
        {"demo": None, "live": None},
    )
    monkeypatch.setattr(ctd, "_stop_client_service", lambda _: None)

    ctd._replace_account_directory_host(
        "demo",
        [
            SimpleNamespace(
                ctidTraderAccountId=47140414,
                isLive=False,
                traderLogin=1105460,
                brokerTitleShort="FP Trading",
            )
        ],
    )
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
                )
            ]
        ),
    )

    ctd._account_directory_probe_list_cb(object(), "live", probe)

    rows = ctd.get_available_accounts()
    assert {row["trader_login"] for row in rows} == {1105460, 2123962}
    assert ctd._ACCOUNT_DIRECTORY_PROBE_CLIENTS == {}


def test_account_list_retains_accounts_and_selection_truth(monkeypatch) -> None:
    sent = []
    monkeypatch.setattr(ctd, "ACCOUNT_ID", 47140414)
    monkeypatch.setattr(ctd, "HOST_TYPE", "demo")
    monkeypatch.setattr(ctd, "CLIENT_HOST_TYPE", "demo")
    monkeypatch.setattr(ctd, "ACTIVE_ACCOUNT_ID", None)
    monkeypatch.setattr(ctd, "ACTIVE_HOST_TYPE", None)
    monkeypatch.setattr(ctd, "AVAILABLE_ACCOUNTS", [])
    monkeypatch.setattr(ctd, "_ACCOUNT_DIRECTORY_BY_HOST", {"demo": [], "live": []})
    monkeypatch.setattr(ctd, "_ACCOUNT_DIRECTORY_PROBE_CLIENTS", {})
    monkeypatch.setattr(ctd.reactor, "running", False, raising=False)
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

    deferred = ctd.account_list_response_cb(object())

    rows_by_id = {
        row["account_id"]: row
        for row in ctd.get_available_accounts()
    }
    assert set(rows_by_id) == {47139918, 47140414, 47140449}
    assert rows_by_id[47139918]["account_type"] == "live"
    assert rows_by_id[47139918]["trader_login"] == 2123962
    assert rows_by_id[47140414]["account_type"] == "demo"
    assert rows_by_id[47140414]["trader_login"] == 1105460
    assert rows_by_id[47140414]["selected"] is True
    assert rows_by_id[47140449]["trader_login"] == 1105462
    assert all(row["broker_title"] == "FP Trading" for row in rows_by_id.values())
    assert len(sent) == 1
    assert sent[0].ctidTraderAccountId == 47140414

    success, _ = deferred.callbacks
    success(object())

    active_rows = ctd.get_available_accounts()
    active = next(row for row in active_rows if row["account_id"] == 47140414)
    assert active["active"] is True
    assert ctd.ACTIVE_ACCOUNT_ID == 47140414
    assert ctd.ACTIVE_HOST_TYPE == "demo"


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
