from __future__ import annotations

from types import SimpleNamespace

from backend import ctrader_client as ctd
from backend.adapters.ctrader import CTraderBrokerAdapter


class _Deferred:
    def __init__(self) -> None:
        self.callbacks = None

    def addCallbacks(self, success, error):
        self.callbacks = (success, error)
        return self


class _FakeClient:
    def __init__(self, *, connected: bool, running: bool) -> None:
        self.isConnected = connected
        self.running = running
        self.started = 0
        self.sent: list[tuple[object, _Deferred]] = []
        self.connected_callback = None
        self.disconnected_callback = None
        self.message_callback = None

    def setConnectedCallback(self, callback) -> None:
        self.connected_callback = callback

    def setDisconnectedCallback(self, callback) -> None:
        self.disconnected_callback = callback

    def setMessageReceivedCallback(self, callback) -> None:
        self.message_callback = callback

    def startService(self) -> None:
        self.running = True
        self.started += 1

    def send(self, request, **kwargs):
        deferred = _Deferred()
        self.sent.append((request, deferred))
        return deferred


def _prime_state(monkeypatch, client: _FakeClient, *, account_id: int = 111) -> None:
    monkeypatch.setattr(ctd, "client", client)
    monkeypatch.setattr(ctd, "ACCOUNT_ID", account_id)
    monkeypatch.setattr(ctd, "HOST_TYPE", "demo")
    monkeypatch.setattr(ctd, "CLIENT_HOST_TYPE", "demo")
    monkeypatch.setattr(ctd, "CONNECTED", True)
    monkeypatch.setattr(ctd, "AUTHORIZED", True)
    monkeypatch.setattr(ctd, "ACCOUNT_IS_DEMO", True)
    monkeypatch.setattr(ctd, "ACTIVE_ACCOUNT_ID", account_id)
    monkeypatch.setattr(ctd, "ACTIVE_HOST_TYPE", "demo")
    monkeypatch.setattr(ctd, "ACCOUNT_SWITCH_IN_PROGRESS", False)
    monkeypatch.setattr(ctd, "ACCOUNT_SWITCH_TARGET_ID", None)
    monkeypatch.setattr(ctd, "ACCOUNT_SWITCH_ERROR", None)
    monkeypatch.setattr(ctd, "ACCOUNT_VERIFICATION_ERROR", None)
    monkeypatch.setattr(ctd, "AUTH_ERROR", None)
    monkeypatch.setattr(ctd, "AVAILABLE_ACCOUNTS", [])
    monkeypatch.setattr(ctd, "reactor", SimpleNamespace(running=False))


def test_same_host_account_switch_reauthenticates_without_replacing_client(monkeypatch) -> None:
    current = _FakeClient(connected=True, running=True)
    _prime_state(monkeypatch, current)

    result = ctd.switch_account(222, "demo")

    assert result["switch_started"] is True
    assert result["cross_host"] is False
    assert ctd.client is current
    assert ctd.AUTHORIZED is False
    assert ctd.ACTIVE_ACCOUNT_ID is None
    assert ctd.ACCOUNT_SWITCH_IN_PROGRESS is True
    assert len(current.sent) == 1
    account_auth_request, deferred = current.sent[0]
    assert account_auth_request.ctidTraderAccountId == 222

    success, _ = deferred.callbacks
    success(object())

    assert ctd.AUTHORIZED is True
    assert ctd.ACTIVE_ACCOUNT_ID == 222
    assert ctd.ACTIVE_HOST_TYPE == "demo"
    assert ctd.ACCOUNT_SWITCH_IN_PROGRESS is False
    assert ctd.ACCOUNT_SWITCH_ERROR is None
    assert len(current.sent) == 2
    asset_request, _ = current.sent[1]
    assert asset_request.ctidTraderAccountId == 222


def test_cross_host_switch_replaces_client_and_ignores_stale_disconnect(monkeypatch) -> None:
    current = _FakeClient(connected=True, running=True)
    replacement = _FakeClient(connected=False, running=False)
    _prime_state(monkeypatch, current)

    stopped = []
    monkeypatch.setattr(ctd, "_new_client", lambda host_type: replacement)
    monkeypatch.setattr(ctd, "_stop_client_service", lambda target: stopped.append(target))

    result = ctd.switch_account(333, "live")

    assert result["switch_started"] is True
    assert result["cross_host"] is True
    assert ctd.client is replacement
    assert ctd.CLIENT_HOST_TYPE == "live"
    assert ctd.HOST_TYPE == "live"
    assert ctd.CONNECTED is False
    assert ctd.AUTHORIZED is False
    assert ctd.ACTIVE_ACCOUNT_ID is None
    assert stopped == [current]
    assert replacement.started == 1
    assert replacement.connected_callback is not None

    replacement.isConnected = True
    replacement.connected_callback(replacement)

    assert ctd.CONNECTED is True
    assert len(replacement.sent) == 1
    app_auth_request, _ = replacement.sent[0]
    assert app_auth_request.clientId == ctd.CLIENT_ID

    ctd._on_disconnected(current, "stale old transport")

    assert ctd.CONNECTED is True
    assert ctd.client is replacement


def test_configure_target_account_rebuilds_client_before_transport_start(monkeypatch) -> None:
    current = _FakeClient(connected=False, running=False)
    replacement = _FakeClient(connected=False, running=False)
    _prime_state(monkeypatch, current)
    monkeypatch.setattr(current, "running", False)
    monkeypatch.setattr(ctd, "_new_client", lambda host_type: replacement)

    ctd.configure_target_account(444, "live")

    assert ctd.ACCOUNT_ID == 444
    assert ctd.HOST_TYPE == "live"
    assert ctd.CLIENT_HOST_TYPE == "live"
    assert ctd.client is replacement
    assert ctd.AUTHORIZED is False
    assert ctd.ACTIVE_ACCOUNT_ID is None
    assert ctd.ACTIVE_HOST_TYPE is None


def test_adapter_start_transport_applies_persisted_target_before_thread_start(monkeypatch) -> None:
    configured = []
    started = []

    class _Thread:
        def __init__(self, *, target, daemon):
            self.target = target
            self.daemon = daemon

        def start(self):
            started.append((self.target, self.daemon))

    monkeypatch.setattr(ctd, "configure_target_account", lambda account_id, account_type: configured.append((account_id, account_type)))
    monkeypatch.setattr("backend.adapters.ctrader.threading.Thread", _Thread)

    adapter = CTraderBrokerAdapter()
    did_start = adapter.start_transport(account_id=555, account_type="live")

    assert did_start is True
    assert configured == [(555, "live")]
    assert started == [(ctd.init_client, True)]
