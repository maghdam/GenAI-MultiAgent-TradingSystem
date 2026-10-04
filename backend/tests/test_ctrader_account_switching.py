from __future__ import annotations

from types import SimpleNamespace

from backend import ctrader_client as ctd
from backend.adapters.ctrader import CTraderBrokerAdapter
from backend.domain.models import EngineConfig


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


def test_broker_status_exposes_account_switch_transition(monkeypatch) -> None:
    monkeypatch.setattr(ctd, "is_connected", lambda: False)
    monkeypatch.setattr(ctd, "is_authorized", lambda: False)
    monkeypatch.setattr(ctd, "get_auth_error", lambda: "cTrader account switch in progress.")
    monkeypatch.setattr(ctd, "get_last_auth_attempt", lambda: None)
    monkeypatch.setattr(ctd, "is_symbol_metadata_ready", lambda: False)
    monkeypatch.setattr(ctd, "is_demo_account_confirmed", lambda: False)
    monkeypatch.setattr(ctd, "get_account_verification_error", lambda: "cTrader account switch in progress.")
    monkeypatch.setattr(ctd, "get_active_account_id", lambda: None)
    monkeypatch.setattr(ctd, "get_active_host_type", lambda: "unknown")
    monkeypatch.setattr(ctd, "is_account_switch_in_progress", lambda: True)
    monkeypatch.setattr(ctd, "get_account_switch_target_id", lambda: 333)
    monkeypatch.setattr(ctd, "get_account_switch_error", lambda: None)
    monkeypatch.setattr(ctd, "CLIENT_HOST_TYPE", "live")

    status = CTraderBrokerAdapter().get_status()

    assert status.account_id is None
    assert status.account_switch_in_progress is True
    assert status.account_switch_target_id == 333
    assert status.account_switch_error is None
    assert status.broker_mode == "live"
    assert status.execution_ready is False
    assert any("switch is in progress" in note for note in status.notes)


def test_app_bootstrap_uses_persisted_account_before_transport_start(monkeypatch) -> None:
    from backend import app_bootstrap

    calls = {}
    config = EngineConfig(
        selected_ctrader_account_id=333,
        selected_ctrader_account_type="live",
    )

    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "1")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_EVENT_INTELLIGENCE_ON_BOOT", "0")
    monkeypatch.setattr(app_bootstrap, "load_engine_config", lambda defaults: config)
    monkeypatch.setattr(app_bootstrap, "configured_feed_urls", lambda: [])
    monkeypatch.setattr(
        app_bootstrap.broker_adapter,
        "start_transport",
        lambda **kwargs: calls.setdefault("transport", kwargs) is not None,
    )

    async def _engine_start():
        calls["engine_start"] = True

    async def _engine_stop():
        calls["engine_stop"] = True

    async def _sleep(_seconds):
        return None

    async def _market_loop(*_args):
        import asyncio

        await asyncio.Event().wait()

    monkeypatch.setattr(app_bootstrap.tradeagent_engine, "start", _engine_start)
    monkeypatch.setattr(app_bootstrap.tradeagent_engine, "stop", _engine_stop)
    monkeypatch.setattr(app_bootstrap.asyncio, "sleep", _sleep)
    monkeypatch.setattr(app_bootstrap, "market_data_probe_loop", _market_loop)

    async def _run():
        async with app_bootstrap.app_lifespan(object()):
            assert calls["transport"] == {
                "account_id": 333,
                "account_type": "live",
            }
            assert calls["engine_start"] is True

    import asyncio

    asyncio.run(_run())

    assert calls["engine_stop"] is True
