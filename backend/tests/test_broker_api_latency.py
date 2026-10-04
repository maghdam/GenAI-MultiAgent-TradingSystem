from __future__ import annotations

import importlib

from fastapi.testclient import TestClient
import pytest

from backend.domain.models import BrokerAccountSnapshot, BrokerStatus, EngineConfig
from backend.services import broker as broker_module
from backend.services.latency_observability import (
    build_broker_api_latency,
    reset_latency_observations,
)


def _ready_status() -> BrokerStatus:
    return BrokerStatus(
        connected=True,
        socket_connected=True,
        account_authorized=True,
        symbols_loaded=100,
        open_positions=1,
        pending_orders=0,
        ready=True,
        market_data_ready=True,
        broker_mode="ctrader_demo",
        account_id=123,
        account_type="demo",
        demo_account_confirmed=True,
        execution_ready=True,
    )


def _unavailable_status() -> BrokerStatus:
    return BrokerStatus(
        connected=False,
        socket_connected=False,
        account_authorized=False,
        symbols_loaded=0,
        open_positions=0,
        pending_orders=0,
        ready=False,
        market_data_ready=False,
        broker_mode="ctrader",
        account_type="unknown",
        demo_account_confirmed=False,
        execution_ready=False,
    )


def _clock(monkeypatch, module, start: float, end: float) -> None:
    values = iter((start, end))
    monkeypatch.setattr(module, "perf_counter", lambda: next(values))


@pytest.fixture(autouse=True)
def _reset_latency_state():
    reset_latency_observations()
    yield
    reset_latency_observations()


def test_latency_report_starts_with_explicit_no_observation_state() -> None:
    report = build_broker_api_latency()

    assert report.persistence == "process_memory"
    assert report.broker.state == "no_observation"
    assert report.broker.duration_ms is None
    assert report.api.state == "no_observation"
    assert report.api.duration_ms is None


def test_local_broker_status_getter_is_not_counted_as_broker_latency(monkeypatch) -> None:
    monkeypatch.setattr(broker_module.adapter, "get_status", _ready_status)

    status = broker_module.get_broker_status()
    report = build_broker_api_latency()

    assert status.execution_ready is True
    assert report.broker.state == "no_observation"


def test_successful_broker_service_call_records_latency_and_preserves_result(monkeypatch) -> None:
    row = {
        "symbol": "NAS100",
        "direction": "buy",
        "position_id": 990001,
    }
    calls = {"positions": 0}

    def _positions():
        calls["positions"] += 1
        return [row]

    monkeypatch.setattr(broker_module.adapter, "list_positions", _positions)
    monkeypatch.setattr(broker_module.ctd, "is_connected", lambda: True)
    monkeypatch.setattr(broker_module.ctd, "is_authorized", lambda: True)
    monkeypatch.setattr(broker_module.ctd, "is_account_confirmed", lambda: True)
    _clock(monkeypatch, broker_module, 100.0, 100.042)

    result = broker_module.list_positions()
    report = build_broker_api_latency()

    assert result == [row]
    assert calls == {"positions": 1}
    assert report.broker.state == "measured"
    assert report.broker.scope == "broker_service_call"
    assert report.broker.operation == "list_positions"
    assert report.broker.duration_ms == pytest.approx(42.0)
    assert report.broker.observed_at is not None


def test_broker_exception_is_unavailable_and_preserves_original_exception(monkeypatch) -> None:
    def _boom():
        raise RuntimeError("simulated broker failure")

    monkeypatch.setattr(broker_module.adapter, "list_positions", _boom)
    monkeypatch.setattr(
        broker_module.ctd,
        "is_connected",
        lambda: pytest.fail("availability classification must not run after broker exception"),
    )
    monkeypatch.setattr(broker_module, "perf_counter", lambda: 200.0)

    with pytest.raises(RuntimeError, match="simulated broker failure"):
        broker_module.list_positions()

    report = build_broker_api_latency()
    assert report.broker.state == "unavailable"
    assert report.broker.operation == "list_positions"
    assert report.broker.duration_ms is None
    assert "simulated broker failure" in report.broker.detail


def test_unavailable_broker_result_is_not_recorded_as_successful_latency(monkeypatch) -> None:
    monkeypatch.setattr(broker_module.adapter, "list_positions", lambda: [])
    monkeypatch.setattr(broker_module.ctd, "is_connected", lambda: False)
    monkeypatch.setattr(broker_module.ctd, "is_authorized", lambda: True)
    monkeypatch.setattr(broker_module.ctd, "is_account_confirmed", lambda: True)
    _clock(monkeypatch, broker_module, 300.0, 300.015)

    assert broker_module.list_positions() == []

    report = build_broker_api_latency()
    assert report.broker.state == "unavailable"
    assert report.broker.operation == "list_positions"
    assert report.broker.duration_ms is None
    assert "without confirmed demo execution availability" in report.broker.detail


def test_unverified_account_snapshot_is_unavailable_not_measured(monkeypatch) -> None:
    monkeypatch.setattr(
        broker_module.adapter,
        "get_account_snapshot",
        lambda force=False: BrokerAccountSnapshot(
            account_id=123,
            source="ctrader",
            verified=False,
            notes=["account snapshot timeout"],
        ),
    )
    _clock(monkeypatch, broker_module, 400.0, 400.020)

    snapshot = broker_module.get_broker_account_snapshot(force=True)
    report = build_broker_api_latency()

    assert snapshot.verified is False
    assert report.broker.state == "unavailable"
    assert report.broker.operation == "get_broker_account_snapshot"
    assert report.broker.duration_ms is None


def test_successful_api_request_records_latency_and_report_endpoint_does_not_overwrite_it(
    monkeypatch,
) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_EVENT_INTELLIGENCE_ON_BOOT", "0")

    app_module = importlib.import_module("backend.app")
    _clock(monkeypatch, app_module, 500.0, 500.025)

    with TestClient(app_module.app) as client:
        response = client.get("/api/config")
        report_response = client.get("/api/reports/broker-api-latency")

    assert response.status_code == 200
    assert report_response.status_code == 200
    payload = report_response.json()
    assert payload["api"]["state"] == "measured"
    assert payload["api"]["scope"] == "api_request"
    assert payload["api"]["method"] == "GET"
    assert payload["api"]["path"] == "/api/config"
    assert payload["api"]["status_code"] == 200
    assert payload["api"]["duration_ms"] == pytest.approx(25.0)


def test_completed_4xx_api_response_is_a_measured_boundary_sample(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_EVENT_INTELLIGENCE_ON_BOOT", "0")

    app_module = importlib.import_module("backend.app")
    _clock(monkeypatch, app_module, 600.0, 600.010)

    with TestClient(app_module.app) as client:
        response = client.post(
            "/api/config",
            json=EngineConfig(allow_live=True).model_dump(mode="json"),
        )

    assert response.status_code == 400
    report = build_broker_api_latency()
    assert report.api.state == "measured"
    assert report.api.status_code == 400
    assert report.api.duration_ms == pytest.approx(10.0)


def test_5xx_api_response_is_unavailable_not_successful_latency(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_EVENT_INTELLIGENCE_ON_BOOT", "0")

    app_module = importlib.import_module("backend.app")
    router_module = importlib.import_module("backend.api.router")

    async def _boom():
        raise RuntimeError("simulated API failure")

    monkeypatch.setattr(router_module, "_status_payload", _boom)
    monkeypatch.setattr(app_module, "perf_counter", lambda: 700.0)

    with TestClient(app_module.app, raise_server_exceptions=False) as client:
        response = client.get("/api/status")

    assert response.status_code == 500
    report = build_broker_api_latency()
    assert report.api.state == "unavailable"
    assert report.api.method == "GET"
    assert report.api.path == "/api/status"
    assert report.api.duration_ms is None
    assert "simulated API failure" in report.api.detail


def test_latency_report_is_read_only_with_respect_to_current_samples(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_EVENT_INTELLIGENCE_ON_BOOT", "0")

    app_module = importlib.import_module("backend.app")
    _clock(monkeypatch, app_module, 800.0, 800.030)

    with TestClient(app_module.app) as client:
        assert client.get("/api/config").status_code == 200
        before = build_broker_api_latency().model_dump()
        response = client.get("/api/reports/broker-api-latency")
        after = build_broker_api_latency().model_dump()

    assert response.status_code == 200
    assert before == after
