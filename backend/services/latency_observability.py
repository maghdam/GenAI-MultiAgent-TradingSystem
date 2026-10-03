from __future__ import annotations

from datetime import UTC, datetime
from threading import Lock

from backend.domain.models import BrokerApiLatencyResponse, LatencyObservation


_LOCK = Lock()


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _empty(scope: str) -> LatencyObservation:
    return LatencyObservation(
        state="no_observation",
        scope=scope,
        detail="No latency observation has been recorded in this process yet.",
    )


_broker_observation = _empty("broker_service_call")
_api_observation = _empty("api_request")


def reset_latency_observations() -> None:
    """Reset process-local latency evidence. Intended for startup/tests only."""
    global _broker_observation, _api_observation
    with _LOCK:
        _broker_observation = _empty("broker_service_call")
        _api_observation = _empty("api_request")


def record_broker_latency(operation: str, duration_ms: float) -> None:
    global _broker_observation
    observation = LatencyObservation(
        state="measured",
        scope="broker_service_call",
        duration_ms=max(0.0, float(duration_ms)),
        operation=operation,
        observed_at=_now(),
        detail="Broker-facing service call completed successfully.",
    )
    with _LOCK:
        _broker_observation = observation


def record_broker_unavailable(operation: str, detail: str) -> None:
    global _broker_observation
    observation = LatencyObservation(
        state="unavailable",
        scope="broker_service_call",
        operation=operation,
        observed_at=_now(),
        detail=detail or "Broker-facing service call did not produce a successful latency sample.",
    )
    with _LOCK:
        _broker_observation = observation


def record_api_latency(method: str, path: str, status_code: int, duration_ms: float) -> None:
    global _api_observation
    observation = LatencyObservation(
        state="measured",
        scope="api_request",
        duration_ms=max(0.0, float(duration_ms)),
        method=method.upper(),
        path=path,
        status_code=int(status_code),
        observed_at=_now(),
        detail="API request completed with a non-5xx response.",
    )
    with _LOCK:
        _api_observation = observation


def record_api_unavailable(
    method: str,
    path: str,
    detail: str,
    *,
    status_code: int | None = None,
) -> None:
    global _api_observation
    observation = LatencyObservation(
        state="unavailable",
        scope="api_request",
        method=method.upper(),
        path=path,
        status_code=status_code,
        observed_at=_now(),
        detail=detail or "API request did not produce a successful latency sample.",
    )
    with _LOCK:
        _api_observation = observation


def build_broker_api_latency() -> BrokerApiLatencyResponse:
    with _LOCK:
        broker = _broker_observation.model_copy(deep=True)
        api = _api_observation.model_copy(deep=True)

    return BrokerApiLatencyResponse(
        broker=broker,
        api=api,
        message=(
            "Latency samples are process-local and describe only the latest observed existing "
            "broker-service/API boundary calls; the report does not generate probe traffic."
        ),
    )
