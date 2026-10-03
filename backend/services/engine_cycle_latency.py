from __future__ import annotations

from backend.domain.models import EngineCycleLatencyResponse
from backend.storage.repositories import load_runtime


def build_engine_cycle_latency() -> EngineCycleLatencyResponse:
    """Return restart-safe latency for the most recently completed engine cycle."""

    runtime = load_runtime()
    available = (
        runtime.last_cycle_duration_ms is not None
        and runtime.last_cycle_completed_at is not None
    )

    if available:
        message = "Last completed engine cycle latency is available."
    else:
        message = "No completed engine cycle latency has been recorded yet."

    return EngineCycleLatencyResponse(
        available=available,
        duration_ms=runtime.last_cycle_duration_ms if available else None,
        cycle_started_at=runtime.last_cycle_at if available else None,
        cycle_completed_at=runtime.last_cycle_completed_at if available else None,
        tick_count=runtime.tick_count,
        last_cycle_summary=runtime.last_cycle_summary,
        message=message,
    )
