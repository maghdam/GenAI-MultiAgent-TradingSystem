import asyncio
import os
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.adapters.ctrader import adapter as broker_adapter
from backend.services import model_service
from backend.services.engine import engine as tradeagent_engine
from backend.services.event_calibration import calibrate_pending_event_outcomes
from backend.services.event_intelligence import configured_feed_urls, refresh_configured_feeds
from backend.services.market_data_monitor import market_data_probe_loop
from backend.services.runtime_state import external_dependency_state
from backend.storage.repositories import log_incident


@lru_cache(maxsize=1)
def _allowed_origins() -> list[str]:
    defaults = [
        "http://localhost:8080",
        "http://localhost:5173",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:5173",
    ]
    raw = os.getenv("ALLOWED_ORIGINS", "")
    if raw.strip():
        items = [origin.strip() for origin in raw.split(",") if origin.strip()]
        return items or defaults
    return defaults


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _boot_default() -> bool:
    return not bool(os.getenv("PYTEST_CURRENT_TEST"))


async def _event_intelligence_loop(interval_sec: int) -> None:
    while True:
        try:
            result = await asyncio.to_thread(refresh_configured_feeds)
            if result.errors:
                log_incident(
                    "warning",
                    "event_source_refresh_degraded",
                    "One or more market-event sources failed to refresh.",
                    result.model_dump(mode="json"),
                )
            if _env_flag("EVENT_CALIBRATION_AUTO", True):
                await asyncio.to_thread(calibrate_pending_event_outcomes, 200)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_incident(
                "error",
                "event_source_refresh_failed",
                "Market-event refresh loop failed.",
                {"error": str(exc)},
            )
        await asyncio.sleep(interval_sec)


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    engine_started = False
    event_task: asyncio.Task | None = None
    market_data_task: asyncio.Task | None = None
    external_dependency_state.notes.clear()
    external_dependency_state.ctrader_started = False
    external_dependency_state.ctrader_reason = ""
    external_dependency_state.ollama_warmed = False
    external_dependency_state.ollama_reason = ""

    if _env_flag("APP_START_CTRADER_ON_BOOT", _boot_default()):
        did_start = broker_adapter.start_transport()
        external_dependency_state.ctrader_started = True
        external_dependency_state.ctrader_reason = "booted on startup" if did_start else "transport already running"
    else:
        external_dependency_state.ctrader_reason = "startup disabled by APP_START_CTRADER_ON_BOOT"

    # Generated Strategy Studio files are intentionally not imported into the
    # trusted API/runtime process. Research execution uses the isolated runner.

    if _env_flag("APP_WARM_OLLAMA_ON_BOOT", _boot_default()):
        try:
            external_dependency_state.ollama_reason = model_service.dispatch_warmup()
            external_dependency_state.ollama_warmed = True
        except Exception as exc:
            external_dependency_state.ollama_reason = f"warmup failed: {exc}"
    else:
        external_dependency_state.ollama_reason = "startup disabled by APP_WARM_OLLAMA_ON_BOOT"

    if external_dependency_state.ctrader_started:
        await asyncio.sleep(5)

    if external_dependency_state.ctrader_started:
        await tradeagent_engine.start()
        engine_started = True
        try:
            probe_interval = int(os.getenv("MARKET_DATA_PROBE_INTERVAL_SEC", "60"))
        except ValueError:
            probe_interval = 60
        try:
            probe_retry = int(os.getenv("MARKET_DATA_PROBE_RETRY_SEC", "5"))
        except ValueError:
            probe_retry = 5
        probe_interval = max(15, min(3_600, probe_interval))
        probe_retry = max(2, min(60, probe_retry))
        market_data_task = asyncio.create_task(market_data_probe_loop(probe_interval, probe_retry))
        external_dependency_state.notes.append(
            f"market data: automatic readiness probe every {probe_interval}s (retry {probe_retry}s)"
        )
    else:
        external_dependency_state.notes.append("engine: autostart skipped because broker startup is disabled")

    feed_urls = configured_feed_urls()
    if feed_urls and _env_flag("APP_START_EVENT_INTELLIGENCE_ON_BOOT", _boot_default()):
        try:
            refresh_interval = int(os.getenv("MARKET_NEWS_AUTO_REFRESH_SEC", "300"))
        except ValueError:
            refresh_interval = 300
        refresh_interval = max(30, min(86_400, refresh_interval))
        event_task = asyncio.create_task(_event_intelligence_loop(refresh_interval))
        external_dependency_state.notes.append(
            f"event intelligence: monitoring {len(feed_urls)} sources every {refresh_interval}s"
        )
    elif not feed_urls:
        external_dependency_state.notes.append("event intelligence: no RSS/Atom sources configured")
    else:
        external_dependency_state.notes.append("event intelligence: automatic refresh disabled")

    try:
        yield
    finally:
        if market_data_task:
            market_data_task.cancel()
            try:
                await market_data_task
            except asyncio.CancelledError:
                pass
        if event_task:
            event_task.cancel()
            try:
                await event_task
            except asyncio.CancelledError:
                pass
        if engine_started:
            await tradeagent_engine.stop()


def configure_app(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.router.lifespan_context = app_lifespan
