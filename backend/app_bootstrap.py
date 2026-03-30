import asyncio
import os
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.strategy import load_generated_strategies
from backend.adapters.ctrader import adapter as broker_adapter
from backend.services import model_service
from backend.services.engine import engine as tradeagent_engine
from backend.services.runtime_state import external_dependency_state


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


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    engine_started = False
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

    try:
        count = load_generated_strategies()
        print(f"[startup] Loaded {count} generated strategies.")
    except Exception as exc:
        print(f"[startup] Failed to load generated strategies: {exc}")

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
    else:
        external_dependency_state.notes.append("engine: autostart skipped because broker startup is disabled")

    try:
        yield
    finally:
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
