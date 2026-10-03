# MONKEY-PATCH: Fix for Twisted dependency issue in ctrader-open-api
# The version of Twisted required by ctrader-open-api has a bug where
# CertificateOptions is not imported. We inject it here before it's used.
try:
    from twisted.internet import endpoints, ssl

    if not hasattr(endpoints, "CertificateOptions"):
        endpoints.CertificateOptions = ssl.CertificateOptions
except ImportError:
    pass  # If twisted isn't installed, app will fail later with a clearer error.

from time import perf_counter

from fastapi import FastAPI, Request

from backend.app_bootstrap import configure_app
from backend.api.router import router as api_router
from backend.services.latency_observability import (
    record_api_latency,
    record_api_unavailable,
)


app = FastAPI()

_LATENCY_REPORT_PATH = "/api/reports/broker-api-latency"


@app.middleware("http")
async def record_api_request_latency(request: Request, call_next):
    path = request.url.path
    if not path.startswith("/api/") or path == _LATENCY_REPORT_PATH:
        return await call_next(request)

    started = perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        try:
            record_api_unavailable(
                request.method,
                path,
                f"{type(exc).__name__}: {exc}",
            )
        except Exception:
            pass
        raise

    duration_ms = max(0.0, (perf_counter() - started) * 1000.0)
    try:
        if response.status_code >= 500:
            record_api_unavailable(
                request.method,
                path,
                f"API request completed with HTTP {response.status_code}.",
                status_code=response.status_code,
            )
        else:
            record_api_latency(
                request.method,
                path,
                response.status_code,
                duration_ms,
            )
    except Exception:
        # Observability must never change API response behavior.
        pass
    return response


app.include_router(api_router)

configure_app(app)
