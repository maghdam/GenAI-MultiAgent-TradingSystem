# MONKEY-PATCH: Fix for Twisted dependency issue in ctrader-open-api
# The version of Twisted required by ctrader-open-api has a bug where
# CertificateOptions is not imported. We inject it here before it's used.
try:
    from twisted.internet import endpoints, ssl

    if not hasattr(endpoints, "CertificateOptions"):
        endpoints.CertificateOptions = ssl.CertificateOptions
except ImportError:
    pass  # If twisted isn't installed, app will fail later with a clearer error.

from fastapi import FastAPI

from backend.app_bootstrap import configure_app
from backend.api.router import router as api_router


app = FastAPI()

app.include_router(api_router)

configure_app(app)
