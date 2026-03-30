import time
from typing import Any, Dict, List

import httpx

from backend.llm_analyzer import FALLBACK_MODEL, MODEL_DEFAULT, OLLAMA_URL, warm_ollama

# Simple cache to prevent blocking the UI
_CACHED_TAGS: Dict[str, Any] = {}
_LAST_FETCH_TS: float = 0
_CACHE_TTL: float = 30.0  # 30 seconds

def default_model() -> str:
    return MODEL_DEFAULT


def fallback_model() -> str:
    return FALLBACK_MODEL


def ollama_url() -> str:
    return OLLAMA_URL


def dispatch_warmup() -> str:
    warm_ollama(MODEL_DEFAULT)
    return f"warmup dispatched for {MODEL_DEFAULT}"


def _extract_model_names(data: Dict[str, Any] | None) -> List[str]:
    models_raw = data.get("models") if isinstance(data, dict) else None
    models: list[str] = []
    if isinstance(models_raw, list):
        for item in models_raw:
            if isinstance(item, dict):
                name = item.get("name") or item.get("model")
                if isinstance(name, str) and name.strip():
                    models.append(name.strip())
    return sorted({name for name in models}, key=lambda value: value.lower())


async def fetch_tags(timeout: float = 10.0, force_refresh: bool = False) -> Dict[str, Any]:
    global _CACHED_TAGS, _LAST_FETCH_TS
    now = time.time()
    if not force_refresh and _CACHED_TAGS and (now - _LAST_FETCH_TS < _CACHE_TTL):
        return _CACHED_TAGS

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
    except Exception as exc:
        # Don't cache failures for too long, but at least for 5 seconds to prevent spam
        res = {
            "ok": False,
            "status_code": None,
            "models": [],
            "error": str(exc),
        }
        if now - _LAST_FETCH_TS > 5.0:
            _CACHED_TAGS = res
            _LAST_FETCH_TS = now
        return res

    payload: Dict[str, Any] | None = None
    if response.headers.get("content-type", "").startswith("application/json"):
        try:
            payload = response.json()
        except Exception:
            payload = None

    _CACHED_TAGS = {
        "ok": response.status_code == 200,
        "status_code": response.status_code,
        "models": _extract_model_names(payload),
        "error": None if response.status_code == 200 else f"ollama /api/tags -> {response.status_code}",
    }
    _LAST_FETCH_TS = now
    return _CACHED_TAGS


async def status_payload(timeout: float = 10.0) -> Dict[str, Any]:
    result = await fetch_tags(timeout=timeout)
    if result["ok"]:
        return {
            "ollama": result["status_code"],
            "model": MODEL_DEFAULT,
            "fallback": FALLBACK_MODEL,
            "reachable": True,
        }
    if result["status_code"] is not None:
        return {
            "ollama": result["status_code"],
            "model": MODEL_DEFAULT,
            "fallback": FALLBACK_MODEL,
            "reachable": False,
            "error": result["error"],
        }
    return {
        "ollama": "unreachable",
        "model": MODEL_DEFAULT,
        "fallback": FALLBACK_MODEL,
        "reachable": False,
        "error": result["error"],
    }


async def models_payload(timeout: float = 10.0) -> Dict[str, Any]:
    result = await fetch_tags(timeout=timeout)
    payload = {
        "provider": "ollama",
        "models": result["models"],
        "default": MODEL_DEFAULT,
        "fallback": FALLBACK_MODEL,
    }
    if not result["ok"]:
        payload["error"] = result["error"]
    return payload
