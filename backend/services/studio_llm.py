from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Dict, List

import httpx

from backend.llm_analyzer import FALLBACK_MODEL, MODEL_DEFAULT, _ollama_generate
from backend.services import model_service

SUPPORTED_PROVIDERS = ("ollama", "gemini")
_GEMINI_API_BASE = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
_GEMINI_DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
_GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-pro").strip()
_OLLAMA_STUDIO_MODEL = os.getenv("STUDIO_OLLAMA_MODEL", "muse-glimmer:latest").strip()
_OLLAMA_STUDIO_THINK = os.getenv("STUDIO_OLLAMA_THINK", "low").strip().lower() or None
_GEMINI_MODELS = [
    item.strip()
    for item in os.getenv("GEMINI_MODELS", "gemini-2.5-flash,gemini-2.5-pro").split(",")
    if item.strip()
]


def normalize_provider(value: str | None) -> str:
    provider = (value or "").strip().lower()
    return provider if provider in SUPPORTED_PROVIDERS else "ollama"


def provider_catalog() -> List[Dict[str, Any]]:
    return [
        {"key": "ollama", "label": "Ollama", "configured": True},
        {"key": "gemini", "label": "Gemini", "configured": bool(os.getenv("GEMINI_API_KEY", "").strip())},
    ]


def default_provider() -> str:
    return normalize_provider(os.getenv("STUDIO_LLM_PROVIDER", "ollama"))


def default_model(provider: str | None) -> str:
    provider_key = normalize_provider(provider)
    if provider_key == "gemini":
        return _GEMINI_DEFAULT_MODEL
    return _OLLAMA_STUDIO_MODEL or MODEL_DEFAULT


async def models_payload(provider: str | None = None, timeout: float = 10.0) -> Dict[str, Any]:
    provider_key = normalize_provider(provider or default_provider())
    payload: Dict[str, Any] = {
        "provider": provider_key,
        "providers": provider_catalog(),
        "models": [],
        "default": default_model(provider_key),
        "fallback": _GEMINI_FALLBACK_MODEL if provider_key == "gemini" else FALLBACK_MODEL,
    }

    if provider_key == "gemini":
        payload["models"] = list(_GEMINI_MODELS)
        if not os.getenv("GEMINI_API_KEY", "").strip():
            payload["error"] = "Gemini API key is not configured."
        return payload

    result = await model_service.fetch_tags(timeout=timeout)
    payload["models"] = result["models"]
    if not result["ok"]:
        payload["error"] = result["error"]
    return payload


def _normalize_model_name(model: str | None, provider: str) -> str:
    candidate = (model or "").strip()
    if candidate:
        try:
            if re.fullmatch(r"[A-Za-z0-9_.:/-]{1,128}", candidate):
                return candidate
        except Exception:
            return candidate
    return default_model(provider)


async def _generate_with_ollama(prompt: str, model: str, timeout: float, num_predict: int) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: _ollama_generate(
            prompt=prompt,
            model=model,
            timeout=timeout,
            json_only=False,
            think=_OLLAMA_STUDIO_THINK,
            options_overrides={"num_predict": num_predict},
        ),
    )


def _extract_gemini_text(payload: Dict[str, Any]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return ""
    chunks: list[str] = []
    for candidate in candidates:
        content = candidate.get("content") if isinstance(candidate, dict) else None
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            continue
        for part in parts:
            text = part.get("text") if isinstance(part, dict) else None
            if isinstance(text, str) and text.strip():
                chunks.append(text)
    return "\n".join(chunks).strip()


async def _generate_with_gemini(prompt: str, model: str, timeout: float, num_predict: int) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Gemini API key is not configured.")

    url = f"{_GEMINI_API_BASE}/models/{model}:generateContent"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.9,
            "maxOutputTokens": max(256, num_predict),
        },
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, params={"key": api_key}, json=body)
    if response.status_code >= 400:
        raise RuntimeError(f"Gemini error {response.status_code}: {response.text[:240]}")
    try:
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(f"Gemini returned invalid JSON: {exc}") from exc
    text = _extract_gemini_text(payload)
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text


async def generate_text(
    *,
    prompt: str,
    provider: str | None = None,
    model: str | None = None,
    timeout: float = 45.0,
    num_predict: int = 512,
) -> Dict[str, str]:
    provider_key = normalize_provider(provider or default_provider())
    model_name = _normalize_model_name(model, provider_key)

    if provider_key == "gemini":
        text = await _generate_with_gemini(prompt=prompt, model=model_name, timeout=timeout, num_predict=num_predict)
    else:
        available = await model_service.fetch_tags(timeout=min(timeout, 5.0))
        if not isinstance(available, dict) or not available.get("ok"):
            detail = (
                str(available.get("error") or "Ollama health check failed.")
                if isinstance(available, dict)
                else "Ollama health check failed."
            )
            raise RuntimeError(
                f"Ollama is unavailable: {detail} "
                "Check /api/llm_status and start or reconnect Ollama before retrying."
            )

        discovered_models = available.get("models") if isinstance(available.get("models"), list) else []
        installed_by_name = {
            str(name).strip().lower(): str(name).strip()
            for name in discovered_models
            if str(name).strip()
        }
        if not installed_by_name:
            raise RuntimeError(
                "Ollama is reachable but no models are installed. "
                "Install a model and verify /api/llm_status before retrying."
            )

        fallback_candidates: list[str] = [model_name, MODEL_DEFAULT]
        fallback_candidates.append(str(discovered_models[0]))
        if FALLBACK_MODEL:
            fallback_candidates.append(FALLBACK_MODEL)

        last_error: Exception | None = None
        seen: set[str] = set()
        attempted: list[str] = []
        text = ""
        resolved_model = model_name
        for candidate in fallback_candidates:
            requested_name = (candidate or "").strip()
            installed_name = installed_by_name.get(requested_name.lower())
            if not installed_name or installed_name.lower() in seen:
                continue
            seen.add(installed_name.lower())
            attempted.append(installed_name)
            try:
                text = await _generate_with_ollama(
                    prompt=prompt,
                    model=installed_name,
                    timeout=timeout,
                    num_predict=num_predict,
                )
                resolved_model = installed_name
                break
            except Exception as exc:
                last_error = exc
                continue
        if not text:
            attempts = ", ".join(attempted) if attempted else "none"
            detail = str(last_error) if last_error else "No installed candidate model was eligible."
            raise RuntimeError(
                f"No installed Ollama model completed generation. Attempted models: {attempts}. "
                f"Last error: {detail}"
            )
        model_name = resolved_model

    return {"provider": provider_key, "model": model_name, "text": (text or "").strip()}
