import os
import re
import json
import asyncio
import time
from typing import Tuple, Optional, Dict, Any, List

import requests
import pandas as pd
from backend.smc_features import build_feature_snapshot


# =========================
# Runtime toggles (env vars)
# =========================
JSON_ONLY_DEFAULT = os.getenv("LLM_JSON_ONLY", "1") == "1"       # prefer compact JSON
MAX_TOK_DEFAULT   = int(os.getenv("LLM_NUM_PREDICT", "96"))      # cap tokens (lower = faster)
# hard cap per attempt to avoid proxy 60s cutters; can raise if your infra allows
ATTEMPT_TIMEOUT_DEFAULT = float(os.getenv("OLLAMA_TIMEOUT", "45"))
MODEL_DEFAULT     = os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip()
FALLBACK_MODEL    = os.getenv("OLLAMA_FALLBACK_MODEL", "llama3.2:3b").strip()
OLLAMA_URL        = os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/")
KEEP_ALIVE        = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
# overall time budget for the whole function (to avoid FastAPI/proxy hard limits)
OVERALL_BUDGET_S  = float(os.getenv("LLM_OVERALL_BUDGET", "110"))

# Serialize heavy calls so we don’t race the CPU/VMM
_ANALYZE_LOCK = asyncio.Semaphore(1)


class TradeDecision:
    def __init__(self, signal: str, sl: float = None, tp: float = None,
                 confidence: float = None, reasons: Optional[List[str]] = None, rationale: str = ''):
        self.signal = signal
        self.sl = sl
        self.tp = tp
        self.confidence = confidence
        self.reasons = reasons or []
        self.rationale = rationale or ' '.join(self.reasons)

    def dict(self):
        return {
            "signal": self.signal,
            "sl": self.sl,
            "tp": self.tp,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "rationale": self.rationale,
        }


# --------------------------
# Utility: Ollama health ping
# --------------------------
def _ollama_healthcheck(ollama_url: str, timeout: float) -> Tuple[bool, str]:
    try:
        r = requests.get(f"{ollama_url}/api/tags", timeout=min(5.0, timeout))
        if r.status_code == 200:
            return True, "ok"
        return False, f"/api/tags -> {r.status_code}"
    except Exception as e:
        return False, f"healthcheck error: {e}"


# --------------------------
# Build prompts (JSON-only or JSON+Explanation)
# --------------------------
def _build_prompts(symbol: str, timeframe: str, smc_text: str,
                   last_rows: pd.DataFrame, json_only: bool) -> str:
    ohlc_text = last_rows.to_string(index=False)
    if json_only:
        return f"""
You are a professional Smart Money Concepts (SMC) trading analyst.
Analyze {symbol} ({timeframe}) using the data provided.
Return ONLY a single JSON object with these keys and no extra text:

{{
  "signal": "long" | "short" | "no_trade",
  "sl": float,
  "tp": float,
  "confidence": float
}}

SMC Summary:
{smc_text}

OHLC (last {len(last_rows)} candles):
{ohlc_text}
""".strip()

    return f"""
You are a professional Smart Money Concepts (SMC) trading analyst.

Analyze the market for {symbol} ({timeframe}) using:
- the data below (no image)
- OHLC data
- extracted SMC features

SMC Summary:
{smc_text}

OHLC (last {len(last_rows)} candles):
{ohlc_text}

Make a trading decision using SMC methodology (CHoCH/BOS, FVGs, OBs, premium/discount, liquidity, structure, trend).

Respond in this exact format:

{{
  "signal": "long" | "short" | "no_trade",
  "sl": float,
  "tp": float,
  "confidence": float
}}

Then on the next line, write a plain English explanation starting with:
Explanation: <text>
""".strip()


def _looks_like_cancelled_or_timeout(status_code: int, body: str) -> bool:
    body_l = (body or "").lower()
    if status_code in (408, 499, 502, 503, 504, 500):
        if "context canceled" in body_l or "timeout" in body_l or "deadline" in body_l:
            return True
    if "post predict" in body_l and "context canceled" in body_l:
        return True
    return False


# --------------------------
# Call Ollama /api/generate
# --------------------------
def _ollama_generate(
    prompt: str,
    model: str,
    timeout: float,
    json_only: bool,
    options_overrides: Optional[Dict[str, Any]] = None,
) -> str:
    # Clamp attempt timeout to avoid infra 60s read timeouts
    per_attempt_timeout = max(5.0, min(timeout, ATTEMPT_TIMEOUT_DEFAULT))

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "num_predict": MAX_TOK_DEFAULT,
            "repeat_penalty": 1.1,
        },
    }
    if json_only:
        payload["format"] = "json"  # compact JSON mode (supported by Llama/LLaVA in Ollama)

    if options_overrides:
        payload["options"].update(options_overrides)

    try:
        # Use (connect, read) timeouts: connect fast, bound read
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=payload,
            timeout=(min(5.0, per_attempt_timeout), per_attempt_timeout),
        )
    except requests.exceptions.Timeout:
        raise ValueError(f"LLM request timed out after {per_attempt_timeout} seconds.")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to contact Ollama API: {e}")

    if r.status_code != 200:
        # If this looks like a timeout/cancel, hint caller to fallback
        if _looks_like_cancelled_or_timeout(r.status_code, r.text):
            raise TimeoutError(f"Ollama cancelled/timeout: {r.status_code} — {r.text}")
        raise RuntimeError(f"Ollama API error: {r.status_code} — {r.text}")

    content = r.json().get("response", "").strip()
    if not content:
        raise ValueError("Empty response from LLM.")
    return content


# --------------------------
# Parse model response
# --------------------------
def _parse_trade_decision(content: str, json_only: bool) -> TradeDecision:
    if json_only:
        parsed = json.loads(content)
        return TradeDecision(**parsed, reasons=[])

    m = re.search(r"\{.*?\}", content, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found in LLM response.\nRaw:\n{content}")
    json_block = m.group(0).strip()
    explanation = content[m.end():].strip()
    explanation = re.sub(r"^Explanation:\s*", "", explanation, flags=re.IGNORECASE).strip()
    parsed = json.loads(json_block)
    return TradeDecision(**parsed, reasons=[explanation] if explanation else [])


def _warm_ollama_sync(model: Optional[str] = None) -> None:
    """Synchronous part of the warm-up logic."""
    mdl = (model or MODEL_DEFAULT).strip()
    print(f"[LLM Warmup] Starting model load for '{mdl}'...")
    try:
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            json={
                "model": mdl,
                "prompt": 'Return {"warm":true}',
                "stream": False,
                "format": "json",
                "options": {"num_predict": 8},
                "keep_alive": KEEP_ALIVE,
            },
            timeout=(5, 120),  # Increased read timeout to 120 seconds
        )
        print(f"[LLM Warmup] Model '{mdl}' is ready.")
    except Exception as e:
        print(f"[LLM Warmup] Failed to warm up model '{mdl}': {e}")

def warm_ollama(model: Optional[str] = None) -> None:
    """Run the model warmup in a background thread to avoid blocking startup."""
    import threading
    thread = threading.Thread(target=_warm_ollama_sync, args=(model,), daemon=True)
    thread.start()


# ==========================================================
# Main entry with health check + progressive fallbacks
# ==========================================================
async def analyze_data_with_llm(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    indicators=None,
    *,
    model: Optional[str] = None,
    options: Optional[dict] = None,
    ollama_url: Optional[str] = None,
) -> TradeDecision:
    async with _ANALYZE_LOCK:
        start = time.time()

        indicators = indicators or []
        model = (model or MODEL_DEFAULT).strip()
        json_only = JSON_ONLY_DEFAULT
        options = (options or {}).copy()

        # Resolve timeout (per attempt) and clamp
        timeout_value = options.pop("timeout", None)
        if timeout_value is None:
            timeout_value = ATTEMPT_TIMEOUT_DEFAULT
        try:
            timeout_value = float(timeout_value)
            if timeout_value <= 0:
                timeout_value = ATTEMPT_TIMEOUT_DEFAULT
        except Exception:
            timeout_value = ATTEMPT_TIMEOUT_DEFAULT
        timeout_value = min(timeout_value, ATTEMPT_TIMEOUT_DEFAULT)

        # Health check first — fail fast if Ollama not responsive
        ok, note = _ollama_healthcheck(OLLAMA_URL, timeout_value)
        if not ok:
            raise ValueError(f"Ollama not healthy: {note}")

        # Prepare SMC + OHLC snapshots
        def build_inputs(candles: int) -> Tuple[str, pd.DataFrame, List[str]]:
            last_rows = df.tail(candles)[["open", "high", "low", "close"]]
            smc_summary = build_feature_snapshot(df)
            reason_lines = []
            for k, v in smc_summary.items():
                if v is None:
                    continue
                if isinstance(v, str) and not v.strip():
                    continue
                reason_lines.append(f"{k}: {v}")
            if not reason_lines:
                reason_lines.append("No strong SMC features detected.")
            smc_text = "\n".join([f"- {line}" for line in reason_lines])
            return smc_text, last_rows, reason_lines

        attempts_summary = []
        latest_reasons: List[str] = []

        def budget_left() -> float:
            return OVERALL_BUDGET_S - (time.time() - start)

        # -------- Attempt 1: Text-only on primary model --------
        try:
            if budget_left() <= 5:
                raise TimeoutError("Budget exhausted before A1.")
            smc_text, last_rows, latest_reasons = build_inputs(24)
            prompt = _build_prompts(symbol, timeframe, smc_text, last_rows, json_only)
            content = _ollama_generate(
                prompt=prompt,
                model=model,
                timeout=min(timeout_value, max(5.0, budget_left() - 2)),
                json_only=json_only,
                options_overrides={"num_predict": min(80, MAX_TOK_DEFAULT)},
            )
            td = _parse_trade_decision(content, json_only)
            td.reasons = (td.reasons or []) + latest_reasons
            return td
        except Exception as e:
            attempts_summary.append(f"A1 (text-only, model={model}): {e}")

        # -------- Attempt 2: Text-only fallback model --------
        try:
            if budget_left() <= 5:
                raise TimeoutError("Budget exhausted before A2.")
            smc_text, last_rows, latest_reasons = build_inputs(24)
            prompt = _build_prompts(symbol, timeframe, smc_text, last_rows, json_only)
            content = _ollama_generate(
                prompt=prompt,
                model=FALLBACK_MODEL,
                timeout=min(timeout_value, max(5.0, budget_left() - 2)),
                json_only=json_only,
                options_overrides={"num_predict": min(80, MAX_TOK_DEFAULT)},
            )
            td = _parse_trade_decision(content, json_only)
            td.reasons = (td.reasons or []) + latest_reasons
            return td
        except Exception as e:
            attempts_summary.append(f"A2 (text-only, fallback={FALLBACK_MODEL}): {e}")

        # If everything failed, raise one consolidated error
        attempts_text = "\n".join(attempts_summary)
        raise ValueError("❌ All inference attempts failed. Summary:\n" + attempts_text)


async def analyze_chart_with_llm(
    fig,  # kept for compatibility; ignored in text-only mode
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    indicators=None,
    *,
    model: Optional[str] = None,
    options: Optional[dict] = None,
    ollama_url: Optional[str] = None,
    **kwargs,
) -> TradeDecision:
    """Backward-compatible wrapper used by `backend.agents.runner`.

    We now operate in text-only mode, so the `fig` argument is unused but
    retained to avoid touching the callers. Any extra keyword arguments are
    ignored.
    """
    return await analyze_data_with_llm(
        df=df,
        symbol=symbol,
        timeframe=timeframe,
        indicators=indicators,
        model=model,
        options=options,
        ollama_url=ollama_url,
    )
