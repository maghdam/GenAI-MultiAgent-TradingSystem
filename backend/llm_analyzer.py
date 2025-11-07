import os
import re
import json
import asyncio
import time
from typing import Tuple, Optional, Dict, Any, List

import requests
import pandas as pd
from backend.smc_features import build_feature_snapshot
from backend import data_fetcher as _dfetch


# =========================
# Runtime toggles (env vars)
# =========================
JSON_ONLY_DEFAULT = os.getenv("LLM_JSON_ONLY", "1") == "1"       # prefer compact JSON
# Smaller default token budget for faster responses; override via env when needed
MAX_TOK_DEFAULT   = int(os.getenv("LLM_NUM_PREDICT", "64"))      # cap tokens (lower = faster)
# hard cap per attempt to avoid proxy 60s cutters; can raise if your infra allows
ATTEMPT_TIMEOUT_DEFAULT = float(os.getenv("OLLAMA_TIMEOUT", "45"))
# Prefer small, fast defaults suitable for exec-style JSON decisions
MODEL_DEFAULT     = os.getenv("OLLAMA_MODEL", "phi3:mini").strip()
FALLBACK_MODEL    = os.getenv("OLLAMA_FALLBACK_MODEL", "llama3.2:3b-instruct-fp16").strip()
OLLAMA_URL        = os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/")
KEEP_ALIVE        = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
# overall time budget for the whole function (to avoid FastAPI/proxy hard limits)
OVERALL_BUDGET_S  = float(os.getenv("LLM_OVERALL_BUDGET", "110"))
GUARDRAIL_MODE    = os.getenv("LLM_GUARDRAIL_MODE", "loose").strip().lower()  # off|loose|strict
GUARDRAIL_CUTOFF  = float(os.getenv("LLM_GUARDRAIL_CUTOFF", "0.70"))
ENFORCE_VOTES     = os.getenv("LLM_ENFORCE_VOTES", "prefer").strip().lower()  # off|prefer|force
VOTE_THRESHOLD    = int(os.getenv("LLM_VOTE_THRESHOLD", "2"))

# Serialize heavy calls so we don’t race the CPU/VMM
try:
    _ANALYZE_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENT", "1"))
except Exception:
    _ANALYZE_CONCURRENCY = 1
_ANALYZE_CONCURRENCY = max(1, _ANALYZE_CONCURRENCY)
_ANALYZE_LOCK = asyncio.Semaphore(_ANALYZE_CONCURRENCY)


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

SL/TP policy:
- Prefer SMC invalidation: set SL beyond prior protected swing or OB boundary with a small buffer; if not available, use ATR-based sizing (ATR len≈14, mult≈1.0).
- Choose TP so RR≈[1.5, 2.5] relative to SL and entry. Ensure SL below entry for long, above for short. Round prices reasonably and avoid placing SL/TP unrealistically close to entry.

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

SL/TP policy:
- Prefer SMC invalidation: set SL beyond prior protected swing or OB boundary with a small buffer; if not available, use ATR-based sizing (ATR len≈14, mult≈1.0).
- Choose TP so RR≈[1.5, 2.5] relative to SL and entry. Ensure SL below entry for long, above for short. Round and avoid unrealistically tight placements.

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

        # Prepare SMC + OHLC snapshots + votes and HTF bias
        def build_inputs(candles: int) -> Tuple[str, pd.DataFrame, List[str], Dict[str, int], int, int]:
            last_rows = df.tail(candles)[["open", "high", "low", "close"]]
            smc_summary = build_feature_snapshot(df)
            reason_lines: List[str] = []
            for k, v in smc_summary.items():
                if v is None:
                    continue
                if isinstance(v, str) and not v.strip():
                    continue
                reason_lines.append(f"{k}: {v}")
            if not reason_lines:
                reason_lines.append("No strong SMC features detected.")

            # Votes from snapshot
            def _votes_from_snapshot(snapshot: Dict[str, Any]) -> Tuple[Dict[str, int], int]:
                votes: Dict[str, int] = {}
                z = (snapshot.get("zone") or "").lower()
                votes["zone"] = 1 if z == "discount" else (-1 if z == "premium" else 0)
                b = (snapshot.get("bos_choch") or "").lower()
                votes["structure"] = 1 if "bullish" in b else (-1 if "bearish" in b else 0)
                f = (snapshot.get("fvg") or "").lower()
                votes["fvg"] = 1 if "bullish" in f else (-1 if "bearish" in f else 0)
                ob = (snapshot.get("ob") or "").lower()
                votes["ob"] = 1 if ob == "near_ob_lo" else (-1 if ob == "near_ob_hi" else 0)
                try:
                    ts = int(snapshot.get("trend_strength") or 0)
                except Exception:
                    ts = 0
                votes["trend"] = 1 if ts > 1 else (-1 if ts < -1 else 0)
                return votes, sum(votes.values())

            def _htf_of(tf: str) -> str:
                tf = (tf or "M5").upper()
                return {
                    "M1": "H1", "M5": "H1", "M15": "H1",
                    "M30": "H4", "H1": "H4", "H4": "D1", "D1": "D1",
                }.get(tf, "H1")

            def _compute_htf_bias(sym: str, tf: str) -> int:
                try:
                    htf = _htf_of(tf)
                    df_htf, _ = _dfetch.fetch_data(sym, htf, num_bars=300)
                    close = df_htf["close"].astype(float)
                    f = close.rolling(50, min_periods=50).mean()
                    s = close.rolling(200, min_periods=200).mean()
                    if pd.isna(f.iloc[-1]) or pd.isna(s.iloc[-1]):
                        return 0
                    return 1 if f.iloc[-1] > s.iloc[-1] else (-1 if f.iloc[-1] < s.iloc[-1] else 0)
                except Exception:
                    return 0

            votes, total_votes = _votes_from_snapshot(smc_summary)
            htf_bias = _compute_htf_bias(symbol, timeframe) if (symbol and timeframe) else 0
            total_with_htf = total_votes + (htf_bias or 0)

            # Add votes summary + rule into reasons to guide the model
            votes_parts = [
                f"zone={votes['zone']:+d}", f"structure={votes['structure']:+d}",
                f"fvg={votes['fvg']:+d}", f"ob={votes['ob']:+d}", f"trend={votes['trend']:+d}", f"htf={htf_bias:+d}"
            ]
            reason_lines.append(f"votes: total={total_with_htf:+d} ({', '.join(votes_parts)})")
            reason_lines.append("rule: total>=+2 → long; total<=-2 → short; otherwise no_trade")

            smc_text = "\n".join([f"- {line}" for line in reason_lines])
            return smc_text, last_rows, reason_lines, votes, htf_bias, total_with_htf

        attempts_summary = []
        latest_reasons: List[str] = []

        def budget_left() -> float:
            return OVERALL_BUDGET_S - (time.time() - start)

        # Optional gate: skip LLM on weak votes to save time
        GATE_ENABLED = (os.getenv("LLM_GATE_WEAK_VOTES", "1").strip() == "1")
        try:
            GATE_THRESHOLD = int(os.getenv("LLM_GATE_THRESHOLD", str(VOTE_THRESHOLD)))
        except Exception:
            GATE_THRESHOLD = VOTE_THRESHOLD

        # Precompute inputs once (24 candles)
        smc_text, last_rows, latest_reasons, votes, htf_bias, total_with_htf = build_inputs(24)
        if GATE_ENABLED and abs(total_with_htf) < GATE_THRESHOLD:
            reasons = latest_reasons + [f"gate: |votes+htf|={total_with_htf:+d} < {GATE_THRESHOLD}"]
            return TradeDecision(signal="no_trade", sl=None, tp=None, confidence=0.0, reasons=reasons)

        # -------- Attempt 1: Text-only on primary model --------
        try:
            if budget_left() <= 5:
                raise TimeoutError("Budget exhausted before A1.")
            # smc_text,... already built
            prompt = _build_prompts(symbol, timeframe, smc_text, last_rows, json_only)
            content = _ollama_generate(
                prompt=prompt,
                model=model,
                timeout=min(timeout_value, max(5.0, budget_left() - 2)),
                json_only=json_only,
                options_overrides={"num_predict": min(80, MAX_TOK_DEFAULT)},
            )
            td = _parse_trade_decision(content, json_only)
            # Guardrail: block directions that strongly contradict votes unless confidence high and mode=off
            try:
                sig = (td.signal or "").lower()
                conf = float(td.confidence or 0.0)
            except Exception:
                sig = str(td.signal).lower() if td.signal is not None else ""
                conf = 0.0
            conflict_long = sig == "long" and total_with_htf <= -VOTE_THRESHOLD
            conflict_short = sig == "short" and total_with_htf >= VOTE_THRESHOLD
            if GUARDRAIL_MODE in ("loose", "strict"):
                if conflict_long or conflict_short:
                    if GUARDRAIL_MODE == "strict" or conf < GUARDRAIL_CUTOFF:
                        td.signal = "no_trade"
                        td.reasons = (td.reasons or []) + [
                            f"guardrail: overrode {sig} due to votes={total_with_htf:+d} (cutoff={GUARDRAIL_CUTOFF:.2f}, mode={GUARDRAIL_MODE})"
                        ]
            # Enforcement: map votes to direction when requested
            if ENFORCE_VOTES in ("prefer", "force"):
                if total_with_htf >= VOTE_THRESHOLD:
                    if ENFORCE_VOTES == "force" or sig not in ("long",):
                        if sig != "long":
                            td.signal = "long"
                            td.reasons = (td.reasons or []) + [f"enforce: set long from votes={total_with_htf:+d} thr={VOTE_THRESHOLD} (mode={ENFORCE_VOTES})"]
                elif total_with_htf <= -VOTE_THRESHOLD:
                    if ENFORCE_VOTES == "force" or sig not in ("short",):
                        if sig != "short":
                            td.signal = "short"
                            td.reasons = (td.reasons or []) + [f"enforce: set short from votes={total_with_htf:+d} thr={VOTE_THRESHOLD} (mode={ENFORCE_VOTES})"]
                else:
                    if ENFORCE_VOTES == "force" and sig != "no_trade":
                        td.signal = "no_trade"
                        td.reasons = (td.reasons or []) + [f"enforce: set no_trade due to mixed votes={total_with_htf:+d} (mode={ENFORCE_VOTES})"]
            td.reasons = (td.reasons or []) + latest_reasons
            return td
        except Exception as e:
            attempts_summary.append(f"A1 (text-only, model={model}): {e}")

        # -------- Attempt 2: Text-only fallback model --------
        try:
            if budget_left() <= 5:
                raise TimeoutError("Budget exhausted before A2.")
            # reuse smc_text, last_rows, latest_reasons, votes
            prompt = _build_prompts(symbol, timeframe, smc_text, last_rows, json_only)
            content = _ollama_generate(
                prompt=prompt,
                model=FALLBACK_MODEL,
                timeout=min(timeout_value, max(5.0, budget_left() - 2)),
                json_only=json_only,
                options_overrides={"num_predict": min(80, MAX_TOK_DEFAULT)},
            )
            td = _parse_trade_decision(content, json_only)
            try:
                sig = (td.signal or "").lower()
                conf = float(td.confidence or 0.0)
            except Exception:
                sig = str(td.signal).lower() if td.signal is not None else ""
                conf = 0.0
            conflict_long = sig == "long" and total_with_htf <= -VOTE_THRESHOLD
            conflict_short = sig == "short" and total_with_htf >= VOTE_THRESHOLD
            if GUARDRAIL_MODE in ("loose", "strict"):
                if conflict_long or conflict_short:
                    if GUARDRAIL_MODE == "strict" or conf < GUARDRAIL_CUTOFF:
                        td.signal = "no_trade"
                        td.reasons = (td.reasons or []) + [
                            f"guardrail: overrode {sig} due to votes={total_with_htf:+d} (cutoff={GUARDRAIL_CUTOFF:.2f}, mode={GUARDRAIL_MODE})"
                        ]
            # Enforcement: map votes to direction when requested
            if ENFORCE_VOTES in ("prefer", "force"):
                if total_with_htf >= VOTE_THRESHOLD:
                    if ENFORCE_VOTES == "force" or sig not in ("long",):
                        if sig != "long":
                            td.signal = "long"
                            td.reasons = (td.reasons or []) + [f"enforce: set long from votes={total_with_htf:+d} thr={VOTE_THRESHOLD} (mode={ENFORCE_VOTES})"]
                elif total_with_htf <= -VOTE_THRESHOLD:
                    if ENFORCE_VOTES == "force" or sig not in ("short",):
                        if sig != "short":
                            td.signal = "short"
                            td.reasons = (td.reasons or []) + [f"enforce: set short from votes={total_with_htf:+d} thr={VOTE_THRESHOLD} (mode={ENFORCE_VOTES})"]
                else:
                    if ENFORCE_VOTES == "force" and sig != "no_trade":
                        td.signal = "no_trade"
                        td.reasons = (td.reasons or []) + [f"enforce: set no_trade due to mixed votes={total_with_htf:+d} (mode={ENFORCE_VOTES})"]
            td.reasons = (td.reasons or []) + latest_reasons
            return td
        except Exception as e:
            attempts_summary.append(f"A2 (text-only, fallback={FALLBACK_MODEL}): {e}")

        # If everything failed, return a votes-based decision as a graceful fallback
        try:
            # Map votes (+htf) to direction and scale confidence by vote strength
            if total_with_htf >= VOTE_THRESHOLD:
                signal = "long"
            elif total_with_htf <= -VOTE_THRESHOLD:
                signal = "short"
            else:
                signal = "no_trade"
            # Confidence scaling (only when we have a directional signal)
            conf = 0.0
            if signal != "no_trade":
                abs_total = abs(int(total_with_htf))
                # count how many components contribute (zone/structure/fvg/ob/trend)
                try:
                    nonzero_votes = sum(1 for v in (votes or {}).values() if int(v) != 0)
                except Exception:
                    nonzero_votes = 0
                # base from total votes
                base = 0.5 + 0.1 * abs_total            # +0.1 per vote of net strength
                # small bonus for broader consensus across components
                breadth = 0.02 * nonzero_votes          # up to +0.10
                # structure/fvg/trend bonuses
                struct_b = 0.05 if ((votes or {}).get("structure") or 0) != 0 else 0.0
                fvg_b    = 0.03 if ((votes or {}).get("fvg") or 0) != 0 else 0.0
                trend_b  = 0.02 if ((votes or {}).get("trend") or 0) != 0 else 0.0
                # HTF bonus when aligned; slight penalty when opposite
                dir_sign = 1 if signal == "long" else -1
                htf_b    = 0.05 if (int(htf_bias or 0) * dir_sign) > 0 else (-0.03 if (int(htf_bias or 0) * dir_sign) < 0 else 0.0)
                conf = base + breadth + struct_b + fvg_b + trend_b + htf_b
                # Clamp to a sane range
                conf = max(0.55, min(0.90, float(conf)))
            fb_reasons = latest_reasons + [
                "fallback: votes-based due to LLM failure",
                f"attempts: {len(attempts_summary)}",
                f"fallback_confidence_scaled={conf:.2f}",
            ]
            return TradeDecision(signal=signal, sl=None, tp=None, confidence=conf, reasons=fb_reasons)
        except Exception:
            # Last resort: raise consolidated error
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
