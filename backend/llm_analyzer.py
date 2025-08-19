# backend/llm_analyzer.py
from __future__ import annotations

import os
import re
import json
import base64
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import httpx

from backend.smc_features import build_feature_snapshot

# --- Environment for Plotly/Kaleido in Docker ---
os.environ.setdefault("PLOTLY_CHROME", "/usr/bin/google-chrome-stable")
os.environ.setdefault("CHROME_PATH", "/usr/bin/google-chrome-stable")


def _ensure_chrome_for_kaleido() -> Optional[str]:
    try:
        import kaleido  # noqa
        get_chrome = getattr(kaleido, "get_chrome_sync", None)
        if callable(get_chrome):
            p = get_chrome()
            p_str = str(p)  # 🔧 ensure str for env
            os.environ["PLOTLY_CHROME"] = p_str
            os.environ["CHROME_PATH"] = p_str
            return p_str
    except Exception as e:
        print("[WARN] kaleido.get_chrome_sync() failed:", e)
    return os.environ.get("PLOTLY_CHROME")

def _figure_to_png(fig: go.Figure) -> str:
    _ensure_chrome_for_kaleido()
    fig.update_layout(width=800, height=400)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    try:
        fig.write_image(tmp.name, format="png", engine="kaleido")
    except Exception:
        img = pio.to_image(fig, format="png", engine="kaleido")
        with open(tmp.name, "wb") as f:
            f.write(img)
    return tmp.name



# --- Config ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava:7b")  # e.g., "llava:7b" for speed

@dataclass
class TradeDecision:
    signal: str
    sl: Optional[float] = None
    tp: Optional[float] = None
    confidence: Optional[float] = None
    reasons: List[str] = None

    def dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal,
            "sl": self.sl,
            "tp": self.tp,
            "confidence": self.confidence,
            "reasons": self.reasons or [],
        }

def _build_prompt(strategy: str, symbol: str, timeframe: str, smc_text: str, last_rows: pd.DataFrame) -> str:
    strategy = (strategy or "smc").lower()
    if strategy == "rsi":
        header = "You are an expert RSI divergence trading analyst."
        body = (
            "Focus on:\n"
            "- RSI(14) vs price swings (HH + lower RSI = bearish divergence; LL + higher RSI = bullish divergence)\n"
            "- Consider basic structure/trend and liquidity context\n"
            "- Produce a clear entry direction with reasonable SL/TP and a confidence score\n"
        )
    else:
        header = "You are a professional Smart Money Concepts (SMC) trading analyst."
        body = (
            "Analyze: market structure (CHoCH/BOS), OBs, FVGs, premium/discount, liquidity grabs, and trend strength.\n"
        )

    return f"""
{header}

Instrument: {symbol} ({timeframe})

Inputs:
- The attached candlestick chart image
- Recent OHLC table
- Extracted SMC features (if applicable)

SMC Summary (if present):
{smc_text}

OHLC (last 50):
{last_rows.to_string(index=False)}

{body}

Return this EXACT JSON on the first line:
{{
  "signal": "long" | "short" | "no_trade",
  "sl": float,
  "tp": float,
  "confidence": float
}}

Then on the next line:
Explanation: <plain English explanation>
""".strip()




def _extract_json_and_expl(text: str) -> Tuple[Dict[str, Any], str]:
    """
    Pull the FIRST valid JSON object from a string, then return (json_obj, explanation_text_after_it).
    This avoids PCRE-only constructs like (?R) and works with Python's 're'.
    """
    # strip code fences if present
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    start = s.find("{")
    if start == -1:
        raise ValueError(f"No JSON found in LLM response:\n{s[:400]}")

    # Walk forward and find the matching closing brace using a simple stack
    depth = 0
    end = None
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end is None:
        raise ValueError("Unbalanced JSON braces in LLM response.")

    json_block = s[start:end].strip()
    explanation = s[end:].strip()

    # Be tolerant to single quotes
    try:
        data = json.loads(json_block)
    except json.JSONDecodeError:
        data = json.loads(json_block.replace("'", '"'))

    # Strip "Explanation:" prefix if present
    explanation = re.sub(r"^Explanation:\s*", "", explanation, flags=re.IGNORECASE)
    return data, explanation


async def analyze_chart_with_llm(
    fig: go.Figure,
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    indicators: List[str] | None = None,
    strategy: str = "smc",
) -> TradeDecision:
    """Render chart to PNG, send to Ollama/llava with SMC context, parse decision."""
    indicators = indicators or []
    last_rows = df.tail(50)[["open", "high", "low", "close"]]
    smc_summary = build_feature_snapshot(df)
    smc_text = "\n".join([f"- {k}: {v}" for k, v in smc_summary.items() if v is not None]) or "No strong SMC features detected."

    png_path = _figure_to_png(fig)
    try:
        with open(png_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("ascii")

        prompt = _build_prompt(strategy, symbol, timeframe, smc_text, last_rows)

        # inside analyze_chart_with_llm(...)
        last_rows = df.tail(30)[["open","high","low","close"]]   # was 50 → smaller prompt


        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "num_predict": 96,   # CPU-friendly; can lower to 64 if needed
                "temperature": 0.2,
                "num_ctx": 2048,
                "top_p": 0.9,
            },
        }

        timeout = httpx.Timeout(connect=30.0, read=600.0, write=120.0, pool=None)

        async with httpx.AsyncClient(timeout=timeout) as client:
            last_err = None
            for attempt in range(3):
                try:
                    r = await client.post(f"{OLLAMA_URL}/api/generate",
                                        json=payload,
                                        headers={"Content-Type": "application/json"})
                    r.raise_for_status()
                    content = (r.json() or {}).get("response", "") or ""
                    content = content.strip()
                    break
                except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.HTTPStatusError) as e:
                    last_err = e
                    await asyncio.sleep(8 * (attempt + 1))
            else:
                raise httpx.ReadTimeout(f"Ollama/LLaVA not ready: {last_err}")



        data, explanation = _extract_json_and_expl(content)
        return TradeDecision(
            signal=data.get("signal", "no_trade"),
            sl=data.get("sl"),
            tp=data.get("tp"),
            confidence=data.get("confidence"),
            reasons=[explanation] if explanation else [],
        )
    finally:
        try:
            os.remove(png_path)
        except Exception:
            pass
