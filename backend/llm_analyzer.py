import requests, json, os, re, base64, tempfile, io
import pandas as pd
import plotly.graph_objects as go
from PIL import Image  # NEW: to recompress PNG -> JPEG for smaller payload
from backend.smc_features import build_feature_snapshot

class TradeDecision:
    def __init__(self, signal: str, sl: float=None, tp: float=None,
                 confidence: float=None, reasons=None):
        self.signal = signal
        self.sl = sl
        self.tp = tp
        self.confidence = confidence
        self.reasons = reasons or []
    def dict(self):
        return {"signal": self.signal, "sl": self.sl, "tp": self.tp,
                "confidence": self.confidence, "reasons": self.reasons}

def _figure_to_jpeg_b64(fig: go.Figure, width=720, height=380, quality=70) -> str:
    """
    Keep using the chart image, but send a compact JPEG to Ollama for speed.
    """
    fig.update_layout(width=width, height=height)
    tmp_png = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_png = tmp.name
        fig.write_image(tmp_png)  # requires kaleido (already used in your project)
        img = Image.open(tmp_png).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buf.getvalue()).decode()
    finally:
        if tmp_png and os.path.exists(tmp_png):
            try: os.remove(tmp_png)
            except: pass

async def analyze_chart_with_llm(
    fig,
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    indicators=[],
    *,
    model: str=None,
    options: dict=None,
    ollama_url: str=None,
):
    """
    Always uses the chart image (no removals), but smaller and faster.
    """
    model = (model or os.getenv("OLLAMA_MODEL", "llava:7b")).strip()
    ollama_url = (ollama_url or os.getenv("OLLAMA_URL", "http://ollama:11434")).rstrip("/")
    options = (options or {}).copy()

    last_rows = df.tail(50)[["open","high","low","close"]]
    smc_summary = build_feature_snapshot(df)
    smc_text = "\n".join([f"- {k}: {v}" for k,v in smc_summary.items() if v is not None]) or "No strong SMC features detected."

    # 🖼️ keep the image, just slimmer bytes
    if not isinstance(fig, go.Figure):
        raise ValueError("Figure is required for vision models; got None.")
    img_b64 = _figure_to_jpeg_b64(fig, width=720, height=380, quality=70)

    prompt = f"""
You are a professional Smart Money Concepts (SMC) trading analyst.

Analyze the market for {symbol} ({timeframe}) using:
- the attached chart image
- OHLC data
- extracted SMC features

SMC Summary:
{smc_text}

OHLC (last 50 candles):
{last_rows.to_string(index=False)}

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

Do NOT include markdown fences or extra prose beyond the JSON and a single 'Explanation:' line.
""".strip()

    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [img_b64],   # ✅ image stays
            "stream": False,
            "options": {
                # sensible fast-ish defaults; UI/env can override
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40,
                **options,
            },
        }
        r = requests.post(f"{ollama_url}/api/generate",
                          headers={"Content-Type":"application/json"},
                          json=payload, timeout=90)
        if r.status_code != 200:
            raise RuntimeError(f"Ollama API error: {r.status_code} — {r.text}")

        content = r.json().get("response","").strip()
        if not content:
            raise ValueError("Empty response from LLM.")

        m = re.search(r"\{.*?\}", content, re.DOTALL)
        if not m:
            raise ValueError(f"No JSON object found in LLM response.\nRaw:\n{content}")
        json_block = m.group(0).strip()

        explanation = content[m.end():].strip()
        explanation = re.sub(r"^Explanation:\s*", "", explanation, flags=re.IGNORECASE).strip()

        parsed = json.loads(json_block)
        return TradeDecision(**parsed, reasons=[explanation] if explanation else [])

    except Exception as e:
        raise ValueError(f"❌ Failed to parse LLM response: {e}")
