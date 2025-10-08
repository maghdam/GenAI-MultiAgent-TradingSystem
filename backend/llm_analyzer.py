import requests, json, os, re, base64, tempfile, io
import pandas as pd
import plotly.graph_objects as go
from PIL import Image  # NEW: to recompress PNG -> JPEG for smaller payload
from backend.smc_features import build_feature_snapshot


class TradeDecision:
    def __init__(self, signal: str, sl: float = None, tp: float = None,
                 confidence: float = None, reasons=None):
        self.signal = signal
        self.sl = sl
        self.tp = tp
        self.confidence = confidence
        self.reasons = reasons or []

    def dict(self):
        return {
            "signal": self.signal,
            "sl": self.sl,
            "tp": self.tp,
            "confidence": self.confidence,
            "reasons": self.reasons,
        }


def _figure_to_jpeg_b64(fig: go.Figure, width=336, height=168, quality=60) -> str:
    """
    Convert a Plotly figure to a base64‑encoded JPEG. By default the image
    resolution is scaled down to a width of 336 pixels and a height of 168
    pixels (maintaining a 2:1 aspect ratio) which aligns better with the
    patch size of many vision models like LLaVA. The JPEG quality is set to
    60 by default to further reduce the payload size. Adjust these values
    if you need higher fidelity images, but keep in mind that larger images
    lead to longer inference times.
    """
    fig.update_layout(width=width, height=height)
    tmp_png = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_png = tmp.name
        fig.write_image(tmp_png)  # requires kaleido
        img = Image.open(tmp_png).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buf.getvalue()).decode()
    finally:
        if tmp_png and os.path.exists(tmp_png):
            try:
                os.remove(tmp_png)
            except Exception:
                pass


async def analyze_chart_with_llm(
    fig,
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    indicators=None,
    *,
    model: str = None,
    options: dict = None,
    ollama_url: str = None,
) -> TradeDecision:
    """
    Analyze a chart using a vision LLM. Sends a compressed image together with
    recent OHLC data and extracted SMC features to the Ollama endpoint and
    returns a `TradeDecision`.

    A few improvements over the original implementation:
    - Uses an environment variable `OLLAMA_TIMEOUT` (or a `timeout` option) to
      control the HTTP request timeout. Default is 120 seconds.
    - Catches network timeouts and API errors explicitly, raising user‑friendly
      exceptions when the Ollama API cannot be reached or responds too slowly.
    - Preserves backward compatibility by respecting the existing `options`
      dictionary for temperature/top_p/top_k but removes the `timeout` entry
      from it so it isn't sent to the API directly.
    """
    indicators = indicators or []
    model = (model or os.getenv("OLLAMA_MODEL", "llava:7b")).strip()
    ollama_url = (ollama_url or os.getenv("OLLAMA_URL", "http://ollama:11434")).rstrip("/")
    options = (options or {}).copy()

    # Determine request timeout: `options` can override the environment variable.
    # The value should be an int or float representing seconds.
    # Default to 120 seconds if unspecified.
    timeout_value = options.pop("timeout", None)
    if timeout_value is None:
        try:
            timeout_value = float(os.getenv("OLLAMA_TIMEOUT", 120))
        except Exception:
            timeout_value = 120.0
    try:
        # ensure timeout is numeric and positive
        timeout_value = float(timeout_value)
        if timeout_value <= 0:
            timeout_value = 120.0
    except Exception:
        timeout_value = 120.0

    # Provide a smaller snapshot of the most recent OHLC candles. Reducing
    # the number of rows helps keep the prompt concise and improves
    # performance of the vision‑language model.
    last_rows = df.tail(30)[["open", "high", "low", "close"]]
    smc_summary = build_feature_snapshot(df)
    smc_text = "\n".join([
        f"- {k}: {v}" for k, v in smc_summary.items() if v is not None
    ]) or "No strong SMC features detected."

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
            "images": [img_b64],
            "stream": False,
            "options": {
                # Set sensible defaults; UI/env may override via `options`.
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40,
                **options,
            },
        }
        try:
            r = requests.post(
                f"{ollama_url}/api/generate",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=timeout_value,
            )
        except requests.exceptions.Timeout:
            raise ValueError(
                f"LLM request timed out after {timeout_value} seconds."
            )
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to contact Ollama API: {e}")

        if r.status_code != 200:
            raise RuntimeError(f"Ollama API error: {r.status_code} — {r.text}")

        content = r.json().get("response", "").strip()
        if not content:
            raise ValueError("Empty response from LLM.")

        # Extract JSON object from response.
        m = re.search(r"\{.*?\}", content, re.DOTALL)
        if not m:
            raise ValueError(
                f"No JSON object found in LLM response.\nRaw:\n{content}"
            )
        json_block = m.group(0).strip()
        explanation = content[m.end():].strip()
        explanation = re.sub(
            r"^Explanation:\s*", "", explanation, flags=re.IGNORECASE
        ).strip()

        parsed = json.loads(json_block)
        return TradeDecision(
            **parsed, reasons=[explanation] if explanation else []
        )

    except Exception as e:
        # Wrap all exceptions in a ValueError with a clear message. This makes
        # it easier for calling code (e.g., FastAPI handlers) to propagate
        # meaningful error messages up to the user.
        raise ValueError(f"❌ Failed to parse LLM response: {e}")