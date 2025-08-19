import requests
import json
import pandas as pd
import plotly.graph_objects as go
import tempfile, os, base64, re
from backend.smc_features import build_feature_snapshot

class TradeDecision:
    def __init__(self, signal: str, sl: float = None, tp: float = None, confidence: float = None, reasons=None):
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
            "reasons": self.reasons
        }

async def analyze_chart_with_llm(fig, df: pd.DataFrame, symbol: str, timeframe: str, indicators=[]):
    last_rows = df.tail(50)[['open', 'high', 'low', 'close']]
    smc_summary = build_feature_snapshot(df)
    smc_text = "\n".join([f"- {k}: {v}" for k, v in smc_summary.items() if v is not None]) or "No strong SMC features detected."

    fig.update_layout(width=800, height=400)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig.write_image(tmp.name)
        chart_path = tmp.name

    with open(chart_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

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

Make a trading decision using SMC methodology:
- CHoCH / BOS
- FVGs
- OBs
- Premium/Discount zones
- Liquidity sweeps
- Market structure
- Trend strength

Respond in this format **exactly**:

{{
  "signal": "long" | "short" | "no_trade",
  "sl": float,
  "tp": float,
  "confidence": float
}}

Then on the next line, write a plain English explanation outside the JSON, like this:

Explanation:
Your explanation goes here.

⚠ Do NOT include triple backticks or any markdown. Respond only with the JSON and plain explanation below it.
""".strip()

    try:
        response = requests.post(
            "http://host.docker.internal:11434/api/generate",
            headers={"Content-Type": "application/json"},
            json={
                "model": "llava",
                "prompt": prompt,
                "images": [img_b64],
                "stream": False
            }
        )

        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error: {response.status_code}, {response.text}")

        content = response.json().get("response", "").strip()

        # Extract JSON part
        match = re.search(r'\{.*?\}', content, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in LLM response.")
        json_block = match.group(0).strip()

        # Extract explanation
        explanation = content[match.end():].strip()
        explanation = re.sub(r"^Explanation:\s*", "", explanation, flags=re.IGNORECASE).strip()

        parsed = json.loads(json_block)
        return TradeDecision(**parsed, reasons=[explanation])

    except Exception as e:
        raise ValueError(f"❌ Failed to parse LLM response: {e}\nRaw content:\n{content}")

    finally:
        if os.path.exists(chart_path):
            os.remove(chart_path)
