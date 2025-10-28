"""
RSI strategy in generated format.

Exposes `trade_decision(df, symbol=None, timeframe=None)` to align with the
generated strategy loader while keeping prior reasoning/confidence.
"""

from typing import Dict, Any
import pandas as pd


def _compute_rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill")


def trade_decision(
    df: pd.DataFrame,
    *,
    symbol: str | None = None,
    timeframe: str | None = None,
    length: int = 14,
    overbought: float = 70.0,
    oversold: float = 30.0,
) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"signal": "no_trade", "confidence": 0.0, "reasons": ["no data"]}

    closes = df["close"].astype(float)
    rsi_series = _compute_rsi(closes, int(length))
    if rsi_series.empty or pd.isna(rsi_series.iloc[-1]):
        return {"signal": "no_trade", "confidence": 0.0, "reasons": ["insufficient data"]}

    latest_rsi = float(rsi_series.iloc[-1])
    signal = "no_trade"
    reasons = [f"RSI={latest_rsi:.2f}"]

    if latest_rsi >= float(overbought):
        signal = "short"
        reasons.append(f"RSI above overbought ({overbought})")
    elif latest_rsi <= float(oversold):
        signal = "long"
        reasons.append(f"RSI below oversold ({oversold})")
    else:
        reasons.append("RSI within neutral range")

    confidence = max(0.0, min(1.0, abs(latest_rsi - 50.0) / 50.0))
    if signal == "no_trade":
        confidence = 0.0

    return {"signal": signal, "sl": None, "tp": None, "confidence": confidence, "reasons": reasons}

