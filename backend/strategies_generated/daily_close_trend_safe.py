"""
Daily close trend strategy (backtest-safe).

This variant avoids end-of-day lookahead leakage by:
- computing completed daily closes via resample('1D').last()
- using prior-day direction (D-1 close vs D-2 close) as today's bias
- forward-filling that bias across intraday bars of the day

Exports:
- signals(df) -> position series in {-1, 0, +1}
- trade_decision(df, ...) -> dict compatible with backend.strategy generated loader
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        return df.copy()
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce", utc=True)
    out = out[~out.index.isna()]
    return out


def _daily_bias_and_strength(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    df = _ensure_datetime_index(df)
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    daily_close = close.resample("1D").last().dropna()
    prev = daily_close.shift(1)
    prev2 = daily_close.shift(2)

    has = prev.notna() & prev2.notna() & (prev2 != 0)
    daily_bias = pd.Series(0.0, index=daily_close.index)
    daily_strength = pd.Series(0.0, index=daily_close.index)
    daily_bias.loc[has] = np.where(prev.loc[has] > prev2.loc[has], 1.0, -1.0)
    daily_strength.loc[has] = ((prev - prev2).abs() / prev2.abs()).fillna(0.0).loc[has]
    return daily_bias, daily_strength, prev, prev2


def signals(df: pd.DataFrame) -> pd.Series:
    """Backtest-friendly position series aligned to df.index."""
    if df is None or df.empty:
        return pd.Series([], dtype=float)

    df = _ensure_datetime_index(df)
    daily_bias, _strength, _prev, _prev2 = _daily_bias_and_strength(df)
    pos = daily_bias.reindex(df.index, method="ffill").fillna(0.0)
    return pos.reindex(df.index).fillna(0.0)


def trade_decision(
    df: pd.DataFrame,
    *,
    symbol: str | None = None,
    timeframe: str | None = None,
    **_kwargs: Any,
) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"signal": "no_trade", "confidence": 0.0, "reasons": ["no data"]}

    df = _ensure_datetime_index(df)
    daily_bias, daily_strength, prev, prev2 = _daily_bias_and_strength(df)
    bias = float(daily_bias.reindex(df.index, method="ffill").fillna(0.0).iloc[-1])
    strength = float(daily_strength.reindex(df.index, method="ffill").fillna(0.0).iloc[-1])

    if bias > 0:
        signal = "long"
    elif bias < 0:
        signal = "short"
    else:
        signal = "no_trade"

    reasons: list[str] = []
    sym_u = (symbol or "").upper()
    tf_u = (timeframe or "").upper()
    if sym_u or tf_u:
        reasons.append(f"{sym_u} {tf_u}".strip())

    try:
        # For the current day, the bias is based on prev and prev2 daily closes
        prev_i = float(prev.reindex(df.index, method="ffill").iloc[-1])
        prev2_i = float(prev2.reindex(df.index, method="ffill").iloc[-1])
        if not np.isnan(prev_i) and not np.isnan(prev2_i):
            reasons.append(f"prior daily close {prev_i:.5f} vs {prev2_i:.5f}")
    except Exception:
        pass

    if signal != "no_trade":
        reasons.append(f"daily_bias= {bias:+.0f} strength={strength:.4f}")

    confidence = 0.0 if signal == "no_trade" else float(max(0.2, min(0.9, strength * 5.0)))
    return {"signal": signal, "confidence": confidence, "reasons": reasons}

