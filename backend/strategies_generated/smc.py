"""
Generated SMC strategy wrapper that uses the shared LLM analyzer.

Contract: expose `trade_decision(df, symbol=None, timeframe=None)` so the
loader can build a Strategy around it. This keeps strategies consistent with
Strategy Studio's generated files while preserving LLM-based reasoning.
"""

from typing import Optional, Dict, Any
import os
import numpy as np
import pandas as pd

from backend.llm_analyzer import analyze_data_with_llm, TradeDecision


async def trade_decision(
    df: pd.DataFrame,
    *,
    symbol: str | None = None,
    timeframe: str | None = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"signal": "no_trade", "confidence": 0.0, "reasons": ["no data"]}

    td: TradeDecision = await analyze_data_with_llm(
        df=df,
        symbol=symbol or "",
        timeframe=timeframe or "",
        indicators=None,
        model=None,
        options=options or {},
        ollama_url=None,
    )
    # Fill SL/TP if missing (risk management within strategy layer)
    out = td.dict()
    try:
        sig = (out.get("signal") or "").lower()
    except Exception:
        sig = ""
    if sig in ("long", "short"):
        has_sl = out.get("sl") is not None
        has_tp = out.get("tp") is not None
        if not (has_sl and has_tp):
            sl_f, tp_f = compute_sltp(df, side=("BUY" if sig == "long" else "SELL"), symbol=(symbol or "").upper())
            if out.get("sl") is None:
                out["sl"] = sl_f
            if out.get("tp") is None:
                out["tp"] = tp_f
            rs = out.get("reasons") or []
            rs.append("strategy: computed SL/TP")
            out["reasons"] = rs
    return out


def _risk_param(symbol: str, key: str, default_val: str) -> str:
    sym_u = (symbol or "").upper()
    env_key_sym = f"{key}_{sym_u}"
    v = os.getenv(env_key_sym)
    if v is None:
        v = os.getenv(key, default_val)
    return str(v)


def compute_sltp(
    df: pd.DataFrame,
    *,
    side: str,
    symbol: str,
    mode: Optional[str] = None,
    rr: Optional[float] = None,
    atr_len: Optional[int] = None,
    atr_mult: Optional[float] = None,
    swing_lookback: Optional[int] = None,
    tick_pct: Optional[float] = None,
) -> tuple[float | None, float | None]:
    """Compute SL/TP from SMC structure with ATR fallback.

    - mode: 'swing' -> last swing hi/lo +/- tick buffer; else 'atr' fallback
    - rr: risk/reward multiple for TP
    - per-symbol overrides read from env SMC_* and SMC_*_{SYMBOL}
    """
    try:
        side_u = (side or "").upper()
        mode_eff = (mode or _risk_param(symbol, "SMC_RISK_MODE", "atr")).strip().lower()
        rr_eff = float(rr if rr is not None else _risk_param(symbol, "SMC_RR", "2.0"))
    except Exception:
        side_u = (side or "").upper()
        mode_eff = "atr"
        rr_eff = 2.0

    if df is None or df.empty or side_u not in ("BUY", "SELL"):
        return None, None

    # Use the latest close as a proxy for entry at strategy stage
    try:
        entry = float(df["close"].astype(float).iloc[-1])
    except Exception:
        return None, None

    if mode_eff == "swing":
        try:
            lb = int(swing_lookback if swing_lookback is not None else _risk_param(symbol, "SMC_SWING_LOOKBACK", "10"))
            pad = float(tick_pct if tick_pct is not None else _risk_param(symbol, "SMC_TICK_PCT", "0.0005"))
        except Exception:
            lb, pad = 10, 0.0005
        try:
            hi = float(df["high"].astype(float).iloc[-lb:].max())
            lo = float(df["low"].astype(float).iloc[-lb:].min())
        except Exception:
            hi = lo = None
        if hi is None or lo is None:
            mode_eff = "atr"  # fallback
        else:
            if side_u == "BUY":
                sl = lo * (1.0 - pad)
                tp = entry + rr_eff * (entry - sl)
            else:
                sl = hi * (1.0 + pad)
                tp = entry - rr_eff * (sl - entry)
            return float(sl), float(tp)

    # ATR fallback or selected
    try:
        n = int(atr_len if atr_len is not None else _risk_param(symbol, "SMC_ATR_LEN", "14"))
        mult = float(atr_mult if atr_mult is not None else _risk_param(symbol, "SMC_ATR_MULT", "1.0"))
    except Exception:
        n, mult = 14, 1.0
    try:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)
        tr_hl = high - low
        tr_hc = (high - prev_close).abs()
        tr_lc = (low - prev_close).abs()
        tr = np.maximum.reduce([tr_hl.values, tr_hc.values, tr_lc.values])
        atr = float(pd.Series(tr, index=close.index).rolling(n, min_periods=n).mean().iloc[-1])
    except Exception:
        atr = 0.0
    if not atr or atr <= 0:
        return None, None

    if side_u == "BUY":
        sl = entry - mult * atr
        tp = entry + rr_eff * (entry - sl)
    else:
        sl = entry + mult * atr
        tp = entry - rr_eff * (sl - entry)
    return float(sl), float(tp)


def signals(
    df: pd.DataFrame,
    *,
    zone_window: int = 50,
    structure_lookback: int = 5,
    ob_window: int = 20,
    trend_window: int = 20,
    fvg_shift: int = 2,
    ob_distance_pct: float = 0.015,
    threshold: int = 2,
) -> pd.Series:
    """Vectorized SMC-style decision rule per bar.

    Emits +1 for long, -1 for short, 0 for flat based on simple votes:
      - zone: premium/discount (rolling window mid with small buffer)
      - structure: BOS/CHOCH approximations vs rolling HH/LL
      - fvg: simple 3-candle imbalance check
      - ob: proximity to recent swing hi/lo
      - trend: recent bias via higher-highs minus lower-lows

    This matches the analyzer's rule of thumb:
      total >= +2 -> long; total <= -2 -> short; else 0
    """
    if df is None or df.empty:
        return pd.Series([], dtype=float)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # zone vote (+1 discount, -1 premium)
    hi_z = high.rolling(zone_window, min_periods=zone_window).max()
    lo_z = low.rolling(zone_window, min_periods=zone_window).min()
    rng_z = (hi_z - lo_z)
    mid_z = (hi_z + lo_z) / 2.0
    buf_z = 0.01 * rng_z
    zone_vote = np.where(close > (mid_z + buf_z), 1, np.where(close < (mid_z - buf_z), -1, 0))

    # structure vote via BOS/CHOCH approximations
    hh = high.rolling(structure_lookback, min_periods=structure_lookback).max()
    ll = low.rolling(structure_lookback, min_periods=structure_lookback).min()
    prev_hh = hh.shift(1)
    prev_ll = ll.shift(1)
    # BOS
    bos_bull = (hh > prev_hh) & (close > prev_hh)
    bos_bear = (ll < prev_ll) & (close < prev_ll)
    # CHOCH (crossing prior extreme)
    choch_bull = (close > prev_hh) & (close.shift(1) < prev_hh)
    choch_bear = (close < prev_ll) & (close.shift(1) > prev_ll)
    structure_vote = (
        np.where(bos_bull | choch_bull, 1, 0)
        + np.where(bos_bear | choch_bear, -1, 0)
    )
    # clamp to -1..+1
    structure_vote = np.where(structure_vote > 0, 1, np.where(structure_vote < 0, -1, 0))

    # fvg vote: 3-candle gap check vs t-2
    fvg_bull = low > high.shift(fvg_shift)
    fvg_bear = high < low.shift(fvg_shift)
    fvg_vote = np.where(fvg_bull, 1, np.where(fvg_bear, -1, 0))

    # ob vote: proximity to recent swing hi/lo
    hi_ob = high.rolling(ob_window, min_periods=ob_window).max()
    lo_ob = low.rolling(ob_window, min_periods=ob_window).min()
    dist_hi = (close - hi_ob).abs() / close
    dist_lo = (close - lo_ob).abs() / close
    near_hi = dist_hi < ob_distance_pct
    near_lo = dist_lo < ob_distance_pct
    # If near both, pick closer bound
    closer_lo = dist_lo <= dist_hi
    ob_vote = np.where(
        near_hi & near_lo,
        np.where(closer_lo, 1, -1),  # closer to low -> +1, else -1
        np.where(near_lo, 1, np.where(near_hi, -1, 0)),
    )

    # trend vote: (# higher-highs - # lower-lows) over window
    high3 = high.rolling(3, min_periods=3).max()
    low3 = low.rolling(3, min_periods=3).min()
    hh_up = (high3 > high3.shift(1)).astype(int)
    ll_down = (low3 < low3.shift(1)).astype(int)
    hh_cnt = hh_up.rolling(trend_window, min_periods=trend_window).sum()
    ll_cnt = ll_down.rolling(trend_window, min_periods=trend_window).sum()
    trend_score = (hh_cnt - ll_cnt).fillna(0)
    trend_vote = np.where(trend_score > 1, 1, np.where(trend_score < -1, -1, 0))

    # Total and map to signal
    total = (
        pd.Series(zone_vote, index=df.index).fillna(0).astype(int)
        + pd.Series(structure_vote, index=df.index).fillna(0).astype(int)
        + pd.Series(fvg_vote, index=df.index).fillna(0).astype(int)
        + pd.Series(ob_vote, index=df.index).fillna(0).astype(int)
        + pd.Series(trend_vote, index=df.index).fillna(0).astype(int)
    )
    sig = np.where(total >= int(threshold), 1.0, np.where(total <= -int(threshold), -1.0, 0.0))
    return pd.Series(sig, index=df.index).fillna(0.0)
