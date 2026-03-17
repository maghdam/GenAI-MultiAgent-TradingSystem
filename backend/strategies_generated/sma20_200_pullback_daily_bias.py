"""
Hybrid strategy: 20/200 SMA pullback entries + Daily Close Trend bias filter.

Idea:
- Use the SMA20/200 pullback system for tactical entries/exits on the current timeframe.
- Use a daily trend proxy as a higher-timeframe bias filter to avoid counter-bias setups.

Backtest note:
This implementation computes the daily bias in a backtest-safe way (by shifting
daily closes before forward-filling), avoiding end-of-day lookahead leakage.

Exports:
- signals(df, ...) -> pandas Series of position values in {-1, 0, +1} for backtesting.
- trade_decision(df, ...) -> dict compatible with backend.strategy generated loader.
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


def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(int(n), min_periods=int(n)).mean()


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(int(n), min_periods=int(n)).mean()


def _pct_slope(series: pd.Series, lookback: int) -> pd.Series:
    lb = int(max(1, lookback))
    prev = series.shift(lb)
    denom = prev.replace(0, np.nan)
    return (series - prev) / denom


def _candle_parts(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    body = (close - open_).abs()
    rng = (high - low).replace(0, np.nan)
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low
    return open_, high, low, close, body, rng, upper_wick, lower_wick


def _rejection_long(
    *,
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    sma20: pd.Series,
    tol: pd.Series,
    wick_body_ratio: float,
    close_pos_min: float,
) -> pd.Series:
    bullish = close > open_
    touch = low <= (sma20 + tol)
    close_back = close >= sma20
    body = (close - open_).abs()
    rng = (high - low).replace(0, np.nan)
    lower_wick = np.minimum(open_, close) - low
    wick_ok = lower_wick >= (float(wick_body_ratio) * body.replace(0, np.nan)).fillna(0.0)
    close_pos = ((close - low) / rng).fillna(0.0)
    close_ok = close_pos >= float(close_pos_min)
    return bullish & touch & close_back & wick_ok & close_ok


def _rejection_short(
    *,
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    sma20: pd.Series,
    tol: pd.Series,
    wick_body_ratio: float,
    close_pos_max: float,
) -> pd.Series:
    bearish = close < open_
    touch = high >= (sma20 - tol)
    close_back = close <= sma20
    body = (close - open_).abs()
    rng = (high - low).replace(0, np.nan)
    upper_wick = high - np.maximum(open_, close)
    wick_ok = upper_wick >= (float(wick_body_ratio) * body.replace(0, np.nan)).fillna(0.0)
    close_pos = ((close - low) / rng).fillna(1.0)
    close_ok = close_pos <= float(close_pos_max)
    return bearish & touch & close_back & wick_ok & close_ok


def _daily_bias(
    df: pd.DataFrame,
    *,
    mode: str = "prior_day_trend",
) -> Tuple[pd.Series, pd.Series]:
    """Return (bias, strength) series aligned to df.index.

    bias:
      +1 bullish, -1 bearish, 0 unknown.

    modes:
      - prior_day_trend: sign(prev_day_close - prev2_day_close), held constant for the day
      - prev_close_cross: sign(current_close - prev_day_close), can flip intraday
      - off: always 0
    """
    if df is None or df.empty:
        empty = pd.Series([], dtype=float)
        return empty, empty

    df = _ensure_datetime_index(df)
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    daily_close = close.resample("1D").last().dropna()
    prev = daily_close.shift(1)
    prev2 = daily_close.shift(2)

    mode_l = (mode or "prior_day_trend").strip().lower()
    if mode_l == "off":
        bias = pd.Series(0.0, index=df.index)
        strength = pd.Series(0.0, index=df.index)
        return bias, strength

    if mode_l == "prev_close_cross":
        prev_intraday = prev.reindex(df.index, method="ffill")
        has_prev = prev_intraday.notna()
        bias = pd.Series(0.0, index=df.index)
        bias.loc[has_prev] = np.where(close.loc[has_prev] > prev_intraday.loc[has_prev], 1.0, -1.0)
        strength = pd.Series(0.0, index=df.index)
        denom = prev_intraday.replace(0, np.nan).abs()
        strength.loc[has_prev] = ((close - prev_intraday).abs() / denom).fillna(0.0).loc[has_prev]
        return bias, strength

    if mode_l == "prior_day_trend":
        has = prev.notna() & prev2.notna() & (prev2 != 0)
        daily_bias = pd.Series(0.0, index=daily_close.index)
        daily_strength = pd.Series(0.0, index=daily_close.index)
        daily_bias.loc[has] = np.where(prev.loc[has] > prev2.loc[has], 1.0, -1.0)
        daily_strength.loc[has] = ((prev - prev2).abs() / prev2.abs()).fillna(0.0).loc[has]
        bias = daily_bias.reindex(df.index, method="ffill").fillna(0.0)
        strength = daily_strength.reindex(df.index, method="ffill").fillna(0.0)
        return bias, strength

    raise ValueError(f"Unknown daily bias mode: {mode}")


def _compute_setups(
    df: pd.DataFrame,
    *,
    fast: int,
    slow: int,
    atr_len: int,
    slope_lookback: int,
    slope_threshold: float,
    touch_atr_mult: float,
    touch_pct_fallback: float,
    extension_atr_mult: float,
    extension_pct_fallback: float,
    max_pullback_atr_mult: float,
    min_200_atr_mult: float,
    wick_body_ratio: float,
    close_pos_min: float,
    squeeze_flat_threshold: float,
    squeeze_proximity_pct: float,
    squeeze_lookback: int,
    squeeze_between_ratio: float,
) -> Dict[str, pd.Series]:
    if df is None or df.empty:
        return {
            "long_setup": pd.Series([], dtype=bool),
            "short_setup": pd.Series([], dtype=bool),
        }

    df = _ensure_datetime_index(df)
    open_, high, low, close, body, rng, upper_wick, lower_wick = _candle_parts(df)

    sma20 = _sma(close, fast)
    sma200 = _sma(close, slow)
    atr = _atr(df, atr_len)

    slope20 = _pct_slope(sma20, slope_lookback)
    slope200 = _pct_slope(sma200, slope_lookback)

    uptrend = slope20 > float(slope_threshold)
    downtrend = slope20 < -float(slope_threshold)

    above200 = close > sma200
    below200 = close < sma200

    atr_eff_touch = atr.fillna(close.abs() * float(touch_pct_fallback))
    tol = (float(touch_atr_mult) * atr_eff_touch).abs()

    atr_eff_ext = atr.fillna(close.abs() * float(extension_pct_fallback))
    dist20 = (close - sma20).abs()
    not_extended = dist20 <= (float(extension_atr_mult) * atr_eff_ext).abs()

    controlled_pullback = atr.isna() | (rng <= (float(max_pullback_atr_mult) * atr))

    dist200 = (close - sma200).abs()
    away_from_200 = atr.isna() | (dist200 >= (float(min_200_atr_mult) * atr))

    bull_reject = _rejection_long(
        open_=open_,
        high=high,
        low=low,
        close=close,
        sma20=sma20,
        tol=tol,
        wick_body_ratio=wick_body_ratio,
        close_pos_min=close_pos_min,
    )
    bear_reject = _rejection_short(
        open_=open_,
        high=high,
        low=low,
        close=close,
        sma20=sma20,
        tol=tol,
        wick_body_ratio=wick_body_ratio,
        close_pos_max=(1.0 - float(close_pos_min)),
    )

    flat200 = slope200.abs() <= float(squeeze_flat_threshold)
    prox = ((sma20 - sma200).abs() / close.replace(0, np.nan)).fillna(np.inf) <= float(squeeze_proximity_pct)
    between = ((close - sma20) * (close - sma200) <= 0).fillna(False)
    between_score = between.rolling(int(squeeze_lookback), min_periods=int(squeeze_lookback)).mean()
    compressing = between_score >= float(squeeze_between_ratio)

    upper_ma = pd.concat([sma20, sma200], axis=1).max(axis=1)
    lower_ma = pd.concat([sma20, sma200], axis=1).min(axis=1)

    breakout_up = (close > upper_ma) & (close.shift(1) <= upper_ma.shift(1))
    breakout_dn = (close < lower_ma) & (close.shift(1) >= lower_ma.shift(1))
    squeeze_long = flat200 & prox & compressing & breakout_up & above200
    squeeze_short = flat200 & prox & compressing & breakout_dn & below200

    long_setup = (above200 & uptrend & bull_reject & not_extended & controlled_pullback & away_from_200) | squeeze_long
    short_setup = (below200 & downtrend & bear_reject & not_extended & controlled_pullback & away_from_200) | squeeze_short

    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "sma20": sma20,
        "sma200": sma200,
        "atr": atr,
        "slope20": slope20,
        "slope200": slope200,
        "long_setup": long_setup.fillna(False),
        "short_setup": short_setup.fillna(False),
        "squeeze_long": squeeze_long.fillna(False),
        "squeeze_short": squeeze_short.fillna(False),
    }


def signals(
    df: pd.DataFrame,
    *,
    daily_bias_mode: str = "prior_day_trend",
    require_daily_bias: bool = True,
    fast: int = 20,
    slow: int = 200,
    atr_len: int = 14,
    slope_lookback: int = 5,
    slope_threshold: float = 0.00005,
    touch_atr_mult: float = 0.25,
    touch_pct_fallback: float = 0.0005,
    extension_atr_mult: float = 1.5,
    extension_pct_fallback: float = 0.0020,
    max_pullback_atr_mult: float = 2.5,
    min_200_atr_mult: float = 0.5,
    wick_body_ratio: float = 1.5,
    close_pos_min: float = 0.60,
    squeeze_flat_threshold: float = 0.00002,
    squeeze_proximity_pct: float = 0.0025,
    squeeze_lookback: int = 8,
    squeeze_between_ratio: float = 0.6,
    exit_on_sma20_break: bool = True,
    exit_on_sma200_break: bool = True,
    exit_on_trend_flip: bool = True,
) -> pd.Series:
    """Backtest-friendly position series in {-1,0,+1}."""
    if df is None or df.empty:
        return pd.Series([], dtype=float)

    df = _ensure_datetime_index(df)
    bias, _strength = _daily_bias(df, mode=daily_bias_mode)
    if bias.empty:
        bias = pd.Series(0.0, index=df.index)

    bias_long_ok = bias > 0
    bias_short_ok = bias < 0
    if not bool(require_daily_bias):
        bias_long_ok = bias_long_ok | (bias == 0)
        bias_short_ok = bias_short_ok | (bias == 0)

    out = _compute_setups(
        df,
        fast=fast,
        slow=slow,
        atr_len=atr_len,
        slope_lookback=slope_lookback,
        slope_threshold=slope_threshold,
        touch_atr_mult=touch_atr_mult,
        touch_pct_fallback=touch_pct_fallback,
        extension_atr_mult=extension_atr_mult,
        extension_pct_fallback=extension_pct_fallback,
        max_pullback_atr_mult=max_pullback_atr_mult,
        min_200_atr_mult=min_200_atr_mult,
        wick_body_ratio=wick_body_ratio,
        close_pos_min=close_pos_min,
        squeeze_flat_threshold=squeeze_flat_threshold,
        squeeze_proximity_pct=squeeze_proximity_pct,
        squeeze_lookback=squeeze_lookback,
        squeeze_between_ratio=squeeze_between_ratio,
    )

    close = out["close"]
    sma20 = out["sma20"]
    sma200 = out["sma200"]
    slope20 = out["slope20"]
    long_setup = out["long_setup"].astype(bool) & bias_long_ok
    short_setup = out["short_setup"].astype(bool) & bias_short_ok

    usable = sma20.notna() & sma200.notna()

    long_exit = pd.Series(False, index=df.index)
    short_exit = pd.Series(False, index=df.index)
    if exit_on_sma20_break:
        long_exit = long_exit | (close < sma20)
        short_exit = short_exit | (close > sma20)
    if exit_on_sma200_break:
        long_exit = long_exit | (close < sma200)
        short_exit = short_exit | (close > sma200)
    if exit_on_trend_flip:
        long_exit = long_exit | (slope20 < 0)
        short_exit = short_exit | (slope20 > 0)

    pos_vals = np.zeros(len(df), dtype=float)
    pos = 0.0
    for i in range(len(df)):
        if not bool(usable.iloc[i]):
            pos = 0.0
            pos_vals[i] = 0.0
            continue

        if pos == 0.0:
            if bool(long_setup.iloc[i]):
                pos = 1.0
            elif bool(short_setup.iloc[i]):
                pos = -1.0
        elif pos > 0.0:
            if bool(long_exit.iloc[i]):
                pos = 0.0
            elif bool(short_setup.iloc[i]):
                pos = -1.0
        else:
            if bool(short_exit.iloc[i]):
                pos = 0.0
            elif bool(long_setup.iloc[i]):
                pos = 1.0

        pos_vals[i] = pos

    return pd.Series(pos_vals, index=df.index).reindex(df.index).fillna(0.0)


def trade_decision(
    df: pd.DataFrame,
    *,
    symbol: str | None = None,
    timeframe: str | None = None,
    daily_bias_mode: str = "prior_day_trend",
    require_daily_bias: bool = True,
    rr: float = 2.0,
    sl_atr_buffer_mult: float = 0.20,
    pullback_lookback: int = 5,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Live decision helper (entry-only) for the hybrid system."""
    if df is None or df.empty:
        return {"signal": "no_trade", "confidence": 0.0, "reasons": ["no data"]}

    df = _ensure_datetime_index(df)
    bias, strength = _daily_bias(df, mode=daily_bias_mode)
    if bias.empty:
        bias = pd.Series(0.0, index=df.index)
        strength = pd.Series(0.0, index=df.index)

    bias_last = float(bias.iloc[-1]) if len(bias) else 0.0
    strength_last = float(strength.iloc[-1]) if len(strength) else 0.0

    params = dict(kwargs)
    setups = _compute_setups(
        df,
        fast=int(params.pop("fast", 20)),
        slow=int(params.pop("slow", 200)),
        atr_len=int(params.pop("atr_len", 14)),
        slope_lookback=int(params.pop("slope_lookback", 5)),
        slope_threshold=float(params.pop("slope_threshold", 0.00005)),
        touch_atr_mult=float(params.pop("touch_atr_mult", 0.25)),
        touch_pct_fallback=float(params.pop("touch_pct_fallback", 0.0005)),
        extension_atr_mult=float(params.pop("extension_atr_mult", 1.5)),
        extension_pct_fallback=float(params.pop("extension_pct_fallback", 0.0020)),
        max_pullback_atr_mult=float(params.pop("max_pullback_atr_mult", 2.5)),
        min_200_atr_mult=float(params.pop("min_200_atr_mult", 0.5)),
        wick_body_ratio=float(params.pop("wick_body_ratio", 1.5)),
        close_pos_min=float(params.pop("close_pos_min", 0.60)),
        squeeze_flat_threshold=float(params.pop("squeeze_flat_threshold", 0.00002)),
        squeeze_proximity_pct=float(params.pop("squeeze_proximity_pct", 0.0025)),
        squeeze_lookback=int(params.pop("squeeze_lookback", 8)),
        squeeze_between_ratio=float(params.pop("squeeze_between_ratio", 0.6)),
    )
    if not setups or setups["close"].empty:
        return {"signal": "no_trade", "confidence": 0.0, "reasons": ["insufficient data"]}

    long_setup = setups["long_setup"].astype(bool)
    short_setup = setups["short_setup"].astype(bool)
    long_event = long_setup & (~long_setup.shift(1).fillna(False))
    short_event = short_setup & (~short_setup.shift(1).fillna(False))

    is_long = bool(long_event.iloc[-1])
    is_short = bool(short_event.iloc[-1])
    if not (is_long or is_short):
        return {"signal": "no_trade", "confidence": 0.0, "reasons": ["no setup"]}

    allow_neutral = not bool(require_daily_bias)
    bias_allows_long = (bias_last > 0) or (allow_neutral and bias_last == 0)
    bias_allows_short = (bias_last < 0) or (allow_neutral and bias_last == 0)

    if is_long and not bias_allows_long:
        return {"signal": "no_trade", "confidence": 0.0, "reasons": [f"blocked by daily bias ({daily_bias_mode}): {bias_last:+.0f}"]}
    if is_short and not bias_allows_short:
        return {"signal": "no_trade", "confidence": 0.0, "reasons": [f"blocked by daily bias ({daily_bias_mode}): {bias_last:+.0f}"]}

    close = setups["close"]
    high = setups["high"]
    low = setups["low"]
    sma20 = setups["sma20"]
    sma200 = setups["sma200"]
    atr = setups["atr"]
    slope20 = setups["slope20"]

    entry = float(close.iloc[-1])
    atr_last = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else None
    buf = (float(sl_atr_buffer_mult) * atr_last) if atr_last is not None else abs(entry) * 0.0005

    lb = int(max(2, pullback_lookback))
    swing_low = float(low.iloc[-lb:].min())
    swing_high = float(high.iloc[-lb:].max())

    rr_f = float(rr)
    reasons: list[str] = []
    sym_u = (symbol or "").upper()
    tf_u = (timeframe or "").upper()
    if sym_u or tf_u:
        reasons.append(f"{sym_u} {tf_u}".strip())
    reasons.append(f"daily_bias({daily_bias_mode})={bias_last:+.0f} strength={strength_last:.4f}")

    try:
        s20 = float(sma20.iloc[-1])
        s200 = float(sma200.iloc[-1])
        reasons.append(f"SMA20={s20:.6f} SMA200={s200:.6f}")
    except Exception:
        pass

    try:
        slp = float(slope20.iloc[-1])
        reasons.append(f"SMA20_slope={slp:+.5%}")
    except Exception:
        pass

    if atr_last is not None:
        reasons.append(f"ATR{int(kwargs.get('atr_len', 14))}={atr_last:.6f}")

    if is_long:
        signal = "long"
        sl = float(swing_low - buf)
        if sl >= entry:
            sl = float(entry - max(buf, abs(entry) * 0.0005))
        tp = float(entry + rr_f * (entry - sl))
        reasons.append("setup: SMA20 pullback + bullish rejection (bias-aligned)")
        if bool(setups.get("squeeze_long", pd.Series([False])).iloc[-1]):
            reasons.append("extra: 20/200 squeeze breakout (up)")
    else:
        signal = "short"
        sl = float(swing_high + buf)
        if sl <= entry:
            sl = float(entry + max(buf, abs(entry) * 0.0005))
        tp = float(entry - rr_f * (sl - entry))
        reasons.append("setup: SMA20 retrace + bearish rejection (bias-aligned)")
        if bool(setups.get("squeeze_short", pd.Series([False])).iloc[-1]):
            reasons.append("extra: 20/200 squeeze breakout (down)")

    conf = 0.65
    try:
        slp_abs = abs(float(slope20.iloc[-1]))
        conf += min(0.15, slp_abs / max(1e-9, float(kwargs.get("slope_threshold", 0.00005))) / 25.0)
    except Exception:
        pass
    conf += min(0.15, float(strength_last) * 5.0)
    conf = float(max(0.0, min(0.95, conf)))

    return {"signal": signal, "sl": sl, "tp": tp, "confidence": conf, "reasons": reasons}

