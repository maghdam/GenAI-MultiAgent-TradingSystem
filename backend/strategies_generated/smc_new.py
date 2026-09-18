# SMC-style strategy generated from the request below.
# Original request: create an XAUUSD M5 strategy using FVG and market structure
#
# Rules:
# - build directional votes from market structure, FVG, premium/discount, and order-block proximity
# - go long when bullish votes reach the threshold
# - go short when bearish votes reach the threshold
import pandas as pd

def signals(
    df: pd.DataFrame,
    zone_window: int = 50,
    structure_lookback: int = 5,
    ob_window: int = 20,
    threshold: int = 2,
    ob_distance_pct: float = 0.015,
) -> pd.Series:
    if df is None or df.empty:
        return pd.Series([], dtype=float)

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    # Premium/discount zone vote.
    zone_hi = high.rolling(zone_window, min_periods=zone_window).max()
    zone_lo = low.rolling(zone_window, min_periods=zone_window).min()
    zone_mid = (zone_hi + zone_lo) / 2.0
    zone_buf = 0.01 * (zone_hi - zone_lo)
    zone_vote = pd.Series(0, index=df.index, dtype=float)
    zone_vote[close < (zone_mid - zone_buf)] = 1.0
    zone_vote[close > (zone_mid + zone_buf)] = -1.0

    # Market structure vote via BOS / CHOCH-style approximations.
    hh = high.rolling(structure_lookback, min_periods=structure_lookback).max()
    ll = low.rolling(structure_lookback, min_periods=structure_lookback).min()
    prev_hh = hh.shift(1)
    prev_ll = ll.shift(1)
    bull_structure = ((hh > prev_hh) & (close > prev_hh)) | ((close > prev_hh) & (close.shift(1) <= prev_hh))
    bear_structure = ((ll < prev_ll) & (close < prev_ll)) | ((close < prev_ll) & (close.shift(1) >= prev_ll))
    structure_vote = pd.Series(0, index=df.index, dtype=float)
    structure_vote[bull_structure.fillna(False)] = 1.0
    structure_vote[bear_structure.fillna(False)] = -1.0

    # FVG vote using a simple 3-candle imbalance check.
    bull_fvg = low > high.shift(2)
    bear_fvg = high < low.shift(2)
    fvg_vote = pd.Series(0, index=df.index, dtype=float)
    fvg_vote[bull_fvg.fillna(False)] = 1.0
    fvg_vote[bear_fvg.fillna(False)] = -1.0

    # Order-block proximity vote using recent swing bounds.
    ob_hi = high.rolling(ob_window, min_periods=ob_window).max()
    ob_lo = low.rolling(ob_window, min_periods=ob_window).min()
    dist_hi = (close - ob_hi).abs() / close.replace(0, pd.NA)
    dist_lo = (close - ob_lo).abs() / close.replace(0, pd.NA)
    near_hi = dist_hi < ob_distance_pct
    near_lo = dist_lo < ob_distance_pct
    ob_vote = pd.Series(0, index=df.index, dtype=float)
    ob_vote[near_lo.fillna(False)] = 1.0
    ob_vote[near_hi.fillna(False)] = -1.0

    total = zone_vote.fillna(0) + structure_vote.fillna(0) + fvg_vote.fillna(0) + ob_vote.fillna(0)
    out = pd.Series(0.0, index=df.index, dtype=float)
    out[total >= float(threshold)] = 1.0
    out[total <= -float(threshold)] = -1.0
    return out.fillna(0.0)