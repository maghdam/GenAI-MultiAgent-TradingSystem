# smc_features.py

import pandas as pd


def detect_bos_choch(df: pd.DataFrame, lookback=5):
    """Return newest BOS/CHOCH string or None."""
    if len(df) < lookback + 2:
        return None

    hh = df['high'].rolling(lookback).max()
    ll = df['low'].rolling(lookback).min()
    bos = choch = None

    if hh.iloc[-1] > hh.iloc[-2] and df['close'].iloc[-1] > hh.iloc[-2]:
        bos = 'bullish_BOS'
    if ll.iloc[-1] < ll.iloc[-2] and df['close'].iloc[-1] < ll.iloc[-2]:
        bos = 'bearish_BOS'
    if df['close'].iloc[-1] < ll.iloc[-2] and df['close'].iloc[-2] > ll.iloc[-2]:
        choch = 'bearish_CHOCH'
    if df['close'].iloc[-1] > hh.iloc[-2] and df['close'].iloc[-2] < hh.iloc[-2]:
        choch = 'bullish_CHOCH'

    return bos or choch


def in_premium_discount(df: pd.DataFrame):
    """Returns 'premium' / 'discount' / 'equilibrium' based on 50% and buffer."""
    if len(df) < 50:
        return None

    swing_hi = df['high'].iloc[-50:].max()
    swing_lo = df['low'].iloc[-50:].min()
    mid = (swing_hi + swing_lo) / 2
    upper = mid + 0.01 * (swing_hi - swing_lo)
    lower = mid - 0.01 * (swing_hi - swing_lo)
    close = df['close'].iloc[-1]

    if close > upper:
        return 'premium'
    elif close < lower:
        return 'discount'
    else:
        return 'equilibrium'


def current_fvg(df: pd.DataFrame):
    """Detect imbalance in last 3 bars."""
    if len(df) < 3:
        return None

    c0, c1, c2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    if c0['low'] > c2['high']:
        return 'bullish_FVG'
    elif c0['high'] < c2['low']:
        return 'bearish_FVG'
    return None


def ob_near_price(df: pd.DataFrame, distance_pct=0.015):
    """Return 'near_OB' if price is near recent OB zone, else None."""
    if len(df) < 20:
        return None

    hi = df['high'].iloc[-20:].max()
    lo = df['low'].iloc[-20:].min()
    close = df['close'].iloc[-1]
    near_hi = abs(close - hi) / close < distance_pct
    near_lo = abs(close - lo) / close < distance_pct

    return 'near_OB' if near_hi or near_lo else None


def trend_strength(df: pd.DataFrame, window=20):
    """Return trend strength as (# higher highs - lower lows)."""
    if len(df) < window + 3:
        return 0

    highs = df['high'].rolling(3).max().dropna()
    lows = df['low'].rolling(3).min().dropna()
    hh_count = sum(highs.diff() > 0)
    ll_count = sum(lows.diff() < 0)

    return int(hh_count - ll_count)


def build_feature_snapshot(df: pd.DataFrame):
    return {
        'bos_choch': detect_bos_choch(df),
        'zone': in_premium_discount(df),
        'fvg': current_fvg(df),
        'ob': ob_near_price(df),
        'trend_strength': trend_strength(df),
        'close': round(df['close'].iloc[-1], 5)
    }
