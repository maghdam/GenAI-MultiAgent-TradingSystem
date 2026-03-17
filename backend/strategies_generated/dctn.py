import pandas as pd


def daily_candle_trend_filtered(df: pd.DataFrame, min_change: float = 0.002) -> pd.DataFrame:
    """
    Daily close trend-following with a strength/volatility filter.

    Core idea:
    - Resample intraday bars to daily closes.
    - Compute today's vs yesterday's daily close.
    - If |daily_change_pct| < min_change -> stay flat (0).
    - Else:
        * if today's daily close > yesterday's -> long (+1)
        * else -> short (-1)
    """
    if df is None or df.empty:
        raise ValueError("Historical data is required.")

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
    else:
        df = df.copy()

    # Clean close
    df["close"] = pd.to_numeric(df.get("close"), errors="coerce")
    df = df.dropna(subset=["close"])

    # Daily closes and previous
    daily = df["close"].resample("1D").last().dropna()
    prev = daily.shift(1)

    # Daily % change
    daily_change_pct = (daily - prev) / prev

    # Map back to intraday index
    df["daily_close"]      = daily.reindex(df.index, method="ffill")
    df["daily_prev"]       = prev.reindex(df.index, method="ffill")
    df["daily_change_pct"] = daily_change_pct.reindex(df.index, method="ffill")

    has_prev = df["daily_prev"].notna()
    strong_move = has_prev & df["daily_change_pct"].abs().ge(min_change)

    # Only take signals when move is strong enough
    df["long_signal"] = ((df["daily_close"] > df["daily_prev"]) & strong_move).astype(int)
    df["short_signal"] = ((df["daily_close"] <= df["daily_prev"]) & strong_move).astype(int)

    # Position: +1 / -1 / 0 (flat when move is too small or first day)
    df["position"] = df["long_signal"] - df["short_signal"]

    # Returns
    df["returns"] = df["close"].pct_change()
    df["strategy_returns"] = df["returns"] * df["position"].shift(1).fillna(0)

    return df


# 🔹 THIS is what your engine expects:
def signals(df: pd.DataFrame) -> pd.Series:
    """
    Backtest entry point. Must be callable as signals(df) by the engine.
    Adjust min_change here to tune the filter.
    """
    enriched = daily_candle_trend_filtered(df, min_change=0.002)  # e.g. 0.2%
    return enriched["position"].reindex(df.index).fillna(0)


def trade_decision(df: pd.DataFrame, symbol=None, timeframe=None, min_change: float = 0.002):
    """
    Live trade decision helper for the filtered daily trend strategy.
    """
    enriched = daily_candle_trend_filtered(df, min_change=min_change)
    last = enriched.iloc[-1]

    pos = float(last.get("position", 0) or 0)
    if pos > 0:
        signal = "long"
    elif pos < 0:
        signal = "short"
    else:
        signal = "no_trade"

    dc = last.get("daily_close")
    dp = last.get("daily_prev")
    chg = last.get("daily_change_pct")

    reasons = []
    if pd.notna(dc) and pd.notna(dp):
        reasons.append(f"Daily close {dc:.5f} vs prev {dp:.5f}")
    if pd.notna(chg):
        reasons.append(f"Daily change {chg * 100:.3f}% (threshold {min_change * 100:.2f}%)")

    # Confidence scaled by how much we exceed the threshold
    confidence = None
    if pd.notna(chg):
        ratio = abs(chg) / max(min_change, 1e-8)
        confidence = max(0.2, min(0.9, 0.2 + 0.15 * ratio))

    return {
        "signal": signal,
        "confidence": confidence,
        "reasons": reasons,
        "symbol": symbol,
        "timeframe": timeframe,
        "params": {"min_change": min_change},
    }
