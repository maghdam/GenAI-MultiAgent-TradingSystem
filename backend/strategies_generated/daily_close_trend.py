# Daily close trend-following strategy.
# Long if today's daily close is above yesterday's; otherwise short.
import pandas as pd


def daily_candle_cross(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample intraday bars to daily closes and go with the daily trend.
    Position: 1 when today's daily close > yesterday's, else -1.
    """
    if df is None or df.empty:
        raise ValueError("Historical data is required.")

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
    else:
        df = df.copy()

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])

    daily = df["close"].resample("1D").last().dropna()
    prev = daily.shift(1)

    df["daily_close"] = daily.reindex(df.index, method="ffill")
    df["daily_prev"] = prev.reindex(df.index, method="ffill")

    has_prev = df["daily_prev"].notna()
    df["long_signal"] = ((df["daily_close"] > df["daily_prev"]) & has_prev).astype(int)
    df["short_signal"] = ((df["daily_close"] <= df["daily_prev"]) & has_prev).astype(int)

    # Position: 1 = long, -1 = short, 0 if no prior day yet
    df["position"] = df["long_signal"] - df["short_signal"]

    df["returns"] = df["close"].pct_change()
    df["strategy_returns"] = df["returns"] * df["position"].shift(1).fillna(0)
    return df


def signals(df: pd.DataFrame) -> pd.Series:
    """Backtest-friendly signal series aligned to input index."""
    enriched = daily_candle_cross(df)
    return enriched["position"].reindex(df.index).fillna(0)


def trade_decision(df: pd.DataFrame, symbol=None, timeframe=None):
    """
    Simple trade decision helper for live use.
    Returns a dict compatible with backend.strategy expectations.
    """
    enriched = daily_candle_cross(df)
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
    reasons = []
    if pd.notna(dc) and pd.notna(dp):
        reasons.append(f"Daily close {dc:.5f} vs prev {dp:.5f}")

    confidence = None
    if pd.notna(dc) and pd.notna(dp) and dp not in (0, 0.0):
        confidence = min(0.9, max(0.2, abs((dc - dp) / dp)))

    return {"signal": signal, "confidence": confidence, "reasons": reasons}
