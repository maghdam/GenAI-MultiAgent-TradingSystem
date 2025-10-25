# Example: SMA crossover strategy skeleton
import pandas as pd

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()

def signals(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
    f = sma(df['close'], fast)
    s = sma(df['close'], slow)
    sig = (f > s).astype(int).diff().fillna(0)
    # 1 = long entry, -1 = exit/short entry depending on rules
    return sig