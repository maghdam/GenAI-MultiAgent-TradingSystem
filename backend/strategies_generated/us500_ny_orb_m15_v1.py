"""Research-only US500 M15 NY opening-range breakout (not a trusted runtime strategy).

Assumptions: input UTC-indexed, complete M15 bars labeled by bar *open*.
The output is a held target position: decisions at completed-bar close are
executed by Studio's research engine at the *next* observed bar open.
An exit signal on the 15:15 NY candle targets the 15:30 open on regular days.
Early closes and missing exit candles are NOT handled; no broker orders here.
"""

import pandas as pd
import numpy as np


def signals(df: pd.DataFrame) -> pd.Series:
    """Return +1 long, -1 short, 0 flat, using past/current bars only."""
    target = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    if df.empty:
        return target

    ny_index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    current_day = None
    first_high = 0.0
    first_low = 0.0
    second_high = 0.0
    second_low = 0.0
    have_first = False
    have_second = False
    traded = False
    position = 0.0

    for i, stamp in enumerate(ny_index):
        day = stamp.date()
        if day != current_day:
            current_day = day
            first_high = first_low = second_high = second_low = 0.0
            have_first = have_second = traded = False
            position = 0.0

        minute = stamp.hour * 60 + stamp.minute
        if minute == 570:  # 09:30–09:45; bar high/low available at its close
            first_high = float(df["high"].iloc[i])
            first_low = float(df["low"].iloc[i])
            have_first = True
        elif minute == 585:  # 09:45–10:00; bar high/low available at its close
            second_high = float(df["high"].iloc[i])
            second_low = float(df["low"].iloc[i])
            have_second = True
        elif minute >= 915:  # 15:15 bar closes 15:30; next-open exit
            position = 0.0
        elif 600 <= minute < 720 and have_first and have_second and not traded:
            top = max(first_high, second_high)
            bottom = min(first_low, second_low)
            close = float(df["close"].iloc[i])
            if close > top:
                position = 1.0
                traded = True
            elif close < bottom:
                position = -1.0
                traded = True

        target.iloc[i] = position

    return target
