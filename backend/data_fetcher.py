# backend/data_fetcher.py
from __future__ import annotations

from typing import Tuple, Optional
import logging
import pandas as pd
import backend.ctrader_client as ctd

logger = logging.getLogger("backend.data_fetcher")

ALLOWED_TF = {"M1", "M5", "M15", "M30", "H1", "H4", "D1"}

def fetch_data(symbol: str, timeframe: str, num_bars: int = 5000) -> Tuple[pd.DataFrame, Optional[float]]:
    """
    Fetch OHLC from cTrader and return:
      - df: pandas DataFrame indexed by UTC timestamps (columns: open, high, low, close, volume)
      - live_price: latest close (float) or None if no data
    On any error: returns (empty DataFrame, None).
    """
    sym = (symbol or "").strip()
    try:
        if not sym:
            raise ValueError("Symbol is required")

        if not ctd.is_connected():
            raise RuntimeError("cTrader feed not connected")

        tf = (timeframe or "M5").upper()
        if tf not in ALLOWED_TF:
            raise ValueError(f"Invalid timeframe '{tf}'. Allowed: {sorted(ALLOWED_TF)}")

        rows = ctd.get_ohlc_data(symbol=sym, tf=tf, n=num_bars)
        df = pd.DataFrame(rows)
        if df.empty:
            return df, None

        # Normalize schema
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"]).set_index("time").sort_index()

        # Ensure numeric dtypes
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows with missing O/H/L/C after coercion
        df = df.dropna(subset=["open", "high", "low", "close"])

        # Keep only the last num_bars (in case we got more after cleaning)
        if num_bars and len(df) > num_bars:
            df = df.iloc[-num_bars:]

        live_price = float(df["close"].iloc[-1]) if not df.empty else None
        return df, live_price

    except Exception as e:
        s = sym or "<EMPTY>"
        logger.warning("❌ Error fetching data for %s: %s", s, e)
        # Fall back to your original behavior:
        return pd.DataFrame(), None
