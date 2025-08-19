# data_fetcher.py

from backend.ctrader_client import get_ohlc_data  # Replace with your actual fetch function
import pandas as pd


def fetch_data(symbol: str, timeframe: str, num_bars: int = 5000):
    try:
        df = get_ohlc_data(symbol=symbol, tf=timeframe, n=num_bars)
        df = pd.DataFrame(df)
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)

        # Extract latest price from last close
        live_price = df["close"].iloc[-1] if not df.empty else None
        return df, live_price

    except Exception as e:
        print(f"❌ Error fetching data for {symbol}:", e)
        return pd.DataFrame(), None
