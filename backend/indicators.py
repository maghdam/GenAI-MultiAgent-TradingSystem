# indicators.py
import ta

def add_indicators(data, selected_indicators):
    """Calculate technical indicators based on user selection."""
    
    if "SMA (20)" in selected_indicators:
        data["SMA_20"] = ta.trend.sma_indicator(data["close"], window=20)
    if "EMA (20)" in selected_indicators:
        data["EMA_20"] = ta.trend.ema_indicator(data["close"], window=20)
    if "VWAP" in selected_indicators:
        data["VWAP"] = ta.volume.volume_weighted_average_price(
            data["high"], data["low"], data["close"], data["tick_volume"]
        )
    if "Bollinger Bands" in selected_indicators:
        bb = ta.volatility.BollingerBands(close=data["close"], window=20, window_dev=2)
        data["BB_Upper"] = bb.bollinger_hband()
        data["BB_Lower"] = bb.bollinger_lband()

    data.dropna(inplace=True)  # Remove NaNs caused by indicator calculations
    return data
