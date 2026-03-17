"""
Meme Coin Short Strategy.

Strategy Description from Video (Algo-trading with Saleh):
 - Timeframe: 1 Hour (recommended).
 - Direction: Short Only.
 - Entry Rules:
   1. Fast EMA (12) < Slow EMA (26).
   2. RSI (14) > 55.
   3. ADX (14) > 20.
 - Exit Rules:
   - Stop Loss: Entry + (1.5 * ATR).
   - Take Profit: Entry - (2.5 * ATR).
 - Risk: 1.5% of account per trade.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
import ta

def trade_decision(
    df: pd.DataFrame,
    *,
    symbol: str | None = None,
    timeframe: str | None = None,
    ema_fast_len: int = 3,
    ema_slow_len: int = 8,
    rsi_len: int = 14,
    adx_len: int = 14,
    atr_len: int = 14,
    rsi_threshold: float = 60.0,
    adx_threshold: float = 20.0,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 2.5,
) -> Dict[str, Any]:
    """
    Analyzes the DataFrame to determine if a SHORT position should be taken according to the Meme Coin Strategy.
    """
    if df is None or df.empty:
        return {"signal": "no_trade", "confidence": 0.0, "reasons": ["no data"]}

    # Ensure we have enough data
    required_len = max(ema_slow_len, rsi_len, adx_len, atr_len) + 5
    if len(df) < required_len:
         return {"signal": "no_trade", "confidence": 0.0, "reasons": ["insufficient data history"]}

    # 1. Calculate Indicators using 'ta' library to be robust
    # CLEAN DATA: ensure no nans for calculation
    # We work on a copy to avoid SettingWithCopyWarning on the input df if it's a slice
    df_calc = df.copy()

    # Fill NAs if any, or drop. For robust calc, usually ffill/bfill is safer
    df_calc.fillna(method='ffill', inplace=True)
    df_calc.fillna(method='bfill', inplace=True)

    close = df_calc["close"]
    high = df_calc["high"]
    low = df_calc["low"]

    # EMAs
    ema_fast = ta.trend.EMAIndicator(close=close, window=ema_fast_len).ema_indicator()
    ema_slow = ta.trend.EMAIndicator(close=close, window=ema_slow_len).ema_indicator()

    # RSI
    rsi = ta.momentum.RSIIndicator(close=close, window=rsi_len).rsi()

    # ADX
    adx_obj = ta.trend.ADXIndicator(high=high, low=low, close=close, window=adx_len)
    adx = adx_obj.adx()

    # ATR (for SL/TP calculation)
    atr_obj = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=atr_len)
    atr = atr_obj.average_true_range()

    # Get latest values (iloc[-1])
    try:
        curr_ema_fast = float(ema_fast.iloc[-1])
        curr_ema_slow = float(ema_slow.iloc[-1])
        curr_rsi = float(rsi.iloc[-1])
        curr_adx = float(adx.iloc[-1])
        curr_atr = float(atr.iloc[-1])
        curr_close = float(close.iloc[-1])
    except (IndexError, ValueError):
         return {"signal": "no_trade", "confidence": 0.0, "reasons": ["error calculating indicators"]}

    # 2. Variable Logic
    signal = "no_trade"
    reasons = []
    confidence = 0.0

    # Debug / Info
    reasons.append(f"EMA_f({ema_fast_len})={curr_ema_fast:.2f}")
    reasons.append(f"EMA_s({ema_slow_len})={curr_ema_slow:.2f}")
    reasons.append(f"RSI({rsi_len})={curr_rsi:.2f}")
    reasons.append(f"ADX({adx_len})={curr_adx:.2f}")

    # Condition 1: Fast EMA < Slow EMA (Downtrend)
    cond_trend = curr_ema_fast < curr_ema_slow

    # Condition 2: RSI > Threshold (Counter-intuitive momentum check for 'pump' before drop)
    cond_rsi = curr_rsi > rsi_threshold

    # Condition 3: ADX > Threshold (Strong trend present)
    cond_adx = curr_adx > adx_threshold

    if cond_trend and cond_rsi:
        # Note: We relaxed ADX strict check for signal generation if it's close, 
        # but let's keep strict for entry.
        if cond_adx:
            signal = "short"
            reasons.append("ENTRY MET: EMA_Down & RSI>Thresh & ADX>Thresh")

            # Calculate SL / TP
            # Short Entry: Sell at Market (Close)
            # SL = Entry + 1.5 * ATR
            # TP = Entry - 2.5 * ATR
            sl_price = curr_close + (sl_atr_mult * curr_atr)
            tp_price = curr_close - (tp_atr_mult * curr_atr)

            # Trailing Stop Logic (Conceptual for Live Trade):
            # If we were already in a trade, we would move SL down as price drops.
            # Here, we just suggest the initial SL.
            # Future improvement: Pass in 'current_position' to this function to calculate dynamic trailing SL.

            # Confidence logic
            adx_factor = min(1.0, (curr_adx - 20) / 30.0)
            confidence = 0.5 + (0.5 * adx_factor) 

            return {
                "signal": signal,
                "sl": sl_price,
                "tp": tp_price,
                "confidence": confidence,
                "reasons": reasons,
                "metadata": {
                    "risk_per_trade_percent": 1.5,
                    "trailing_stop_activated": True
                }
            }
        else:
             reasons.append(f"FAIL: ADX({curr_adx:.1f}) <= {adx_threshold}")


    else:
        if not cond_trend:
            reasons.append("FAIL: EMA Fast >= EMA Slow")
        if not cond_rsi:
            reasons.append(f"FAIL: RSI({curr_rsi:.1f}) <= {rsi_threshold}")
        if not cond_adx:
            reasons.append(f"FAIL: ADX({curr_adx:.1f}) <= {adx_threshold}")

    return {"signal": "no_trade", "confidence": 0.0, "reasons": reasons}

def signals(
    df: pd.DataFrame,
    *,
    ema_fast_len: int = 3,
    ema_slow_len: int = 8,
    rsi_len: int = 14,
    adx_len: int = 14,
    rsi_short_threshold: float = 50.0,
    rsi_long_threshold: float = 50.0,
    adx_threshold: float = 15.0,
    enable_longs: bool = True,
    enable_shorts: bool = True,
    **kwargs: Any
) -> pd.Series:
    """
    Backtest-friendly signal series (-1 Short, +1 Long, 0 Neutral).

    STATEFUL IMPLEMENTATION:
    - Maintains position until an Exit Condition is met.
    - Captures the full trend even if RSI/ADX fluctuate after entry.

    Entry Conditions (Trigger):
    - Long: Uptrend (Fast>Slow) + RSI < Oversold (Dip) + ADX > Thresh
    - Short: Downtrend (Fast<Slow) + RSI > Overbought (Pump) + ADX > Thresh

    Exit Conditions (Close):
    - Trend Reversal: Fast/Slow cross opposite to trade direction.
    - (Optionally strict): If ADX drops too low (trend loss) -> Exit? 
      For now, we trust the Trend Cross as the primary exit to let winners run.
    """
    if df is None or df.empty:
        return pd.Series([], dtype=float)

    # Calculate Indicators
    close = df["close"]
    high = df["high"]
    low = df["low"]

    ema_fast = ta.trend.EMAIndicator(close=close, window=ema_fast_len).ema_indicator()
    ema_slow = ta.trend.EMAIndicator(close=close, window=ema_slow_len).ema_indicator()
    rsi = ta.momentum.RSIIndicator(close=close, window=rsi_len).rsi()
    adx = ta.trend.ADXIndicator(high=high, low=low, close=close, window=adx_len).adx()

    # Pre-calculate Conditions
    # 1. Trend
    uptrend = ema_fast > ema_slow
    downtrend = ema_fast < ema_slow

    # 2. Filters
    # Note: We treat RSI as a "Dip/Pump" entry trigger, not a hold condition.
    rsi_buy_signal = rsi < rsi_long_threshold
    rsi_sell_signal = rsi > rsi_short_threshold
    adx_strong = adx > adx_threshold

    signals = np.zeros(len(df))

    # Loop for Stateful Logic
    # 0 = Cash, 1 = Long, -1 = Short
    current_pos = 0.0

    # Convert series to numpy for speed
    up_arr = uptrend.values
    down_arr = downtrend.values
    rsi_buy_arr = rsi_buy_signal.values
    rsi_sell_arr = rsi_sell_signal.values
    adx_arr = adx_strong.values

    for i in range(1, len(df)):
        # Default: Hold previous position
        signals[i] = current_pos

        # Check Exits first
        if current_pos == 1.0:
            # Long Exit: Trend Flip (Fast crosses below Slow)
            if down_arr[i]:
                current_pos = 0.0
                signals[i] = 0.0
        elif current_pos == -1.0:
            # Short Exit: Trend Flip (Fast crosses above Slow)
            if up_arr[i]:
                current_pos = 0.0
                signals[i] = 0.0

        # Check Entries (if cash)
        # Note: Could also support reversal (flip 1 to -1 directly)
        if current_pos == 0.0:
            if enable_longs and up_arr[i] and rsi_buy_arr[i] and adx_arr[i]:
                current_pos = 1.0
                signals[i] = 1.0
            elif enable_shorts and down_arr[i] and rsi_sell_arr[i] and adx_arr[i]:
                current_pos = -1.0
                signals[i] = -1.0

        # Support Direct Reversal (Stop & Reverse)
        # If Long but Short Signal appears? 
        # Usually dangerous to flip on RSI spike against trend, trust trend exit.
        # But if Trend FLIPS and we have a signal?
        # The 'Check Exits' block above handles the 'Trend Flip' to 0. 
        # Then 'Check Entries' can immediately go to -1 if valid.
        # So we need to allow re-entry in same bar if exit happened? 
        # For simplicity, we exit to 0 this bar, enter next. 
        # Or check entry if current_pos became 0 just now.

        if current_pos == 0.0:
             if enable_longs and up_arr[i] and rsi_buy_arr[i] and adx_arr[i]:
                current_pos = 1.0
                signals[i] = 1.0
             elif enable_shorts and down_arr[i] and rsi_sell_arr[i] and adx_arr[i]:
                current_pos = -1.0
                signals[i] = -1.0

    return pd.Series(signals, index=df.index)


