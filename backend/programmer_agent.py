import textwrap
from typing import Literal

TaskKind = Literal["indicator", "backtest", "strategy"]


class ProgrammerAgent:
    """Very lightweight code generator stub.

    In a full setup this would call an LLM to synthesize code. For now we
    return concise, readable snippets that the UI can show in the results view.
    """

    async def generate_code(self, goal: str, task_type: TaskKind) -> str:
        goal = (goal or "").strip()
        goal_lower = goal.lower()
        if task_type == "indicator":
            src = f"""
                # Example: Simple RSI (14) using pandas
                import pandas as pd

                def rsi(series: pd.Series, period: int = 14) -> pd.Series:
                    delta = series.diff()
                    gain = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
                    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
                    rs = gain / loss.replace(0, 1e-9)
                    return 100 - (100 / (1 + rs))

                # Usage: df['rsi14'] = rsi(df['close'], 14)
            """
            return textwrap.dedent(src).strip()

        if task_type == "strategy":
            # Lightweight keyword router so Strategy Studio returns code closer to the user's intent.
            smc_keywords = (
                "smc",
                "fair value gap",
                "fvg",
                "market structure",
                "structure break",
                "bos",
                "choch",
                "order block",
                "premium",
                "discount",
            )
            if any(keyword in goal_lower for keyword in smc_keywords):
                src = f"""
                    # SMC-style strategy generated from the request below.
                    # Original request: {goal or "no description provided"}
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
                """
                return textwrap.dedent(src).strip()

            if "daily" in goal_lower and "close" in goal_lower:
                src = f"""
                    # Trend-following on daily closes: long if today's close is above yesterday's, else short.
                    import pandas as pd

                    def daily_candle_cross(df: pd.DataFrame) -> pd.DataFrame:
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
                        df["position"] = df["long_signal"] - df["short_signal"]
                        df["returns"] = df["close"].pct_change()
                        df["strategy_returns"] = df["returns"] * df["position"].shift(1).fillna(0)
                        return df

                    def signals(df: pd.DataFrame) -> pd.Series:
                        enriched = daily_candle_cross(df)
                        return enriched["position"].reindex(df.index).fillna(0)
                """
                return textwrap.dedent(src).strip()

            if "rsi" in goal_lower:
                src = f"""
                    # Mean-reversion using RSI: long when RSI < 30, short when RSI > 70.
                    import pandas as pd

                    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
                        delta = series.diff()
                        gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
                        loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
                        rs = gain / loss.replace(0, 1e-9)
                        return 100 - (100 / (1 + rs))

                    def signals(df: pd.DataFrame, period: int = 14, lower: float = 30, upper: float = 70) -> pd.Series:
                        r = rsi(df["close"], period)
                        sig = pd.Series(0, index=df.index, dtype=float)
                        sig[r < lower] = 1   # long bias
                        sig[r > upper] = -1  # short bias
                        return sig.ffill().fillna(0)
                """
                return textwrap.dedent(src).strip()

            if "macd" in goal_lower:
                src = f"""
                    # MACD line/signal crossover.
                    import pandas as pd

                    def ema(series: pd.Series, span: int) -> pd.Series:
                        return series.ewm(span=span, adjust=False).mean()

                    def signals(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
                        price = df["close"]
                        macd_line = ema(price, fast) - ema(price, slow)
                        signal_line = ema(macd_line, signal)
                        cross_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
                        cross_dn = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
                        sig = pd.Series(0, index=df.index, dtype=float)
                        sig[cross_up] = 1
                        sig[cross_dn] = -1
                        return sig.ffill().fillna(0)
                """
                return textwrap.dedent(src).strip()

            if "bollinger" in goal_lower:
                src = f"""
                    # Bollinger Band mean reversion: long at lower band touch, short at upper band touch.
                    import pandas as pd

                    def signals(df: pd.DataFrame, length: int = 20, mult: float = 2.0) -> pd.Series:
                        close = df["close"]
                        ma = close.rolling(length, min_periods=length).mean()
                        std = close.rolling(length, min_periods=length).std(ddof=0)
                        upper = ma + mult * std
                        lower = ma - mult * std
                        sig = pd.Series(0, index=df.index, dtype=float)
                        sig[close < lower] = 1
                        sig[close > upper] = -1
                        return sig.ffill().fillna(0)
                """
                return textwrap.dedent(src).strip()

            if "breakout" in goal_lower or "range" in goal_lower:
                src = f"""
                    # Simple breakout: long on highest-high breakout, short on lowest-low breakdown.
                    import pandas as pd

                    def signals(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
                        highs = df["high"].rolling(lookback, min_periods=lookback).max()
                        lows = df["low"].rolling(lookback, min_periods=lookback).min()
                        close = df["close"]
                        long_trig = (close > highs.shift(1))
                        short_trig = (close < lows.shift(1))
                        sig = pd.Series(0, index=df.index, dtype=float)
                        sig[long_trig] = 1
                        sig[short_trig] = -1
                        return sig.ffill().fillna(0)
                """
                return textwrap.dedent(src).strip()

            # Fallback: SMA crossover (kept as a default) but at least echo the request.
            src = f"""
                # Default crossover strategy (fallback when the request is unclear).
                # Original request: {goal or "no description provided"}
                import pandas as pd

                def sma(series: pd.Series, n: int) -> pd.Series:
                    return series.rolling(n, min_periods=n).mean()

                def signals(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
                    f = sma(df["close"], fast)
                    s = sma(df["close"], slow)
                    sig = (f > s).astype(int).diff().fillna(0)
                    # 1 = long entry, -1 = exit/short entry depending on rules
                    return sig
            """
            return textwrap.dedent(src).strip()

        # For backtest requests, return a helper snippet to show intent
        src = f"""
            # Backtest outline (pseudocode):
            # - Fetch OHLCV data
            # - Generate signals
            # - Iterate bars and track PnL with simple costs
            # - Compute metrics (total return, win rate, max drawdown)
        """
        return textwrap.dedent(src).strip()
