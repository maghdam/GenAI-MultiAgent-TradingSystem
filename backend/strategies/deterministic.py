from __future__ import annotations

from typing import Dict

import pandas as pd

from backend.domain.models import StrategyAnalysis
from backend.strategies.base import StrategyV2


def _atr(df: pd.DataFrame, length: int = 14) -> float | None:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    value = tr.rolling(length, min_periods=length).mean().iloc[-1]
    if pd.isna(value):
        return None
    return float(value)


def _sltp(entry: float, side: str, atr: float | None, rr: float = 2.0) -> tuple[float | None, float | None]:
    if atr is None or atr <= 0:
        return None, None
    if side == "long":
        stop = entry - atr
        take = entry + rr * atr
    else:
        stop = entry + atr
        take = entry - rr * atr
    return float(stop), float(take)


class SmaCrossStrategy(StrategyV2):
    key = "sma_cross"
    label = "SMA Cross"
    description = "Directional crossover with ATR-based exits."
    parameters = {"fast": 20, "slow": 50, "rr": 2.0}

    def analyze(self, df: pd.DataFrame, symbol: str, timeframe: str, params: Dict[str, object]) -> StrategyAnalysis:
        fast = max(2, int(params.get("fast", 20)))
        slow = max(fast + 1, int(params.get("slow", 50)))
        rr = float(params.get("rr", 2.0))

        close = df["close"].astype(float)
        fast_ma = close.rolling(fast, min_periods=fast).mean()
        slow_ma = close.rolling(slow, min_periods=slow).mean()
        entry = float(close.iloc[-1])
        atr = _atr(df)
        signal = "no_trade"
        reasons = [
            f"fast_ma={fast_ma.iloc[-1]:.4f}" if pd.notna(fast_ma.iloc[-1]) else "fast_ma=unavailable",
            f"slow_ma={slow_ma.iloc[-1]:.4f}" if pd.notna(slow_ma.iloc[-1]) else "slow_ma=unavailable",
        ]

        if pd.notna(fast_ma.iloc[-1]) and pd.notna(slow_ma.iloc[-1]):
            gap = float((fast_ma.iloc[-1] - slow_ma.iloc[-1]) / entry) if entry else 0.0
            if fast_ma.iloc[-1] > slow_ma.iloc[-1]:
                signal = "long"
                reasons.append("Fast SMA is above slow SMA.")
            elif fast_ma.iloc[-1] < slow_ma.iloc[-1]:
                signal = "short"
                reasons.append("Fast SMA is below slow SMA.")
            confidence = min(0.85, 0.5 + abs(gap) * 400)
        else:
            confidence = 0.0
            reasons.append("Not enough bars to compute the moving averages.")

        stop, take = _sltp(entry, signal, atr, rr) if signal != "no_trade" else (None, None)
        return StrategyAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            strategy=self.key,
            signal=signal,
            confidence=float(confidence),
            entry_price=entry,
            stop_loss=stop,
            take_profit=take,
            reasons=reasons,
            context={"fast": fast, "slow": slow, "rr": rr, "atr": atr},
        )


class RsiReversalStrategy(StrategyV2):
    key = "rsi_reversal"
    label = "RSI Reversal"
    description = "Mean reversion around RSI extremes with ATR exits."
    parameters = {"length": 14, "lower": 30.0, "upper": 70.0, "rr": 1.8}

    def analyze(self, df: pd.DataFrame, symbol: str, timeframe: str, params: Dict[str, object]) -> StrategyAnalysis:
        length = max(2, int(params.get("length", 14)))
        lower = float(params.get("lower", 30.0))
        upper = float(params.get("upper", 70.0))
        rr = float(params.get("rr", 1.8))

        close = df["close"].astype(float)
        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(alpha=1 / length, adjust=False).mean()
        loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / length, adjust=False).mean()
        rs = gain / loss.replace(0.0, pd.NA)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi_last = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None
        entry = float(close.iloc[-1])
        atr = _atr(df)

        signal = "no_trade"
        reasons = [f"rsi={rsi_last:.2f}" if rsi_last is not None else "rsi=unavailable"]
        confidence = 0.0

        if rsi_last is not None:
            if rsi_last <= lower:
                signal = "long"
                confidence = min(0.82, 0.55 + (lower - rsi_last) / 100)
                reasons.append("RSI is in the oversold region.")
            elif rsi_last >= upper:
                signal = "short"
                confidence = min(0.82, 0.55 + (rsi_last - upper) / 100)
                reasons.append("RSI is in the overbought region.")
            else:
                reasons.append("RSI is between thresholds; no trade.")

        stop, take = _sltp(entry, signal, atr, rr) if signal != "no_trade" else (None, None)
        return StrategyAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            strategy=self.key,
            signal=signal,
            confidence=float(confidence),
            entry_price=entry,
            stop_loss=stop,
            take_profit=take,
            reasons=reasons,
            context={"length": length, "lower": lower, "upper": upper, "rr": rr, "atr": atr},
        )


class BreakoutStrategy(StrategyV2):
    key = "breakout"
    label = "Range Breakout"
    description = "Break above rolling high or below rolling low."
    parameters = {"lookback": 20, "rr": 2.2}

    def analyze(self, df: pd.DataFrame, symbol: str, timeframe: str, params: Dict[str, object]) -> StrategyAnalysis:
        lookback = max(5, int(params.get("lookback", 20)))
        rr = float(params.get("rr", 2.2))

        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        recent_high = float(high.rolling(lookback, min_periods=lookback).max().shift(1).iloc[-1])
        recent_low = float(low.rolling(lookback, min_periods=lookback).min().shift(1).iloc[-1])
        entry = float(close.iloc[-1])
        atr = _atr(df)

        signal = "no_trade"
        confidence = 0.0
        reasons = [f"recent_high={recent_high:.4f}", f"recent_low={recent_low:.4f}"]
        if entry > recent_high:
            signal = "long"
            confidence = min(0.86, 0.58 + (entry - recent_high) / entry * 250)
            reasons.append("Close broke above the rolling range high.")
        elif entry < recent_low:
            signal = "short"
            confidence = min(0.86, 0.58 + (recent_low - entry) / entry * 250)
            reasons.append("Close broke below the rolling range low.")
        else:
            reasons.append("Price remains inside the rolling range.")

        stop, take = _sltp(entry, signal, atr, rr) if signal != "no_trade" else (None, None)
        return StrategyAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            strategy=self.key,
            signal=signal,
            confidence=float(confidence),
            entry_price=entry,
            stop_loss=stop,
            take_profit=take,
            reasons=reasons,
            context={"lookback": lookback, "rr": rr, "atr": atr},
        )
