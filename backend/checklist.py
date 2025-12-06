from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from typing import Literal, Optional

import pandas as pd

from backend import data_fetcher, ctrader_client as ctd
from backend.agent_state import recent_signals
from backend.data_fetcher import ALLOWED_TF

Bias = Literal["bullish", "bearish", "flat", "unknown"]
Scenario = Literal["", "A", "B", "C", "D"]
VolumeHint = Literal["rising", "falling", "flat", "unknown"]
StructureHint = Literal["bullish", "bearish", "range", "unknown"]


@dataclass
class ComponentStatus:
    symbol: str
    bias: Bias
    change_pct: float | None = None


@dataclass
class AutoChecklist:
    ts: float
    us30_bias: Bias
    xau_bias: Bias
    dxy_bias: Bias = "unknown"
    dxy_change_pct: float | None = None
    correlation: Literal["normal", "fear", "dollar_crash", "weird", "unknown"] = "unknown"
    scenario: Scenario = ""
    components: dict[str, ComponentStatus] | None = None
    component_score: float | None = None
    top_movers: list[ComponentStatus] | None = None
    structure_hint: StructureHint = "unknown"
    volume_hint: VolumeHint = "unknown"
    structure_tf: str | None = None
    volume_tf: str | None = None
    smc_signal: dict | None = None
    notes: str | None = None

    def dict(self):
        return {
            "ts": self.ts,
            "us30_bias": self.us30_bias,
            "xau_bias": self.xau_bias,
            "dxy_bias": self.dxy_bias,
            "dxy_change_pct": self.dxy_change_pct,
            "correlation": self.correlation,
            "scenario": self.scenario,
            "components": {k: asdict(v) for k, v in (self.components or {}).items()},
            "component_score": self.component_score,
            "top_movers": [asdict(v) for v in self.top_movers] if self.top_movers else [],
            "structure_hint": self.structure_hint,
            "volume_hint": self.volume_hint,
            "structure_tf": self.structure_tf,
            "volume_tf": self.volume_tf,
            "smc_signal": self.smc_signal,
            "notes": self.notes,
        }


def _bias_from_df(df: pd.DataFrame) -> tuple[Bias, float | None]:
    """Derive a simple bias and latest percent change from a price series."""
    try:
        if df is None or df.empty or "close" not in df:
            return "unknown", None
        close = df["close"].astype(float)
        if len(close) < 5:
            return "unknown", None
        last = close.iloc[-1]
        prev = close.iloc[-2]
        change_pct = (last / prev - 1.0) * 100.0 if prev else 0.0
        sma = close.rolling(20, min_periods=5).mean().iloc[-1]
        bias = "flat"
        if pd.isna(sma):
            sma = prev
        if change_pct > 0.15 or last > sma:
            bias = "bullish"
        if change_pct < -0.15 or last < sma:
            bias = "bearish"
        return bias, float(change_pct)
    except Exception:
        return "unknown", None


def _structure_hint(df: pd.DataFrame) -> StructureHint:
    try:
        if df is None or df.empty:
            return "unknown"
        close = df["close"].astype(float)
        if len(close) < 30:
            return "unknown"
        sma_fast = close.rolling(20, min_periods=10).mean().iloc[-1]
        sma_slow = close.rolling(50, min_periods=20).mean().iloc[-1]
        if pd.isna(sma_fast) or pd.isna(sma_slow):
            return "unknown"
        if sma_fast > sma_slow and close.iloc[-1] > sma_fast:
            return "bullish"
        if sma_fast < sma_slow and close.iloc[-1] < sma_fast:
            return "bearish"
        return "range"
    except Exception:
        return "unknown"


def _volume_hint(df: pd.DataFrame) -> VolumeHint:
    try:
        if df is None or df.empty or "volume" not in df:
            return "unknown"
        vol = df["volume"].astype(float)
        if len(vol) < 10:
            return "unknown"
        recent = vol.iloc[-1]
        avg = vol.iloc[-20:].mean()
        if avg == 0 or pd.isna(avg):
            return "unknown"
        if recent > avg * 1.15:
            return "rising"
        if recent < avg * 0.85:
            return "falling"
        return "flat"
    except Exception:
        return "unknown"


def _correlation(us30_bias: Bias, xau_bias: Bias) -> Literal["normal", "fear", "dollar_crash", "weird", "unknown"]:
    if "unknown" in (us30_bias, xau_bias):
        return "unknown"
    if us30_bias == "bullish" and xau_bias == "bearish":
        return "normal"
    if us30_bias == "bearish" and xau_bias == "bullish":
        return "fear"
    if us30_bias == "bullish" and xau_bias == "bullish":
        return "dollar_crash"
    return "weird"


def _pick_scenario(components: dict[str, ComponentStatus], us30_bias: Bias, xau_bias: Bias, corr: str) -> Scenario:
    gs = components.get("GS")
    cat = components.get("CAT")
    msft = components.get("MSFT")
    heavy_up = all(c and c.bias == "bullish" for c in (gs, cat, msft))
    heavy_down = all(c and c.bias == "bearish" for c in (gs, cat))

    if heavy_up and corr in ("normal", "dollar_crash"):
        return "A"
    if us30_bias == "bullish" and gs and gs.bias == "bearish":
        return "B"
    if heavy_down and xau_bias == "bullish":
        return "D"
    return "C"


def _fetch_symbol(sym: str, timeframe: str = "M5", bars: int = 150) -> pd.DataFrame:
    df, _ = data_fetcher.fetch_data(sym, timeframe, bars)
    return df


_ALIASES = {
    "US30": ["US30", "US30.cash", "USA30"],
    "XAUUSD": ["XAUUSD", "GOLD"],
    "DXY": ["DXY", "USDOLLAR", "DX", "USDX", "USDX.US"],
    "GS": ["GS.US", "GS"],
    "CAT": ["CAT.US", "CAT"],
    "MSFT": ["MSFT.US", "MSFT"],
    "UNH": ["UNH.US", "UNH"],
    "HD": ["HD.US", "HD"],
    "GDX": ["GDX.US", "GDX"],
    "GDXJ": ["GDXJ.US", "GDXJ"],
    "FDX": ["FDX.US", "FDX", "FDX.US-24", "FDX-24"],
}


def _resolve_symbol(key: str) -> str:
    """Return the first matching symbol from env override, available aliases, or the key itself."""
    env_key = f"CHECKLIST_SYMBOL_{key.upper()}"
    override = os.getenv(env_key)
    if override:
        return override.strip()
    aliases = _ALIASES.get(key.upper(), [key])
    try:
        available = {k.upper() for k in ctd.symbol_name_to_id.keys()}
        for a in aliases:
            if a.upper() in available:
                return a
    except Exception:
        pass
    return aliases[0]


def _valid_tf(tf: str, fallback: str) -> str:
    t = (tf or fallback or "").upper()
    return t if t in ALLOWED_TF else fallback.upper()


def compute_auto_checklist(bias_tf: str = "M5", structure_tf: str = "H1", volume_tf: Optional[str] = None) -> AutoChecklist:
    bias_tf = _valid_tf(bias_tf, "M5")
    structure_tf = _valid_tf(structure_tf, "H1")
    volume_tf = _valid_tf(volume_tf or bias_tf, bias_tf)

    # Symbols configurable via env? Keep defaults focused on US30/XAU + components.
    watch = {
        "US30": bias_tf,
        "XAUUSD": bias_tf,
        "DXY": bias_tf,
        "GS": bias_tf,
        "CAT": bias_tf,
        "MSFT": bias_tf,
        "UNH": bias_tf,
        "HD": bias_tf,
        # Gold confirmation
        "GDX": bias_tf,
        "GDXJ": bias_tf,
        # Transports confirmation
        "FDX": bias_tf,
    }
    resolved = {k: _resolve_symbol(k) for k in watch.keys()}
    components: dict[str, ComponentStatus] = {}
    us30_bias: Bias = "unknown"
    xau_bias: Bias = "unknown"
    dxy_bias: Bias = "unknown"
    dxy_change_pct: float | None = None
    top_movers: list[ComponentStatus] = []

    for base, tf in watch.items():
        sym = resolved.get(base, base)
        df = _fetch_symbol(sym, tf, 150)
        bias, pct = _bias_from_df(df)
        if base == "US30":
            us30_bias = bias
        elif base == "XAUUSD":
            xau_bias = bias
        elif base == "DXY":
            dxy_bias = bias
            dxy_change_pct = pct
        else:
            components[base] = ComponentStatus(symbol=sym, bias=bias, change_pct=pct)
            if pct is not None:
                top_movers.append(ComponentStatus(symbol=base, bias=bias, change_pct=pct))

    top_movers = sorted(top_movers, key=lambda c: abs(c.change_pct or 0.0), reverse=True)[:3]
    corr = _correlation(us30_bias, xau_bias)
    scenario = _pick_scenario(components, us30_bias, xau_bias, corr)

    # Weighted component score: map bias to +1/0/-1 times weight, normalized
    weights = {"GS": 10.5, "CAT": 7.5, "MSFT": 6.3, "UNH": 4.2, "HD": 4.6}
    score_raw = 0.0
    w_sum = 0.0
    for k, w in weights.items():
        st = components.get(k)
        if st:
            w_sum += w
            score_raw += w * (1.0 if st.bias == "bullish" else (-1.0 if st.bias == "bearish" else 0.0))
    component_score = score_raw / w_sum if w_sum else None

    # Structure/volume hints from higher timeframe and intraday volume
    us30_tf = _fetch_symbol(resolved.get("US30", "US30"), structure_tf, 300)
    xau_tf = _fetch_symbol(resolved.get("XAUUSD", "XAUUSD"), structure_tf, 300)
    structure_hint = _structure_hint(us30_tf)
    vol_df = _fetch_symbol(resolved.get("US30", "US30"), volume_tf, 200)
    volume_hint = _volume_hint(vol_df)

    # Latest SMC signal (if available) from agent signals
    smc_sig = None
    try:
        for sig in recent_signals(20):
            if str(sig.get("symbol", "")).upper() in ("US30", resolved.get("US30", "US30").upper()):
                smc_sig = {
                    "symbol": sig.get("symbol"),
                    "timeframe": sig.get("timeframe"),
                    "strategy": sig.get("strategy"),
                    "signal": sig.get("signal"),
                    "confidence": sig.get("confidence"),
                    "rationale": sig.get("rationale"),
                }
                break
    except Exception:
        smc_sig = None

    notes = None
    try:
        score_txt = f"{component_score:.2f}" if component_score is not None else "n/a"
        notes = (
            f"Scenario {scenario or 'n/a'} from GS/CAT/MSFT alignment, "
            f"US30 bias {us30_bias}, XAU bias {xau_bias}, corr {corr}, "
            f"DXY {dxy_bias}, score {score_txt}."
        )
    except Exception:
        notes = None

    return AutoChecklist(
        ts=time.time(),
        us30_bias=us30_bias,
        xau_bias=xau_bias,
        dxy_bias=dxy_bias,
        dxy_change_pct=dxy_change_pct,
        correlation=corr,
        scenario=scenario,
        components=components,
        component_score=component_score,
        top_movers=top_movers,
        structure_hint=structure_hint,
        volume_hint=volume_hint,
        structure_tf=structure_tf,
        volume_tf=volume_tf,
        smc_signal=smc_sig,
        notes=notes,
    )
