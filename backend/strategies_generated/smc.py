"""
Generated SMC strategy wrapper that uses the shared LLM analyzer.

Contract: expose `trade_decision(df, symbol=None, timeframe=None)` so the
loader can build a Strategy around it. This keeps strategies consistent with
Strategy Studio's generated files while preserving LLM-based reasoning.
"""

from typing import Optional, Dict, Any
import pandas as pd

from backend.llm_analyzer import analyze_data_with_llm, TradeDecision


async def trade_decision(
    df: pd.DataFrame,
    *,
    symbol: str | None = None,
    timeframe: str | None = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"signal": "no_trade", "confidence": 0.0, "reasons": ["no data"]}

    td: TradeDecision = await analyze_data_with_llm(
        df=df,
        symbol=symbol or "",
        timeframe=timeframe or "",
        indicators=None,
        model=None,
        options=options or {},
        ollama_url=None,
    )
    return td.dict()

