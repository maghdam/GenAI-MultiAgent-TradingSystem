# backend/models.py
from pydantic import BaseModel, Field
from typing import Literal, Optional

class SignalRequest(BaseModel):
    symbol: str
    timeframe: str = "M5"
    num_bars: int = 500
    strategy: str = "smc"   # pluggable, default SMC
    indicators: list[str] = []

class SignalResponse(BaseModel):
    signal: Literal["long", "short", "no_trade", "flat"] = "no_trade"
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    confidence: float = Field(0, ge=0, le=1)
    rationale: str = ""
