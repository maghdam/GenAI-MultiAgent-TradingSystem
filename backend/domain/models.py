from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


SignalValue = Literal["long", "short", "no_trade"]
IncidentLevel = Literal["info", "warning", "error"]
StudioTaskType = Literal[
    "calculate_indicator",
    "backtest_strategy",
    "save_strategy",
    "research_strategy",
    "create_strategy",
    "backtest",
    "optimize",
    "chat",
]


def utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class BrokerStatus(BaseModel):
    connected: bool
    socket_connected: bool = False
    account_authorized: bool = False
    auth_error: Optional[str] = None
    last_auth_attempt_at: Optional[datetime] = None
    symbols_loaded: int
    open_positions: int
    pending_orders: int
    ready: bool
    market_data_ready: bool = False
    broker_mode: str
    account_id: Optional[int] = None
    notes: List[str] = Field(default_factory=list)


class SymbolLimits(BaseModel):
    symbol: str
    source: str = "fallback"
    min_lots: float
    step_lots: float
    max_lots: float
    min_api_units: int
    step_api_units: int
    max_api_units: int
    hard_min: bool = False
    hard_step: bool = False


class EngineConfig(BaseModel):
    enabled: bool = False
    paper_autotrade: bool = False
    allow_live: bool = False
    kill_switch: bool = True
    default_symbol: str = "XAUUSD"
    default_timeframe: str = "M5"
    default_strategy: str = "sma_cross"
    scan_interval_sec: int = 10
    min_confidence: float = 0.6
    paper_trade_size: float = 1.0
    risk_per_trade_pct: float = 0.5
    daily_loss_limit_pct: float = 2.0
    max_daily_trades: int = 12
    max_open_positions: int = 3
    max_positions_per_symbol: int = 1
    cooldown_minutes: int = 30
    session_filter_enabled: bool = False
    session_start_hour_utc: int = 6
    session_end_hour_utc: int = 21
    require_stops: bool = True
    operator_note: str = ""
    watchlist: List["WatchlistItem"] = Field(default_factory=list)


class WatchlistItem(BaseModel):
    symbol: str
    timeframe: str
    strategy: str = "sma_cross"
    enabled: bool = True
    params: Dict[str, Any] = Field(default_factory=dict)


class StrategyInfo(BaseModel):
    key: str
    label: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class AnalyzeRequest(BaseModel):
    symbol: str = Field(..., min_length=1)
    timeframe: str = Field(default="M5", min_length=1)
    strategy: str = Field(default="sma_cross", min_length=1)
    num_bars: int = Field(default=500, ge=100, le=5000)
    params: Dict[str, Any] = Field(default_factory=dict)


class StudioTaskRequest(BaseModel):
    task_type: StudioTaskType
    goal: str = ""
    params: Dict[str, Any] = Field(default_factory=dict)


class StudioTaskResponse(BaseModel):
    status: Literal["success", "error"]
    message: str = ""
    result: Optional[Any] = None


class ManualOrderRequest(BaseModel):
    symbol: str = Field(..., min_length=1)
    timeframe: str = Field(default="M5", min_length=1)
    strategy: str = Field(default="manual", min_length=1)
    signal: Literal["long", "short"]
    quantity: float = Field(default=1.0, gt=0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasons: List[str] = Field(default_factory=list)
    rationale: str = ""


class StrategyAnalysis(BaseModel):
    symbol: str
    timeframe: str
    strategy: str
    signal: SignalValue
    confidence: float = 0.0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasons: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow_naive)


class PaperPosition(BaseModel):
    id: int
    symbol: str
    timeframe: str
    strategy: str
    direction: Literal["long", "short"]
    quantity: float
    status: Literal["open", "closed"]
    entry_price: float
    current_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    opened_at: datetime
    closed_at: Optional[datetime] = None
    exit_price: Optional[float] = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    close_reason: Optional[str] = None


class PaperEvent(BaseModel):
    id: int
    created_at: datetime
    event_type: str
    summary: str
    details: Dict[str, Any] = Field(default_factory=dict)


class OrderIntentRecord(BaseModel):
    id: int
    created_at: datetime
    symbol: str
    timeframe: str
    strategy: str
    direction: SignalValue
    intent_type: Literal["open", "close", "update", "hold", "skip"]
    status: Literal["pending", "accepted", "rejected", "executed", "cancelled"]
    confidence: float = 0.0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    quantity: Optional[float] = None
    rationale: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)


class TradeAuditRecord(BaseModel):
    id: int
    created_at: datetime
    event_type: str
    symbol: str
    timeframe: str
    strategy: str
    position_id: Optional[int] = None
    intent_id: Optional[int] = None
    summary: str
    details: Dict[str, Any] = Field(default_factory=dict)


class IncidentRecord(BaseModel):
    id: int
    level: IncidentLevel
    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ReadinessCheck(BaseModel):
    name: str
    ok: bool
    detail: str


class EngineRuntime(BaseModel):
    running: bool = False
    loop_active: bool = False
    ollama_ready: bool = False
    last_cycle_at: Optional[datetime] = None
    last_cycle_summary: str = ""
    last_reconcile_at: Optional[datetime] = None
    last_reconcile_summary: str = ""
    last_error: Optional[str] = None
    tick_count: int = 0
    active_watchlist: List[str] = Field(default_factory=list)


class EngineStatus(BaseModel):
    version: str
    mode: Literal["paper_only", "live_enabled"]
    broker: BrokerStatus
    config: EngineConfig
    runtime: EngineRuntime
    readiness: List[ReadinessCheck] = Field(default_factory=list)
    strategies: List[StrategyInfo] = Field(default_factory=list)
    recent_incidents: List[IncidentRecord] = Field(default_factory=list)
    recent_analyses: List[StrategyAnalysis] = Field(default_factory=list)
    paper_positions: List[PaperPosition] = Field(default_factory=list)
    recent_events: List[PaperEvent] = Field(default_factory=list)
    recent_order_intents: List[OrderIntentRecord] = Field(default_factory=list)
    recent_trade_audits: List[TradeAuditRecord] = Field(default_factory=list)
