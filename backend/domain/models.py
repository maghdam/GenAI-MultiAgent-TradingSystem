from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


SignalValue = Literal["long", "short", "no_trade"]
IncidentLevel = Literal["info", "warning", "error"]
StrategyLifecycleStage = Literal["draft", "backtested", "validated", "paper", "eligible", "retired"]
StrategyEvidenceType = Literal["development_backtest", "out_of_sample", "regime", "paper"]
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


class BrokerAccountSnapshot(BaseModel):
    account_id: Optional[int] = None
    currency: Optional[str] = None
    balance: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    equity: Optional[float] = None
    money_digits: Optional[int] = None
    deposit_asset_id: Optional[int] = None
    source: str = "ctrader"
    verified: bool = False
    as_of: Optional[datetime] = None
    notes: List[str] = Field(default_factory=list)


class CTraderAccount(BaseModel):
    account_id: int
    account_type: Literal["demo", "live"]
    is_live: bool
    trader_login: Optional[int] = None
    broker_title: Optional[str] = None
    selected: bool = False
    active: bool = False


class CTraderAccountSelectionRequest(BaseModel):
    account_id: int = Field(..., gt=0)


class CTraderAccountSelectionResponse(BaseModel):
    selected_account: CTraderAccount
    active_account_id: Optional[int] = None
    transport_switch_required: bool = False
    switch_started: bool = False


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
    account_type: Literal["demo", "live", "unknown"] = "unknown"
    active_host_type: Literal["demo", "live", "unknown"] = "unknown"
    account_switch_in_progress: bool = False
    account_switch_target_id: Optional[int] = None
    account_switch_error: Optional[str] = None
    account_verified: bool = False
    demo_account_confirmed: bool = False
    execution_ready: bool = False
    account_snapshot: Optional["BrokerAccountSnapshot"] = None
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


class InstrumentSpec(BaseModel):
    """Monetary contract used for position sizing and paper P&L."""

    symbol: str
    source: str = "fallback"
    account_currency: str = "USD"
    quote_currency: Optional[str] = None
    lot_size_units: Optional[float] = None
    tick_size: Optional[float] = None
    tick_value_per_lot: Optional[float] = None
    cash_per_price_unit_per_lot: Optional[float] = None
    conversion_rate_to_account: Optional[float] = None
    valuation_ready: bool = False
    verified: bool = False
    notes: List[str] = Field(default_factory=list)


class EngineConfig(BaseModel):
    enabled: bool = False
    selected_ctrader_account_id: Optional[int] = Field(default=None, gt=0)
    selected_ctrader_account_type: Optional[Literal["demo", "live"]] = None
    paper_autotrade: bool = False
    ctrader_autotrade: bool = False
    kill_switch: bool = True
    default_symbol: str = "XAUUSD"
    default_timeframe: str = "M5"
    default_strategy: str = "sma_cross"
    scan_interval_sec: int = 10
    min_confidence: float = 0.6
    paper_trade_size: float = 1.0
    account_currency: str = Field(default="USD", min_length=3, max_length=3, pattern=r"^[A-Za-z]{3}$")
    paper_starting_equity_amount: float = Field(default=100_000.0, gt=0)
    risk_per_trade_pct: float = Field(default=0.5, ge=0, le=5)
    daily_loss_limit_pct: float = Field(default=2.0, gt=0, le=50)
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

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_execution_flags(cls, value):
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if "ctrader_autotrade" not in payload and "demo_autotrade" in payload:
            payload["ctrader_autotrade"] = bool(payload.get("demo_autotrade"))
        payload.pop("demo_autotrade", None)
        payload.pop("allow_live", None)
        return payload

    @property
    def demo_autotrade(self) -> bool:
        """Backward-compatible read alias for pre-Phase-10.4 callers/tests."""
        return self.ctrader_autotrade


class WatchlistItem(BaseModel):
    symbol: str
    timeframe: str
    strategy: str = "sma_cross"
    enabled: bool = True
    trading_enabled: bool = False
    lot_size: Optional[float] = Field(default=None, gt=0, le=100)
    params: Dict[str, Any] = Field(default_factory=dict)


class StrategyInfo(BaseModel):
    key: str
    label: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class StrategyLifecycleEvidence(BaseModel):
    id: int
    lifecycle_id: int
    evidence_type: StrategyEvidenceType
    passed: bool
    summary: str = ""
    metrics: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class StrategyLifecycleTransition(BaseModel):
    id: int
    lifecycle_id: int
    from_stage: StrategyLifecycleStage
    to_stage: StrategyLifecycleStage
    operator: str
    reason: str = ""
    created_at: datetime


class StrategyLifecycle(BaseModel):
    id: int
    strategy: str
    version: int
    version_hash: str
    stage: StrategyLifecycleStage = "draft"
    hypothesis: str = ""
    gates: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    evidence: List[StrategyLifecycleEvidence] = Field(default_factory=list)
    transitions: List[StrategyLifecycleTransition] = Field(default_factory=list)
    current_source: bool = True
    next_stage: Optional[StrategyLifecycleStage] = None
    promotion_ready: bool = False
    blockers: List[str] = Field(default_factory=list)


class StrategyLifecycleUpdateRequest(BaseModel):
    hypothesis: str = Field(default="", max_length=10_000)


class StrategyLifecyclePromoteRequest(BaseModel):
    operator: str = Field(..., min_length=2, max_length=120)
    reason: str = Field(..., min_length=3, max_length=2000)


class StrategyLifecycleEvidenceRequest(BaseModel):
    evidence_type: StrategyEvidenceType
    metrics: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    summary: str = Field(default="", max_length=2000)


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
    lifecycle_version_hash: Optional[str] = None
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
    account_currency: str = "USD"
    cash_per_price_unit_per_lot: float = 1.0
    instrument_spec_source: str = "legacy"
    broker_position_id: Optional[int] = None
    realized_pnl_source: str = "paper_estimate"
    broker_entry_price: Optional[float] = None
    broker_stop_loss: Optional[float] = None
    broker_take_profit: Optional[float] = None
    broker_protection_status: Literal["protected", "partial", "unprotected", "unavailable"] = "unavailable"
    broker_last_synced_at: Optional[datetime] = None
    broker_sync_status: str = "not_applicable"


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
    status: Literal["pending", "accepted", "rejected", "executed", "cancelled", "failed"]
    confidence: float = 0.0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    quantity: Optional[float] = None
    rationale: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)
    decision_id: Optional[int] = None


class OrderIntentTransitionRecord(BaseModel):
    id: int
    intent_id: int
    created_at: datetime
    from_status: Optional[str] = None
    to_status: Literal["pending", "accepted", "rejected", "executed", "cancelled", "failed"]
    reason: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)


class DecisionRecord(BaseModel):
    id: int
    created_at: datetime
    correlation_id: str
    decision_type: str
    symbol: str
    timeframe: str
    strategy: str
    outcome: str
    summary: str
    evidence: Dict[str, Any] = Field(default_factory=dict)


class DailyRejectionReason(BaseModel):
    reason: str
    count: int


class DailySummaryResponse(BaseModel):
    date_utc: date
    account_currency: str
    trades: int
    trades_opened: int
    trades_closed: int
    realized_pnl: float
    max_realized_drawdown: float
    wins: int
    losses: int
    breakeven: int
    win_rate_pct: Optional[float] = None
    rejected_signals: int
    rejected_by_reason: List[DailyRejectionReason] = Field(default_factory=list)
    broker_deal_count: int = 0
    pure_paper_close_count: int = 0
    pnl_basis: str = "broker_deals_preferred"
    drawdown_basis: str = "realized_pnl_sequence"


ReconciliationHealthState = Literal[
    "healthy",
    "degraded",
    "unavailable",
    "unresolved",
    "missing_local",
    "missing_broker",
]


class ReconciliationHealthItem(BaseModel):
    status: ReconciliationHealthState
    scope: Literal["local_tracker", "broker_position"]
    symbol: str
    direction: Literal["long", "short"]
    local_position_id: Optional[int] = None
    broker_position_id: Optional[int] = None
    intent_id: Optional[int] = None
    match_status: str
    message: str
    action_required: str


class ReconciliationHealthResponse(BaseModel):
    status: ReconciliationHealthState
    broker_truth_available: bool
    broker_execution_ready: bool
    checked_at: datetime
    local_managed_positions: int
    canonical_broker_positions: Optional[int] = None
    ignored_broker_positions: int = 0
    counts: Dict[str, int] = Field(default_factory=dict)
    items: List[ReconciliationHealthItem] = Field(default_factory=list)
    summary: str


ProtectionHealthState = Literal[
    "fully_protected",
    "partially_protected",
    "unprotected",
    "unavailable",
    "identity_unresolved",
]


class ProtectionHealthItem(BaseModel):
    status: ProtectionHealthState
    scope: Literal["local_tracker", "broker_position"]
    symbol: str
    direction: Literal["long", "short"]
    local_position_id: Optional[int] = None
    broker_position_id: Optional[int] = None
    intent_id: Optional[int] = None
    broker_sync_status: str
    broker_stop_loss: Optional[float] = None
    broker_take_profit: Optional[float] = None
    message: str
    action_required: str


class ProtectionHealthResponse(BaseModel):
    status: ProtectionHealthState
    broker_truth_available: bool
    broker_execution_ready: bool
    checked_at: datetime
    managed_positions: int
    assessable_positions: int
    fully_protected_positions: int
    full_protection_coverage_pct: Optional[float] = None
    ignored_broker_positions: int = 0
    counts: Dict[str, int] = Field(default_factory=dict)
    items: List[ProtectionHealthItem] = Field(default_factory=list)
    summary: str


class ConfluenceShadowRecord(BaseModel):
    id: int
    created_at: datetime
    analysis_created_at: datetime
    mode: Literal["shadow"] = "shadow"
    symbol: str
    timeframe: str
    strategy: str
    original_signal: SignalValue
    original_confidence: float
    shadow_signal: SignalValue
    shadow_confidence: float
    confidence_adjustment: float
    action: Literal["confirm", "reduce", "neutral", "context_only", "insufficient_evidence"]
    target_horizon: Literal["5m", "30m", "4h", "1d"]
    event_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    eligible_event_count: int = 0
    event_ids: List[int] = Field(default_factory=list)
    original_would_pass: bool
    shadow_would_pass: bool
    rationale: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    execution_unchanged: bool = True


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


class ActiveIncident(BaseModel):
    level: IncidentLevel
    code: str
    message: str


class EngineRuntime(BaseModel):
    running: bool = False
    loop_active: bool = False
    ollama_ready: bool = False
    last_cycle_at: Optional[datetime] = None
    last_cycle_completed_at: Optional[datetime] = None
    last_cycle_duration_ms: Optional[float] = Field(default=None, ge=0)
    last_cycle_summary: str = ""
    last_reconcile_at: Optional[datetime] = None
    last_reconcile_summary: str = ""
    last_error: Optional[str] = None
    tick_count: int = 0
    active_watchlist: List[str] = Field(default_factory=list)


class EngineCycleLatencyResponse(BaseModel):
    available: bool
    unit: Literal["ms"] = "ms"
    duration_ms: Optional[float] = None
    cycle_started_at: Optional[datetime] = None
    cycle_completed_at: Optional[datetime] = None
    tick_count: int = 0
    last_cycle_summary: str = ""
    message: str


LatencyObservationState = Literal["no_observation", "measured", "unavailable"]


class LatencyObservation(BaseModel):
    state: LatencyObservationState
    scope: Literal["broker_service_call", "api_request"]
    unit: Literal["ms"] = "ms"
    duration_ms: Optional[float] = Field(default=None, ge=0)
    operation: Optional[str] = None
    method: Optional[str] = None
    path: Optional[str] = None
    status_code: Optional[int] = None
    observed_at: Optional[datetime] = None
    detail: str


class BrokerApiLatencyResponse(BaseModel):
    broker: LatencyObservation
    api: LatencyObservation
    persistence: Literal["process_memory"] = "process_memory"
    message: str


ErrorEvidenceSource = Literal["incident", "order_intent_failure", "event_source_failure"]


class ErrorIncidentGroup(BaseModel):
    group_id: str
    source: ErrorEvidenceSource
    code: str
    level: IncidentLevel
    occurrences: int
    first_seen_at: datetime
    last_seen_at: datetime
    sample_message: str


class ErrorIncidentReportResponse(BaseModel):
    window_hours: int
    window_start_utc: datetime
    window_end_utc: datetime
    denominator_name: Literal[
        "terminal_order_intent_transitions_plus_event_source_runs"
    ] = "terminal_order_intent_transitions_plus_event_source_runs"
    denominator_count: int
    failure_evidence_count: int
    failure_evidence_rate_pct: Optional[float] = None
    terminal_intent_outcomes: int
    failed_intent_outcomes: int
    event_source_runs: int
    failed_event_source_runs: int
    incident_occurrences: int
    distinct_incident_groups: int
    distinct_group_count: int
    group_limit: int
    groups_truncated: bool
    groups: List[ErrorIncidentGroup] = Field(default_factory=list)
    message: str


class StrategyPerformanceSlice(BaseModel):
    symbol: str
    timeframe: str
    strategy: str
    account_currency: str
    trades: int
    realized_pnl: float
    average_realized_pnl: float
    wins: int
    losses: int
    breakeven: int
    win_rate_pct: Optional[float] = None
    broker_backed_trades: int
    pure_paper_trades: int
    broker_deal_count: int
    first_closed_at: datetime
    last_closed_at: datetime


class StrategyPerformanceResponse(BaseModel):
    window_days: int
    window_start_utc: datetime
    window_end_utc: datetime
    symbol_filter: Optional[str] = None
    timeframe_filter: Optional[str] = None
    strategy_filter: Optional[str] = None
    sample_count: int
    slice_count: int
    group_limit: int
    groups_truncated: bool
    account_currencies: List[str] = Field(default_factory=list)
    pnl_basis: Literal["broker_deals_preferred_per_closed_trade"] = (
        "broker_deals_preferred_per_closed_trade"
    )
    outcome_basis: Literal["one_closed_position_one_trade"] = (
        "one_closed_position_one_trade"
    )
    slices: List[StrategyPerformanceSlice] = Field(default_factory=list)
    message: str


JournalPnlBasis = Literal["broker_deals", "paper_estimate"]


class JournalExportRow(BaseModel):
    row_id: str
    local_position_id: int
    broker_position_id: Optional[int] = None
    broker_identity_status: Literal["resolved", "not_applicable", "conflict"]
    broker_identity_detail: str = ""
    broker_deal_ids: List[int] = Field(default_factory=list)
    audit_event_ids: List[int] = Field(default_factory=list)
    symbol: str
    timeframe: str
    strategy: str
    direction: Literal["long", "short"]
    opened_at_utc: datetime
    closed_at_utc: datetime
    entry_price: float
    exit_price: Optional[float] = None
    account_currency: str
    realized_pnl: float
    realized_pnl_basis: JournalPnlBasis
    broker_deal_count: int
    close_reason: Optional[str] = None


class JournalExportResponse(BaseModel):
    window_days: int
    window_start_utc: datetime
    window_end_utc: datetime
    row_count: int
    account_currencies: List[str] = Field(default_factory=list)
    rows: List[JournalExportRow] = Field(default_factory=list)
    message: str


class BrokerStatementRowInput(BaseModel):
    row_id: str = Field(..., min_length=1, max_length=200)
    broker_position_id: Optional[int] = Field(default=None, gt=0)
    deal_id: Optional[int] = Field(default=None, gt=0)
    symbol: str = Field(..., min_length=1, max_length=64)
    direction: Optional[Literal["long", "short"]] = None
    closed_at: datetime
    account_currency: str = Field(..., min_length=1, max_length=12)
    realized_pnl: float


class StatementComparisonRequest(BaseModel):
    rows: List[BrokerStatementRowInput] = Field(..., min_length=1, max_length=5000)
    window_days: int = Field(default=30, ge=1, le=365)
    pnl_tolerance: float = Field(default=0.01, ge=0.0, le=1000000.0)
    close_time_tolerance_seconds: int = Field(default=60, ge=0, le=86400)


StatementComparisonStatus = Literal[
    "matched",
    "mismatch",
    "local_only",
    "statement_only",
    "identity_unresolved",
]


class StatementComparisonItem(BaseModel):
    status: StatementComparisonStatus
    local_row_id: Optional[str] = None
    local_position_id: Optional[int] = None
    statement_row_ids: List[str] = Field(default_factory=list)
    broker_position_id: Optional[int] = None
    local_deal_ids: List[int] = Field(default_factory=list)
    statement_deal_ids: List[int] = Field(default_factory=list)
    symbol: Optional[str] = None
    local_currency: Optional[str] = None
    statement_currencies: List[str] = Field(default_factory=list)
    local_realized_pnl: Optional[float] = None
    statement_realized_pnl: Optional[float] = None
    pnl_difference: Optional[float] = None
    local_closed_at_utc: Optional[datetime] = None
    statement_closed_at_utc: Optional[datetime] = None
    mismatch_reasons: List[str] = Field(default_factory=list)


class StatementComparisonResponse(BaseModel):
    window_days: int
    window_start_utc: datetime
    window_end_utc: datetime
    local_row_count: int
    statement_row_count: int
    matched: int
    mismatched: int
    local_only: int
    statement_only: int
    identity_unresolved: int
    pnl_tolerance: float
    close_time_tolerance_seconds: int
    items: List[StatementComparisonItem] = Field(default_factory=list)
    message: str


class EngineStatus(BaseModel):
    version: str
    mode: Literal["paper_only", "demo_enabled", "live_enabled"]
    broker: BrokerStatus
    config: EngineConfig
    runtime: EngineRuntime
    readiness: List[ReadinessCheck] = Field(default_factory=list)
    status_truth: List[ReadinessCheck] = Field(default_factory=list)
    active_incidents: List[ActiveIncident] = Field(default_factory=list)
    strategies: List[StrategyInfo] = Field(default_factory=list)
    recent_incidents: List[IncidentRecord] = Field(default_factory=list)
    recent_analyses: List[StrategyAnalysis] = Field(default_factory=list)
    paper_positions: List[PaperPosition] = Field(default_factory=list)
    recent_events: List[PaperEvent] = Field(default_factory=list)
    recent_order_intents: List[OrderIntentRecord] = Field(default_factory=list)
    recent_trade_audits: List[TradeAuditRecord] = Field(default_factory=list)
    recent_decisions: List[DecisionRecord] = Field(default_factory=list)
    recent_confluence_shadows: List[ConfluenceShadowRecord] = Field(default_factory=list)


MarketBias = Literal["bullish", "bearish", "neutral", "unknown"]
MarketRegime = Literal["risk_on", "risk_off", "mixed", "unknown"]


class MarketIntelligenceDriver(BaseModel):
    label: str
    detail: str
    impact: Literal["bullish", "bearish", "neutral"]


class MarketIntelligenceInstrument(BaseModel):
    symbol: str
    timeframe: str
    name: str
    category: str
    price: Optional[float] = None
    change: Optional[float] = None
    change_pct: Optional[float] = None
    low_range: Optional[float] = None
    high_range: Optional[float] = None
    bias: MarketBias = "unknown"
    confidence: float = 0.0
    situation: str = ""
    direction_note: str = ""
    drivers: List[MarketIntelligenceDriver] = Field(default_factory=list)
    support: List[float] = Field(default_factory=list)
    resistance: List[float] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=utcnow_naive)
    data_status: Literal["live", "cached", "unavailable"] = "unavailable"
    error: Optional[str] = None


class MarketIntelligenceMacroEvent(BaseModel):
    title: str
    impact: Literal["high", "medium", "low", "unknown"] = "unknown"
    source: str = "system"
    ts: Optional[int] = None


class MarketEventInput(BaseModel):
    source: str = Field(..., min_length=1, max_length=120)
    title: str = Field(..., min_length=1, max_length=1000)
    summary: str = Field(default="", max_length=10_000)
    url: str = Field(default="", max_length=4000)
    published_at: Optional[datetime] = None
    symbols: List[str] = Field(default_factory=list)
    raw: Dict[str, Any] = Field(default_factory=dict)


class MarketEventRecord(BaseModel):
    id: int
    content_hash: str
    source: str
    title: str
    summary: str = ""
    url: str = ""
    published_at: Optional[datetime] = None
    ingested_at: datetime
    symbols: List[str] = Field(default_factory=list)
    event_type: str = "general"
    sentiment: Literal["bullish", "bearish", "neutral", "mixed"] = "neutral"
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    impact: Literal["high", "medium", "low", "unknown"] = "unknown"
    horizon: Literal["immediate", "intraday", "swing", "long_term", "unknown"] = "unknown"
    credibility_score: float = Field(default=0.5, ge=0.0, le=1.0)
    classification_version: str = "deterministic-v1"
    raw: Dict[str, Any] = Field(default_factory=dict)


class EventAlertRecord(BaseModel):
    id: int
    alert_key: str
    created_at: datetime
    alert_type: str
    symbol: str
    severity: Literal["info", "warning", "critical"] = "warning"
    summary: str
    details: Dict[str, Any] = Field(default_factory=dict)


class EventRefreshResponse(BaseModel):
    ok: bool
    configured_sources: int = 0
    fetched_items: int = 0
    inserted_events: int = 0
    duplicate_events: int = 0
    alerts_created: int = 0
    errors: List[str] = Field(default_factory=list)


class EventOutcomeRecord(BaseModel):
    id: int
    event_id: int
    symbol: str
    market_symbol: str
    horizon: Literal["5m", "30m", "4h", "1d"]
    status: Literal["pending", "evaluated", "unavailable"]
    reference_at: Optional[datetime] = None
    target_at: Optional[datetime] = None
    reference_price: Optional[float] = None
    target_price: Optional[float] = None
    forward_return_pct: Optional[float] = None
    max_favorable_excursion_pct: Optional[float] = None
    max_adverse_excursion_pct: Optional[float] = None
    predicted_direction: Literal["bullish", "bearish", "neutral", "mixed"] = "neutral"
    realized_direction: Literal["up", "down", "flat", "unknown"] = "unknown"
    direction_hit: Optional[bool] = None
    brier_score: Optional[float] = None
    threshold_pct: float = 0.0
    reason: str = ""
    computed_at: datetime


class EventCalibrationGroup(BaseModel):
    event_type: str
    horizon: Literal["5m", "30m", "4h", "1d"]
    samples: int = 0
    hit_rate: Optional[float] = None
    average_return_pct: Optional[float] = None
    median_return_pct: Optional[float] = None
    average_brier_score: Optional[float] = None
    average_favorable_excursion_pct: Optional[float] = None
    average_adverse_excursion_pct: Optional[float] = None
    gate: Literal["insufficient_samples", "observe", "eligible", "degraded"] = "insufficient_samples"
    gate_reason: str = ""


class EventCalibrationResponse(BaseModel):
    generated_at: datetime = Field(default_factory=utcnow_naive)
    research_only: bool = True
    methodology_version: str = "event-calibration-v1"
    methodology: Dict[str, Any] = Field(default_factory=dict)
    evaluated_outcomes: int = 0
    pending_outcomes: int = 0
    unavailable_outcomes: int = 0
    minimum_samples: int = 30
    groups: List[EventCalibrationGroup] = Field(default_factory=list)


class EventCalibrationRunResponse(BaseModel):
    ok: bool
    events_checked: int = 0
    outcomes_evaluated: int = 0
    outcomes_pending: int = 0
    outcomes_unavailable: int = 0
    errors: List[str] = Field(default_factory=list)


class ConfluenceReplayMetrics(BaseModel):
    candidate_decisions: int = 0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate_pct: float = 0.0
    expectancy_pct: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: Optional[float] = None
    trades_per_day: float = 0.0


class ConfluenceReplayResponse(BaseModel):
    generated_at: datetime = Field(default_factory=utcnow_naive)
    research_only: bool = True
    methodology: str
    total_records: int = 0
    priced_records: int = 0
    pending_records: int = 0
    unavailable_records: int = 0
    fee_bps_per_side: float = 0.0
    original: ConfluenceReplayMetrics
    shadow: ConfluenceReplayMetrics
    deltas: Dict[str, float] = Field(default_factory=dict)
    verdict: Literal["insufficient_data", "keep_shadow", "candidate_for_review"] = "insufficient_data"
    verdict_reason: str
    warnings: List[str] = Field(default_factory=list)


class MarketIntelligenceResponse(BaseModel):
    generated_at: datetime = Field(default_factory=utcnow_naive)
    regime: MarketRegime = "unknown"
    headline: str = ""
    summary: str = ""
    instruments: List[MarketIntelligenceInstrument] = Field(default_factory=list)
    macro_events: List[MarketIntelligenceMacroEvent] = Field(default_factory=list)
    market_events: List[MarketEventRecord] = Field(default_factory=list)
    event_alerts: List[EventAlertRecord] = Field(default_factory=list)
    source_notes: List[str] = Field(default_factory=list)
