from __future__ import annotations

import asyncio
from threading import Lock
from time import monotonic
from typing import List

from fastapi import APIRouter, HTTPException

from backend.adapters.ctrader import adapter as broker_adapter
from backend.calendar import get_next_event
from backend.config import SETTINGS
from backend.domain.models import (
    AnalyzeRequest,
    ConfluenceReplayResponse,
    EngineConfig,
    EngineStatus,
    EventRefreshResponse,
    EventCalibrationResponse,
    EventCalibrationRunResponse,
    InstrumentSpec,
    ManualOrderRequest,
    MarketIntelligenceResponse,
    MarketEventInput,
    SymbolLimits,
    StudioTaskRequest,
    StudioTaskResponse,
    StrategyLifecyclePromoteRequest,
    StrategyLifecycleUpdateRequest,
    StrategyAnalysis,
    StrategyInfo,
    WatchlistItem,
)
from backend.services.broker import get_broker_status, get_instrument_spec, get_symbol_limits, list_positions, list_symbols
from backend.services.engine import engine
from backend.services.execution_engine import execute_paper_signal
from backend.services.confluence_shadow import record_confluence_shadow
from backend.services.confluence_replay import run_confluence_replay
from backend.services.event_intelligence import detect_abnormal_events, ingest_events, refresh_configured_feeds
from backend.services.event_calibration import build_event_calibration, calibrate_pending_event_outcomes
from backend.services.market_data import MarketDataError, get_bars, get_market_data_status
from backend.services.market_intelligence import build_market_intelligence
from backend.services.reconciler import reconcile_open_positions, recover_runtime_state
from backend.services.broker_ledger import reconcile_closed_demo_history
from backend.services.risk import build_readiness
from backend.services import model_service
from backend.services import studio_llm
from backend.services.studio_backtests import list_saved_strategy_files, run_saved_strategy_backtest
from backend.services.studio_tasks import execute_studio_task
from backend.services.strategy_lifecycle import (
    StrategyLifecycleError,
    get_lifecycle,
    list_lifecycles,
    promote,
    record_paper_evidence,
    retire,
    update_hypothesis,
)
from backend.storage.repositories import (
    add_analysis,
    list_decision_records,
    list_confluence_shadows,
    list_incidents,
    list_event_alerts,
    list_event_outcomes,
    list_market_events,
    list_order_intent_transitions,
    list_order_intents,
    list_paper_events,
    list_paper_positions,
    list_recent_analyses,
    list_trade_audits,
    load_engine_config,
    load_runtime,
    log_incident,
    save_engine_config,
)
from backend.strategies.registry import get_strategy, list_strategies


router = APIRouter(prefix="/api", tags=["tradeagent"])

_DEFAULT_CONFIG = EngineConfig()
_LLM_READY_TTL_SEC = 10.0
_llm_ready_lock = Lock()
_llm_ready_cache: bool | None = None
_llm_ready_expires_at = 0.0


def _current_config() -> EngineConfig:
    return load_engine_config(_DEFAULT_CONFIG)


async def _get_cached_ollama_ready() -> bool:
    global _llm_ready_cache, _llm_ready_expires_at

    now = monotonic()
    with _llm_ready_lock:
        if _llm_ready_cache is not None and now < _llm_ready_expires_at:
            return _llm_ready_cache

    llm_result = await model_service.fetch_tags(timeout=1.0)
    models = llm_result.get("models") if isinstance(llm_result.get("models"), list) else []
    ready = bool(llm_result.get("ok")) and model_service.is_model_available(models)

    with _llm_ready_lock:
        _llm_ready_cache = ready
        _llm_ready_expires_at = monotonic() + _LLM_READY_TTL_SEC

    return ready


async def _status_payload() -> EngineStatus:
    config = _current_config()
    broker = await asyncio.to_thread(get_broker_status)
    strategies = [strategy.info() for strategy in list_strategies()]
    readiness = await asyncio.to_thread(build_readiness, config)
    runtime = load_runtime()
    runtime.ollama_ready = await _get_cached_ollama_ready()
    
    return EngineStatus(
        version=SETTINGS.version,
        mode="demo_enabled" if config.demo_autotrade and broker.execution_ready else "paper_only",
        broker=broker,
        config=config,
        runtime=runtime,
        readiness=readiness,
        strategies=strategies,
        recent_incidents=list_incidents(8),
        recent_analyses=list_recent_analyses(8),
        paper_positions=list_paper_positions("open"),
        recent_events=list_paper_events(8),
        recent_order_intents=list_order_intents(8),
        recent_trade_audits=list_trade_audits(8),
        recent_decisions=list_decision_records(8),
        recent_confluence_shadows=list_confluence_shadows(8),
    )


@router.get("/health")
async def health() -> dict:
    status = await _status_payload()
    ready = status.broker.ready
    return {
        "status": "ok",
        "version": SETTINGS.version,
        "paper_ready": all(item.ok for item in status.readiness if item.name != "live_permission"),
        "mode": status.mode,
        "broker_ready": ready,
        "connected": status.broker.socket_connected,
        "authorized": status.broker.account_authorized,
        "market_data": status.broker.market_data_ready,
    }


@router.get("/llm_status")
async def llm_status() -> dict:
    return await model_service.status_payload()


@router.get("/status", response_model=EngineStatus)
async def v2_status() -> EngineStatus:
    return await _status_payload()


@router.get("/config", response_model=EngineConfig)
async def v2_get_config() -> EngineConfig:
    return _current_config()


@router.post("/config", response_model=EngineConfig)
async def v2_set_config(config: EngineConfig) -> EngineConfig:
    if config.allow_live:
        raise HTTPException(
            status_code=400,
            detail="Live-account execution is not supported. Connect and verify a cTrader demo account instead.",
        )
    saved = save_engine_config(config)
    engine.wake()
    return saved


@router.get("/strategies", response_model=List[StrategyInfo])
async def v2_strategies() -> List[StrategyInfo]:
    return [strategy.info() for strategy in list_strategies()]


@router.get("/incidents")
async def v2_incidents(limit: int = 20) -> list:
    limit = max(1, min(100, limit))
    return [record.model_dump() for record in list_incidents(limit)]


@router.get("/positions")
async def v2_positions() -> list:
    return list_positions()


@router.get("/symbols")
async def v2_symbols() -> dict:
    import time
    start = time.perf_counter()
    config = _current_config()
    all_symbols = list_symbols()
    
    # Prioritize 'Major' symbols for better UX in large lists
    majors = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "US30", "US500", "NAS100", "USDOLLAR", "XAGUSD", "WTI", "BRENT"]
    # Filter for majors that actually exist in the broker list
    prioritized = [s for s in majors if s in all_symbols]
    # Rest of the symbols sorted alphabetically, excluding duplicates already in prioritized
    others = sorted([s for s in all_symbols if s not in prioritized])
    
    symbols = prioritized + others
    default = config.default_symbol.upper() if config.default_symbol.upper() in symbols else (symbols[0] if symbols else None)
    
    elapsed = time.perf_counter() - start
    print(f"[API] symbols fetched in {elapsed:.4f}s (count={len(symbols)})")
    return {"symbols": symbols, "default": default}


@router.get("/symbol-limits", response_model=SymbolLimits)
async def v2_symbol_limits(symbol: str) -> SymbolLimits:
    if not (symbol or "").strip():
        raise HTTPException(status_code=400, detail="Symbol is required.")
    return get_symbol_limits(symbol.upper())


@router.get("/instrument-spec", response_model=InstrumentSpec)
async def v2_instrument_spec(symbol: str) -> InstrumentSpec:
    if not (symbol or "").strip():
        raise HTTPException(status_code=400, detail="Symbol is required.")
    return get_instrument_spec(symbol.upper(), _current_config().account_currency)


@router.get("/market/status")
async def v2_market_status(symbol: str | None = None, timeframe: str | None = None) -> dict:
    config = _current_config()
    probe_symbol = (symbol or config.default_symbol).upper()
    probe_timeframe = (timeframe or config.default_timeframe).upper()
    return await asyncio.to_thread(get_market_data_status, probe_symbol, probe_timeframe)


@router.get("/market/intelligence", response_model=MarketIntelligenceResponse)
async def v2_market_intelligence() -> MarketIntelligenceResponse:
    return await asyncio.to_thread(build_market_intelligence, _current_config())


@router.get("/market/events")
async def v2_market_events(limit: int = 50, symbol: str | None = None) -> list:
    limit = max(1, min(200, limit))
    return [event.model_dump(mode="json") for event in list_market_events(limit, symbol)]


@router.post("/market/events/ingest", response_model=EventRefreshResponse)
async def v2_ingest_market_events(items: list[MarketEventInput]) -> EventRefreshResponse:
    if len(items) > 200:
        raise HTTPException(status_code=400, detail="At most 200 events may be ingested per request.")
    inserted, duplicates = ingest_events(items)
    alerts = detect_abnormal_events(inserted)
    return EventRefreshResponse(
        ok=True,
        fetched_items=len(items),
        inserted_events=len(inserted),
        duplicate_events=duplicates,
        alerts_created=alerts,
    )


@router.post("/market/events/refresh", response_model=EventRefreshResponse)
async def v2_refresh_market_events() -> EventRefreshResponse:
    return await asyncio.to_thread(refresh_configured_feeds)


@router.get("/market/event-alerts")
async def v2_event_alerts(limit: int = 20) -> list:
    limit = max(1, min(100, limit))
    return [alert.model_dump(mode="json") for alert in list_event_alerts(limit)]


@router.post("/market/events/calibrate", response_model=EventCalibrationRunResponse)
async def v2_calibrate_market_events(event_limit: int = 200) -> EventCalibrationRunResponse:
    event_limit = max(1, min(2000, event_limit))
    return await asyncio.to_thread(calibrate_pending_event_outcomes, event_limit)


@router.get("/market/events/outcomes")
async def v2_market_event_outcomes(limit: int = 200, status: str | None = None) -> list:
    limit = max(1, min(2000, limit))
    return [outcome.model_dump(mode="json") for outcome in list_event_outcomes(limit, status)]


@router.get("/market/events/calibration", response_model=EventCalibrationResponse)
async def v2_market_event_calibration() -> EventCalibrationResponse:
    return build_event_calibration()


@router.get("/market/confluence-shadow")
async def v2_confluence_shadow(limit: int = 50, symbol: str | None = None) -> list:
    limit = max(1, min(500, limit))
    return [record.model_dump(mode="json") for record in list_confluence_shadows(limit, symbol)]


@router.get("/market/confluence-shadow/replay", response_model=ConfluenceReplayResponse)
async def v2_confluence_shadow_replay(
    limit: int = 1000,
    symbol: str | None = None,
    fee_bps_per_side: float = 0.0,
    num_bars: int = 5000,
) -> ConfluenceReplayResponse:
    limit = max(1, min(5000, limit))
    num_bars = max(100, min(5000, num_bars))
    fee_bps_per_side = max(0.0, min(100.0, fee_bps_per_side))
    return await asyncio.to_thread(
        run_confluence_replay,
        limit=limit,
        symbol=(symbol or "").strip().upper() or None,
        fee_bps_per_side=fee_bps_per_side,
        num_bars=num_bars,
    )


@router.get("/market/candles")
async def v2_market_candles(
    symbol: str,
    timeframe: str = "M5",
    num_bars: int = 5000,
    live: bool = False,
) -> dict:
    try:
        df = await asyncio.to_thread(get_bars, symbol.upper(), timeframe.upper(), num_bars, prefer_live=live)
    except MarketDataError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    candles = [
        {
            "time": int(ts.timestamp()),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
        }
        for ts, row in df[["open", "high", "low", "close"]].iterrows()
    ]
    return {"candles": candles, "indicators": {}}




@router.get("/calendar/next")
async def v2_calendar_next() -> dict:
    try:
        event = get_next_event()
    except Exception as exc:
        return {"ts": None, "title": None, "impact": "unknown", "source": f"error: {exc}"}
    return event or {"ts": None, "title": None, "impact": "unknown", "source": None}


@router.get("/models")
async def v2_models() -> dict:
    return await model_service.models_payload()


@router.get("/studio/models")
async def v2_studio_models(provider: str | None = None) -> dict:
    return await studio_llm.models_payload(provider=provider)


@router.get("/studio/strategy-files")
async def v2_studio_strategy_files() -> dict:
    return list_saved_strategy_files()


@router.get("/studio/backtest")
async def v2_studio_backtest(
    strategy: str,
    symbol: str,
    timeframe: str = "M5",
    num_bars: int = 1500,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    validation_kind: str = "development_backtest",
):
    kwargs = dict(
        strategy=strategy,
        symbol=symbol,
        timeframe=timeframe,
        num_bars=num_bars,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    if validation_kind != "development_backtest":
        kwargs["validation_kind"] = validation_kind
    return run_saved_strategy_backtest(**kwargs)


@router.post("/studio/tasks", response_model=StudioTaskResponse)
async def v2_studio_tasks(request: StudioTaskRequest) -> StudioTaskResponse:
    return await execute_studio_task(request)



@router.get("/studio/lifecycles")
async def v2_studio_lifecycles() -> list:
    return [item.model_dump(mode="json") for item in list_lifecycles()]


@router.get("/studio/lifecycle/{strategy}")
async def v2_studio_lifecycle(strategy: str) -> dict:
    try:
        return get_lifecycle(strategy).model_dump(mode="json")
    except StrategyLifecycleError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.put("/studio/lifecycle/{strategy}")
async def v2_update_studio_lifecycle(strategy: str, request: StrategyLifecycleUpdateRequest) -> dict:
    try:
        return update_hypothesis(strategy, request.hypothesis).model_dump(mode="json")
    except StrategyLifecycleError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/studio/lifecycle/{strategy}/promote")
async def v2_promote_studio_lifecycle(strategy: str, request: StrategyLifecyclePromoteRequest) -> dict:
    try:
        return promote(strategy, request.operator, request.reason).model_dump(mode="json")
    except StrategyLifecycleError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/studio/lifecycle/{strategy}/paper-evidence")
async def v2_studio_paper_evidence(strategy: str) -> dict:
    try:
        return record_paper_evidence(strategy).model_dump(mode="json")
    except StrategyLifecycleError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/studio/lifecycle/{strategy}/retire")
async def v2_retire_studio_lifecycle(strategy: str, request: StrategyLifecyclePromoteRequest) -> dict:
    try:
        return retire(strategy, request.operator, request.reason).model_dump(mode="json")
    except StrategyLifecycleError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/paper/positions")
async def v2_paper_positions(status: str | None = None) -> list:
    if status:
        return [item.model_dump(mode="json") for item in list_paper_positions(status)]
    return [item.model_dump(mode="json") for item in list_paper_positions()]


@router.get("/paper/events")
async def v2_paper_events(limit: int = 20) -> list:
    limit = max(1, min(100, limit))
    return [item.model_dump(mode="json") for item in list_paper_events(limit)]


@router.get("/paper/order-intents")
async def v2_order_intents(limit: int = 20) -> list:
    limit = max(1, min(100, limit))
    return [item.model_dump(mode="json") for item in list_order_intents(limit)]


@router.get("/paper/order-intents/{intent_id}/transitions")
async def v2_order_intent_transitions(intent_id: int) -> list:
    return [item.model_dump(mode="json") for item in list_order_intent_transitions(intent_id)]


@router.get("/decisions")
async def v2_decisions(limit: int = 20) -> list:
    limit = max(1, min(100, limit))
    return [item.model_dump(mode="json") for item in list_decision_records(limit)]


@router.get("/paper/audit")
async def v2_trade_audit(limit: int = 20) -> list:
    limit = max(1, min(100, limit))
    return [item.model_dump(mode="json") for item in list_trade_audits(limit)]


@router.post("/engine/start")
async def v2_engine_start() -> dict:
    config = _current_config()
    config.enabled = True
    saved = save_engine_config(config)
    engine.wake()
    return {"ok": True, "enabled": saved.enabled}


@router.post("/engine/stop")
async def v2_engine_stop() -> dict:
    config = _current_config()
    config.enabled = False
    saved = save_engine_config(config)
    engine.wake()
    return {"ok": True, "enabled": saved.enabled}


@router.post("/engine/scan")
async def v2_engine_scan() -> dict:
    summary = await engine.run_once()
    return {"ok": True, "summary": summary}


@router.post("/engine/reconcile")
async def v2_engine_reconcile() -> dict:
    summary = reconcile_open_positions(reason="manual")
    history = reconcile_closed_demo_history(limit=100) if _current_config().demo_autotrade else {
        "checked": 0,
        "reconciled": 0,
        "missing_broker_id": 0,
        "unavailable": 0,
    }
    return {"ok": True, **summary, "closed_history": history}


@router.post("/engine/recover")
async def v2_engine_recover() -> dict:
    recovered = recover_runtime_state(_current_config())
    return {"ok": True, **recovered}


@router.post("/watchlist")
async def v2_set_watchlist(watchlist: List[WatchlistItem]) -> dict:
    config = _current_config()
    config.watchlist = watchlist
    save_engine_config(config)
    engine.wake()
    return {"ok": True, "count": len(watchlist)}


@router.post("/orders/manual")
async def v2_manual_order(request: ManualOrderRequest) -> dict:
    config = _current_config()
    analysis = StrategyAnalysis(
        symbol=request.symbol.upper(),
        timeframe=request.timeframe.upper(),
        strategy=request.strategy,
        signal=request.signal,
        confidence=request.confidence,
        entry_price=request.entry_price,
        stop_loss=request.stop_loss,
        take_profit=request.take_profit,
        reasons=request.reasons or ([request.rationale] if request.rationale else []),
        context={"source": "manual_dashboard"},
    )
    configured_item = next(
        (
            item
            for item in config.watchlist
            if item.symbol.upper() == request.symbol.upper()
            and item.timeframe.upper() == request.timeframe.upper()
        ),
        None,
    )
    watch_item = (
        configured_item.model_copy(
            update={
                "symbol": request.symbol.upper(),
                "timeframe": request.timeframe.upper(),
                "strategy": request.strategy,
            }
        )
        if configured_item
        else WatchlistItem(
            symbol=request.symbol.upper(),
            timeframe=request.timeframe.upper(),
            strategy=request.strategy,
            enabled=True,
            trading_enabled=False,
            lot_size=request.quantity,
            params={},
        )
    )
    mark_price = request.entry_price
    mark_timestamp = None
    bar_snapshot = None
    if mark_price is None:
        try:
            market_status = get_market_data_status(request.symbol.upper(), request.timeframe.upper())
            if market_status.get("ok"):
                df = get_bars(request.symbol.upper(), request.timeframe.upper(), 5)
                mark_price = float(df["close"].iloc[-1])
                mark_timestamp = df.index[-1].to_pydatetime()
                bar_snapshot = {
                    "open": float(df["open"].iloc[-1]),
                    "high": float(df["high"].iloc[-1]),
                    "low": float(df["low"].iloc[-1]),
                    "close": float(df["close"].iloc[-1]),
                }
        except Exception:
            mark_price = None
    if mark_price is None:
        raise HTTPException(status_code=503, detail="Unable to resolve a mark price for the manual order.")

    result = execute_paper_signal(
        config=config,
        watch_item=watch_item,
        analysis=analysis,
        mark_price=float(mark_price),
        bar_timestamp=mark_timestamp,
        bar_snapshot=bar_snapshot,
        quantity=request.quantity,
        source="manual",
    )
    return {
        "ok": True,
        "intent_id": result.intent_id,
        "status": result.status,
        "summary": result.summary,
        "position_id": result.position_id,
        "broker_position_id": result.broker_position_id,
        "mode": result.mode,
    }


@router.post("/analyze", response_model=StrategyAnalysis)
async def v2_analyze(request: AnalyzeRequest) -> StrategyAnalysis:
    config = _current_config()
    try:
        strategy = get_strategy(request.strategy)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        df = get_bars(request.symbol.upper(), request.timeframe.upper(), request.num_bars)
    except MarketDataError as exc:
        log_incident(
            level="warning",
            code="market_data_unavailable",
            message=str(exc),
            details=request.model_dump(),
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    analysis = strategy.analyze(
        df=df,
        symbol=request.symbol.upper(),
        timeframe=request.timeframe.upper(),
        params=request.params,
    )
    analysis.context.setdefault("engine_mode", "paper_only")
    analysis.context.setdefault("kill_switch", config.kill_switch)
    saved = add_analysis(analysis)
    try:
        record_confluence_shadow(saved, config.min_confidence)
    except Exception as exc:
        log_incident(
            "warning",
            "confluence_shadow_failed",
            f"Shadow confluence failed for {saved.symbol}:{saved.timeframe}",
            {"error": str(exc), "execution_unchanged": True},
        )
    return saved
