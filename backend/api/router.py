from __future__ import annotations

from threading import Lock
from time import monotonic
from typing import List

from fastapi import APIRouter, HTTPException

from backend.adapters.ctrader import adapter as broker_adapter
from backend.config import SETTINGS
from backend.domain.models import (
    AnalyzeRequest,
    EngineConfig,
    EngineStatus,
    ManualOrderRequest,
    SymbolLimits,
    StudioTaskRequest,
    StudioTaskResponse,
    StrategyAnalysis,
    StrategyInfo,
    WatchlistItem,
)
from backend.services.broker import get_broker_status, get_symbol_limits, list_positions, list_symbols
from backend.services.checklist_feed import get_auto_checklist, get_calendar_next
from backend.services.engine import engine
from backend.services.execution_engine import execute_paper_signal
from backend.services.market_data import MarketDataError, get_bars, get_market_data_status
from backend.services.reconciler import reconcile_open_positions, recover_runtime_state
from backend.services.risk import build_readiness
from backend.services import model_service
from backend.services import studio_llm
from backend.services.studio_backtests import list_saved_strategy_files, run_saved_strategy_backtest
from backend.services.studio_tasks import execute_studio_task
from backend.storage.repositories import (
    add_analysis,
    list_incidents,
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
    ready = bool(llm_result.get("ok"))

    with _llm_ready_lock:
        _llm_ready_cache = ready
        _llm_ready_expires_at = monotonic() + _LLM_READY_TTL_SEC

    return ready


async def _status_payload() -> EngineStatus:
    config = _current_config()
    broker = get_broker_status()
    strategies = [strategy.info() for strategy in list_strategies()]
    readiness = build_readiness(config)
    runtime = load_runtime()
    runtime.ollama_ready = await _get_cached_ollama_ready()
    
    return EngineStatus(
        version=SETTINGS.version,
        mode="live_enabled" if config.allow_live and not config.kill_switch else "paper_only",
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
    saved = save_engine_config(config)
    if saved.allow_live:
        log_incident(
            level="warning",
            code="live_mode_request",
            message="Live mode was requested, but execution remains disabled until the new engine is built.",
            details=saved.model_dump(),
        )
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


@router.get("/market/status")
async def v2_market_status(symbol: str | None = None, timeframe: str | None = None) -> dict:
    config = _current_config()
    probe_symbol = (symbol or config.default_symbol).upper()
    probe_timeframe = (timeframe or config.default_timeframe).upper()
    return get_market_data_status(probe_symbol, probe_timeframe)


@router.get("/market/candles")
async def v2_market_candles(
    symbol: str,
    timeframe: str = "M5",
    num_bars: int = 5000,
) -> dict:
    try:
        df = get_bars(symbol.upper(), timeframe.upper(), num_bars)
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


@router.get("/checklist/auto")
async def v2_checklist_auto(tf: str = "M5", structure_tf: str = "H1") -> dict:
    return get_auto_checklist(tf=tf, structure_tf=structure_tf)


@router.get("/calendar/next")
async def v2_calendar_next() -> dict:
    return get_calendar_next()


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
):
    return run_saved_strategy_backtest(
        strategy=strategy,
        symbol=symbol,
        timeframe=timeframe,
        num_bars=num_bars,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )


@router.post("/studio/tasks", response_model=StudioTaskResponse)
async def v2_studio_tasks(request: StudioTaskRequest) -> StudioTaskResponse:
    return await execute_studio_task(request)


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
    return {"ok": True, **summary}


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
    watch_item = WatchlistItem(
        symbol=request.symbol.upper(),
        timeframe=request.timeframe.upper(),
        strategy=request.strategy,
        enabled=True,
        params={},
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
        "mode": "paper_only",
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
    return saved
