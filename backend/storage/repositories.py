from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.domain.models import (
    DecisionRecord,
    ConfluenceShadowRecord,
    EngineConfig,
    EngineRuntime,
    EventAlertRecord,
    EventOutcomeRecord,
    IncidentRecord,
    MarketEventRecord,
    OrderIntentRecord,
    OrderIntentTransitionRecord,
    PaperEvent,
    PaperPosition,
    StrategyAnalysis,
    TradeAuditRecord,
)
from backend.storage.db import get_db


_CONFIG_KEY = "engine_config"
_RUNTIME_KEY = "engine_runtime"
_BAR_STATE_KEY = "bar_state"


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _load_json_state(key: str, default: Any) -> Any:
    with get_db() as db:
        row = db.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
    if not row:
        return default
    try:
        return json.loads(row["value"])
    except Exception:
        return default


def _save_json_state(key: str, payload: Any) -> None:
    now = _utcnow().isoformat()
    with get_db() as db:
        db.execute(
            """
            INSERT INTO state(key, value, updated_at)
            VALUES(?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, json.dumps(payload, ensure_ascii=False), now),
        )
        db.commit()


def load_engine_config(defaults: EngineConfig) -> EngineConfig:
    with get_db() as db:
        row = db.execute("SELECT value FROM config WHERE key = ?", (_CONFIG_KEY,)).fetchone()
    if not row:
        return defaults
    try:
        payload = json.loads(row["value"])
        config = EngineConfig(**payload)
        for item in config.watchlist:
            if item.lot_size is None:
                item.lot_size = config.paper_trade_size
        return config
    except Exception:
        return defaults


def save_engine_config(config: EngineConfig) -> EngineConfig:
    payload = json.dumps(config.model_dump(), ensure_ascii=False)
    now = _utcnow().isoformat()
    with get_db() as db:
        db.execute(
            """
            INSERT INTO config(key, value, updated_at)
            VALUES(?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (_CONFIG_KEY, payload, now),
        )
        db.commit()
    return config


def load_runtime(defaults: Optional[EngineRuntime] = None) -> EngineRuntime:
    if defaults is None:
        defaults = EngineRuntime()
    payload = _load_json_state(_RUNTIME_KEY, defaults.model_dump())
    try:
        return EngineRuntime(**payload)
    except Exception:
        return defaults


def save_runtime(runtime: EngineRuntime) -> EngineRuntime:
    _save_json_state(_RUNTIME_KEY, runtime.model_dump(mode="json"))
    return runtime


def load_bar_state() -> Dict[str, int]:
    payload = _load_json_state(_BAR_STATE_KEY, {})
    return payload if isinstance(payload, dict) else {}


def save_bar_state(state: Dict[str, int]) -> Dict[str, int]:
    _save_json_state(_BAR_STATE_KEY, state)
    return state


def load_cached_market_bars(symbol: str, timeframe: str, limit: int) -> tuple[pd.DataFrame, datetime | None]:
    sym = (symbol or "").strip().upper()
    tf = (timeframe or "").strip().upper()
    if not sym or not tf or limit <= 0:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]), None

    with get_db() as db:
        rows = db.execute(
            """
            SELECT bar_time, open, high, low, close, volume, fetched_at
            FROM market_bars
            WHERE symbol = ? AND timeframe = ?
            ORDER BY bar_time DESC
            LIMIT ?
            """,
            (sym, tf, int(limit)),
        ).fetchall()

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]), None

    ordered = list(reversed(rows))
    fetched_at: datetime | None = None
    data: list[dict[str, object]] = []
    for row in ordered:
        if fetched_at is None and row["fetched_at"]:
            try:
                fetched_at = datetime.fromisoformat(str(row["fetched_at"]))
            except Exception:
                fetched_at = None
        data.append(
            {
                "time": str(row["bar_time"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        )

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    return df[["open", "high", "low", "close", "volume"]], fetched_at


def upsert_market_bars(symbol: str, timeframe: str, df: pd.DataFrame) -> None:
    sym = (symbol or "").strip().upper()
    tf = (timeframe or "").strip().upper()
    if not sym or not tf or df is None or df.empty:
        return

    fetched_at = _utcnow().isoformat()
    payload: list[tuple[object, ...]] = []
    for ts, row in df[["open", "high", "low", "close", "volume"]].iterrows():
        if pd.isna(ts):
            continue
        ts_value = pd.Timestamp(ts)
        if ts_value.tzinfo is None:
            ts_value = ts_value.tz_localize(UTC)
        else:
            ts_value = ts_value.tz_convert(UTC)
        payload.append(
            (
                sym,
                tf,
                ts_value.isoformat(),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
                fetched_at,
            )
        )

    if not payload:
        return

    with get_db() as db:
        db.executemany(
            """
            INSERT INTO market_bars(symbol, timeframe, bar_time, open, high, low, close, volume, fetched_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, timeframe, bar_time) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume,
                fetched_at = excluded.fetched_at
            """,
            payload,
        )
        db.commit()


def log_incident(level: str, code: str, message: str, details: Dict[str, Any] | None = None) -> None:
    now = _utcnow().isoformat()
    details_json = json.dumps(details or {}, ensure_ascii=False)
    with get_db() as db:
        db.execute(
            """
            INSERT INTO incidents(created_at, level, code, message, details_json)
            VALUES(?, ?, ?, ?, ?)
            """,
            (now, level, code, message, details_json),
        )
        db.commit()


def list_incidents(limit: int = 20) -> List[IncidentRecord]:
    with get_db() as db:
        rows = db.execute(
            """
            SELECT id, created_at, level, code, message, details_json
            FROM incidents
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    out: List[IncidentRecord] = []
    for row in rows:
        try:
            details = json.loads(row["details_json"] or "{}")
        except Exception:
            details = {}
        out.append(
            IncidentRecord(
                id=int(row["id"]),
                level=str(row["level"]),
                code=str(row["code"]),
                message=str(row["message"]),
                details=details if isinstance(details, dict) else {},
                created_at=datetime.fromisoformat(str(row["created_at"])),
            )
        )
    return out


def add_analysis(analysis: StrategyAnalysis) -> StrategyAnalysis:
    with get_db() as db:
        db.execute(
            """
            INSERT INTO analyses(
                created_at, symbol, timeframe, strategy, signal, confidence,
                entry_price, stop_loss, take_profit, reasons_json, context_json
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis.created_at.isoformat(),
                analysis.symbol,
                analysis.timeframe,
                analysis.strategy,
                analysis.signal,
                analysis.confidence,
                analysis.entry_price,
                analysis.stop_loss,
                analysis.take_profit,
                json.dumps(analysis.reasons, ensure_ascii=False),
                json.dumps(analysis.context, ensure_ascii=False),
            ),
        )
        db.commit()
    return analysis


def list_recent_analyses(limit: int = 12) -> List[StrategyAnalysis]:
    with get_db() as db:
        rows = db.execute(
            """
            SELECT created_at, symbol, timeframe, strategy, signal, confidence,
                   entry_price, stop_loss, take_profit, reasons_json, context_json
            FROM analyses
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    out: List[StrategyAnalysis] = []
    for row in rows:
        try:
            reasons = json.loads(row["reasons_json"] or "[]")
        except Exception:
            reasons = []
        try:
            context = json.loads(row["context_json"] or "{}")
        except Exception:
            context = {}
        out.append(
            StrategyAnalysis(
                symbol=str(row["symbol"]),
                timeframe=str(row["timeframe"]),
                strategy=str(row["strategy"]),
                signal=str(row["signal"]),
                confidence=float(row["confidence"] or 0.0),
                entry_price=row["entry_price"],
                stop_loss=row["stop_loss"],
                take_profit=row["take_profit"],
                reasons=reasons if isinstance(reasons, list) else [],
                context=context if isinstance(context, dict) else {},
                created_at=datetime.fromisoformat(str(row["created_at"])),
            )
        )
    return out


def add_paper_event(event_type: str, summary: str, details: Dict[str, Any] | None = None) -> None:
    with get_db() as db:
        db.execute(
            """
            INSERT INTO paper_events(created_at, event_type, summary, details_json)
            VALUES(?, ?, ?, ?)
            """,
            (_utcnow().isoformat(), event_type, summary, json.dumps(details or {}, ensure_ascii=False)),
        )
        db.commit()


def create_order_intent(
    *,
    symbol: str,
    timeframe: str,
    strategy: str,
    direction: str,
    intent_type: str,
    status: str,
    confidence: float,
    entry_price: float | None,
    stop_loss: float | None,
    take_profit: float | None,
    quantity: float | None,
    rationale: str,
    details: Dict[str, Any] | None = None,
    decision_id: int | None = None,
) -> OrderIntentRecord:
    now = _utcnow().isoformat()
    with get_db() as db:
        cur = db.execute(
            """
            INSERT INTO order_intents(
                created_at, symbol, timeframe, strategy, direction, intent_type, status,
                confidence, entry_price, stop_loss, take_profit, quantity, rationale, details_json, decision_id
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                symbol.upper(),
                timeframe.upper(),
                strategy,
                direction,
                intent_type,
                status,
                confidence,
                entry_price,
                stop_loss,
                take_profit,
                quantity,
                rationale,
                json.dumps(details or {}, ensure_ascii=False),
                decision_id,
            ),
        )
        row_id = int(cur.lastrowid)
        cur.execute(
            """
            INSERT INTO order_intent_transitions(intent_id, created_at, from_status, to_status, reason, details_json)
            VALUES(?, ?, NULL, ?, ?, ?)
            """,
            (row_id, now, status, "intent_created", json.dumps(details or {}, ensure_ascii=False)),
        )
        db.commit()
    return get_order_intent_by_id(row_id)


_ALLOWED_INTENT_TRANSITIONS = {
    "pending": {"accepted", "rejected", "cancelled", "failed"},
    "accepted": {"accepted", "executed", "cancelled", "failed"},
    "rejected": set(),
    "executed": set(),
    "cancelled": set(),
    "failed": set(),
}


def update_order_intent_status(
    intent_id: int,
    status: str,
    details: Dict[str, Any] | None = None,
    reason: str = "",
) -> OrderIntentRecord:
    current = get_order_intent_by_id(intent_id)
    if status not in _ALLOWED_INTENT_TRANSITIONS.get(current.status, set()):
        raise ValueError(f"Invalid order intent transition: {current.status} -> {status}")
    merged_details = dict(current.details)
    if details:
        merged_details.update(details)
    with get_db() as db:
        db.execute(
            """
            UPDATE order_intents
            SET status = ?, details_json = ?
            WHERE id = ?
            """,
            (status, json.dumps(merged_details, ensure_ascii=False), intent_id),
        )
        db.execute(
            """
            INSERT INTO order_intent_transitions(intent_id, created_at, from_status, to_status, reason, details_json)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (
                intent_id,
                _utcnow().isoformat(),
                current.status,
                status,
                reason or "status_updated",
                json.dumps(details or {}, ensure_ascii=False),
            ),
        )
        db.commit()
    return get_order_intent_by_id(intent_id)


def get_order_intent_by_id(intent_id: int) -> OrderIntentRecord:
    with get_db() as db:
        row = db.execute(
            """
            SELECT id, created_at, symbol, timeframe, strategy, direction, intent_type, status,
                   confidence, entry_price, stop_loss, take_profit, quantity, rationale, details_json, decision_id
            FROM order_intents
            WHERE id = ?
            """,
            (intent_id,),
        ).fetchone()
    if not row:
        raise KeyError(f"Unknown order intent {intent_id}")
    try:
        details = json.loads(row["details_json"] or "{}")
    except Exception:
        details = {}
    return OrderIntentRecord(
        id=int(row["id"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        symbol=str(row["symbol"]),
        timeframe=str(row["timeframe"]),
        strategy=str(row["strategy"]),
        direction=str(row["direction"]),
        intent_type=str(row["intent_type"]),
        status=str(row["status"]),
        confidence=float(row["confidence"] or 0.0),
        entry_price=row["entry_price"],
        stop_loss=row["stop_loss"],
        take_profit=row["take_profit"],
        quantity=row["quantity"],
        rationale=str(row["rationale"] or ""),
        details=details if isinstance(details, dict) else {},
        decision_id=row["decision_id"],
    )


def list_order_intents(limit: int = 20) -> List[OrderIntentRecord]:
    with get_db() as db:
        rows = db.execute(
            """
            SELECT id, created_at, symbol, timeframe, strategy, direction, intent_type, status,
                   confidence, entry_price, stop_loss, take_profit, quantity, rationale, details_json, decision_id
            FROM order_intents
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [get_order_intent_by_id(int(row["id"])) for row in rows]


def create_decision_record(
    *,
    correlation_id: str,
    decision_type: str,
    symbol: str,
    timeframe: str,
    strategy: str,
    outcome: str,
    summary: str,
    evidence: Dict[str, Any] | None = None,
) -> DecisionRecord:
    now = _utcnow().isoformat()
    with get_db() as db:
        cur = db.execute(
            """
            INSERT INTO decision_records(
                created_at, correlation_id, decision_type, symbol, timeframe, strategy,
                outcome, summary, evidence_json
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                correlation_id,
                decision_type,
                symbol.upper(),
                timeframe.upper(),
                strategy,
                outcome,
                summary,
                json.dumps(evidence or {}, ensure_ascii=False),
            ),
        )
        decision_id = int(cur.lastrowid)
        db.commit()
    return get_decision_record(decision_id)


def get_decision_record(decision_id: int) -> DecisionRecord:
    with get_db() as db:
        row = db.execute("SELECT * FROM decision_records WHERE id = ?", (decision_id,)).fetchone()
    if not row:
        raise KeyError(f"Unknown decision record {decision_id}")
    try:
        evidence = json.loads(row["evidence_json"] or "{}")
    except Exception:
        evidence = {}
    return DecisionRecord(
        id=int(row["id"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        correlation_id=str(row["correlation_id"]),
        decision_type=str(row["decision_type"]),
        symbol=str(row["symbol"]),
        timeframe=str(row["timeframe"]),
        strategy=str(row["strategy"]),
        outcome=str(row["outcome"]),
        summary=str(row["summary"]),
        evidence=evidence if isinstance(evidence, dict) else {},
    )


def list_decision_records(limit: int = 20) -> List[DecisionRecord]:
    with get_db() as db:
        rows = db.execute("SELECT id FROM decision_records ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [get_decision_record(int(row["id"])) for row in rows]


def list_order_intent_transitions(intent_id: int) -> List[OrderIntentTransitionRecord]:
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM order_intent_transitions WHERE intent_id = ? ORDER BY id",
            (intent_id,),
        ).fetchall()
    records: List[OrderIntentTransitionRecord] = []
    for row in rows:
        try:
            details = json.loads(row["details_json"] or "{}")
        except Exception:
            details = {}
        records.append(
            OrderIntentTransitionRecord(
                id=int(row["id"]),
                intent_id=int(row["intent_id"]),
                created_at=datetime.fromisoformat(str(row["created_at"])),
                from_status=row["from_status"],
                to_status=str(row["to_status"]),
                reason=str(row["reason"] or ""),
                details=details if isinstance(details, dict) else {},
            )
        )
    return records


def add_trade_audit(
    *,
    event_type: str,
    symbol: str,
    timeframe: str,
    strategy: str,
    summary: str,
    position_id: int | None = None,
    intent_id: int | None = None,
    details: Dict[str, Any] | None = None,
) -> None:
    with get_db() as db:
        db.execute(
            """
            INSERT INTO trade_audit(created_at, event_type, symbol, timeframe, strategy, position_id, intent_id, summary, details_json)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _utcnow().isoformat(),
                event_type,
                symbol.upper(),
                timeframe.upper(),
                strategy,
                position_id,
                intent_id,
                summary,
                json.dumps(details or {}, ensure_ascii=False),
            ),
        )
        db.commit()


def list_trade_audits(limit: int = 20) -> List[TradeAuditRecord]:
    with get_db() as db:
        rows = db.execute(
            """
            SELECT id, created_at, event_type, symbol, timeframe, strategy, position_id, intent_id, summary, details_json
            FROM trade_audit
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    out: List[TradeAuditRecord] = []
    for row in rows:
        try:
            details = json.loads(row["details_json"] or "{}")
        except Exception:
            details = {}
        out.append(
            TradeAuditRecord(
                id=int(row["id"]),
                created_at=datetime.fromisoformat(str(row["created_at"])),
                event_type=str(row["event_type"]),
                symbol=str(row["symbol"]),
                timeframe=str(row["timeframe"]),
                strategy=str(row["strategy"]),
                position_id=row["position_id"],
                intent_id=row["intent_id"],
                summary=str(row["summary"]),
                details=details if isinstance(details, dict) else {},
            )
        )
    return out


def _row_to_market_event(row) -> MarketEventRecord:
    try:
        symbols = json.loads(row["symbols_json"] or "[]")
    except Exception:
        symbols = []
    try:
        raw = json.loads(row["raw_json"] or "{}")
    except Exception:
        raw = {}
    return MarketEventRecord(
        id=int(row["id"]),
        content_hash=str(row["content_hash"]),
        source=str(row["source"]),
        title=str(row["title"]),
        summary=str(row["summary"] or ""),
        url=str(row["url"] or ""),
        published_at=datetime.fromisoformat(str(row["published_at"])) if row["published_at"] else None,
        ingested_at=datetime.fromisoformat(str(row["ingested_at"])),
        symbols=symbols if isinstance(symbols, list) else [],
        event_type=str(row["event_type"]),
        sentiment=str(row["sentiment"]),
        sentiment_score=float(row["sentiment_score"] or 0.0),
        impact=str(row["impact"]),
        horizon=str(row["horizon"]),
        credibility_score=float(row["credibility_score"] or 0.0),
        classification_version=str(row["classification_version"]),
        raw=raw if isinstance(raw, dict) else {},
    )


def insert_market_event(event: MarketEventRecord) -> tuple[MarketEventRecord, bool]:
    with get_db() as db:
        existing = db.execute(
            "SELECT * FROM market_events WHERE content_hash = ?",
            (event.content_hash,),
        ).fetchone()
        if existing:
            return _row_to_market_event(existing), False
        cur = db.execute(
            """
            INSERT INTO market_events(
                content_hash, source, title, summary, url, published_at, ingested_at,
                symbols_json, event_type, sentiment, sentiment_score, impact, horizon,
                credibility_score, classification_version, raw_json
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.content_hash,
                event.source,
                event.title,
                event.summary,
                event.url,
                event.published_at.isoformat() if event.published_at else None,
                event.ingested_at.isoformat(),
                json.dumps(event.symbols, ensure_ascii=False),
                event.event_type,
                event.sentiment,
                event.sentiment_score,
                event.impact,
                event.horizon,
                event.credibility_score,
                event.classification_version,
                json.dumps(event.raw, ensure_ascii=False),
            ),
        )
        event_id = int(cur.lastrowid)
        db.commit()
        row = db.execute("SELECT * FROM market_events WHERE id = ?", (event_id,)).fetchone()
    return _row_to_market_event(row), True


def list_market_events(limit: int = 50, symbol: str | None = None) -> List[MarketEventRecord]:
    fetch_limit = max(limit, 1) if not symbol else max(limit * 8, 100)
    with get_db() as db:
        rows = db.execute(
            """
            SELECT * FROM market_events
            ORDER BY COALESCE(published_at, ingested_at) DESC, id DESC
            LIMIT ?
            """,
            (fetch_limit,),
        ).fetchall()
    events = [_row_to_market_event(row) for row in rows]
    if symbol:
        wanted = symbol.upper()
        events = [event for event in events if wanted in {item.upper() for item in event.symbols}]
    return events[:limit]


def get_market_event(event_id: int) -> MarketEventRecord:
    with get_db() as db:
        row = db.execute("SELECT * FROM market_events WHERE id = ?", (event_id,)).fetchone()
    if not row:
        raise KeyError(f"Unknown market event {event_id}")
    return _row_to_market_event(row)


def _row_to_event_outcome(row) -> EventOutcomeRecord:
    return EventOutcomeRecord(
        id=int(row["id"]),
        event_id=int(row["event_id"]),
        symbol=str(row["symbol"]),
        market_symbol=str(row["market_symbol"] or row["symbol"]),
        horizon=str(row["horizon"]),
        status=str(row["status"]),
        reference_at=datetime.fromisoformat(str(row["reference_at"])) if row["reference_at"] else None,
        target_at=datetime.fromisoformat(str(row["target_at"])) if row["target_at"] else None,
        reference_price=row["reference_price"],
        target_price=row["target_price"],
        forward_return_pct=row["forward_return_pct"],
        max_favorable_excursion_pct=row["max_favorable_excursion_pct"],
        max_adverse_excursion_pct=row["max_adverse_excursion_pct"],
        predicted_direction=str(row["predicted_direction"]),
        realized_direction=str(row["realized_direction"]),
        direction_hit=None if row["direction_hit"] is None else bool(row["direction_hit"]),
        brier_score=row["brier_score"],
        threshold_pct=float(row["threshold_pct"] or 0.0),
        reason=str(row["reason"] or ""),
        computed_at=datetime.fromisoformat(str(row["computed_at"])),
    )


def upsert_event_outcome(outcome: EventOutcomeRecord) -> EventOutcomeRecord:
    with get_db() as db:
        db.execute(
            """
            INSERT INTO event_outcomes(
                event_id, symbol, market_symbol, horizon, status, reference_at, target_at, reference_price,
                target_price, forward_return_pct, max_favorable_excursion_pct,
                max_adverse_excursion_pct, predicted_direction, realized_direction,
                direction_hit, brier_score, threshold_pct, reason, computed_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_id, symbol, horizon) DO UPDATE SET
                status = excluded.status,
                market_symbol = excluded.market_symbol,
                reference_at = excluded.reference_at,
                target_at = excluded.target_at,
                reference_price = excluded.reference_price,
                target_price = excluded.target_price,
                forward_return_pct = excluded.forward_return_pct,
                max_favorable_excursion_pct = excluded.max_favorable_excursion_pct,
                max_adverse_excursion_pct = excluded.max_adverse_excursion_pct,
                predicted_direction = excluded.predicted_direction,
                realized_direction = excluded.realized_direction,
                direction_hit = excluded.direction_hit,
                brier_score = excluded.brier_score,
                threshold_pct = excluded.threshold_pct,
                reason = excluded.reason,
                computed_at = excluded.computed_at
            """,
            (
                outcome.event_id,
                outcome.symbol.upper(),
                outcome.market_symbol.upper(),
                outcome.horizon,
                outcome.status,
                outcome.reference_at.isoformat() if outcome.reference_at else None,
                outcome.target_at.isoformat() if outcome.target_at else None,
                outcome.reference_price,
                outcome.target_price,
                outcome.forward_return_pct,
                outcome.max_favorable_excursion_pct,
                outcome.max_adverse_excursion_pct,
                outcome.predicted_direction,
                outcome.realized_direction,
                None if outcome.direction_hit is None else int(outcome.direction_hit),
                outcome.brier_score,
                outcome.threshold_pct,
                outcome.reason,
                outcome.computed_at.isoformat(),
            ),
        )
        db.commit()
        row = db.execute(
            "SELECT * FROM event_outcomes WHERE event_id = ? AND symbol = ? AND horizon = ?",
            (outcome.event_id, outcome.symbol.upper(), outcome.horizon),
        ).fetchone()
    return _row_to_event_outcome(row)


def list_event_outcomes(
    limit: int = 1000,
    status: str | None = None,
    event_id: int | None = None,
) -> List[EventOutcomeRecord]:
    clauses: list[str] = []
    params: list[Any] = []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if event_id is not None:
        clauses.append("event_id = ?")
        params.append(event_id)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(max(1, limit))
    with get_db() as db:
        rows = db.execute(
            f"SELECT * FROM event_outcomes{where} ORDER BY id DESC LIMIT ?",
            tuple(params),
        ).fetchall()
    return [_row_to_event_outcome(row) for row in rows]


def create_confluence_shadow(record: ConfluenceShadowRecord) -> ConfluenceShadowRecord:
    with get_db() as db:
        cur = db.execute(
            """
            INSERT INTO confluence_shadow_records(
                created_at, analysis_created_at, mode, symbol, timeframe, strategy,
                original_signal, original_confidence, shadow_signal, shadow_confidence,
                confidence_adjustment, action, target_horizon, event_score,
                eligible_event_count, event_ids_json, original_would_pass,
                shadow_would_pass, rationale, evidence_json, execution_unchanged
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.created_at.isoformat(),
                record.analysis_created_at.isoformat(),
                record.mode,
                record.symbol.upper(),
                record.timeframe.upper(),
                record.strategy,
                record.original_signal,
                record.original_confidence,
                record.shadow_signal,
                record.shadow_confidence,
                record.confidence_adjustment,
                record.action,
                record.target_horizon,
                record.event_score,
                record.eligible_event_count,
                json.dumps(record.event_ids),
                int(record.original_would_pass),
                int(record.shadow_would_pass),
                record.rationale,
                json.dumps(record.evidence, ensure_ascii=False),
                int(record.execution_unchanged),
            ),
        )
        record_id = int(cur.lastrowid)
        db.commit()
        row = db.execute("SELECT * FROM confluence_shadow_records WHERE id = ?", (record_id,)).fetchone()
    return _row_to_confluence_shadow(row)


def _row_to_confluence_shadow(row) -> ConfluenceShadowRecord:
    try:
        event_ids = json.loads(row["event_ids_json"] or "[]")
    except Exception:
        event_ids = []
    try:
        evidence = json.loads(row["evidence_json"] or "{}")
    except Exception:
        evidence = {}
    return ConfluenceShadowRecord(
        id=int(row["id"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        analysis_created_at=datetime.fromisoformat(str(row["analysis_created_at"])),
        mode=str(row["mode"]),
        symbol=str(row["symbol"]),
        timeframe=str(row["timeframe"]),
        strategy=str(row["strategy"]),
        original_signal=str(row["original_signal"]),
        original_confidence=float(row["original_confidence"]),
        shadow_signal=str(row["shadow_signal"]),
        shadow_confidence=float(row["shadow_confidence"]),
        confidence_adjustment=float(row["confidence_adjustment"]),
        action=str(row["action"]),
        target_horizon=str(row["target_horizon"]),
        event_score=float(row["event_score"]),
        eligible_event_count=int(row["eligible_event_count"]),
        event_ids=event_ids if isinstance(event_ids, list) else [],
        original_would_pass=bool(row["original_would_pass"]),
        shadow_would_pass=bool(row["shadow_would_pass"]),
        rationale=str(row["rationale"]),
        evidence=evidence if isinstance(evidence, dict) else {},
        execution_unchanged=bool(row["execution_unchanged"]),
    )


def list_confluence_shadows(limit: int = 50, symbol: str | None = None) -> List[ConfluenceShadowRecord]:
    if symbol:
        sql = "SELECT * FROM confluence_shadow_records WHERE symbol = ? ORDER BY id DESC LIMIT ?"
        params: tuple[Any, ...] = (symbol.upper(), max(1, limit))
    else:
        sql = "SELECT * FROM confluence_shadow_records ORDER BY id DESC LIMIT ?"
        params = (max(1, limit),)
    with get_db() as db:
        rows = db.execute(sql, params).fetchall()
    return [_row_to_confluence_shadow(row) for row in rows]


def add_event_alert(
    *,
    alert_key: str,
    alert_type: str,
    symbol: str,
    severity: str,
    summary: str,
    details: Dict[str, Any] | None = None,
) -> tuple[EventAlertRecord, bool]:
    now = _utcnow().isoformat()
    with get_db() as db:
        existing = db.execute("SELECT * FROM event_alerts WHERE alert_key = ?", (alert_key,)).fetchone()
        if existing:
            return _row_to_event_alert(existing), False
        cur = db.execute(
            """
            INSERT INTO event_alerts(alert_key, created_at, alert_type, symbol, severity, summary, details_json)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (alert_key, now, alert_type, symbol.upper(), severity, summary, json.dumps(details or {}, ensure_ascii=False)),
        )
        alert_id = int(cur.lastrowid)
        db.commit()
        row = db.execute("SELECT * FROM event_alerts WHERE id = ?", (alert_id,)).fetchone()
    return _row_to_event_alert(row), True


def _row_to_event_alert(row) -> EventAlertRecord:
    try:
        details = json.loads(row["details_json"] or "{}")
    except Exception:
        details = {}
    return EventAlertRecord(
        id=int(row["id"]),
        alert_key=str(row["alert_key"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        alert_type=str(row["alert_type"]),
        symbol=str(row["symbol"]),
        severity=str(row["severity"]),
        summary=str(row["summary"]),
        details=details if isinstance(details, dict) else {},
    )


def list_event_alerts(limit: int = 20) -> List[EventAlertRecord]:
    with get_db() as db:
        rows = db.execute("SELECT * FROM event_alerts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_row_to_event_alert(row) for row in rows]


def add_event_source_run(
    *,
    started_at: datetime,
    completed_at: datetime,
    source: str,
    fetched_items: int,
    inserted_events: int,
    error: str = "",
) -> None:
    with get_db() as db:
        db.execute(
            """
            INSERT INTO event_source_runs(started_at, completed_at, source, fetched_items, inserted_events, error)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (started_at.isoformat(), completed_at.isoformat(), source, fetched_items, inserted_events, error),
        )
        db.commit()


def list_paper_events(limit: int = 20) -> List[PaperEvent]:
    with get_db() as db:
        rows = db.execute(
            """
            SELECT id, created_at, event_type, summary, details_json
            FROM paper_events
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    out: List[PaperEvent] = []
    for row in rows:
        try:
            details = json.loads(row["details_json"] or "{}")
        except Exception:
            details = {}
        out.append(
            PaperEvent(
                id=int(row["id"]),
                created_at=datetime.fromisoformat(str(row["created_at"])),
                event_type=str(row["event_type"]),
                summary=str(row["summary"]),
                details=details if isinstance(details, dict) else {},
            )
        )
    return out


def _row_to_position(row) -> PaperPosition:
    return PaperPosition(
        id=int(row["id"]),
        symbol=str(row["symbol"]),
        timeframe=str(row["timeframe"]),
        strategy=str(row["strategy"]),
        direction=str(row["direction"]),
        quantity=float(row["quantity"]),
        status=str(row["status"]),
        entry_price=float(row["entry_price"]),
        current_price=row["current_price"],
        stop_loss=row["stop_loss"],
        take_profit=row["take_profit"],
        opened_at=datetime.fromisoformat(str(row["opened_at"])),
        closed_at=datetime.fromisoformat(str(row["closed_at"])) if row["closed_at"] else None,
        exit_price=row["exit_price"],
        realized_pnl=float(row["realized_pnl"] or 0.0),
        unrealized_pnl=float(row["unrealized_pnl"] or 0.0),
        close_reason=row["close_reason"],
        account_currency=str(row["account_currency"] or "USD"),
        cash_per_price_unit_per_lot=float(row["cash_per_price_unit_per_lot"] or 1.0),
        instrument_spec_source=str(row["instrument_spec_source"] or "legacy"),
        broker_position_id=(int(row["broker_position_id"]) if row["broker_position_id"] is not None else None),
        realized_pnl_source=str(row["realized_pnl_source"] or "paper_estimate"),
    )


def list_paper_positions(status: Optional[str] = None) -> List[PaperPosition]:
    sql = """
        SELECT id, symbol, timeframe, strategy, direction, quantity, status, entry_price, current_price,
               stop_loss, take_profit, opened_at, closed_at, exit_price, realized_pnl, unrealized_pnl, close_reason,
               account_currency, cash_per_price_unit_per_lot, instrument_spec_source,
               broker_position_id, realized_pnl_source
        FROM paper_positions
    """
    params: tuple[Any, ...] = ()
    if status:
        sql += " WHERE status = ?"
        params = (status,)
    sql += " ORDER BY id DESC"
    with get_db() as db:
        rows = db.execute(sql, params).fetchall()
    return [_row_to_position(row) for row in rows]


def get_open_position(symbol: str, timeframe: str) -> Optional[PaperPosition]:
    with get_db() as db:
        row = db.execute(
            """
            SELECT id, symbol, timeframe, strategy, direction, quantity, status, entry_price, current_price,
                   stop_loss, take_profit, opened_at, closed_at, exit_price, realized_pnl, unrealized_pnl, close_reason,
                   account_currency, cash_per_price_unit_per_lot, instrument_spec_source,
                   broker_position_id, realized_pnl_source
            FROM paper_positions
            WHERE status = 'open' AND symbol = ? AND timeframe = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (symbol.upper(), timeframe.upper()),
        ).fetchone()
    return _row_to_position(row) if row else None


def open_paper_position(
    *,
    symbol: str,
    timeframe: str,
    strategy: str,
    direction: str,
    quantity: float,
    entry_price: float,
    stop_loss: float | None,
    take_profit: float | None,
    account_currency: str = "USD",
    cash_per_price_unit_per_lot: float = 1.0,
    instrument_spec_source: str = "legacy",
    broker_position_id: int | None = None,
) -> PaperPosition:
    now = _utcnow().isoformat()
    with get_db() as db:
        cur = db.execute(
            """
            INSERT INTO paper_positions(
                symbol, timeframe, strategy, direction, quantity, status, entry_price, current_price,
                stop_loss, take_profit, opened_at, realized_pnl, unrealized_pnl,
                account_currency, cash_per_price_unit_per_lot, instrument_spec_source,
                broker_position_id, realized_pnl_source
            )
            VALUES(?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?, 'paper_estimate')
            """,
            (
                symbol.upper(),
                timeframe.upper(),
                strategy,
                direction,
                quantity,
                entry_price,
                entry_price,
                stop_loss,
                take_profit,
                now,
                account_currency.upper(),
                cash_per_price_unit_per_lot,
                instrument_spec_source,
                broker_position_id,
            ),
        )
        row_id = int(cur.lastrowid)
        db.commit()
    add_paper_event(
        "position_opened",
        f"Opened paper {direction} on {symbol.upper()} {timeframe.upper()}",
        {
            "position_id": row_id,
            "symbol": symbol.upper(),
            "timeframe": timeframe.upper(),
            "direction": direction,
            "entry_price": entry_price,
        },
    )
    add_trade_audit(
        event_type="paper_position_opened",
        symbol=symbol,
        timeframe=timeframe,
        strategy=strategy,
        position_id=row_id,
        summary=f"Opened paper {direction} position.",
        details={
            "entry_price": entry_price,
                "quantity": quantity,
                "account_currency": account_currency.upper(),
                "cash_per_price_unit_per_lot": cash_per_price_unit_per_lot,
                "instrument_spec_source": instrument_spec_source,
                "broker_position_id": broker_position_id,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        },
    )
    return get_position_by_id(row_id)


def get_position_by_id(position_id: int) -> PaperPosition:
    with get_db() as db:
        row = db.execute(
            """
            SELECT id, symbol, timeframe, strategy, direction, quantity, status, entry_price, current_price,
                   stop_loss, take_profit, opened_at, closed_at, exit_price, realized_pnl, unrealized_pnl, close_reason,
                   account_currency, cash_per_price_unit_per_lot, instrument_spec_source,
                   broker_position_id, realized_pnl_source
            FROM paper_positions
            WHERE id = ?
            """,
            (position_id,),
        ).fetchone()
    if not row:
        raise KeyError(f"Unknown paper position {position_id}")
    return _row_to_position(row)


def update_paper_position_mark(position_id: int, current_price: float, unrealized_pnl: float) -> None:
    with get_db() as db:
        db.execute(
            """
            UPDATE paper_positions
            SET current_price = ?, unrealized_pnl = ?
            WHERE id = ?
            """,
            (current_price, unrealized_pnl, position_id),
        )
        db.commit()


def update_paper_position_targets(position_id: int, stop_loss: float | None, take_profit: float | None) -> None:
    position = get_position_by_id(position_id)
    with get_db() as db:
        db.execute(
            """
            UPDATE paper_positions
            SET stop_loss = ?, take_profit = ?
            WHERE id = ?
            """,
            (stop_loss, take_profit, position_id),
        )
        db.commit()
    add_trade_audit(
        event_type="paper_position_updated",
        symbol=position.symbol,
        timeframe=position.timeframe,
        strategy=position.strategy,
        position_id=position_id,
        summary="Updated paper position targets.",
        details={"stop_loss": stop_loss, "take_profit": take_profit},
    )


def close_paper_position(
    position_id: int,
    exit_price: float,
    reason: str,
    *,
    realized_pnl_override: float | None = None,
    closed_at_override: datetime | None = None,
    realized_pnl_source: str = "paper_estimate",
) -> PaperPosition:
    position = get_position_by_id(position_id)
    if position.status != "open":
        return position
    signed_move = (exit_price - position.entry_price) if position.direction == "long" else (position.entry_price - exit_price)
    estimated = signed_move * position.quantity * position.cash_per_price_unit_per_lot
    realized = float(realized_pnl_override) if realized_pnl_override is not None else estimated
    now = (closed_at_override or _utcnow()).isoformat()
    with get_db() as db:
        db.execute(
            """
            UPDATE paper_positions
            SET status = 'closed',
                current_price = ?,
                closed_at = ?,
                exit_price = ?,
                realized_pnl = ?,
                unrealized_pnl = 0,
                close_reason = ?,
                realized_pnl_source = ?
            WHERE id = ?
            """,
            (exit_price, now, exit_price, realized, reason, realized_pnl_source, position_id),
        )
        db.commit()
    add_paper_event(
        "position_closed",
        f"Closed paper {position.direction} on {position.symbol} {position.timeframe}",
        {
            "position_id": position_id,
            "exit_price": exit_price,
            "reason": reason,
            "realized_pnl": realized,
            "realized_pnl_source": realized_pnl_source,
            "broker_position_id": position.broker_position_id,
            "closed_at": now,
        },
    )
    add_trade_audit(
        event_type="paper_position_closed",
        symbol=position.symbol,
        timeframe=position.timeframe,
        strategy=position.strategy,
        position_id=position_id,
        summary=f"Closed paper {position.direction} position.",
        details={
            "exit_price": exit_price,
            "reason": reason,
            "realized_pnl": realized,
            "realized_pnl_source": realized_pnl_source,
            "broker_position_id": position.broker_position_id,
            "closed_at": now,
        },
    )
    return get_position_by_id(position_id)


def set_paper_position_broker_id(position_id: int, broker_position_id: int | None) -> PaperPosition:
    with get_db() as db:
        db.execute(
            "UPDATE paper_positions SET broker_position_id = ? WHERE id = ?",
            (broker_position_id, position_id),
        )
        db.commit()
    return get_position_by_id(position_id)


def reconcile_closed_paper_position_from_broker(
    position_id: int,
    *,
    exit_price: float,
    realized_pnl: float,
    closed_at: datetime | None,
    broker_position_id: int,
    broker_details: Dict[str, Any] | None = None,
) -> PaperPosition:
    position = get_position_by_id(position_id)
    if position.status != "closed":
        raise ValueError(f"Paper position {position_id} must be closed before broker P&L reconciliation.")

    closed_at_value = (closed_at or position.closed_at or _utcnow()).isoformat()
    details = dict(broker_details or {})
    details.update(
        {
            "exit_price": float(exit_price),
            "realized_pnl": float(realized_pnl),
            "realized_pnl_source": "ctrader_deal",
            "broker_position_id": int(broker_position_id),
            "closed_at": closed_at_value,
        }
    )

    with get_db() as db:
        db.execute(
            """
            UPDATE paper_positions
            SET current_price = ?,
                closed_at = ?,
                exit_price = ?,
                realized_pnl = ?,
                unrealized_pnl = 0,
                broker_position_id = ?,
                realized_pnl_source = 'ctrader_deal'
            WHERE id = ?
            """,
            (
                float(exit_price),
                closed_at_value,
                float(exit_price),
                float(realized_pnl),
                int(broker_position_id),
                position_id,
            ),
        )

        # Keep the operator-facing close row current while retaining a separate
        # reconciliation audit record below.
        audit_rows = db.execute(
            """
            SELECT id, details_json
            FROM trade_audit
            WHERE position_id = ? AND event_type = 'paper_position_closed'
            """,
            (position_id,),
        ).fetchall()
        for row in audit_rows:
            try:
                audit_details = json.loads(row["details_json"] or "{}")
            except Exception:
                audit_details = {}
            if not isinstance(audit_details, dict):
                audit_details = {}
            audit_details.update(details)
            db.execute(
                "UPDATE trade_audit SET details_json = ? WHERE id = ?",
                (json.dumps(audit_details, ensure_ascii=False), int(row["id"])),
            )
        db.commit()

    add_trade_audit(
        event_type="ctrader_demo_close_reconciled",
        symbol=position.symbol,
        timeframe=position.timeframe,
        strategy=position.strategy,
        position_id=position_id,
        summary="Reconciled close price and realized P&L from cTrader deal history.",
        details=details,
    )
    return get_position_by_id(position_id)


def daily_realized_pnl() -> float:
    today = _utcnow().date().isoformat()
    with get_db() as db:
        row = db.execute(
            """
            SELECT COALESCE(SUM(realized_pnl), 0) AS total
            FROM paper_positions
            WHERE status = 'closed' AND substr(closed_at, 1, 10) = ?
            """,
            (today,),
        ).fetchone()
    return float((row["total"] if row else 0.0) or 0.0)


def daily_trade_count() -> int:
    today = _utcnow().date().isoformat()
    with get_db() as db:
        row = db.execute(
            """
            SELECT COUNT(*) AS total
            FROM paper_positions
            WHERE substr(opened_at, 1, 10) = ?
            """,
            (today,),
        ).fetchone()
    return int((row["total"] if row else 0) or 0)
