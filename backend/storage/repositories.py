from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.domain.models import (
    EngineConfig,
    EngineRuntime,
    IncidentRecord,
    OrderIntentRecord,
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
        return EngineConfig(**payload)
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
) -> OrderIntentRecord:
    now = _utcnow().isoformat()
    with get_db() as db:
        cur = db.execute(
            """
            INSERT INTO order_intents(
                created_at, symbol, timeframe, strategy, direction, intent_type, status,
                confidence, entry_price, stop_loss, take_profit, quantity, rationale, details_json
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        row_id = int(cur.lastrowid)
        db.commit()
    return get_order_intent_by_id(row_id)


def update_order_intent_status(intent_id: int, status: str, details: Dict[str, Any] | None = None) -> OrderIntentRecord:
    current = get_order_intent_by_id(intent_id)
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
        db.commit()
    return get_order_intent_by_id(intent_id)


def get_order_intent_by_id(intent_id: int) -> OrderIntentRecord:
    with get_db() as db:
        row = db.execute(
            """
            SELECT id, created_at, symbol, timeframe, strategy, direction, intent_type, status,
                   confidence, entry_price, stop_loss, take_profit, quantity, rationale, details_json
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
    )


def list_order_intents(limit: int = 20) -> List[OrderIntentRecord]:
    with get_db() as db:
        rows = db.execute(
            """
            SELECT id, created_at, symbol, timeframe, strategy, direction, intent_type, status,
                   confidence, entry_price, stop_loss, take_profit, quantity, rationale, details_json
            FROM order_intents
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [get_order_intent_by_id(int(row["id"])) for row in rows]


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
    )


def list_paper_positions(status: Optional[str] = None) -> List[PaperPosition]:
    sql = """
        SELECT id, symbol, timeframe, strategy, direction, quantity, status, entry_price, current_price,
               stop_loss, take_profit, opened_at, closed_at, exit_price, realized_pnl, unrealized_pnl, close_reason
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
                   stop_loss, take_profit, opened_at, closed_at, exit_price, realized_pnl, unrealized_pnl, close_reason
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
) -> PaperPosition:
    now = _utcnow().isoformat()
    with get_db() as db:
        cur = db.execute(
            """
            INSERT INTO paper_positions(
                symbol, timeframe, strategy, direction, quantity, status, entry_price, current_price,
                stop_loss, take_profit, opened_at, realized_pnl, unrealized_pnl
            )
            VALUES(?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, 0, 0)
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
                   stop_loss, take_profit, opened_at, closed_at, exit_price, realized_pnl, unrealized_pnl, close_reason
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


def close_paper_position(position_id: int, exit_price: float, reason: str) -> PaperPosition:
    position = get_position_by_id(position_id)
    if position.status != "open":
        return position
    signed_move = (exit_price - position.entry_price) if position.direction == "long" else (position.entry_price - exit_price)
    realized = signed_move * position.quantity
    now = _utcnow().isoformat()
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
                close_reason = ?
            WHERE id = ?
            """,
            (exit_price, now, exit_price, realized, reason, position_id),
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
        },
    )
    add_trade_audit(
        event_type="paper_position_closed",
        symbol=position.symbol,
        timeframe=position.timeframe,
        strategy=position.strategy,
        position_id=position_id,
        summary=f"Closed paper {position.direction} position.",
        details={"exit_price": exit_price, "reason": reason, "realized_pnl": realized},
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
