from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from io import StringIO
from typing import Any

from backend.domain.models import (
    BrokerStatementRowInput,
    JournalExportResponse,
    JournalExportRow,
    StatementComparisonItem,
    StatementComparisonRequest,
    StatementComparisonResponse,
)
from backend.storage.db import get_db


def _utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _utc_aware(value: datetime) -> datetime:
    return _utc_naive(value).replace(tzinfo=UTC)


def _parse_instant(value: str) -> datetime:
    return _utc_aware(datetime.fromisoformat(str(value)))


def _now_utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _bounded_days(value: int) -> int:
    return max(1, min(365, int(value)))


def build_journal_export(
    *,
    window_days: int = 30,
    now: datetime | None = None,
    all_time: bool = False,
) -> JournalExportResponse:
    """Build one deterministic journal row per persisted closed position."""

    bounded_days = None if all_time else _bounded_days(window_days)
    window_end = _utc_naive(now or _now_utc())
    window_start = (
        None
        if bounded_days is None
        else window_end - timedelta(days=bounded_days)
    )

    with get_db() as db:
        if all_time:
            positions = db.execute(
                """
                SELECT
                    id, symbol, timeframe, strategy, direction, quantity,
                    entry_price, opened_at, closed_at, exit_price,
                    realized_pnl, close_reason, account_currency,
                    broker_position_id
                FROM paper_positions
                WHERE status = 'closed'
                  AND closed_at IS NOT NULL
                  AND julianday(closed_at) < julianday(?)
                ORDER BY julianday(closed_at), id
                """,
                (window_end.isoformat(),),
            ).fetchall()
        else:
            positions = db.execute(
                """
                SELECT
                    id, symbol, timeframe, strategy, direction, quantity,
                    entry_price, opened_at, closed_at, exit_price,
                    realized_pnl, close_reason, account_currency,
                    broker_position_id
                FROM paper_positions
                WHERE status = 'closed'
                  AND closed_at IS NOT NULL
                  AND julianday(closed_at) >= julianday(?)
                  AND julianday(closed_at) < julianday(?)
                ORDER BY julianday(closed_at), id
                """,
                (window_start.isoformat(), window_end.isoformat()),
            ).fetchall()

        position_ids = [int(row["id"]) for row in positions]
        deals: list[Any] = []
        audits: list[Any] = []
        if position_ids:
            placeholders = ",".join("?" for _ in position_ids)
            deals = db.execute(
                f"""
                SELECT
                    deal_id, broker_position_id, local_position_id,
                    execution_at, net_profit
                FROM broker_deals
                WHERE local_position_id IN ({placeholders})
                ORDER BY local_position_id, julianday(execution_at), deal_id
                """,
                tuple(position_ids),
            ).fetchall()
            audits = db.execute(
                f"""
                SELECT id, position_id, intent_id, event_type
                FROM trade_audit
                WHERE position_id IN ({placeholders})
                ORDER BY position_id, id
                """,
                tuple(position_ids),
            ).fetchall()

            opening_intent_ids = sorted(
                {
                    int(row["intent_id"])
                    for row in audits
                    if row["intent_id"] is not None
                    and str(row["event_type"]) in {
                        "ctrader_order_executed",
                        "paper_signal_open",
                        "ctrader_order_ack_timeout_tracking_retained",
                        "ctrader_unprotected_tracking_retained",
                    }
                }
            )
            intents: list[Any] = []
            if opening_intent_ids:
                intent_placeholders = ",".join("?" for _ in opening_intent_ids)
                intents = db.execute(
                    f"""
                    SELECT id, details_json
                    FROM order_intents
                    WHERE id IN ({intent_placeholders})
                    """,
                    tuple(opening_intent_ids),
                ).fetchall()
        else:
            intents = []

    deals_by_position: dict[int, list[Any]] = defaultdict(list)
    for row in deals:
        deals_by_position[int(row["local_position_id"])].append(row)

    audits_by_position: dict[int, list[int]] = defaultdict(list)
    opening_intents_by_position: dict[int, list[int]] = defaultdict(list)
    opening_event_types = {
        "ctrader_order_executed",
        "paper_signal_open",
        "ctrader_order_ack_timeout_tracking_retained",
        "ctrader_unprotected_tracking_retained",
    }
    for row in audits:
        position_id = int(row["position_id"])
        audits_by_position[position_id].append(int(row["id"]))
        if row["intent_id"] is not None and str(row["event_type"]) in opening_event_types:
            opening_intents_by_position[position_id].append(int(row["intent_id"]))

    source_by_intent: dict[int, str] = {}
    for row in intents:
        try:
            details = json.loads(row["details_json"] or "{}")
        except Exception:
            details = {}
        source = str(details.get("source") or "").strip().lower() if isinstance(details, dict) else ""
        if source in {"manual", "auto"}:
            source_by_intent[int(row["id"])] = source

    export_rows: list[JournalExportRow] = []
    currencies: set[str] = set()

    for position in positions:
        position_id = int(position["id"])
        position_deals = deals_by_position.get(position_id, [])
        broker_deal_ids = [int(row["deal_id"]) for row in position_deals]
        deal_broker_ids = {
            int(row["broker_position_id"])
            for row in position_deals
            if int(row["broker_position_id"] or 0) > 0
        }
        persisted_broker_id = int(position["broker_position_id"] or 0) or None

        identity_status = "not_applicable"
        identity_detail = "Pure-paper trade has no broker identity."
        broker_position_id = persisted_broker_id

        if persisted_broker_id is not None:
            if deal_broker_ids and deal_broker_ids != {persisted_broker_id}:
                identity_status = "conflict"
                identity_detail = (
                    "Persisted broker position id conflicts with one or more linked broker deals."
                )
            else:
                identity_status = "resolved"
                identity_detail = "Persisted broker position id is canonical for this trade."
        elif deal_broker_ids:
            if len(deal_broker_ids) == 1:
                broker_position_id = next(iter(deal_broker_ids))
                identity_status = "resolved"
                identity_detail = (
                    "Broker position id is resolved from the immutable linked broker-deal ledger."
                )
            else:
                identity_status = "conflict"
                identity_detail = (
                    "Linked broker deals reference multiple broker position ids."
                )
                broker_position_id = None

        if position_deals:
            realized_pnl = sum(float(row["net_profit"] or 0.0) for row in position_deals)
            pnl_basis = "broker_deals"
        else:
            realized_pnl = float(position["realized_pnl"] or 0.0)
            pnl_basis = "paper_estimate"

        currency = str(position["account_currency"] or "USD").upper()
        currencies.add(currency)

        linked_sources = {
            source_by_intent[int(intent_id)]
            for intent_id in opening_intents_by_position.get(position_id, [])
            if int(intent_id) in source_by_intent
        }
        execution_source = next(iter(linked_sources)) if len(linked_sources) == 1 else "unknown"

        export_rows.append(
            JournalExportRow(
                row_id=f"local_position:{position_id}",
                local_position_id=position_id,
                broker_position_id=broker_position_id,
                broker_identity_status=identity_status,
                broker_identity_detail=identity_detail,
                broker_deal_ids=broker_deal_ids,
                audit_event_ids=audits_by_position.get(position_id, []),
                symbol=str(position["symbol"]).upper(),
                timeframe=str(position["timeframe"]).upper(),
                strategy=str(position["strategy"]),
                direction=str(position["direction"]),
                quantity=float(position["quantity"]),
                execution_source=execution_source,
                opened_at_utc=_parse_instant(str(position["opened_at"])),
                closed_at_utc=_parse_instant(str(position["closed_at"])),
                entry_price=float(position["entry_price"]),
                exit_price=(
                    float(position["exit_price"])
                    if position["exit_price"] is not None
                    else None
                ),
                account_currency=currency,
                realized_pnl=realized_pnl,
                realized_pnl_basis=pnl_basis,
                broker_deal_count=len(position_deals),
                close_reason=(
                    str(position["close_reason"])
                    if position["close_reason"] is not None
                    else None
                ),
            )
        )

    return JournalExportResponse(
        window_days=bounded_days,
        window_start_utc=_utc_aware(window_start) if window_start is not None else None,
        window_end_utc=_utc_aware(window_end),
        all_time=all_time,
        row_count=len(export_rows),
        account_currencies=sorted(currencies),
        rows=export_rows,
        message=(
            "Each export row is one persisted closed TradeAgent position. Linked broker deals, "
            "when present, replace local paper realized-P&L estimates; audit_event_ids preserve "
            "traceability back to the TradeAgent journal."
        ),
    )


def render_journal_export_csv(report: JournalExportResponse) -> str:
    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        [
            "row_id",
            "local_position_id",
            "broker_position_id",
            "broker_identity_status",
            "broker_deal_ids",
            "audit_event_ids",
            "symbol",
            "timeframe",
            "strategy",
            "direction",
            "quantity",
            "execution_source",
            "opened_at_utc",
            "closed_at_utc",
            "entry_price",
            "exit_price",
            "account_currency",
            "realized_pnl",
            "realized_pnl_basis",
            "broker_deal_count",
            "close_reason",
        ]
    )
    for row in report.rows:
        writer.writerow(
            [
                row.row_id,
                row.local_position_id,
                row.broker_position_id or "",
                row.broker_identity_status,
                "|".join(str(item) for item in row.broker_deal_ids),
                "|".join(str(item) for item in row.audit_event_ids),
                row.symbol,
                row.timeframe,
                row.strategy,
                row.direction,
                row.quantity,
                row.execution_source,
                row.opened_at_utc.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                row.closed_at_utc.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                row.entry_price,
                "" if row.exit_price is None else row.exit_price,
                row.account_currency,
                row.realized_pnl,
                row.realized_pnl_basis,
                row.broker_deal_count,
                row.close_reason or "",
            ]
        )
    return output.getvalue()


def _statement_row_utc(row: BrokerStatementRowInput) -> datetime:
    return _utc_aware(row.closed_at)


def _external_group_key(row: BrokerStatementRowInput) -> str:
    if row.broker_position_id is not None:
        return f"broker_position:{row.broker_position_id}"
    if row.deal_id is not None:
        return f"deal:{row.deal_id}"
    return f"row:{row.row_id}"


def _aggregate_statement_rows(rows: list[BrokerStatementRowInput]) -> dict[str, Any]:
    return {
        "row_ids": [row.row_id for row in rows],
        "deal_ids": sorted({int(row.deal_id) for row in rows if row.deal_id is not None}),
        "broker_position_ids": sorted(
            {int(row.broker_position_id) for row in rows if row.broker_position_id is not None}
        ),
        "symbols": sorted({row.symbol.upper() for row in rows}),
        "directions": sorted({row.direction for row in rows if row.direction is not None}),
        "currencies": sorted({row.account_currency.upper() for row in rows}),
        "realized_pnl": sum(float(row.realized_pnl) for row in rows),
        "closed_at": max(_statement_row_utc(row) for row in rows),
    }


def _comparison_item_for_local(
    local: JournalExportRow,
    rows: list[BrokerStatementRowInput],
    *,
    pnl_tolerance: float,
    close_time_tolerance_seconds: int,
) -> StatementComparisonItem:
    aggregate = _aggregate_statement_rows(rows)
    reasons: list[str] = []

    if local.broker_identity_status == "conflict":
        reasons.append("local_broker_identity_conflict")

    broker_ids = aggregate["broker_position_ids"]
    if (
        local.broker_position_id is not None
        and broker_ids
        and broker_ids != [local.broker_position_id]
    ):
        reasons.append("broker_position_id_mismatch")

    if aggregate["symbols"] != [local.symbol]:
        reasons.append("symbol_mismatch")

    directions = aggregate["directions"]
    if directions and directions != [local.direction]:
        reasons.append("direction_mismatch")

    currencies = aggregate["currencies"]
    if currencies != [local.account_currency]:
        reasons.append("currency_mismatch")

    statement_pnl = float(aggregate["realized_pnl"])
    pnl_difference = statement_pnl - float(local.realized_pnl)
    if abs(pnl_difference) > float(pnl_tolerance) + 1e-12:
        reasons.append("realized_pnl_mismatch")

    statement_closed_at = aggregate["closed_at"]
    close_diff = abs(
        (statement_closed_at - local.closed_at_utc.astimezone(UTC)).total_seconds()
    )
    if close_diff > int(close_time_tolerance_seconds):
        reasons.append("close_time_mismatch")

    statement_deal_ids = aggregate["deal_ids"]
    if local.broker_deal_ids and statement_deal_ids:
        if set(statement_deal_ids) != set(local.broker_deal_ids):
            reasons.append("deal_id_set_mismatch")

    return StatementComparisonItem(
        status="mismatch" if reasons else "matched",
        local_row_id=local.row_id,
        local_position_id=local.local_position_id,
        statement_row_ids=aggregate["row_ids"],
        broker_position_id=local.broker_position_id,
        local_deal_ids=local.broker_deal_ids,
        statement_deal_ids=statement_deal_ids,
        symbol=local.symbol,
        local_currency=local.account_currency,
        statement_currencies=currencies,
        local_realized_pnl=local.realized_pnl,
        statement_realized_pnl=statement_pnl,
        pnl_difference=pnl_difference,
        local_closed_at_utc=local.closed_at_utc,
        statement_closed_at_utc=statement_closed_at,
        mismatch_reasons=reasons,
    )


def compare_external_statement(
    request: StatementComparisonRequest,
    *,
    now: datetime | None = None,
) -> StatementComparisonResponse:
    journal = build_journal_export(window_days=request.window_days, now=now)
    local_rows = journal.rows

    broker_map: dict[int, list[JournalExportRow]] = defaultdict(list)
    deal_map: dict[int, list[JournalExportRow]] = defaultdict(list)
    local_by_id = {row.local_position_id: row for row in local_rows}

    for row in local_rows:
        if row.broker_position_id is not None:
            broker_map[int(row.broker_position_id)].append(row)
        for deal_id in row.broker_deal_ids:
            deal_map[int(deal_id)].append(row)

    deal_counts = Counter(
        int(row.deal_id)
        for row in request.rows
        if row.deal_id is not None
    )
    duplicate_deal_ids = {deal_id for deal_id, count in deal_counts.items() if count > 1}

    matched_statement_by_local: dict[int, list[BrokerStatementRowInput]] = defaultdict(list)
    unresolved_items: list[StatementComparisonItem] = []
    unmatched_external_groups: dict[str, list[BrokerStatementRowInput]] = defaultdict(list)

    for row in request.rows:
        reasons: list[str] = []
        if row.broker_position_id is None and row.deal_id is None:
            reasons.append("missing_statement_identity")
        if row.deal_id is not None and int(row.deal_id) in duplicate_deal_ids:
            reasons.append("duplicate_statement_deal_id")

        candidates: set[int] = set()
        broker_candidates: set[int] = set()
        deal_candidates: set[int] = set()

        if row.broker_position_id is not None:
            broker_candidates = {
                item.local_position_id
                for item in broker_map.get(int(row.broker_position_id), [])
            }
            candidates.update(broker_candidates)

        if row.deal_id is not None:
            deal_candidates = {
                item.local_position_id
                for item in deal_map.get(int(row.deal_id), [])
            }
            candidates.update(deal_candidates)

        if broker_candidates and deal_candidates and broker_candidates != deal_candidates:
            reasons.append("statement_identity_conflict")

        if len(candidates) > 1:
            reasons.append("ambiguous_local_identity")

        if reasons:
            local = local_by_id[next(iter(candidates))] if len(candidates) == 1 else None
            unresolved_items.append(
                StatementComparisonItem(
                    status="identity_unresolved",
                    local_row_id=local.row_id if local is not None else None,
                    local_position_id=local.local_position_id if local is not None else None,
                    statement_row_ids=[row.row_id],
                    broker_position_id=(
                        local.broker_position_id if local is not None else row.broker_position_id
                    ),
                    local_deal_ids=local.broker_deal_ids if local is not None else [],
                    statement_deal_ids=[row.deal_id] if row.deal_id is not None else [],
                    symbol=local.symbol if local is not None else row.symbol.upper(),
                    local_currency=local.account_currency if local is not None else None,
                    statement_currencies=[row.account_currency.upper()],
                    local_realized_pnl=local.realized_pnl if local is not None else None,
                    statement_realized_pnl=float(row.realized_pnl),
                    local_closed_at_utc=local.closed_at_utc if local is not None else None,
                    statement_closed_at_utc=_statement_row_utc(row),
                    mismatch_reasons=reasons,
                )
            )
            continue

        if len(candidates) == 1:
            local_id = next(iter(candidates))
            local = local_by_id[local_id]
            if local.broker_identity_status == "conflict":
                unresolved_items.append(
                    StatementComparisonItem(
                        status="identity_unresolved",
                        local_row_id=local.row_id,
                        local_position_id=local.local_position_id,
                        statement_row_ids=[row.row_id],
                        broker_position_id=local.broker_position_id,
                        local_deal_ids=local.broker_deal_ids,
                        statement_deal_ids=[row.deal_id] if row.deal_id is not None else [],
                        symbol=local.symbol,
                        local_currency=local.account_currency,
                        statement_currencies=[row.account_currency.upper()],
                        local_realized_pnl=local.realized_pnl,
                        statement_realized_pnl=float(row.realized_pnl),
                        local_closed_at_utc=local.closed_at_utc,
                        statement_closed_at_utc=_statement_row_utc(row),
                        mismatch_reasons=["local_broker_identity_conflict"],
                    )
                )
                continue

            matched_statement_by_local[local_id].append(row)
            continue

        unmatched_external_groups[_external_group_key(row)].append(row)

    items: list[StatementComparisonItem] = []

    for local in local_rows:
        statement_rows = matched_statement_by_local.get(local.local_position_id, [])
        if statement_rows:
            items.append(
                _comparison_item_for_local(
                    local,
                    statement_rows,
                    pnl_tolerance=request.pnl_tolerance,
                    close_time_tolerance_seconds=request.close_time_tolerance_seconds,
                )
            )
            continue

        if any(item.local_position_id == local.local_position_id for item in unresolved_items):
            continue

        if local.broker_identity_status == "not_applicable":
            reason = "no_broker_identity"
        elif local.broker_identity_status == "conflict":
            reason = "local_broker_identity_conflict"
        else:
            reason = "missing_from_statement"

        status = "identity_unresolved" if reason == "local_broker_identity_conflict" else "local_only"
        items.append(
            StatementComparisonItem(
                status=status,
                local_row_id=local.row_id,
                local_position_id=local.local_position_id,
                broker_position_id=local.broker_position_id,
                local_deal_ids=local.broker_deal_ids,
                symbol=local.symbol,
                local_currency=local.account_currency,
                local_realized_pnl=local.realized_pnl,
                local_closed_at_utc=local.closed_at_utc,
                mismatch_reasons=[reason],
            )
        )

    for group_rows in unmatched_external_groups.values():
        aggregate = _aggregate_statement_rows(group_rows)
        broker_ids = aggregate["broker_position_ids"]
        items.append(
            StatementComparisonItem(
                status="statement_only",
                statement_row_ids=aggregate["row_ids"],
                broker_position_id=broker_ids[0] if len(broker_ids) == 1 else None,
                statement_deal_ids=aggregate["deal_ids"],
                symbol=aggregate["symbols"][0] if len(aggregate["symbols"]) == 1 else None,
                statement_currencies=aggregate["currencies"],
                statement_realized_pnl=float(aggregate["realized_pnl"]),
                statement_closed_at_utc=aggregate["closed_at"],
                mismatch_reasons=["missing_from_tradeagent_journal"],
            )
        )

    items.extend(unresolved_items)
    items.sort(
        key=lambda item: (
            {
                "mismatch": 0,
                "identity_unresolved": 1,
                "local_only": 2,
                "statement_only": 3,
                "matched": 4,
            }[item.status],
            item.local_position_id if item.local_position_id is not None else 10**18,
            ",".join(item.statement_row_ids),
        )
    )

    counts = Counter(item.status for item in items)

    return StatementComparisonResponse(
        window_days=journal.window_days,
        window_start_utc=journal.window_start_utc,
        window_end_utc=journal.window_end_utc,
        local_row_count=journal.row_count,
        statement_row_count=len(request.rows),
        matched=int(counts.get("matched", 0)),
        mismatched=int(counts.get("mismatch", 0)),
        local_only=int(counts.get("local_only", 0)),
        statement_only=int(counts.get("statement_only", 0)),
        identity_unresolved=int(counts.get("identity_unresolved", 0)),
        pnl_tolerance=float(request.pnl_tolerance),
        close_time_tolerance_seconds=int(request.close_time_tolerance_seconds),
        items=items,
        message=(
            "External rows are compared transiently and are never persisted. Matching uses exact "
            "broker position/deal identity only; linked broker deals remain authoritative for local "
            "realized P&L and partial-close rows aggregate into one trade."
        ),
    )
