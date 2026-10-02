from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

from backend.domain.models import (
    DailyRejectionReason,
    DailySummaryResponse,
    EngineConfig,
)
from backend.storage.db import get_db
from backend.storage.repositories import load_engine_config


def _utc_day() -> date:
    return datetime.now(UTC).date()


def _utc_instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _realized_drawdown(entries: list[tuple[datetime, int, float]]) -> float:
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for _, _, pnl in sorted(entries, key=lambda item: (item[0], item[1])):
        cumulative += float(pnl)
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    return max_drawdown


def build_daily_summary(day: date | None = None) -> DailySummaryResponse:
    """Build a read-only UTC daily summary from persisted TradeAgent truth.

    Realized P&L prefers immutable cTrader broker-deal rows. A paper-position
    close contributes only when no broker deals exist for that local position.
    Drawdown is therefore realized-P&L drawdown, not mark-to-market equity
    drawdown; the runtime does not persist an intraday equity curve.
    """

    report_day = day or _utc_day()
    day_key = report_day.isoformat()
    config = load_engine_config(EngineConfig())

    with get_db() as db:
        opened_row = db.execute(
            """
            SELECT COUNT(*) AS total
            FROM paper_positions
            WHERE date(opened_at) = ?
            """,
            (day_key,),
        ).fetchone()

        closed_rows = db.execute(
            """
            SELECT id, closed_at, realized_pnl, account_currency
            FROM paper_positions
            WHERE status = 'closed' AND date(closed_at) = ?
            ORDER BY closed_at, id
            """,
            (day_key,),
        ).fetchall()

        broker_entries = db.execute(
            """
            SELECT deal_id, local_position_id, execution_at, net_profit, account_currency
            FROM broker_deals
            WHERE date(execution_at) = ?
            ORDER BY execution_at, deal_id
            """,
            (day_key,),
        ).fetchall()

        paper_entries = db.execute(
            """
            SELECT p.id, p.closed_at, p.realized_pnl, p.account_currency
            FROM paper_positions p
            WHERE p.status = 'closed'
              AND date(p.closed_at) = ?
              AND NOT EXISTS (
                  SELECT 1
                  FROM broker_deals d
                  WHERE d.local_position_id = p.id
              )
            ORDER BY p.closed_at, p.id
            """,
            (day_key,),
        ).fetchall()

        rejection_rows = db.execute(
            """
            SELECT
                CASE
                    WHEN trim(summary) = '' THEN 'Unspecified rejection.'
                    ELSE trim(summary)
                END AS reason,
                COUNT(*) AS total
            FROM decision_records
            WHERE date(created_at) = ?
              AND decision_type = 'paper_execution_gate'
              AND outcome LIKE 'rejected_%'
            GROUP BY reason
            ORDER BY total DESC, reason ASC
            """,
            (day_key,),
        ).fetchall()

        closed_outcomes: list[float] = []
        for row in closed_rows:
            broker_total = db.execute(
                """
                SELECT COUNT(*) AS deal_count, COALESCE(SUM(net_profit), 0) AS net_profit
                FROM broker_deals
                WHERE local_position_id = ?
                """,
                (int(row["id"]),),
            ).fetchone()
            if broker_total and int(broker_total["deal_count"] or 0) > 0:
                closed_outcomes.append(float(broker_total["net_profit"] or 0.0))
            else:
                closed_outcomes.append(float(row["realized_pnl"] or 0.0))

    entries: list[tuple[datetime, int, float]] = []
    for row in broker_entries:
        entries.append(
            (
                _utc_instant(str(row["execution_at"])),
                int(row["deal_id"]),
                float(row["net_profit"] or 0.0),
            )
        )
    for row in paper_entries:
        entries.append(
            (
                _utc_instant(str(row["closed_at"])),
                -int(row["id"]),
                float(row["realized_pnl"] or 0.0),
            )
        )

    realized_pnl = sum(item[2] for item in entries)
    wins = sum(1 for value in closed_outcomes if value > 1e-12)
    losses = sum(1 for value in closed_outcomes if value < -1e-12)
    breakeven = len(closed_outcomes) - wins - losses
    resolved = wins + losses
    win_rate_pct = (wins / resolved * 100.0) if resolved else None

    rejected_by_reason = [
        DailyRejectionReason(reason=str(row["reason"]), count=int(row["total"] or 0))
        for row in rejection_rows
    ]
    rejected_signals = sum(item.count for item in rejected_by_reason)
    trades_opened = int((opened_row["total"] if opened_row else 0) or 0)

    return DailySummaryResponse(
        date_utc=report_day,
        account_currency=config.account_currency.upper(),
        trades=trades_opened,
        trades_opened=trades_opened,
        trades_closed=len(closed_rows),
        realized_pnl=realized_pnl,
        max_realized_drawdown=_realized_drawdown(entries),
        wins=wins,
        losses=losses,
        breakeven=breakeven,
        win_rate_pct=win_rate_pct,
        rejected_signals=rejected_signals,
        rejected_by_reason=rejected_by_reason,
        broker_deal_count=len(broker_entries),
        pure_paper_close_count=len(paper_entries),
    )
