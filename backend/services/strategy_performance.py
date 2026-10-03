from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from backend.domain.models import StrategyPerformanceResponse, StrategyPerformanceSlice
from backend.storage.db import get_db


def _utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _now_utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _parse_instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    return _utc_naive(parsed)


def build_strategy_performance_report(
    *,
    window_days: int = 30,
    symbol: str | None = None,
    timeframe: str | None = None,
    strategy: str | None = None,
    group_limit: int = 100,
    now: datetime | None = None,
) -> StrategyPerformanceResponse:
    """Aggregate closed-trade performance from persisted broker/local truth.

    One closed paper position is one trade sample. When broker deals exist for
    that local position, every linked deal is summed once and replaces the local
    paper estimate for the trade outcome. This preserves broker-ledger precedence
    and avoids treating partial-close deals as separate trades.
    """

    bounded_days = max(1, min(365, int(window_days)))
    bounded_limit = max(1, min(200, int(group_limit)))
    window_end = _utc_naive(now or _now_utc())
    window_start = window_end - timedelta(days=bounded_days)

    symbol_filter = (symbol or "").strip().upper() or None
    timeframe_filter = (timeframe or "").strip().upper() or None
    strategy_filter = (strategy or "").strip() or None

    clauses = [
        "p.status = 'closed'",
        "p.closed_at IS NOT NULL",
        "julianday(p.closed_at) >= julianday(?)",
        "julianday(p.closed_at) < julianday(?)",
    ]
    params: list[Any] = [window_start.isoformat(), window_end.isoformat()]
    if symbol_filter:
        clauses.append("p.symbol = ?")
        params.append(symbol_filter)
    if timeframe_filter:
        clauses.append("p.timeframe = ?")
        params.append(timeframe_filter)
    if strategy_filter:
        clauses.append("p.strategy = ?")
        params.append(strategy_filter)

    where = " AND ".join(clauses)
    with get_db() as db:
        rows = db.execute(
            f"""
            WITH broker_totals AS (
                SELECT
                    local_position_id,
                    COUNT(*) AS deal_count,
                    COALESCE(SUM(net_profit), 0) AS broker_net_profit
                FROM broker_deals
                GROUP BY local_position_id
            )
            SELECT
                p.id,
                p.symbol,
                p.timeframe,
                p.strategy,
                p.account_currency,
                p.closed_at,
                p.realized_pnl,
                COALESCE(b.deal_count, 0) AS deal_count,
                COALESCE(b.broker_net_profit, 0) AS broker_net_profit
            FROM paper_positions p
            LEFT JOIN broker_totals b
              ON b.local_position_id = p.id
            WHERE {where}
            ORDER BY p.closed_at, p.id
            """,
            tuple(params),
        ).fetchall()

    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "trades": 0,
            "realized_pnl": 0.0,
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "broker_backed_trades": 0,
            "pure_paper_trades": 0,
            "broker_deal_count": 0,
            "first_closed_at": None,
            "last_closed_at": None,
        }
    )

    for row in rows:
        symbol_key = str(row["symbol"]).upper()
        timeframe_key = str(row["timeframe"]).upper()
        strategy_key = str(row["strategy"])
        currency_key = str(row["account_currency"] or "USD").upper()
        key = (symbol_key, timeframe_key, strategy_key, currency_key)

        deal_count = int(row["deal_count"] or 0)
        pnl = (
            float(row["broker_net_profit"] or 0.0)
            if deal_count > 0
            else float(row["realized_pnl"] or 0.0)
        )
        closed_at = _parse_instant(str(row["closed_at"]))

        bucket = grouped[key]
        bucket["trades"] += 1
        bucket["realized_pnl"] += pnl
        bucket["broker_deal_count"] += deal_count
        if deal_count > 0:
            bucket["broker_backed_trades"] += 1
        else:
            bucket["pure_paper_trades"] += 1

        if pnl > 1e-12:
            bucket["wins"] += 1
        elif pnl < -1e-12:
            bucket["losses"] += 1
        else:
            bucket["breakeven"] += 1

        if bucket["first_closed_at"] is None or closed_at < bucket["first_closed_at"]:
            bucket["first_closed_at"] = closed_at
        if bucket["last_closed_at"] is None or closed_at > bucket["last_closed_at"]:
            bucket["last_closed_at"] = closed_at

    slices: list[StrategyPerformanceSlice] = []
    for (symbol_key, timeframe_key, strategy_key, currency_key), bucket in grouped.items():
        trades = int(bucket["trades"])
        resolved = int(bucket["wins"]) + int(bucket["losses"])
        slices.append(
            StrategyPerformanceSlice(
                symbol=symbol_key,
                timeframe=timeframe_key,
                strategy=strategy_key,
                account_currency=currency_key,
                trades=trades,
                realized_pnl=float(bucket["realized_pnl"]),
                average_realized_pnl=(
                    float(bucket["realized_pnl"]) / trades if trades else 0.0
                ),
                wins=int(bucket["wins"]),
                losses=int(bucket["losses"]),
                breakeven=int(bucket["breakeven"]),
                win_rate_pct=(
                    int(bucket["wins"]) / resolved * 100.0
                    if resolved
                    else None
                ),
                broker_backed_trades=int(bucket["broker_backed_trades"]),
                pure_paper_trades=int(bucket["pure_paper_trades"]),
                broker_deal_count=int(bucket["broker_deal_count"]),
                first_closed_at=bucket["first_closed_at"],
                last_closed_at=bucket["last_closed_at"],
            )
        )

    slices.sort(
        key=lambda item: (
            -item.trades,
            item.symbol,
            item.timeframe,
            item.strategy,
            item.account_currency,
        )
    )
    slice_count = len(slices)
    selected = slices[:bounded_limit]
    currencies = sorted({item.account_currency for item in slices})

    return StrategyPerformanceResponse(
        window_days=bounded_days,
        window_start_utc=window_start,
        window_end_utc=window_end,
        symbol_filter=symbol_filter,
        timeframe_filter=timeframe_filter,
        strategy_filter=strategy_filter,
        sample_count=len(rows),
        slice_count=slice_count,
        group_limit=bounded_limit,
        groups_truncated=slice_count > bounded_limit,
        account_currencies=currencies,
        slices=selected,
        message=(
            "Each closed position is one trade sample. Broker deals, when present, "
            "replace local paper realized-P&L estimates and are summed once per trade; "
            "account currencies remain separate performance slices."
        ),
    )
