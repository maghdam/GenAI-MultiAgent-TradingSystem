from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Callable, Iterable

import pandas as pd

from backend.domain.models import ConfluenceReplayMetrics, ConfluenceReplayResponse, ConfluenceShadowRecord
from backend.services.market_data import get_bars
from backend.storage.repositories import list_confluence_shadows


_HORIZON_DURATION = {
    "5m": timedelta(minutes=5),
    "30m": timedelta(minutes=30),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
}
_METHODOLOGY = (
    "Immutable decision-cohort replay. Entry uses the next bar open strictly after analysis time; "
    "exit uses the first bar close at or after the stored target horizon. Each policy allows at most "
    "one overlapping position per symbol/timeframe/strategy. Returns include round-trip configured costs."
)


def _utc_timestamp(value: datetime) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize(UTC) if timestamp.tzinfo is None else timestamp.tz_convert(UTC)


def _prepare_bars(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty or "close" not in frame.columns:
        return pd.DataFrame()
    bars = frame.copy()
    bars.index = pd.to_datetime(bars.index, utc=True, errors="coerce")
    bars = bars[~bars.index.isna()].sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]
    bars["close"] = pd.to_numeric(bars["close"], errors="coerce")
    bars["open"] = pd.to_numeric(bars["open"], errors="coerce") if "open" in bars else bars["close"]
    return bars.dropna(subset=["open", "close"])


def _price_record(record: ConfluenceShadowRecord, bars: pd.DataFrame) -> dict[str, object] | None:
    if bars.empty or record.original_signal == "no_trade":
        return None
    signal_at = _utc_timestamp(record.analysis_created_at)
    entry_candidates = bars.loc[bars.index > signal_at]
    if entry_candidates.empty:
        return None
    entry_at = entry_candidates.index[0]
    entry_price = float(entry_candidates.iloc[0]["open"])
    exit_target = entry_at + _HORIZON_DURATION[record.target_horizon]
    exit_candidates = bars.loc[bars.index >= exit_target]
    if exit_candidates.empty:
        return None
    exit_at = exit_candidates.index[0]
    exit_price = float(exit_candidates.iloc[0]["close"])
    if entry_price <= 0 or exit_price <= 0:
        return None
    gross_return = (exit_price - entry_price) / entry_price
    if record.original_signal == "short":
        gross_return *= -1.0
    return {"record": record, "entry_at": entry_at, "exit_at": exit_at, "gross_return": float(gross_return)}


def _metrics(
    observations: Iterable[dict[str, object]], policy: str, fee_bps_per_side: float
) -> ConfluenceReplayMetrics:
    candidates = 0
    returns: list[float] = []
    trade_times: list[pd.Timestamp] = []
    active_until: dict[tuple[str, str, str], pd.Timestamp] = {}
    for item in sorted(observations, key=lambda row: row["entry_at"]):
        record = item["record"]
        assert isinstance(record, ConfluenceShadowRecord)
        accepted = record.original_would_pass if policy == "original" else record.shadow_would_pass
        if not accepted:
            continue
        candidates += 1
        key = (record.symbol, record.timeframe, record.strategy)
        entry_at = item["entry_at"]
        exit_at = item["exit_at"]
        assert isinstance(entry_at, pd.Timestamp) and isinstance(exit_at, pd.Timestamp)
        if key in active_until and entry_at < active_until[key]:
            continue
        active_until[key] = exit_at
        returns.append(float(item["gross_return"]) - (2.0 * fee_bps_per_side / 10_000.0))
        trade_times.append(entry_at)

    wins = sum(value > 0 for value in returns)
    losses = sum(value < 0 for value in returns)
    positive = sum(value for value in returns if value > 0)
    negative = abs(sum(value for value in returns if value < 0))
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in returns:
        equity *= max(0.0, 1.0 + value)
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = min(max_drawdown, equity / peak - 1.0)
    elapsed_days = 0.0
    if len(trade_times) >= 2:
        elapsed_days = max(1.0, (max(trade_times) - min(trade_times)).total_seconds() / 86_400.0)
    elif trade_times:
        elapsed_days = 1.0
    return ConfluenceReplayMetrics(
        candidate_decisions=candidates,
        trades=len(returns),
        wins=wins,
        losses=losses,
        win_rate_pct=round((wins / len(returns) * 100.0) if returns else 0.0, 2),
        expectancy_pct=round((sum(returns) / len(returns) * 100.0) if returns else 0.0, 4),
        total_return_pct=round((equity - 1.0) * 100.0 if returns else 0.0, 4),
        max_drawdown_pct=round(max_drawdown * 100.0, 4),
        profit_factor=round(positive / negative, 4) if negative > 0 else None,
        trades_per_day=round(len(returns) / elapsed_days, 3) if elapsed_days else 0.0,
    )


def evaluate_confluence_replay(
    records: list[ConfluenceShadowRecord],
    bars_by_key: dict[tuple[str, str], pd.DataFrame],
    *,
    fee_bps_per_side: float = 0.0,
    unavailable_keys: set[tuple[str, str]] | None = None,
) -> ConfluenceReplayResponse:
    observations: list[dict[str, object]] = []
    pending = 0
    unavailable = 0
    unavailable_keys = unavailable_keys or set()
    prepared = {key: _prepare_bars(frame) for key, frame in bars_by_key.items()}
    for record in sorted(records, key=lambda item: item.analysis_created_at):
        if record.original_signal == "no_trade":
            continue
        key = (record.symbol.upper(), record.timeframe.upper())
        if key in unavailable_keys or key not in prepared:
            unavailable += 1
            continue
        priced = _price_record(record, prepared[key])
        if priced is None:
            pending += 1
            continue
        observations.append(priced)

    original = _metrics(observations, "original", fee_bps_per_side)
    shadow = _metrics(observations, "shadow", fee_bps_per_side)
    deltas = {
        "trades": float(shadow.trades - original.trades),
        "win_rate_pct": round(shadow.win_rate_pct - original.win_rate_pct, 4),
        "expectancy_pct": round(shadow.expectancy_pct - original.expectancy_pct, 4),
        "total_return_pct": round(shadow.total_return_pct - original.total_return_pct, 4),
        "max_drawdown_pct": round(shadow.max_drawdown_pct - original.max_drawdown_pct, 4),
    }
    warnings: list[str] = []
    if len(observations) < 100 or min(original.trades, shadow.trades) < 30:
        verdict = "insufficient_data"
        verdict_reason = "Collect at least 100 priced decisions and 30 trades under each policy before review."
    elif shadow.expectancy_pct <= original.expectancy_pct or shadow.max_drawdown_pct < original.max_drawdown_pct:
        verdict = "keep_shadow"
        verdict_reason = "The adjusted policy has not improved expectancy without worsening maximum drawdown."
    else:
        verdict = "candidate_for_review"
        verdict_reason = "Minimum evidence gates passed; independent out-of-sample review is still required before promotion."
    if unavailable:
        warnings.append(f"Market bars were unavailable for {unavailable} decision records.")
    if pending:
        warnings.append(f"Future exit bars are not available yet for {pending} decision records.")
    warnings.append("This is a decision-cohort replay, not a broker-fill or portfolio-capital simulation.")
    return ConfluenceReplayResponse(
        methodology=_METHODOLOGY,
        total_records=len(records),
        priced_records=len(observations),
        pending_records=pending,
        unavailable_records=unavailable,
        fee_bps_per_side=fee_bps_per_side,
        original=original,
        shadow=shadow,
        deltas=deltas,
        verdict=verdict,  # type: ignore[arg-type]
        verdict_reason=verdict_reason,
        warnings=warnings,
    )


def run_confluence_replay(
    *,
    limit: int = 1000,
    symbol: str | None = None,
    fee_bps_per_side: float = 0.0,
    num_bars: int = 5000,
    bars_loader: Callable[[str, str, int], pd.DataFrame] | None = None,
) -> ConfluenceReplayResponse:
    records = list_confluence_shadows(limit, symbol)
    keys = {(record.symbol.upper(), record.timeframe.upper()) for record in records if record.original_signal != "no_trade"}
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    unavailable: set[tuple[str, str]] = set()
    loader = bars_loader or (lambda market_symbol, timeframe, count: get_bars(market_symbol, timeframe, count))
    for key in keys:
        try:
            frames[key] = loader(key[0], key[1], num_bars)
        except Exception:
            unavailable.add(key)
    return evaluate_confluence_replay(
        records, frames, fee_bps_per_side=fee_bps_per_side, unavailable_keys=unavailable
    )
