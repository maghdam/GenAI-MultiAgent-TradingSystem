"""Read-only Alpaca SPY/QQQ baseline using existing trusted-runtime audits.

No order, account, position, engine, or scheduler interfaces are imported.
Run explicitly; this module has no import-time broker or file side effects.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi import HTTPException

from backend.services.alpaca_market_data import AlpacaMarketDataError, _stock_feed, get_bars
from backend.services.runtime_strategy_validation import _audit_on_frame
from backend.services.studio_backtests import _prepare_market_frame, _safe_cost_bps


SYMBOLS = ("SPY", "QQQ")
STRATEGIES = ("sma_cross", "rsi_reversal", "breakout")
MIN_COMPLETE_BARS = 250
MAX_REQUESTED_BARS = 1000



def _validate_alpaca_daily_session_labels(frame: pd.DataFrame, symbol: str) -> None:
    """Check only Alpaca's US-stock D1 session-start timestamp convention.

    A stock daily bar is labeled at midnight America/New_York, which is
    04:00 or 05:00 UTC across daylight-saving transitions. This is not an
    NYSE holiday calendar and does not establish a bar's actual close time.
    """
    eastern = frame.index.tz_convert("America/New_York")
    for position, timestamp in enumerate(eastern):
        if timestamp.dayofweek >= 5:
            raise ValueError(
                f"{symbol}: Alpaca D1 bar {position} has a weekend New York session label "
                f"({timestamp.isoformat()})."
            )
        if timestamp != timestamp.normalize():
            raise ValueError(
                f"{symbol}: Alpaca D1 bar {position} must be labeled at midnight "
                f"America/New_York ({timestamp.isoformat()})."
            )


def run_baseline(
    *,
    symbols: tuple[str, ...] = SYMBOLS,
    num_bars: int = 500,
    fee_bps: float = 1.0,
    slippage_bps: float = 1.0,
    spread_bps: float = 2.0,
) -> dict:
    """Evaluate fixed, existing strategies without tuning or submitting trades."""
    selected = tuple(str(value).strip().upper() for value in symbols)
    if not selected or len(set(selected)) != len(selected) or any(s not in SYMBOLS for s in selected):
        raise ValueError("Select SPY and/or QQQ once each; other symbols are out of this baseline's scope.")
    if type(num_bars) is not int or not MIN_COMPLETE_BARS <= num_bars <= MAX_REQUESTED_BARS:
        raise ValueError(f"num_bars must be an integer from {MIN_COMPLETE_BARS} to {MAX_REQUESTED_BARS}.")

    fee = _safe_cost_bps(fee_bps, "fee_bps")
    slippage = _safe_cost_bps(slippage_bps, "slippage_bps")
    spread = _safe_cost_bps(spread_bps, "spread_bps")
    # A fixed normalized research allocation, not an implementable $500
    # stock/fractional-share order-sizing simulation.
    size_pct = 100.0
    observed_at = datetime.now(UTC)
    cutoff = pd.Timestamp(observed_at.date(), tz="UTC")
    stock_feed = _stock_feed()

    rows: list[dict] = []
    for symbol in selected:
        raw = get_bars(symbol=symbol, timeframe="D1", num_bars=num_bars, asset_class="us_equity")
        frame = _prepare_market_frame(raw, "D1")
        # A possibly still-forming current UTC daily bar must not enter a
        # historical experiment, even when Alpaca returns it.
        frame = frame.loc[frame.index < cutoff].copy()
        # Validate observed US trading-day labels, not fixed UTC midnight:
        # New York midnight converts to 05:00 or 04:00 UTC across DST.
        _validate_alpaca_daily_session_labels(frame, symbol)
        if len(frame) < MIN_COMPLETE_BARS:
            raise ValueError(
                f"{symbol}: only {len(frame)} completed daily bars; "
                f"at least {MIN_COMPLETE_BARS} are required for this baseline."
            )
        for key in STRATEGIES:
            result = _audit_on_frame(
                strategy_key=key,
                symbol=symbol,
                timeframe="D1",
                df=frame,
                params=None,
                fee_bps=fee,
                slippage_bps=slippage,
                spread_bps=spread,
                position_size_pct=size_pct,
            )
            rows.append(result)

    return {
        "study": "option-a-alpaca-spy-qqq-trusted-baseline-v1",
        "mode": "research_only",
        "read_only": True,
        "broker_order_requests": 0,
        "observed_at_utc": observed_at.isoformat(),
        "symbols": list(selected),
        "strategies": list(STRATEGIES),
        "timeframe": "D1",
        "requested_bars_per_symbol": num_bars,
        "data_source": {
            "provider": "alpaca",
            "asset_class": "us_equity",
            "stock_feed": stock_feed,
            "adjustment": "raw",
            "completed_daily_bars_only": True,
        },
        "cost_assumptions_bps": {"fee_per_side": fee, "slippage_per_side": slippage, "quoted_spread": spread},
        "position_size_pct": size_pct,
        "methodology": "Existing trusted runtime StrategyV2 auditor: chronological 70/30, completed-bar/next-open, costs, regime and development-only sensitivity.",
        "limitations": [
            "Research percentages use normalized portfolio sizing, not a $500 cash/fractional-share/margin model.",
            "The existing runtime accounting is not stock-exchange fill or borrow/short-sale simulation.",
            "Raw bars may omit split/dividend adjustment; IEX and SIP feed coverage can differ.",
            "Costs are assumptions, not Alpaca-confirmed execution costs.",
            "No model selection is performed on the out-of-sample window; no strategy is approved for trading.",
        ],
        "results": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only trusted strategy baseline for Alpaca SPY and QQQ.")
    parser.add_argument("--symbol", action="append", choices=SYMBOLS, dest="symbols", help="Repeat to select both; default is SPY and QQQ.")
    parser.add_argument("--bars", type=int, default=500, help="Daily bars requested per symbol (250-1000).")
    parser.add_argument("--output", type=Path, help="Optional local JSON report path (otherwise prints JSON).")
    args = parser.parse_args(argv)
    try:
        report = run_baseline(symbols=tuple(args.symbols or SYMBOLS), num_bars=args.bars)
        payload = json.dumps(report, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
            print(f"Read-only research report written to {args.output}")
        else:
            print(payload)
        return 0
    except (ValueError, HTTPException, AlpacaMarketDataError, OSError) as exc:
        print(f"Alpaca research failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
