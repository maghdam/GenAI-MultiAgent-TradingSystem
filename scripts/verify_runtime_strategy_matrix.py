from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = "http://127.0.0.1:4000"
RUNTIME_STRATEGIES = ("sma_cross", "rsi_reversal", "breakout")


def _normalize_base_url(value: str) -> str:
    return str(value or DEFAULT_BASE_URL).strip().rstrip("/")


def _headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"x-api-key": api_key}


def _get_json(
    session: requests.Session,
    *,
    base_url: str,
    path: str,
    api_key: str | None,
    timeout: float,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = session.get(
        f"{_normalize_base_url(base_url)}{path}",
        params=params,
        headers=_headers(api_key),
        timeout=timeout,
    )
    try:
        payload = response.json()
    except Exception:
        payload = {"detail": response.text}

    if response.status_code >= 400:
        detail = payload.get("detail") if isinstance(payload, dict) else payload
        raise RuntimeError(f"{path} -> HTTP {response.status_code}: {detail}")

    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} returned a non-object JSON payload.")
    return payload


def _enabled_targets(config: dict[str, Any]) -> list[tuple[str, str]]:
    watchlist = config.get("watchlist")
    if not isinstance(watchlist, list):
        return []

    targets: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in watchlist:
        if not isinstance(item, dict) or not bool(item.get("enabled")):
            continue
        symbol = str(item.get("symbol") or "").strip().upper()
        timeframe = str(item.get("timeframe") or "").strip().upper()
        if not symbol or not timeframe:
            continue
        target = (symbol, timeframe)
        if target in seen:
            continue
        seen.add(target)
        targets.append(target)
    return targets


def _broker_evidence(status: dict[str, Any]) -> dict[str, Any]:
    broker = status.get("broker")
    return broker if isinstance(broker, dict) else {}


def collect_matrix(
    *,
    session: requests.Session,
    base_url: str,
    api_key: str | None = None,
    timeout: float = 300.0,
    num_bars: int = 1500,
    fee_bps: float = 1.0,
    slippage_bps: float = 1.0,
    spread_bps: float = 2.0,
    position_size_pct: float = 100.0,
) -> dict[str, Any]:
    status = _get_json(
        session,
        base_url=base_url,
        path="/api/status",
        api_key=api_key,
        timeout=timeout,
    )
    broker = _broker_evidence(status)

    if not bool(broker.get("socket_connected")):
        raise RuntimeError("cTrader socket is not connected in the running backend.")
    if not bool(broker.get("account_authorized")):
        raise RuntimeError("cTrader account is not authorized in the running backend.")
    if not bool(broker.get("demo_account_confirmed")) or str(broker.get("account_type") or "").lower() != "demo":
        raise RuntimeError("Connected cTrader account is not positively confirmed as demo.")

    config = _get_json(
        session,
        base_url=base_url,
        path="/api/config",
        api_key=api_key,
        timeout=timeout,
    )
    targets = _enabled_targets(config)
    if not targets:
        raise RuntimeError("No enabled watchlist targets are available for Phase 5.1 verification.")

    assumptions = {
        "num_bars": int(num_bars),
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "spread_bps": float(spread_bps),
        "position_size_pct": float(position_size_pct),
    }

    results: list[dict[str, Any]] = []
    for symbol, timeframe in targets:
        for strategy in RUNTIME_STRATEGIES:
            params = {
                "strategy": strategy,
                "symbol": symbol,
                "timeframe": timeframe,
                **assumptions,
            }
            try:
                audit = _get_json(
                    session,
                    base_url=base_url,
                    path="/api/studio/runtime-strategy-audit",
                    api_key=api_key,
                    timeout=timeout,
                    params=params,
                )
            except Exception as exc:
                results.append(
                    {
                        "status": "unavailable",
                        "strategy": strategy,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "error": str(exc),
                    }
                )
                continue

            results.append(
                {
                    "status": "ok",
                    "strategy": strategy,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "audit": audit,
                }
            )

    available = sum(1 for row in results if row.get("status") == "ok")
    return {
        "phase": "5.1",
        "purpose": "Read-only connected cTrader demo historical-feed evidence for trusted runtime strategies.",
        "broker": {
            "account_type": broker.get("account_type"),
            "demo_account_confirmed": broker.get("demo_account_confirmed"),
            "socket_connected": broker.get("socket_connected"),
            "account_authorized": broker.get("account_authorized"),
            "market_data_ready": broker.get("market_data_ready"),
            "account_id": broker.get("account_id"),
        },
        "strategies": list(RUNTIME_STRATEGIES),
        "targets": [{"symbol": symbol, "timeframe": timeframe} for symbol, timeframe in targets],
        "assumptions": assumptions,
        "results": results,
        "available_rows": available,
        "total_rows": len(results),
        "all_ok": available == len(results),
    }


def _metric(container: dict[str, Any], key: str) -> Any:
    return container.get(key, "n/a") if isinstance(container, dict) else "n/a"


def print_summary(matrix: dict[str, Any]) -> None:
    broker = matrix.get("broker") or {}
    print(
        "BROKER "
        f"account_type={broker.get('account_type')} "
        f"demo_confirmed={broker.get('demo_account_confirmed')} "
        f"socket_connected={broker.get('socket_connected')} "
        f"authorized={broker.get('account_authorized')} "
        f"market_data_ready={broker.get('market_data_ready')}"
    )

    targets = matrix.get("targets") or []
    print("TARGETS " + ", ".join(f"{item['symbol']}/{item['timeframe']}" for item in targets))
    print(f"STRATEGIES {', '.join(matrix.get('strategies') or [])}")
    print(f"ASSUMPTIONS {json.dumps(matrix.get('assumptions') or {}, sort_keys=True)}")

    for row in matrix.get("results") or []:
        prefix = f"{row.get('symbol')}/{row.get('timeframe')}/{row.get('strategy')}"
        if row.get("status") != "ok":
            print(f"[UNAVAILABLE] {prefix} :: {row.get('error')}")
            continue

        audit = row.get("audit") or {}
        full = audit.get("Full Backtest") or {}
        oos = audit.get("Out of Sample") or {}
        costs = audit.get("Costs") or {}
        regimes = audit.get("Regime Analysis") or []
        sensitivity = (audit.get("Parameter Sensitivity") or {}).get("Cases") or []
        print(
            f"[OK] {prefix} "
            f"bars={audit.get('Fetched Bars', 'n/a')} "
            f"full_return={_metric(full, 'Total Return [%]')}% "
            f"trades={_metric(full, 'Number of Trades')} "
            f"win_rate={_metric(full, 'Win Rate [%]')}% "
            f"expectancy={_metric(full, 'Expectancy [%]')}% "
            f"max_dd={_metric(full, 'Max Drawdown [%]')}% "
            f"trades_day={_metric(full, 'Trades/Day')} "
            f"oos_return={_metric(oos, 'Total Return [%]')}% "
            f"oos_trades={_metric(oos, 'Number of Trades')} "
            f"oos_expectancy={_metric(oos, 'Expectancy [%]')}% "
            f"cost_drag={_metric(costs, 'Cost Drag [%]')}% "
            f"regimes={len(regimes)} "
            f"sensitivity_cases={len(sensitivity)}"
        )

    available = int(matrix.get("available_rows") or 0)
    total = int(matrix.get("total_rows") or 0)
    label = "MATRIX COMPLETE" if matrix.get("all_ok") else "MATRIX INCOMPLETE"
    print(f"{label}: {available}/{total} audit rows available.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect Phase 5.1 read-only runtime-strategy evidence through the already-running "
            "TradeAgent backend so the audit uses that process's connected cTrader demo session."
        )
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("TRADEAGENT_API_BASE", DEFAULT_BASE_URL),
        help=f"TradeAgent backend URL (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("API_KEY") or None,
        help="Optional API key; defaults to the API_KEY environment variable.",
    )
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout in seconds.")
    parser.add_argument("--num-bars", type=int, default=1500)
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--spread-bps", type=float, default=2.0)
    parser.add_argument("--position-size-pct", type=float, default=100.0)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON evidence output path.")
    return parser


def main() -> int:
    args = _parser().parse_args()
    with requests.Session() as session:
        try:
            matrix = collect_matrix(
                session=session,
                base_url=args.base_url,
                api_key=args.api_key,
                timeout=args.timeout,
                num_bars=args.num_bars,
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
                spread_bps=args.spread_bps,
                position_size_pct=args.position_size_pct,
            )
        except Exception as exc:
            print(f"PHASE 5.1 FIELD VERIFICATION BLOCKED: {exc}")
            return 2

    print_summary(matrix)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(matrix, indent=2, sort_keys=True), encoding="utf-8")
        print(f"EVIDENCE_JSON {args.output}")

    return 0 if matrix.get("all_ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
