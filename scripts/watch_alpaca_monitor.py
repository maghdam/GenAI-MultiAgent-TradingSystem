"""Operator-started local watchdog of Phase 10.21 *completed* sweep timestamps.

No API credentials, Alpaca HTTP requests, orders, external notification or
automatic startup. To perform one offline check from the project root:
  python -m scripts.watch_alpaca_monitor --once
The nonzero exit code can be consumed by a separately configured supervisor;
this command does NOT install or enable a Windows scheduled task.
"""
from __future__ import annotations

import argparse
import json
import sys

from backend.services.alpaca_bracket_watchdog import explicit_watchdog_cycles
from backend.storage.db import init_db


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local SQLite-only Alpaca monitor watchdog")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="Check monitor heartbeat once")
    mode.add_argument("--continuous", action="store_true",
                      help="Check repeatedly until operator stops this separate process")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--stale-after", type=int, default=120)
    args = parser.parse_args(argv)
    if not 10 <= args.interval <= 3600 or not 30 <= args.stale_after <= 86400:
        parser.error("Interval must be 10..3600s; stale threshold 30..86400s")

    try:
        init_db()  # Local DB schema only; no broker access.
        for result in explicit_watchdog_cycles(
            interval_seconds=args.interval, max_age_seconds=args.stale_after,
            cycles=1 if args.once else None,
        ):
            print(json.dumps(result, sort_keys=True), flush=True)
            if result["severity"] == "critical":
                print("ACTION REQUIRED: inspect System incidents and the Paper monitor process. "
                      "No recovery orders are authorized.", file=sys.stderr, flush=True)
            if args.once:
                return 2 if result["severity"] == "critical" else 0
    except KeyboardInterrupt:
        print("Watchdog stopped. No independent liveness assurance.", file=sys.stderr)
        return 130
    except Exception:
        print("Watchdog database check failed; liveness UNVERIFIED. "
              "Operator intervention required.", file=sys.stderr, flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
