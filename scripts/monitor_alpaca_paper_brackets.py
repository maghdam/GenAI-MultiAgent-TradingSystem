"""Explicit operator-only Paper bracket monitoring; NEVER started by backend boot.

Usage (ONLY after read-only broker field queries are separately authorized):
  python -m scripts.monitor_alpaca_paper_brackets --once --acknowledge-paper-reads
  python -m scripts.monitor_alpaca_paper_brackets --continuous --acknowledge-paper-reads
"""
from __future__ import annotations

import argparse
import json

from backend.services.alpaca_bracket_monitor import explicit_monitor_cycles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Operator-started Alpaca Paper READ-ONLY monitor")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="One read-only sweep")
    mode.add_argument("--continuous", action="store_true", help="Keep polling while this process runs")
    parser.add_argument("--acknowledge-paper-reads", action="store_true",
                        help="Confirm separately authorized real Paper broker read-only queries")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--max-intents", type=int, default=25)
    args = parser.parse_args(argv)
    if not args.acknowledge_paper_reads:
        parser.error("Real broker read-only field queries need explicit operator authorization")
    print("READ-ONLY PAPER MONITOR: incident escalation is local; NO ORDERS or automatic recovery.", flush=True)
    print("NOT an unattended protection guarantee. Stop with Ctrl+C; restart manually after reboot.", flush=True)
    try:
        for result in explicit_monitor_cycles(
            interval_seconds=args.interval, max_intents=args.max_intents,
            cycles=None if args.continuous else 1,
        ):
            print(json.dumps(result, sort_keys=True), flush=True)
    except KeyboardInterrupt:
        print("Monitor stopped by operator. Protection is NOT being monitored.", flush=True)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
