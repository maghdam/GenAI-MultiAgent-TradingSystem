"""Operator-owned OFFLINE proposal / approved draft commit, never an MCP tool.

Procedure:
  1. propose --name NEW --source-file reviewed.py  (pending correlation ID)
  2. python -m scripts.review_studio_draft_request --request-id ID --source-file reviewed.py
     (independent human types exact name + full source hash in a terminal)
  3. commit --request-id ID --name NEW --source-file reviewed.py

No network gateway, backtest, broker, source execution, strategy promotion or
orders. All field acceptance must use isolated pytest storage, not real saves.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.services.studio_draft_approval import (
    DraftApprovalError,
    request_draft_approval,
)
from backend.services.studio_draft_create import (
    MAX_SOURCE_BYTES,
    DraftCreateError,
    create_from_approved_local_request,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline human-approved research draft proposal/save")
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("propose", "commit"):
        action = actions.add_parser(name)
        action.add_argument("--name", required=True, help="Exact new lower-case strategy identifier")
        action.add_argument("--source-file", required=True, type=Path, help="Independently reviewed local source")
        if name == "commit":
            action.add_argument("--request-id", required=True, help="Human-approved proposal correlation ID")
    args = parser.parse_args(argv)

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("BLOCKED: interactive_operator_required")
        return 2

    try:
        source_path = args.source_file
        if source_path.is_symlink() or not source_path.is_file():
            raise DraftCreateError("source_file_unavailable")
        with source_path.open("rb") as stream:
            raw = stream.read(MAX_SOURCE_BYTES + 1)
        if len(raw) > MAX_SOURCE_BYTES:
            raise DraftCreateError("invalid_source_size")
        try:
            source = raw.decode("utf-8")
        except UnicodeError:
            raise DraftCreateError("invalid_source") from None

        if args.action == "propose":
            result = request_draft_approval(args.name, source)
        else:
            result = create_from_approved_local_request(args.request_id, args.name, source)
    except (DraftCreateError, DraftApprovalError) as exc:
        print("BLOCKED:", exc.code)
        return 2
    except OSError:
        print("BLOCKED: source_file_unavailable")
        return 2

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
