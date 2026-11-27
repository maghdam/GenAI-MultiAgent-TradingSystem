"""Human-owned terminal review of an existing proposed draft approval request.

This CLI authorizes no file save, MCP write tool, backtest, broker or trade.
A request must first be registered by local-only code; no transport is wired.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.services.studio_draft_approval import DraftApprovalError, approve_from_local_console
from backend.services.studio_draft_create import MAX_SOURCE_BYTES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Approve an existing draft proposal in a human terminal")
    parser.add_argument("--request-id", required=True, help="Existing pending correlation ID, not a credential")
    parser.add_argument("--source-file", type=Path, required=True, help="Independently reviewed local source file")
    args = parser.parse_args(argv)

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("BLOCKED: interactive_operator_required")
        return 2
    try:
        source_file = args.source_file
        if source_file.is_symlink() or not source_file.is_file():
            raise DraftApprovalError("source_file_unavailable")
        with source_file.open("rb") as stream:
            raw = stream.read(MAX_SOURCE_BYTES + 1)
        if len(raw) > MAX_SOURCE_BYTES:
            raise DraftApprovalError("invalid_source_size")
        try:
            source = raw.decode("utf-8")
        except UnicodeError:
            raise DraftApprovalError("invalid_source") from None
        outcome = approve_from_local_console(args.request_id, source)
    except DraftApprovalError as exc:
        print("BLOCKED:", exc.code)
        return 2
    except OSError:
        print("BLOCKED: source_file_unavailable")
        return 2
    print(json.dumps(outcome, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
