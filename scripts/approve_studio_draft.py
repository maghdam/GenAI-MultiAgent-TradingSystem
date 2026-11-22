"""Trusted local-terminal-only Strategy Studio draft creation.

Usage from repository root:
  python -m scripts.approve_studio_draft --name new_research_name --source-file path/to/reviewed.py

Never expose as an MCP tool, unattended automation, HTTP endpoint, or broker
workflow. A completed §31 preflight is NOT approval to save.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.services.studio_draft_create import (
    MAX_SOURCE_BYTES,
    DraftCreateError,
    create_from_interactive_console,
    prepare_candidate,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Operator-approved create-only research draft")
    parser.add_argument("--name", required=True, help="Exact new lower-case strategy name")
    parser.add_argument("--source-file", required=True, type=Path, help="Locally reviewed Python source")
    args = parser.parse_args(argv)

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("BLOCKED: interactive_console_required")
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

        candidate = prepare_candidate(args.name, source)
        result = create_from_interactive_console(candidate)
    except DraftCreateError as exc:
        print("BLOCKED:", exc.code)
        return 2
    except OSError:
        print("BLOCKED: source_file_unavailable")
        return 2

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
