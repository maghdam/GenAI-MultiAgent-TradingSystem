"""Independent local-console approval for creating a NEW Strategy Studio draft.

Deliberately not an HTTP endpoint, MCP tool or broker action. A model-supplied
approval string cannot authorize a write through the existing Read-only plugin.
The trusted operator runs the CLI from a separate interactive Windows terminal.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from backend.services.strategy_lifecycle import DEFAULT_GATES
from backend.services.strategy_policy import StrategyPolicyError, validate_strategy_source
from backend.storage.db import get_db


STRATEGY_DIR = Path(__file__).resolve().parents[1] / "strategies_generated"
MAX_SOURCE_BYTES = 16_384
MAX_SOURCE_LINES = 400
APPROVAL_SECONDS = 120
_NAME_RE = re.compile(r"[a-z0-9_-]{1,64}", re.ASCII)


class DraftCreateError(Exception):
    """Fixed, non-sensitive outcome for an operator-controlled draft save."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class DraftCandidate:
    name: str
    canonical_bytes: bytes
    source_sha256: str
    source_lines: int


def prepare_candidate(name: str, source: str) -> DraftCandidate:
    """Apply the exact v0.1.2 preflight limits without executing source."""
    if type(name) is not str or not _NAME_RE.fullmatch(name):
        raise DraftCreateError("invalid_strategy_name")
    if type(source) is not str:
        raise DraftCreateError("invalid_source")
    try:
        raw = source.encode("utf-8")
    except UnicodeError:
        raise DraftCreateError("invalid_source") from None
    if not source.strip() or len(raw) > MAX_SOURCE_BYTES:
        raise DraftCreateError("invalid_source_size")
    if len(source.splitlines()) > MAX_SOURCE_LINES:
        raise DraftCreateError("invalid_source_lines")
    if any(ord(char) < 32 and char not in "\t\n\r" for char in source):
        raise DraftCreateError("invalid_source_controls")
    try:
        normalized = validate_strategy_source(source)
        canonical = normalized.strip() + "\n"
        encoded = canonical.encode("utf-8")
        if len(encoded) > MAX_SOURCE_BYTES or len(canonical.splitlines()) > MAX_SOURCE_LINES:
            raise DraftCreateError("invalid_source_size")
        validate_strategy_source(canonical)
    except (StrategyPolicyError, SyntaxError, UnicodeError, RecursionError, MemoryError, ValueError):
        raise DraftCreateError("source_policy_rejected") from None
    return DraftCandidate(
        name=name,
        canonical_bytes=encoded,
        source_sha256=hashlib.sha256(encoded).hexdigest(),
        source_lines=len(canonical.splitlines()),
    )


def _same_created_file(path: Path, identity: tuple[int, int] | None) -> bool:
    if identity is None:
        return False
    try:
        stat = path.lstat()
    except OSError:
        return False
    return (stat.st_dev, stat.st_ino) == identity and not path.is_symlink()


def _create_exclusively(
    candidate: DraftCandidate, *, approved_request_id: str | None = None
) -> dict:
    """Atomically create the file, then commit one draft lifecycle row.

    SQLite and the filesystem cannot share a transaction. Before COMMIT starts,
    a failed insert/write is rolled back and the exact created inode is removed.
    If COMMIT may have occurred, retain the file and require manual
    reconciliation; never blindly retry or remove a potentially saved draft.
    """
    # Extra approval checks are relevant only to the new offline bound-save
    # path. The released §32 human-confirmed CLI keeps its existing behavior.
    # Import here to avoid a module-initialization cycle.
    approvals = None
    if approved_request_id is not None:
        from backend.services import studio_draft_approval as approvals

        # Revalidate exact code and SHA at commit, not merely at preflight.
        if type(candidate) is not DraftCandidate:
            raise DraftCreateError("invalid_source")
        try:
            source = candidate.canonical_bytes.decode("utf-8")
        except (AttributeError, UnicodeError):
            raise DraftCreateError("invalid_source") from None
        checked = prepare_candidate(candidate.name, source)
        if checked != candidate:
            raise DraftCreateError("invalid_source")

    directory = STRATEGY_DIR
    if directory.is_symlink() or not directory.is_dir():
        raise DraftCreateError("strategy_directory_unavailable")

    dest = directory / (candidate.name + ".py")
    created = False
    identity: tuple[int, int] | None = None
    commit_started = False
    try:
        with get_db() as db:
            try:
                db.execute("BEGIN IMMEDIATE")
                # Names with an existing lifecycle, even if no file remains,
                # are never eligible for a new first draft.
                row = db.execute(
                    "SELECT 1 FROM strategy_lifecycles WHERE strategy = ? LIMIT 1",
                    (candidate.name,),
                ).fetchone()
                if row is not None:
                    raise DraftCreateError("strategy_lifecycle_already_exists")

                if approvals is not None:
                    try:
                        approvals._consume_in_transaction(
                            db, approved_request_id, candidate, approvals._now()
                        )
                    except approvals.DraftApprovalError:
                        raise DraftCreateError("operator_approval_not_available") from None

                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
                try:
                    fd = os.open(dest, flags, 0o600)
                except FileExistsError:
                    raise DraftCreateError("strategy_name_already_exists") from None
                created = True
                with os.fdopen(fd, "wb") as stream:
                    stat = os.fstat(stream.fileno())
                    identity = (stat.st_dev, stat.st_ino)
                    stream.write(candidate.canonical_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())

                now = datetime.now(UTC).replace(tzinfo=None).isoformat()
                db.execute(
                    """
                    INSERT INTO strategy_lifecycles(
                        strategy, version, version_hash, stage, hypothesis,
                        gates_json, created_at, updated_at
                    ) VALUES (?, 1, ?, 'draft', '', ?, ?, ?)
                    """,
                    (
                        candidate.name,
                        candidate.source_sha256,
                        json.dumps(DEFAULT_GATES),
                        now,
                        now,
                    ),
                )
                if approvals is not None:
                    try:
                        approvals._verify_consumed_before_commit(
                            db, approved_request_id, candidate
                        )
                    except approvals.DraftApprovalError:
                        raise DraftCreateError("operator_approval_not_available") from None
                commit_started = True
                db.commit()
            except Exception as exc:
                try:
                    db.rollback()
                except Exception:
                    # Unknown DB state must never trigger automatic file removal.
                    commit_started = True

                if created and commit_started:
                    raise DraftCreateError("manual_reconciliation_required") from None
                if created:
                    if not _same_created_file(dest, identity):
                        raise DraftCreateError("manual_reconciliation_required") from None
                    try:
                        dest.unlink()
                    except OSError:
                        raise DraftCreateError("manual_reconciliation_required") from None

                if isinstance(exc, DraftCreateError):
                    raise
                if isinstance(exc, sqlite3.IntegrityError):
                    raise DraftCreateError("strategy_lifecycle_already_exists") from None
                raise DraftCreateError("draft_create_failed") from None
    except DraftCreateError:
        raise
    except Exception:
        raise DraftCreateError("draft_create_failed") from None

    return {
        "strategy": candidate.name,
        "source_sha256": candidate.source_sha256,
        "lifecycle_version": 1,
        "stage": "draft",
        "saved": True,
        "warning": "Draft saved only. No source execution, backtest, broker connection or trade.",
    }


def create_from_interactive_console(candidate: DraftCandidate) -> dict:
    """Prompt in a real local terminal, independent of all MCP inputs.

    No --yes flag, remote HTTP route, approved=true parameter or reusable token.
    This boundary protects the v0.1.2 MCP plugin; it is NOT a protection
    against an attacker already able to execute arbitrary code on this PC.
    """
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise DraftCreateError("interactive_console_required")

    print("\nLOCAL OPERATOR APPROVAL — CREATE-ONLY STRATEGY STUDIO DRAFT")
    print("New name:", candidate.name)
    print("Canonical UTF-8 bytes:", len(candidate.canonical_bytes))
    print("Canonical lines:", candidate.source_lines)
    print("Exact source SHA256:", candidate.source_sha256)
    print("Confirm only after independently reviewing the entire source.")
    print("This does NOT run a backtest, connect to a broker or authorize trading.")
    print("Type the exact NAME + SPACE + FULL SHA256 to approve once.")
    started = time.monotonic()
    try:
        confirmation = input("Approval: ")
    except (EOFError, KeyboardInterrupt):
        raise DraftCreateError("operator_approval_denied") from None
    if time.monotonic() - started > APPROVAL_SECONDS:
        raise DraftCreateError("operator_approval_expired")
    if confirmation != candidate.name + " " + candidate.source_sha256:
        raise DraftCreateError("operator_approval_denied")
    return _create_exclusively(candidate)


def create_from_approved_local_request(request_id: str, name: str, source: str) -> dict:
    """Offline §34 create-only commit after independent §33 human approval.

    The approval's atomic consumed transition, new file, and version-1 draft
    lifecycle row share one SQLite transaction. No broker/backtest/execution.
    NOT an authenticated service: only the operator-owned CLI may call this
    function in this release. Do not expose it directly to MCP/HTTP.
    """
    candidate = prepare_candidate(name, source)
    return _create_exclusively(candidate, approved_request_id=request_id)
