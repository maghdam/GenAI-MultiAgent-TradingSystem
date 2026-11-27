"""Local-only proposed-draft approval state machine, NOT a network service.

No HTTP/MCP tools are registered here. Approval is entered in an independent,
interactive operator-owned terminal and bound to a validated name + SHA256.
This module does not save strategies or authorize backtests/broker operations.

A later separately permissioned write service must authenticate its callers
and enforce authorization before using the one-time consumption primitive.
"""

from __future__ import annotations

import re
import secrets
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime

from backend.services.studio_draft_create import DraftCandidate, DraftCreateError, prepare_candidate
from backend.storage.db import get_db


REQUEST_TTL_SECONDS = 600
APPROVAL_TTL_SECONDS = 120
MAX_OUTSTANDING_REQUESTS = 16
_ID_RE = re.compile(r"[a-f0-9]{32}\Z", re.ASCII)


class DraftApprovalError(Exception):
    """Fixed, non-sensitive failure code, never internal DB details or source."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _now() -> int:
    return int(time.time())


def _request_id(value: object) -> str:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        raise DraftApprovalError("invalid_request_id")
    return value


@contextmanager
def _write_transaction():
    with get_db() as db:
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise


def _validated(name: str, source: str) -> DraftCandidate:
    try:
        return prepare_candidate(name, source)
    except DraftCreateError:
        raise DraftApprovalError("invalid_candidate") from None


def request_draft_approval(name: str, source: str) -> dict:
    """Register a *proposal*, never an approval or a reserved strategy name.

    A request ID is only a correlation ID, NOT a bearer authorization token.
    Raw source never enters SQLite. No writes to strategy files or lifecycles.
    """
    candidate = _validated(name, source)
    now = _now()
    request_id = secrets.token_hex(16)
    expiry = now + REQUEST_TTL_SECONDS
    try:
        with _write_transaction() as db:
            count = db.execute(
                """SELECT COUNT(*) AS n FROM studio_draft_approval_requests
                   WHERE expires_at_epoch > ? AND state IN ('pending', 'approved')""",
                (now,),
            ).fetchone()["n"]
            if count >= MAX_OUTSTANDING_REQUESTS:
                raise DraftApprovalError("too_many_pending_requests")
            db.execute(
                """INSERT INTO studio_draft_approval_requests(
                    request_id, strategy, source_sha256, created_at_epoch,
                    expires_at_epoch, state
                ) VALUES (?, ?, ?, ?, ?, 'pending')""",
                (request_id, candidate.name, candidate.source_sha256, now, expiry),
            )
    except DraftApprovalError:
        raise
    except Exception:
        raise DraftApprovalError("approval_store_unavailable") from None
    return {
        "request_id": request_id,
        "strategy": candidate.name,
        "source_sha256": candidate.source_sha256,
        "expires_at_epoch": expiry,
        "status": "pending",
        "warning": "Pending operator review only. No draft saved or trading authorized.",
    }


def _read_pending(request_id: str):
    request_id = _request_id(request_id)
    try:
        with get_db() as db:
            row = db.execute(
                """SELECT strategy, source_sha256, state, expires_at_epoch
                   FROM studio_draft_approval_requests WHERE request_id = ?""",
                (request_id,),
            ).fetchone()
    except Exception:
        raise DraftApprovalError("approval_store_unavailable") from None
    if row is None or row["state"] != "pending":
        raise DraftApprovalError("request_not_pending")
    if int(row["expires_at_epoch"]) <= _now():
        raise DraftApprovalError("approval_expired")
    return row


def approve_from_local_console(request_id: str, reviewed_source: str) -> dict:
    """Operator MUST independently inspect exact local source and type tuple.

    Called only from an explicit trusted local-console CLI, never from MCP.
    No user-supplied approval boolean, token, --yes or noninteractive path.
    """
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise DraftApprovalError("interactive_operator_required")
    row = _read_pending(request_id)
    candidate = _validated(str(row["strategy"]), reviewed_source)
    if candidate.source_sha256 != str(row["source_sha256"]):
        raise DraftApprovalError("source_fingerprint_mismatch")

    print("\nLOCAL RESEARCH DRAFT APPROVAL (NO SAVE)")
    print("Request ID:", request_id)
    print("Strategy:", candidate.name)
    print("Canonical UTF-8 bytes:", len(candidate.canonical_bytes))
    print("Canonical lines:", candidate.source_lines)
    print("Source SHA256:", candidate.source_sha256)
    print("Review the entire local source independently before confirming.")
    print("No strategy will be saved, executed or backtested by this approval.")
    print("Type the exact STRATEGY + SPACE + full SHA256 to approve once.")

    start = time.monotonic()
    try:
        response = input("Approval: ")
    except (EOFError, KeyboardInterrupt):
        raise DraftApprovalError("operator_approval_denied") from None
    if time.monotonic() - start > APPROVAL_TTL_SECONDS:
        raise DraftApprovalError("operator_approval_expired")
    if response != candidate.name + " " + candidate.source_sha256:
        raise DraftApprovalError("operator_approval_denied")

    now = _now()
    try:
        with _write_transaction() as db:
            fresh = db.execute(
                """SELECT strategy, source_sha256, state, expires_at_epoch
                   FROM studio_draft_approval_requests WHERE request_id = ?""",
                (request_id,),
            ).fetchone()
            if (
                fresh is None or fresh["state"] != "pending"
                or int(fresh["expires_at_epoch"]) <= now
            ):
                raise DraftApprovalError("request_not_pending")
            if (fresh["strategy"], fresh["source_sha256"]) != (
                candidate.name, candidate.source_sha256
            ):
                raise DraftApprovalError("source_fingerprint_mismatch")
            expires = min(int(fresh["expires_at_epoch"]), now + APPROVAL_TTL_SECONDS)
            updated = db.execute(
                """UPDATE studio_draft_approval_requests
                   SET state='approved', approved_at_epoch=?,
                       expires_at_epoch=?
                   WHERE request_id=? AND state='pending'""",
                (now, expires, request_id),
            ).rowcount
            if updated != 1:
                raise DraftApprovalError("request_not_pending")
    except DraftApprovalError:
        raise
    except Exception:
        raise DraftApprovalError("approval_store_unavailable") from None

    return {
        "request_id": request_id,
        "strategy": candidate.name,
        "source_sha256": candidate.source_sha256,
        "expires_at_epoch": expires,
        "status": "approved",
        "warning": "One-use research approval only; not a saved strategy or trading authorization.",
    }


def consume_approved_once(request_id: str, name: str, source: str) -> dict:
    """Test-only integration primitive, NOT called by any live write endpoint.

    Atomically consume one outstanding human approval for the exact tuple.
    Future write integration needs additional authenticated caller isolation
    and must design commit atomicity with create-exclusive draft persistence.
    """
    request_id = _request_id(request_id)
    candidate = _validated(name, source)
    now = _now()
    try:
        with _write_transaction() as db:
            row = db.execute(
                """SELECT strategy, source_sha256, state, expires_at_epoch
                   FROM studio_draft_approval_requests WHERE request_id = ?""",
                (request_id,),
            ).fetchone()
            if row is None or row["state"] != "approved":
                raise DraftApprovalError("approval_not_available")
            if int(row["expires_at_epoch"]) <= now:
                raise DraftApprovalError("approval_expired")
            if (row["strategy"], row["source_sha256"]) != (
                candidate.name, candidate.source_sha256
            ):
                raise DraftApprovalError("approval_tuple_mismatch")
            changed = db.execute(
                """UPDATE studio_draft_approval_requests
                   SET state='consumed', consumed_at_epoch=?
                   WHERE request_id=? AND state='approved'""",
                (now, request_id),
            ).rowcount
            if changed != 1:
                raise DraftApprovalError("approval_not_available")
    except DraftApprovalError:
        raise
    except Exception:
        raise DraftApprovalError("approval_store_unavailable") from None
    return {
        "strategy": candidate.name,
        "source_sha256": candidate.source_sha256,
        "approval_consumed": True,
        "warning": "Approval consumed; no strategy was saved or executed.",
    }
