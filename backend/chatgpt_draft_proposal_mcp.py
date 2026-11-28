"""Opt-in, token-gated, LOCAL-ONLY MCP server for DRAFT PROPOSALS, not saving.

This is an independent port and process from TradeAgent's five-tool
unauthenticated read-only research gateway. It has exactly ONE MCP tool, which
creates only an unapproved §33 SQLite proposal record (name/hash/expiry).
No HTTP/MCP tool is allowed to approve or commit an actual strategy.

The transport gate protects HTTP requests; in-process Python/stdio do NOT
confer this authentication boundary and must never be exported as a client.
The caller's Bearer token is an independent, operator-provisioned secret,
never a broker credential and never returned or logged.
"""

from __future__ import annotations

import hmac
import os
import re

from backend.services.studio_draft_approval import (
    DraftApprovalError,
    request_draft_approval,
)

_HOST = b"127.0.0.1:8766"
_PORT = 8766
_MAX_HTTP_BODY_BYTES = 24_576
_TOKEN_RE = re.compile(r"[a-f0-9]{64}\Z", re.ASCII)


def _validate_token(token: object) -> str:
    """Require a full 256-bit independently provisioned local client token."""
    if type(token) is not str or _TOKEN_RE.fullmatch(token) is None:
        raise ValueError("proposal_mcp_token_unavailable")
    return token


def propose_studio_draft_for_review(strategy: str, source: str) -> dict:
    """Create an UNAPPROVED pending correlation ID for a new research draft.

    No source, local path, secret, approval credential or executable code is
    stored by the ledger or returned to the MCP client. The operator must
    independently review a local source file in a separate terminal before
    any subsequent offline approval and save.
    """
    try:
        proposed = request_draft_approval(strategy, source)
    except DraftApprovalError as exc:
        # Fixed codes only; no source, path or upstream exception disclosure.
        raise ValueError(exc.code) from None
    return {
        "status": "pending",
        "request_id": proposed["request_id"],
        "strategy": proposed["strategy"],
        "source_sha256": proposed["source_sha256"],
        "expires_at_epoch": proposed["expires_at_epoch"],
        "warning": (
            "Unapproved proposal only. NO SAVED STRATEGY, BACKTEST, "
            "BROKER CONNECTION OR TRADE. Independent operator review required."
        ),
    }


def create_server():
    """Exactly one *proposal-only* tool; do not register any approval/save."""
    # Optional SDK, never imported for normal TradeAgent server startup.
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("TradeAgent Local Draft Proposal (Restricted)")
    server.tool()(propose_studio_draft_for_review)
    return server


async def _send_error(send, status: int) -> None:
    """Do not reflect requests, tokens, Python source or internal details."""
    from starlette.status import HTTP_401_UNAUTHORIZED

    body = b'{"error":"request_denied"}'
    headers = [
        (b"content-type", b"application/json"),
        (b"cache-control", b"no-store"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    if status == HTTP_401_UNAUTHORIZED:
        headers.append((b"www-authenticate", b"Bearer"))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class _TokenGate:
    """Authenticate the full MCP HTTP transport BEFORE parsing any tool input."""

    def __init__(self, app, token: str) -> None:
        self._app = app
        self._expected = "Bearer " + _validate_token(token)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            # Keep the SDK's own session-manager lifespan intact.
            await self._app(scope, receive, send)
            return
        if scope["type"] != "http":
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            return
        if scope.get("path") not in ("/mcp", "/mcp/"):
            await _send_error(send, 404)
            return
        client = scope.get("client")
        # No remote network access, even if a future launcher is misconfigured.
        if not client or client[0] not in ("127.0.0.1", "::1"):
            await _send_error(send, 403)
            return
        headers = scope.get("headers") or []
        host = [value for key, value in headers if key.lower() == b"host"]
        auth = [value for key, value in headers if key.lower() == b"authorization"]
        # Browser origins, alternate/duplicate hosts and duplicate credentials
        # are denied. The private desktop MCP client uses no Origin header.
        if (
            len(host) != 1 or host[0] != _HOST
            or any(key.lower() == b"origin" for key, _ in headers)
        ):
            await _send_error(send, 403)
            return
        if (
            len(auth) != 1 or len(auth[0]) != len(self._expected)
            or not hmac.compare_digest(auth[0], self._expected.encode("ascii"))
        ):
            await _send_error(send, 401)
            return
        await self._app(scope, receive, send)


def create_app(token: str):
    """Build an explicitly authenticated, independent loopback MCP ASGI app."""
    _validate_token(token)
    server = create_server()
    app = server.streamable_http_app(
        stateless_http=True,
        json_response=True,
        max_request_body_size=_MAX_HTTP_BODY_BYTES,
        host="127.0.0.1",
    )
    return _TokenGate(app, token)


def main() -> None:
    """Manual opt-in ONLY; NEVER auto-start with TradeAgent/read-only MCP."""
    if os.environ.get("TRADEAGENT_ENABLE_DRAFT_PROPOSALS") != "1":
        raise SystemExit("BLOCKED: proposal_mcp_not_opted_in")
    try:
        token = _validate_token(os.environ.get("TRADEAGENT_PROPOSAL_MCP_TOKEN"))
    except ValueError:
        raise SystemExit("BLOCKED: proposal_mcp_token_unavailable") from None

    import uvicorn

    uvicorn.run(
        create_app(token),
        host="127.0.0.1",
        port=_PORT,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
