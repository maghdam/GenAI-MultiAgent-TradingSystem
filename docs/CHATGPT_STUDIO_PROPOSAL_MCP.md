# TradeAgent §35 — separate token-gated **proposal-only** local MCP

**Status: draft PR candidate, NOT released / not installed.**
This is the next small piece of the authenticated Strategy Studio integration,
not the completed ChatGPT saving workflow.

## What changes, and what emphatically does not

§32 released an operator-owned create-exclusive research draft CLI.
§33 released an exact-name/source-hash, short-lived human approval ledger.
§34 joined one-time approval consumption to create-only lifecycle persistence
in an offline transaction, with explicit crash-reconciliation limits.

§35 introduces **one separately launched, explicitly opted-in local MCP server**
at `http://127.0.0.1:8766/mcp`, with exactly ONE tool:

- `propose_studio_draft_for_review(strategy, source)`: validate a new strategy
  name and caller-provided Python source under the existing §33/§32 policy,
  and create only an **unapproved** short-lived request row. The tool returns
  request correlation ID, exact canonical name, SHA256, expiry, and warning.
  **It cannot approve, consume approval, save a file, create a lifecycle,
  backtest, connect a broker, promote a strategy or trade.**

The prior private five-tool `tradeagent-research` plugin and its read-only
`http://127.0.0.1:8765/mcp` listener remain entirely unchanged.
**No new desktop plugin is installed or started by this PR.**
No change is made to `start-local.cmd`, the broker backend, generic
`/api/studio/tasks` endpoint, strategy runtime, or read-only MCP plugin
manifest. A proposal request is not a reserved strategy name.

## Authentication and minimum trust boundary

The independent MCP server has its own 256-bit lowercase hexadecimal Bearer
token. **Do not reuse cTrader, Alpaca, GitHub, ChatGPT or broker credentials.**
A token must be provisioned by the operator outside Git and outside chat;
never place it in a committed plugin descriptor or issue/CI log.

The server **refuses to start** unless both environment values are present:

- `TRADEAGENT_ENABLE_DRAFT_PROPOSALS=1` (explicit manual operator opt-in).
- `TRADEAGENT_PROPOSAL_MCP_TOKEN`: 64 lowercase hex characters (32 bytes).
  Generate fresh random bytes securely; rotate by stopping the server,
  replacing the value in server and client configuration, and restarting.

Even on loopback, localhost is not authentication. The ASGI gate denies
unauthenticated MCP requests **before** parsing tools; uses constant-time
token comparison, requires exactly one `Authorization: Bearer <token>`
header, exact `Host: 127.0.0.1:8766`, rejects all `Origin` headers and
non-loopback clients, and exposes **only** `/mcp`. The MCP SDK also applies
its own localhost transport checks. Request-body limit is 24 KiB.
No token, strategy Python source, filesystem path, broker details or
underlying Python exception text is returned by HTTP denials.

**Limitations:** This is an operator-provisioned static token, not OAuth
login, hardware-backed consent, OS-level process identity or remote/cloud
authorization. Any other process with this local token can submit proposals;
it still cannot approve or save them. A malicious process already executing
with the operator's privileges may read configuration or call Python directly.
The HTTP gate does not apply to an in-memory Python call or stdio transport;
neither is supported as a privileged client. Rotating a static local token
requires manual action. This does **not** make private localhost reachable
from ChatGPT Work's cloud environment or justify a tunnel.

**Source privacy:** Supplying `source` to ChatGPT can disclose that code
to a connected model/client. Never include secrets, private account data,
API keys, broker credentials or private code the operator does not intend
to share. Input policy is a syntax restriction, not a confidentiality scan.

## Local setup only after code + Windows/native-host acceptance

Do not activate this draft server or install any new MCP plugin for initial
PR field acceptance. The tests use isolated SQLite and a synthetic ASGI
transport, with no broker or real strategies.

For a later independently authorized local native-client smoke, an operator
could generate a 256-bit local secret in a trusted PowerShell session:

```powershell
$env:TRADEAGENT_ENABLE_DRAFT_PROPOSALS = "1"
$env:TRADEAGENT_PROPOSAL_MCP_TOKEN = python -c "import secrets; print(secrets.token_hex(32))"
python -m backend.chatgpt_draft_proposal_mcp
```

The server stays in that terminal. In a **separate, private** MCP client
configuration (not the repository and not a chat message), the connector
needs the exact `http://127.0.0.1:8766/mcp` endpoint and a Bearer
`Authorization` header matching the generated token. The token value
must be communicated to that client by an operator-controlled secret
mechanism, not posted into ChatGPT or committed to Git. The current
`tradeagent-research` plugin remains on `8765`; it is not converted into
a writer. A native desktop client might reject a nonstandard static-token
configuration or include an `Origin` header; acceptance must verify
actual host behavior and fail closed, not weaken auth to make it attach.

This request-only tool deliberately cannot complete the save path through
MCP. The trusted human terminal still independently reviews and approves
the exact local source file with §33, and separately commits with the
released §34 CLI. A **future PR** must design the protected approval/save
transport and end-to-end host consent before a ChatGPT-accessible save tool
is released. Do not give the model the human operator's terminal or any
generic command-execution permission to bypass these controls.

## Tests and release gates

```powershell
python -m pytest backend/tests/test_chatgpt_draft_proposal_mcp.py backend/tests/test_studio_approved_draft.py backend/tests/test_studio_draft_approval.py backend/tests/test_studio_draft_create.py backend/tests/test_chatgpt_mcp_readonly.py backend/tests/test_tradeagent_desktop_plugin_package.py -q
python scripts/validate.py
git diff --check origin/main...HEAD
git status -sb
```

Test acceptance checks unauthorized/malformed/duplicate token rejection,
wrong host, remote peer and cross-origin rejection, fixed tool allowlist,
pending-only metadata, opt-in startup guard and absence of real saves.
The ASGI unit tests validate the gate with a fake downstream app;
**they do not prove live native MCP handshake/authorization**. That
requires separately approved desktop-host testing once Windows checks and
GitHub CI pass. No actual saved strategy, broker connection, backtest,
paper/live order or trade is permitted by PR acceptance.
