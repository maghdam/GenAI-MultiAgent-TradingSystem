# TradeAgent §36 — separate private proposal plugin package (NOT installed)

**Status: draft PR candidate; native host attachment UNVERIFIED.**
The §35 token-gated pending-proposal gateway at
`http://127.0.0.1:8766/mcp` was merged in PR #165. This item bundles a
**separately selectable** local plugin identity and a narrow proposal skill,
rather than broadening the existing research plugin.

## User-facing distinction

| Plugin | Scope | Auth and lifecycle |
| --- | --- | --- |
| `tradeagent-research` | Five existing read-only market/Studio tools, `127.0.0.1:8765/mcp` | Remains `Read`; no new permissions, token or behavior |
| `tradeagent-proposal` | Exactly one `propose_studio_draft_for_review(strategy, source)` tool at `127.0.0.1:8766/mcp` | Declares `Write` because pending ledger metadata is created; requires separate operator-managed authentication; **cannot approve or save** |

The repo-local marketplace now makes both plugins separately **AVAILABLE**,
with `authentication: ON_INSTALL`; that is **not an installation**, a
server start, or automatic permission grant. The operator must explicitly
review and enable the proposal plugin to use it. Nothing in
`start-local.cmd` auto-starts the proposal server.

The new portable bundle contains only:

- `plugins/tradeagent-proposal/plugin.json` — identity, truthful
  metadata-`Write` capability and local-executor requirement.
- `plugins/tradeagent-proposal/mcp.json` — one Streamable HTTP endpoint,
  exactly `127.0.0.1:8766/mcp`. **No credential, placeholder or headers.**
- `plugins/tradeagent-proposal/skills/proposal-review/SKILL.md` — explicit
  proposal-only authorization and hard **NO SAVE / NO TRADE** instructions.

The package has **no commands, hooks, local-execution tools, browser tunnel,
source inspection bypass or strategy file-saving tool**.

## Why the token is not inside mcp.json

The Agent Plugins 1.0.0 portable `mcp.json` header values are **literal
visible package data**, with no portable environment variable interpolation.
Hard-coding a bearer token would disclose the credential to everyone
with access to the repo or installed plugin. An apparent
`"${TRADEAGENT_PROPOSAL_MCP_TOKEN}"` substitute would be forwarded literally
by a conformant client, not securely resolved. This package intentionally
does neither.

The gateway continues to demand an independent, operator-generated **256-bit
lower-case hex bearer token**, exact host `127.0.0.1:8766`, no `Origin`
header and local peer. It refuses to start unless
`TRADEAGENT_ENABLE_DRAFT_PROPOSALS=1` and
`TRADEAGENT_PROPOSAL_MCP_TOKEN` are properly set.

**Expected default result after merely discovering this plugin:**
the proposal server is off; no request can be submitted. If the server
is manually started but the client lacks token authorization, it returns
HTTP 401 and must not submit a proposal. That is a safe failure, not a
reason to remove authentication.

## Native Codex / ChatGPT host attachment — separate field gate

Current Codex documentation supports `bearer_token_env_var` for
**user-scoped** Streamable HTTP MCP connections. A candidate private
user-configured connection could look like:

```toml
[mcp_servers.tradeagent-local-proposal]
url = "http://127.0.0.1:8766/mcp"
bearer_token_env_var = "TRADEAGENT_PROPOSAL_MCP_TOKEN"
enabled_tools = ["propose_studio_draft_for_review"]
default_tools_approval_mode = "prompt"
```

This is a **reference example**, not a command to paste now.
It does not store the secret in the TOML file. The environment variable
must already be provided securely to the *native Codex process* as well
as to the separately launched Python proposal gateway; different launch
environments may not share it. The **precedence/merging behavior between a
user-scoped MCP entry and the same server packaged in a plugin has not yet
been verified on the operator's exact Windows host version**. Do not
register duplicate active connections until that behavior is confirmed.
Do not write a real bearer secret to TOML, plugin files, chat messages,
Git, screenshot captures or logs.

A client-managed API-key prompt may be supported by some OpenAI-native
hosts, but the historically tested Codex host rejected server-level
`extensions` metadata (`unknown field extensions`). We therefore
**do not add** an unproven vendor extension or any shared/static
authorization header to this portable package. A separate operator
Windows native-host smoke must identify the actual supported credential
source and prove that it attaches **exactly one** proposal tool, with
appropriate tool approval and no duplicate research or saved-strategy
write tools. A `401`, malformed Host, missing token, rejected plugin
config or unavailable local executor is a **BLOCKED** attachment.

ChatGPT Work running in the cloud cannot reach `127.0.0.1` on the
operator's Windows computer. This package does not enable remote or
cloud connectivity, and **no public relay, proxy, firewall change,
port forwarding or tunnel** should be added.

## Human consent remains separate from proposing

The permitted MCP tool only checks canonical name/source and records
pending request metadata: correlation ID, name, SHA256 and expiry.
It cannot approve or consume that request and cannot create an
on-disk strategy. A pending request expires; it is **not approval**,
a name reservation, a trade signal, or a broker/backtest authorization.

The operator must independently inspect a trusted local source file
and use the released §33 console review and §34 offline create-only
commit. The AI client has **no access** to that terminal, the request
secret, any account or broker credential or the strategy files. Nothing
in this PR exercises the human approval/commit sequence or publishes
it through MCP.

## PR acceptance versus future host test

First validate **without** installing, starting the proposal server, or
configuring any credential:

```powershell
python -m pytest backend/tests/test_tradeagent_proposal_plugin_package.py backend/tests/test_tradeagent_desktop_plugin_package.py backend/tests/test_chatgpt_draft_proposal_mcp.py backend/tests/test_studio_approved_draft.py backend/tests/test_chatgpt_mcp_readonly.py -q
python scripts/validate.py
git diff --check origin/main...HEAD
git status -sb
```

The tests check exact three-file plugin package, two separately available
local marketplace entries, `Write` labeling for pending metadata,
no secret/header in source, existing five-tool `Read` package untouched,
and fixed/disjoint gateway tool allowlists. They do **not** validate a
real native ChatGPT attachment, operator consent, cloud access, an actual
strategy save, broker connectivity, a backtest or any trade.

After Windows tests and matching GitHub CI pass, record evidence and retest
the final documentation-only PR head before ready/squash merge. Following
clean local-main synchronization and post-merge CI, a separately authorized
**native Windows host attachment smoke** can test consent and secure
credential delivery with fake/no-secret-revealing metadata. At that
stage obtain the operator's explicit installation action and never
copy a secret into chat. No token or live plugin installation is needed
for the present PR.
