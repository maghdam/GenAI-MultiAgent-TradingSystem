# TradeAgent §34 — offline, approval-bound create-only research draft

**Status:** PR candidate only. Not merged or operator-verified. **No ChatGPT
write tool, HTTP route or broker integration** is added by this change.

## Why

§32 shipped the independently human-confirmed local create-exclusive CLI.
§33 shipped a local SQLite request/approval/one-time-consumption ledger but
did **not** connect approval to a draft save. §34 joins these two already
reviewed primitives inside a single **offline, operator-owned** workflow.

An approval is neither an authorization for general Python execution nor a
guarantee that a strategy name remains unused. It authorizes at most one
**new version-1 research draft**, with exact canonical name and source hash.

## Operator-only workflow (after release; NOT required for test acceptance)

In the local `tradeagent-v2` environment, create and **independently inspect**
a new Python source file. Use a new name; existing names cannot be replaced.

1. From an independently trusted local terminal:

   ```powershell
   python -m scripts.studio_approved_draft propose --name new_unique_research_draft --source-file "C:\path\reviewed.py"
   ```

   This validates the name/source and returns an **unapproved** short-lived
   correlation ID and the canonical SHA256; no code is saved. The ID is not
   a secret, token or approval. An outstanding request expires in at most
   600 seconds.

2. Review the same source file in an editor, then explicitly approve in the
   **human-owned** console (no noninteractive flag or simulated consent):

   ```powershell
   python -m scripts.review_studio_draft_request --request-id <request-id> --source-file "C:\path\reviewed.py"
   ```

   The operator must type the exact **name, space, full 64-character SHA256**
   displayed in the terminal. Approval expires within 120 seconds and
   is still **not a strategy save**.

3. Only after approval, explicitly commit the **same** reviewed file:

   ```powershell
   python -m scripts.studio_approved_draft commit --request-id <request-id> --name new_unique_research_draft --source-file "C:\path\reviewed.py"
   ```

   The commit CLI also requires interactive stdin/stdout, but does not prompt
   for duplicate approval: the independent step above already recorded that
   exact decision. A fresh new request/approval is required for another save.

**Never run these operator commands against real Strategy Studio during
initial PR field testing.** Unit tests use temporary strategy paths and
an isolated test SQLite database. Do not reuse the existing saved strategy
names, and never send the underlying source or secrets to an MCP tool.

## One transaction boundary

The new `create_from_approved_local_request` path in
`backend/services/studio_draft_create.py`:

- Re-runs the exact §32 source/name policy and checks canonical bytes/hash
  at commit rather than trusting a preview or a caller-supplied digest.
- Acquires `BEGIN IMMEDIATE` in the same SQLite connection used for
  the lifecycle write; rejects any historical lifecycle with that name.
- Verifies the §33 approval **approved**, not expired, and matching exact
  name and SHA256; transitions it to `consumed` **inside the open DB transaction**.
- Creates the **new** strategy file with `O_CREAT | O_EXCL` in the fixed
  generated-strategy directory. Existing files or symlinks are not overwritten.
  Inserts one `version=1`, `stage=draft` lifecycle row with matching hash.
- Rechecks the approval tuple and its expiry immediately **before** committing
  that same DB transaction. A normal successful DB commit records the new
  lifecycle and consumed approval together, with one exclusively created file.
- If any operation fails **before COMMIT**, rolls back consumed approval
  and lifecycle state and removes only the exact new file inode it owns.
  Existing saved strategies and lifecycle records remain untouched.
- If COMMIT was attempted and its outcome is uncertain, leaves the file
  in place and returns `manual_reconciliation_required`. There is **no
  blind retry**. A process crash between file creation and SQLite COMMIT
  can still produce an orphan file requiring manual reconciliation;
  SQLite and the filesystem do not have a shared atomic transaction.

Returned metadata is only strategy name, SHA256, draft lifecycle version,
saved status, and a no-trading warning. No raw source, source path or credentials.

## Trust boundary and explicit exclusions

This is a *local console* release, not an authenticated ChatGPT write service.
The new functions must **never be registered directly** as MCP tools or
exposed as generic loopback HTTP routes. Being on `127.0.0.1`, knowing the
request correlation ID or passing `approved=true` does **not** prove
independent operator consent. A terminal check does not protect against
malicious code that already runs with arbitrary operator-user privileges.

The existing five-tool private desktop plugin stays **Read-only**. No MCP
restart, new plugin manifest, cloud tunnel, broker connection, automatic
backtest, optimization, executable strategy load, paper/live promotion,
trade, order or account change is introduced.

A separate next roadmap item must design and validate **authenticated,
opt-in, permission-isolated** ChatGPT/Work/Codex request/write transport before
any conversational saving, retaining this same approval binding and
create-exclusive commit boundary.

## Acceptance gates

Focused tests:

```powershell
python -m pytest backend/tests/test_studio_approved_draft.py backend/tests/test_studio_draft_approval.py backend/tests/test_studio_draft_create.py backend/tests/test_strategy_lifecycle.py backend/tests/test_chatgpt_mcp_readonly.py backend/tests/test_tradeagent_desktop_plugin_package.py -q
python scripts/validate.py
git diff --check origin/main...HEAD
git status -sb
```

Require a clean exact GitHub PR SHA, six-file scope, Windows-focused and full
backend + frontend validation, no accidental strategy writes, matching
exact-head GitHub CI, and an independent final documentation-head release
gate before merge. Do not use real operator DB or saved strategy files
for acceptance.
