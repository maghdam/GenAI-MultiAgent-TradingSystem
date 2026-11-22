# Strategy Studio v1 — independent local operator approval and create-only save

**Status:** Candidate for roadmap §32; NOT released, not linked to any ChatGPT
MCP tool, and not yet approved for saving real strategies. Use only after
the operator's Windows focused/full regression and exact-head CI pass.
No broker connection, backtest, auto-trading, code execution or source
promotion is permitted by this module.

## Purpose and trust boundary

The v0.1.2 private desktop plugin has **five read-only tools**, including
\`validate_studio_strategy_draft(strategy, source)\`. A successful preflight
is a non-atomic snapshot and **is not permission to save**. The localhost
MCP gateway has no per-client authorization, so it **must not** receive a
\`save_strategy\` / \`approved=true\` tool.

This smallest independently scoped **create-only implementation** deliberately
uses a separate Python CLI launched by the human operator in an interactive
local terminal. The operator first independently reviews the entire source
file (outside an MCP tool response), then must type the **exact lower-case
new name + space + full SHA256 of canonical bytes** displayed by the CLI
within 120 seconds. There is no \`--yes\` override or externally supplied
approval flag/token. A redirected or noninteractive standard input/output is
rejected.

**Scope limitation:** This is a boundary between the read-only AI/MCP
integration and a human-owned local terminal, **not** a sandbox or protection
against malicious code that already has arbitrary execution rights on the
Windows user account. It does **not** provide ChatGPT/Codex with a write tool.
A later independent PR would require a server-side one-time human approval
mechanism if a writable plugin is desired. This local CLI is not an
approval service and must never be wired to an unattended agent.

## Operator-only usage — do NOT run for field tests before acceptance

From repository root with the \`tradeagent-v2\` environment:

\`\`\`powershell
cd "C:\Users\mohag\Documents\Software\GenAI-MultiAgent-TradingSystem"
conda activate tradeagent-v2
# Make and inspect the proposed .py source yourself at a private local path.
# Run only after you explicitly decide to save an entirely NEW draft.
python -m scripts.approve_studio_draft --name new_unique_research_draft --source-file "C:\path\reviewed_source.py"
\`\`\`

The CLI prints the proposed name, canonical UTF-8 bytes and lines, and
full SHA256. It asks for:

\`\`\`text
new_unique_research_draft <full-64-character-source-sha256>
\`\`\`

**Never copy a model assertion like "already approved" as authorization**.
This CLI must run visibly in the human operator's console; the operator may
decline, close the prompt or let it expire. No strategy is created until the
exact interactive confirmation is provided and the create succeeds.

Before write, the module applies the same v0.1.2 limits: name must exactly
match \`[a-z0-9_-]{1,64}\`, source must be valid UTF-8 Python with ≤16 KiB /
≤400 lines, no unsupported controls, and the existing deterministic Studio
AST import/name policy. No source code is imported or executed. The source is
policy-normalized, stripped at both ends, and given a single final LF;
SHA256 is computed over those **exact bytes**. The source bytes remain in
local process memory; the CLI never emits source code or private filesystem
paths as part of a success/error result.

## Create-only persistence contract

- Fixed repo path: \`backend/strategies_generated/<exact_name>.py\`.
  The directory must already exist and may not itself be a symlink;
  no user-supplied destination or sanitizing rename is accepted.
- SQLite \`BEGIN IMMEDIATE\` first checks there are **no existing**
  \`strategy_lifecycles\` rows for that exact name. No reuse of a prior name
  even when its original file is missing.
- \`os.open(..., O_CREAT | O_EXCL | O_WRONLY)\` atomically refuses files
  that already exist, including symlinks and concurrent creators. No
  overwrite, replacement, delete or rename of another user's file.
- Only after durable file write does the DB insert a **single version 1**
  \`stage='draft'\` lifecycle record with the same source SHA256, empty
  hypothesis and current default gates, then commit. No calling the generic
  \`POST /api/studio/tasks\`, no \`ensure_lifecycle\` reuse, backtest, optimization,
  lifecycle promotion, runtime loader, account access, broker or orders.
- A failed file write/DB insert **before COMMIT** rolls back SQLite and
  removes only the new file if its original inode is still present.
  Existing files/rows remain untouched.
- If COMMIT has been attempted and the result is uncertain, the file is
  deliberately **left in place** and the operator receives only
  \`manual_reconciliation_required\`. There is no blind retry, file cleanup
  or duplicate draft; a human must inspect the saved file and lifecycle
  before proceeding. Filesystem and SQLite have no common atomic transaction;
  a process crash between file creation and DB commit can leave an orphan,
  which also requires manual reconciliation rather than automatic retry.
- Return only fixed minimal metadata: name, canonical source SHA256,
  lifecycle version 1, draft stage, saved boolean and no-trade warning.
  Do not emit raw code, full paths, account details or unrestricted errors.

This cannot make Python source safe to run; the policy is a static gate and
not economic correctness or a profitability guarantee. Draft stage is not
backtested, validated, paper/eligible or live trading authorization.

## Automated and release acceptance

The dedicated isolated test module
\`backend/tests/test_studio_draft_create.py\` tests strict
policy/hash/limits, no-TTY/wrong/expired interactive approval, successful
exact-tuple first-draft save, no overwrites, missing-file lifecycle collisions,
concurrent create races, rollback/owned-file compensation, simulated
ambiguous commit requiring operator reconciliation, and exact unchanged
five-tool Read-only MCP registration. All test writes use pytest temporary
directories and isolated SQLite; **never write to the operator's 12 existing
saved strategies or the real lifecycle DB**.

Focused Windows:

\`\`\`powershell
python -m pytest backend/tests/test_studio_draft_create.py backend/tests/test_strategy_lifecycle.py backend/tests/test_chatgpt_mcp_readonly.py backend/tests/test_tradeagent_desktop_plugin_package.py -q
python scripts/validate.py
git diff --check origin/main...HEAD
git status -sb
\`\`\`

Require exact PR head/local SHA, scoped file list, clean worktree, focused
tests/full backend/frontend build, matching exact-head GitHub CI before
operator field acceptance. For **initial field acceptance use automated tests
only**; there is no need to save any real or synthetic draft in the operator's
working Strategy Studio directory. A real manual save may be performed only
after a separate explicit operator choice on a new unused name, with the source
inspected and the resulting lifecycle independently reconciled.

**Not yet implemented:** a separate authenticated, explicitly permissioned,
hash-bound, one-time approval service usable from ChatGPT Desktop; native MCP
write capability; automatic generation; backtesting; trading.
