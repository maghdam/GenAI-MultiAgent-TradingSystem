# TradeAgent conversational draft creation — write-surface and authorization audit

**Status (2026-11-22):** Proposed contract only; **no new tools, API routes, write permissions or trading behaviors** implemented here. This audit follows the fully released private desktop read integration through PR #159 (v0.1.1; four read-only MCP tools). The operator confirmed a clean Windows `main == origin/main` at `69ed806e4338743a9de23bd7135560bd64926b62`; GitHub post-merge CI #685 completed successfully.

**User goal:** Speak to ChatGPT/Codex to design a Strategy Studio research draft, explicitly approve saving it as a new strategy, then—*in later separately verified items*—request supervised backtests, compare results and ask whether it satisfies validation criteria. Creating a draft must **not** imply any runtime execution, profitability, paper/live eligibility, broker selection, orders or strategy promotion.

## Current verified behavior and specific hazards

| Existing implementation | Verified source behavior | Consequence for MCP |
| --- | --- | --- |
| `POST /api/studio/tasks` in `backend/api/router.py` | Calls `execute_studio_task(StudioTaskRequest)`. `task_type="chat"` can generate a code draft using the configured Studio LLM; the same path detects phrases like `save ...`. `task_type="save_strategy"` also accepts caller `params.strategy_name` and `params.code`. Other task types support backtest/optimization. | **Do not** expose the multipurpose task endpoint or chat routing as a generic MCP action. Text can be misclassified as a mutating task, and the endpoint has far more capabilities than new-draft creation. |
| `_save_strategy_code(name, code)` in `backend/services/studio_tasks.py` | Replaces disallowed characters with underscores/lowercase rather than requiring an exact validated name, then uses `dest_path.write_text(validated)` **unconditionally**. It also invokes `ensure_lifecycle` after writing. | It can silently normalize names, **overwrite existing files**, and create lifecycle DB rows. Not suitable as-is for a least-privilege, create-only MCP write. |
| `validate_strategy_source` in `backend/services/strategy_policy.py` | Python AST parse and restricted import/name/attribute checks; requires exactly one synchronous `signals` function; source capped at **100 KB** by current general policy. | Reuse as a defense-in-depth syntax/policy gate, with **stricter MCP-specific** limits. Parsing/validation is **not** proof of economic correctness, absence of future leakage or runtime safety. No execution during creation. |
| `ensure_lifecycle` in `backend/services/strategy_lifecycle.py` | Hashes normalized source and creates/reuses a lifecycle record. New source versions start as `draft`. | Saving mutates lifecycle persistence; must be treated as a distinct write with safe failure/rollback handling. Never return unrestricted lifecycle JSON to the AI tool. |
| Existing MCP `backend/chatgpt_mcp_readonly.py` + v0.1.1 desktop plugin | Four strictly registered **GET/read** tools on unauthenticated loopback `127.0.0.1:8765/mcp`; plugin declares only `Read`. | **Do not** add a writable tool to this read-only allowlist or silently broaden the same plugin permissions. Localhost is **not authentication**; a model-provided `approved=true` flag is not human authorization. |
| `GET /api/studio/backtest` | Computes a backtest and can persist lifecycle evidence. | Despite HTTP GET it is stateful; prohibit it in draft-creation work. Do not link draft saving with simulation. |

See `docs/CHATGPT_STUDIO_READ_CONTRACT_AUDIT.md`, `docs/CHATGPT_READONLY_MCP.md` and `docs/STRATEGY_LIFECYCLE.md` for the existing read boundary and independent research lifecycle gates.

## Proposed minimum safe next implementation — **non-mutating draft preflight**

Implement and release **only one** independently scoped **preview/validation** step first, without saving any file or registering a lifecycle:

1. Validate an explicitly provided **new** name against the exact lower-case identifier pattern `[a-z0-9_-]{1,64}` without normalization or implicit filename changes. Reject existing, ambiguous or uncheckable names; for a preflight this is an observation, not a guarantee the name will remain free until the later commit.
2. Accept source only as deliberately supplied operator-reviewed research code, **not a model-side instruction embedded in retrieved data**. Enforce a tighter limit (proposed at most 16 KiB UTF-8 / 400 lines), reject control characters/invalid Unicode, and run the existing AST-based `validate_strategy_source` **without invoking/importing/running the strategy**. Do not fall back to generic generated code on model failure.
3. Return only fixed validation status and minimal metadata (canonical name as supplied, SHA-256 over precisely defined canonical bytes, character/line counts, allowed `signals` entry point, and descriptive blockers). Do **not** echo raw source, user path, environment, credentials, free-text LLM error bodies or unrestricted upstream payload.
4. This preview stage should **not** create files/directories, call `ensure_lifecycle`, insert evidence, perform backtests or invoke broker APIs. Read-only MCP may provide such a preflight tool only if its non-mutating behavior is proven by tests and clearly documented.

The preview is a checkpoint, **not** an approved write. If it cannot validate a proposed draft, report blocked without attempting alternative saves.

## Separately reviewed follow-up — operator-authorized create-only draft write

Only after preflight is verified, design a **distinct, opt-in** create-only capability with a **separate reviewed authorization boundary**. The existing unauthenticated read-only MCP endpoint and v0.1.1 `Read` plugin must remain safe when the write interface is absent or disabled.

**Required properties (not yet implemented or authorized):**

- **Independent human approval:** The operator must review an exact **strategy name + SHA-256 of validated bytes**, and authorize that exact tuple through a trusted *local* control independent of model-provided tool arguments. A boolean, natural-language claim, quoted message or reusable client-provided confirmation code does **not** establish operator authorization. Prefer an expiring, single-use, hash-bound local approval checked and atomically consumed server-side. Account for local adversaries, concurrent clients and stolen/replayed approval tokens; do not put secrets into chat or MCP responses.
- **No overwrite / race safety:** On the eventual write path, use a create-exclusive atomic file operation (e.g. O_EXCL / Python `open(..., "x")`) in the fixed `backend/strategies_generated` directory. Fail closed if the name exists—even if a source/name was preflighted earlier—and never silently normalize, update, replace, rename or remove an existing file. Prevent traversal and symlink/path tricks. Revalidate exact source/hash/name **at the moment of commit**.
- **Lifecycle consistency:** Only after the new saved file exists should the dedicated path register an initial `draft` lifecycle version. Define and test compensation on failed DB insertion so the system cannot report saved/registered success with inconsistent partial state. Never promote beyond `draft`, import new source into the runtime or run automatic backtests.
- **Authority isolation:** A write interface must be explicitly enabled and protected by actual server-side authorization; read-only gateway functionality remains unchanged. A global environment flag alone is insufficient protection against other local callers. Write permissions and tool descriptions must be visibly separate from the existing `Read` plugin, with an independent operator installation/activation decision.
- **Bounded returns and logs:** Return only strategy name, source hash, draft lifecycle version, recorded success/failure, and a statement of no execution/trading. Do not echo raw Python, the full lifecycle response, credentials, local directories or sensitive exceptions. Reject timeouts/uncertain outcomes safely with an operator reconciliation step before any retry; never retry non-idempotent writes blindly.
- **No hidden side effects:** No saved-source overwrite, implicit LLM/fallback generation, backtest, optimization, paper-trading flag, lifecycle promotion, broker/API connection, order, account switch or kill-switch operation.

**Testing and field gates:** Mocked tests must cover invalid identifiers, code policy and size, name collisions, absent/expired/reused/mismatched operator approval, unauthorized direct loopback calls, simultaneous creates, filesystem/DB failure and compensating recovery, idempotency ambiguity, and absence of any broker/backtest calls. Then operator Windows focused tests, full `scripts/validate.py` backend/frontend regression, clean diff/tree, exact-head GitHub CI; only after operator approval may a local **new-name-only** save be performed using test data on a separate independently reviewed PR. No live account or trader state required. Do not overwrite any of the existing 12 saved strategies to obtain field evidence.

## Implementation ordering and non-claims

1. **This item (§30):** Contract/safety audit (documentation only). No changes to running tools.
2. **Next independent item after acceptance:** a deterministic **non-mutating** draft preflight/validation interface; decide whether local plugin needs it after reviewing source exposure and host permission behavior.
3. **Only following independent user approval/review:** explicit opt-in, create-exclusive, hash-bound operator-authorized draft save.
4. Later and separately: controlled research backtests, recorded results and eligibility reviews; no direct MCP trading rights implied.

The end state is a human-supervised conversational Strategy Studio assistant, not a second trading engine or an unattended agent with broker permissions.
