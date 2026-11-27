# TradeAgent §33 — local, non-executing draft-approval ledger

**Status:** Candidate, NOT released. No network endpoint, plugin capability or strategy-saving authority is enabled.

## Purpose and narrow scope

Roadmap §32 implemented a separate human-operated create-only draft CLI. This item is the **next prerequisite** for eventual ChatGPT Desktop strategy saving: a short-lived, one-use, exact-name-and-source-fingerprint approval ledger that **does not save or execute code**.

The existing five `tradeagent-research` MCP tools stay Read-only on their original unauthenticated loopback gateway. This PR introduces no sixth tool, Write permission, HTTP route, model `approved=true` flag, backtest, broker connection, strategy promotion or trade.

## State machine

- `request_draft_approval(name, source)` applies the existing `prepare_candidate` deterministic name/code policy before recording a **pending** request in local SQLite. It stores only a random 128-bit correlation ID, canonical validated name, SHA256, timestamps and state; **never raw Python, source paths, credentials or approval secrets**. Maximum 16 outstanding requests, pending lifetime at most 600 seconds. A request ID **is not a bearer credential or approval**.
- The operator may independently inspect the proposed source in a local editor and run a **separate trusted terminal**: `python -m scripts.review_studio_draft_request --request-id <id> --source-file <reviewed.py>`. The command requires real interactive stdin and stdout, revalidates source/name/hash and requires typing the **exact strategy name plus full source SHA256** within 120 seconds. Any mismatch fails closed. An approved request expires within 120 seconds and never beyond its original pending deadline.
- `consume_approved_once(request_id, name, source)` is a **local, test-only integration primitive**, not linked to a write route. It checks state, expiry and canonical tuple again under a SQLite `BEGIN IMMEDIATE` transaction; one simultaneous consumer can succeed, replay cannot. It **does not save strategy code or create a lifecycle**.

## Important limits of the security boundary

The operator console is separate from the AI-visible read tools, but it is **not** an OS-enforced privilege boundary against arbitrary code already running on the operator's PC. The state-machine library itself must **not** be directly exposed to unauthenticated HTTP or MCP. In particular, a model-supplied confirmation flag, a request correlation ID or simply being on `localhost` is insufficient human authorization.

**Future separately reviewed item(s), NOT in this PR:** secure local request/review transport, authenticated and independently permissioned write service, operator identity/consent isolation, explicitly opted-in Write-capable plugin and create-exclusive strategy commit. The future write service must revalidate the source and new name at commit, atomically reconcile one-time approval consumption with file/DB persistence, protect against concurrent clients, and fail closed on partial/uncertain outcomes. Neither preflight nor approval reserves a name, creates a strategy or permits backtest/trading.

Do not run the CLI manually on the actual Strategy Studio DB for initial field acceptance. There is no plugin integration yet to originate a real request.

## Tests and release gates

Isolated tests use temporary files and pytest SQLite. They cover invalid candidates/IDs, pending caps, only-hash metadata, non-TTY/declined/expired/mismatched approval, exact approval, single-use consume, simultaneous consumption, CLI argument behavior and preservation of the **exact five** read-only MCP tools.

Focused Windows validation:

```powershell
python -m pytest backend/tests/test_studio_draft_approval.py backend/tests/test_studio_draft_create.py backend/tests/test_strategy_lifecycle.py backend/tests/test_chatgpt_mcp_readonly.py backend/tests/test_tradeagent_desktop_plugin_package.py -q
python scripts/validate.py
git diff --check origin/main...HEAD
git status -sb
```

Require exact branch/SHA, expected six-file scope, clean worktree, matching GitHub backend/frontend CI, operator Windows focused + full regression and frontend build. Add evidence to roadmap/work log on the same draft PR, repeat exact-final-status-head gates, then merge and clean-sync main. **No real strategy save, MCP write, broker connection, backtest or trade is authorized.**
