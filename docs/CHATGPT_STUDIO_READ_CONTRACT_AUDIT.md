# TradeAgent Strategy Studio read-contract audit for future local MCP tools

**Audit baseline:** `main` at `b23e3d723a1ca62b2c5af29f5b9e54face70cc15` (after merged launcher PR #154).
**Status:** documentation/contract review only. **No new MCP tool is registered by this item.** Existing gateway still exposes solely `get_ctrader_candles` at `127.0.0.1:8765/mcp`.
**Product boundary:** This is the existing **TradeAgent** Strategy Studio, not the historical standalone SMC assistant. No public tunnel, cloud plugin connection, trading execution, strategy save, backtest initiation or lifecycle promotion is authorized.

## Verified existing interfaces

The routes below are in `backend/api/router.py` and are mounted under `/api`. The implementation detail is important: **HTTP GET does not by itself mean a read-only research action.**

| Existing route | Implementation | Actual semantics | Initial MCP decision |
| --- | --- | --- | --- |
| `GET /api/studio/strategy-files` | `studio_backtests.list_saved_strategy_files()` | Lists `*.py` filenames under `backend/strategies_generated`; also returns the server **absolute `cwd`**. The filenames are not guaranteed sorted, bounded, validated by lifecycle or safe to publish verbatim to an untrusted client. | **Candidate** only after a fixed output schema strips `cwd`, caps results and validates names. |
| `GET /api/studio/strategy-file/{strategy}` | `studio_backtests.load_saved_strategy_source()` | Name normalized to lower case and constrained to `[a-z0-9_-]+`; returns strategy, filename and **full raw Python source**, with no current size bound or redaction. | **Defer raw source** until explicit operator opt-in, byte cap, secret review, exact name allowlist and provenance rules. Never execute the source from MCP. |
| `GET /api/studio/lifecycles` | `strategy_lifecycle.list_lifecycles()` | SQLite query returns **latest version for each strategy**, then hydrates full nested evidence/transitions; includes hypothesis, version hash, readiness blockers, detailed metrics/context, operator names and reasons. No response count/size cap in this route. | **Candidate** only through a metadata-only projection with result limits. Never return unfiltered nested payload. |
| `GET /api/studio/lifecycle/{strategy}` | `strategy_lifecycle.get_lifecycle()` | Retrieves latest version and stored lifecycle evidence. Internal service accepts an optional `version_hash`, but this API route does **not** expose version selection. | **Candidate** for a single-name, latest-version bounded summary. Never promise historical-version lookup via this route. |
| `GET /api/studio/backtest` | `studio_backtests.run_saved_strategy_backtest()` | Fetches market data, executes saved research source in sandbox and, on the normal non-walk-forward path, calls `record_backtest()` which registers lifecycle and inserts evidence into SQLite. | **Prohibited as an MCP read tool.** It computes a new test and can persist evidence despite using HTTP GET. |
| `GET /api/studio/runtime-strategy-audit` | `runtime_strategy_validation.run_runtime_strategy_audit()` | On-demand runtime strategy computation over fetched market observations, **not** retrieval of an already completed report. | **Excluded** from initial read-only MCP contract. |
| `POST /api/studio/tasks` | `studio_tasks.execute_studio_task()` | Can generate or save source, run backtests or optimization and create lifecycle evidence. | **Prohibited**. |
| `PUT /api/studio/lifecycle/{strategy}` and `POST /api/studio/lifecycle/{strategy}/{promote,paper-evidence,retire}` | `strategy_lifecycle` mutation functions | Modify hypothesis, stage, evidence or lifecycle. | **Prohibited**. |

## Completed research results: what is actually persisted

`backend/storage/db.py` defines `strategy_lifecycles`, `strategy_lifecycle_evidence` and `strategy_lifecycle_transitions`. Each recorded evidence row stores `evidence_type`, `passed`, `summary`, `metrics_json`, `context_json` and `created_at`. The lifecycle reader hydrates those rows; thus **previously recorded evidence/metrics** can be read without starting a new backtest.

This **does not establish** a general durable catalog of full completed backtest reports or their original time series/equity curves. A request for the most recent completed backtest should initially be answered, if supported, as **latest recorded lifecycle evidence for a particular saved strategy**, clearly labeled by evidence type and version; otherwise return unavailable. Results from draft tasks that were not recorded must not be invented or represented as persisted.

`backend/domain/models.py` defines `StrategyLifecycleEvidence` with open-ended `metrics` and `context` dictionaries. These can be large and may contain operator-supplied content. The API's raw response also includes transition `operator` names and reasons. **No direct raw pass-through to an AI client**: project explicitly allowed scalar fields, drop free-form context and operator data, enforce limits.

The FastAPI middleware in `backend/app.py` records API latency and unavailable-request telemetry for requests under `/api/`. Therefore even a domain-read GET may result in **operational telemetry writes**; here “read-only” must mean no requested trade, strategy or lifecycle mutations, not a guarantee of zero disk writes.

## Proposed first independently scoped MCP change (NOT implemented here)

**One new local-only tool**: `list_studio_saved_strategies` (illustrative contract name, to be finalized in its own PR). Its sole purpose is to list safe saved-strategy **names** from the existing `GET /api/studio/strategy-files` endpoint.

Proposed minimal contract:
- Connect only to hardcoded `http://127.0.0.1:4000/api/studio/strategy-files`; **GET only**, `trust_env=False`, redirect disabled and finite timeout. Never accept an arbitrary URL or filesystem path.
- Validate the upstream JSON envelope and `files` array. Permit only `[a-z0-9_-]+.py` with a bounded name length, normalize/sort/deduplicate, and limit the number of returned names. Reject malformed payloads and non-finite/invalid data rather than inventing saved strategies.
- Return a small structured `{strategies, observed_count, returned_count, truncated, source_route, warning}` envelope; **omit `cwd`, full paths, source text, secrets, database IDs, execution status, broker account info and arbitrary payload fields**. Provide an honest count/truncation statement.
- No strategy code execution, HTTP backtest GET, Studio task POST, lifecycle evidence insertion, trade entry/exit or account switch.
- Keep current one-tool candle gateway behavior intact and test the new registration count explicitly. Use mocked HTTP and official MCP SDK registration tests, plus local client smoke and focused/full CI gates.

**Later and separately scoped**: bounded `get_studio_strategy_lifecycle_summary` and/or `get_studio_completed_evidence` over **existing** lifecycle GET reads. Select exact safe fields and allowlisted scalar metrics after checking representative persisted records, cap nested evidence (e.g., last 10 records), validate the exact saved strategy name, state source hash/version and timestamp, redact arbitrary `metrics/context` and transition details, and clearly flag that evidence is not performance proof or live approval. **Raw saved-source access** requires an independent explicit review and operator approval; do not silently combine it with catalog/evidence tools.

## Acceptance and security gates for next implementation

1. Reconfirm the repository main/PR/CI state and this audit before implementing the next item. Work **one read-only MCP tool at a time**.
2. Unit-test bounded outputs, malformed upstream errors, traversal/invalid strategy names, no arbitrary URL, no proxy/redirect, no state-changing calls, and exactly the expected MCP tool allowlist.
3. Run Windows local SDK-client smoke only on the existing loopback service and on already authorized research data; verify gateway/API ports remain on loopback. No broker account or runtime control changes.
4. Require exact-head Windows focused tests and `python scripts/validate.py`, matching GitHub CI, roadmap/work-log evidence, merge and a clean local-main sync before starting another capability.
5. Until a **separately reviewed authenticated HTTPS** design exists, the MCP server remains unauthenticated **localhost-only**, never exposed by tunnel, port forwarding or ChatGPT Work/cloud. All retrieved source, strategy names and notes are untrusted AI input, not instructions to execute trades.

## Source locations inspected

- `backend/api/router.py`: Studio GET/POST/PUT handlers.
- `backend/services/studio_backtests.py`: saved-file list/source and `run_saved_strategy_backtest` persistence path.
- `backend/services/strategy_lifecycle.py`: latest-version query, evidence hydration, `record_backtest` / `record_evidence` writes.
- `backend/domain/models.py`: nested lifecycle, evidence and transition response models.
- `backend/storage/db.py`: durable lifecycle/evidence schema.
- `backend/services/studio_tasks.py`: source-writing/backtest/optimization task behaviors.
- `backend/app.py`: telemetry middleware and API routing.
- `frontend/src/services/api.ts`: Strategy Studio currently uses file/catalog/source/backtest/lifecycle endpoints.
- `docs/STRATEGY_LIFECYCLE.md`, `docs/TRADEAGENT_ROADMAP.md` §§21–24, `docs/CHATGPT_READONLY_MCP.md`.
