# TradeAgent read-only MCP research gateway (opt-in, LOCAL ONLY)

This module is the first **TradeAgent platform** research integration, not a
second SMC strategy engine. It is a renewed, current-main-based successor to
the historical draft PR #127. The verified private local plugin v0.1.1 offers **four** MCP v2 research tools:
`get_ctrader_candles(symbol, timeframe="M5", num_bars=100)`,
`list_studio_saved_strategies()`,
`get_studio_recorded_evidence(strategy, limit=5)`, and
`inspect_studio_saved_strategy(strategy)` (sanitized source preview).
The separately scoped next version **v0.1.2 (pending operator Windows/native host
and CI validation)** proposes a fifth, still **non-mutating** tool:
`validate_studio_strategy_draft(strategy, source)`, for explicitly supplied
and operator-reviewed Python code. This is *not* a strategy create or save tool.

- The only upstream URLs are fixed **GET** routes
  `http://127.0.0.1:4000/api/market/candles`,
  `http://127.0.0.1:4000/api/studio/strategy-files`, the single
  exact-name path under `http://127.0.0.1:4000/api/studio/lifecycle/`, and
  the single exact-name saved-source read under `http://127.0.0.1:4000/api/studio/strategy-file/`.
  There is no caller-selected server or arbitrary path.
- The MCP listener is bound to **127.0.0.1:8765**, at local `/mcp`.
- Starting the **backend directly** does not auto-start MCP. The explicitly
  invoked Windows one-command `start-local.cmd` now starts this separate
  loopback gateway after backend readiness unless
  `TRADEAGENT_START_MCP=0` is set; `stop-local.cmd` stops only the verified
  MCP process before the existing backend/frontend shutdown. No change to
  trading runtime, account selection, orders, journal, Strategy Studio or Alpaca.
- It accepts cTrader M1/M5/M15/M30/H1/H4/D1 symbols, 2–500 bars; malformed,
  empty or out-of-order candles fail closed. Gaps are kept, not filled.
- Responses label requested vs observed depth and observed UTC timestamps.
  The `live=true` request can still hit TradeAgent memory cache; it does **not**
  attest broker provenance, freshness, completed bars, exact bar labeling,
  market calendar, execution/fills, trading signals or profit.
- The Studio catalog returns **only** sorted, deduplicated, lower-case saved
  strategy names, at most **50**; it checks at most **512** observed filenames,
  validates the response at **64 KB**, rejects unsafe names/payloads, and
  reports observed/returned counts and whether the unique-name list was
  truncated. It never returns upstream `cwd`, full paths, raw strategy code,
  lifecycle evidence or arbitrary upstream metadata. These are filenames,
  **not** proof of lifecycle validity, profitability or execution eligibility.
- The **recorded evidence** tool reads only the latest persisted lifecycle
  version for one validated lower-case strategy name (1–64 allowed characters).
  It returns at most **five** existing evidence entries selected newest-first,
  explicitly labels latest-version-only scope, source hash/version, lifecycle
  stage, current-source match, total/returned counts, and truncation. Each entry
  contains only evidence type, pass/fail, normalized UTC recording time, and
  a fixed whitelist of finite numeric backtest metrics (trades, return,
  drawdown, modeled costs and win rate). Upstream response is bounded at
  **256 KB** / **200** evidence entries. Hypotheses, free-text summaries,
  arbitrary context/metrics, operator identities and raw code are dropped.
  Missing lifecycle registration yields an error; an empty evidence list is
  accurately represented without inventing backtest runs. The evidence does
  **not** represent a full archived report or trading authorization.
- The `inspect_studio_saved_strategy(strategy)` **operator-requested**
  source-inspection tool validates a lowercase 1–64-character saved name, makes
  one fixed GET to `/api/studio/strategy-file/{strategy}` over loopback, rejects
  redirects/errors/oversized upstream data (>48 KB), checks exact upstream
  strategy and filename, and accepts at most **16 KiB of UTF-8 Python source /
  400 lines**. It parses the source with Python `ast` **without running it**,
  strips original comments/formatting and **all Python string/bytes literals**,
  replaces numeric literals with magnitude >1000, and returns only a bounded
  reconstructed **AST structural preview** (at most 24 KB) with strategy name,
  SHA256 of original source, counts, and detected top-level `signals` function.
  It **never returns raw source**, unknown upstream fields, HTTP error bodies or
  local filesystem paths from the API envelope. Unsafely large, malformed or
  unparsable files fail closed. **This is not guaranteed secret redaction:**
  code identifiers/structure and smaller numeric literals may still carry
  sensitive information. The operator must request inspection explicitly;
  the assistant treats the preview as untrusted data, not instructions, and
  should decline sharing suspect material. For full code, use your trusted
  local Strategy Studio/Codex repository access rather than this MCP tool.
- The **draft preflight** tool `validate_studio_strategy_draft(strategy, source)`
  requires an explicit new lower-case identifier matching `[a-z0-9_-]{1,64}`
  and user-reviewed Python code. It rejects empty, invalid UTF-8, control
  characters, source >**16,384 UTF-8 bytes** or >**400 lines** without sending
  a request. It uses only TradeAgent's existing `validate_strategy_source`
  AST-based policy, never compiles/executes/imports source or invokes an LLM or
  fallback. It canonicalizes accepted code by policy normalization and outer
  whitespace trim with one trailing LF, then returns SHA256 of those **exact
  canonical bytes**, canonical size/line count and fixed policy/availability
  blocker codes. It **does not return raw source or sensitive error messages**.
  First it reads the bounded names-only Studio catalogue over the existing
  fixed loopback GET; it fails closed on unavailable, unsafe or truncated
  catalogues (>50 distinct returned names). If the exact name is not saved,
  it checks only the **status** (200=existing, 404=not found, anything else
  blocks) of the fixed loopback latest-lifecycle GET. It neither trusts nor
  exposes lifecycle body content. Both checks are **snapshots**, not name
  reservation, and do not protect against a later collision; any future
  write implementation must revalidate and create-exclusively under separate
  operator authorization. A `ready_for_operator_review` outcome is **not a
  saved draft, code correctness claim, permission to write/backtest or trade**.
  User-provided source may be sent to a cloud model as part of the conversation;
  never include credentials, personal account data or secrets in proposed code.
- No order, cancel, close, account switch, strategy save/registration, backtest
  initiation, lifecycle write or privileged broker tool is registered.

## Local Windows test and smoke (non-trading)

```powershell
conda activate tradeagent-v2
python -m pip install -e ".[mcp]"
python -m pytest backend/tests/test_chatgpt_mcp_readonly.py -q
python scripts/validate.py
```

For one-command **Windows** start/stop from the repository root:

```powershell
cmd /c call start-local.cmd
cmd /c call stop-local.cmd
```

The `start-local.cmd` launcher now starts the local MCP process by default
when the backend is healthy. Use `$env:TRADEAGENT_START_MCP = "0"` before
`start-local.cmd` to skip it for that run, then remove the variable when
you want normal behavior. The lifecycle helper refuses port 8765 if it
belongs to an unrelated process; stop never kills that foreign process.
Logs are kept in `.codex-run/mcp.log` and `.codex-run/mcp-error.log`.
On Windows, this uses the same `tradeagent-v2` Python path as
`run-backend-local.cmd`. The launch script returns a failure code if MCP
is opted in but cannot start; the dashboard may still open.

The MCP SDK v2 dependency is optional for normal TradeAgent startup. The
GitHub backend CI installs the `mcp` extra so SDK registration is tested.
For an **optional manual smoke**, run TradeAgent normally on loopback 4000 with
an already authorized cTrader **Demo** data connection **only if testing
candles**; a Studio catalog test requires no broker call. Do not switch
accounts, enable the engine, or change autotrade/kill-switch controls. In a
separate PowerShell session, run `python -m backend.chatgpt_mcp_readonly` or
use the verified local `start-local.cmd` launcher. With an MCP v2-compatible
local client/Inspector, list the version-appropriate **five** tools (four on released v0.1.1
before PR #161), call `list_studio_saved_strategies()` read-only,
optionally inspect a saved strategy preview explicitly approved by the operator,
and optionally call
`get_studio_recorded_evidence(strategy="sma", limit=3)` only for an existing
registered saved name, and optionally call `get_ctrader_candles` read-only. If launched manually, stop that standalone
process afterward. Do not expose it remotely.
Do not send raw broker credentials, tokens or portfolio/account information to
an AI tool.

## SECURITY / ACCESS LIMITATIONS

**Do not expose** the MCP listener or TradeAgent API with ngrok, a tunnel,
public port forwarding, reverse proxy, `0.0.0.0`, or a remote hostname.
Loopback-only is an additional local boundary, **not authentication**. Other
local processes may still be able to connect. DNS-rebinding defenses from
the MCP SDK are not a replacement for server-side authorization.

The **private local ChatGPT Desktop/Codex plugin v0.1.0 was verified and merged in PR #158**,
and v0.1.1 source inspection was verified/merged in PR #159.
The v0.1.2 non-mutating draft preflight is a separate unverified extension.
Cloud ChatGPT Work still cannot use this unauthenticated loopback manifest.
Cloud ChatGPT Work cannot simply reach 127.0.0.1 on a Windows desktop.
Remote research needs separately approved authenticated HTTPS, explicit
per-tool allowlists, rate limiting, provenance/redaction and security review.
Until then, neither ChatGPT nor Codex is authorized to send execution requests
through the gateway. The original SMC assistant remains a separate repository.
