# TradeAgent private desktop research plugin — v0.1.1 released; v0.1.2 draft preflight awaiting host validation

**Scope:** This repository's existing TradeAgent, **not** the historical standalone SMC assistant. This item packages existing, independently field-tested local MCP research capabilities and one disciplined daily-research skill for a **private, local desktop** plugin. It does **not** implement or claim ChatGPT Work/cloud access, broker execution, a new MCP service, plugin publishing or an installed personal account plugin.

**Repository baseline and release state:** the original private research plugin (v0.1.0, three read-only tools) was verified through an actual native desktop invocation in PR #158, then final Windows tests and GitHub CI #674 passed; it was squash-merged to `main` `a4ba1486ba470bdf95174ea0bddeca5a8528749e` and the operator clean-synchronized local `main`. Post-merge CI #675 passed. The narrowly scoped v0.1.1 fourth-tool extension passed native desktop field tests, exact final-head Windows validation and CI #684; PR #159 merged as `69ed806e4338743a9de23bd7135560bd64926b62`, operator clean-synchronized main and post-merge CI #685 passed. PR #160 separately released a documentation-only contract audit for draft creation, merged as `8828496d7e3aa12a67fe16edd9f9d5f8cc9cc5f2` after final local/CI validation and clean Windows main sync. The new v0.1.2 proposal is non-mutating draft preflight only, not draft saving or execution.

## Released extension — source-inspection preview (PR #159)

The already verified private desktop plugin in PR #158 bundled exactly three
research tools at installed version `0.1.0`. The separately reviewed next
candidate increments the plugin manifest to `0.1.1` and registers **only one
additional read-only MCP tool**: `inspect_studio_saved_strategy(strategy)`.
It does not create, load into a runtime, save, backtest, optimize, or deploy a
strategy. This is a limited, reconstructed **Python AST preview**, not raw source:
the gateway strips comments and masks string/bytes literals and large numeric
literals, requires strict lower-case saved names, enforces byte/line/output
limits, and never executes source. Identifiers, code structure and smaller
numbers can still be sensitive, so preview access requires an operator request
and is not a guarantee of secret removal. The plugin remains localhost only,
with no execution or cloud Work access.

**Field check after exact-head Windows tests and GitHub CI, without any broker
connection or restart of the backend:** refresh/reinstall the **supported
desktop plugin cache** rather than editing cached files manually; verify the
installed `plugin.json` version is `0.1.1`, and use only the process-owner-
verified local gateway. A running gateway loaded before the new source code
still registers three tools; stopping/restarting it is a **separately controlled,
nontrading operator action** and must never kill a foreign PID. In a fresh
local repo-scoped desktop task, verify the four-tool list, then explicitly
request `inspect_studio_saved_strategy(strategy="us500_ny_orb_m15_v1")`.
Check that returned preview is bounded, source SHA and field labels are present,
no raw comments/literals are echoed, and no saved source, strategy lifecycle,
backtest or account state changes occur. If a broker-free backend cannot serve
that existing saved strategy, report blocked honestly.

**No source mutation, simulation or trading power was introduced by v0.1.1.**
PR #159 subsequently passed the native host, final Windows/CI, merge and clean-main gates.

### v0.1.1 source-preview native desktop acceptance (2026-11-22)

The operator ran the focused MCP/Studio/plugin tests and full backend regression
with **707-module frontend Vite/TypeScript build** on PR #159 implementation
head `381c5ed56923383549d7627c178bb88b3cf237c6`, with exact SHA,
expected eight-file scope, `git diff --check` and clean worktree. Matching
GitHub Actions **CI #682** (`44945941684`) passed both jobs.

After a supported desktop plugin reinstall, local PowerShell confirmed that
the installed v`0.1.1` `plugin.json` SHA256 matched the repository. The
operator verified an existing loopback-only TradeAgent backend on port 4000,
then used the process-owner-checked local MCP lifecycle script to stop owned
gateway PID **22856** and restart it on loopback `127.0.0.1:8765/mcp`
(PID **12360**). This was a gateway-only restart, not a broker startup.

A fresh local desktop task then used the **native attached TradeAgent plugin MCP
tool**, not a standalone SDK or direct HTTP request, and discovered exactly
four read tools: `get_ctrader_candles`, `get_studio_recorded_evidence`,
`inspect_studio_saved_strategy`, `list_studio_saved_strategies`. Native
`inspect_studio_saved_strategy(strategy="us500_ny_orb_m15_v1")` succeeded
(`isError: false`). Returned metadata identified **2,320 bytes / 63 lines**,
source hash `77f6b978e19754e92c0ece72576f04a0ff38e1b5a7db7c1b0dffcf5afb70b633`,
a detected top-level `signals` function and
`preview_mode=python_ast_without_comments_or_string_literals`; the sanitized
preview was reported as successfully returned, without sharing its body. No
source execution, mutation, backtest, broker connection or trade occurred.
**Native host tool attachment is verified; confidentiality is still bounded
by the documented limits and is not a general secret-scanning guarantee.**

**Pending final release:** this documentation acceptance commit must pass
fresh exact-HEAD Windows focused/full regression, clean scope/tree and matching
GitHub CI. PR #159 remains draft until those checks pass. Only after merge
may the operator clean-sync `main` and start planning separately permissioned
strategy draft creation and backtesting.


## v0.1.2 proposal — operator-requested non-mutating draft preflight (PR pending validation)

The fifth read-only MCP tool is `validate_studio_strategy_draft(strategy, source)`.
It takes an **explicitly operator-reviewed new lower-case strategy name** and
the proposed Python source, validates the syntax and existing Strategy Studio
policy **without executing the code**, checks both the complete available
saved-strategy file catalogue and the existing persisted lifecycle-name status
using only fixed loopback GET routes, and returns a fixed compact result:
`strategy`, `ready_for_operator_review`, `source_sha256` (over canonical
normalized code), canonical bytes/lines, `catalog_checked`,
`lifecycle_checked`, blocker codes and a warning. A successful result
**does not reserve the name or save anything**. Failed checks return only
bounded blocker codes, never raw code, traceback or sensitive backend text.
No source file, strategy lifecycle, backtest evidence or account state should
change. The existing plugin remains local/unauthenticated `Read`-only;
v0.1.2 deliberately does not introduce an agent write permission.

### Broker-free desktop field check after exact-head Windows/CI tests

1. Through the supported local Plugins Directory, refresh/reinstall the
   private plugin as **v0.1.2**, restart the local desktop client as necessary
   and verify the installed `plugin.json` SHA256 matches the repo. Do **not**
   edit the client cache manually or change MCP host binding.
2. With the existing research-only backend healthy on loopback port 4000,
   restart **only** the verified-owner MCP listener on 8765 using the existing
   `scripts/local-mcp-lifecycle.ps1` Stop/Start helper. A foreign owner or
   non-loopback listener is a hard stop. Do not run a normal broker-starting
   launcher, connect cTrader or change the trading engine.
3. Open a fresh local repo-scoped desktop task. Confirm exactly **five native
   plugin-attached MCP tools**, including `validate_studio_strategy_draft`,
   and invoke it **once** with a deliberately synthetic, non-sensitive name
   such as `mcp_preflight_us30_demo_20261122` and the following research-only
   minimal code (no external APIs, credentials or orders):

   ```python
   import pandas as pd

   def signals(df):
       return pd.Series(0, index=df.index)
   ```

4. Expect `ready_for_operator_review=True` **only if** the specified name is
   not already a saved strategy and has no persisted lifecycle. Check returned
   SHA256 is 64 lowercase hex, both catalogue and lifecycle checks succeeded,
   returned counts are sensible, source code is **not echoed**, and no file
   or lifecycle was created. If the candidate name is already registered,
   expect an explicit name-collision blocker rather than inventing success.
5. Use the **native attached MCP tool** in the desktop plugin, not a standalone
   Python SDK or direct HTTP fallback. Report any blocked response honestly.
   Do not ask the model to write, save, backtest, promote or execute the code.

This preflight receives user-supplied code. A cloud-backed model/chat may see
that code even though the MCP gateway itself is loopback-only; **never send
real secrets, broker account data or proprietary source snippets** for this
field test. The future create-exclusive, human-authorized save feature is
a **different item**, not part of the read-only plugin.

## Portable package and scope

- `plugins/tradeagent-research/plugin.json`: portable manifest, version 0.1.0, `Read` capability only, local executor required.
- `plugins/tradeagent-research/mcp.json`: **one** Streamable HTTP MCP configuration, `http://127.0.0.1:8765/mcp`, requiring the *already running* local research gateway; no OAuth, credentials, remote host, process lifecycle hooks or automatic broker connection.
- `plugins/tradeagent-research/skills/daily-desk/SKILL.md`: short repeatable US30-first, XAUUSD-second, optional US500 confirmation research workflow, source/freshness honesty and explicit **NO TRADE**.
- `.agents/plugins/marketplace.json`: local repo marketplace pointing **only** to this plugin.
- `backend/tests/test_tradeagent_desktop_plugin_package.py`: offline contract tests for package boundaries and exact existing three-tool SDK registration.

The gateway itself is unchanged: `get_ctrader_candles`, `list_studio_saved_strategies`, `get_studio_recorded_evidence`. The portable MCP manifest is a **local HTTP client configuration** for the desktop/plugin runtime, not an HTTP endpoint deployed to the internet and not a remote ChatGPT connector.

## Operator-safe local test — after PR branch sync

Start **only** the existing research backend in a separate PowerShell window, from the repo root, without broker or model startup:

```powershell
conda activate tradeagent-v2
$env:APP_START_CTRADER_ON_BOOT = "0"
$env:APP_WARM_OLLAMA_ON_BOOT = "0"
$env:APP_START_EVENT_INTELLIGENCE_ON_BOOT = "0"
$env:PYTHONPATH = (Get-Location).Path
python -m uvicorn backend.app:app --host 127.0.0.1 --port 4000
```

In a second shell, check `http://127.0.0.1:4000/api/health` and start only the process-owner-verified MCP gateway:

```powershell
conda activate tradeagent-v2
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\local-mcp-lifecycle.ps1 -Action Start
```

The API and MCP listener **must** be restricted to loopback. A failed listener/foreign owner is a **stop** condition; never force-kill another PID. Do not run the normal `start-local.cmd` for a broker-free acceptance test because its backend launcher explicitly opts in to broker startup.

### Discovery and private installation (after code/CI approval)

1. In a supported local ChatGPT Desktop/Codex plugin-capable session, open the **Plugins Directory**, select the repo's local marketplace `TradeAgent Local Research`, and look for **TradeAgent Research (Local)**. The repo marketplace is discoverable after a desktop restart on supported clients. If it is absent, report the app version, menu options and marketplace discovery status; **do not** replace the localhost URL with a public endpoint.
2. Install/enable the private repo-local plugin **only for this local environment**. Discover exactly the three tools already verified on TradeAgent; the host may require permission to use the local executor. The package does not install itself into ChatGPT just because files exist on GitHub.
3. With the broker disabled, run an **actual local** plugin/agent request: “Use TradeAgent Daily Research Desk to list my saved strategy names and inspect existing recorded evidence for `ema20_50_xauusd_m5_research`. No new backtests, account actions, or trades.” Expect only saved names and at most five stored evidence entries. Request a US30/XAUUSD daily brief afterward; without broker market data, it must honestly report **NO LIVE DATA / NO TRADE**, not fabricate market prices.
4. When finished, stop the **owned** MCP process:
   ```powershell
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\local-mcp-lifecycle.ps1 -Action Stop
   ```
   Stop the research API with Ctrl+C in its own shell. Do not alter cTrader selection, autotrading, the engine, kill switch, stored lifecycle state or source code.

**Alternative Codex CLI check (not required to install the plugin):** `codex mcp add tradeagent-local-readonly --url http://127.0.0.1:8765/mcp`, then `codex mcp list`. This manually adds a local MCP connection to Codex's own user configuration; do **not** run it alongside plugin auto-wiring without checking for duplicate registrations. It is a separate operator opt-in, not performed by this PR.

## Security and capability boundaries

- Private market/strategy data remains on the user's Windows machine **until a local agent chooses to send retrieved values into its conversation**. Neither a localhost endpoint nor a private plugin prevents an AI cloud model from receiving data deliberately used in a chat. Avoid account secrets, client tokens, trade history, personal identifiers or raw source.
- The local gateway has no authentication and binds to `127.0.0.1`; **loopback is not authentication**. Its client manifest omits auth metadata and credential headers. Other local processes may access it. This package is **not appropriate for sharing publicly** without replacing the authorization model.
- No ngrok, public reverse proxy, port forwarding, `0.0.0.0`, remote network bind, or cloud-hosted `ChatGPT Work` connections. Cloud Work cannot reach the user's Windows loopback through this manifest.
- A Secure MCP Tunnel or private authenticated HTTPS endpoint is a **separate** item requiring threat modeling, explicit user approval, authenticated least-privilege exposure and rate limits. This PR does not provision one.
- Read-only means no requested domain/strategy/trading mutation; backend API calls can still produce operational telemetry. Saved strategies and past backtest evidence are **not** evidence of live performance or authorization to trade.

## Acceptance and release

### Connection diagnosis on PR #158 (2026-11-21; acceptance still pending)

The task ran locally on Windows 11 build 26300 / PowerShell 7.6.9, with
Codex Desktop 26.1117.4957.0 and CLI 0.171.0-alpha.13. The restricted shell's
localhost requests timed out and CIM listener inspection was denied. An
approved host-side inspection found the existing loopback listeners on 4000
(PID 16532) and 8765 (PID 12308); neither process was restarted. The installed
MCP v2 SDK, outside the sandbox, discovered exactly the three approved tools
and returned 12 saved strategy names, `truncated=false`, without broker calls.

The plugin was enabled and its skill was visible, but no TradeAgent tools were
attached. The local host log (`codex_core_plugins::loader`) identifies the
repository packaging defect before any MCP handshake:

```text
failed to parse plugin MCP server ... server="tradeagent-local-readonly"
error="unknown field `extensions`, expected `url` or `headers`"
```

Remove only the server-level `extensions.com.openai.auth` metadata from
`mcp.json`; retain `type=streamable-http` and the exact loopback URL. Plugin-level
`extensions` in `plugin.json` remain required and unchanged. The package test
now guards the host-compatible server keys. No stdio bridge, new server,
public endpoint or user-wide MCP registration is justified by this failure.
Sandbox network restrictions explain the failed shell probes separately; they
do not establish that the desktop host is cloud-only or the gateway is down.

The installed cache and user settings were not edited. After review, reload or
reinstall through the supported local host UI and start a fresh local chat.
Confirm the installed `mcp.json` matches the repository, then invoke the plugin
to list tools and call `list_studio_saved_strategies`. Require exactly three
tools and an actual successful catalog response. SDK success alone does not
verify plugin attachment; keep roadmap §28 unverified until this host check
and the exact-head Windows/CI gates pass.

With the existing backend/gateway already running, this PowerShell check uses
only the MCP catalog (no service start, broker connection or strategy write):

```powershell
conda activate tradeagent-v2
Get-NetTCPConnection -State Listen |
    Where-Object LocalPort -in 4000,8765 |
    Select-Object LocalAddress,LocalPort,OwningProcess
@'
import asyncio, json, httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

async def main():
    async with httpx.AsyncClient(trust_env=False, timeout=15) as client:
        async with streamable_http_client(
            "http://127.0.0.1:8765/mcp", http_client=client
        ) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                names = [tool.name for tool in (await session.list_tools()).tools]
                assert set(names) == {
                    "get_ctrader_candles", "list_studio_saved_strategies",
                    "get_studio_recorded_evidence",
                } and len(names) == 3
                print("Tools:", names)
                result = await session.call_tool("list_studio_saved_strategies", {})
                assert not result.is_error
                payload = json.loads(result.content[0].text)
                print("Strategies:", payload["strategies"])
                print("Returned:", payload["returned_count"],
                      "Truncated:", payload["truncated"])

asyncio.run(main())
'@ | python -
if ($LASTEXITCODE -ne 0) { throw "Read-only MCP smoke failed" }
```

After the branch is pulled, run:

```powershell
conda activate tradeagent-v2
# Use a dedicated test shell; these variables affect only this shell/children.
$env:APP_START_CTRADER_ON_BOOT = "0"
$env:APP_WARM_OLLAMA_ON_BOOT = "0"
$env:APP_START_EVENT_INTELLIGENCE_ON_BOOT = "0"
python -m pytest backend/tests/test_tradeagent_desktop_plugin_package.py backend/tests/test_chatgpt_mcp_readonly.py -q
if ($LASTEXITCODE -ne 0) { throw "Focused tests failed" }
python scripts/validate.py
if ($LASTEXITCODE -ne 0) { throw "Full regression failed" }
git diff --check
git diff --check origin/main...HEAD
git status --porcelain
```

Run the full suite in the normal Windows operator context: a restricted agent
sandbox can fail collection with Credential Manager error 5 before tests run.
This is distinct from the MCP manifest parser failure. Do not print credentials
or change the credential store to work around it.

### Native desktop plugin connection — operator field acceptance (2026-11-21)

After Codex identified the exact host parse error `unknown field 'extensions', expected 'url' or 'headers'`, PR #158 dropped the unsupported **server-level** MCP `extensions` key without changing the Streamable HTTP transport, endpoint, plugin-level read-only permission, gateway implementation or security boundaries. Windows focused package/MCP tests, full backend validation and 707-module frontend production build passed on implementation commit `a50d6b4b8f495dee481c29e70ac1fc3008e8dfc7`. GitHub Actions CI **#672** (run `44934254438`) passed Python backend and Node frontend on that exact implementation head.

The operator **uninstalled and reinstalled** TradeAgent Research (Local) through the supported desktop plugin UI. Installed `0.1.0/mcp.json` then exactly matched the corrected source (SHA256 `36B79D1DF6D2CE7DD0494F74DE2B2FE1385E4BA9F746F1710541A064A6755E7E`), with **`type,url`** keys; the previous cached `extensions,type,url` mismatch is resolved.

Most importantly, the operator's new local desktop task reported **genuine direct attachment of the plugin's native MCP tool**: the three permitted read-only tools `get_ctrader_candles`, `get_studio_recorded_evidence` and `list_studio_saved_strategies` were exposed, and `list_studio_saved_strategies` returned **12** stored names, `truncated=false`, using the *attached plugin tool*, **not** a separate Python MCP SDK, PowerShell request or direct HTTP client. No broker connections, backtests, strategy edits or trades were initiated. This completes **§28 functional plugin-host acceptance**; it does not establish live market data, broker readiness, cloud ChatGPT Work connectivity or public plugin distribution.

**Remaining release requirements:** verify final documentation-status commit with exact-head Windows focused and full regression, clean worktree/diff and new exact-head GitHub CI, then merge PR #158 and fast-forward synchronize a clean local `main`. Do not merge before that final checkpoint or introduce a localhost tunnel.

Operator native desktop plugin-host field evidence is now recorded above; this is **separate** from SDK-only validation. Before merging, require exact local/remote documentation-status HEAD, expected file scope, clean diff/worktree, final Windows focused and full regression, and matching final-head GitHub CI frontend/backend. Then merge PR #158 and clean-sync local `main` before starting another roadmap item.

Authoritative interface docs: `docs/CHATGPT_READONLY_MCP.md` and `docs/TRADEAGENT_ROADMAP.md` §21 and §28.
