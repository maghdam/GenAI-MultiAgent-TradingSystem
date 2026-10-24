# Alpaca integration — fixed-scope completion review (2026-10-24)

**Repository baseline:** `main` at `c41109b3b0c90827c84966e45497907bea876f79` (merged Phase 10.22 / PR #129). This is a source-and-roadmap **review**, not new broker functionality, a Paper field test, or approval to trade.

## Decision: finish the intended product, not an endless sequence of safety phases

The word **complete** has three distinct meanings:

| Exit target | Current evidence | Honest closeout |
| --- | --- | --- |
| **A. Daily research and explicitly supervised Paper experiments** (immediate product goal) | Phase 10.10 venue selection, 10.11 market-data shadow (read-only real SPY/BTC bars), Phase 10.9/10.12 simple Paper broker order/reconciliation adapter (2026-10-18 isolated SOL/USD Paper entry/close field cycle). TradeAgent can run trusted-strategy research without enabling auto entries. | **Core integration delivered for research**; supervision, exposure checks, instrument constraints and operator discretion still apply. The isolated historical Paper cycle does not prove native equity brackets safe. |
| **B. Autonomous, broker-protected US-equity Alpaca Paper execution** | Phase 10.14 hard-blocks automatic entry. Phases 10.15–10.22 added private one-shot equity bracket services, reconciliation, partial-fill accounting, local incident monitor and independent **manually started** SQLite watchdog. Latest lifecycle proof is mock/CI only; no native equity bracket broker field validation. | **Blocked.** Finite mandatory gates below must close, followed by explicit operator approval and a distinct activation PR. |
| **C. Alpaca crypto unattended entries or Alpaca Live** | Crypto only has simple order class under the provider decisions recorded in Phase 10.13; equity-bracket safeguards do not transfer. Live routing is prohibited. | **Out of scope for the US-equity Paper closeout.** Require separate design, evidence and explicit permission. |

**Product boundary:** cTrader remains independently governed for forex/gold/indices. The unrelated open ChatGPT MCP-gateway PR is not an Alpaca activation dependency. Generated Strategy Studio runtime (roadmap Phase 4.9/4.10) is separately governed and not required to close this **trusted-strategy** Alpaca work.

## Evidence-based inventory (do not reopen validated work)

| Capability | Authoritative evidence | Remaining caveat |
| --- | --- | --- |
| Venue/configuration and symbol routing | Roadmap 10.10, `backend/services/alpaca_runtime.py` | Explicit Alpaca rows do not inherit cTrader permissions. |
| Data/research | Roadmap 10.11 read-only SPY and BTC/USD field data | Price availability does not certify strategy profitability. |
| Simple supervised Paper execution | Roadmap 10.9/10.12: real isolated SOL/USD Paper entry, fill/flatten, and cloned-DB restart replay | Was a **simple order**, not protected native US-equity bracket execution. The original field experiment does not authorize new orders. |
| Risk and activation gate | `backend/services/alpaca_bracket_readiness.py`; `backend/services/alpaca_runtime.py` rejects autonomous new entries | Existing supervised close path is distinct; past snapshots may not bypass gate. |
| Native US-equity bracket foundations | Roadmap 10.14–10.20, `backend/services/alpaca_bracket_{supervised,exit,lifecycle,reconcile}.py` | Mock-tested internal Python services, not an automatically wired UI/API/engine workflow or Paper field-verified lifecycle. |
| Operator-started snapshot monitor | Roadmap 10.21, `backend/services/alpaca_bracket_monitor.py` | No account-wide orphan scan; local incidents alone do not deliver notifications. |
| Separate offline watchdog | Roadmap 10.22, `backend/services/alpaca_bracket_watchdog.py` | Only checks last **completed** local heartbeat when started; neither process is OS-supervised. |

## Finite blocker register — **only** for target B (autonomous protected US-equity Paper)

The following **five acceptance packages** replace indefinitely adding monitoring layers. Keep one PR and one verified result at a time, prioritizing the highest remaining failure risk. Work can be combined where a single test/demo genuinely proves multiple criteria; no package is pre-approved for real orders.

**B1 — Account-wide truth, identity and bounded observations (P0).** Verify full Paper position/order inventory, including **orphan/external** positions, alternate accounts, pagination/incomplete responses and symbol collisions. Missing/incomplete truth must block automatic action and raise operator incidents. Define read timeouts and documented fallbacks. Unit/restart tests first; read-only broker observations only with separate authorization.

**B2 — Protected exit and failure recovery policy (P0).** Resolve how to handle disappearing/replaced protective legs, consumed/unknown claim outcomes, rejected or partially filled closing orders, OCO races and cancellations. Explicitly address the **unprotected cancellation-to-close interval** and lack of broker-enforced reduce-only market close; require an operator-approved fail-closed response that never guesses quantities or blindly retries. Mock race/crash cases and safety review are mandatory. Do not promise atomic flattening where the broker offers none.

**B3 — Broker-native equity bracket field evidence (P0; explicit approval required).** In a controlled **Paper US-equity only** test, check parent submission acknowledgement, exact IDs, full/partial fills, actual stop/target leg quantities/resizing, OCO effects, cancellation response versus terminal truth, broker position truth and restart reconciliation. Document outcomes and discrepancies. Never transmit orders merely because this document exists; confirm authorization, a small test quantity and a no-exposure baseline beforehand.

**B4 — Durable lifecycle accounting and risk release (P0).** Reconcile parent/stop/target/supervised-close fills with broker positions and realized P/L; handle fees, partial fills, ledger idempotence, terminal versus ambiguous close, and reservation release. Prove journal totals and daily-loss accounting with realistic replay/broker evidence, and prevent double-counting or premature new entries.

**B5 — Independently supervised liveness, notifications and final Paper acceptance (P0).** OS-supervised service/watchdog with bounded failure detection **and operator-delivered, acknowledged escalation**; verify outages/restarts and that safety action never depends on a stale healthy heartbeat. Exercise an isolated end-to-end Paper US-equity lifecycle through normal permitted interfaces with clear rollback/stop conditions. Then review all blockers and only **in a separate explicitly authorized activation PR** consider Paper automatic entries. Paper crypto and Live stay disabled.

**Dependencies:** B1 and B2 provide a safe evidence basis for B3; B3 can expose new B2 findings. B4 must be validated before risk capacity is released or automatic entries enabled. B5 must not infer readiness from mocked tests or successful snapshots. Never label any package complete without its operator-local, CI and required provider evidence.

## Practical stopping rule

- **If the immediate goal is disciplined daily market research:** stop Alpaca infrastructure expansion after this review. Continue analysis, backtesting and strategy evaluation on already-available Alpaca market data; any supervised Paper order needs explicit, separately considered authorization. No watchdog service, bracket field orders, auto entry or Live activation is implied.
- **If the goal is unattended US-equity Paper trading:** use **B1 → B2 → B3 → B4 → B5** as a finite, reviewed acceptance list. New sub-issues must reference a failing gate instead of becoming unrelated phases.
- **If the goal is Live or unattended crypto:** open separate workstreams **only after** Paper safety is justified; do not silently expand B1–B5.

## Stop conditions and review status

At this review's creation, **B1–B5 are open**, Phase 10.20–10.22 internal code is mock-verified within its scope, and automatic Alpaca Paper/crypto entries and Alpaca Live are **disabled**. This review introduces **no Python execution change, endpoint, credentials, scheduler or external action**. Review acceptance needs a clean local docs diff, full regression and GitHub CI on the exact PR head; it does not require new Alpaca API field queries.

Sources: `docs/TRADEAGENT_ROADMAP.md` (Phase 10.9–10.22); `backend/services/alpaca_runtime.py`, `alpaca_bracket_readiness.py`, `alpaca_bracket_supervised.py`, `alpaca_bracket_exit.py`, `alpaca_bracket_reconcile.py`, `alpaca_bracket_monitor.py`, `alpaca_bracket_watchdog.py`; `docs/operations/alpaca-bracket-lifecycle.md`. This fixed scope is an **internal plan**, not a claim that broker-provider behavior has been freshly verified.
