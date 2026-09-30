# TradeAgent — Local Validation & Improvement Roadmap

> Canonical project roadmap. Keep this file in Git and update it with verified implementation/test evidence as work progresses.
>
> **Status legend:** ⬜ Not started · 🟨 In progress · 🧪 Ready to test · ✅ Verified · ⛔ Blocked · ↩ Revisit later

## 0. Operating rule

Work **one item at a time**:

1. Reproduce / establish baseline.
2. Make the smallest coherent change.
3. Run focused tests.
4. Run full regression tests where appropriate.
5. Validate against the real cTrader **demo** account or UI when relevant.
6. Record evidence below.
7. Mark the item ✅ only after verification.
8. Move to the next item.

Live-account routing remains intentionally blocked.

---

## 1. Current verified baseline

| Area | Status | Evidence / note |
|---|---|---|
| Engine background loop | ✅ | `running=True`, `loop_active=True`, 3 watched, no runtime error after restart |
| Demo order routing | ✅ | XAUUSD, US30, NAS100 demo orders opened in cTrader |
| Lot-size conversion | ✅ | Fresh demo trades use 0.10 lot as configured |
| Broker SL/TP synchronization | ✅ | Broker target precision/repair and unprotected-position fail-safe are implemented and regression-tested; live routing remains blocked |
| Broker close detection | ✅ | Local positions close when broker position disappears |
| Broker realized P&L | ✅ | Journal reconciled to cTrader closing deals; NAS100 -2.71, US30 -0.73, XAUUSD -9.24 example matched cTrader total -12.68 CHF |
| Broker close price/time | ✅ | Closing deal execution price/time persisted from cTrader |
| Dashboard local time | ✅ | UTC backend timestamps now display in local browser time |
| Cooldown / confidence gating | ✅ | 75%/71% XAUUSD signals correctly rejected during 30-minute cooldown; later 67% signal executed after cooldown expired |
| SQLite persistence | ✅ | Runtime DB moved outside synchronized Google Drive worktree |
| Git / source of truth | ✅ | Canonical repository is on local C: storage and synchronized with GitHub `main`; roadmap is now tracked in `docs/TRADEAGENT_ROADMAP.md` |
| Full backend test suite | ✅ | Full backend suite passes on the canonical repository after readiness/test-environment fixes |
| Frontend production build | ✅ | Vite production build passes (699 modules in the latest verified runs) |

---

## 2. Phase 0 — Restore a clean test baseline

### 0.1 Investigate Python / pytest path contamination
**Status:** ✅

Observed full-suite failure paths reference:

`G:\My Drive\Software\GenAI-MultiAgent-TradingSystem - V2\backend\tests\...`

while the active repository is:

`G:\My Drive\Software\GenAI-MultiAgent-TradingSystem`

This may indicate an old editable install, `PYTHONPATH`, package import, or environment path still points at the previous `- V2` copy.

**Checks**
- [x] Migrate the canonical repository out of the old Google Drive worktree.
- [x] Reinstall the editable backend package from the canonical C: repository.
- [x] Confirm full-suite execution resolves against the canonical repository.
- [x] Confirm the full backend suite passes.

**Done when**
- Full-suite traceback paths point only to the canonical repository.
- No imports resolve from the old `- V2` folder.

### 0.2 Update demo workflow tests for strict broker readiness
**Status:** ✅

Current production execution intentionally defers demo trading unless the cTrader account/symbol is confirmed ready. The failing workflow tests do not currently mock this new readiness gate.

**Tasks**
- [x] Mock `get_demo_symbol_execution_readiness()` as ready in the happy-path/failure-routing tests.
- [x] Ensure the "order failure" test reaches `place_demo_market_order()` rather than stopping at readiness.
- [x] Ensure the engine test uses a fresh/current bar or mocks readiness as needed.
- [x] Add regression coverage for not-ready demo execution → deferred/retryable behavior.

**Done when**
- `python -m pytest backend/tests -q` passes completely.
- Focused demo workflow tests explicitly cover both ready and not-ready states.

### 0.3 Confirm frontend production build
**Status:** ✅

- [x] `npm --prefix frontend run build`
- [x] Production build completed successfully in repeated local and CI runs.

---

## 3. Phase 1 — Execution safety & broker truth

### 1.1 Hard failsafe for unprotected demo positions
**Priority:** P0  
**Status:** ✅

Observed event:

`ctrader_demo_order_unprotected`

Current behavior retries broker protection. Add a bounded retry/failsafe policy so a demo position cannot remain indefinitely without verified SL/TP.

**Target behavior**
1. Order opens.
2. Attempt SL/TP attach/verification.
3. Retry for a short bounded interval.
4. If still unprotected, close the **demo** broker position and verify closure.
5. Persist a high-severity incident/audit trail.

**Tests**
- [x] Immediate protection success.
- [x] Delayed/repair path covered.
- [x] Repeated protection failure → broker fail-safe close.
- [x] Broker close failure → critical/audited state with local tracking retained; no false local closure.

### 1.2 Broker position ID as primary reconciliation identity
**Priority:** P0/P1  
**Status:** ✅

Broker position IDs are now persisted. Replace remaining symbol+direction matching with broker-position-ID matching wherever possible.

**Done when**
- Exact `broker_position_id` is primary.
- Symbol+direction is only a recovery fallback.
- Multiple same-direction positions cannot be confused.

### 1.3 Broker account currency / balance / equity synchronization
**Priority:** P0  
**Status:** ✅

Recent execution intent details showed `account_currency=USD` and `starting_equity_amount=100000`, while the cTrader demo statement is CHF-denominated.

**Target separation**
- **cTrader demo mode:** broker-reported account currency and balance/equity drive monetary risk calculations.
- **Pure paper mode:** configurable virtual starting equity remains available.

**Validate**
- [x] Deposit/account currency (real demo account verified as CHF).
- [x] Balance/equity (real broker snapshot verified).
- [x] Daily loss budget derived from broker equity.
- [x] Risk-per-trade amount uses the broker monetary basis.
- [x] Auto sizing uses the broker monetary basis.
- [x] Cross-currency valuation covered by automated tests.

### 1.4 Broker ledger completeness
**Priority:** P1  
**Status:** 🟨

- [x] Legacy missing-deal case is fail-safe: never fabricate broker P&L.
- [x] Partial-close ledger/reconciliation support implemented and regression-tested.
- [x] Multiple closing deals are weighted/summed correctly in automated tests.
- [x] Commission/swap/conversion fees are reflected in broker net P&L.
- [x] Broker deal IDs are ingested idempotently so a deal cannot be counted twice.
- [ ] Real cTrader demo partial-close field verification (waiting for the next normal TradeAgent-managed open position).

---

## 4. Phase 2 — Trade workspace & operator UX

### 2.1 Show actual rejection reason in Trade Journal
**Priority:** P1  
**Status:** ✅

Replace generic:

`Signal rejected by V2 risk engine.`

with useful summaries such as:
- `Rejected - XAUUSD in 30-minute cooldown.`
- `Rejected - confidence 54% < minimum 60%.`
- `Rejected - max daily trade count reached.`
- `Rejected - daily loss cap reached.`
- `Rejected - stale M5 market bar.`

Keep full structured details available in Intents/Incidents.

**Verification evidence (2026-09-27)**
- Focused `backend/tests/test_engine_intents.py -q`: 23 passed.
- Full `backend/tests -q`: passed.
- GitHub CI: passed.
- Runtime manual rejection: XAUUSD 0.001 lot was rejected below broker minimum 0.01 lot with no position opened.
- Trade Journal persisted: `Rejected - Requested quantity is below the symbol minimum of 0.01 lots.`

### 2.2 Clarify signal confidence semantics
**Priority:** P1  
**Status:** ✅

Internal `confidence` values remain deterministic strategy-strength heuristics, not proven win probabilities.

**Implemented**
- [x] Operator-facing analysis, signal, and intent labels use `Signal strength`.
- [x] Tooltips explain that the score is not a calibrated win probability and show the current execution threshold where available.
- [x] System safety configuration names and explains the minimum signal-strength gate.
- [x] Trade Journal threshold rejections use `signal strength` wording while API/database field names remain backward compatible.
- [ ] Display calibrated probability separately only after enough out-of-sample evidence exists.

### 2.3 Position panel broker truth
**Priority:** P1  
**Status:** ✅

Show:
- [x] broker position ID,
- [x] broker entry from a read-only matched cTrader snapshot,
- [x] current broker SL/TP,
- [x] quantity,
- [x] protection status,
- [x] account-currency unrealized P&L from the tracked position,
- [x] broker-sourced realized P&L for partial closes when available,
- [x] last successful broker snapshot timestamp,
- [x] explicit broker identity/sync status without falling back to local values when a persisted cTrader position ID is not found.

**Verification**
- [x] Focused broker-truth/API tests, full backend suite, frontend production build, and CI.
- [ ] Real cTrader field observation on the next normal TradeAgent-managed open demo position.

### 2.4 Journal filtering / drill-down
**Priority:** P2  
**Status:** ✅

Filters:
- [x] execution only,
- [x] rejected signals,
- [x] protection incidents,
- [x] symbol,
- [x] strategy,
- [x] local date,
- [x] broker vs paper.

Drill-down:
- [x] linked intent type/status,
- [x] intent/risk/quantity/sizing reasons,
- [x] broker execution/protection/close details when available,
- [x] raw audit details,
- [x] demo-backed `paper_position_*` rows remain classified as broker-backed when broker metadata is present.

**Verification**
- [x] Local frontend production build.
- [x] Local full backend regression suite.
- [x] GitHub CI frontend production build + full backend suite.

---

## 5. Phase 3 — System page acceptance audit

Test every operator control against backend behavior.

### 3.1 Engine lifecycle
- [x] Start.
- [x] Stop.
- [x] Restart.
- [x] One-shot scan.
- [x] Recover.
- [x] Reconcile.
- [ ] Restart backend while broker position is open.

**Verification**
- [x] Focused lifecycle/API tests.
- [x] Local full backend regression suite.
- [x] Local frontend production build.
- [x] GitHub CI.
- [ ] Real cTrader demo field check: restart the backend while a normal TradeAgent-managed broker position is open and confirm recovery/reconciliation preserves broker identity and protection truth.

### 3.2 Safety controls
- [x] Kill switch.
- [x] Minimum confidence.
- [x] Cooldown.
- [x] Maximum daily trades.
- [x] Maximum open positions.
- [x] Maximum positions per symbol.
- [x] Daily loss limit.
- [x] Require protective stops.
- [x] Session filter.
- [x] Trading-enabled per watchlist item.

**Verification**
- [x] Focused safety-control acceptance tests.
- [x] Local full backend regression suite.
- [x] Local frontend production build.
- [x] GitHub CI.
- [x] System page exposes all listed global safety controls, including maximum positions per symbol.
- [x] Per-symbol automatic execution remains gated by watchlist `trading_enabled`.

### 3.3 Persistence
- [x] Settings survive backend restart.
- [x] Watchlist survives restart.
- [x] Runtime DB remains in nonsynchronized local path.
- [x] No `.db`, `-wal`, or `-shm` state appears in Git.

**Verification**
- [x] Focused persistence acceptance tests, including close/reopen storage restart.
- [x] Settings and complete watchlist reload unchanged after storage restart.
- [x] Windows default resolves to `%LOCALAPPDATA%\TradeAgent\data\tradeagent.db`.
- [x] Legacy `backend/data/tradeagent.db` has a one-time backup migration path when the new local DB does not yet exist.
- [x] `.gitignore` covers `.db`, `-wal`, `-shm`, and journal state.
- [x] `git ls-files` confirms no SQLite runtime state is tracked.
- [x] Local full backend regression suite.
- [x] GitHub CI.

### 3.4 Status truthfulness
- [x] Connected.
- [x] Demo confirmed.
- [x] Execution ready.
- [x] Symbol metadata ready.
- [x] Engine scanning.
- [x] Model ready.
- [x] Incidents reflect actual current faults, not stale state.

**Verification**
- [x] Focused status-truthfulness acceptance tests.
- [x] Connected, demo-confirmed, execution-ready, symbol-metadata, engine-scanning, and model-ready flags are derived from current broker/runtime/model state.
- [x] Engine scanning is sourced from `runtime.loop_active`, not merely `config.enabled`.
- [x] Current incidents are derived from present faults; historical incidents remain separate audit history.
- [x] Runtime recovery clears stale persisted `last_error` state.
- [x] Local full backend regression suite.
- [x] Local frontend production build.
- [x] GitHub CI.

---

## 6. Phase 4 — Build & Test / Strategy Studio acceptance audit

### 4.1 LLM research workflow
- [x] Chat request.
- [x] Strategy drafting.
- [x] Provider/model selection.
- [x] Failure/fallback path.
- [x] Save strategy.
- [x] Reload saved strategy.

**Verification**
- [x] Focused Strategy Studio LLM workflow acceptance tests.
- [x] Chat requests return research guidance without forcing code generation.
- [x] Strategy drafting returns validated editable source and records the provider/model used.
- [x] Selected provider/model are forwarded into generation.
- [x] LLM generation failure falls back to the built-in safe strategy template path.
- [x] Saved strategies are validated, persisted, and registered as lifecycle `draft`.
- [x] Saved strategy source can be reloaded read-only into the editable draft without importing it into the trusted runtime.
- [x] Local full backend regression suite.
- [x] Local frontend production build.
- [x] GitHub CI.

### 4.2 Generated strategy sandbox
- [x] Normal strategy executes.
- [x] Forbidden imports rejected.
- [x] File/network/system access rejected.
- [x] Timeout enforced.
- [x] Invalid output rejected.
- [x] Resource limits reviewed.

**Verification**
- [x] Focused generated-strategy sandbox acceptance suite: 18 tests passed locally on Windows.
- [x] Normal pandas/numpy strategy executes in an isolated child Python process and returns aligned finite signals.
- [x] Forbidden imports and file/network/system capabilities are rejected before execution.
- [x] Wall-clock execution limit is enforced and capped by `STRATEGY_SANDBOX_MAX_TIMEOUT_SECONDS` (default 30s).
- [x] Non-Series and non-finite outputs are rejected.
- [x] Resource limits: 100 KB source cap, `STRATEGY_SANDBOX_MAX_BARS` (default 20,000), and child-process RSS ceiling via `STRATEGY_SANDBOX_MEMORY_MB` (default 512 MB).
- [x] Memory ceiling verified in GitHub Linux CI and local Windows tests.
- [x] Local full backend regression suite.
- [x] GitHub CI.

### 4.3 Draft backtest correctness
- [x] Fix win-rate calculation.
- [x] Align draft and saved-strategy accounting logic.
- [x] Verify flips, entries, exits and final open trade handling.

**Verification**
- [x] Focused draft-backtest correctness acceptance suite: 3 tests passed locally on Windows.
- [x] Draft win rate is derived from trade-level outcomes instead of being hard-coded to `0.0`.
- [x] Draft and saved-strategy backtests use the same accounting implementation for return, trade count, win rate, average trade, drawdown, Sharpe/SQN, trade frequency, and hold duration.
- [x] Long-to-short flips close the prior trade and open the new direction.
- [x] Explicit flat signals close the active trade.
- [x] A trade already open before the final bar is marked through the final bar; a final-bar-only entry with no holding period is not counted.
- [x] Local full backend regression suite.
- [x] GitHub CI.

### 4.4 Backtest realism
- [x] Fees.
- [x] Slippage.
- [x] Spread assumptions.
- [x] Execution timing / no lookahead.
- [x] Missing bars.
- [x] Time zones/session assumptions.
- [x] Trade-level return math.
- [x] Sharpe calculations.
- [x] Drawdown.
- [x] Hold duration.
- [x] Position sizing.

**Verification**
- [x] Focused Phase 4.4 acceptance suite plus Phase 4.3 accounting regression: 15 tests passed locally on Windows.
- [x] Signals generated from a completed bar execute only at the next bar open; same-bar close information cannot earn the same bar's return.
- [x] Fees and slippage are modeled per transaction; quoted spread is modeled as half-spread per transaction.
- [x] Strategy Studio exposes fee, slippage, spread, and position-size assumptions for both draft and saved-strategy backtests.
- [x] Portfolio allocation is explicit and constrained to greater than 0% and at most 100%; P&L scales with configured position size.
- [x] Long and short P&L use linear CFD-style price-return math rather than reciprocal short returns.
- [x] Open trades are marked to the final close without inventing an unexecuted exit cost.
- [x] Backtest input timestamps are normalized to UTC; duplicate timestamps/invalid OHLC are rejected.
- [x] Missing broker bars are reported as observed gaps and are never synthetically forward-filled; session assumption is broker-observed bars only.
- [x] Total return, trade return, drawdown, Sharpe/SQN, and hold duration are derived from the same mark-to-market equity/accounting path.
- [x] Sharpe uses observed UTC daily equity returns annualized with `sqrt(252)`; trade Sharpe/SQN remain separately reported.
- [x] Hold duration is reported in both bars and elapsed minutes.
- [x] Local full backend regression suite.
- [x] Local frontend production build.
- [x] GitHub CI #65 and #66.

### 4.5 Validation methodology
- [x] Development split.
- [x] 30% out-of-sample holdout.
- [x] Regime/different-market test.
- [x] Walk-forward option.
- [x] Parameter optimization without leakage.
- [x] Minimum sample sizes.

**Verification**
- [x] Development evidence uses the first chronological 70% of the requested dataset; holdout evidence uses the final untouched 30%, with split metadata recorded in the result.
- [x] Development and holdout windows enforce minimum bar counts; lifecycle evidence continues to enforce minimum trade counts.
- [x] Regime evidence must differ from passing development evidence by market/timeframe or use a non-overlapping period; regime sample sufficiency uses the lifecycle trade-count gate.
- [x] Strategy Studio exposes expanding-window walk-forward validation; each fold trains on prior history and evaluates only on a strictly later test window.
- [x] Parameter grids are bounded and selected on development data only; the chosen parameters are evaluated once on the untouched holdout.
- [x] Leakage acceptance case deliberately makes development favor one direction while holdout favors the opposite; parameter selection remains driven solely by development data.
- [x] Focused Phase 4.5/local lifecycle/API acceptance suite: 25 tests passed locally on Windows.
- [x] Local full backend regression suite.
- [x] Local frontend production build.
- [x] GitHub CI #71.

### 4.6 Strategy lifecycle
Validate:

`draft → backtested → validated → paper → eligible`

- [x] Hypothesis required.
- [x] Development evidence.
- [x] OOS evidence.
- [x] Regime evidence.
- [x] Paper evidence.
- [x] Operator/reason required for promotion.
- [x] Source hash/version changes invalidate old evidence appropriately.
- [x] Generated strategy can never jump directly into autonomous execution.

**Verification**
- [x] Focused Phase 4.6 lifecycle/API/execution/recovery acceptance suite: 49 tests passed locally on Windows.
- [x] Promotion remains sequential across `draft → backtested → validated → paper → eligible`.
- [x] A measurable hypothesis plus passing development, holdout, regime, and paper evidence are required at the appropriate promotion gates.
- [x] Promotion requires a named operator and non-empty audit reason at both API and service boundaries.
- [x] Source edits create a fresh lifecycle version at `draft`; prior evidence and transitions remain attached to the old source hash.
- [x] Governed paper positions persist the exact lifecycle source hash, including cTrader demo tracker recovery.
- [x] Paper evidence is calculated only from closed positions carrying the current lifecycle source hash; older-version paper trades cannot satisfy a new version's gate.
- [x] Generated strategy files remain research/backtest artifacts and are not imported into the trusted runtime strategy registry.
- [x] The central execution risk gate rejects lifecycle-governed strategies before `paper` and allows only the current source version at `paper`/`eligible`.
- [x] Local full backend regression suite.
- [x] Local frontend production build.
- [x] GitHub CI #76 on implementation head `a82f2312`.

---

## 7. Phase 5 — Strategy quality & confidence calibration

### 5.1 Runtime deterministic strategies
**Status:** 🟨

Current runtime strategies include:
- `sma_cross`
- `rsi_reversal`
- `breakout`

**Implemented audit coverage**
- [x] Read-only runtime strategy backtest path.
- [x] 70/30 development and out-of-sample split.
- [x] Chronological regime analysis.
- [x] Fee, slippage, spread, and cost-drag reporting.
- [x] Trade-frequency reporting.
- [x] Maximum-drawdown reporting.
- [x] Expectancy reporting.
- [x] Development-only parameter-sensitivity diagnostics.
- [x] Strategy × symbol × timeframe matrix runner.
- [x] Runtime `no_trade` hold semantics and next-bar-open execution preserved.
- [x] No broker order-routing or live-account behavior changed.

**Verification**
- [x] Focused `backend/tests/test_runtime_strategy_validation.py -q`: 10 passed locally on Windows.
- [x] Full `backend/tests -q`: passed locally.
- [x] Frontend production build: passed locally (699 modules).
- [x] GitHub CI #79 on implementation head `f4e6038a`: backend + frontend passed.
- [x] Connected-feed audit attempt correctly failed closed when no cTrader historical feed was available.
- [ ] Run the real connected cTrader demo historical-feed matrix for enabled targets and record actual backtest/OOS/regime/cost/frequency/drawdown/expectancy/sensitivity evidence.

**Field-attempt evidence (2026-09-29)**
- Enabled runtime targets were `NAS100 / M5`, `US30 / M5`, and `XAUUSD / M5`.
- All three returned `No market data available` because the cTrader feed was not connected.
- Validation cost assumptions were fee 1 bps + slippage 1 bps per transaction and quoted spread 2 bps; these remain validation assumptions, not broker-verified transaction costs.

### 5.2 Confidence calibration
**Status:** ✅

Compare signal-strength buckets with realized outcomes:

- <60%
- 60–65%
- 65–70%
- 70–75%
- 75–80%
- 80%+

**Implemented**
- [x] Link each closed trade to the exact opening order intent via persisted trade-audit identity; no symbol/direction/timestamp guessing.
- [x] Report win rate by bucket.
- [x] Report expectancy by bucket while keeping account currencies separate.
- [x] Report average R using opening stop-defined monetary risk.
- [x] Keep trades without a valid initial-risk denominator in win-rate/expectancy counts but exclude them from R metrics.
- [x] Report chronological cumulative-R drawdown contribution by bucket.
- [x] Report trade counts plus broker-vs-paper and realized-P&L-source composition.
- [x] Exclude manual trades by default, with explicit diagnostic opt-in.
- [x] Support strategy, symbol, and timeframe filters.
- [x] Expose a read-only `/api/studio/confidence-calibration` endpoint.
- [x] Keep signal strength explicitly labeled as a deterministic heuristic, not a calibrated probability.
- [x] Never change execution thresholds automatically from this report.

**Verification**
- [x] Focused `backend/tests/test_confidence_calibration.py -q`: 7 passed locally on Windows.
- [x] Full `backend/tests -q`: passed locally.
- [x] GitHub CI #82 on implementation head `c3ee736`: backend tests + frontend production build passed.
- [x] Boundary, manual-exclusion, mixed-currency, missing-risk, filter, exact-linkage, and API-forwarding cases covered.

**Interpretation**
- The workflow is verified; current bucket statistics remain descriptive evidence only.
- No bucket is treated as a win probability and no per-strategy/symbol threshold is changed until Phase 5.3 has sufficient sample evidence.

### 5.3 Per-strategy / per-symbol thresholds
**Status:** 🟨 pending evidence

Only after enough evidence, evaluate whether one global 60% threshold is inferior to calibrated thresholds per strategy/symbol/timeframe.

**Implemented screening**
- [x] Read-only sample-sufficiency assessment by exact strategy × symbol × timeframe.
- [x] Exclude manual trades from the automatic-threshold evidence pool.
- [x] Require at least 100 closed automatic trades at/above the current 60% baseline.
- [x] Require worst-case 95% binomial win-rate margin of error <= 10 percentage points.
- [x] Require at least 80% initial-risk/R coverage.
- [x] Require at least 10 wins and 10 losses.
- [x] Require at least 3 populated signal-strength buckets with at least 10 trades each.
- [x] Return no selected/recommended threshold and never mutate execution settings.
- [x] Expose read-only `/api/studio/confidence-threshold-sufficiency`.

**Verification**
- [x] Focused `backend/tests/test_confidence_thresholds.py -q`: 7 passed locally on Windows.
- [x] Full `backend/tests -q`: passed locally.
- [x] GitHub CI #85 on implementation head `b3691d1`: backend tests + frontend production build passed.
- [x] Real persisted-runtime sample assessment completed.

**Real sample assessment (2026-09-29)**
- `breakout × NAS100 × M5`: 17 baseline trades; 6 wins / 11 losses; 100% R coverage; insufficient sample/bucket coverage.
- `breakout × US30 × M5`: 15 baseline trades; 6 wins / 9 losses; 100% R coverage; insufficient sample/bucket coverage.
- `sma_cross × XAUUSD × M5`: 25 baseline trades; 7 wins / 18 losses; 100% R coverage; insufficient sample/bucket coverage.
- 0 of 3 cells passed the screening gate.
- No threshold comparison or execution-setting change is authorized from the current sample.

**Pending acceptance**
- [ ] Re-run the sufficiency screen after substantially more closed automatic trades accumulate.
- [ ] Only if a cell passes the screen, perform a leakage-safe development/holdout comparison of threshold alternatives.
- [ ] Keep the global 60% threshold unchanged until such evidence exists.

---

## 8. Phase 6 — Market intelligence / events / confluence research

These features remain research/shadow-only until validated.

### 6.1 Calendar / events
**Status:** ✅

- [x] Event ingestion.
  - Content-hash idempotency prevents duplicate persisted events.
  - Configured RSS refreshes persist source-run evidence and bound feed size/timeouts.
- [x] Upcoming event display.
  - Market Context displays the next scheduled calendar event as read-only evidence.
  - Past/non-finite `CALENDAR_NEXT_TS` values are rejected so expired events cannot be shown as upcoming.
- [x] Symbol/event mapping.
  - Deterministic aliases map recognized companies/instruments to symbols.
  - Nasdaq-component events propagate NAS100 context.
- [x] Missing/stale data handling.
  - Read-only `/api/market/events/status` distinguishes `not_configured`, never-refreshed/missing, stale, failed/degraded, and healthy sources.
  - Source freshness is derived from persisted refresh runs; no source health is invented when feeds are absent.

**Verification**
- [x] Focused `backend/tests/test_event_intelligence.py backend/tests/test_calendar_v2.py -q`: 11 passed locally on Windows.
- [x] Full `backend/tests -q`: passed locally.
- [x] Frontend production build: passed locally (699 modules transformed).
- [x] GitHub CI #88 on implementation head `b80aca6`: backend tests + frontend production build passed.
- [x] Real runtime truth check: event sources currently `not_configured`; next calendar event currently `null`. Both are valid fail-closed states.
- [x] Event/calendar context remains research/read-only and cannot create, approve, or route trades.

### 6.2 Event calibration
**Status:** ✅

- [x] Outcome windows.
  - Fixed 5m, 30m, 4h, and 1d research horizons use the first closed M5 bar at/after the event timestamp as the reference.
  - Each horizon exposes its exact duration and flat-return threshold in the calibration methodology.
- [x] Sample counts.
  - Reports evaluated, pending, and unavailable outcome totals plus per-event-type/per-horizon sample counts.
- [x] Calibration gates.
  - Minimum-sample, hit-rate, and Brier-score gates remain research-only and classify groups as insufficient, observe, eligible, or degraded.
- [x] Persistence.
  - Outcomes persist per event × symbol × horizon and evaluated outcomes remain unchanged on rerun.
- [x] Reproducibility.
  - Stable methodology version `event-calibration-v1` exposes the bar timeframe, reference rule, outcome windows, thresholds, and gate policy.
  - Fully evaluated symbols skip market-data refetch on rerun, preventing transient feed outages from falsely invalidating completed calibration evidence.

**Verification**
- [x] Focused `backend/tests/test_event_calibration.py -q`: 7 passed locally on Windows.
- [x] Full `backend/tests -q`: passed locally.
- [x] Frontend production build: passed locally (699 modules transformed).
- [x] GitHub CI #91 on implementation head `84611a5`: backend tests + frontend production build passed.
- [x] Real runtime calibration report inspected: methodology version `event-calibration-v1`, research-only true, 0 evaluated / 0 pending / 0 unavailable outcomes, and no groups.
- [x] Calibration remains research/read-only and does not alter confluence promotion or execution behavior.

### 6.3 Confluence shadow mode
**Status:** ✅

- [x] Original vs shadow confidence.
  - Persist both the original strategy confidence and the shadow-adjusted confidence plus pass/fail threshold outcomes.
  - Execution continues to receive the untouched original `StrategyAnalysis`.
- [x] No-trade can never be promoted.
  - `no_trade` keeps zero confidence adjustment, remains `no_trade`, and cannot pass the shadow execution threshold.
- [x] Event evidence caps.
  - Aligned evidence can add at most +12 confidence points; conflicting evidence can remove at most 20 points.
- [x] Replay.
  - Decision-cohort replay uses next-bar entry, stored horizon exit, transaction costs, and one overlapping position per symbol/timeframe/strategy.
  - Replay remains research-only and can only return a candidate for human review.
- [x] Forward paper observation.
  - Normal `auto_loop` scans persist shadow observations before the unchanged original execution path.
  - Real restarted-runtime observation confirmed new policy-stamped forward records.
- [x] Promotion remains manual/reviewed.
  - New shadow records stamp `promotion_policy=manual_review_only`, `automatic_promotion=false`, and `execution_source=original_strategy_analysis`.
  - Shadow subsystem failure is logged but cannot block or modify the original execution decision path.

**Verification**
- [x] Focused confluence shadow/integration/replay tests: 13 passed locally on Windows.
- [x] Full `backend/tests -q`: passed locally.
- [x] GitHub CI #94 on implementation head `97daad4`: backend tests + frontend production build passed.
- [x] Existing persisted sample audit: no no-trade violations, no execution-changed records, and no promotion-policy violations.
- [x] Real forward observation after backend restart: 4 policy-stamped `auto_loop` records observed.
- [x] Latest real `XAUUSD × M5 × sma_cross` record preserved original confidence 0.85, shadow confidence 0.85, `execution_unchanged=true`, `automatic_promotion=false`, and `execution_source=original_strategy_analysis`.
- [x] Replay remained `insufficient_data` with 0 priced decisions because market bars were unavailable for 255 historical records; no promotion was authorized.

---

## 9. Phase 7 — Resilience / failure testing

Simulate deliberately:

- [x] cTrader disconnect while flat.
  - Demo execution preflight now requires an active cTrader transport and authorization before cached demo-account/symbol metadata can count as execution-ready.
  - Known pre-submission disconnects defer safely with `retryable=true`; the same bar is not consumed and may be retried after recovery.
  - Broker submission itself refuses disconnected/unauthorized state as defense in depth.
  - Ambiguous failures after a broker submission remain non-retryable to avoid duplicate demo orders.
  - Status exposes `broker_disconnected` while the transport is down and clears it after recovery.
  - Verification:
    - 5 scenario-specific resilience tests passed locally.
    - 69 focused broker/demo/engine/status tests passed locally.
    - Full backend suite passed locally.
    - GitHub CI #99 on implementation head `66e9261` passed backend tests + frontend production build.
    - Real cTrader demo transport probe started flat with 0 positions / 0 pending orders, observed a real disconnect, fail-closed preflight reason `cTrader transport is not connected.`, actionable `broker_disconnected`, no new local order intent, and deterministic recovery to connected/authorized/demo with execution ready.
    - Field-probe baseline was connected/authorized/demo but its first status snapshot had `execution_ready=false`; automated acceptance tests provide the pre-submission retry/duplicate-order proof, while the real probe supplies transport/status/recovery evidence.
- [ ] cTrader disconnect with open protected position.
- [ ] Backend restart with broker position open.
- [ ] Frontend restart.
- [ ] Ollama/model unavailable.
- [ ] Symbol metadata delayed.
- [ ] Market bars stale.
- [ ] Market data malformed.
- [ ] Order acknowledgement timeout.
- [ ] Protection amend rejected.
- [ ] Broker close rejected.
- [ ] Partial close.
- [ ] SQLite busy/locked.
- [ ] DB restart/recovery.
- [ ] Google Drive unavailable (repo code still usable).
- [ ] Clock/time-zone edge cases / DST.

For every scenario verify:
- no unintended live routing,
- no duplicate demo orders,
- broker remains source of truth,
- incident is actionable,
- recovery is deterministic.

---

## 10. Phase 8 — Observability & reporting

- [ ] Daily summary: trades, realized P&L, drawdown, win/loss, rejected signals by reason.
- [ ] Broker-vs-local reconciliation health.
- [ ] Protection-health metric.
- [ ] Engine cycle latency.
- [ ] Broker/API latency.
- [ ] Error-rate / incident grouping.
- [ ] Strategy performance by symbol/timeframe.
- [ ] Export journal / statement comparison.

---

## 11. Phase 9 — Repository / documentation cleanup

- [ ] Update README test count and verification date.
- [ ] Update screenshots after UI stabilizes.
- [ ] Clearly mark/remove stale legacy architecture files where safe.
- [ ] Ensure operational docs use the current runtime DB path.
- [ ] Add a single canonical validation command.
- [ ] CI for backend tests + frontend build.
- [ ] Optional lint/type-check gate.
- [ ] Dependency/deprecation cleanup (`protobuf`, Starlette/httpx warning).

---

## 12. Acceptance definition

TradeAgent is considered **demo-runtime validated** when all of the following are true:

- [ ] Full backend test suite passes.
- [ ] Frontend production build passes.
- [ ] Repeated demo trades use correct quantities.
- [ ] Every broker-managed position is either verified protected or fails closed.
- [ ] Broker position ID is the canonical identity.
- [ ] Closed-trade price/time/net P&L reconcile to cTrader.
- [ ] Broker account currency/equity drive demo monetary risk controls.
- [ ] Restart/recovery with open positions is verified.
- [ ] Safety controls are acceptance-tested.
- [ ] Trade Journal explains decisions clearly.
- [ ] Build & Test is methodologically validated.
- [ ] Generated strategy code remains isolated from autonomous execution until explicit lifecycle approval.
- [ ] Failure scenarios do not create duplicate/unprotected/untracked exposure.
- [ ] Documentation matches current implementation.

---

## 13. Work log

Add one row after every completed task.

| Date | Item | Change / test | Result | Commit / PR | Follow-up |
|---|---|---|---|---|---|
| 2026-09-23 | Broker realized P&L + local time | cTrader closing-deal reconciliation and browser-local timestamp rendering | ✅ Verified against cTrader statement | PR #7 / `ec5fc19` | Observe fresh closes |
| 2026-09-23 | Engine restart | Restarted after merge and verified loop | ✅ `running=True`, `loop_active=True` | `main` | None |
| 2026-09-24 | Full backend suite baseline | `pytest backend/tests -q` | ⛔ 3 demo workflow failures; old `- V2` path appears in failures | — | Phase 0.1 |
| 2026-09-24 | Frontend build baseline | `npm --prefix frontend run build` | 🧪 In progress when recorded | — | Record final output |
| 2026-09-27 | Broker partial-close ledger | Immutable cTrader deal ledger + partial-close reconciliation | 🧪 Implementation/tests/CI verified; real field partial-close pending | PR #14 / `5296a30` | Field verify next normal demo position |
| 2026-09-27 | cTrader asset metadata cache | Cache account currency asset map and compact huge asset-list logging | ✅ Local regression + runtime log + CI | PR #15 / `3ad4517` | None |
| 2026-09-27 | Intent rejection UX | Exact rejection/failure rationale shown in Intents panel | ✅ Frontend build + CI | PR #16 / `d1d6eef` | Trade Journal summary remains Phase 2.1 |
| 2026-09-27 | Position identity UX | Local and cTrader IDs + account-currency P&L exposed in Positions panel | ✅ Frontend build + CI | PR #17 / `1cb5e8f` | Finish broker entry/protection/sync fields in 2.3 |
| 2026-09-27 | Monetary execution readiness | Demo execution readiness requires verified account currency and positive equity | ✅ Full tests + real CHF demo runtime + CI | PR #18 / `8938c2b` | None |
| 2026-09-27 | Trade Journal rejection reasons | Replace generic rejection summary with actionable operator-facing reason | ✅ Focused/full tests + runtime rejected-order field check + CI | PR #19 | Phase 2.2 |
| 2026-09-27 | Lot display precision | Align operator-facing lot values with cTrader-style precision (0.01 rather than 0.0100; preserve finer broker steps) | ✅ Focused tests + full backend suite + API regression + CI | PR #20 | None |
| 2026-09-27 | Signal strength semantics | Relabel heuristic confidence as signal strength, explain the threshold, and update rejection wording without changing schema/execution behavior | ✅ Frontend production build + full backend suite + CI | PR #21 | Phase 2.3 |
| 2026-09-27 | Position broker truth | Read-only cTrader snapshot exposes broker entry, SL/TP, protection, sync time, and identity status without mutating the local ledger | ✅ Focused tests + full backend suite + frontend build + CI; real open-demo field observation pending | PR #22 | Phase 2.4 |
| 2026-09-27 | Trade Journal filtering / drill-down | Add execution/rejection/protection, symbol, strategy, date, and broker-vs-paper filters plus expandable intent/risk/broker details | ✅ Local frontend build + full backend suite + CI | PR #23 | Phase 3.1 |
| 2026-09-27 | System engine lifecycle controls | Add explicit runtime-loop Restart and acceptance-test Start, Stop, Restart, one-shot Scan, Recover, and Reconcile | ✅ Focused/full backend tests + frontend build + CI; real open-demo backend restart pending | PR #24 | Phase 3.2 |
| 2026-09-27 | System safety controls | Expose per-symbol position cap on System page and acceptance-test all Phase 3.2 safety gates | ✅ 11 focused tests + full backend suite + frontend build + CI | PR #25 | Phase 3.3 |
| 2026-09-27 | Persistence acceptance | Move runtime SQLite DB to OS-local state, preserve legacy data via one-time migration, and verify config/watchlist persistence plus Git hygiene | ✅ 6 focused tests + full backend suite + resolved-path/tracked-state checks + CI | PR #26 | Phase 3.4 |
| 2026-09-27 | Status truthfulness | Separate current status/current incidents from readiness and history; derive engine scanning from runtime loop activity and clear stale recovery errors | ✅ 6 focused tests + full backend suite + frontend build + CI | PR #27 | Phase 4.1 |
| 2026-09-27 | Strategy Studio LLM workflow | Acceptance-test chat/drafting/provider selection/fallback/save and add safe saved-source reload into the editor | ✅ 6 focused tests + full backend suite + frontend build + CI | PR #28 | Phase 4.2 |
| 2026-09-27 | Generated strategy sandbox | Harden isolated strategy execution with explicit timeout/bar/memory limits and invalid-output checks; acceptance-test restricted capabilities | ✅ 18 focused Windows tests + full backend suite + Linux CI + Windows memory-limit verification | PR #29 | Phase 4.3 |
| 2026-09-28 | Draft backtest correctness | Share draft/saved accounting, calculate trade-level win rate, and verify flips/exits/final-open handling | ✅ 3 focused tests + full backend suite + CI | PR #30 | Phase 4.4 |
| 2026-09-28 | Backtest realism | Remove same-bar lookahead; unify mark-to-market accounting; model fees/slippage/spread, UTC gaps, linear short P&L, Sharpe/drawdown/hold duration, and explicit position sizing | ✅ 15 focused tests + full backend suite + frontend build + CI #65/#66 | PR #31 | Phase 4.5 |
| 2026-09-28 | Validation methodology | Formalize 70/30 chronological development/holdout validation, independent regime evidence, walk-forward folds, leakage-safe parameter optimization, and sample gates | ✅ 25 focused Windows tests + full backend suite + frontend build + CI #71 | PR #32 | Phase 4.6 |
| 2026-09-29 | Strategy lifecycle | Bind governed paper evidence to exact source versions, require promotion audit metadata, preserve sequential promotion, and block generated strategies from bypassing lifecycle governance | ✅ 49 focused Windows tests + full backend suite + frontend build + CI #76 | PR #33 | Phase 5.1 |
| 2026-09-29 | Runtime deterministic strategy validation | Add read-only Phase 5.1 audit service/matrix for trusted runtime strategies with OOS, regime, costs, frequency, drawdown, expectancy, and development-only parameter sensitivity | 🟨 10 focused tests + full backend + frontend build + CI #79 passed; connected cTrader historical-feed matrix attempted but unavailable because feed was disconnected | PR #34 / `f4e6038` | Keep real connected-feed evidence pending; proceed to Phase 5.2 development |
| 2026-09-29 | Signal-strength calibration | Add exact intent→closed-trade calibration report with bucketed win rate, account-currency expectancy, average R, R-drawdown contribution, sample composition, and safe filtering | ✅ 7 focused tests + full backend suite + CI #82 backend/frontend | PR #35 / `c3ee736` | Phase 5.3 only after sufficient sample evidence |
| 2026-09-29 | Threshold sample sufficiency | Add fail-closed per-cell screening before any threshold study; verify against real persisted runtime data | 🟨 7 focused tests + full backend suite + CI #85; real sample: 0/3 cells sufficient (17, 15, and 25 baseline trades) | PR #36 / `b3691d1` | Keep 60% unchanged; revisit after more evidence and proceed to Phase 6.1 |
| 2026-09-30 | Calendar / event acceptance | Accept existing ingestion, upcoming-event display, and deterministic symbol mapping; add truthful calendar expiry and persisted event-source freshness states | ✅ 11 focused tests + full backend suite + frontend build + CI #88; runtime sources not configured and next event null fail closed correctly | PR #37 / `b80aca6` | Proceed to Phase 6.2 event calibration acceptance |
| 2026-09-30 | Event calibration acceptance | Accept fixed outcome windows, persisted samples, research gates, and reproducible reruns; expose stable calibration methodology | ✅ 7 focused tests + full backend suite + frontend build + CI #91; runtime currently has 0 calibration outcomes/groups | PR #38 / `84611a5` | Proceed to Phase 6.3 confluence shadow-mode acceptance |
| 2026-09-30 | Confluence shadow-mode acceptance | Lock the execution boundary, no-trade non-promotion, bounded event influence, replay, forward observation, and manual-review-only promotion | ✅ 13 focused tests + full backend suite + CI #94; restarted runtime produced 4 policy-stamped auto-loop observations with execution unchanged | PR #39 / `97daad4` | Proceed to Phase 7 first resilience scenario: cTrader disconnect while flat |
| 2026-09-30 | Resilience: cTrader disconnect while flat | Fail closed before broker submission on disconnect/authorization loss; preserve same-bar retry only before submission and prevent duplicate replay after recovery | ✅ 5 scenario tests + 69 focused tests + full backend suite + CI #99; real flat demo transport disconnect/recovery probe passed all safety invariants | PR #40 / `66e9261` | Proceed to Phase 7 next scenario: cTrader disconnect with an open protected position |

---

## 14. Next item

**Phase 7 — Resilience / failure testing, second scenario only: acceptance-test cTrader disconnect with an open protected TradeAgent-managed demo position. Verify the broker remains the source of truth, no local-only SL/TP close occurs while disconnected, no duplicate/competing broker action is created, protection state remains explicit/actionable, and recovery deterministically reconciles the same broker position by canonical broker position ID. Do not proceed to the next failure scenario until this one is locally validated and CI passes. If the required real protected demo position is not available, leave only the field-observation acceptance check pending after safe implementation/testing and continue according to the roadmap. Phase 5.3 threshold comparison remains pending; keep the global 60% threshold unchanged. Phase 5.1 real connected historical-feed matrix, Phase 3.1 real backend restart with an open TradeAgent-managed demo broker position, Phase 1.4 real partial-close verification, and Phase 2.3 real broker-truth field observation remain pending until the required demo field conditions are available.**
