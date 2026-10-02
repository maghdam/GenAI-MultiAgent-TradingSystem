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
  - [x] Same-bar maintenance checks cTrader transport, authorization, and demo confirmation before broker position reads or protection mutation.
  - [x] While disconnected, the local TradeAgent tracker stays open even if the local mark crosses stored SL/TP; no synthetic local protective exit is allowed.
  - [x] While disconnected, broker position reads, protection amendments, broker closes, and competing demo orders are suppressed.
  - [x] A position-specific `ctrader_demo_protection_verification_deferred` incident records the canonical broker position ID and explicitly marks broker protection as unverified while unavailable.
  - [x] Recovery resumes verification against the persisted canonical broker position ID rather than adopting another same-side position.
  - [x] New-bar protection-sync failure remains retryable so the bar is not consumed while broker protection cannot be verified.
  - Verification:
    - [x] 3 scenario-specific protected-position resilience tests passed locally.
    - [x] 47 focused Phase 7 / engine / reconciler / status tests passed locally.
    - [x] Full backend suite passed locally.
    - [x] GitHub CI #103 on implementation head `ea4f899` passed backend tests + frontend production build.
    - [ ] Real protected-position disconnect/recovery field observation is pending. Candidate probe found 0 broker positions and 0 local open trackers, so no qualifying TradeAgent-managed protected demo position was available; no trade was manufactured for testing.
- [ ] Backend restart with broker position open.
  - [x] Startup recovery keeps the existing cTrader-first boot order and runs tracker recovery before local open-position reconciliation.
  - [x] A broker position is recoverable only through a TradeAgent open intent whose broker position ID, symbol, and direction match the live broker row.
  - [x] Normal recovery requires an executed open intent; the only failed-intent exception is an explicitly retained still-open fail-safe case with `tracking_retained=true` and no successful fail-safe close.
  - [x] Recovery is idempotent: repeating startup reconciliation does not duplicate the local tracker or create a competing broker position/order.
  - [x] Broker entry price and broker volume remain the source of truth when reconstructing a missing local tracker.
  - [x] Startup before broker execution readiness preserves an existing demo tracker and performs no broker read/amend/close or synthetic local protective exit.
  - Verification:
    - [x] 6 scenario-specific backend-restart resilience tests passed locally.
    - [x] 53 focused restart / reconciler / engine / Phase 7 status tests passed locally.
    - [x] Full backend suite passed locally.
    - [x] GitHub CI #106 on implementation head `4b6286a` passed backend tests + frontend production build.
    - [ ] Real backend-restart-with-open-position field observation is pending. The latest real candidate probe found 0 broker positions and 0 local open trackers, so no qualifying TradeAgent-managed demo position was available; no trade was manufactured for testing.
- [x] Frontend restart.
  - [x] Frontend mount/reload is read-only with respect to backend engine/config/order state; mutating actions remain behind explicit operator controls.
  - [x] Dashboard status and strategy reads settle independently so one transient failure does not discard the other successful response.
  - [x] Last-known-good frontend state is retained during transient read degradation, with an explicit degraded banner and automatic retry.
  - [x] Recovery is deterministic: a later successful refresh clears the degraded state and reconstructs current backend truth.
  - [x] React StrictMode may duplicate read effects in development, but restart bootstrap exposes no mutation callback and cannot create duplicate orders/config writes.
  - Verification:
    - [x] 3 frontend-restart acceptance tests passed locally.
    - [x] Frontend production build passed locally.
    - [x] GitHub CI #109 on implementation head `669139b` passed backend tests, frontend restart acceptance tests, and frontend production build.
    - [x] Real frontend-only restart observation passed: backend config, engine enabled/running/loop state, mode, demo confirmation, and position identity were unchanged; the UI reloaded normally without operator mutation clicks.
    - [x] One new intent (`2128`) appeared during the observation window, but it was a backend-generated rejected `skip` intent from the autonomous scan path, not a frontend-triggered open/close/update or duplicate broker action.
- [ ] Ollama/model unavailable.
  - [x] Deterministic strategy analysis/execution remains independent from Ollama availability; `/api/analyze` does not query model health or create LLM-dependent execution state.
  - [x] Failed Ollama health responses are cached only briefly (5 seconds) so recovery is rechecked promptly instead of remaining stale behind the healthy-response TTL.
  - [x] Strategy Studio fails fast when Ollama is unreachable or reachable with no installed models; it does not cycle through hidden generation attempts while the provider itself is down.
  - [x] Installed-model fallback remains bounded and explicit when Ollama is reachable.
  - [x] `model_not_ready` status is actionable and explicitly states that deterministic trading remains independent while LLM/research actions are degraded.
  - [x] Strategy-drafting fallback is research-only and creates no order intents.
  - Verification:
    - [x] 7 scenario-specific Ollama/model-unavailable resilience tests passed locally.
    - [x] 64 focused model / Studio / status / engine tests passed locally.
    - [x] Full backend suite passed locally.
    - [x] GitHub CI #113 on implementation head `637fc00` passed backend tests + frontend production build.
    - [ ] Real forced Ollama-down/recovery observation remains pending. The attempted field probe could not hold Ollama offline because the Windows Ollama supervisor immediately respawned the listener on port 11434; the probe therefore never observed `ollama_ready=false`. During that attempted probe, config/engine/mode/position identity stayed unchanged and no dangerous open/close/update intent appeared.
- [x] Symbol metadata delayed.
  - [x] Current-session cTrader symbol-contract readiness is tracked explicitly; stale prior-session maps cannot satisfy readiness after reconnect.
  - [x] Symbol metadata is invalidated on connect/disconnect, including cached symbol IDs, lot sizes, volume limits, and verification hints.
  - [x] Fallback/light symbol data remains available for discovery but is never treated as broker-verified contract metadata for demo execution.
  - [x] Global broker readiness stays false until the current broker full-contract response contains complete lot-size/min/step/max metadata for the current symbol set.
  - [x] Symbol limits and instrument specs fall back safely while current-session metadata is incomplete instead of labeling stale/synthetic values as broker-verified.
  - [x] Demo execution defers retryably before intent creation or broker submission while metadata is delayed; direct broker submission independently rechecks the same readiness boundary.
  - [x] Status exposes actionable `symbol_metadata_unavailable` / execution-not-ready state while delayed, and readiness recovers deterministically after complete broker contract metadata arrives.
  - Verification:
    - [x] 6 scenario-specific delayed-symbol-metadata resilience tests passed locally.
    - [x] Focused broker / demo-account / disconnect / engine / status regression set passed locally.
    - [x] Full backend suite passed locally.
    - [x] GitHub CI #116 on implementation head `6e8e87c` passed backend tests + frontend production build.
- [x] Market bars stale.
  - [x] Market-bar freshness is classified by one shared timeframe-aware rule used by both runtime preflight and execution risk checks.
  - [x] Auto-loop freshness is checked immediately after bars are fetched and before same-bar marking, broker protection synchronization, strategy analysis, order-intent creation, or broker submission.
  - [x] A stale bar is treated as a retryable market-data dependency skip rather than a consumed trading decision; bar state is not advanced and no order intent is created.
  - [x] Stale same-bar data cannot mark/close local positions or mutate broker protection.
  - [x] Repeated stale observations for one symbol/timeframe episode produce one actionable `market_data_stale` incident instead of an incident/rejection storm.
  - [x] Current broker/market status exposes the latest bar timestamp and an active stale-market-data reason while stale; the incident clears after a fresh bar is observed.
  - [x] Recovery is deterministic: the first fresh bar is processed normally, then the persisted bar-state guard prevents duplicate replay of that same bar.
  - Verification:
    - [x] 6 scenario-specific stale-market-bar resilience tests passed locally.
    - [x] Focused market-data / engine / risk / status / demo regression set passed locally.
    - [x] Full backend suite passed locally.
    - [x] GitHub CI #119 on implementation head `fdb24cc` passed backend tests + frontend production build.
- [x] Market data malformed.
  - [x] OHLC integrity validation is centralized and shared by the cTrader adapter, market-data service/cache layer, auto-loop trading boundary, and execution risk engine.
  - [x] Incomplete, non-numeric/non-finite, non-positive, invalid high/low, open/close outside range, and extreme-range bars fail closed.
  - [x] cTrader no longer silently drops malformed OHLC rows and falls back to an older valid bar; established invalid-timestamp row filtering is preserved when valid timestamped rows remain.
  - [x] Memory-cache and persisted-cache frames are validated before being marked healthy or returned to the engine.
  - [x] Auto-loop validation occurs before same-bar marking, broker protection synchronization, strategy analysis, order-intent creation, or broker submission.
  - [x] Malformed data is treated as retryable: bar state is not advanced and no automatic order intent or broker action is created.
  - [x] Repeated malformed observations for one symbol/timeframe episode are incident-deduplicated and surfaced as actionable `market_data_malformed`; the active incident clears after valid data returns.
  - [x] Recovery is deterministic: the first valid bar after a malformed episode processes normally, then the normal bar-state guard prevents duplicate replay.
  - Verification:
    - [x] 12 scenario-specific malformed-market-data pytest cases passed locally.
    - [x] Focused market / engine / risk / status / adapter regression set passed locally.
    - [x] Full backend suite passed locally.
    - [x] GitHub CI #123 on implementation head `b21a9f9` passed backend tests + frontend production build.
- [x] Order acknowledgement timeout.
  - [x] A cTrader demo acknowledgement timeout is classified explicitly as a post-submission ambiguity, distinct from a known pre-submit/rejected order failure.
  - [x] Broker position IDs are snapshotted immediately before demo submission; timeout reconciliation may auto-resolve only from that baseline.
  - [x] Exactly one new broker position matching symbol, direction, and quantity may resolve the timeout automatically; broker position ID, entry price, and volume become the canonical local tracker truth.
  - [x] Missing baseline or multiple new broker candidates remain unresolved; symbol-only adoption is forbidden.
  - [x] Unresolved ambiguity persists on the failed open intent with `retryable=false`, `automatic_retry=false`, and `submission_may_have_succeeded=true`.
  - [x] Later automatic open attempts for the same symbol/timeframe are blocked before a new intent or broker submission is created while the ambiguity remains unresolved.
  - [x] Current status surfaces actionable `order_acknowledgement_ambiguous`; resolved broker-confirmed timeout cases do not leave that active incident.
  - [x] Live-account routing remains blocked before submission.
  - Verification:
    - [x] 6 scenario-specific order-acknowledgement-timeout resilience tests passed locally.
    - [x] Focused execution / broker / reconciliation / disconnect / restart / status regression set passed locally.
    - [x] Full backend suite passed locally.
    - [x] GitHub CI #126 on implementation head `77c6261` passed backend tests + frontend production build.
- [x] Protection amend rejected.
  - [x] cTrader demo protection amend/verification failures are classified separately from ordinary broker precondition errors.
  - [x] Rejected/unverified protection never causes new local SL/TP targets to be written before broker verification succeeds.
  - [x] The protection fail-safe close policy is shared across new-bar signal execution, same-bar maintenance, and startup/manual reconciliation.
  - [x] Fail-safe close requires the persisted canonical broker position ID to match the broker row; no symbol-only adoption or competing open order is allowed.
  - [x] A successful fail-safe close closes the broker position first and only then closes the local tracker.
  - [x] If fail-safe close is also rejected/unavailable, the canonical local tracker is retained, the result is non-retryable for the current bar, and the incident is actionable.
  - [x] After a failed fail-safe close, repeated scans of the same bar suppress duplicate protection amendments/closes and only watch broker truth.
  - [x] Broker-truth recovery clears same-bar suppression once the intended SL/TP is later verified, without sending another amend.
  - [x] Generic disconnect/precondition failures keep the existing deferred/retry behavior and do not trigger fail-safe close.
  - Verification:
    - [x] 7 scenario-specific rejected-protection-amend resilience tests passed locally.
    - [x] Focused protection / execution / reconciliation regression set passed locally after one narrow compatibility fix.
    - [x] Full backend suite passed locally on corrected head `23c6de4`.
    - [x] GitHub CI #130 on corrected implementation head `23c6de4` passed backend tests + frontend production build.
    - [x] Initial CI #129 on `4c757a2` failed only because new optional engine bookkeeping assumed an older test double exposed `position_id`/`status`; corrected with optional `getattr` access and no protection-safety behavior change.
- [x] Broker close rejected.
  - [x] Explicit broker close rejection is classified separately from ambiguous post-submit outcomes.
  - [x] A rejected or unresolved demo close never marks the local tracker closed without verified broker truth.
  - [x] Local close requires the canonical persisted broker position ID to match the verified broker-close payload; no symbol-only close is accepted.
  - [x] Rejected/ambiguous close state is persisted against the canonical local+broker identity and surfaced as an actionable active incident.
  - [x] Automatic duplicate/competing close submissions are blocked while the prior close outcome remains unresolved.
  - [x] Later verified broker absence reconciles the existing tracker closed without resubmitting or synthesizing a local-only exit.
  - [x] New-position protection fail-safe close rejection retains canonical tracking and submits only one close.
  - Verification:
    - [x] 7 broker-close-rejected scenario tests passed locally.
    - [x] Previously failing compatibility tests for protection-amend/startup reconciliation passed locally.
    - [x] Focused close / execution / protection / reconciliation / status regression set passed locally (62 tests).
    - [x] Full backend suite passed locally on corrected head `fde9b62`.
    - [x] GitHub CI #134 on corrected implementation head `fde9b62` passed backend tests + frontend production build.
    - [x] Initial CI #133 exposed a legacy reconciler monkeypatch compatibility issue; the public close call surface was preserved without weakening close-safety behavior.
- [x] Partial close.
  - [x] Residual broker exposure keeps the same canonical broker position ID and remains locally open.
  - [x] Local quantity/P&L only advances when cumulative authoritative broker closing-deal volume supports the observed broker reduction.
  - [x] Broker deal IDs remain immutable/idempotent and repeated reconciliation does not double-count realized P&L.
  - [x] A second/later partial close recovers deterministically if its deal was persisted before a process interruption but local quantity was not yet updated.
  - [x] Missing later deal history stays pending instead of reusing prior deal volume or fabricating a local reduction.
  - [x] Broker-position identity mismatch fails closed before deal-history lookup or local mutation.
  - [x] Partial-close reconciliation does not synthesize a full local close or submit a competing broker close.
  - [x] Protection remains bound to the residual canonical broker position during pending, recovered, and repeated reconciliation.
  - Verification:
    - [x] 5 partial-close resilience scenario tests passed locally on `37b60dd`.
    - [x] Focused partial-close / ledger / reconciliation / close-safety / restart / execution regression set passed locally.
    - [x] Full backend suite passed locally on `37b60dd`.
    - [x] GitHub CI #137 on `37b60dd` passed backend tests + frontend restart/build.
    - [ ] Real cTrader demo partial-close field observation remains pending until a safe qualifying TradeAgent-managed open demo position exists.
- [x] SQLite busy/locked.
  - [x] Lock/busy failures roll back cleanly and are reported distinctly.
  - [x] Pre-submit persistence failure prevents any demo broker submission.
  - [x] A durable pre-submit reservation prevents duplicate submission if persistence later becomes unavailable.
  - [x] Broker-confirmed identity is retained across a local tracker persistence interruption and recovered deterministically.
  - [x] A post-submit persistence interruption is reconciled against broker truth before another order can be considered.
  - [x] Real SQLite exclusive-lock coverage verifies rollback and idempotent broker-deal recovery.
  - Verification:
    - [x] 5 SQLite resilience tests passed locally on `ac50103`.
    - [x] Focused persistence/execution/reconciliation regression passed locally on `ac50103`.
    - [x] Full backend suite passed locally on `ac50103` with a clean working tree.
    - [x] GitHub CI #141 on `ac50103` passed backend tests + frontend restart/build.
- [x] DB restart/recovery.
  - [x] Durable engine config/runtime state survives an actual SQLite connection close/reopen against the same database file.
  - [x] Open TradeAgent tracker and executed order intent survive restart with the same canonical broker position ID.
  - [x] Immutable broker deal IDs remain idempotent after database reopen.
  - [x] Repeated startup reconciliation preserves exactly one local tracker/intent and resumes protection against the canonical broker position.
  - [x] A broker-confirmed tracking handoff survives restart and recovers exactly one canonical local tracker.
  - [x] An unresolved durable submission reservation survives restart and continues blocking duplicate demo submission.
  - [x] Tracker-recovery dependency failure suppresses broker mutation/automatic adoption and records an actionable durable incident.
  - Verification:
    - [x] 5 DB restart/recovery resilience tests passed locally on `f02c926`.
    - [x] Focused restart/persistence/reconciliation/execution regression passed locally on `f02c926`.
    - [x] Full backend suite passed locally on `f02c926` with a clean working tree.
    - [x] GitHub CI #148 on cleaned head `f02c926` passed backend tests + frontend restart/build.
- [x] Google Drive unavailable (repo code still usable).
  - [x] Runtime dependencies remain free of Google Drive SDK coupling.
  - [x] Local API/config/incident endpoints remain usable while Drive network endpoints are unavailable.
  - [x] SQLite-backed configuration persists normally during the simulated Drive outage.
  - [x] The live-account guard still rejects `allow_live=true`; Drive unavailability cannot weaken demo-only routing.
  - [x] Engine kill-switch execution remains local and persists runtime state without Drive access.
  - [x] Runtime DB resolution remains under local application state and ignores Drive environment paths.
  - Verification:
    - [x] 5 Google Drive unavailable resilience tests passed locally on `154f02b`.
    - [x] Focused API/persistence/engine-safety regression passed locally on `154f02b`.
    - [x] Full backend suite passed locally on `154f02b` with a clean working tree.
    - [x] GitHub CI #151 on `154f02b` passed backend tests + frontend build.
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
| 2026-09-30 | Resilience: cTrader disconnect with protected position | Preserve broker source-of-truth semantics during disconnect, suppress local/broker mutation, expose deferred protection verification, and recover by canonical broker position ID | 🟨 3 scenario tests + 47 focused tests + full backend suite + CI #103 passed; real field observation pending because candidate probe found 0 broker positions / 0 local open trackers | PR #41 / `ea4f899` | Leave field check pending; proceed to Phase 7 backend-restart-with-open-position development |
| 2026-09-30 | Resilience: backend restart with open demo broker position | Harden canonical startup recovery identity, preserve broker source of truth, prove idempotent tracker reconstruction, and fail closed before broker readiness | 🟨 6 scenario tests + 53 focused tests + full backend suite + CI #106 passed; real restart field observation pending because no qualifying broker/local open position exists | PR #42 / `4b6286a` | Leave field check pending; proceed to Phase 7 frontend-restart scenario |
| 2026-09-30 | Resilience: frontend restart | Keep restart bootstrap read-only, retain partial/last-known-good UI state through transient read failures, and automatically reconstruct backend truth after reload | ✅ 3 restart tests + frontend build + CI #109; real frontend-only restart preserved config/engine/mode/demo/position invariants and UI reloaded normally; observed intent 2128 was normal backend rejected skip activity | PR #43 / `669139b` | Proceed to Phase 7 Ollama/model-unavailable scenario |
| 2026-10-01 | Resilience: Ollama/model unavailable | Keep deterministic trading independent from model availability, fail Studio model calls fast and explicitly during outage, bound fallback attempts, and recheck recovery promptly | 🟨 7 scenario tests + 64 focused tests + full backend suite + CI #113 passed; real forced-outage observation pending because Windows immediately respawned Ollama before `ollama_ready=false` could be observed | PR #44 / `637fc00` | Leave field check pending; proceed to Phase 7 symbol-metadata-delayed scenario |
| 2026-10-01 | Resilience: symbol metadata delayed | Invalidate stale cTrader contract state across sessions, keep fallback/light symbols non-executable, fail closed before intent/order submission, and restore readiness only from complete current-session broker contracts | ✅ 6 scenario tests + focused broker/demo/engine/status regression + full backend suite + CI #116 passed | PR #45 / `6e8e87c` | Proceed to Phase 7 market-bars-stale scenario |
| 2026-10-01 | Resilience: stale market bars | Fail closed before same-bar maintenance/strategy/intent/broker mutation, preserve retryable bar state, expose active stale-feed status, and process the first fresh recovery bar exactly once | ✅ 6 scenario tests + focused market/engine/risk/status regression + full backend suite + CI #119 passed | PR #46 / `fdb24cc` | Proceed to Phase 7 malformed-market-data scenario |
| 2026-10-01 | Resilience: malformed market data | Centralize OHLC integrity checks, reject malformed broker/cache data before trading logic, preserve retryable bar state, deduplicate actionable malformed-feed incidents, and recover on the first valid bar exactly once | ✅ 12 scenario cases + focused market/engine/risk/status/adapter regression + full backend suite + CI #123 passed | PR #47 / `b21a9f9` | Proceed to Phase 7 order-acknowledgement-timeout scenario |
| 2026-10-02 | Resilience: order acknowledgement timeout | Distinguish ambiguous post-submit timeouts from known failures, reconcile against a pre-submit broker-position baseline, forbid symbol-only adoption/resubmission, and track broker-confirmed recovery without a second order | ✅ 6 scenario tests + focused execution/broker/reconciliation/status regression + full backend suite + CI #126 passed | PR #48 / `77c6261` | Proceed to Phase 7 protection-amend-rejected scenario |
| 2026-10-02 | Resilience: protection amend rejected | Keep local SL/TP truthful, apply one canonical fail-safe close on rejected/unverified broker protection, retain tracking if close also fails, suppress duplicate same-bar amend loops, and recover from later broker-truth verification | ✅ 7 scenario tests + focused protection/execution/reconciliation regression + full backend suite + corrected CI #130 passed; initial CI #129 exposed one legacy test-double compatibility issue only | PR #49 / `23c6de4` | Proceed to Phase 7 broker-close-rejected scenario |
| 2026-10-02 | Resilience: broker close rejected | Separate explicit rejection from ambiguous post-submit outcomes, require verified canonical broker truth before local closure, block duplicate close submissions, retain tracking on failure, and reconcile later broker absence deterministically | ✅ 7 scenario tests + 4 compatibility tests + 62 focused regression tests + full backend suite + CI #134 passed on `fde9b62`; CI #133 compatibility regression corrected | PR #50 / `fde9b62` | Proceed to Phase 7 partial-close scenario |
| 2026-10-02 | Resilience: partial close | Make broker-volume reconciliation cumulative and identity-safe; preserve residual exposure, recover persisted-deal/local-quantity crash gaps, keep deal ingestion idempotent, and fail closed while later broker history is incomplete | 🟨 5 scenario tests + focused regression + full backend suite + CI #137 passed on `37b60dd`; real cTrader demo partial-close field observation pending | PR #51 / `37b60dd` | Leave field check pending; proceed to Phase 7 SQLite busy/locked scenario |
| 2026-10-02 | Resilience: SQLite busy/locked | Make SQLite lock handling transactional and keep demo-order recovery deterministic across pre-submit and post-submit persistence interruptions | ✅ 5 scenario tests + focused regression + full backend suite + CI #141 passed on `ac50103` | PR #52 / `ac50103` | Proceed to Phase 7 DB restart/recovery scenario |
| 2026-10-02 | Resilience: DB restart/recovery | Verify durable SQLite reopen behavior, canonical tracker/intent recovery, idempotent broker ledger, protection resumption, and persistent duplicate-order blocking across restart | ✅ 5 scenario tests + focused regression + full backend suite + CI #148 passed on cleaned head `f02c926` | PR #53 / `f02c926` | Proceed to Phase 7 Google Drive unavailable scenario |
| 2026-10-02 | Resilience: Google Drive unavailable | Verify the repo/runtime remains independent of Drive, local API/SQLite continue to work, and demo/live safety controls are unchanged when Drive is unavailable | ✅ 5 scenario tests + focused regression + full backend suite + CI #151 passed on `154f02b` | PR #54 / `154f02b` | Proceed to Phase 7 clock/time-zone/DST edge cases |

---

## 14. Next item

**Phase 7 — Resilience / failure testing, sixteenth scenario only: acceptance-test clock/time-zone edge cases and DST. Verify UTC/local-time boundaries, DST transitions, persisted timestamps, session filters, bar freshness, daily counters/limits, event timing, and reconciliation logic remain deterministic and do not create duplicate demo actions or incorrect safety state when local time shifts or clocks cross day boundaries. Preserve cTrader demo-only routing and broker source-of-truth/canonical-position safeguards. Do not proceed beyond Phase 7 resilience completion until this item is locally validated and CI passes. The existing pending real cTrader field observations and Phase 5.3 threshold comparison remain pending; keep the global 60% threshold unchanged.**
