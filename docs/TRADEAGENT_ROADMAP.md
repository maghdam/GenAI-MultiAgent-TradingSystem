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
5. Validate against the real cTrader account or UI when relevant; do not manufacture broker exposure solely for testing.
6. Record evidence below.
7. Mark the item ✅ only after verification.
8. Move to the next item.

Phase 10 supersedes the former permanent demo-only product boundary. The target architecture is account-type neutral: cTrader reports whether each authorized account is Demo or Live, the operator selects the active account, and the same guarded execution/reconciliation pipeline serves either type.

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
**Status:** ✅

- [x] Legacy missing-deal case is fail-safe: never fabricate broker P&L.
- [x] Partial-close ledger/reconciliation support implemented and regression-tested.
- [x] Multiple closing deals are weighted/summed correctly in automated tests.
- [x] Commission/swap/conversion fees are reflected in broker net P&L.
- [x] Broker deal IDs are ingested idempotently so a deal cannot be counted twice.
- [x] Real cTrader demo partial-close field verification.
  - Verified 2026-10-06 on normal TradeAgent-managed XAUUSD local `109` / broker `57783302` on verified Demo account `44089601`.
  - Broker quantity was `0.10`; the operator partially closed exactly `0.05` through cTrader, leaving `0.05` open on the same canonical broker position ID.
  - TradeAgent synchronized the residual tracker to quantity `0.05`, realized P&L `36.29 CHF`, and `realized_pnl_source=ctrader_deal_partial`.
  - The partial-close audit recorded `status=partial_close_synced`, `previous_quantity=0.1`, `remaining_quantity=0.05`, `inserted_deals=1`, `total_closed_lots=0.05`, `required_closed_lots=0.05`, and `tracked_initial_quantity=0.1`.
  - Reconciliation remained healthy with `id_match`, and the residual broker position remained fully protected with 100% protection coverage.
  - Local docs-only staging validation passed on `9b127bf`; PR #99 CI #397 passed the normal backend/frontend CI contract before this verification update.

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
- [x] Real cTrader field observation on the next normal TradeAgent-managed open demo position.
  - Verified 2026-10-06 on naturally opened XAUUSD local `107` / broker `57779123` on verified Demo account `44089601`.
  - The enriched position exposed quantity `0.1`, tracked entry `4131.68`, broker entry `4131.83`, broker SL `4135.97`, broker TP `4123.1`, account currency `CHF`, protection `protected`, broker sync `id_match`, and broker snapshot timestamp `2026-10-06T06:43:58.676505`.
  - The immediately preceding protection-health observation reported broker truth/execution ready, 1/1 assessable position fully protected, and 100% protection coverage for the same broker position.
  - Local docs-only staging validation passed on `174d26e`; PR #98 CI #394 passed the normal backend/frontend CI contract before this verification update.

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

### 2.5 cTrader-style lot-size entry
**Priority:** P1
**Status:** ✅

Keep Trade setup lot entry visually consistent with cTrader while leaving broker/account/symbol contract enforcement in the backend.

**Implemented**
- [x] Replace the misleading four-decimal UI minimum `0.0001` with the normal two-decimal lot-entry floor/step `0.01`.
- [x] Render configured lot values in cTrader-style two-decimal form such as `0.01`, `0.10`, and `1.00`.
- [x] Normalize legacy sub-`0.01` Trade setup values to `0.01` in the editable UI model.
- [x] Do not hard-code broker-specific symbol minima into the display; existing backend broker contract validation remains authoritative at execution.
- [x] Remove the misleading universal frontend `max=100` constraint because broker maxima are contract-specific.

**Verification**
- [x] Focused frontend lot-size test: 3/3 passed locally.
- [x] Local frontend production build passed (702 modules transformed).
- [x] Local browser smoke check confirmed Trade setup now uses the expected two-decimal cTrader-style presentation and no longer shows `0.0001`.
- [x] Local final implementation-head checks passed: exact head `02997a0`, clean working tree, and `git diff --check origin/main...HEAD`.
- [x] GitHub PR #101 CI #404 passed backend + frontend on implementation head `02997a0`.

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
- [x] Restart backend while broker position is open.
  - Verified 2026-10-06 with two normal TradeAgent-managed Demo positions: US30 local `108` / broker `57782376` and XAUUSD local `109` / broker `57783302`.
  - Pre-restart, both positions were canonical `id_match`, broker-protected, and the protection report measured 2/2 fully protected with 100% coverage.
  - The backend process was stopped and relaunched through `run-backend-local.cmd`. Startup briefly reported unavailable lot-size metadata before loading complete broker contract metadata, then re-authorized Demo account `44089601`.
  - Post-restart API confirmation showed engine enabled/scanning, account authorized/verified/execution-ready, 6460 symbols loaded, and the same two broker position IDs still canonical, `protected`, and `id_match` with broker SL/TP preserved.

**Verification**
- [x] Focused lifecycle/API tests.
- [x] Local full backend regression suite.
- [x] Local frontend production build.
- [x] GitHub CI.
- [x] Real cTrader demo field check: backend restart occurred while two normal TradeAgent-managed broker positions were open; both identities and broker protection truth survived recovery. Local docs-only validation passed on staging head `b2048b4`, and PR #97 CI #391 passed the full backend/frontend CI contract before this verification update.

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
**Status:** ✅

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
- [x] Run the real connected cTrader demo historical-feed matrix for enabled targets and record actual backtest/OOS/regime/cost/frequency/drawdown/expectancy/sensitivity evidence.
- [x] PR #86 adds a read-only field-verification helper that queries the already-running backend session, requires a positively confirmed demo account, discovers enabled watchlist targets, and never routes or mutates broker orders.
- [x] Local verification on `7fb61ca`: 14 focused Phase 5.1 tests passed, the full backend suite passed, the working tree remained clean, and the real connected demo matrix completed 9/9 rows. PR CI #252 passed the full backend suite plus frontend restart/Journal acceptance and production build.

**Field-attempt evidence (2026-09-29)**
- Enabled runtime targets were `NAS100 / M5`, `US30 / M5`, and `XAUUSD / M5`.
- All three returned `No market data available` because the cTrader feed was not connected.
- Validation cost assumptions were fee 1 bps + slippage 1 bps per transaction and quoted spread 2 bps; these remain validation assumptions, not broker-verified transaction costs.

**Connected cTrader demo evidence (2026-10-04)**
- Running backend reported cTrader `account_type=demo`, `demo_account_confirmed=true`, socket connected, and account authorized. The general `market_data_ready` status flag was false at the observation instant, but the read-only historical OHLC requests themselves succeeded for every audited target.
- Enabled targets were `NAS100 / M5`, `US30 / M5`, and `XAUUSD / M5`; each returned 1,379 broker bars and all three trusted runtime strategies completed, producing 9/9 evidence rows.
- Cost assumptions remained fee 1 bps + slippage 1 bps per transaction and quoted spread 2 bps, with 100% validation position size. These are validation assumptions, not broker-verified transaction costs.
- Full-period net returns were negative in all 9 cells: NAS100 SMA -7.5341%, RSI -1.8496%, breakout -2.1417%; US30 SMA -6.5659%, RSI -2.9291%, breakout -1.1970%; XAUUSD SMA -5.6441%, RSI -2.6288%, breakout -0.4054%.
- OOS returns were also negative except XAUUSD RSI (+0.1760%) and XAUUSD breakout (+0.5236%). This field evidence verifies the audit path and records current strategy quality; it does **not** justify promotion, parameter retuning, or any confidence-threshold change.

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

**Re-check protocol (2026-10-04)**
- Use the existing read-only `GET /api/studio/confidence-threshold-sufficiency` endpoint against the current persisted runtime database.
- Review every exact strategy × symbol × timeframe cell; manual trades remain excluded by the service.
- A cell may proceed to threshold-comparison research only if `sufficient_for_threshold_study=true` and every screening check passes.
- If no cell passes, record the updated sample evidence and stop; do not tune parameters, run threshold selection, or change the global 60% execution threshold.
- This re-check is evidence-only and does not route broker orders or mutate execution settings.

**Real sample re-check (2026-10-04)**
- Local focused `backend/tests/test_confidence_thresholds.py -q`: 7 passed; full `backend/tests -q`: passed; branch remained clean on `601f776`. PR CI #255 passed backend + frontend.
- Read-only runtime check confirmed `min_confidence=0.60` before and after the request, `automatic_threshold_change=false`, and no selected threshold.
- Summary: 3 cells assessed, 0 sufficient, 3 insufficient, 0 manual trades excluded.
- `breakout × NAS100 × M5`: 27 baseline trades; 8 wins / 19 losses; 100% R coverage; strength buckets 10/4/6/1/6; failed minimum trades, worst-case 95% win-rate margin of error, outcome diversity, and bucket coverage.
- `breakout × US30 × M5`: 26 baseline trades; 11 wins / 15 losses; 100% R coverage; strength buckets 16/4/4/1/1; failed minimum trades, worst-case 95% win-rate margin of error, and bucket coverage.
- `sma_cross × XAUUSD × M5`: 41 baseline trades; 14 wins / 27 losses; 100% R coverage; 40 broker-backed + 1 paper trade; strength buckets 5/3/3/2/28; failed minimum trades, worst-case 95% win-rate margin of error, and bucket coverage.
- The sample has grown since 2026-09-29, but no cell is eligible for threshold-comparison research. No threshold optimization or execution-setting change is authorized.

**Real sample re-check (2026-10-06)**
- Read-only runtime request again reported `status=insufficient_evidence`, `baseline_threshold=0.60`, `automatic_threshold_change=false`, and `selected_threshold=null`.
- Summary: 3 cells assessed, 0 sufficient, 3 insufficient, 0 manual trades excluded.
- `breakout × NAS100 × M5`: 31 baseline trades; 10 wins / 21 losses; 100% R coverage; 31 broker trades; signal-strength buckets 12/5/6/2/6. Outcome diversity now passes, but minimum trades, worst-case 95% win-rate margin of error (17.6013 pp), and bucket coverage still fail.
- `breakout × US30 × M5`: 31 baseline trades; 12 wins / 19 losses; 100% R coverage; 31 broker trades; signal-strength buckets 17/4/5/2/3. Outcome diversity passes, but minimum trades, worst-case 95% win-rate margin of error (17.6013 pp), and bucket coverage still fail.
- `sma_cross × XAUUSD × M5`: 48 baseline trades; 16 wins / 32 losses; 100% R coverage; 47 broker trades + 1 paper trade; signal-strength buckets 5/5/4/2/32. Outcome diversity passes, but minimum trades, worst-case 95% win-rate margin of error (14.1451 pp), and bucket coverage still fail.
- Compared with the 2026-10-04 re-check, baseline samples increased from 27→31 NAS100, 26→31 US30, and 41→48 XAUUSD. No cell is authorized to proceed to threshold-comparison research, and the global 60% execution threshold remains unchanged.

**Pending acceptance**
- [x] Re-run the sufficiency screen after more closed automatic trades accumulated; 0 of 3 cells qualified on 2026-10-04.
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
- [x] Backend restart with broker position open.
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
    - [x] Real backend-restart-with-open-position field observation passed 2026-10-06: US30 broker `57782376` and XAUUSD broker `57783302` were canonical + fully protected before restart and recovered afterward as the same `id_match` + `protected` positions on verified Demo account `44089601`. Local staging validation passed on `b2048b4`; PR #97 CI #391 passed backend + frontend before final verification.
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
- [x] Ollama/model unavailable.
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
    - [x] Real forced Ollama-down/recovery observation passed on 2026-10-04: both Windows Ollama server and app/supervisor processes were stopped, port 11434 was confirmed closed, `/api/llm_status` reported unreachable/not ready, `/api/status` reported `model_ready=false` plus actionable `model_not_ready`, engine scanning remained unchanged, deterministic NAS100/M5 breakout analysis returned through `manual_analyze`, no new order intent or broker/local exposure appeared, and restarting the original Ollama app restored `reachable=true`, `ready=true`, `model_ready=true`, and cleared the incident while config/mode remained unchanged.
    - [x] Local verification on `e14068c`: 16 focused Ollama/model/status tests passed; the timing-sensitive market-data-monitor regression passed 3/3 on retry; the full backend suite passed; working tree remained clean. PR CI #258 passed backend tests + frontend restart/Journal acceptance + production build.
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
    - [x] Real cTrader demo partial-close field observation.
      - Verified 2026-10-06 from the preserved normal TradeAgent-managed XAUUSD local `109` / broker `57783302` observation on verified Demo account `44089601`: broker quantity `0.10` was partially closed by exactly `0.05`, leaving the same canonical broker position open at `0.05` lots.
      - TradeAgent synchronized realized P&L `36.29 CHF` from authoritative broker deal history with `realized_pnl_source=ctrader_deal_partial`; the audit recorded `status=partial_close_synced`, `inserted_deals=1`, `total_closed_lots=0.05`, `required_closed_lots=0.05`, and `tracked_initial_quantity=0.1`.
      - Reconciliation remained `id_match`, and the residual broker position remained fully protected with 100% protection coverage.
      - This is the same preserved broker observation already verified for Phase 1.4 in PR #99; no second partial close or manufactured broker exposure was required.
      - Local docs-only staging validation passed on `341ab60`; PR #100 CI #400 passed the normal backend/frontend CI contract before this verification update.
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
- [x] Clock/time-zone edge cases / DST.
  - [x] UTC session filters normalize aware timestamps before comparing configured UTC hours, including Zurich spring-forward and repeated autumn-hour cases.
  - [x] Symbol cooldown timestamps are normalized to UTC before comparison.
  - [x] Market-bar freshness normalizes offset-aware timestamps to UTC and rejects materially future-dated bars instead of treating negative age as fresh.
  - [x] Persisted market-cache age compares UTC instants across offsets/DST.
  - [x] Daily trade-count and realized-P&L SQL date filters use SQLite UTC-aware `date(...)` semantics for offset-bearing timestamps.
  - [x] Broker-reconciled close timestamps preserve the source instant while daily safety accounting uses the UTC trading date.
  - [x] Event calibration selects the correct closed bar through the repeated autumn DST hour.
  - [x] The same market-bar instant represented in UTC vs Europe/Zurich remains idempotent and does not create a duplicate action.
  - Verification:
    - [x] 7 clock/time-zone/DST resilience tests passed locally on `99067bd`.
    - [x] Focused time/risk/market/event/reconciliation regression passed locally on `99067bd`.
    - [x] Full backend suite passed locally on `99067bd` with a clean working tree.
    - [x] GitHub CI #154 on `99067bd` passed backend tests + frontend build.

For every scenario verify:
- no unintended live routing,
- no duplicate demo orders,
- broker remains source of truth,
- incident is actionable,
- recovery is deterministic.

---

## 10. Phase 8 — Observability & reporting

- [x] Daily summary: trades, realized P&L, drawdown, win/loss, rejected signals by reason.
  - [x] Read-only `GET /api/reports/daily-summary?date=YYYY-MM-DD` reports a deterministic UTC-day summary without broker calls or execution mutations.
  - [x] `trades` stays aligned with the existing daily safety counter (positions opened on the UTC day), with closed-trade count exposed separately.
  - [x] Realized P&L prefers immutable cTrader broker deals and falls back to local paper-close P&L only when no broker deal exists for that local position, preventing double counting.
  - [x] Drawdown is explicitly realized-P&L sequence drawdown; no unsupported mark-to-market/equity drawdown is invented because no intraday equity curve is persisted.
  - [x] Win/loss/breakeven is computed per closed position using broker-deal totals where available.
  - [x] Rejected execution decisions are grouped deterministically by persisted rejection reason.
  - [x] UTC date handling remains correct for offset-bearing timestamps and realized entries are ordered by their actual UTC instant.
  - Verification:
    - [x] 7 daily-summary acceptance tests passed locally on `00f1eb0`.
    - [x] Focused reporting/ledger/decision/API regression passed locally on `00f1eb0`.
    - [x] Full backend suite passed locally on `00f1eb0` with a clean working tree.
    - [x] GitHub CI #157 on `00f1eb0` passed backend tests + frontend build.
- [x] Broker-vs-local reconciliation health.
  - [x] Read-only `GET /api/reports/reconciliation-health` classifies canonical TradeAgent-managed demo position health without running reconciliation or mutating broker/local state.
  - [x] Persisted positive broker position ID remains authoritative for existing local trackers; same-symbol fallback never replaces a missing canonical broker ID.
  - [x] Broker-only positions are treated as TradeAgent-managed only when the existing canonical recovery-intent identity confirms broker position ID + symbol + direction.
  - [x] Unrelated manual/non-TradeAgent broker positions are ignored rather than silently adopted.
  - [x] Broker unavailability is reported as unavailable instead of guessing missing/closed broker state.
  - [x] Healthy, degraded legacy identity, unresolved/ambiguous identity, missing-local, and missing-broker conditions are distinguished with actionable operator guidance.
  - [x] Health reporting is read-only and preserves cTrader demo-only/live-routing safeguards.
  - Verification:
    - [x] 8 reconciliation-health acceptance tests passed locally on `37a4f99`.
    - [x] Focused reconciliation/position-truth/status regression passed locally on `37a4f99`.
    - [x] Full backend suite passed locally on `37a4f99` with a clean working tree.
    - [x] GitHub CI #160 on `37a4f99` passed backend tests + frontend build.
- [x] Protection-health metric.
  - [x] Read-only `GET /api/reports/protection-health` reports broker-truth protection health for canonical TradeAgent-managed demo positions without broker/local mutations.
  - [x] Broker protection classification reuses the same truth semantics as position enrichment: both SL + TP = fully protected, exactly one = partially protected, neither = unprotected.
  - [x] Local requested SL/TP values are never treated as proof of broker protection.
  - [x] Full-protection coverage is computed only across broker-identity-assessable positions; unavailable and identity-unresolved positions remain explicit and outside the coverage denominator.
  - [x] Persisted positive broker position ID remains authoritative; canonical broker-only positions require the existing TradeAgent recovery-intent identity rule.
  - [x] Unrelated manual/non-TradeAgent broker positions are ignored rather than silently adopted.
  - [x] Broker outages/read failures report unavailable, and legacy/duplicate/mismatched identity reports identity-unresolved with actionable guidance.
  - [x] Health reporting is read-only and performs no amend, close, order, adoption, reconciliation, or live-routing action.
  - Verification:
    - [x] 9 protection-health acceptance tests passed locally on `d1e7acc`.
    - [x] Focused protection/position-truth/reconciliation/status regression passed locally on `d1e7acc`.
    - [x] Full backend suite passed locally on `d1e7acc` with a clean working tree.
    - [x] GitHub CI #163 on `d1e7acc` passed backend tests + frontend build.
- [x] Engine cycle latency.
  - [x] Read-only `GET /api/reports/engine-cycle-latency` exposes the most recently completed engine cycle duration in explicit milliseconds with start/completion timestamps.
  - [x] `run_once()` is measured with a monotonic `perf_counter()` boundary without changing scan order, trading decisions, broker execution, reconciliation, or scan-interval/sleep behavior.
  - [x] Disabled-engine, kill-switch, empty-watchlist, normal-scan, and exceptioning cycle attempts share the same latency measurement boundary.
  - [x] Runtime JSON persists only minimal restart-safe evidence: optional `last_cycle_duration_ms` and `last_cycle_completed_at`, preserving backward compatibility without a DB schema migration.
  - [x] No-cycle-yet state is explicit (`available=false`) rather than inventing a zero latency.
  - [x] Telemetry persistence is non-fatal: a latency write failure cannot alter the cycle result or replace the original exception.
  - [x] API consumers are read-only and cTrader demo-only/live-routing safeguards remain unchanged.
  - Verification:
    - [x] 9 engine-cycle latency acceptance tests passed locally on `b39a515`.
    - [x] Focused engine/runtime/status/persistence regression passed locally on `b39a515`.
    - [x] Full backend suite passed locally on `b39a515` with a clean working tree.
    - [x] GitHub CI #166 on `b39a515` passed backend tests + frontend build.
- [x] Broker/API latency.
  - [x] Read-only `GET /api/reports/broker-api-latency` reports the latest process-local broker-service and API request latency observations with explicit millisecond units and timing scope.
  - [x] Broker timing reuses existing broker-facing service calls only; no parallel/probe broker request is generated solely for metrics.
  - [x] Broker scope is explicitly `broker_service_call`, not raw network RTT; local status/symbol/readiness getters are excluded.
  - [x] Successful broker calls are classified with local connection/authorization/demo-confirmation flags only, avoiding extra reconcile/account-snapshot traffic.
  - [x] Broker exceptions or unavailable broker state report `unavailable` with no successful duration sample.
  - [x] API middleware measures existing `/api/*` request handling; completed non-5xx responses, including intentional 4xx responses, are measured while exceptions/5xx report unavailable.
  - [x] The latency-report endpoint is excluded from API timing so reading the metric cannot overwrite its own sample.
  - [x] Samples are intentionally process-local, adding no SQLite writes/lock contention to broker or API paths; restart truthfully resets to `no_observation`.
  - [x] Observability is guarded so telemetry cannot replace broker exceptions, API responses, retry semantics, execution ordering, or demo-only/live-routing safeguards.
  - Verification:
    - [x] 10 broker/API latency acceptance tests passed locally on `4ba520b`.
    - [x] Focused broker/API/status/execution regression passed locally on `4ba520b`.
    - [x] Full backend suite passed locally on `4ba520b` with a clean working tree.
    - [x] GitHub CI #169 on `4ba520b` passed backend tests + frontend build.
- [x] Error-rate / incident grouping.
  - [x] Read-only `GET /api/reports/error-incidents` aggregates existing persisted failure evidence over an explicit bounded UTC window (default 24h, allowed 1–168h).
  - [x] Persisted incidents are grouped by stable incident `code`; repeated occurrences are counted separately from distinct groups.
  - [x] Failed terminal order-intent outcomes are grouped by persisted terminal transition `reason`.
  - [x] Failed event-source runs are grouped by persisted `source`.
  - [x] Failure-evidence denominator is explicit and limited to persisted sources with real success/failure outcomes: latest terminal order-intent outcome per intent plus completed event-source runs.
  - [x] Persisted incidents are intentionally excluded from the percentage denominator because the incident log has no corresponding success rows.
  - [x] Multiple terminal history rows for one intent are deduplicated to the latest terminal transition within the reporting window.
  - [x] Group ordering is deterministic by occurrence count, last-seen UTC timestamp, then stable group id; group output is bounded (1–100).
  - [x] Reporting is read-only and does not change incident creation, retries, event ingestion, broker actions, reconciliation, execution control flow, or demo-only/live-routing safeguards.
  - Verification:
    - [x] 8 error-rate / incident-grouping acceptance tests passed locally on `32d397c`.
    - [x] Focused incident/intent/event/status/API regression passed locally on `32d397c`.
    - [x] Full backend suite passed locally on `32d397c` with a clean working tree.
    - [x] GitHub CI #172 on `32d397c` passed backend tests + frontend build.
- [x] Strategy performance by symbol/timeframe.
  - [x] Read-only `GET /api/reports/strategy-performance` reports bounded performance slices by symbol, timeframe, strategy, and account currency.
  - [x] One persisted closed local position is one trade sample; open positions are excluded.
  - [x] Broker-ledger truth takes precedence: when linked cTrader deals exist, all linked deal `net_profit` values are summed once for the trade and replace the local paper estimate.
  - [x] Partial-close/final-close deal sequences remain one trade sample and do not inflate trade counts.
  - [x] Pure-paper closed positions use their persisted local realized P&L only when no broker deals exist.
  - [x] Account currencies remain separate slices so CHF/USD history is never summed into one misleading P&L figure.
  - [x] Each slice exposes sample count, realized P&L, average realized P&L, wins/losses/breakeven, win rate, broker-backed vs pure-paper counts, broker-deal count, and first/last close timestamps.
  - [x] UTC report window is explicit and bounded (default 30 days, allowed 1–365 days), with optional symbol/timeframe/strategy filters and bounded deterministic group output.
  - [x] Reporting is read-only and does not change execution, broker calls, reconciliation, strategy lifecycle, risk controls, or demo-only/live-routing safeguards.
  - Verification:
    - [x] 8 strategy-performance acceptance tests passed locally on `146cbfb`.
    - [x] Focused strategy/reporting/broker-ledger regression passed locally on `146cbfb`.
    - [x] Full backend suite passed locally on `146cbfb` with a clean working tree.
    - [x] GitHub CI #175 on `146cbfb` passed backend tests + frontend build.
- [x] Export journal / statement comparison.
  - [x] Read-only JSON and CSV journal export surfaces expose deterministic closed-trade rows without broker calls or execution mutations.
  - [x] One persisted closed TradeAgent position maps to one export trade row with stable local identity, broker position/deal identity, linked audit IDs, UTC timestamps, currency, strategy/timeframe, prices, close reason, and explicit realized-P&L basis.
  - [x] Immutable broker deals take precedence over local paper estimates; partial-close/final-close deal rows aggregate once per local trade without double counting.
  - [x] Broker identity conflicts are explicit rather than repaired or fuzzily matched.
  - [x] External broker-statement comparison is transient/read-only and resolves trades only by exact broker position ID or persisted deal ID; symbol/time proximity is never used as identity.
  - [x] Deal-level statement rows aggregate into one trade before comparison, while duplicate/missing/conflicting statement identity fails closed as `identity_unresolved`.
  - [x] Matched rows compare symbol, optional direction, account currency, realized P&L, close time, and supplied deal-ID sets with explicit tolerances/reasons; unmatched rows remain explicit as `local_only` or `statement_only`.
  - [x] UTC window normalization is deterministic and reporting preserves cTrader demo-only/live-routing safeguards.
  - Verification:
    - [x] 9 journal export / statement comparison acceptance tests passed locally on `6e41d84`.
    - [x] Focused journal/broker-ledger/reporting regression passed locally on `6e41d84`.
    - [x] Full backend suite passed locally on `6e41d84` with a clean working tree.
    - [x] GitHub CI #178 on `6e41d84` passed backend tests + frontend build.

---

## 11. Phase 9 — Repository / documentation cleanup

- [x] Update README test count and verification date.
  - README verification metadata updated to October 3, 2026 and 428 canonical backend tests.
  - Test count derived from the actual pytest suite in successful CI output rather than estimated.
  - Local full backend regression passed on `c7621aa` with a clean working tree.
  - GitHub CI #181 passed backend tests and frontend production build on `c7621aa`.
- [x] Update screenshots after UI stabilizes.
  - Added current Trade, Build & Test, and System screenshots captured from the stabilized October 2026 UI.
  - README Product Gallery now uses `Trade_Main.png`, `Build_Test.png`, and `System.png`; April 2026 screenshots remain available only under the explicitly labeled prototype-history foldout.
  - Local README/image reference checks passed and the frontend production build completed successfully on `32489e6` with a clean working tree.
  - GitHub CI #184 passed the full backend suite plus frontend restart acceptance and production build on `32489e6`.
  - Demo-only cTrader execution and the live-account routing block remain explicitly documented.
- [x] Clearly mark/remove stale legacy architecture files where safe.
  - `IMPLEMENTATION_PLAN.md` and `STRATEGY_INTEGRATION_PLAN.md` are retained for provenance but explicitly marked historical and non-authoritative.
  - Their retired `/api/agent/execute_task`, standalone `/strategy-studio`, and old local-path assumptions are called out directly.
  - `ARCHITECTURE.md` remains the current technical source of truth, and `docs/TRADEAGENT.md` now indexes the two planning files as historical only.
  - Local warning/reference/current-route checks plus `git diff --check` passed on `0ab49c4` with a clean working tree.
  - GitHub CI #187 passed the full backend suite plus frontend restart acceptance and production build on `0ab49c4`.
- [x] Ensure operational docs use the current runtime DB path.
  - `docs/operations/local-run.md` now documents the exact resolver precedence from `backend/config.py`: explicit `TRADEAGENT_DB_PATH`, Windows `%LOCALAPPDATA%\TradeAgent\data\tradeagent.db`, `XDG_STATE_HOME`, then `~/.local/state/tradeagent/tradeagent.db`.
  - The old `backend/data/tradeagent.db` path is explicitly documented as legacy migration input rather than the active local-runtime database.
  - One-time migration semantics match `backend/storage/db.py`: copy only when the active DB is absent and the legacy DB exists; never overwrite an existing active DB.
  - Local verification resolved the actual Windows runtime path to `C:\Users\mohag\AppData\Local\TradeAgent\data\tradeagent.db`, matched code/docs, and `git diff --check` passed on `27f5982` with a clean working tree.
  - GitHub CI #190 passed the full backend suite plus frontend restart acceptance and production build on `27f5982`.
- [x] Add a single canonical validation command.
  - Added cross-platform `scripts/validate.py` as the single repository validation entry point.
  - The wrapper preserves accepted validation semantics: `python -m pytest backend/tests -q`, then `npm --prefix frontend run build`, and exits non-zero on the first failed step.
  - README and `docs/operations/local-run.md` now point to `python scripts\\validate.py` as the canonical command.
  - Local canonical validation passed on `a5bf536`: 428 backend tests passed, frontend production build passed, `VALIDATION PASSED` printed, `git diff --check` passed, and the working tree remained clean.
  - GitHub CI #193 passed the full backend suite plus frontend restart acceptance and production build on `a5bf536`.
- [x] CI for backend tests + frontend build.
  - Audited `.github/workflows/ci.yml` against the accepted repository validation contract; no workflow change was required.
  - Pull requests targeting `main` and pushes to `main` both run CI, with manual `workflow_dispatch` retained.
  - Backend CI uses Python 3.12, installs `.[broker,dev]`, and runs `python -m pytest backend/tests -q`.
  - Frontend CI uses Node 22, runs `npm ci --prefix frontend`, preserves the frontend restart acceptance test, and runs `npm --prefix frontend run build`.
  - Local audit checks confirmed trigger/job commands, one-file documentation scope, clean whitespace, and a clean working tree on `e7b6113`.
  - PR CI #196 passed the full backend suite plus frontend restart acceptance and production build on `e7b6113`; post-merge push-to-main CI #195 had already passed on `1e43f01`, confirming both required trigger paths.
- [x] Optional lint/type-check gate.
  - Evaluated the repository's existing Ruff and ESLint configurations as candidate CI gates without adding dependencies or changing lint rules.
  - Baseline evidence showed `ruff check backend` has 304 existing violations (114 reported fixable) and `npm --prefix frontend run lint` has 32 problems (28 errors, 4 warnings), so enabling either as a required gate would require broad unrelated cleanup.
  - Trial CI #199 confirmed both lint steps fail before the accepted backend test suite and frontend restart/build can run; the candidate workflow changes were therefore reverted.
  - TypeScript type checking remains enforced by the existing production build because `npm --prefix frontend run build` starts with `tsc -b`.
  - Final branch scope is documentation-only; local scope/whitespace checks passed with a clean tree on `e05ebdb`, and CI #200 passed the full backend suite plus frontend restart acceptance and production build.
  - Ruff and ESLint remain available as optional cleanup tools until their baselines are intentionally reduced in a separate scoped change.
- [x] Dependency/deprecation cleanup (`protobuf`, Starlette/httpx warning).
  - Added dev/test-only `httpx2>=2.13,<3` and `starlette>=1.7,<2` so FastAPI/Starlette `TestClient` uses the supported client backend and current AnyIO portal API without changing runtime `httpx`.
  - Kept `ctrader-open-api==0.9.2` and its declared `protobuf==3.20.1` constraint intact; instead, pytest suppresses only the exact third-party Python 3.12 `utcfromtimestamp()` deprecation warning from `google.protobuf.internal.well_known_types`.
  - Removed tracked generated `tradeagent_v2.egg-info/*` metadata and added `*.egg-info/` to `.gitignore` so editable installs no longer dirty the repository.
  - Local validation resolved Starlette 1.7.0, httpx2 2.13.1, AnyIO 4.15.1, protobuf 3.20.1, and ctrader-open-api 0.9.2; `TestClient` import passed under `-W error`, canonical backend/frontend validation passed without the targeted warning summary, regenerated egg-info left Git clean, and `git diff --check` passed on `a08891f`.
  - CI #203 passed backend + frontend on the dependency-cleanup implementation head, and CI #207 passed the final branch shape including packaging-metadata hygiene.

---

## 12. Acceptance definition

TradeAgent is considered **demo-runtime validated** when all of the following are true:

- [x] Full backend test suite passes.
  - Verified through the accepted canonical validation path and current GitHub CI: local canonical validation passed the full backend suite during Phase 9 dependency cleanup, post-merge CI #209 passed the backend test job on `9343eea`, and PR CI #210 passed the backend test job on `b03601d`.
- [x] Frontend production build passes.
  - Verified through the accepted canonical validation path and current GitHub CI: local canonical validation passed the frontend TypeScript/Vite production build during Phase 9 dependency cleanup, post-merge CI #212 passed the frontend restart acceptance test plus production build on `6faf1d5`, and PR CI #213 passed the same frontend job on `c353daa`.
- [x] Repeated demo trades use correct quantities.
  - Verified from preserved real cTrader demo-field evidence plus current implementation continuity: the original canonical roadmap baseline (`e41ccc`) records real demo orders for XAUUSD, US30, and NAS100, records that fresh demo trades used the configured 0.10 lot size, and reconciles realized P&L for those three symbols against cTrader.
  - Current demo execution still converts requested lots through broker contract metadata before submission; post-merge CI #215 passed the current backend suite on `6ccc18a`, local roadmap/scope/whitespace validation passed on `300c154`, and PR CI #216 passed backend + frontend on that staged-evidence head.
- [ ] Every broker-managed position is either verified protected or fails closed.
  - Acceptance audit pending real post-hardening field evidence: Phase 1.1 implements and regression-tests immediate/delayed protection success, repeated protection failure → verified broker fail-safe close, and broker-close failure → critical incident with local tracking retained; protection-health reporting also distinguishes protected/partial/unprotected/unavailable states.
  - The original canonical roadmap baseline (`e41ccc`) records that broker SL/TP synchronization worked but also records at least one real `ctrader_demo_order_unprotected` event. No preserved post-hardening real cTrader observation currently proves that an unprotected broker position was subsequently verified protected or actually fail-safe closed, so this checkbox remains pending until the next normal TradeAgent-managed demo position provides safe field evidence. No new trade should be opened solely for this acceptance check.
- [x] Broker position ID is the canonical identity.
  - Verified as a current implementation invariant: Phase 1.2 / PR #11 made persisted `broker_position_id` authoritative across execution, reconciliation, and recovery; symbol+direction fallback is restricted to legacy rows without a broker ID and only when unambiguous, while mismatched or ambiguous identities fail safely instead of switching positions.
  - Original PR #11 CI #4 passed backend + frontend, later DB restart/recovery acceptance (PR #53) preserved canonical identity across durable recovery without symbol-only adoption, post-merge CI #221 passed the current suite on `a1cd3e2`, local roadmap/scope/whitespace validation passed on `41ff991`, and PR CI #222 passed backend + frontend on that staged-evidence head. The separate Phase 2.3 real broker-truth UI observation remains pending but does not change the canonical matching rule.
- [x] Closed-trade price/time/net P&L reconcile to cTrader.
  - Verified from preserved real cTrader demo-close evidence plus current broker-ledger continuity: the original canonical roadmap baseline (`e41ccc`) records real closes whose journal P&L matched closing-deal truth (NAS100 -2.71, US30 -0.73, XAUUSD -9.24; total -12.68 CHF) and records broker closing-deal execution price/time persisted from cTrader.
  - Current final-close reconciliation uses cTrader `exit_price`, `closed_at`, and `net_profit` with `realized_pnl_source=ctrader_deal`; the immutable broker-deal ledger preserves gross profit, swap, commission, conversion fee, and net profit, while missing broker-deal truth does not fabricate broker P&L. PR #63 adds exact-identity read-only statement comparison with broker-deal precedence. Post-merge CI #224 passed the current suite on `7f79cbe`, local roadmap/scope/whitespace validation passed on `17af65a`, and PR CI #225 passed backend + frontend on that staged-evidence head. The separate real cTrader partial-close field observation remains pending under Phase 1.4.
- [x] Broker account currency/equity drive demo monetary risk controls.
  - Verified from Phase 1.3 real-field evidence plus current implementation continuity: the cTrader demo account was verified as CHF with a real broker balance/equity snapshot, while `resolve_monetary_basis()` keeps configurable `paper_config` equity exclusive to pure-paper mode and requires verified broker currency plus positive equity in demo mode, failing closed otherwise.
  - Daily-loss budget, risk-per-trade amount, and demo auto-sizing all use the broker monetary basis; the current workflow verifies CHF 20,000 at 0.5% risk produces CHF 100 risk and the expected account-currency-valued quantity, including cross-currency conversion. PR #18 CI #19 passed the original hardening, post-merge CI #227 passed on `2160709`, local roadmap/scope/whitespace validation passed on `813bcea`, and PR CI #228 passed backend + frontend on that staged-evidence head.
- [x] Restart/recovery with open positions is verified.
  - PR #53 and the current resilience suite verify durable SQLite reopen, canonical tracker/intent recovery, broker-confirmed handoff recovery exactly once, protection resumption on the persisted broker position ID, idempotent broker ledger state, and persistent duplicate-order blocking across restart; final PR #53 CI #149 and post-merge CI #230 passed backend + frontend.
  - Real cTrader Demo field verification completed 2026-10-06: the backend was restarted while US30 broker `57782376` and XAUUSD broker `57783302` were both open, canonical, and fully protected. After startup recovery, Demo account `44089601` was authorized/verified/execution-ready, the engine was scanning, and both same broker IDs remained `id_match` + `protected` with broker SL/TP truth preserved. PR #97 staging head `b2048b4` passed local docs-only validation and CI #391 before this acceptance checkbox was closed.
- [x] Safety controls are acceptance-tested.
  - Verified from Phase 3.2 / PR #25 plus current failure-mode regression coverage: kill switch, minimum confidence, cooldown, maximum daily trades, maximum open positions, maximum positions per symbol, daily loss limit, required protective stops, session filter, and per-symbol automatic-trading permission are directly acceptance-tested; PR #25 CI #44/#45 passed focused safety tests plus full backend/frontend validation.
  - Current demo-account safety tests reject live-account authorization/routing and require verified demo readiness, while Phase 7 resilience acceptance covers ambiguous order acknowledgements, rejected protection amendments, rejected/ambiguous broker closes, stale/malformed data, disconnects, SQLite persistence faults, and duplicate-submission suppression. Post-merge CI #233 passed on `fad6185`, local roadmap/scope/whitespace validation passed on `4f22aa7`, and PR CI #234 passed backend + frontend on that staged-evidence head. Separately pending real-field observations remain tracked under their own acceptance/phase items and are not silently closed by this acceptance result.
- [x] Trade Journal explains decisions clearly.
  - Verified across Phase 2.1/2.4 and Phase 8: human-readable rejection summaries, linked intent type/status, deduplicated risk/quantity/sizing reasons, broker execution/protection/close details, raw audit drill-down, deterministic reporting/export, and exact-identity statement comparison give a reviewer the decision and execution context without reconstructing raw logs.
  - PR #81 closes the remaining explainability gap by promoting broker-deal vs partial-broker-deal vs paper-estimate provenance into an explicit `Realized result` drill-down block. Local validation on `7457d87` passed the new Journal acceptance test (3/3), existing frontend restart acceptance (3/3), production build, scope/whitespace checks, and a clean working tree; PR CI #237 passed the full backend suite plus both frontend acceptance tests and production build. Post-merge baseline CI #236 had already passed on `4c85653`.
- [x] Build & Test is methodologically validated.
  - Verified from Phase 4.3–4.6 methodology and current regression evidence: PR #30 established one shared draft/saved accounting path with trade-derived win rate and correct flip/flat/final-trade handling; PR #31 added next-bar-open/no-lookahead execution, explicit costs/position sizing, UTC/gap handling, linear CFD returns, and one mark-to-market equity path; PR #32 enforces chronological 70/30 development/OOS separation, independent regime evidence, expanding-window walk-forward validation, development-only parameter selection with one untouched-holdout evaluation, and minimum bar/trade gates; PR #33 enforces sequential lifecycle promotion, source-hash/version isolation, operator/reason audit, and generated-strategy execution gating.
  - Original Phase 4 CI remained green through PR #30 CI #60/#61, PR #31 CI #65/#66/#67, PR #32 CI #71/#72, and PR #33 CI #76/#77. Current local validation on `044805f` passed the focused Phase 4 methodology suite, canonical backend suite + frontend production build, roadmap-only scope, whitespace, and clean-tree checks; PR CI #240 passed backend plus frontend restart/Journal acceptance and production build. Post-merge baseline CI #239 had already passed on `9409eb6`. Phase 5.1 real connected-feed evidence and Phase 5.3 threshold comparison remain pending separately; the global 60% threshold remains unchanged.
- [x] Generated strategy code remains isolated from autonomous execution until explicit lifecycle approval.
  - Verified from Phase 4.2 / PR #29 and Phase 4.6 / PR #33 plus current implementation continuity: generated Strategy Studio source remains a research/backtest artifact inside the isolated sandbox; forbidden imports/file/network/system access are rejected; timeout/bar/memory/output limits apply; `load_generated_strategies()` only validates saved research files and returns `0`; the legacy private `_register_generated_module()` has no repository callers; generated names do not enter either runtime registry.
  - Lifecycle governance remains sequential `draft → backtested → validated → paper → eligible`, source edits create a fresh `draft` version/hash without inheriting old evidence, paper evidence is exact-source-version bound, and the central `risk_engine` calls `paper_execution_gate()` before execution so governed strategies are rejected before `paper` and only the current source version can execute at `paper` / `eligible`. Original PR #29 CI #57/#58 and PR #33 CI #76/#77 passed; post-merge CI #242 passed on `dcf3a9f`; local focused sandbox/lifecycle tests + canonical validation + scope/whitespace/clean-tree checks passed on `354744b`; PR CI #243 passed backend + frontend on that staged-evidence head.
- [x] Failure scenarios do not create duplicate/unprotected/untracked exposure.
  - Verified from Phase 1 safety invariants plus Phase 7 resilience and current regression coverage: ambiguous post-submit opens are non-retryable and block competing submissions until canonical broker truth resolves; protection rejection either verifies/repairs protection, closes broker-first, or retains canonical local tracking with an actionable incident; rejected/ambiguous closes never fabricate local closure and block duplicate close submissions; SQLite interruptions preserve durable submission reservations/tracking handoffs and suppress resubmission; restart recovery preserves canonical identity, one tracker/intent, broker-deal idempotence, protection resumption, and duplicate-order blocking; protected-position disconnect/restart paths preserve tracking and suppress unsafe mutation/adoption while broker truth is unavailable.
  - Original resilience evidence remains green through PR #41/#42, PR #48, PR #49, PR #50, PR #52, and PR #53. Local current validation on `2912ae2` passed the focused failure-containment suite, canonical backend suite + frontend production build, roadmap-only scope, whitespace, and clean-tree checks; PR CI #246 passed the full backend suite plus frontend restart/Journal acceptance and production build. Post-merge baseline CI #245 had already passed on `0adb4a8`. Separately pending real-field observations remain tracked under their own acceptance/phase items and are not silently closed here.
- [x] Documentation matches current implementation.
  - Verified after the final current-implementation documentation audit: README and architecture text describe local paper execution plus explicitly enabled cTrader demo routing while preserving the live-account block; generated Strategy Studio source remains outside the trusted runtime registry; the active `/build-test/results` route, CI frontend gates, runtime SQLite resolver/legacy migration semantics, current screenshots/assets, and historical-plan warnings are documented consistently.
  - Original documentation-consistency validation on `2ea33d7` passed the six-file documentation-only scope checks, current-route/CI/demo-live/generated-strategy wording checks, canonical backend suite + frontend production build, whitespace/clean-tree checks, and the corrected read-only SQLite documentation lookup confirming `TRADEAGENT_DB_PATH`, `%LOCALAPPDATA%\TradeAgent\data\tradeagent.db`, and the legacy migration-input wording; PR CI #249 passed backend + frontend acceptance/build.
  - Re-confirmed at the deployment-ready baseline on 2026-10-04 through merged `main` `00d2666` / PR #88: subsequent Phase 5.1 connected-feed evidence, Phase 5.3 sufficiency re-check, and Phase 7 Ollama outage/recovery verification are reflected in the roadmap; remaining broker-position/sample-dependent checks are explicitly classified as post-deployment validation rather than blockers; at that baseline cTrader was still demo-only and live-account routing remained blocked, while the global 60% threshold remained unchanged. Push-to-main CI #260 passed backend tests plus frontend restart/Journal acceptance and production build.
  - Re-confirmed after Phase 10.5 / PR #94: current README, architecture, environment example, operations guidance, generated-strategy security wording, and architecture diagram now match the superseding account-neutral cTrader architecture. Demo remains recommended for development/testing, while a selected authenticated Live account can submit real-money orders only through the same guarded execution/risk/protection/reconciliation path. Local documentation consistency, full backend regression, frontend build, and PR CI #323 all passed on `257e90b`.

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
| 2026-10-02 | Resilience: clock/time-zone/DST | Normalize risk/market/daily-accounting time boundaries to UTC; reject future-clock market data; verify Zurich DST, event timing, reconciliation timestamps, and same-instant replay idempotence | ✅ 7 scenario tests + focused regression + full backend suite + CI #154 passed on `99067bd` | PR #55 / `99067bd` | Phase 7 automated resilience scenarios complete; proceed to Phase 8 daily observability summary while pending real cTrader field observations remain open |
| 2026-10-03 | Observability: deterministic daily summary | Add read-only UTC-day reporting for trades, broker-preferred realized P&L, realized drawdown, win/loss/breakeven, win rate, and rejected signals by persisted reason | ✅ 7 acceptance tests + focused reporting/ledger/API regression + full backend suite + CI #157 passed on `00f1eb0` | PR #56 / `00f1eb0` | Proceed to Phase 8 broker-vs-local reconciliation health |
| 2026-10-03 | Observability: broker-vs-local reconciliation health | Add read-only canonical identity health for TradeAgent-managed demo positions, distinguishing healthy/degraded/unavailable/unresolved/missing-local/missing-broker while ignoring unrelated manual broker positions | ✅ 8 acceptance tests + focused reconciliation/position-truth/status regression + full backend suite + CI #160 passed on `37a4f99` | PR #57 / `37a4f99` | Proceed to Phase 8 protection-health metric |
| 2026-10-03 | Observability: protection-health metric | Add read-only broker-truth SL/TP health and full-protection coverage for canonical TradeAgent-managed demo positions; keep unavailable/unresolved identity explicit and exclude it from assessable coverage | ✅ 9 acceptance tests + focused protection/broker-truth regression + full backend suite + CI #163 passed on `d1e7acc` | PR #58 / `d1e7acc` | Proceed to Phase 8 engine cycle latency |
| 2026-10-03 | Observability: engine cycle latency | Measure `run_once()` with a monotonic clock, persist minimal restart-safe duration/completion evidence, and expose a read-only last-cycle latency report with explicit no-cycle-yet state | ✅ 9 acceptance tests + focused engine/runtime/status/persistence regression + full backend suite + CI #166 passed on `b39a515` | PR #59 / `b39a515` | Proceed to Phase 8 broker/API latency |
| 2026-10-03 | Observability: broker/API latency | Measure existing broker-service and FastAPI request boundaries without extra broker probes, classify failed/unavailable calls separately, and expose process-local read-only latency evidence | ✅ 10 acceptance tests + focused broker/API/status/execution regression + full backend suite + CI #169 passed on `4ba520b` | PR #60 / `4ba520b` | Proceed to Phase 8 error-rate / incident grouping |
| 2026-10-03 | Observability: error-rate / incident grouping | Group persisted incidents, failed terminal intent outcomes, and event-source failures by stable identity; expose an explicit persisted-outcome denominator and bounded UTC reporting window without inventing synthetic events | ✅ 8 acceptance tests + focused incident/intent/event/status/API regression + full backend suite + CI #172 passed on `32d397c` | PR #61 / `32d397c` | Proceed to Phase 8 strategy performance by symbol/timeframe |
| 2026-10-03 | Observability: strategy performance | Aggregate persisted closed trades by symbol/timeframe/strategy/currency with broker-deal precedence, one-position/one-trade semantics, explicit UTC windows, and partial-close-safe accounting | ✅ 8 acceptance tests + focused strategy/reporting/broker-ledger regression + full backend suite + CI #175 passed on `146cbfb` | PR #62 / `146cbfb` | Proceed to Phase 8 export journal / statement comparison |
| 2026-10-03 | Observability: journal export / statement comparison | Add deterministic read-only JSON/CSV journal export and exact-identity external statement comparison with broker-deal precedence, partial-close aggregation, explicit mismatch/unmatched reasons, and no persistence/broker mutation | ✅ 9 acceptance tests + focused journal/ledger/reporting regression + full backend suite + CI #178 passed on `6e41d84` | PR #63 / `6e41d84` | Proceed to Phase 9 README verification metadata |
| 2026-10-03 | Repository/docs: README verification metadata | Refresh stale README verification date and backend-test count from the canonical suite without mixing other cleanup work | ✅ README shows October 3, 2026 + 428 tests; local full backend regression passed with clean tree; CI #181 passed backend + frontend on `c7621aa` | PR #64 / `c7621aa` | Proceed to Phase 9 screenshot update/review |
| 2026-10-03 | Repository/docs: current UI screenshots | Refresh the README Product Gallery with current Trade, Build & Test, and System captures while retaining older screenshots only as labeled prototype history | ✅ Local asset/reference checks + frontend production build passed with clean tree; CI #184 passed backend + frontend on `32489e6` | PR #65 / `32489e6` | Proceed to Phase 9 legacy architecture file review |
| 2026-10-03 | Repository/docs: legacy architecture plans | Preserve historical planning provenance while preventing stale routes/API assumptions from being mistaken for current architecture | ✅ Historical warnings + current-route/API reference checks + whitespace check passed locally; CI #187 passed backend + frontend on `0ab49c4` | PR #66 / `0ab49c4` | Proceed to Phase 9 runtime DB path documentation |
| 2026-10-03 | Repository/docs: runtime SQLite path | Document the actual runtime DB resolver precedence and distinguish the repository-local legacy migration source from active OS-local state | ✅ Local code/doc/path resolution + whitespace checks passed; actual Windows path resolved to `%LOCALAPPDATA%\TradeAgent\data\tradeagent.db`; CI #190 passed backend + frontend on `27f5982` | PR #67 / `27f5982` | Proceed to Phase 9 canonical validation command |
| 2026-10-03 | Repository/docs: canonical validation command | Add one cross-platform validation entry point for the accepted backend suite and frontend production build, then point current docs to it | ✅ Local canonical validator passed 428 backend tests + frontend production build with clean tree; CI #193 passed backend + frontend on `a5bf536` | PR #68 / `a5bf536` | Proceed to Phase 9 CI audit/acceptance |
| 2026-10-03 | Repository/docs: CI audit / acceptance | Audit the existing GitHub Actions workflow against the accepted backend-suite/frontend-build contract and document the verified behavior without unnecessary YAML changes | ✅ Local trigger/job/scope/whitespace checks passed; PR CI #196 passed backend + frontend on `e7b6113`; push-to-main CI #195 passed on `1e43f01` | PR #69 / `e7b6113` | Proceed to Phase 9 optional lint/type-check gate |
| 2026-10-03 | Repository/docs: optional lint/type-check gate | Evaluate existing Ruff/ESLint checks without widening scope; keep TypeScript checking through the production build and document why lint remains optional | ✅ Trial found Ruff 304 violations + ESLint 32 problems; candidate CI failed as expected in #199 and was reverted; cleaned docs-only head `e05ebdb` passed local scope/whitespace checks and CI #200 backend + frontend | PR #70 / `e05ebdb` | Proceed to Phase 9 dependency/deprecation cleanup |
| 2026-10-03 | Repository/docs: dependency/deprecation cleanup | Move TestClient validation to supported Starlette/httpx2 dev dependencies, preserve cTrader's protobuf pin with a precise third-party warning filter, and stop tracking generated egg-info metadata | ✅ Local warning-free TestClient + canonical backend/frontend validation + clean editable-install Git state; CI #203 and final-shape CI #207 passed | PR #71 / `a08891f` | Proceed to acceptance-definition closure: full backend suite |
| 2026-10-03 | Acceptance definition: full backend suite | Close the first acceptance checkbox using the accepted canonical validation path plus merged-main and PR CI evidence | ✅ Local canonical backend suite passed; CI #209 passed backend on merged `main` `9343eea`; PR CI #210 passed backend on `b03601d` | PR #72 / `b03601d` | Proceed to acceptance-definition frontend production build |
| 2026-10-03 | Acceptance definition: frontend production build | Close the second acceptance checkbox using the accepted canonical validation path plus merged-main and PR CI evidence | ✅ Local canonical frontend production build passed; CI #212 passed restart acceptance + build on merged `main` `6faf1d5`; PR CI #213 passed the same frontend job on `c353daa` | PR #73 / `c353daa` | Proceed to acceptance-definition repeated demo trade quantities |
| 2026-10-03 | Acceptance definition: repeated demo trade quantities | Close the third acceptance checkbox from preserved real cTrader demo observations plus current lot-to-protocol-volume implementation continuity | ✅ Original verified baseline records XAUUSD/US30/NAS100 real demo orders at configured 0.10 lot; current backend regression remained green in CI #215/#216; local scope/whitespace checks passed on `300c154` | PR #74 / `300c154` | Proceed to acceptance-definition protection/fail-closed evidence |
| 2026-10-03 | Acceptance definition: protection / fail-closed audit | Evaluate whether current automated hardening plus preserved cTrader field evidence is sufficient to close the protection acceptance checkbox without creating a new trade | 🟨 Automated protection/fail-safe coverage is green and CI #218/#219 passed, but preserved post-hardening real-field evidence is insufficient; checkbox intentionally remains pending | PR #75 / `558322e` | Proceed to acceptance-definition broker position identity; revisit protection on the next normal TradeAgent-managed demo position |
| 2026-10-03 | Acceptance definition: broker position identity | Close the fifth acceptance checkbox from the Phase 1.2 canonical-ID hardening, current matching tests, and restart/recovery identity guarantees | ✅ Persisted broker ID is authoritative; legacy symbol+direction fallback is unique-only; ambiguity/mismatch fails safely; original PR #11 CI #4, current CI #221/#222, and local scope/whitespace checks on `41ff991` passed | PR #76 / `41ff991` | Proceed to acceptance-definition closed-trade cTrader reconciliation |
| 2026-10-03 | Acceptance definition: closed-trade cTrader reconciliation | Close the sixth acceptance checkbox from preserved real cTrader close evidence plus current broker-ledger and exact-identity statement-comparison behavior | ✅ Real NAS100/US30/XAUUSD closes reconciled to cTrader total -12.68 CHF with broker close price/time; current broker-deal truth remained green in CI #224/#225; local scope/whitespace checks passed on `17af65a` | PR #77 / `17af65a` | Proceed to acceptance-definition broker monetary-basis controls; keep real partial-close field observation pending |
| 2026-10-03 | Acceptance definition: broker monetary basis | Close the seventh acceptance checkbox from the verified real CHF cTrader account snapshot plus current fail-closed monetary-basis, daily-loss, risk-sizing, and cross-currency behavior | ✅ Demo mode uses verified broker currency/equity while paper mode keeps virtual equity separate; PR #18 CI #19, current CI #227/#228, and local scope/whitespace checks on `813bcea` passed | PR #78 / `813bcea` | Proceed to acceptance-definition restart/recovery with open positions |
| 2026-10-03 | Acceptance definition: restart/recovery with open positions audit | Evaluate whether durable restart/recovery coverage is sufficient to close the open-position restart acceptance checkbox without manufacturing a broker position | 🟨 PR #53 durable restart/recovery and CI #230/#231 are green, but the required real cTrader restart with an actually open TradeAgent-managed demo position has not occurred; checkbox intentionally remains pending | PR #79 / `26aa162` | Proceed to acceptance-definition safety controls; revisit restart/recovery on the next normal open demo position |
| 2026-10-03 | Acceptance definition: safety controls | Close the ninth acceptance checkbox from Phase 3.2 direct safety-control acceptance tests plus current demo/live routing and Phase 7 failure-mode safeguards | ✅ PR #25 directly acceptance-tested operator safety gates; live routing remains blocked; ambiguous/rejected broker paths fail safely; post-merge CI #233, PR CI #234, and local scope/whitespace checks on `4f22aa7` passed | PR #80 / `4f22aa7` | Proceed to acceptance-definition Trade Journal decision explainability |
| 2026-10-03 | Acceptance definition: Trade Journal decision explainability | Close the tenth acceptance checkbox after auditing reviewer-facing decision/risk/broker/result context and promoting realized-P&L provenance into the normal Journal drill-down | ✅ Local Journal explainability 3/3 + restart acceptance 3/3 + frontend production build passed on `7457d87`; PR CI #237 passed full backend + both frontend acceptance tests + build; no execution/broker/storage behavior changed | PR #81 / `7457d87` | Proceed to acceptance-definition Build & Test methodological validation |
| 2026-10-04 | Acceptance definition: Build & Test methodology | Close the eleventh acceptance checkbox from Phase 4.3–4.6 accounting, realism, leakage-safe validation, and lifecycle-governance evidence | ✅ Focused current Phase 4 methodology suite + canonical validation passed locally on `044805f`; PR CI #240 passed backend + frontend acceptance/build; Phase 5 real-feed/threshold evidence remains separate and 60% stays unchanged | PR #82 / `044805f` | Proceed to acceptance-definition generated-strategy isolation |
| 2026-10-04 | Acceptance definition: generated-strategy isolation | Close the twelfth acceptance checkbox from sandbox isolation, trusted-registry exclusion, exact-source lifecycle versioning, and central execution gating | ✅ Local focused sandbox/lifecycle suite + canonical validation passed on `354744b`; corrected safety-boundary text check, scope/whitespace, and clean tree passed; PR CI #243 passed backend + frontend | PR #83 / `354744b` | Proceed to acceptance-definition failure-scenario exposure containment |
| 2026-10-04 | Acceptance definition: failure-scenario exposure containment | Close the thirteenth acceptance checkbox from ambiguous-open, protection, close, persistence, disconnect, and restart containment invariants | ✅ Focused current containment suite + canonical validation passed locally on `2912ae2`; PR CI #246 passed backend + frontend acceptance/build; separate real-field observations remain pending without weakening automated containment | PR #84 / `2912ae2` | Proceed to acceptance-definition documentation/current-implementation consistency |
| 2026-10-04 | Acceptance definition: documentation consistency | Close the fourteenth acceptance checkbox after auditing current product/runtime/CI/SQLite/generated-strategy documentation against the implementation | ✅ Six-file documentation scope and current wording checks + canonical validation + whitespace/clean-tree checks passed locally on `2ea33d7`; corrected SQLite lookup confirmed active path and legacy-migration wording; PR CI #249 passed backend + frontend restart/Journal acceptance + production build | PR #85 / `2ea33d7` | Acceptance-definition sweep complete; preserve remaining real-field/research checks as pending until qualifying evidence is safely available |
| 2026-10-04 | Phase 5.1 connected historical-feed evidence | Add a read-only verifier through the running backend and execute the trusted-strategy × enabled-target matrix against the positively confirmed cTrader demo session | ✅ 14 focused tests + full backend suite passed locally on `7fb61ca`; real NAS100/US30/XAUUSD M5 matrix completed 9/9 with 1,379 bars per target; PR CI #252 passed backend + frontend; all 9 full-period net returns were negative, so no strategy/threshold change is authorized | PR #86 / `7fb61ca` | Proceed to Phase 5.3 read-only sample-sufficiency re-check; keep global 60% unchanged |
| 2026-10-04 | Phase 5.3 threshold sample-sufficiency re-check | Re-run the existing read-only per-cell screening against current persisted automatic-trade evidence without selecting or mutating a threshold | 🟨 7 focused tests + full backend suite passed locally on `601f776`; runtime threshold stayed 0.60; 0/3 cells qualified (27 NAS100 breakout, 26 US30 breakout, 41 XAUUSD SMA baseline trades); PR CI #255 passed | PR #87 / `601f776` | Keep Phase 5.3 pending until more closed automatic trades accumulate; proceed to the next safely actionable field check |
| 2026-10-04 | Phase 7 Ollama/model unavailable field verification | Force a real Windows Ollama outage by stopping both the server and app supervisor; verify truthful degraded state, deterministic-trading independence, zero exposure side effects, and full recovery | ✅ Real outage/recovery passed; 16 focused tests passed; timing-sensitive monitor test passed 3/3 on retry; full backend suite passed locally on `e14068c`; PR CI #258 passed backend + frontend | PR #88 / `e14068c` | Deployment-ready baseline reached; remaining broker-position/sample-dependent checks move to post-deployment evidence and must not block normal demo use |
| 2026-10-04 | Final deployment documentation consistency | Re-confirm the documentation-consistency acceptance evidence against the merged deployment-ready baseline after PRs #86–#88 | ✅ Local docs-only validation passed on `15b5706`: exactly one changed file (`docs/TRADEAGENT_ROADMAP.md`), `git diff --check` clean, all final acceptance wording present, and working tree clean; PR CI #261 passed backend tests + frontend restart/Journal acceptance + production build | PR #89 / `15b5706` | Baseline documentation closure verified; merge and return local `main` to a clean synchronized state |
| 2026-10-04 | Phase 10.1 cTrader account discovery foundation | Retain and expose every authorized cTrader account with broker-reported Demo/Live type and selected-account identity, without yet changing account switching or order routing | ✅ Local focused suite 20/20 + full backend regression + frontend build passed on `419ab23`; final local head `46fdf75` clean under `git status --short` and `git diff --check`; PR CI #264/#265 passed | PR #90 / `46fdf75` | Proceed to Phase 10.2 dashboard account selector and persisted active account |
| 2026-10-04 | Phase 10.2 cTrader dashboard account selector | Persist the operator-selected authorized cTrader account and expose a cTrader-style broker/Demo-Live/login selector while keeping active transport truth separate | ✅ Local focused suite 28/28 + full backend regression + frontend build passed on `c3b79ff`; local tree/diff clean; PR CI #274 passed | PR #91 / `c3b79ff` | Proceed to Phase 10.3 account-aware transport switching |
| 2026-10-04 | Phase 10.3 cTrader account-aware transport switching | Make the persisted account selection drive safe same-host re-authentication or Demo↔Live client replacement without restarting the Twisted reactor | ✅ Local focused suite 59/59 + full backend regression + frontend build passed on `451c5d3`; local tree/diff clean; PR CI #287 passed | PR #92 / `451c5d3` | Proceed to Phase 10.4 account-neutral execution path |
| 2026-10-04 | Phase 10.4 cTrader account-neutral execution | Generalize the guarded cTrader execution path from Demo-specific semantics to the explicitly selected authenticated Demo or Live account without weakening risk/protection/reconciliation safeguards | ✅ Initial local regression exposed stale demo-only fixtures and a real monetary-snapshot Demo guard; repaired final head `a6941c0` passed focused regression, full backend suite, frontend build, clean tree/diff checks, and PR CI #320 | PR #93 / `a6941c0` | Proceed to Phase 10.5 documentation and operator/risk disclaimer cleanup |
| 2026-10-04 | Phase 10.5 documentation and operator disclaimer | Align README, architecture, environment bootstrap guidance, operations, generated-strategy security, and architecture diagram with the account-neutral Demo/Live execution model and explicit Live-risk warning | ✅ Local documentation consistency checks + full backend regression + frontend production build passed on `257e90b`; working tree/diff clean; PR CI #323 passed | PR #94 / `257e90b` | Phase 10 implementation complete; require final docs-only CI, merge, sync local main, then gather remaining broker-position/sample evidence naturally |
| 2026-10-06 | Phase 10.6 Trade-dashboard cTrader selector + Live arming + transport isolation | Expose authorized Demo/Live account selection on Trade, preserve selected-vs-active truth, add runtime-only Live arming, serialize cross-host auth/directory refresh, redact auth logs, and isolate OpenApiPy TCP queues per connection | ✅ Focused cTrader suite + full backend regression passed locally on `91cdf9e`; real Demo `44089601` → Live `48922568` → Demo smoke passed with Live disarmed and 7-account Demo-directory refresh clean; PR CI #388 passed backend + frontend | PR #96 / `91cdf9e` | Merge after final docs-only CI, sync local main, then resume only the remaining evidence-dependent roadmap checks without manufacturing trades or changing the 60% threshold |
| 2026-10-06 | Real backend restart with open broker positions | Restart the backend through the normal launcher while two canonical, fully protected Demo positions are open, then verify the same broker IDs and protection truth after recovery | ✅ US30 `57782376` + XAUUSD `57783302` recovered as `id_match` + `protected`; Demo `44089601` authorized/verified/execution-ready; local docs-only staging validation passed on `b2048b4`; PR CI #391 passed backend + frontend | PR #97 | Final docs-only validation/CI, merge, sync local main, then close the already-captured Phase 2.3 real broker-truth observation as the next smallest evidence item |
| 2026-10-06 | Phase 2.3 real broker-truth field observation | Verify the Position-panel broker-truth fields against a naturally opened TradeAgent-managed cTrader Demo position without mutating the trade | ✅ XAUUSD local `107` / broker `57779123`: quantity `0.1`, broker entry `4131.83`, SL `4135.97`, TP `4123.1`, CHF basis, `protected`, `id_match`, timestamped broker snapshot; 1/1 protection-health coverage; local staging validation passed on `174d26e`; PR CI #394 passed backend + frontend | PR #98 | Final docs-only validation/CI, merge, sync local main, then resume only the remaining naturally evidence-dependent partial-close / protected-disconnect checks |
| 2026-10-06 | Phase 1.4 real cTrader Demo partial-close field verification | Partially close one normal TradeAgent-managed broker position and verify residual quantity, authoritative broker-deal P&L, canonical identity, and continued broker protection | ✅ XAUUSD local `109` / broker `57783302`: `0.10 → 0.05` lots, `36.29 CHF` realized P&L from `ctrader_deal_partial`, one new broker deal, canonical `id_match`, residual 100% protected; local docs-only staging validation passed on `9b127bf`; PR CI #397 passed backend + frontend | PR #99 | Final docs-only validation/CI, merge, sync local main, then reuse this preserved evidence for the separate Phase 7 partial-close field-observation item |
| 2026-10-06 | Phase 7 real cTrader Demo partial-close field observation | Close the remaining Phase 7 field-evidence checkbox by reusing the already-preserved normal partial-close observation without manufacturing a second trade | ✅ XAUUSD local `109` / broker `57783302`: `0.10 → 0.05` lots, `36.29 CHF` from `ctrader_deal_partial`, `partial_close_synced`, exact authoritative `0.05` closed/required lots, canonical `id_match`, residual 100% protected; local staging validation passed on `341ab60`; PR CI #400 passed backend + frontend | PR #100 | Final docs-only validation/CI, merge, sync local main, then perform the still-pending Phase 7 protected-position disconnect/recovery field observation only when a safe qualifying TradeAgent-managed Demo position exists |
| 2026-10-06 | Phase 2.5 cTrader-style lot-size entry | Replace misleading four-decimal Trade setup lot entry with two-decimal cTrader-style input formatting while preserving backend broker-contract validation | ✅ Focused frontend lot-size test 3/3 + production build (702 modules) + clean diff/worktree + browser smoke showing correct two-decimal UI; PR CI #404 passed on implementation head `02997a0` | PR #101 / `02997a0` | Final docs-only validation/CI, merge, sync local main, then resume the pending Phase 7 protected-position disconnect field observation only when a natural qualifying Demo position exists |
| 2026-10-06 | Phase 5.3 threshold sample-sufficiency re-check | Re-run the existing read-only sufficiency screen after additional naturally closed automatic trades while Phase 7 field evidence remained blocked by zero open managed positions | 🟨 0/3 cells sufficient; NAS100 31, US30 31, XAUUSD 48 baseline trades; all cells have 100% R coverage and sufficient win/loss diversity, but minimum sample, 95% margin-of-error, and bucket-coverage gates still fail; threshold stayed 0.60 and no threshold was selected | Evidence-only PR | Keep 60% unchanged; resume Phase 7 disconnect/recovery only when a natural protected Demo position exists |

---

## 14. Phase 10 — cTrader multi-account architecture

### 10.1 Authorized account discovery foundation
**Priority:** P0
**Status:** ✅

**Target behavior**
- [x] Retain every cTrader account returned for the configured access token instead of discarding all but the fixed account ID.
- [x] Normalize broker-reported account type as `demo` or `live`.
- [x] Identify which account is currently selected.
- [x] Expose the authorized account directory through `GET /api/broker/accounts`.
- [x] Add the typed frontend API contract needed by the upcoming account selector.
- [x] Add focused regression coverage for mixed Demo/Live account discovery.
- [x] Local focused tests: 20 passed.
- [x] Full backend regression suite passed.
- [x] Frontend production build passed (701 modules).
- [x] GitHub CI #264 and final functional-head CI #265 passed.

**Verification**
- Local functional validation on `419ab23`: focused account-directory + account-safety suite 20/20, full backend suite, and frontend production build all passed with a clean working tree.
- `git diff --check` exposed one roadmap-only trailing-space issue; docs-only commit `46fdf75` corrected it, after which local `git status --short` and `git diff --check origin/main...HEAD` were both clean.
- PR #90 CI #264 passed the implementation head and CI #265 passed the corrected final functional head.

**Scope boundary**
This item does not yet change the active account, reconnect transports, or route orders differently. It establishes broker-truth account discovery so account selection can be implemented without environment-variable assumptions.

### 10.2 Dashboard account selector and persisted active account
**Status:** ✅

**Target behavior**
- [x] Persist the explicitly selected cTrader account ID in the existing SQLite-backed engine configuration.
- [x] Keep broker truth explicit by distinguishing the saved `selected` account from the currently authenticated `active` transport account.
- [x] Validate account selection against the authorized account directory before saving it.
- [x] Expose an explicit `POST /api/broker/accounts/select` operation without reconnecting or mutating broker execution state.
- [x] Add a cTrader-style account selector to the System dashboard with broker title, automatic Demo/Live label, and the trader login/account number shown in cTrader.
- [x] Show the active authenticated account separately and disclose when a saved selection is pending a transport switch.
- [x] Persist the selected account through the existing config serializer without a database schema migration.
- [x] Local focused backend tests: 28 passed.
- [x] Local full backend regression suite passed.
- [x] Local frontend production build passed (701 modules).
- [x] GitHub CI #274 passed on the final functional head.

**Verification**
- Local final-head validation on `c3b79ff`: focused account-selection/persistence/account-safety suite 28/28, full backend suite, and frontend production build all passed.
- `git status --short` and `git diff --check origin/main...HEAD` were clean before and after the test/build run.
- The selector presents broker title + Demo/Live + trader login/account number while keeping internal cTrader account IDs for API routing and persistence.
- PR #91 CI #274 passed the final functional head.

**Scope boundary**
This item stores operator intent and presents it truthfully in the dashboard. It does not reconnect the cTrader client, change the active host/account, or alter order routing; those remain Phase 10.3 and 10.4.

### 10.3 Account-aware transport switching
**Status:** ✅

**Target behavior**
- [x] Persist the selected account's broker-reported Demo/Live type alongside its internal cTrader account ID.
- [x] Apply a persisted account ID/type before the cTrader transport thread starts so restarts can boot directly on the correct host.
- [x] Preserve a one-time migration path for Phase 10.2 selections that stored an account ID before account type was persisted.
- [x] Re-authenticate Demo→Demo or Live→Live account changes on the existing cTrader client without restarting Twisted's reactor.
- [x] Replace only the cTrader `ClientService` for Demo↔Live host changes; keep the existing Twisted reactor running.
- [x] Ignore stale connection/auth callbacks from the previous cTrader client after a host switch.
- [x] Separate desired account state from authenticated active-account state; execution/readiness fail closed while a switch is incomplete.
- [x] Clear symbol, asset, monetary, conversion, and adapter caches before changing account context.
- [x] Expose switch-in-progress, target-account, active-host, and last-switch-error state through broker status/UI.
- [x] Permit a Live account to authenticate for broker truth/market data while keeping the existing demo-only order-submission guard unchanged until Phase 10.4.
- [x] Reject account switching while the engine is active without the kill switch or while TradeAgent tracks an open broker-backed position.
- [x] Poll account-directory truth during the transition so the System page automatically converges from selected→active.
- [x] Local focused account-switching/safety tests: 59 passed.
- [x] Local full backend regression suite passed.
- [x] Local frontend production build passed (701 modules).
- [x] GitHub CI #287 passed on the final functional head.

**Verification**
- Local final-head validation on `451c5d3`: focused account-switching/directory/account-safety/broker-adapter/persistence/symbol-metadata suite 59/59, full backend suite, and frontend production build all passed.
- `git status --short` and `git diff --check origin/main...HEAD` were clean before and after validation.
- Same-host re-authentication, Demo↔Live client replacement, persisted startup targeting, stale-callback suppression, switch-state truthfulness, and safety-gate rejection are covered by focused regression tests.
- PR #92 CI #287 passed the final functional head.

**Scope boundary**
Phase 10.3 changes account/transport activation only. Demo-specific execution APIs, `demo_autotrade`, `allow_live` rejection, and the final live-order guard remain unchanged until Phase 10.4.

### 10.4 Account-neutral execution path
**Status:** ✅

**Target behavior**
- [x] Replace the active execution configuration flag with account-neutral `ctrader_autotrade`.
- [x] Migrate persisted legacy `demo_autotrade` state into `ctrader_autotrade` and discard the obsolete `allow_live` bypass flag.
- [x] Replace runtime broker operations with generic `symbol_execution_readiness`, `place_market_order`, `sync_position_targets`, and `close_position` semantics.
- [x] Treat the explicitly selected authenticated Demo or Live account as confirmed broker truth through `account_verified`.
- [x] Require the same verified account currency/equity, symbol contract metadata, quantity/risk, protective-stop, position-limit, reconciliation, recovery, and audit controls for Demo and Live.
- [x] Permit the config API and execution engine to use a selected Live account without a second `allow_live` permission switch.
- [x] Preserve backward-compatible read/wrapper aliases where needed for persisted state and older focused tests without using demo-only names in the active runtime path.
- [x] Update System/Trade settings to label Live execution explicitly and warn that Live auto-trade can place real-money orders.
- [x] Add focused regression coverage for legacy config migration, verified Live readiness, generic Live market-order routing, Live status mode, and config acceptance.
- [x] Local focused account-neutral execution/safety tests passed.
- [x] Local full backend regression suite passed.
- [x] Local frontend production build passed (701 modules).
- [x] GitHub CI #320 passed on the repaired final functional head.

**Verification**
- Initial local regression on the first 10.4 head exposed stale demo-only test hooks plus one real runtime gap: low-level cTrader monetary snapshots still required demo confirmation.
- Repaired final functional head `a6941c0` uses generic active-account confirmation for Demo or Live monetary truth, updates runtime/audit identifiers for new account-neutral activity while retaining legacy-history readability, and aligns resilience/regression fixtures with `ctrader_autotrade`.
- Final local validation on `a6941c0`: focused Phase 10.4 regression suite passed, full backend suite passed, frontend production build passed, and both `git status --short` and `git diff --check origin/main...HEAD` were clean.
- PR #93 CI #320 passed on the same final functional head.

**Scope boundary**
This item changes the execution contract from Demo-specific to account-neutral while preserving all existing safety gates. Historical persisted audit/event identifiers are not rewritten in place. Final operator-facing documentation, environment examples, screenshots, and risk/disclaimer cleanup remain Phase 10.5.

### 10.5 Documentation and operator disclaimer
**Status:** ✅

**Target behavior**
- [x] Update README product/runtime wording from Demo-only to the single account-neutral cTrader architecture.
- [x] Document automatic authorized-account discovery from the cTrader access token and persisted System-dashboard account selection.
- [x] Clarify that `CTRADER_HOST_TYPE` / `CTRADER_ACCOUNT_ID` are bootstrap/fallback settings, not a manual account directory.
- [x] Update architecture and architecture-diagram wording for selected Demo/Live execution through the same guarded path.
- [x] Update operations guidance with account-switching behavior, selected-versus-active truth, and fail-closed readiness.
- [x] Update generated-strategy security wording so the trust boundary is independent of broker account type.
- [x] Remove current operator-facing claims that Live execution is permanently blocked.
- [x] Recommend Demo for development/testing and clearly warn that an authenticated selected Live account can place real-money orders when cTrader/per-symbol auto-trade and safety gates permit execution.
- [x] Review current screenshot captions/UI wording; no separate Demo/Live product architecture is described.
- [x] Local documentation consistency checks pass.
- [x] Full backend regression suite passes.
- [x] Frontend production build passes.
- [x] GitHub CI passes on the final implementation head.

**Verification**
- Local validation on implementation head `257e90b` used the actual `phase-10/ctrader-account-neutral-docs` branch with a clean working tree and clean `git diff --check origin/main...HEAD`.
- Documentation consistency checks found no stale current Live-blocked wording and confirmed the required access-token discovery, bootstrap/fallback, Demo recommendation, generated-strategy boundary, and real-money Live warning text.
- The full backend regression suite passed locally on `257e90b`.
- The frontend production build passed locally on `257e90b` with Vite transforming 701 modules.
- PR #94 CI #323 passed backend and frontend validation on the same implementation head.

**Scope boundary**
This phase changes documentation and operator guidance only. It does not alter runtime execution policy, risk parameters, account selection state, broker credentials, or the global 0.60 signal-strength threshold.

### 10.6 Trade-dashboard cTrader account selector
**Status:** ✅

**Trigger**
A real operator review on 2026-10-05 confirmed that the verified cTrader account selector was only exposed on System, not on the main Trade dashboard. After the selector was added to Trade, the cTrader platform showed more accounts than the current Open API token returned. Field testing and Spotware's official multi-environment sample clarified the correct boundary: the token-granted account directory is obtained through the Demo client, each returned account carries its broker-reported `isLive` flag, and that flag determines whether Demo or Live must be used later for account authorization/trading. OAuth was then re-authorized with the intended accounts and the Demo-sourced directory returned all seven accounts, including FP Trading Live trader login `2123962`. A subsequent real switch attempt was correctly blocked because TradeAgent still tracked open broker-backed XAUUSD position `57748750`. After that position closed, a real Demo→Live smoke attempt reached the Live selection state but the cTrader authentication request hit the SDK's 5-second response timeout; TradeAgent failed closed with no active account and Live Trading remained disarmed, while the Live→Demo recovery succeeded. A second field attempt with an explicit 15-second auth timeout failed the same way, so further blind timeout increases are not accepted as a fix. Stage-labelled field diagnostics then proved that Live application authentication succeeds and the timeout occurs specifically on Live account authentication for account `48922568`. Spotware's official OpenApiPy sample handles `ProtoOAAccountAuthRes` from the incoming message stream, so TradeAgent accepts that broker event as authoritative while retaining the request Deferred as a timeout/error fallback. A subsequent field run still timed out at Live account auth and also showed the concurrent Demo-directory probe timing out at application auth; the only broker error surfaced in that window was `ALREADY_LOGGED_IN` ("Open API application is already authorized"), while no `CONNECTIONS_LIMIT_EXCEEDED` or account-not-authorized error appeared. To reduce cross-host auth concurrency and make any remaining error attributable, TradeAgent now serializes the sequence: complete Live account authorization first, then start the Demo-only directory refresh, with safe per-request correlation IDs in diagnostics. Field validation on that serialized path successfully authenticated and verified Live account `48922568` / trader login `2123962` while the engine stayed off and Live Trading stayed disarmed, then returned cleanly to Demo account `44089601` / trader login `4206993`. The remaining issue was a Demo-directory probe warning whose broker error carried a different client message ID from the outstanding probe request; TradeAgent had incorrectly attributed that connection-level error to the probe stage. The probe now ignores uncorrelated broker errors and only fails on an error correlated to its outstanding request. Final diagnostics then showed the probe's application-auth response appearing on the active Live callback; forwarding that response allowed the probe to advance, but the correlated token account-list request was then rejected with `UNSUPPORTED_MESSAGE Trading account is not authorized`. Inspection of the upstream OpenApiPy transport revealed the root cause: `TcpProtocol._send_queue`, `_send_task`, and `_lastSendMessageTime` are class-level attributes, so concurrent Demo and Live protocol instances can share outbound transport state. TradeAgent now uses a small protocol subclass that shadows those fields per instance, keeping the Live trading socket and short-lived Demo directory socket truly isolated instead of routing responses across clients. That field evidence also exposed that broker event logging must redact access/refresh tokens and client secrets.

**Target behavior**
- [x] Reuse the existing authorized cTrader account directory on the Trade dashboard.
- [x] Show broker title, broker-reported Demo/Live type, and trader login/account number in the Trade toolbar.
- [x] Show saved selection and currently authenticated active-account truth distinctly on the Trade toolbar.
- [x] Mark the currently authenticated account as Active.
- [x] Use the existing `POST /api/broker/accounts/select` path and existing transport-switching/safety guards rather than creating a second account-selection mechanism.
- [x] Disable the selector while an account switch is in progress and surface account-directory/selection failures on the Trade page.
- [x] Show an explicit LIVE MONEY warning when the selected account is Live.
- [x] Treat the Demo endpoint as the authoritative source for the token-granted account directory, matching Spotware's multi-environment sample.
- [x] Preserve each broker-reported `isLive` value and use it to route later account authorization/trading to the correct Demo or Live host.
- [x] Reuse the active Demo trading connection for directory discovery when Demo is selected.
- [x] When Live is selected, authenticate/trade on the Live connection while refreshing the directory through a short-lived Demo-only discovery connection.
- [x] Do not query the Live endpoint for `ProtoOAGetAccountListByAccessTokenReq`.
- [x] Preserve the authorization boundary: accounts visible in the cTrader platform but not granted to the current access token are not invented or exposed by TradeAgent.
- [x] Add focused regression coverage for mixed Demo/Live rows returned from the Demo directory source and for Live transport + Demo directory separation.
- [x] Require an explicitly selected cTrader account before cTrader auto-trade can be enabled.
- [x] Keep bootstrap-active transport truth distinct from persisted operator selection; if no account is explicitly selected, the UI must show no selected account rather than silently treating the active fallback account as selected.
- [x] Keep Live accounts visible/selectable for broker truth, but default new real-money entry submission to disarmed.
- [x] Use an explicit 15-second response timeout for cTrader application/account authentication instead of the SDK's 5-second default; authentication failure still clears active-account truth and fails closed. Field validation proved that timeout length alone is not the Live-switch fix.
- [x] Add stage-labeled application/account-auth failure diagnostics; field validation identified the failing stage as Live account authentication for account `48922568`, not application authentication.
- [x] Accept matching `ProtoOAAccountAuthRes` from the active client's incoming broker message stream as successful account authorization, matching Spotware's OpenApiPy sample behavior; keep the handler idempotent in case the SDK Deferred also resolves.
- [x] Serialize cross-host Live authentication before starting the Demo-only directory refresh; field evidence showed the concurrent Live account-auth + Demo probe path timing out on both sides and surfacing `ALREADY_LOGGED_IN`, so the directory probe is deferred until Live account authorization succeeds.
- [x] Attach safe client message correlation IDs to application/account-auth and Demo-directory requests so any remaining broker error can be attributed to the exact request stage without exposing credentials.
- [x] Ignore Demo-directory broker errors whose client message ID does not match the outstanding probe request; field evidence showed an uncorrelated `INVALID_REQUEST Trading account is not authorized` event being misattributed to probe application auth after the Live account itself had already authenticated successfully.
- [x] Isolate OpenApiPy TCP send queues/tasks per protocol instance so simultaneous Live trading and Demo directory clients cannot share outbound transport state; field evidence plus upstream source inspection showed the library's class-level send queue could put Demo-probe requests onto the Live socket.
- [x] Redact cTrader access tokens, refresh tokens, authorization codes, and client secrets before broker payloads are written to logs.
- [x] Add a runtime-only, account-bound Live Trading arm; it is not persisted across backend restart.
- [x] Automatically disarm Live Trading on account changes and engine restart.
- [x] Require the selected account to be the authenticated active Live account before arming.
- [x] Require the engine to be stopped before arming Live Trading.
- [x] Require Live Trading to be armed before an engine start can proceed with cTrader auto-trade on a Live account.
- [x] Gate new Live entry submission on the arm while preserving protection, reconciliation, and verified-close handling for already-open broker-backed positions.
- [x] Surface armed/disarmed Live state in System, Trade setup, status/readiness, incidents, and the Trade toolbar.
- [x] Local frontend production build passed on the Phase 10.6 frontend implementation (701 modules).
- [x] Local browser verification confirmed the selector is visible directly on the Trade dashboard.
- [x] GitHub CI #338 passed on the pre-simplification implementation head; final-head CI remains required.
- [x] Re-authorize the cTrader Open API token with the intended FP Trading Demo/Live accounts and confirm the Demo-sourced directory returns all seven intended accounts.
- [x] Local Demo/Live selection smoke check confirms selected-versus-active truth and existing switching guards remain intact.
- [x] Full backend regression passes on the final simplified implementation head.
- [x] GitHub CI passes on the final simplified implementation head.

**Verification**
- Final local validation on implementation head `91cdf9e` passed the focused cTrader account-switching/directory/demo-safety/Live-arm/status suite, the full backend regression suite, clean working-tree checks, and `git diff --check origin/main...HEAD`.
- Final real broker smoke started from Demo account `44089601` / trader login `4206993` with the engine stopped, account verified, Live Trading disarmed, and zero broker-backed open positions.
- Demo→Live switched to account `48922568` / trader login `2123962`; selected and active truth matched, application/account authorization succeeded, account verification stayed true, no switch error was present, Live Trading remained disarmed, and the authorized directory contained all seven accounts.
- With per-connection OpenApiPy transport state isolated, the post-Live Demo directory probe authenticated on its own socket and returned all seven token-granted accounts without `ALREADY_LOGGED_IN`, `UNSUPPORTED_MESSAGE`, or timeout failure.
- Live→Demo returned cleanly to account `44089601` / trader login `4206993` with selected/active/verified Demo truth restored, no switch error, and Live Trading still disarmed.
- PR #96 CI #388 passed the backend suite and frontend restart/Journal acceptance plus production build on `91cdf9e`.

**Scope boundary**
This item changes account-directory refresh behavior, Trade/System UX, and the fail-closed eligibility gate for new Live entries. It does not change persisted account selection, Demo↔Live transport replacement, broker credentials, existing position protection/reconciliation/close safety, sizing/risk limits, or the global 0.60 signal-strength threshold. The account directory reflects only accounts granted to the current cTrader Open API access token. Live arming is deliberately runtime-only and account-bound; it must not silently persist through backend restart or account changes.

### 10.7 Windows Credential Manager for cTrader secrets
**Status:** 🧪 Ready to test

**Trigger**
The operator explicitly requested stronger local protection after Live-account support was enabled. The current ignored `backend/.env` prevents normal Git commits but still leaves the cTrader client secret/access token in a plaintext project file.

**Target behavior**
- [x] On Windows, read `CTRADER_CLIENT_ID`, `CTRADER_CLIENT_SECRET`, and `CTRADER_ACCESS_TOKEN` from Windows Credential Manager before environment-variable fallback.
- [x] Keep environment-variable credentials available for non-Windows/CI compatibility.
- [x] Keep `CTRADER_HOST_TYPE` and `CTRADER_ACCOUNT_ID` as non-secret bootstrap/fallback configuration; normal selected account ID/type remains persisted in local SQLite.
- [x] Add a local credential-management CLI that never prints secret values.
- [x] Support secure interactive credential entry for a fresh Windows setup.
- [x] Support migration from an existing ignored `.env`, verify every Credential Manager write by reading it back, and only then optionally scrub the three plaintext secret assignments.
- [x] Roll back prior Credential Manager values if migration fails before completion.
- [x] Provide an explicit emergency `restore-env` path without changing normal runtime precedence.
- [x] Update the environment template and local-run/README guidance so Windows users do not treat plaintext `.env` secrets as the preferred configuration.
- [x] Preserve the existing `*.env` Git ignore rule; no credential value is introduced into tracked source, SQLite, logs, or API responses.

**Verification**
- [ ] Focused credential-store/migration tests pass locally on Windows.
- [ ] Focused cTrader account/auth safety regression passes locally.
- [ ] Full backend regression suite passes locally.
- [ ] Real Windows migration copies the three current cTrader credentials into Credential Manager and reports manager precedence without displaying values.
- [ ] After verified migration, `backend/.env` contains no `CTRADER_CLIENT_ID`, `CTRADER_CLIENT_SECRET`, or `CTRADER_ACCESS_TOKEN` assignment while retaining bootstrap/runtime settings.
- [ ] Backend restart authenticates the existing Demo account successfully using Credential Manager with the plaintext cTrader secrets absent from `.env`.
- [ ] `git check-ignore backend/.env`, scope/whitespace checks, and clean working-tree checks pass.
- [ ] GitHub CI passes on the exact implementation head.

**Scope boundary**
This item changes only local secret storage/loading and operator tooling/documentation. It does not rotate/re-authorize the cTrader token, alter authorized accounts, change account selection, arm Live Trading, route orders, change risk/sizing/protection/reconciliation behavior, or modify the global 60% signal-strength threshold.

---

## 15. Next item

**Phase 10.7 Windows Credential Manager hardening is the active operator-requested item while the Phase 7 protected-position disconnect/recovery field observation remains blocked by zero open managed positions. Validate the credential migration and backend restart without printing or committing any secret. After Phase 10.7 is verified and merged, return to the Phase 7 field observation only when a natural qualifying protected Demo position exists. Phase 5.3 still has 0 of 3 qualifying cells, so no threshold-comparison study is authorized and the global 60% signal-strength threshold remains unchanged.**
