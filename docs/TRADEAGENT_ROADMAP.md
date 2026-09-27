# TradeAgent — Local Validation & Improvement Roadmap

> Local working document. Keep this file out of Git and use it as the project checklist for validation, fixes, and improvements.
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
| Broker SL/TP synchronization | 🟨 | Works, but at least one `ctrader_demo_order_unprotected` event observed; hardening still required |
| Broker close detection | ✅ | Local positions close when broker position disappears |
| Broker realized P&L | ✅ | Journal reconciled to cTrader closing deals; NAS100 -2.71, US30 -0.73, XAUUSD -9.24 example matched cTrader total -12.68 CHF |
| Broker close price/time | ✅ | Closing deal execution price/time persisted from cTrader |
| Dashboard local time | ✅ | UTC backend timestamps now display in local browser time |
| Cooldown / confidence gating | ✅ | 75%/71% XAUUSD signals correctly rejected during 30-minute cooldown; later 67% signal executed after cooldown expired |
| SQLite persistence | ✅ | Runtime DB moved outside synchronized Google Drive worktree |
| Git / source of truth | ✅ | `main` synchronized with GitHub; clean worktree after PR #7 |
| Full backend test suite | ⛔ | 3 failures in `test_demo_execution_workflow.py`; investigate before further feature work |
| Frontend production build | 🧪 | Build started successfully; confirm final completion after current run |

---

## 2. Phase 0 — Restore a clean test baseline

### 0.1 Investigate Python / pytest path contamination
**Status:** ⛔

Observed full-suite failure paths reference:

`G:\My Drive\Software\GenAI-MultiAgent-TradingSystem - V2\backend\tests\...`

while the active repository is:

`G:\My Drive\Software\GenAI-MultiAgent-TradingSystem`

This may indicate an old editable install, `PYTHONPATH`, package import, or environment path still points at the previous `- V2` copy.

**Checks**
- [ ] Print `backend.__file__`
- [ ] Print `backend.tests.test_demo_execution_workflow.__file__`
- [ ] Inspect `sys.path`
- [ ] Inspect `pip show tradeagent-v2`
- [ ] Remove/reinstall stale editable package if necessary
- [ ] Confirm pytest collects tests from the canonical repository only

**Done when**
- Full-suite traceback paths point only to the canonical repository.
- No imports resolve from the old `- V2` folder.

### 0.2 Update demo workflow tests for strict broker readiness
**Status:** ⬜

Current production execution intentionally defers demo trading unless the cTrader account/symbol is confirmed ready. The failing workflow tests do not currently mock this new readiness gate.

**Tasks**
- [ ] Mock `get_demo_symbol_execution_readiness()` as ready in the happy-path/failure-routing tests.
- [ ] Ensure the "order failure" test reaches `place_demo_market_order()` rather than stopping at readiness.
- [ ] Ensure the engine test uses a fresh/current bar or mocks readiness as needed.
- [ ] Add regression coverage for "not confirmed as demo" → deferred/retryable behavior.

**Done when**
- `python -m pytest backend/tests -q` passes completely.
- Focused demo workflow tests explicitly cover both ready and not-ready states.

### 0.3 Confirm frontend production build
**Status:** 🧪

- [ ] `npm --prefix frontend run build`
- [ ] Record final result and warnings.

---

## 3. Phase 1 — Execution safety & broker truth

### 1.1 Hard failsafe for unprotected demo positions
**Priority:** P0  
**Status:** ⬜

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
- [ ] Immediate protection success.
- [ ] Delayed protection success.
- [ ] Repeated protection failure → broker close.
- [ ] Broker close itself fails → critical incident, no false local closure.

### 1.2 Broker position ID as primary reconciliation identity
**Priority:** P0/P1  
**Status:** ⬜

Broker position IDs are now persisted. Replace remaining symbol+direction matching with broker-position-ID matching wherever possible.

**Done when**
- Exact `broker_position_id` is primary.
- Symbol+direction is only a recovery fallback.
- Multiple same-direction positions cannot be confused.

### 1.3 Broker account currency / balance / equity synchronization
**Priority:** P0  
**Status:** ⬜

Recent execution intent details showed `account_currency=USD` and `starting_equity_amount=100000`, while the cTrader demo statement is CHF-denominated.

**Target separation**
- **cTrader demo mode:** broker-reported account currency and balance/equity drive monetary risk calculations.
- **Pure paper mode:** configurable virtual starting equity remains available.

**Validate**
- [ ] Deposit/account currency.
- [ ] Balance/equity.
- [ ] Daily loss budget.
- [ ] Risk-per-trade amount.
- [ ] Auto sizing.
- [ ] Cross-currency instruments.

### 1.4 Broker ledger completeness
**Priority:** P1  
**Status:** 🟨

- [ ] Investigate legacy position 7 with no returned closing deal.
- [ ] Confirm partial close support.
- [ ] Confirm multiple closing deals are weighted/summed correctly.
- [ ] Confirm commission/swap/conversion fees are reflected in net P&L.
- [ ] Confirm no duplicate broker deal can be counted twice.

---

## 4. Phase 2 — Trade workspace & operator UX

### 2.1 Show actual rejection reason in Trade Journal
**Priority:** P1  
**Status:** ⬜

Replace generic:

`Signal rejected by V2 risk engine.`

with useful summaries such as:
- `Rejected — XAUUSD in 30-minute cooldown.`
- `Rejected — confidence 54% < minimum 60%.`
- `Rejected — max daily trade count reached.`
- `Rejected — daily loss cap reached.`
- `Rejected — stale M5 market bar.`

Keep full structured details available in Intents/Incidents.

### 2.2 Clarify signal confidence semantics
**Priority:** P1  
**Status:** ⬜

Current confidence values are deterministic strategy-strength heuristics, not proven win probabilities.

**Improve UI wording**
- Consider `Signal strength` / `Model confidence` rather than implying empirical probability.
- Add tooltip explaining meaning and threshold.
- Later display calibrated probability separately if enough evidence exists.

### 2.3 Position panel broker truth
**Priority:** P1  
**Status:** ⬜

Show:
- broker position ID,
- broker entry,
- current broker SL/TP,
- quantity,
- protection status,
- unrealized broker P&L if available,
- last synchronization timestamp.

### 2.4 Journal filtering / drill-down
**Priority:** P2  
**Status:** ⬜

Filters:
- execution only,
- rejected signals,
- protection incidents,
- symbol,
- strategy,
- date,
- broker vs paper.

Row expansion should show intent/risk reasons and broker details.

---

## 5. Phase 3 — System page acceptance audit

Test every operator control against backend behavior.

### 3.1 Engine lifecycle
- [ ] Start.
- [ ] Stop.
- [ ] Restart.
- [ ] One-shot scan.
- [ ] Recover.
- [ ] Reconcile.
- [ ] Restart backend while broker position is open.

### 3.2 Safety controls
- [ ] Kill switch.
- [ ] Minimum confidence.
- [ ] Cooldown.
- [ ] Maximum daily trades.
- [ ] Maximum open positions.
- [ ] Maximum positions per symbol.
- [ ] Daily loss limit.
- [ ] Require protective stops.
- [ ] Session filter.
- [ ] Trading-enabled per watchlist item.

### 3.3 Persistence
- [ ] Settings survive backend restart.
- [ ] Watchlist survives restart.
- [ ] Runtime DB remains in nonsynchronized local path.
- [ ] No `.db`, `-wal`, or `-shm` state appears in Git.

### 3.4 Status truthfulness
- [ ] Connected.
- [ ] Demo confirmed.
- [ ] Execution ready.
- [ ] Symbol metadata ready.
- [ ] Engine scanning.
- [ ] Model ready.
- [ ] Incidents reflect actual current faults, not stale state.

---

## 6. Phase 4 — Build & Test / Strategy Studio acceptance audit

### 4.1 LLM research workflow
- [ ] Chat request.
- [ ] Strategy drafting.
- [ ] Provider/model selection.
- [ ] Failure/fallback path.
- [ ] Save strategy.
- [ ] Reload saved strategy.

### 4.2 Generated strategy sandbox
- [ ] Normal strategy executes.
- [ ] Forbidden imports rejected.
- [ ] File/network/system access rejected.
- [ ] Timeout enforced.
- [ ] Invalid output rejected.
- [ ] Resource limits reviewed.

### 4.3 Draft backtest correctness
**Known improvement:** draft backtest currently reports `Win Rate [%] = 0.0` rather than calculating trade-level wins.

- [ ] Fix win-rate calculation.
- [ ] Align draft and saved-strategy accounting logic.
- [ ] Verify flips, entries, exits and final open trade handling.

### 4.4 Backtest realism
- [ ] Fees.
- [ ] Slippage.
- [ ] Spread assumptions.
- [ ] Execution timing / no lookahead.
- [ ] Missing bars.
- [ ] Time zones/session assumptions.
- [ ] Trade-level return math.
- [ ] Sharpe calculations.
- [ ] Drawdown.
- [ ] Hold duration.
- [ ] Position sizing.

### 4.5 Validation methodology
- [ ] Development split.
- [ ] 30% out-of-sample holdout.
- [ ] Regime/different-market test.
- [ ] Walk-forward option.
- [ ] Parameter optimization without leakage.
- [ ] Minimum sample sizes.

### 4.6 Strategy lifecycle
Validate:

`draft → backtested → validated → paper → eligible`

- [ ] Hypothesis required.
- [ ] Development evidence.
- [ ] OOS evidence.
- [ ] Regime evidence.
- [ ] Paper evidence.
- [ ] Operator/reason required for promotion.
- [ ] Source hash/version changes invalidate old evidence appropriately.
- [ ] Generated strategy can never jump directly into autonomous execution.

---

## 7. Phase 5 — Strategy quality & confidence calibration

### 5.1 Runtime deterministic strategies
Current runtime strategies include:
- `sma_cross`
- `rsi_reversal`
- `breakout`

For each strategy × symbol × timeframe:
- [ ] Backtest.
- [ ] OOS.
- [ ] Regime analysis.
- [ ] Costs.
- [ ] Trade frequency.
- [ ] Drawdown.
- [ ] Expectancy.
- [ ] Parameter sensitivity.

### 5.2 Confidence calibration
Compare confidence buckets with realized outcomes, e.g.:

- 60–65%
- 65–70%
- 70–75%
- 75–80%
- 80%+

Measure:
- win rate,
- expectancy,
- average R,
- drawdown contribution,
- trade count.

Do **not** relabel strategy confidence as probability until calibration supports it.

### 5.3 Per-strategy / per-symbol thresholds
Only after enough evidence, evaluate whether one global 60% threshold is inferior to calibrated thresholds per strategy/symbol/timeframe.

---

## 8. Phase 6 — Market intelligence / events / confluence research

These features remain research/shadow-only until validated.

### 6.1 Calendar / events
- [ ] Event ingestion.
- [ ] Upcoming event display.
- [ ] Symbol/event mapping.
- [ ] Missing/stale data handling.

### 6.2 Event calibration
- [ ] Outcome windows.
- [ ] Sample counts.
- [ ] Calibration gates.
- [ ] Persistence.
- [ ] Reproducibility.

### 6.3 Confluence shadow mode
- [ ] Original vs shadow confidence.
- [ ] No-trade can never be promoted.
- [ ] Event evidence caps.
- [ ] Replay.
- [ ] Forward paper observation.
- [ ] Promotion remains manual/reviewed.

---

## 9. Phase 7 — Resilience / failure testing

Simulate deliberately:

- [ ] cTrader disconnect while flat.
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

---

## 14. Next item

**Phase 0.1 — determine why full-suite failures resolve from `GenAI-MultiAgent-TradingSystem - V2`, then restore a canonical clean test environment before modifying production logic.**
