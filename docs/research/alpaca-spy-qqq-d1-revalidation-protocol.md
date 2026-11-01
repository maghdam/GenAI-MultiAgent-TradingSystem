# Option A — repeat the Alpaca SPY/QQQ D1 field study after research-integrity fixes

**Status (2026-11-01):** ✅ **Actual operator field attempt succeeded and the uploaded JSON's six model-output rows have been reviewed; final PR documentation-only head Windows/CI verification and merge remain pending.** This is a read-only evidence checkpoint, **not** an order-authorization, model-selection, or strategy-promotion request.

**Code baseline:** `main` at PR #141 squash merge `defb14e4f16f3f53d098672814229b13cd78334d`. Prior first study: [observed 2026-10-25 report](alpaca-spy-qqq-baseline-2026-10-25.md). That earlier report was generated **before** observed-session daily-return corrections (#134), OHLC exit realism (#135), historical candle validity (#136), intraday overlap (#137), D1 New York bar labels (#138), exchange-session calendar (#139), missing expected session fail-closed (#140), and requested/returned/retained sample-depth provenance (#141). **Never compare raw old/new strategy returns as an isolated strategy performance effect**, because model assumptions, validation gates, date ranges and sample windows may differ.

## Scope and exact operator procedure

Use the unchanged audited `scripts.research_alpaca_strategy_baseline` on a clean, GitHub-matching branch. **No new broker credentials, third-party feeds, orders, positions, account balance requests, persistence, autonomous engine or strategy modifications are part of this step.** The code requests historical stock bars via Alpaca's read-only market-data GET and exchange dates via Paper GET `/v2/calendar`. The output path MUST be **outside the Git repository**, e.g. the operator's `$env:TEMP` directory. Never commit the full JSON or credential/environment files.

Run after local branch/diff verification and required validation:

```powershell
$report = Join-Path $env:TEMP "alpaca-spy-qqq-d1-revalidation-2026-11-01.json"
if (Test-Path $report) {
    throw "Existing report path would be overwritten: $report. Rename it before rerunning."
}
python -m scripts.research_alpaca_strategy_baseline --bars 500 --output "$report"
if ($LASTEXITCODE -ne 0) {
    Write-Host "READ-ONLY FIELD STUDY FAILED CLOSED: review provider/calendar/data error, do not suppress guards or enable orders."
    throw "Read-only Alpaca D1 field-study check did not complete."
}
if (-not (Test-Path $report)) {
    throw "Expected local research JSON was not produced."
}
Get-FileHash -Algorithm SHA256 "$report"
```

A **failed** provider/calendar/missing-bar check is a valid research outcome, not permission to repair dates or synthesize candles. Save the exact redacted error text and report no strategy scores. Do not retry by silently switching feed subscriptions/adjustment modes or reducing the historical window without a separate reviewed decision.

## Required evidence for an accepted report

- **Identity:** report `study`, `mode=research_only`, `read_only=true`, `broker_order_requests=0`, UTC observation timestamp, selected symbols **SPY and QQQ**, D1 timeframe, stock feed (`iex` or explicitly configured provider), `raw` adjustment, cost assumptions and StrategyV2 methodology.
- **Source sample truth:** for each symbol, `data_source.sample_depth_by_symbol`: 500 requested, provider-returned, retained completed, conservative current-UTC-day exclusions, distinct shortfall counts and `requested_completed_sample_satisfied`; verify observed bar minimum `>=250` independently of whether 500 was fulfilled. Never rewrite the provider count as the retained count.
- **Exchange dates:** per-symbol `calendar_validation_by_symbol`: exchange core-session source, trading dates covered, observed dates, early-close observations, and missing scheduled session outcome. If any **in-range expected trading session is unobserved**, the CLI must fail closed without a report. This is not independent confirmation of IEX/SIP consolidation or price accuracy.
- **Strategy results:** retain all six SPY/QQQ × SMA cross / RSI reversal / breakout results as **new, dated research outputs**; review development, out-of-sample and full-run returns, trade counts/exit types, Sharpe/drawdown and cost assumptions **without selecting or promoting a winner**. Past out-of-sample 2026-10-25 comparisons have already been seen and are **not fresh holdout evidence**.
- **Reproducibility:** capture exact Git branch SHA, clean worktree, command, UTC report timestamp, outside-repo report filename and SHA-256, **not secrets or account identifiers**. Report any warnings and failed-closed outcomes accurately; do not promise a pass.

## Acceptance and subsequent decision

Operator supplied the exact Windows validation/field output and full JSON report for this item. Exact report hash and fields have been verified and six model-output rows reviewed below. **Evidence is received and evaluated, but PR #142's final docs-only head Windows/CI gate and clean-main merge closeout remain outstanding.** Do not make strategy-selection or external-data-quality claims based on this acceptance.

**Explicit nonclaims:** No independent source/feed completeness comparison is included in this item; no verification of raw-vs-adjusted corporate actions, IEX/SIP equivalence, realized slippage/fills, borrowing, deployable $500 equity sizing, fresh out-of-sample validation or strategy profitability. Independent cross-source coverage, H4 provider timestamps, intrabar event-time assumptions, and broker-cost realism remain future **separate** roadmap items; optional Alpaca Paper B1–B5/autonomous, crypto and Live execution are not enabled.


## Actual observed field result and reviewed JSON (2026-11-01)

**Evidence:** Operator Windows on `research/alpaca-d1-readonly-field-revalidation` at `33165685fc7af0d8ef2d9d00dbf7aff0389390f0` passed **77/77** focused research tests and canonical `python scripts/validate.py` full backend regression/TypeScript Vite **707-module** frontend build; clean diff/working tree. GitHub Actions **#603** completed successfully (backend Python 3.12, frontend Node 22). The operator then explicitly invoked `python -m scripts.research_alpaca_strategy_baseline --bars 500 --output "$report"` once, successfully (`exit=0`) and without changing the repository. Saved JSON remained **outside** the repository in the user's TEMP directory. **Uploaded report bytes were read and independently SHA-256 checked** (96,138 bytes); the digest matches the PowerShell console:

`5D722D667F9AAB412DBB1D6E5F0B261E0B8C01BBFB868ACC161F37A5BF9C67E7`

**Recorded time:** `2026-11-01T17:28:11.364126+00:00` UTC. JSON identity: `study=option-a-alpaca-spy-qqq-trusted-baseline-v1`, `mode=research_only`, `read_only=true`, `broker_order_requests=0`, `symbols=[SPY, QQQ]`, `strategies=[sma_cross, rsi_reversal, breakout]`, `timeframe=D1`. Historical stock feed is **Alpaca IEX, raw adjustment**; this is not SIP consolidated or adjusted total-return data. **No trading orders were requested by the research script.**

**Daily source/coverage evidence:** For **each** of SPY and QQQ: **500 requested, 500 provider returned, 500 completed retained**, 0 conservative current-UTC-day exclusions, 0 count shortfalls, and requested completed sample satisfied. Exchange-calendar cross-check: **500 observed NY trading dates == 500 scheduled dates**, 0 missing in-range expected sessions, calendar response-shape validated and **5 early-close observations**. Both `independent_feed_completeness_verified=false` and `observed_bar_completeness_verified=false` remain reported: calendar/date/count checks are **not** independent verification of OHLC quality, comprehensive IEX/SIP coverage, corporate actions, or actual exchange fills. Observed daily session gaps: 112 per symbol in the study's generic calendar-gap display; these are not 112 missing exchange trading dates.

**Model and windows:** 500 completed NY-session D1 bars dated **2024-11-01 to 2026-10-30**. Development window first **350** dated 2024-11-01–2026-03-27; reused OOS last **150** dated 2026-03-30–2026-10-30 (all dates session-local). Each split restarts normalized equity/position state; segment returns and trade counts are not additive to the independent full run. Assumed **1 bp fee per side, 1 bp slippage per side, 2 bp full quoted spread**, `position_size_pct=100`; stop-first when both daily OHLC thresholds hit and adverse gap-open protection modeled. Intrabar exit *time* is still approximated; live borrow/cash/share sizing and actual broker fills are not modeled.

### Six independently extracted reported outcomes

**All return/drawdown/drag figures are percentages or percentage-point gross-to-net differences; results are model outputs, not executable returns.**

| Symbol / strategy | Development net return | Reused OOS net return | Full net return | OOS trades | OOS win rate | OOS max drawdown | OOS Sharpe | Full max drawdown | Full cost drag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SPY / SMA cross | -13.0287% | -9.2445% | -21.6047% | 31 | 35.48% | -10.51% | -1.397 | -24.02% | 4.4667 pp |
| SPY / RSI reversal | -3.0655% | -2.0312% | -5.1763% | 5 | 60.00% | -5.73% | -0.731 | -13.02% | 0.9182 pp |
| SPY / Range breakout | -10.7728% | +5.6937% | -5.5502% | 9 | 33.33% | -1.83% | +1.756 | -11.74% | 2.1282 pp |
| QQQ / SMA cross | -7.0535% | -5.5536% | -12.4010% | 33 | 30.30% | -12.01% | -0.421 | -26.56% | 4.7655 pp |
| QQQ / RSI reversal | +15.0814% | -6.5688% | +6.9410% | 8 | 25.00% | -11.40% | -1.432 | -13.43% | 1.2302 pp |
| QQQ / Range breakout | -23.6212% | +10.1691% | -16.0731% | 11 | 45.45% | -6.40% | +1.623 | -26.17% | 1.8960 pp |

**Interpretation and audit notes:**

1. **The system successfully collected, validated and reported six analyses** under its current mocked/regression-tested research accounting. That is the positive technical finding. **It is not a positive strategy-verdict**: the full-window net results are **negative in five of six**; QQQ RSI reversal is **+6.941% full** but **-6.5688% reused OOS** with only **8 OOS trades**. The 100%-allocation model is not an executable small cash-account simulation.
2. **Breakout's reused OOS results are positive** (SPY **+5.6937% / nine trades**; QQQ **+10.1691% / eleven trades**) but its **development results are negative** (SPY -10.7728%, QQQ -23.6212%), and its **full-run returns are negative**. Crucially, the OOS report labels **all nine SPY and eleven QQQ exits as stop exits, with zero target exits**. A stop exit may be profitable after protective-stop movement, but aggregate exit labels cannot establish **when/how the favorable stop was modified**. This remains an explicit **future per-trade execution trace / intrabar timing audit**, not a reason to promote breakout or change strategy parameters in this PR.
3. **Very small OOS trade samples / multiple comparisons:** RSI SPY 5, RSI QQQ 8, breakout SPY 9 and breakout QQQ 11 OOS trades. None establishes stable performance. OOS 2026 dates overlap the already-seen October 25 study; these segments **cannot be represented as fresh independent holdouts or used to select a winning configuration**.
4. **Historical comparison is diagnostic only.** The October 25 observed report (before the fixes, different D1 endpoints) described SPY/QQQ breakout OOS **+8.1941%/+18.2009%**, full **+9.5983%/+21.4474%**; the updated analysis instead reports OOS **+5.6937%/+10.1691%** and full **-5.5502%/-16.0731%**. This is a **large change in modeled results**, not evidence that any individual code fix caused exactly that change. Dates/splits, historical samples, accounting, execution assumptions and calendar/data gates differ; no controlled attribution is possible from these two aggregate reports.
5. **Observed cost drag matters:** full-period gross-to-net drag ranges from **0.9182 to 4.7655 percentage points** under the stated synthetic fee/slippage/spread model. This is neither independently realized broker cost nor a robustness test over alternative costs. Development-only parameter sensitivity remains post-hoc diagnostic rather than OOS selection.
6. **Unresolved research limitations:** provider/feed provenance is not independently audited, raw IEX data may exclude corporate-action adjustments, historical OHLC stop/target collision order and approximate intrabar time are not broker fills, and shorting/borrowing/cash/fractional-share constraints remain unmodeled. No forward/Paper/live execution inference is authorized.

**Decision:** Mark **field collection, report integrity envelope and six-row diagnostic review as observed and reviewed**. Do **not** approve any of the three strategies for trading or call any a validated edge. Continue PR #142's **final documentation-only Windows/exact-GitHub-CI verification** before merge; later remediation remains separately scoped.

**Reproducibility/privacy:** Full 96,138-byte local JSON remains outside GitHub; only its hash, window, data/provenance counts and aggregate research findings are recorded here. No credentials/account IDs and no actual order outputs are included.
