# Option A — repeat the Alpaca SPY/QQQ D1 field study after research-integrity fixes

**Status (2026-11-01):** 🧪 **Protocol staged; no new operator field observation yet.** This is a read-only, manually initiated evidence checkpoint, **not** an order-authorization, model-selection, or strategy-promotion request.

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

Operator supplies the exact PowerShell validation/field output and the saved report (or its relevant redacted research-only JSON excerpt). Review **whether it succeeded or failed closed** before updating this document with observations. Do not manufacture prices/metrics or mark this item verified prematurely. After operator evidence, update roadmap/work log and the field-study review on the **same one-item PR**, then require final documentation-only SHA Windows verification and GitHub CI before merge and clean `main` sync.

**Explicit nonclaims:** No independent source/feed completeness comparison is included in this item; no verification of raw-vs-adjusted corporate actions, IEX/SIP equivalence, realized slippage/fills, borrowing, deployable $500 equity sizing, fresh out-of-sample validation or strategy profitability. Independent cross-source coverage, H4 provider timestamps, intrabar event-time assumptions, and broker-cost realism remain future **separate** roadmap items; optional Alpaca Paper B1–B5/autonomous, crypto and Live execution are not enabled.
