# Option A — Alpaca SPY/QQQ D1 baseline: observed field-study review

**Observation date:** 2026-10-25 (UTC report timestamp `2026-10-25T19:58:09.930476+00:00`)  
**Source:** Operator-invoked `python -m scripts.research_alpaca_strategy_baseline --bars 500 --output <local JSON>` on merged `main` `88a38d6`.  
**Original operator-supplied JSON:** `alpaca-spy-qqq-baseline.json`, **SHA-256** `390c1eea3dfd1e9e069d057af7c45ae946d9d4227f4e14c907f0bf5faf75b6e8` (retained outside repository, not committed).  
**Scope:** Read-only historical-data observations; no Paper/Live orders or automatic entry activation.

## Coverage and comparison

- Alpaca `us_equity`, **IEX**, `raw` adjustment, `D1`; 500 completed bars each for SPY and QQQ, dated **2024-10-25 through 2026-10-23**. Chronological development window: 350 bars through **2026-03-20**; held-out window: 150 bars from **2026-03-23** through 2026-10-23.
- The JSON reports **112 calendar gaps** for each symbol, largest **5,760 minutes** (four days), plausibly including non-trading closures. This is *not* an independent exchange-calendar completeness audit; do not label the data complete or defective solely from the gap count.
- Assumed **1 bp fee per side + 1 bp slippage per side + 2 bp quoted spread**, normalized **100%** research allocation. These are *not* provider-confirmed fills or an executable **$500** fractional-share, cash-limited account.
- The rows below are **reported model outputs**, not attainable investor returns. Returns, drawdowns, and win rates require accounting/model-integrity review before strategy approval.

| Symbol | Strategy | Development net return | OOS net return | OOS trades | OOS win rate | OOS max drawdown | Full net return | Full gross-to-net drag |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SPY | SMA cross | -2.8079% | +1.1736% | 14 | 42.86% | -9.14% | -2.1336% | 2.5419 pp |
| SPY | RSI reversal | +1.9517% | -1.4892% | 4 | 50.00% | -5.48% | -2.6934% | 0.8833 pp |
| SPY | Range breakout | -0.0151% | +8.1941% | 8 | 62.50% | -1.57% | +9.5983% | 1.9221 pp |
| QQQ | SMA cross | +16.2992% | +1.3609% | 22 | 50.00% | -11.46% | +17.1274% | 3.6110 pp |
| QQQ | RSI reversal | +5.3692% | -7.4584% | 8 | 25.00% | -11.76% | -2.4896% | 1.0689 pp |
| QQQ | Range breakout | -0.5295% | +18.2009% | 9 | 55.56% | -4.68% | +21.4474% | 2.0130 pp |

*OOS = previously reserved out-of-sample segment; pp = percentage points. Development, OOS and full backtests each start accounting at normalized equity 1, so period returns cannot simply be added; period trade counts may differ from the full run at reset boundaries.*

## Diagnostic interpretation (research, not strategy promotion)

1. **Range breakout warrants a methodological audit, not promotion.** Its held-out returns (+8.19% SPY; +18.20% QQQ) contrast sharply with essentially flat development (-0.015% and -0.530%). The held-out counts are only **8 and 9**, of which QQQ has **8 closed + 1 marked-open**; the evidence is far too thin for a reliable profitability claim. Post-hoc choosing breakout as the winner now consumes the apparent independence of this window for future selection.
2. **Breakout reported all OOS completed exits as stops**, with **zero target exits**: SPY **8 stop exits / 8 trades**, QQQ **8 stop exits / 8 closed trades**. A stop exit can realize a gain if protection has moved; this does *not* mean eight losses. Without trade-by-trade fills, prices and stop updates, the mechanism behind the strong returns cannot yet be audited.
3. **Regime concentration:** Breakout's third 168-bar regime (2026-02-25–2026-10-23) reports **+12.8503% SPY** and **+22.0938% QQQ**, while the first two regimes report SPY **-2.6266%, -1.1788%** and QQQ **+0.2584%, -1.7047%**. Its edge is not demonstrated as regime-stable.
4. **Development-only parameter sensitivity is fragile:** breakout `lookback=20, rr=2.2` baseline is near flat for both symbols; `lookback=12` reports **+4.1336% SPY / +5.2493% QQQ**; `lookback=30` reports **-3.5335% SPY / -3.3538% QQQ**. Do not optimize parameters on the already-seen OOS data.
5. **SMA cross** had substantial development profits on QQQ (+16.2992%) but only +1.3609% OOS with -11.46% OOS drawdown; SPY was negative in development and full run. **RSI reversal** was negative OOS on both; SPY's OOS sample is just four trades. None qualifies for promotion.

## Accounting and execution limitations requiring separate corrective tasks

**P0 model integrity before relying on reported risk-adjusted scores or promoting a strategy:**

- **Pandas warning / Sharpe calculation:** Operator console emitted repeated `FutureWarning` at `backend/services/studio_backtests.py:208` calling `daily_equity.pct_change()` without an explicit fill policy. `_daily_metrics` resamples to **calendar days** even though bars are trading sessions; implicit weekend/holiday padding can alter daily-return observations while Sharpe is annualized with `sqrt(252)`. Add deterministic trading-session handling, explicit `pct_change(fill_method=None)`, focused tests across market-closure gaps, and revalidate scores. The extent/direction of score error has not been quantified.
- **Protective exit realism:** Static review of `backend/services/runtime_strategy_validation.py::_runtime_accounting` shows the research auditor checks stop/target conditions against **next open and daily close**, but does not inspect **intraday high/low**. It also uses the stored stop/target **level as the exit fill when a daily open gaps through it**, instead of modeling adverse gap-through execution. This can materially misstate executions, returns and drawdowns. Propose a separate bounded, tested realism fix with explicit gap, intrabar stop/target collision priority, and next-bar timing rules. Do not infer the actual direction or magnitude of distortion without rerunning a corrected model.
- **Trade trace and allocation:** The 84-KB report contains aggregate metrics and sensitivity cases but no executable per-trade fill/stop-change ledger, cash constraints, fractional-share sizes, borrow restrictions, corporate-action-adjusted prices, or broker-validated cost/slippage. These remain required for stronger claims; the IEX raw-feed results are not SIP-adjusted or total-return benchmarked.
- **Split-window accounting:** `_runtime_accounting` resets equity and position state per development/OOS/regime window, which also explains why trades can be boundary-counted differently than the independent full run. Mark window metrics as independent scenarios, not additive subsequences.

## Decision, guardrails, and next gate

- **Observed field retrieval and report parsing: successful.** The user's Windows transcript confirmed report creation and six strategy rows; full JSON has been reviewed here against the script's output schema. No Paper field account/position/order calls are implied by this market-data GET.
- **Strategy verdict: no candidate approved for autonomous or supervised orders based on this report.** Breakout remains only a **diagnostic research candidate**; no holding out OOS results as fresh evidence after this comparison.
- **Next engineering priority (separate roadmap item / PR):** fix the deterministic daily Sharpe/calendar-padding warning and test it first, then audit protective-exit fill realism in its own bounded slice; do not expand Alpaca automatic Paper B1–B5, crypto, or Live.
- **Closeout for this docs-only evidence PR:** obtain Windows clean branch/diff check and GitHub exact-head CI before marking this field-review checkpoint verified, merging, and syncing `main`.

This document records an operator-observed historical-data study, **not** independently reconstructed Alpaca bars or an audited profitable strategy.
