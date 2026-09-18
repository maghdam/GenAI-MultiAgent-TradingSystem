# Strategy lifecycle

V2 treats AI-generated strategy code as a research artifact. Saving a strategy registers an immutable SHA-256 source version; it does not load the code into the trusted runtime or authorize execution.

## Stages

1. `draft` ? define a measurable hypothesis and run development evidence.
2. `backtested` ? development gate passed; collect independent validation.
3. `validated` ? both holdout and alternate-regime gates passed.
4. `paper` ? an operator explicitly approved supervised paper observation.
5. `eligible` ? paper-trading evidence passed. This is eligibility for a later supervised deployment decision, not automatic live activation.
6. `retired` ? version is no longer eligible for promotion or execution.

Promotion is sequential and requires a named operator. Every transition, reason, evidence record, dataset window, metric set, and source hash is stored in SQLite.

## Default research gates

- Development: at least 30 closed trades.
- Holdout: at least 20 closed trades.
- Alternate regime: at least 10 closed trades.
- Backtest return must be positive.
- Maximum absolute backtest drawdown is 15%.
- Fees plus slippage must be greater than zero; zero-cost results cannot pass.
- Paper observation: at least 20 closed trades, positive return, and maximum drawdown of 10%.

The defaults are stored on each strategy version so later policy changes do not silently rewrite historical decisions.

## Validation windows

- **Development 70%** uses the first chronological 70% of fetched bars.
- **Holdout 30%** uses the final chronological 30% and remains unseen during the matching development run.
- **Regime / alternate market** uses the full requested sample. Select a materially different symbol, timeframe, or market regime before recording this evidence.

Backtest evidence attaches only to the exact saved source hash. Editing and saving the strategy creates a new version at `draft`; evidence from an older version is not inherited.

## Daily workflow

1. Create a draft in Strategy Studio and save it under a stable name.
2. Write and save the measurable hypothesis.
3. Enter realistic fee and slippage assumptions.
4. Run Development 70%, review the evidence, then promote to `backtested` if the gate passes.
5. Run Holdout 30% without changing the code.
6. Select an alternate market, timeframe, or historical regime and run Regime evidence.
7. Promote to `validated`, then explicitly approve `paper` observation.
8. Run the exact governed strategy in paper mode. Use **Collect paper evidence** to calculate results from closed V2 paper positions.
9. Promote to `eligible` only after the paper gate passes.

For lifecycle-registered strategies, the central risk engine rejects paper execution unless the current source hash matches and the stage is `paper` or `eligible`. Existing built-in strategies without a lifecycle record keep their legacy behavior until they are enrolled.
