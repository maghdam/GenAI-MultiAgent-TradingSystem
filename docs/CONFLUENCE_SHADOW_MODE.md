# Event Confluence Shadow Mode

The workstation now measures whether calibrated market events confirm or conflict with a technical strategy decision. This layer is deliberately **observation-only**: the risk engine and paper executor continue to receive the original strategy analysis.

## Eligibility

An event can contribute only when:

- its symbol resolves to the analyzed instrument (including `NAS100`, `US100`, `USTEC`, and `NDX` aliases);
- its event type and target horizon have passed the outcome-calibration gate;
- it is recent enough for that horizon;
- its historical group has at least 30 evaluated observations, at least 55% directional accuracy, and a Brier score no greater than 0.24.

Timeframes map to outcome horizons as follows: `M1 → 5m`, `M5 → 30m`, `M15/M30 → 4h`, and `H1/H4/D1 → 1d`.

## Bounded comparison

Eligible evidence is weighted by source credibility, impact, calibration quality, and recency. Aligned evidence can add no more than 12 confidence points; conflicting evidence can remove no more than 20 points. A technical `no_trade` can never be promoted into a trade.

Each analysis stores both values, the event IDs used, threshold outcomes, rationale, and the explicit `execution_unchanged` flag. The Workbench shows these records under **Event Confluence · Shadow**.

## Promotion criteria

Do not connect this layer to risk or execution until replay and forward paper observation show that it improves out-of-sample expectancy and drawdown without introducing unstable trade frequency. Promotion should require a separately reviewed configuration change, not happen automatically.
