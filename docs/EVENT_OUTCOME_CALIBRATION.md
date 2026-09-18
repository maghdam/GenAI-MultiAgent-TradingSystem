# Event outcome calibration

Event labels are research hypotheses, not trade signals. The calibration layer measures whether each hypothesis has historically aligned with subsequent market movement before the application may consider using it in a strategy-confidence model.

## Forward evaluation

For each event/symbol pair, the evaluator records outcomes at:

- 5 minutes
- 30 minutes
- 4 hours
- 1 day

The reference is the close of the first completed bar at or after the event timestamp. This avoids using a bar close that was not yet known when the event arrived. The target is the first completed bar at or after the requested horizon, which handles market gaps without fabricating prices.

Each outcome stores:

- Reference and target timestamps/prices.
- Forward percentage return.
- Directional maximum favorable excursion (MFE).
- Directional maximum adverse excursion (MAE).
- Predicted and realized direction.
- Directional hit/miss.
- Brier calibration score.
- Pending or unavailable reason.

Rows are uniquely keyed by event, symbol, and horizon. Pending or unavailable outcomes can be re-evaluated when new market data becomes available; completed outcomes are skipped.

Canonical event symbols are resolved to broker aliases before bars are requested. For example, `NAS100` can resolve to `US100`, `USTEC`, `NDX`, or another configured Nasdaq alias while the stored research symbol remains `NAS100`.

## Direction thresholds

Small movements are classified as flat using horizon-specific thresholds:

```text
5m   0.05%
30m  0.10%
4h   0.25%
1d   0.50%
```

These thresholds are explicit research assumptions and should later be adjusted for symbol volatility and trading costs.

## Evidence gate

Calibration is grouped by event type and horizon. The default minimum sample size is 30 evaluated outcomes.

An event group becomes `eligible` only when:

```text
samples >= minimum sample size
directional hit rate >= 55%
average Brier score <= 0.24
```

Possible states are:

- `insufficient_samples`
- `observe`
- `eligible`
- `degraded`

Eligibility is informational in this tranche. It does not alter strategy confidence or execution.

## APIs

```text
POST /api/market/events/calibrate
GET  /api/market/events/outcomes
GET  /api/market/events/calibration
```

The Market Intelligence screen includes an **Evaluate outcomes** control and a calibration panel showing sample size, hit rate, Brier score, average return, MFE, MAE, and gate state.

## Configuration

```dotenv
EVENT_CALIBRATION_AUTO=1
EVENT_CALIBRATION_MIN_SAMPLES=30
```

When automatic event monitoring is enabled, pending outcomes are evaluated after each source-refresh cycle.
