# Decision and monetary sizing foundation

## Safety boundary

LLMs and strategies may propose a trade. Deterministic application code remains the only authority that can qualify size, accept an intent, and update the paper ledger.

## Monetary sizing

Risk-based size is calculated in account currency:

```text
risk amount = paper equity × risk percent
loss per lot at stop = abs(entry - stop) × cash per price unit per lot
lots = risk amount / loss per lot at stop
```

The cTrader adapter obtains the full symbol contract and snapshots the resulting monetary specification into every new paper position. This prevents later metadata changes from rewriting historical P&L.

Automatic execution is rejected when risk sizing is enabled but contract valuation or account-currency conversion is unavailable. Set risk per trade to zero only when explicitly testing fixed-size behavior.

The current conversion implementation values contracts directly when the inferred quote currency equals the configured account currency. Cross-currency contracts remain blocked for automatic risk sizing until a deterministic conversion-rate service is added.

## Immutable decisions

Every paper execution gate appends a `decision_records` row containing:

- A unique correlation identifier.
- Strategy analysis and market evidence.
- Instrument contract and sizing inputs.
- Quantity and risk checks.
- A concise outcome and summary.

Decision rows are never updated. Order intents reference the decision that authorized or rejected them.

## Intent lifecycle

`order_intents.status` is the current projection. Every change also appends an `order_intent_transitions` row. Invalid transitions fail closed.

Supported terminal states are `rejected`, `executed`, `cancelled`, and `failed`. The transition API is:

```text
GET /api/paper/order-intents/{intent_id}/transitions
```

Decision evidence is available through:

```text
GET /api/decisions
GET /api/instrument-spec?symbol=US100
```

The Workbench displays recent immutable decisions beside order intents and trade audit records.

