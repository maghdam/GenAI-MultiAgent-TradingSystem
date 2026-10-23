# Alpaca Paper bracket lifecycle development

The Phase 10.20 implementation is private and supervised. It does not enable automatic trading and has no engine, API, UI or scheduler mutation caller. Use mocked broker responses for development. Actual Paper and Live orders are not authorized by the development task.

`inspect_paper_bracket_exit(intent_id=...)` reads current broker state and persists historical audit evidence and identity bindings. It never mutates the broker. `advance_supervised_paper_bracket_exit(...)` requires the exact confirmation `CANCEL AND FLATTEN PAPER BRACKET <intent_id>` and an operator reason. It performs at most one cancellation or close submission per call, after persisting a single-use claim. These are internal Python services, not operator-facing commands.

## Sequence and recovery

1. Fetch the exact reserved parent with nested legs, the independently reported position, account identity and open-order inventory. Reject unknown ownership, changed identity, replacement, overfill, position mismatch, foreign same-symbol orders or a potentially truncated inventory.
2. Cancel a known outstanding native order. Alpaca may cancel the other group members; reread before deciding anything further. A DELETE 204 response only acknowledges the request.
3. Wait until parent and both exits are explicitly terminal. Missing orders and pending cancellation do not qualify. A partially filled entry may still require cancellation of its unfilled remainder.
4. Reconcile remaining quantity from entry fills minus protective and supervised-close fills. Close only that quantity with an independently reserved client ID, after all original orders are terminal. If already flat, no close is sent.
5. On restart, read the same identities. A claimed action is never retransmitted, even if lookup returns 404. Partial or rejected closes with residual exposure require operator recovery; no second close is synthesized.
6. Flat completion requires independent flat position truth, consistent cumulative fills, terminal parent/SL/TP states and, if a close was claimed, an identified terminal close. This is historical evidence, not release of the original reservation or continuous protection.

Cancellation can remove protection before the subsequent close fills. Account activity outside this workflow can invalidate the quantity snapshot. The separate market exit is not reduce-only. These constraints block unattended readiness and require a separately approved recovery/monitoring design and authorized provider evidence. Kill switch blocks new supervised entries; it does not suppress these explicitly supervised risk-reducing exit steps.

SL/TP resizing is broker-owned: the service verifies actual remaining quantities and never PATCHes quantities or creates replacement stops. Unsupported/missing/replaced protection fails closed and requires manual reconciliation. No symbol-wide or account-wide bulk cancellation is used.

## Durable evidence

- `alpaca_bracket_exit_sessions`: first account/parent/leg/request identity binding.
- `alpaca_bracket_exit_actions`: immutable single-use cancellation and close claims, with exact request, reason and UTC time; observed close identity is also pinned here.
- `trade_audit`: each inspection, claimed action and outcome. A crash after the claim commits leaves the claim consumed even if later audit or HTTP work fails.

Do not delete/reset claims to retry orders. The same database must survive restart; restoring an old database snapshot is not a safe mutation recovery method.

## Validation work log — 2026-10-21

- Synced local main to `b29151c`; developed on `codex/alpaca-bracket-lifecycle`.
- Existing reconciliation/supervised baseline passed outside the Windows sandbox. Initial sandbox run failed at pytest temporary-directory access before test execution; no product failure was inferred.
- Initial lifecycle accounting slice: 19 focused tests passed; committed as `7a7a28d`.
- Initial exit/recovery plus existing Paper execution slice: 37 tests passed.
- Broader focused tests and final canonical validation: results to be recorded after completion.
- No actual broker orders, no automatic activation, no merge, no claim of field verification.

Provider reference: [Alpaca orders documentation](https://docs.alpaca.markets/us/docs/orders-at-alpaca).
