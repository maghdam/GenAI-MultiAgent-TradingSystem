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

## Phase 10.21 — operator-started read-only protection monitor (not yet validated)

The monitored scope is accepted Alpaca **Paper US-equity native bracket** intents; unsubmitted reservations never become assumed broker exposure. Each explicitly started poll uses the Phase 10.20 fresh account/order/leg/position inspector and existing durable account/leg identity binding. It classifies protection only *at the observed instant*. Unknown/missing/changed broker state, partial-entry exposure, unprotected remaining positions, slow reads and excessive monitoring scope require operator attention. A snapshot of an apparently protected or flat position does **not** clear a prior incident, close an order intent or authorize a new trade.

Durable alpaca_bracket_monitor_states records the last classification to deduplicate identical incidents across restart. The existing System incidents list receives a new critical/warning event on changes to an unsafe classification and a separate warning when a previously unsafe snapshot later looks better; **only an operator** can review the original incident. A monitor heartbeat records completed sweeps and raises a restart/missed-poll warning when a later sweep detects a gap. It cannot detect its own failure or alert while the process is stopped.

**No broker field queries are authorized by mock test validation.** After separately authorizing READ-ONLY Alpaca Paper queries, the operator may explicitly start the CLI with:

~~~powershell
python -m scripts.monitor_alpaca_paper_brackets --once --acknowledge-paper-reads
python -m scripts.monitor_alpaca_paper_brackets --continuous --interval 30 --acknowledge-paper-reads
~~~

Start manually after system restart and stop with Ctrl+C. This CLI is **not** started on import, backend boot, engine start or scheduled trades; there is no webhook, push notification, guaranteed unattended watchdog or automatic restoration after a restart. HTTP calls use the existing bounded per-request client timeout; a late aggregate sweep is flagged but cannot forcibly interrupt a hung network client. Full-account orphan exposure outside accepted local intents is **not** yet surveyed. More than the configured intent cap fails closed with a durable critical incident, rather than checking just the most recent 200 intents.

The code never sends an order POST, DELETE, replace, flatten, recovery or broker retry and cannot enable Alpaca Paper/crypto automatic entries or Alpaca Live. Keep the existing Phase 10.20 cancellation-to-close exposure and non-reduce-only limitations open. Until operational liveness, authorized provider field testing, account-wide reconciliation, response escalation and completion accounting are separately proven, the monitor is **not a production trading safety layer**.

## Phase 10.22 — independent local heartbeat watchdog (validation pending)

**Scope:** a separate, explicitly started Windows/Python process checks **only the local SQLite completed-monitor heartbeat** produced by Phase 10.21. This can run without broker credentials or approval for Paper field queries because it performs no Alpaca HTTP requests. It never starts the broker monitor itself. It never places, cancels, replaces or closes a broker order.

From the project root, after selecting the tradeagent-v2 environment:

~~~powershell
python -m scripts.watch_alpaca_monitor --once
# Optional second *manually started* console, not a Windows service:
python -m scripts.watch_alpaca_monitor --continuous --interval 30 --stale-after 120
~~~

A one-shot command exits **0** if a recent completed sweep exists with a recognized non-alert status, **2** if the completed sweep is missing, stale, invalid, future-dated or itself reports an unsafe status, and **3** if the local DB check fails. In all cases the console output explicitly says a recent completed sweep is **not** proof of continuously running monitoring or broker-side protection. The result and a short operator message may be observed in a terminal or by a **separately configured** external supervisor. This PR does not create a scheduled task, webhook, email or push delivery.

A dedicated one-row SQLite transition state deduplicates repeated critical incidents. New unsafe states are saved in the existing System incidents log as level error, retaining monitor_severity=critical; later improvement records an additional warning and **does not acknowledge** old incidents. Clock skew, missing heartbeat, and missed intervals fail closed. A fresh completed sweep reporting attention_required or scope_exceeded_unverified also triggers a watchdog alert.

**Important limitations:** this watchdog process can also stop, lose DB access or be unscheduled. A stopped watchdog cannot generate alerts; stdout/stderr and local incident storage are **not** an independently delivered notification. A recent completion timestamp proves neither broker account-wide exposure completeness nor operational SLA. Do not enable unattended broker trading based on this watchdog. Later roadmap work must separately establish OS-supervised process liveness, reliable external operator alert delivery, account-wide orphan detection, authorized Paper field observations, and broker failure/close safety.
