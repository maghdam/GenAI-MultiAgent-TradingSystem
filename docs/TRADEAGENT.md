# TradeAgent Docs

This file is kept as a lightweight documentation index for older links and bookmarks.

For the current documentation structure, use:

- [README.md](../README.md)
  public-facing project overview, screenshots, quick start, verification, and high-level agent summary
- [ARCHITECTURE.md](../ARCHITECTURE.md)
  current system architecture, runtime flow, Strategy Studio flow, diagrams, and active boundaries
- [operations/local-run.md](operations/local-run.md)
  local startup, environment flags, verification commands, and troubleshooting

## Current Documentation Policy

- `README.md` is the front door for GitHub visitors and employers.
- `ARCHITECTURE.md` is the technical source of truth.
- historical or overlapping architecture notes should be merged into `ARCHITECTURE.md` rather than duplicated here.

## Current Repo Shape

TradeAgent now runs on one active consolidated stack:

- `backend/`
- `frontend/`

The autonomous runtime supports local paper execution and explicitly enabled cTrader execution on the selected authenticated Demo or Live account. The access token supplies the authorized account directory, the System dashboard persists the operator's account selection, and the same broker-readiness/risk/protection/reconciliation controls apply to both account types. Demo is recommended for development and testing; enabling cTrader auto-trade on a selected Live account can place real-money orders. Strategy Studio remains a separate research workflow for drafting and backtesting strategies.


## Historical Planning Documents

The following root-level files are intentionally retained for provenance only and are not current architecture or operational guidance:

- [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) — March 2026 backtesting / Strategy Studio upgrade plan using superseded API and local-path assumptions.
- [STRATEGY_INTEGRATION_PLAN.md](../STRATEGY_INTEGRATION_PLAN.md) — October 2025 standalone Strategy Studio integration plan using retired routes.

For current architecture, always use [ARCHITECTURE.md](../ARCHITECTURE.md).
