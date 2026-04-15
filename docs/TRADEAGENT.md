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

The autonomous runtime is the paper-only V2 engine. Strategy Studio remains a separate research workflow for drafting and backtesting strategies.
