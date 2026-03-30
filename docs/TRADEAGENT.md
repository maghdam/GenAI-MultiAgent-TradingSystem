# TradeAgent

TradeAgent now uses a single consolidated backend. The old parallel `backend_v2` package and the legacy execution stack have been removed from the active repo shape.

## Goals

- deterministic live-path strategies
- explicit risk and operator controls
- persistent config, incidents, and analysis history
- cleaner API boundaries
- paper-first rollout

## Current Scope

TradeAgent is exposed under `/api/*` and currently provides:

- health and engine status
- persistent operator config
- incident log
- deterministic strategy analysis
- recent analysis history
- broker status and position snapshots
- a background paper-trading loop
- persistent paper positions and engine events
- operator actions for start, stop, and manual scans
- startup runtime recovery and manual reconciliation

## Structure

- `backend/domain`
  - typed request/response and engine models
- `backend/storage`
  - SQLite-backed config, incidents, analyses, paper positions, intents, and audit records
- `backend/services`
  - broker, market data, readiness, engine, reconciliation, studio, and execution helpers
- `backend/strategies`
  - deterministic strategy implementations and registry
- `backend/api`
  - primary FastAPI router for the app

## Deterministic Strategies

The initial deterministic strategies are:

- `sma_cross`
- `rsi_reversal`
- `breakout`

These are intended as clean, testable building blocks. Strategy Studio and saved generated strategies still exist as research tooling, but autonomous paper execution is now built around the deterministic engine path.

## Current Boundaries

1. The repo now has one active backend package: `backend/`.
2. The main app surface, Strategy Studio flow, checklist flow, and operator controls all run through the consolidated API.
3. Execution remains paper-only by design.
4. Live trading should only be enabled after a separate broker-intent and reconciliation hardening pass.
