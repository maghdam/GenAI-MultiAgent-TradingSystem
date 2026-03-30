# TradeAgent

TradeAgent is a local-first trading workstation with a single consolidated backend and a React frontend.

The repo now runs on one active backend package: [`backend/`](C:/Users/mohag/My%20Drive/Github/GenAI-MultiAgent-TradingSystem%20-%202/backend). The old parallel `backend_v2` package and the legacy agent API layer have been removed from the active code path.

## What It Does

- deterministic strategy analysis for manual and automated paper workflows
- persistent operator config, incidents, analyses, order intents, and paper-trade audit
- cTrader-backed market data and broker status
- Strategy Studio for research, code generation, saved strategies, and backtests
- checklist and operator workbench pages in the frontend

## Current Execution Boundary

- autonomous execution is paper-only
- live mode requests are persisted and surfaced, but live execution is still intentionally disabled
- risk checks now include confidence thresholds, stale-bar guards, malformed-bar rejection, stop/target geometry checks, sizing limits, cooldowns, and daily loss controls

## Repo Shape

- [`backend/app.py`](C:/Users/mohag/My%20Drive/Github/GenAI-MultiAgent-TradingSystem%20-%202/backend/app.py)
  - FastAPI composition root
- [`backend/api/router.py`](C:/Users/mohag/My%20Drive/Github/GenAI-MultiAgent-TradingSystem%20-%202/backend/api/router.py)
  - main `/api/*` surface
- [`backend/services/`](C:/Users/mohag/My%20Drive/Github/GenAI-MultiAgent-TradingSystem%20-%202/backend/services)
  - engine, execution, reconciliation, broker/model/checklist/studio services
- [`backend/storage/`](C:/Users/mohag/My%20Drive/Github/GenAI-MultiAgent-TradingSystem%20-%202/backend/storage)
  - SQLite-backed persistence
- [`backend/strategies/`](C:/Users/mohag/My%20Drive/Github/GenAI-MultiAgent-TradingSystem%20-%202/backend/strategies)
  - deterministic strategy registry
- [`backend/strategies_generated/`](C:/Users/mohag/My%20Drive/Github/GenAI-MultiAgent-TradingSystem%20-%202/backend/strategies_generated)
  - saved research strategies used by Strategy Studio
- [`frontend/src/`](C:/Users/mohag/My%20Drive/Github/GenAI-MultiAgent-TradingSystem%20-%202/frontend/src)
  - dashboard, workbench, checklist, and Strategy Studio

## Main API Surface

All active endpoints are under `/api`.

- `GET /api/health`
- `GET /api/llm_status`
- `GET /api/status`
- `GET|POST /api/config`
- `GET /api/symbols`
- `GET /api/market/candles`
- `POST /api/analyze`
- `POST /api/orders/manual`
- `POST /api/engine/start`
- `POST /api/engine/stop`
- `POST /api/engine/scan`
- `POST /api/engine/reconcile`
- `POST /api/engine/recover`
- `GET /api/studio/strategy-files`
- `GET /api/studio/backtest`
- `POST /api/studio/tasks`

## Frontend Routes

- `/`
  - main dashboard
- `/workbench`
  - operator workbench
- `/heavyweight-checklist`
  - checklist view
- `/strategy-studio`
  - Strategy Studio

`/v2` is kept as a compatibility route to the same workbench page.

## Local Verification

The current consolidated repo was validated with:

```powershell
python -m compileall backend
C:\Users\mohag\miniconda3\python.exe -m pytest backend\tests -q -p no:cacheprovider
npm.cmd run build
```

Result:

- backend compile passed
- `40` backend tests passed
- frontend production build passed

## Notes

- The remaining root `pytest-cache-files-*` directories are OS-protected temp folders that could not be removed from this session even with elevated deletion attempts.
- Some older planning docs still exist in the repo for historical context, but the active implementation is the consolidated backend and frontend described above.
