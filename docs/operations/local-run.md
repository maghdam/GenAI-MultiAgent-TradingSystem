# Local Run Guide

This guide covers the practical local developer workflow for TradeAgent.

Use this document when you want startup details, environment flags, and troubleshooting without bloating the root `README`.

## Startup Options

### Recommended: one-command startup

From the repo root:

```powershell
cmd /c call start-local.cmd
```

This script attempts to:

- launch Ollama if needed
- start the backend on `127.0.0.1:4000`
- start the frontend on `127.0.0.1:5173`
- open the dashboard

### Manual backend startup

From the repo root:

```powershell
set APP_START_CTRADER_ON_BOOT=1
set APP_WARM_OLLAMA_ON_BOOT=1
set OLLAMA_URL=http://127.0.0.1:11434
set PYTHONPATH=%CD%
C:\Users\mohag\miniconda3\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 4000
```

### Manual frontend startup

```powershell
cd frontend
set VITE_API_BASE=http://127.0.0.1:4000
npm.cmd run dev -- --host 127.0.0.1 --port 5173
```

## Stop Local Processes

From the repo root:

```powershell
cmd /c call stop-local.cmd
```

This script kills listeners on ports `4000` and `5173` and aggressively cleans up common local frontend/backend orphans.

## Useful Environment Flags

### Backend boot flags

- `APP_START_CTRADER_ON_BOOT=1`
  start broker transport at boot
- `APP_WARM_OLLAMA_ON_BOOT=1`
  warm the configured local model path at boot
- `APP_START_CTRADER_ON_BOOT=0`
  useful for tests or offline API work
- `APP_WARM_OLLAMA_ON_BOOT=0`
  useful if you want backend startup without model warmup

### Frontend API target

- `VITE_API_BASE=http://127.0.0.1:4000`
  points the frontend to the local FastAPI server

### cTrader credentials and account selection

Configure the cTrader Open API application credentials and one access token in `backend/.env` (using `backend/.env.example` as the template). The access token is the source of truth for the authorized account directory: TradeAgent discovers every account available to that token and reads the broker-reported Demo/Live type.

Do **not** create separate environment-variable entries for every broker account. `CTRADER_HOST_TYPE` and `CTRADER_ACCOUNT_ID` are bootstrap/fallback values used for initial connection/discovery and older persisted-selection migration. During normal operation, choose the desired account from the System dashboard. TradeAgent persists that account ID/type in SQLite and restores it on restart.

Account switching is deliberately guarded:

- selected account and currently authenticated active account are shown separately
- execution remains blocked while a switch is incomplete
- Demo→Demo and Live→Live can re-authenticate on the same host/client
- Demo↔Live switches replace the cTrader client service without restarting the Twisted reactor
- account switching is blocked while TradeAgent tracks an open broker-backed position
- when the engine is active, stop it or activate the kill switch before switching accounts

> **Live-account warning:** use Demo for development/testing. A Live account can be selected for broker truth without arming real-money entries. New Live orders require `cTrader auto-trade`, per-symbol `Auto-trade`, kill switch off, and the runtime-only **Arm Live Trading** control in System for the currently selected/active Live account. Arming requires the engine to be stopped and resets on account changes, engine restart, and backend restart. Existing broker-backed positions continue through protection/reconciliation/verified-close handling even when Live entry arming is off.

## Runtime SQLite Database

TradeAgent keeps runtime SQLite state outside the repository by default. The active database path is resolved in this order:

1. `TRADEAGENT_DB_PATH`, when explicitly set.
2. Windows: `%LOCALAPPDATA%\TradeAgent\data\tradeagent.db`.
3. Systems with `XDG_STATE_HOME`: `$XDG_STATE_HOME/tradeagent/tradeagent.db`.
4. Other Unix-like systems: `~/.local/state/tradeagent/tradeagent.db`.

The Windows launcher `run-backend-local.cmd` sets the same `%LOCALAPPDATA%` path when no override is already present. Container or other controlled environments may set `TRADEAGENT_DB_PATH` explicitly to a different runtime location.

The old repository-local `backend/data/tradeagent.db` path is **legacy migration input, not the active local-runtime path**. On first database open, if the resolved active database does not yet exist and that legacy database does exist, TradeAgent copies it into the active path using SQLite backup. Existing active databases are never overwritten by this migration.

To inspect the path that the current environment resolves without opening or modifying the database:

```powershell
C:\Users\mohag\miniconda3\envs\tradeagent-v2\python.exe -c "from backend.config import resolve_db_path; print(resolve_db_path())"
```

## Verification Command

From the repository root, run the canonical validator:

```powershell
python scripts\validate.py
```

The script uses the Python interpreter that invokes it. It runs the accepted backend suite first and, only if that passes, the frontend production build. `npm` must be available on `PATH`. A failing backend or frontend step produces a non-zero exit code.

### GitHub CI contract

`.github/workflows/ci.yml` applies the repository validation gates automatically to pull requests targeting `main` and pushes to `main` (and also supports manual `workflow_dispatch` runs):

- backend: Python 3.12, install `.[broker,dev]`, then run `python -m pytest backend/tests -q`
- frontend: Node 22, run `npm ci --prefix frontend`, then run `npm --prefix frontend run test:restart`, `npm --prefix frontend run test:journal`, and `npm --prefix frontend run build`

The two jobs run independently, and any failed command fails its job/workflow. The CI workflow intentionally remains separate from the local `scripts/validate.py` wrapper so backend and frontend jobs can run in parallel; CI extends the local backend-suite/frontend-build core contract with the frontend restart and Trade Journal explainability acceptance tests.

### Optional lint / type-check baseline

The repository already has Ruff configured for Python and ESLint configured for the frontend, but they are **not CI gates yet** because the current baseline contains substantial pre-existing cleanup:

- `ruff check backend`: 304 violations in the current baseline; 114 are reported as automatically fixable.
- `npm --prefix frontend run lint`: 32 problems in the current baseline (28 errors, 4 warnings), mostly explicit-`any` findings plus React hook dependency warnings.
- TypeScript type checking already passes as part of the canonical frontend production build because `npm --prefix frontend run build` starts with `tsc -b`.

A trial CI gate failed both lint jobs before the accepted backend test suite and frontend restart/build could run (CI #199). Keep Ruff/ESLint available for cleanup work, but do not make them required CI gates until the lint baseline is intentionally reduced in a separate scoped change.

## What "working" looks like

Backend health:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:4000/api/health
```

Frontend root:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5173
```

Expected signs:

- backend responds on `/api/health`
- frontend responds on `/`
- dashboard shows a reachable backend
- broker/model status reflects your local environment rather than throwing frontend fetch errors

## Troubleshooting

### Backend does not start

Check:

- Python path in `run-backend-local.cmd`
- whether port `4000` is already occupied
- whether broker startup flags are causing boot-time dependency issues

Try:

```powershell
netstat -ano | Select-String ':4000'
```

### Frontend does not start

Check:

- `npm.cmd` availability
- whether port `5173` is already occupied
- whether `VITE_API_BASE` points to the backend

Try:

```powershell
netstat -ano | Select-String ':5173'
```

### Dashboard loads but API actions fail

Check:

- backend is actually listening on `127.0.0.1:4000`
- frontend is pointing to the same base URL
- CORS origins still match local host usage

### Broker-related status is degraded

That can still be valid local behavior.

Check the System page for selected-versus-active account state, authorization, account verification, symbol metadata, monetary snapshot readiness, and any account-switch error. cTrader execution intentionally fails closed until those prerequisites agree.

The app is designed to boot in constrained environments, and broker/model readiness can be partially unavailable while the API and UI still work for inspection, paper-state review, or Strategy Studio work.

## Documentation Links

- [README.md](../../README.md)
- [ARCHITECTURE.md](../../ARCHITECTURE.md)
