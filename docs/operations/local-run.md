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

## Verification Commands

Backend tests:

```powershell
python -m pytest backend\tests -q
```

Frontend production build:

```powershell
cd frontend
npm.cmd run build
```

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

The app is designed to boot in constrained environments, and broker/model readiness can be partially unavailable while the API and UI still work for inspection, paper-state review, or Strategy Studio work.

## Documentation Links

- [README.md](../../README.md)
- [ARCHITECTURE.md](../../ARCHITECTURE.md)
