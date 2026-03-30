@echo off
setlocal

set "ROOT=%~dp0"
set "RUN_DIR=%ROOT%.codex-run"
if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"

cd /d "%ROOT%"
set APP_START_CTRADER_ON_BOOT=1
set APP_WARM_OLLAMA_ON_BOOT=1
set OLLAMA_URL=http://127.0.0.1:11434
set PYTHONPATH=%ROOT%

"C:\Users\mohag\miniconda3\python.exe" -m uvicorn backend.app:app --host 127.0.0.1 --port 4000 1>"%RUN_DIR%\backend.log" 2>&1
