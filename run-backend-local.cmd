@echo off
setlocal

set "ROOT=%~dp0"
set "RUN_DIR=%ROOT%.codex-run"
if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"

cd /d "%ROOT%"

set APP_START_CTRADER_ON_BOOT=1
set APP_WARM_OLLAMA_ON_BOOT=1
set OLLAMA_URL=http://127.0.0.1:11434
set OLLAMA_MODEL=phi3:mini
set OLLAMA_FALLBACK_MODEL=muse-glimmer:latest
set OLLAMA_THINK=off
set STUDIO_OLLAMA_THINK=low
set STUDIO_OLLAMA_MODEL=muse-glimmer:latest
set PYTHONPATH=%ROOT%

rem Keep the live SQLite runtime outside Google Drive / other sync folders.
rem Callers may override TRADEAGENT_DB_PATH before launching.
if not defined TRADEAGENT_DB_PATH (
    if defined LOCALAPPDATA (
        set "TRADEAGENT_DB_PATH=%LOCALAPPDATA%\TradeAgent\data\tradeagent.db"
    ) else (
        set "TRADEAGENT_DB_PATH=%ROOT%backend\data\tradeagent.db"
    )
)

set "PYTHON=C:\Users\mohag\miniconda3\envs\tradeagent-v2\python.exe"

if not exist "%PYTHON%" (
    >"%RUN_DIR%\backend.log" echo [ERROR] Conda environment tradeagent-v2 was not found:
    >>"%RUN_DIR%\backend.log" echo %PYTHON%
    exit /b 1
)

"%PYTHON%" -m uvicorn backend.app:app --host 127.0.0.1 --port 4000 1>"%RUN_DIR%\backend.log" 2>&1
