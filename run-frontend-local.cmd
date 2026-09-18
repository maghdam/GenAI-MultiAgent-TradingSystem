@echo off
setlocal

set "ROOT=%~dp0"
set "RUN_DIR=%ROOT%.codex-run"
if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"

cd /d "%ROOT%frontend"
set "VITE_API_BASE=http://127.0.0.1:4000"
echo [Frontend] Starting Vite directly with Node... 1>"%RUN_DIR%\frontend.log"
node "%ROOT%frontend\node_modules\vite\bin\vite.js" --host 127.0.0.1 --port 5173 1>>"%RUN_DIR%\frontend.log" 2>&1
