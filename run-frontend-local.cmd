@echo off
setlocal

set "ROOT=%~dp0"
set "RUN_DIR=%ROOT%.codex-run"
if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"

cd /d "%ROOT%frontend"
set "VITE_API_BASE=http://127.0.0.1:4000"
call npm.cmd run dev -- --host 127.0.0.1 --port 5173
