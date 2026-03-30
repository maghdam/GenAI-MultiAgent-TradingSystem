@echo off
setlocal

set "ROOT=%~dp0"
set "RUN_DIR=%ROOT%.codex-run"
if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"

echo Checking Ollama...
tasklist /fi "imagename eq ollama.exe" | find ":" > nul
if errorlevel 1 (
    echo [Ollama] Already running.
) else (
    echo [Ollama] Launching app...
    start "" "C:\Users\mohag\AppData\Local\Programs\Ollama\ollama app.exe"
    timeout /t 5 /nobreak > nul
)

echo Starting backend on http://127.0.0.1:4000
start "tradeagent-backend" /min cmd /c call "%ROOT%run-backend-local.cmd"
timeout /t 3 /nobreak > nul

echo Starting frontend on http://127.0.0.1:5173
start "tradeagent-frontend" /min cmd /c call "%ROOT%run-frontend-local.cmd"

echo.
echo Launching Dashboard...
timeout /t 10 /nobreak > nul
start http://localhost:5173

echo.
echo Logs:
echo   %RUN_DIR%\backend.log
echo   %RUN_DIR%\frontend.log
echo.
echo System components initialized.
