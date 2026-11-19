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
curl.exe --silent --fail --max-time 2 "http://127.0.0.1:4000/docs" >nul 2>&1
if errorlevel 1 (
    start "tradeagent-backend" /min cmd /c call "%ROOT%run-backend-local.cmd"
) else (
    echo [Backend] Already running.
)

echo Starting frontend on http://127.0.0.1:5173
curl.exe --silent --fail --max-time 2 "http://127.0.0.1:5173/" >nul 2>&1
if errorlevel 1 (
    start "tradeagent-frontend" /min cmd /c call "%ROOT%run-frontend-local.cmd"
) else (
    echo [Frontend] Already running.
)

echo.
echo Waiting for frontend readiness...
call :wait_for_url "http://127.0.0.1:5173/" 90
if errorlevel 1 (
    echo.
    echo [ERROR] Frontend did not become ready within 90 seconds.
    echo Review: %RUN_DIR%\frontend.log
    if exist "%RUN_DIR%\frontend.log" type "%RUN_DIR%\frontend.log"
    echo.
    echo This window will close in 20 seconds.
    timeout /t 20 /nobreak >nul
    exit /b 1
)

echo [Frontend] Ready.
echo Waiting briefly for backend readiness...
call :wait_for_url "http://127.0.0.1:4000/docs" 45
if errorlevel 1 (
    echo [WARNING] Backend is not ready yet. The dashboard will open, but data may remain unavailable.
    echo Review: %RUN_DIR%\backend.log
    set "BACKEND_READY=0"
) else (
    echo [Backend] Ready.
    set "BACKEND_READY=1"
)

rem Manually running this local launcher opts in to the loopback-only MCP research
rem gateway by default. Set TRADEAGENT_START_MCP=0 to disable it.
rem This is not an authenticated ChatGPT/Work connector or trading service.
if not defined TRADEAGENT_START_MCP set "TRADEAGENT_START_MCP=1"
if "%TRADEAGENT_START_MCP%"=="1" (
    if "%BACKEND_READY%"=="1" (
        echo Starting read-only local MCP research gateway on 127.0.0.1:8765...
        powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\local-mcp-lifecycle.ps1" -Action Start
        if errorlevel 1 (
            echo [MCP][ERROR] Gateway startup failed or port 8765 belongs to a different process.
            echo Review: %RUN_DIR%\mcp-error.log
            set "MCP_FAILED=1"
        )
    ) else (
        echo [MCP][WARNING] Backend not ready; MCP startup skipped.
        set "MCP_FAILED=1"
    )
) else (
    echo [MCP] Skipped by TRADEAGENT_START_MCP=%TRADEAGENT_START_MCP%.
)

echo Launching Dashboard...
start "" "http://127.0.0.1:5173/"

echo.
echo Logs:
echo   %RUN_DIR%\backend.log
echo   %RUN_DIR%\frontend.log
echo   %RUN_DIR%\mcp-error.log  ^(if MCP enabled^)
echo.
if defined MCP_FAILED (
    echo [WARNING] Dashboard started, but the opted-in MCP gateway is not ready.
    exit /b 1
)
echo System components initialized.
exit /b 0

:wait_for_url
set "WAIT_URL=%~1"
set /a "WAIT_SECONDS=%~2"
for /l %%S in (1,1,%WAIT_SECONDS%) do (
    curl.exe --silent --fail --max-time 2 "%WAIT_URL%" >nul 2>&1
    if not errorlevel 1 exit /b 0
    timeout /t 1 /nobreak >nul
)
exit /b 1
