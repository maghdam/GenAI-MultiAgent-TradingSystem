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
) else (
    echo [Backend] Ready.
)

echo Launching Dashboard...
start "" "http://127.0.0.1:5173/"

echo.
echo Logs:
echo   %RUN_DIR%\backend.log
echo   %RUN_DIR%\frontend.log
echo.
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
