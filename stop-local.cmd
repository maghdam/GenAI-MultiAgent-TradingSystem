@echo off
setlocal

set "ROOT=%~dp0"
echo Stopping trading system processes...

rem Stop only our verified local MCP gateway. Never kill an unrelated listener
rem that happens to be using port 8765.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\local-mcp-lifecycle.ps1" -Action Stop
if errorlevel 1 (
    echo [MCP][WARNING] Could not safely stop MCP. Check port 8765 manually.
    set "MCP_STOP_FAILED=1"
)

rem Kill by ports 4000 (backend) and 5173 (frontend)
for %%P in (4000 5173) do (
  for /f "tokens=5" %%I in ('netstat -ano ^| findstr :%%P ^| findstr LISTENING 2^>nul') do (
    echo [Stop] Killing process PID %%I on port %%P
    taskkill /PID %%I /F >nul 2>&1
  )
)

rem Aggressively kill common orphans
taskkill /f /im node.exe >nul 2>&1
taskkill /f /im uvicorn.exe >nul 2>&1

if defined MCP_STOP_FAILED (
    echo [WARNING] Backend/frontend stopped; MCP shutdown needs inspection.
    exit /b 1
)

echo System stopped successfully.
