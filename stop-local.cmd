@echo off
setlocal

echo Stopping trading system processes...

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

echo System stopped successfully.
