$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Run scripts\setup-local.ps1 first."
}
$env:APP_START_CTRADER_ON_BOOT = "0"
$env:APP_WARM_OLLAMA_ON_BOOT = "0"
$env:PYTHONPATH = $root
& $python -m pytest backend\tests
Push-Location (Join-Path $root "frontend")
try {
    & npm.cmd ci
    & npm.cmd run build
} finally {
    Pop-Location
}
