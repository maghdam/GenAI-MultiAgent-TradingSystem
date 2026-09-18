param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root ".venv"

& $Python -c "import sys; assert (3, 11) <= sys.version_info[:2] < (3, 15), sys.version"
if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
    & $Python -m venv $venv
}
$venvPython = Join-Path $venv "Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -e "$root[dev,broker,research]"
Write-Host "TradeAgent environment is ready: $venvPython" -ForegroundColor Green
