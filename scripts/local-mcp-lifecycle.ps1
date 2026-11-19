<#
Lifecycle of the opt-in local TradeAgent MCP research gateway.
This script manages only the expected Python module on loopback port 8765.
It must never stop a different process that happens to own that port.
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Start", "Stop")]
    [string]$Action
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = "C:\Users\mohag\miniconda3\envs\tradeagent-v2\python.exe"
$module = "backend.chatgpt_mcp_readonly"
$port = 8765
$runDir = Join-Path $root ".codex-run"

function Get-McpListener {
    @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
}

function Assert-LoopbackOnly($listeners) {
    foreach ($listener in $listeners) {
        if ($listener.LocalAddress -notin @("127.0.0.1", "::1")) {
            throw "[MCP] Port $port is listening beyond loopback. Refusing lifecycle operation."
        }
    }
}

function Test-McpOwner([int]$ProcessId) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction Stop
    if (-not $process) { return $false }

    $expectedModule = '(?i)(?:^|\s)-m\s+backend\.chatgpt_mcp_readonly(?:\s|$)'
    return (
        [string]$process.ExecutablePath -ieq $python -and
        [string]$process.CommandLine -match $expectedModule
    )
}

function Assert-OwnedListener($listeners) {
    Assert-LoopbackOnly $listeners
    $ids = @($listeners | Select-Object -ExpandProperty OwningProcess -Unique)
    foreach ($processIdValue in $ids) {
        if (-not (Test-McpOwner ([int]$processIdValue))) {
            throw "[MCP] Port $port is occupied by PID $processIdValue, not our verified gateway. Refusing."
        }
    }
    return $ids
}

if ($Action -eq "Stop") {
    $listeners = @(Get-McpListener)
    if ($listeners.Count -eq 0) {
        Write-Host "[MCP] Not running on port $port."
        exit 0
    }
    $ownedPids = @(Assert-OwnedListener $listeners)
    foreach ($processIdValue in $ownedPids) {
        Write-Host "[MCP] Stopping verified gateway PID $processIdValue."
        Stop-Process -Id $processIdValue -Force -ErrorAction Stop
    }

    for ($i = 0; $i -lt 20; $i++) {
        if (@(Get-McpListener).Count -eq 0) {
            Write-Host "[MCP] Stopped; port $port is free."
            exit 0
        }
        Start-Sleep -Milliseconds 250
    }
    throw "[MCP] Port $port still has a listener after stopping the verified gateway."
}

# Start is only invoked from the operator's manually run start-local.cmd.
if (-not (Test-Path -LiteralPath $python)) {
    throw "[MCP] Python environment missing: $python"
}
$existing = @(Get-McpListener)
if ($existing.Count -gt 0) {
    $ownedPids = @(Assert-OwnedListener $existing)
    Write-Host "[MCP] Already running on loopback port $port (PID $($ownedPids -join ', '))."
    exit 0
}

& $python -c "from mcp.server.mcpserver import MCPServer" 1>$null
if ($LASTEXITCODE -ne 0) {
    throw '[MCP] Optional MCP SDK not installed; run python -m pip install -e ".[mcp]" in tradeagent-v2.'
}
if (-not (Test-Path -LiteralPath $runDir)) {
    New-Item -ItemType Directory -Path $runDir | Out-Null
}
$stdoutLog = Join-Path $runDir "mcp.log"
$stderrLog = Join-Path $runDir "mcp-error.log"

$started = Start-Process -FilePath $python -ArgumentList @("-m", $module) `
    -WorkingDirectory $root -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru

for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Milliseconds 500
    $listeners = @(Get-McpListener)
    if ($listeners.Count -gt 0) {
        Assert-LoopbackOnly $listeners
        $ids = @($listeners | Select-Object -ExpandProperty OwningProcess -Unique)
        if ($ids.Count -ne 1 -or [int]$ids[0] -ne $started.Id) {
            throw "[MCP] Unexpected listener owns port $port; inspect $stderrLog."
        }
        if (-not (Test-McpOwner $started.Id)) {
            throw "[MCP] Listener command line does not match the expected research gateway."
        }
        Write-Host "[MCP] Ready at http://127.0.0.1:$port/mcp (PID $($started.Id))."
        Write-Host "[MCP] Logs: $stdoutLog and $stderrLog"
        exit 0
    }
    if ($started.HasExited) { break }
}

throw "[MCP] Did not become ready on loopback port $port. Check $stderrLog."
