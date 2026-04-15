param(
    [string]$ApiBase = 'http://127.0.0.1:4000',
    [string]$AppBase = 'http://127.0.0.1:5173',
    [string]$ProfileDir = '',
    [int]$DebugPort = 9222
)

$ErrorActionPreference = 'Stop'

if (-not $ProfileDir) {
    $ProfileDir = Join-Path (Get-Location) '.codex-run\edge-shot-profile'
}

$edgePath = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
if (-not (Test-Path $edgePath)) {
    throw "Edge not found at $edgePath"
}

New-Item -ItemType Directory -Force -Path $ProfileDir | Out-Null

$body = @{
    task_type = 'backtest'
    goal = 'backtest smc on XAUUSD M5'
    params = @{
        strategy_name = 'smc'
        symbol = 'XAUUSD'
        timeframe = 'M5'
        num_bars = 1500
    }
} | ConvertTo-Json -Depth 8

$response = Invoke-RestMethod -UseBasicParsing -Method Post -Uri "$ApiBase/api/studio/tasks" -ContentType 'application/json' -Body $body
if ($response.status -ne 'success' -or -not $response.result) {
    throw "Studio backtest did not return a usable result."
}

$result = $response.result
$compact = [ordered]@{
    metrics = $result.metrics
    equity = @($result.equity | Select-Object -First 160)
    trades = @($result.trades | Select-Object -First 24)
}

if ($result.candles) {
    $compact.candles = @($result.candles | Select-Object -First 180)
}

$meta = [ordered]@{
    ts = [int64]([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds())
    strategy = 'smc'
    symbol = 'XAUUSD'
    timeframe = 'M5'
    numBars = 1500
}

Start-Process -FilePath $edgePath -ArgumentList @(
    "--remote-debugging-port=$DebugPort",
    "--user-data-dir=$ProfileDir",
    '--new-window',
    '--start-maximized',
    "--app=$AppBase/strategy-studio/results"
) | Out-Null

$targets = $null
for ($i = 0; $i -lt 50; $i++) {
    Start-Sleep -Milliseconds 500
    try {
        $targets = Invoke-RestMethod -UseBasicParsing -Uri "http://127.0.0.1:$DebugPort/json/list"
        if ($targets) { break }
    } catch {
    }
}

if (-not $targets) {
    throw "Unable to reach Edge DevTools on port $DebugPort."
}

$target = $targets | Where-Object {
    $_.type -eq 'page' -and $_.url -like "$AppBase/strategy-studio*"
} | Select-Object -First 1

if (-not $target) {
    throw "Unable to find the Strategy Studio target in DevTools."
}

$ws = [System.Net.WebSockets.ClientWebSocket]::new()
$uri = [Uri]$target.webSocketDebuggerUrl
$cts = [Threading.CancellationTokenSource]::new()
$ws.ConnectAsync($uri, $cts.Token).GetAwaiter().GetResult()

function Send-CdpMessage {
    param(
        [System.Net.WebSockets.ClientWebSocket]$Socket,
        [int]$Id,
        [string]$Method,
        [hashtable]$Params = @{}
    )

    $payload = @{
        id = $Id
        method = $Method
        params = $Params
    } | ConvertTo-Json -Depth 20 -Compress

    $buffer = [Text.Encoding]::UTF8.GetBytes($payload)
    $segment = [ArraySegment[byte]]::new($buffer)
    $null = $Socket.SendAsync($segment, [System.Net.WebSockets.WebSocketMessageType]::Text, $true, $cts.Token).GetAwaiter().GetResult()
}

function Receive-CdpMessage {
    param([System.Net.WebSockets.ClientWebSocket]$Socket)

    $buffer = New-Object byte[] 262144
    $stream = New-Object System.IO.MemoryStream

    do {
        $segment = [ArraySegment[byte]]::new($buffer)
        $result = $Socket.ReceiveAsync($segment, $cts.Token).GetAwaiter().GetResult()
        if ($result.Count -gt 0) {
            $stream.Write($buffer, 0, $result.Count)
        }
    } while (-not $result.EndOfMessage)

    [Text.Encoding]::UTF8.GetString($stream.ToArray())
}

$resultJson = ($compact | ConvertTo-Json -Depth 20 -Compress)
$metaJson = ($meta | ConvertTo-Json -Depth 10 -Compress)

$resultB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($resultJson))
$metaB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($metaJson))

$expression = @"
(() => {
  const result = JSON.parse(atob('$resultB64'));
  const meta = JSON.parse(atob('$metaB64'));
  localStorage.setItem('strategyStudio.backtest.lastResult', JSON.stringify(result));
  localStorage.setItem('strategyStudio.backtest.lastMeta', JSON.stringify(meta));
  localStorage.setItem('strategyStudio.layout.mode', 'stack');
  localStorage.setItem('strategyStudio.layout.userSet', '1');
  location.href = '$AppBase/strategy-studio/results';
  return 'ok';
})()
"@

Send-CdpMessage -Socket $ws -Id 1 -Method 'Runtime.enable'
$null = Receive-CdpMessage -Socket $ws
Send-CdpMessage -Socket $ws -Id 2 -Method 'Runtime.evaluate' -Params @{
    expression = $expression
    awaitPromise = $true
}
$null = Receive-CdpMessage -Socket $ws

Start-Sleep -Seconds 2

if ($ws.State -eq [System.Net.WebSockets.WebSocketState]::Open) {
    try {
        $null = $ws.CloseOutputAsync([System.Net.WebSockets.WebSocketCloseStatus]::NormalClosure, 'done', $cts.Token).GetAwaiter().GetResult()
    } catch {
    }
}

Write-Output "$AppBase/strategy-studio/results"
