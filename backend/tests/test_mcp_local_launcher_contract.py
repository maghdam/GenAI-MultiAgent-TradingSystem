"""Static safeguards for Windows one-command local MCP process lifecycle.

These are cross-platform source-contract tests. Native Windows start/stop
behavior still requires operator smoke validation before the PR is merged.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_start_local_starts_only_loopback_gateway_with_opt_out():
    start = (ROOT / "start-local.cmd").read_text(encoding="utf-8")
    assert 'if not defined TRADEAGENT_START_MCP set "TRADEAGENT_START_MCP=1"' in start
    assert 'if "%TRADEAGENT_START_MCP%"=="1"' in start
    assert 'if "%BACKEND_READY%"=="1"' in start
    assert 'scripts\\local-mcp-lifecycle.ps1" -Action Start' in start
    assert "if defined MCP_FAILED" in start
    assert "backend.chatgpt_mcp_readonly" not in start


def test_stop_local_stops_mcp_before_backend_and_frontend():
    stop = (ROOT / "stop-local.cmd").read_text(encoding="utf-8")
    assert 'scripts\\local-mcp-lifecycle.ps1" -Action Stop' in stop
    assert stop.index('-Action Stop') < stop.index("for %%P in (4000 5173)")
    assert "for %%P in (4000 5173 8765)" not in stop
    assert "if defined MCP_STOP_FAILED" in stop


def test_mcp_lifecycle_refuses_foreign_port_owners():
    script = (ROOT / "scripts/local-mcp-lifecycle.ps1").read_text(encoding="utf-8")
    assert "Get-NetTCPConnection -LocalPort $port -State Listen" in script
    assert '$port = 8765' in script
    assert 'backend.chatgpt_mcp_readonly' in script
    assert 'Get-CimInstance Win32_Process' in script
    assert "Assert-LoopbackOnly" in script
    assert "Test-McpOwner" in script
    assert "Assert-OwnedListener" in script
    assert "Stop-Process -Id $processIdValue" in script
    assert "Stop-Process -Name python" not in script
    assert "taskkill" not in script
    assert '0.0.0.0' not in script


def test_python_launcher_is_consistent_with_backend_environment():
    backend = (ROOT / "run-backend-local.cmd").read_text(encoding="utf-8")
    script = (ROOT / "scripts/local-mcp-lifecycle.ps1").read_text(encoding="utf-8")
    expected = 'C:\\Users\\mohag\\miniconda3\\envs\\tradeagent-v2\\python.exe'
    assert expected in backend
    assert expected in script
    assert 'Start-Process -FilePath $python' in script
    assert ' -WindowStyle Hidden ' in script
    assert '.codex-run' in script
