"""Contract checks for the draft *local-only* TradeAgent research plugin package.

These tests validate package structure and boundaries; they do NOT establish
that a particular ChatGPT Desktop installation has loaded the plugin.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "plugins" / "tradeagent-research"
MARKETPLACE = ROOT / ".agents" / "plugins" / "marketplace.json"
SKILL = PLUGIN / "skills" / "daily-desk" / "SKILL.md"
LOCAL_MCP_URL = "http://127.0.0.1:8765/mcp"
ALLOWED_TOOL_NAMES = (
    "get_ctrader_candles",
    "list_studio_saved_strategies",
    "get_studio_recorded_evidence",
    "inspect_studio_saved_strategy",
    "validate_studio_strategy_draft",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_plugin_contains_only_reviewed_portable_components() -> None:
    found = {
        path.relative_to(PLUGIN).as_posix()
        for path in PLUGIN.rglob("*")
        if path.is_file()
    }
    assert found == {"plugin.json", "mcp.json", "skills/daily-desk/SKILL.md"}


def test_portable_manifest_is_private_local_read_only() -> None:
    payload = load_json(PLUGIN / "plugin.json")
    assert payload["$schema"] == "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
    assert payload["name"] == "tradeagent-research"
    assert payload["version"] == "0.1.2"
    ext = payload["extensions"]["com.openai"]
    assert ext["requires_local_executor"] is True
    assert ext["interface"]["capabilities"] == ["Read"]
    assert "TradeAgent" in ext["interface"]["displayName"]


def test_mcp_manifest_has_exact_one_local_research_server() -> None:
    payload = load_json(PLUGIN / "mcp.json")
    assert payload["$schema"] == "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json"
    assert set(payload["mcpServers"]) == {"tradeagent-local-readonly"}
    server = payload["mcpServers"]["tradeagent-local-readonly"]
    assert server["type"] == "streamable-http"
    assert server["url"] == LOCAL_MCP_URL
    # Codex 0.171.0-alpha.13 rejects server-level extensions before attaching
    # tools ("unknown field `extensions`, expected `url` or `headers`").
    # The loopback gateway needs no auth metadata or credential headers.
    assert set(server) == {"type", "url"}
    assert not any(key in server for key in ("command", "args", "headers", "env"))


def test_marketplace_points_only_to_this_repo_local_package() -> None:
    payload = load_json(MARKETPLACE)
    assert payload["name"] == "tradeagent-local"
    assert len(payload["plugins"]) == 1
    entry = payload["plugins"][0]
    assert entry["name"] == load_json(PLUGIN / "plugin.json")["name"]
    assert entry["source"] == {"source": "local", "path": "./plugins/tradeagent-research"}
    assert entry["policy"] == {"installation": "AVAILABLE", "authentication": "ON_INSTALL"}
    assert entry["category"] == "Productivity"

    # Resolve relative to repository root, not .agents/plugins.
    target = (ROOT / entry["source"]["path"]).resolve()
    assert target == PLUGIN.resolve()
    assert target.is_relative_to(ROOT.resolve())


@pytest.mark.parametrize("tool", ALLOWED_TOOL_NAMES)
def test_skill_includes_existing_read_only_tool_names(tool: str) -> None:
    skill = SKILL.read_text(encoding="utf-8")
    assert f"`{tool}`" in skill


def test_skill_has_explicit_no_trade_provenance_and_risk_gates() -> None:
    skill = SKILL.read_text(encoding="utf-8")
    assert skill.startswith("---\nname: tradeagent-daily-desk\n")
    assert "description:" in skill.split("---", 2)[1]
    for required in (
        "NO TRADE",
        "NO LIVE DATA",
        "US30",
        "XAUUSD",
        "US500",
        "M15",
        "M5",
        "freshness",
        "unverified",
        "operator",
        "No live or paper-trading action",
        "Never invoke broker order/cancel/close",
        "Never expose the localhost server",
    ):
        assert required in skill


def test_existing_gateway_keeps_exact_five_read_only_tools() -> None:
    from backend.chatgpt_mcp_readonly import create_server

    server = create_server()
    names = [tool.name for tool in asyncio.run(server.list_tools())]
    assert names == list(ALLOWED_TOOL_NAMES)
