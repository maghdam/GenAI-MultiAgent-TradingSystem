"""§36 private native proposal plugin packaging contract; no host install or secrets."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from backend.chatgpt_draft_proposal_mcp import create_server as proposal_server
from backend.chatgpt_mcp_readonly import create_server as research_server


ROOT = Path(__file__).resolve().parents[2]
PROPOSAL = ROOT / "plugins" / "tradeagent-proposal"
RESEARCH = ROOT / "plugins" / "tradeagent-research"
MARKETPLACE = ROOT / ".agents" / "plugins" / "marketplace.json"
SKILL = PROPOSAL / "skills" / "proposal-review" / "SKILL.md"
PROPOSAL_URL = "http://127.0.0.1:8766/mcp"
RESEARCH_URL = "http://127.0.0.1:8765/mcp"
READ_TOOLS = {
    "get_ctrader_candles",
    "list_studio_saved_strategies",
    "get_studio_recorded_evidence",
    "inspect_studio_saved_strategy",
    "validate_studio_strategy_draft",
}
PROPOSE_TOOL = "propose_studio_draft_for_review"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_proposal_plugin_contains_only_reviewed_portable_components():
    found = {
        item.relative_to(PROPOSAL).as_posix()
        for item in PROPOSAL.rglob("*") if item.is_file()
    }
    assert found == {
        "plugin.json", "mcp.json", "skills/proposal-review/SKILL.md"
    }


def test_plugin_manifest_advertises_distinct_metadata_write_and_local_requirement():
    manifest = _json(PROPOSAL / "plugin.json")
    assert manifest["$schema"] == "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
    assert manifest["name"] == "tradeagent-proposal"
    assert manifest["version"] == "0.1.0"
    extension = manifest["extensions"]["com.openai"]
    assert extension["requires_local_executor"] is True
    interface = extension["interface"]
    assert interface["capabilities"] == ["Write"]
    assert "proposal" in interface["displayName"].lower()
    assert "pending" in interface["longDescription"].lower()
    assert "cannot save" in interface["longDescription"].lower()
    assert "tradeagent-research" not in json.dumps(manifest).lower()


def test_only_one_bounded_loopback_mcp_server_no_headers_or_secrets():
    payload = _json(PROPOSAL / "mcp.json")
    assert payload["$schema"] == "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json"
    assert set(payload) == {"$schema", "mcpServers"}
    assert set(payload["mcpServers"]) == {"tradeagent-local-proposal"}
    server = payload["mcpServers"]["tradeagent-local-proposal"]
    assert server == {"type": "streamable-http", "url": PROPOSAL_URL}
    assert server["url"] != RESEARCH_URL
    assert not any(key in server for key in (
        "headers", "authorization", "env", "command", "args", "extensions",
    ))
    # The portable package must not contain a literal token or a fake
    # environment-variable substitution that the host might forward verbatim.
    assert "${" not in json.dumps(payload)
    assert "TRADEAGENT_PROPOSAL_MCP_TOKEN" not in json.dumps(payload)


def test_marketplace_has_two_separate_opt_in_local_sources():
    payload = _json(MARKETPLACE)
    assert payload["name"] == "tradeagent-local"
    assert set(item["name"] for item in payload["plugins"]) == {
        "tradeagent-research", "tradeagent-proposal"
    }
    for item in payload["plugins"]:
        assert item["source"] == {
            "source": "local", "path": "./plugins/" + item["name"]
        }
        assert item["policy"] == {
            "installation": "AVAILABLE", "authentication": "ON_INSTALL"
        }
        assert item["category"] == "Productivity"
        target = (ROOT / item["source"]["path"]).resolve()
        assert target == (ROOT / "plugins" / item["name"]).resolve()
        assert target.is_relative_to(ROOT.resolve())
    research = _json(RESEARCH / "plugin.json")
    assert research["extensions"]["com.openai"]["interface"]["capabilities"] == ["Read"]


def test_skill_prohibits_unapproved_saving_and_default_trade_actions():
    contents = SKILL.read_text(encoding="utf-8")
    assert contents.startswith("---\nname: tradeagent-proposal-review\n")
    for required in (
        PROPOSE_TOOL, "PENDING", "NO SAVE", "NO TRADE",
        "127.0.0.1:8766/mcp", "8765",
        "human", "terminal", "NO SAVE",
    ):
        assert required.lower() in contents.lower()
    assert "Never submit source containing keys" in contents
    assert "Do not bypass authentication" in contents


def test_proposal_and_read_gateway_expose_disjoint_fixed_tool_allowlists():
    proposed = {
        t.name for t in asyncio.run(proposal_server().list_tools())
    }
    research = {
        t.name for t in asyncio.run(research_server().list_tools())
    }
    assert proposed == {PROPOSE_TOOL}
    assert research == READ_TOOLS
    assert proposed.isdisjoint(research)
    assert "create_from_approved_local_request" not in proposed


def test_normal_windows_launcher_never_starts_proposal_server():
    start = (ROOT / "start-local.cmd").read_text(encoding="utf-8")
    stop = (ROOT / "stop-local.cmd").read_text(encoding="utf-8")
    lifecycle = (ROOT / "scripts" / "local-mcp-lifecycle.ps1").read_text(
        encoding="utf-8"
    )
    for text in (start, stop, lifecycle):
        assert "backend.chatgpt_draft_proposal_mcp" not in text
        assert "8766" not in text
    assert "backend.chatgpt_mcp_readonly" in lifecycle
