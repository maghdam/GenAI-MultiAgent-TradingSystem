# ChatGPT TradeAgent MCP gateway — Phase 1 (local research only)

This opt-in proof of concept **does not** change the existing TradeAgent API,
trading runtime, broker selection, or any order-execution path.

It provides only `get_ctrader_candles`, a read-only broker-market-data tool.
The gateway forwards a GET request to TradeAgent's existing
`http://127.0.0.1:4000/api/market/candles`, with the backend providing the
cTrader connection. It does **not** connect Alpaca: use the existing ChatGPT
Alpaca Paper plugin for that during this phase.

## Local test (Windows PowerShell)

1. Pull the feature branch after it is approved. Activate `tradeagent-v2`.
2. Install the optional SDK: `python -m pip install -e ".[mcp]"`.
3. Run isolated tests: `python -m pytest backend/tests/test_chatgpt_mcp_readonly.py -q`.
4. Start the existing TradeAgent backend on `127.0.0.1:4000`, ensure the
   cTrader **Demo** account is selected and connected, and then open another
   PowerShell terminal in the repository root.
5. Start the gateway: `python -m backend.chatgpt_mcp_readonly`.

The local MCP endpoint is `http://127.0.0.1:8765/mcp` (Streamable HTTP).
Do not enter this localhost address as a remote ChatGPT plugin connection:
ChatGPT cannot reach a loopback address on your own machine.

**Security restriction:** Do not deploy or tunnel this unauthenticated POC,
change its bind address to `0.0.0.0`, or expose the existing `/api` service
on the public internet. A later phase must add an authenticated HTTPS MCP
interface with explicit access control and verified read-only tool scope.

No execution or mutation tools are exposed. Quotes may be unavailable or stale;
this tool is for research, not automated trading.
