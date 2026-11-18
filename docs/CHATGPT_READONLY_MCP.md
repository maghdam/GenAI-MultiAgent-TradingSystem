# TradeAgent read-only MCP research gateway (opt-in, LOCAL ONLY)

This module is the first **TradeAgent platform** research integration, not a
second SMC strategy engine. It is a renewed, current-main-based successor to
the historical draft PR #127. It offers exactly one MCP v2 tool:
`get_ctrader_candles(symbol, timeframe="M5", num_bars=100)`.

- The only upstream URL is **GET** `http://127.0.0.1:4000/api/market/candles`.
- The MCP listener is bound to **127.0.0.1:8765**, at local `/mcp`.
- Starting the **backend directly** does not auto-start MCP. The explicitly
  invoked Windows one-command `start-local.cmd` now starts this separate
  loopback gateway after backend readiness unless
  `TRADEAGENT_START_MCP=0` is set; `stop-local.cmd` stops only the verified
  MCP process before the existing backend/frontend shutdown. No change to
  trading runtime, account selection, orders, journal, Strategy Studio or Alpaca.
- It accepts cTrader M1/M5/M15/M30/H1/H4/D1 symbols, 2–500 bars; malformed,
  empty or out-of-order candles fail closed. Gaps are kept, not filled.
- Responses label requested vs observed depth and observed UTC timestamps.
  The `live=true` request can still hit TradeAgent memory cache; it does **not**
  attest broker provenance, freshness, completed bars, exact bar labeling,
  market calendar, execution/fills, trading signals or profit.
- No order, cancel, close, account switch, strategy registration, backtest
  initiation or privileged broker tool is registered.

## Local Windows test and smoke (non-trading)

```powershell
conda activate tradeagent-v2
python -m pip install -e ".[mcp]"
python -m pytest backend/tests/test_chatgpt_mcp_readonly.py -q
python scripts/validate.py
```

For one-command **Windows** start/stop from the repository root:

```powershell
cmd /c call start-local.cmd
cmd /c call stop-local.cmd
```

The `start-local.cmd` launcher now starts the local MCP process by default
when the backend is healthy. Use `$env:TRADEAGENT_START_MCP = "0"` before
`start-local.cmd` to skip it for that run, then remove the variable when
you want normal behavior. The lifecycle helper refuses port 8765 if it
belongs to an unrelated process; stop never kills that foreign process.
Logs are kept in `.codex-run/mcp.log` and `.codex-run/mcp-error.log`.
On Windows, this uses the same `tradeagent-v2` Python path as
`run-backend-local.cmd`. The launch script returns a failure code if MCP
is opted in but cannot start; the dashboard may still open.

The MCP SDK v2 dependency is optional for normal TradeAgent startup. The
GitHub backend CI installs the `mcp` extra so SDK registration is tested.
For an **optional manual smoke**, run TradeAgent normally on loopback 4000 with
an already authorized cTrader **Demo** data connection; do not switch accounts,
enable the engine, or change autotrade/kill-switch controls. In a separate
PowerShell session, run `python -m backend.chatgpt_mcp_readonly` and use an
MCP v2-compatible local client/Inspector to list exactly one tool and call
`get_ctrader_candles` read-only. Stop the standalone process afterward.
Do not send raw broker credentials, tokens or portfolio/account information to
an AI tool.

## SECURITY / ACCESS LIMITATIONS

**Do not expose** the MCP listener or TradeAgent API with ngrok, a tunnel,
public port forwarding, reverse proxy, `0.0.0.0`, or a remote hostname.
Loopback-only is an additional local boundary, **not authentication**. Other
local processes may still be able to connect. DNS-rebinding defenses from
the MCP SDK are not a replacement for server-side authorization.

This is **not yet a connected ChatGPT plugin**. ChatGPT desktop local
connectivity depends on client/plan features and separate plugin packaging.
Cloud ChatGPT Work cannot simply reach 127.0.0.1 on a Windows desktop.
Remote research needs separately approved authenticated HTTPS, explicit
per-tool allowlists, rate limiting, provenance/redaction and security review.
Until then, neither ChatGPT nor Codex is authorized to send execution requests
through the gateway. The original SMC assistant remains a separate repository.
