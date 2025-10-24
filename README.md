# GenAI-MultiAgent-TradingSystem

A full-stack, local-first trading platform with a FastAPI backend and a modern React (Vite) frontend. It integrates cTrader OpenAPI for market data + order placement and uses an LLM (via Ollama) for analysis. It now includes a Strategy Studio for creating, backtesting, and saving custom strategies that the dashboard and agent can use.

---

## Highlights

- Live dashboard with chart, signals, positions, and pending orders
- AI Analysis button for one-off strategy decisions
- Autonomous agent with watchlist and confidence threshold
- Strategy Studio for code generation + backtesting + saving strategies
- Saved strategies auto-load and appear in both strategy dropdowns

---

## Architecture

`mermaid
graph TD
  subgraph Frontend
    UI[Dashboard + Strategy Studio]
  end

  subgraph Backend
    API[FastAPI /api/*]
    PA[ProgrammerAgent]
    BA[BacktestingAgent]
    STRAT[Strategy Loader]
    CTD[cTrader OpenAPI]
    OLL[Ollama LLM]
  end

  UI -->|/api/analyze| API
  UI -->|/api/agent/status| API
  UI -->|/api/agent/config| API
  UI -->|/api/agent/execute_task| API
  UI -->|/api/strategies/reload| API
  API --> PA
  API --> BA
  API --> STRAT
  API --> CTD
  API --> OLL
`

---

## Repository Layout

- backend/
  - app.py — FastAPI app and routes (includes strategies reload/list endpoints)
  - strategy.py — base Strategy classes (SMC, RSI) + loader for generated strategies
  - programmer_agent.py — generates indicator/strategy code (used by Strategy Studio)
  - backtesting_agent.py — backtests (SMA crossover; optional vectorbt)
  - strategies_generated/ — saved strategies from Strategy Studio (auto-loaded)
  - ctrader_client.py — cTrader OpenAPI integration
  - llm_analyzer.py, data_fetcher.py, indicators.py, smc_features.py …
  - journal/ — trade journaling API + DB
  - agents/ — autonomous agent runner (optional)
- frontend/
  - src/
    - App.tsx — routes ("/" dashboard, "/strategy-studio")
    - components/ — Header, Chart, SidePanel, Journal, AIOutput, AgentSettings …
      - StrategyChat.tsx — chat UI for Studio
      - CodeDisplay.tsx — code viewer with Copy
      - BacktestResult.tsx — backtest metrics table
    - pages/StrategyStudio/index.tsx — Strategy Studio page
    - services/api.ts — backend calls (executeTask + strategies reload)
    - styles/global.css
- docker-compose.yml — services: ollama, llm-smc (backend), frontend

---

## Strategy Studio

- Create strategies (code), run backtests, and save to ackend/strategies_generated/.
- Saved files with a top-level signals(df, ...) function are auto-loaded and appear in both strategy dropdowns (header + Agent Settings).

Endpoints used:
- POST /api/agent/execute_task
  - 	ask_type: calculate_indicator | create_strategy | backtest_strategy | save_strategy
  - params (backtests): { symbol, timeframe, num_bars }
  - params (save): { strategy_name, code }
- GET or POST /api/strategies/reload — re-scan ackend/strategies_generated and register any signals(df, ...) strategies
- GET /api/strategies — list available strategy names and last load errors

---

## Creating Custom Strategies

1) Location: put files in ackend/strategies_generated/. Example: ackend/strategies_generated/my_sma.py.

2) Minimal template (no indentation at the left edge):
`python
import pandas as pd

def signals(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
    """Return +1 (long), 0 (flat), or -1 (short) per bar.
    The dashboard uses the last value to infer the current side.
    """
    f = df['close'].rolling(fast, min_periods=fast).mean()
    s = df['close'].rolling(slow, min_periods=slow).mean()
    return (f > s).astype(int).diff().fillna(0)
`

3) Make it appear: save from Strategy Studio (auto-reloads), click "Reload Strategies" in the header, or call GET /api/strategies/reload.

4) Use it: select in the top-left strategy selector or Agent Settings and run analysis or enable the agent.

---

## Quickstart (Docker)

`sh
docker compose down
docker compose build --no-cache
docker compose up
`

- Frontend: http://localhost:8080
- Backend:  http://localhost:4000

---