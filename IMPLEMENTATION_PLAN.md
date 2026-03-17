# Advanced Backtesting & Strategy Studio Upgrade

## Goal Description
Enhance the existing "Strategy Studio" to rival frameworks like "Jesse" or "VectorBT" in terms of visualization and capability. This involves integrating `vectorbt` for high-performance backtesting and parameter optimization, and creating a rich frontend dashboard to visualize equity curves, trade lists, and optimization heatmaps.

## User Review Required
> [!IMPORTANT]
> This plan introduces `vectorbt` as a dependency. It is a powerful library but can be heavy. We will aim for the standard (non-PRO) version which is open source.

> [!NOTE]
> We will use `recharts` for the frontend visualization as it is already in the project, creating a consistent look and feel with the existing dashboard.

## Proposed Changes

### Backend

#### [MODIFY] [requirements.txt](file:///c:/Users/mohag/My%20Drive/Github/GenAI-MultiAgent-TradingSystem/requirements.txt)
- Add `vectorbt` and `schedule` (if needed for advanced timing, though `vectorbt` usually suffices for backtest).

#### [MODIFY] [backend/backtesting_agent.py](file:///c:/Users/mohag/My%20Drive/Github/GenAI-MultiAgent-TradingSystem/backend/backtesting_agent.py)
- Fully implement `run_backtest` using `vectorbt`.
- Add support for **Parameter Optimization**: Detect if parameters are lists/ranges and run `vbt.Portfolio.from_signals` in batch mode.
- Return comprehensive JSON:
    - `metrics`: Existing scalar metrics.
    - `equity`: Time series of equity.
    - `trades`: List of individual trades (entry/exit time, price, PnL).
    - `optimization_results`: (Optional) Table of Params vs. Return/Sharpe for heatmaps.

#### [NEW] [backend/optimizer_utils.py](file:///c:/Users/mohag/My%20Drive/Github/GenAI-MultiAgent-TradingSystem/backend/optimizer_utils.py)
- Helper functions to parse natural language ranges (e.g., "fast 10 to 50 step 10") into python lists for `vectorbt`.

### Frontend

#### [NEW] [frontend/src/components/BacktestDashboard.tsx](file:///c:/Users/mohag/My%20Drive/Github/GenAI-MultiAgent-TradingSystem/frontend/src/components/BacktestDashboard.tsx)
- A detailed dashboard component replacing the simple table.
- **Charts**: Equity Curve (Recharts AreaChart), Drawdown (Recharts AreaChart).
- **Tables**: Trade List (Scrollable table with Win/Loss coloring).
- **Optimization View**: If multiple results exist, show a heatmap or sorted table of best parameter sets.

#### [MODIFY] [frontend/src/pages/StrategyStudio/index.tsx](file:///c:/Users/mohag/My%20Drive/Github/GenAI-MultiAgent-TradingSystem/frontend/src/pages/StrategyStudio/index.tsx)
- Integrate `BacktestDashboard`.
- Add UI inputs for parameter ranges (or rely on the Chatbot to parse "optimize ...").
- Update `runBacktest` to handle the richer response.

## Verification Plan

### Automated Tests
- **Backend Unit Test**: Create `backend/tests/test_vectorbt.py` to ensure `vectorbt` runs a simple crossover and returns the expected JSON structure.
- **API Check**: Use `curl` to POST to `/api/agent/execute_task` with a backtest task and verify the JSON response contains `equity` and `trades`.

### Manual Verification
1.  **Optimization Flow**:
    -   Go to Strategy Studio.
    -   Type: "Optimize SMA strategy on XAUUSD H1. Fast from 10 to 50 step 10, Slow from 60 to 100 step 10."
    -   Verify the backend runs multiple combinations.
    -   Verify the Frontend displays a heatmap/table of results.
    -   Click a result to see its specific Equity Curve.
2.  **Visual Check**:
    -   Run a single backtest.
    -   Check that the Equity Curve looks correct (starts at 0% or 100%, fluctuates).
    -   Check the Trade List matches the chart logic.
