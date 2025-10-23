Strategy Studio Integration Plan

Overview
- Add a dedicated page at `/strategy-studio` for strategy creation and backtesting.
- Keep the main dashboard at `/` unchanged and stable.
- Use a small task API shim in the backend to map UI task types to the controller.

Frontend
- Route: Use `react-router-dom` with two routes: `/` and `/strategy-studio`.
- Page: `frontend/src/pages/StrategyStudio/index.tsx`
  - Left panel: chat-like input to describe tasks.
  - Right panel: results viewer that can show code or backtest metrics.
- Components
  - `StrategyChat.tsx`: basic message list and input box.
  - `CodeDisplay.tsx`: monospaced code block (handles long text).
  - `BacktestResult.tsx`: simple metrics table for backtest outputs.
- API
  - `executeTask(request: TaskRequest): Promise<TaskResponse>` in `frontend/src/services/api.ts`.
  - Supported `task_type` values: `calculate_indicator | backtest_strategy | save_strategy | research_strategy | create_strategy`.
  - Response (normalized): `{ status: 'success'|'error', message?: string, result?: any }`.

Backend
- Endpoint: `POST /api/agent/execute_task` (already added)
  - Maps UI task types to controller: indicator/backtest/strategy.
  - Normalizes responses. If controller returns `{ code: string }`, expose as `{ result: { stdout: string } }` for the UI to render.

Tasks (examples)
- Calculate indicator: "Calculate 14-period RSI for XAUUSD on H1 and summarize key divergences"
- Create strategy: "Create an SMA crossover strategy with 50/200 and risk management notes"
- Backtest strategy: "Backtest SMA-50/200 on XAUUSD H1 from 2023-01-01 to 2023-06-30"
- Save strategy: "Save this strategy as 'sma_crossover_v1'"

Non-goals (initial)
- Do not start autonomous agents from this page.
- Do not require a live data feed for code generation tasks.

Validation
- Hitting `/strategy-studio` loads the page.
- A prompt that generates code displays under the results area.
- Backtest task returns a structured metrics object or error; UI renders whatever the backend returns in the `result` field.

