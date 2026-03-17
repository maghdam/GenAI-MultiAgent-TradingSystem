import React, { useState } from 'react';
import { executeTask, getLlmModels, listStrategyFiles, backtestSavedStrategy, type TaskRequest, type TaskResponse } from '../../services/api';
import StrategyChat, { type ChatMessage } from '../../components/StrategyChat';
import { CodeDisplay } from '../../components/CodeDisplay';
import { BacktestResult } from '../../components/BacktestResult';
import { BacktestDashboard } from '../../components/BacktestDashboard';

const RESULT_KEY = 'strategyStudio.backtest.lastResult';
const META_KEY = 'strategyStudio.backtest.lastMeta';
const CHAT_MODEL_KEY = 'strategyStudio.chat.llmModel';
const LAYOUT_KEY = 'strategyStudio.layout.mode';
const LAYOUT_USER_KEY = 'strategyStudio.layout.userSet';

export default function StrategyStudioPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [lastResult, setLastResult] = useState<any>(null);
  const [view, setView] = useState<'auto' | 'raw'>('auto');
  const [symbol, setSymbol] = useState('XAUUSD');
  const [timeframe, setTimeframe] = useState('M5');
  const [numBars, setNumBars] = useState(1500);
  const [savedStrategy, setSavedStrategy] = useState<string>('');
  const [availableSaved, setAvailableSaved] = useState<string[]>([]);
  const [showCosts, setShowCosts] = useState(false);
  const [feeBps, setFeeBps] = useState<number>(0);
  const [slippageBps, setSlippageBps] = useState<number>(0);
  const [llmModels, setLlmModels] = useState<string[]>([]);
  const [llmModel, setLlmModel] = useState<string>(() => {
    try {
      return localStorage.getItem(CHAT_MODEL_KEY) || '';
    } catch {
      return '';
    }
  });
  const [llmModelError, setLlmModelError] = useState<string>('');
  const [layoutMode, setLayoutMode] = useState<'split' | 'stack'>(() => {
    try {
      const userSet = localStorage.getItem(LAYOUT_USER_KEY) === '1';
      const saved = localStorage.getItem(LAYOUT_KEY);
      if (userSet && (saved === 'split' || saved === 'stack')) return saved;
      // Default to vertical layout unless the user explicitly chose otherwise.
      return 'stack';
    } catch {
      return 'stack';
    }
  });

  const isBacktestLikeResult = (r: any) => {
    if (!r) return false;
    if (r.metrics && (r.equity || r.optimization_results || r.plots)) return true;
    if (typeof r === 'object' && r && r['Total Return [%]'] !== undefined) return true;
    return false;
  };

  const extractBacktestMetaFromText = (text: string) => {
    const lower = (text || '').toLowerCase();
    const tf = (text.match(/\b(M1|M5|M15|M30|H1|H4|D1|W1)\b/i)?.[1] || timeframe).toUpperCase();

    let sym = symbol;
    const symFromOn = text.match(/\bon\s+([A-Za-z0-9_]{3,20})\b/i)?.[1];
    const symFromPair = text.match(/\b([A-Za-z]{3,10}USD)\b/i)?.[1];
    if (symFromOn) sym = symFromOn.toUpperCase();
    else if (symFromPair) sym = symFromPair.toUpperCase();

    let strategy = savedStrategy || 'sma';
    if (lower.includes('smc')) strategy = 'smc';
    else if (lower.includes('rsi')) strategy = 'rsi';
    else if (lower.includes('sma')) strategy = 'sma';

    const barsMatch = text.match(/(\d{2,6})\s*bars?\b/i)?.[1];
    const bars = barsMatch ? Math.max(200, Number.parseInt(barsMatch, 10)) : numBars;

    return { strategy, symbol: sym, timeframe: tf, numBars: bars };
  };

  const persistBacktestResult = (
    result: any,
    meta?: { strategy: string; symbol: string; timeframe: string; numBars: number }
  ) => {
    try {
      localStorage.setItem(RESULT_KEY, JSON.stringify(result));
      const m = meta ?? { strategy: savedStrategy || 'sma', symbol, timeframe, numBars };
      localStorage.setItem(META_KEY, JSON.stringify({
        ts: Date.now(),
        ...m,
      }));
    } catch {
      // ignore quota / serialization failures
    }
  };

  const openResults = () => {
    // Ensure whatever is currently displayed is available in the results tab
    if (lastResult && isBacktestLikeResult(lastResult)) {
      persistBacktestResult(lastResult);
    }
    window.open('/strategy-studio/results', '_blank', 'noopener,noreferrer');
  };

  const resetStudio = () => {
    try {
      localStorage.removeItem(RESULT_KEY);
      localStorage.removeItem(META_KEY);
    } catch {
      // ignore
    }
    setMessages([]);
    setInput('');
    setLastResult(null);
    setView('auto');
  };

  React.useEffect(() => {
    // Restore last backtest output (useful after refresh)
    try {
      const raw = localStorage.getItem(RESULT_KEY);
      if (raw) setLastResult(JSON.parse(raw));
    } catch {
      // ignore
    }
  }, []);

  React.useEffect(() => {
    let mounted = true;
    getLlmModels()
      .then((res) => {
        if (!mounted) return;
        const models = Array.isArray(res?.models) ? res.models : [];
        setLlmModels(models);
        if (!llmModel && res?.default && models.includes(res.default)) {
          setLlmModel(res.default);
        } else if (!llmModel && models.length > 0) {
          setLlmModel(models[0]);
        }
        if (res?.error) setLlmModelError(String(res.error));
      })
      .catch((e: any) => {
        if (!mounted) return;
        setLlmModelError(e?.message || 'Failed to load models');
      });
    return () => { mounted = false };
  }, []);

  React.useEffect(() => {
    try {
      if (llmModel) localStorage.setItem(CHAT_MODEL_KEY, llmModel);
    } catch {
      // ignore
    }
  }, [llmModel]);

  React.useEffect(() => {
    try {
      localStorage.setItem(LAYOUT_KEY, layoutMode);
    } catch {
      // ignore
    }
  }, [layoutMode]);

  React.useEffect(() => {
    let mounted = true;
    listStrategyFiles()
      .then((files) => {
        if (!mounted) return;
        const list = Array.isArray(files) ? files : [];
        setAvailableSaved(list);
        // Preselect 'smc' if available and nothing selected
        if (!savedStrategy && list.includes('smc')) {
          setSavedStrategy('smc');
        }
      })
      .catch(() => { })
    return () => { mounted = false };
  }, []);

  const parseMessage = (message: string): TaskRequest => {
    const lower = message.toLowerCase();
    if (lower.startsWith('save this strategy as')) {
      const name = message.substring('save this strategy as'.length).trim();
      return { task_type: 'save_strategy', goal: message, params: { strategy_name: name } };
    }

    const hasOptimizeWord = /\boptimi[sz](e|ation|ing)?\b/i.test(message);
    const hasBacktestWord = /\bbacktest(ing)?\b/i.test(message);
    const wantsHelp =
      /\bhelp\b/i.test(message) ||
      /\bwhat can you do\b/i.test(message) ||
      /\bcommands?\b/i.test(message) ||
      /\bhow do i\b/i.test(message);

    const hasTarget =
      /\bon\s+([A-Za-z0-9_]{3,20})\b/i.test(message) ||
      /\b([A-Za-z]{3,10}USD)\b/i.test(message) ||
      /\b(M1|M5|M15|M30|H1|H4|D1|W1)\b/i.test(message) ||
      /\b\d{2,9}\s*bars?\b/i.test(message) ||
      /\b(sma|rsi|smc)\b/i.test(message) ||
      /\bstrategy\b/i.test(message);

    const isActionOptimize =
      hasOptimizeWord &&
      (/^\s*optimi[sz](e|ation|ing)?\b/i.test(message) ||
        (/\b(can you|could you|please|plz|run|do|perform|execute|help me)\b/i.test(message) && hasTarget) ||
        hasTarget);

    const isActionBacktest =
      hasBacktestWord &&
      (/^\s*backtest(ing)?\b/i.test(message) ||
        (/\b(can you|could you|please|plz|run|do|perform|execute|help me)\b/i.test(message) && hasTarget) ||
        hasTarget);

    if (isActionOptimize) {
      const meta = extractBacktestMetaFromText(message);
      return {
        task_type: 'optimize',
        goal: message,
        params: {
          symbol: meta.symbol,
          timeframe: meta.timeframe,
          num_bars: meta.numBars,
          strategy_name: meta.strategy,
          fee_bps: feeBps,
          slippage_bps: slippageBps,
        },
      };
    }
    if (isActionBacktest) {
      const meta = extractBacktestMetaFromText(message);
      return {
        task_type: 'backtest_strategy',
        goal: message,
        params: {
          symbol: meta.symbol,
          timeframe: meta.timeframe,
          num_bars: meta.numBars,
          strategy_name: meta.strategy,
          fee_bps: feeBps,
          slippage_bps: slippageBps,
        },
      };
    }
    if (lower.includes('create') && lower.includes('strategy')) {
      return { task_type: 'create_strategy', goal: message };
    }

    // Default to chat for general questions instead of always generating code.
    // Treat explicit indicator/code requests as "calculate_indicator".
    const wantsCode =
      /\b(indicator|code|script|function|generate|write)\b/i.test(message);

    if (wantsHelp || message.trim().endsWith('?')) {
      const meta = extractBacktestMetaFromText(message);
      return {
        task_type: 'chat',
        goal: message,
        params: {
          symbol: meta.symbol,
          timeframe: meta.timeframe,
          num_bars: meta.numBars,
          strategy_name: meta.strategy,
        },
      };
    }

    return wantsCode
      ? { task_type: 'calculate_indicator', goal: message }
      : { task_type: 'chat', goal: message, params: { symbol, timeframe, num_bars: numBars, strategy_name: savedStrategy || 'sma' } };
  };

  const send = async (text?: string) => {
    const message = (text ?? input).trim();
    if (!message || isLoading) return;
    const req = parseMessage(message);
    if (req.task_type === 'chat') {
      const history = messages
        .slice(-10)
        .map((m) => ({ role: m.role, content: String(m.content || '').slice(0, 400) }));
      req.params = { ...(req.params || {}), history, llm_model: llmModel || undefined };
    }
    setIsLoading(true);
    setMessages(prev => [...prev, { role: 'user', content: message }]);
    if (req.task_type !== 'chat') setLastResult(null);
    setInput('');
    try {
      const res: TaskResponse = await executeTask(req);
      if (res.status === 'success') {
        if (req.task_type !== 'chat') setLastResult(res.result);
        if (req.task_type === 'backtest_strategy' || req.task_type === 'backtest' || req.task_type === 'optimize') {
          if (isBacktestLikeResult(res.result)) {
            persistBacktestResult(res.result, extractBacktestMetaFromText(message));
          }
        }
        setMessages(prev => [
          ...prev,
          { role: 'assistant', type: res.result?.stdout ? 'code' : 'text', content: res.message || 'Task completed.' },
        ]);
      } else {
        setMessages(prev => [...prev, { role: 'assistant', type: 'error', content: res.message || 'Task failed.' }]);
      }
    } catch (e: any) {
      setMessages(prev => [...prev, { role: 'assistant', type: 'error', content: e?.message || 'Request failed.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  const runBacktest = async () => {
    if (isLoading) return;
    const meta = { strategy: savedStrategy || 'sma', symbol, timeframe, numBars };
    setIsLoading(true);
    setMessages(prev => [...prev, { role: 'user', content: `Backtest ${symbol} ${timeframe} (${numBars} bars)` }]);
    setLastResult(null);
    try {
      // Use new Agent/VectorBT backend
      const req: TaskRequest = {
        task_type: 'backtest',
        goal: `backtest ${savedStrategy || 'SMA'} on ${symbol}`,
        params: {
          symbol,
          timeframe,
          num_bars: numBars,
          strategy_name: savedStrategy || 'sma', // pass along, though backend currently defaults to SMA
          fee_bps: feeBps,
          slippage_bps: slippageBps
        },
      };

      const res: TaskResponse = await executeTask(req);
      
      if (res.status === 'success') {
         setLastResult(res.result);
         persistBacktestResult(res.result, meta);
         setMessages(prev => [...prev, { role: 'assistant', content: res.message || 'Backtest complete.' }]);
      } else {
         setMessages(prev => [...prev, { role: 'assistant', type: 'error', content: res.message || 'Backtest failed.' }]);
      }
    } catch (e: any) {
      setMessages(prev => [...prev, { role: 'assistant', type: 'error', content: e?.message || 'Request failed.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  // Removed secondary backtest handler; keeping dropdown for future use

  const saveStrategy = async () => {
    const code = lastResult?.stdout;
    if (!code || isLoading) return;
    const name = prompt('Strategy name to save as (filename)');
    if (!name) return;
    setIsLoading(true);
    try {
      const req: TaskRequest = {
        task_type: 'save_strategy',
        goal: `save strategy ${name}`,
        params: { strategy_name: name, code },
      };
      const res: TaskResponse = await executeTask(req);
      if (res.status === 'success') {
        setMessages(prev => [...prev, { role: 'assistant', content: res.message || `Saved as ${name}` }]);
        try {
          const files = await listStrategyFiles();
          setAvailableSaved(Array.isArray(files) ? files : []);
          if (files && files.includes(name)) setSavedStrategy(name);
        } catch { }
      } else {
        setMessages(prev => [...prev, { role: 'assistant', type: 'error', content: res.message || 'Save failed.' }]);
      }
    } catch (e: any) {
      setMessages(prev => [...prev, { role: 'assistant', type: 'error', content: e?.message || 'Request failed.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={`strategy-studio-layout ${layoutMode === 'stack' ? 'stack' : ''}`}>
      <div
        className="box"
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '8px',
          ...(layoutMode === 'stack' ? { height: 'min(520px, 45vh)', minHeight: 360 } : {}),
        }}
      >
        <StrategyChat
          messages={messages}
          isLoading={isLoading}
          onSendMessage={send}
          headerRight={(
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className="muted" style={{ fontSize: 12 }}>Layout</span>
              <select
                value={layoutMode}
                onChange={(e) => {
                  const next = e.target.value === 'split' ? 'split' : 'stack';
                  setLayoutMode(next);
                  try {
                    localStorage.setItem(LAYOUT_USER_KEY, '1');
                  } catch {
                    // ignore
                  }
                }}
                disabled={isLoading}
                title="Page layout"
                style={{ maxWidth: 160 }}
              >
                <option value="stack">Vertical</option>
                <option value="split">Side-by-side</option>
              </select>
              <span className="muted" style={{ fontSize: 12 }}>Model</span>
              <select
                value={llmModel}
                onChange={(e) => setLlmModel(e.target.value)}
                disabled={isLoading || llmModels.length === 0}
                title={llmModelError ? `LLM models error: ${llmModelError}` : 'Chat model'}
                style={{ maxWidth: 200 }}
              >
                {llmModels.length === 0 ? (
                  <option value="">(no models)</option>
                ) : (
                  llmModels.map((m) => (<option key={m} value={m}>{m}</option>))
                )}
              </select>
            </div>
          )}
        />
      </div>
      <div className="box" style={{ display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8, gap: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{ fontWeight: 600 }}>Result</div>
            <button className="btn primary" type="button" onClick={openResults} title="Open the latest backtest results in a new tab">
              Open Results
            </button>
            <button className="btn" type="button" onClick={resetStudio} disabled={isLoading} title="Clear the last backtest result and reset the page">
              Reset
            </button>
          </div>
          <div className="stack">
            <input value={symbol} onChange={e => setSymbol(e.target.value.toUpperCase())} style={{ width: 90 }} title="Symbol" />
            <select value={timeframe} onChange={e => setTimeframe(e.target.value)} title="Timeframe">
              {['M1', 'M5', 'M15', 'H1', 'H4', 'D1'].map(tf => (<option key={tf} value={tf}>{tf}</option>))}
            </select>
            <input type="number" min={200} step={100} value={numBars}
              onChange={e => setNumBars(Math.max(200, parseInt(e.target.value || '1500', 10)))}
              style={{ width: 110 }} title="Bars" />
            <button className="btn" type="button" onClick={() => setShowCosts(s => !s)} title="Toggle fee/slippage inputs">{showCosts ? 'Hide Costs' : 'Costs'}</button>
            {showCosts && (
              <>
                <input
                  type="number"
                  min={0}
                  step={0.1}
                  value={feeBps}
                  onChange={e => setFeeBps(Math.max(0, Number.parseFloat(e.target.value || '0')))}
                  style={{ width: 110 }}
                  title="Fees (bps)"
                  placeholder="Fee bps"
                />
                <input
                  type="number"
                  min={0}
                  step={0.1}
                  value={slippageBps}
                  onChange={e => setSlippageBps(Math.max(0, Number.parseFloat(e.target.value || '0')))}
                  style={{ width: 130 }}
                  title="Slippage (bps)"
                  placeholder="Slippage bps"
                />
              </>
            )}
            <button className="btn" type="button" onClick={runBacktest} disabled={isLoading}>Run Backtest</button>
            <select value={savedStrategy} onChange={e => setSavedStrategy(e.target.value)} title="Saved Strategy">
              <option value="">Select strategy…</option>
              {availableSaved.map(name => (<option key={name} value={name}>{name}</option>))}
            </select>
            <button className="btn" type="button" onClick={saveStrategy} disabled={isLoading || !lastResult?.stdout}>Save Strategy</button>
            <button className="btn" type="button" onClick={() => setView('auto')} disabled={view === 'auto'}>Formatted</button>
            <button className="btn" type="button" onClick={() => setView('raw')} disabled={view === 'raw'}>Raw JSON</button>
          </div>
        </div>
        <div style={{ flex: 1, minHeight: 0 }}>
          {renderResult(lastResult, view)}
        </div>
      </div>
    </div>
  );
}

function renderResult(lastResult: any, view: 'auto' | 'raw') {
  if (!lastResult) return <div className="muted">No result yet.</div>;
  if (view === 'raw') return <CodeDisplay code={JSON.stringify(lastResult, null, 2)} />;
  if (lastResult.stdout) return <CodeDisplay code={lastResult.stdout} />;

  // New Backtest Data Shape
  if (lastResult.metrics && (lastResult.equity || lastResult.optimization_results)) {
    return <BacktestDashboard data={lastResult} />;
  }

  // Legacy flat metrics
  if (typeof lastResult === 'object' && lastResult && lastResult['Total Return [%]'] !== undefined) {
    return <BacktestResult metrics={lastResult} />;
  }

  return <CodeDisplay code={JSON.stringify(lastResult, null, 2)} />;
}
