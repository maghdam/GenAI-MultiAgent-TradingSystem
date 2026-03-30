import React, { Suspense, lazy, useState } from 'react';
import {
  backtestV2SavedStrategy,
  executeV2StudioTask,
  getV2StudioModels,
  listV2StudioStrategyFiles,
  type V2StudioTaskRequest,
  type V2StudioTaskResponse,
  type V2StudioProviderInfo,
} from '../../services/api';
import StrategyChat, { type ChatMessage } from '../../components/StrategyChat';
import { CodeDisplay } from '../../components/CodeDisplay';
import { BacktestResult } from '../../components/BacktestResult';

const BacktestDashboard = lazy(() =>
  import('../../components/BacktestDashboard').then((module) => ({ default: module.BacktestDashboard }))
);

const RESULT_KEY = 'strategyStudio.backtest.lastResult';
const META_KEY = 'strategyStudio.backtest.lastMeta';
const CHAT_PROVIDER_KEY = 'strategyStudio.chat.llmProvider';
const CHAT_MODEL_KEY = 'strategyStudio.chat.llmModel';
const DRAFT_CODE_KEY = 'strategyStudio.draft.code';
const LAYOUT_KEY = 'strategyStudio.layout.mode';
const LAYOUT_USER_KEY = 'strategyStudio.layout.userSet';

export default function StrategyStudioPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [lastResult, setLastResult] = useState<any>(null);
  const [draftCode, setDraftCode] = useState<string>(() => {
    try {
      return localStorage.getItem(DRAFT_CODE_KEY) || '';
    } catch {
      return '';
    }
  });
  const [view, setView] = useState<'auto' | 'raw'>('auto');
  const [symbol, setSymbol] = useState('XAUUSD');
  const [timeframe, setTimeframe] = useState('M5');
  const [numBars, setNumBars] = useState(1500);
  const [savedStrategy, setSavedStrategy] = useState<string>('');
  const [availableSaved, setAvailableSaved] = useState<string[]>([]);
  const [showCosts, setShowCosts] = useState(false);
  const [feeBps, setFeeBps] = useState<number>(0);
  const [slippageBps, setSlippageBps] = useState<number>(0);
  const [providers, setProviders] = useState<V2StudioProviderInfo[]>([]);
  const [llmProvider, setLlmProvider] = useState<string>(() => {
    try {
      return localStorage.getItem(CHAT_PROVIDER_KEY) || 'ollama';
    } catch {
      return 'ollama';
    }
  });
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

  const persistBacktestResult = (
    result: any,
    meta?: { strategy: string; symbol: string; timeframe: string; numBars: number }
  ) => {
    try {
      localStorage.setItem(RESULT_KEY, JSON.stringify(result));
      const m = meta ?? { strategy: savedStrategy || 'draft', symbol, timeframe, numBars };
      localStorage.setItem(META_KEY, JSON.stringify({
        ts: Date.now(),
        ...m,
      }));
    } catch {
      // ignore
    }
  };

  const openResults = () => {
    if (lastResult && isBacktestLikeResult(lastResult)) {
      persistBacktestResult(lastResult);
    }
    window.open('/strategy-studio/results', '_blank', 'noopener,noreferrer');
  };

  const resetStudio = () => {
    try {
      localStorage.removeItem(RESULT_KEY);
      localStorage.removeItem(META_KEY);
      localStorage.removeItem(DRAFT_CODE_KEY);
    } catch {
      // ignore
    }
    setMessages([]);
    setDraftCode('');
    setLastResult(null);
    setView('auto');
  };

  React.useEffect(() => {
    try {
      const raw = localStorage.getItem(RESULT_KEY);
      if (raw) setLastResult(JSON.parse(raw));
    } catch {
      // ignore
    }
  }, []);

  React.useEffect(() => {
    try {
      localStorage.setItem(DRAFT_CODE_KEY, draftCode || '');
    } catch {
      // ignore
    }
  }, [draftCode]);

  React.useEffect(() => {
    let mounted = true;
    getV2StudioModels(llmProvider)
      .then((res) => {
        if (!mounted) return;
        const providerList = Array.isArray(res?.providers) ? res.providers : [];
        const models = Array.isArray(res?.models) ? res.models : [];
        setProviders(providerList);
        setLlmModels(models);
        setLlmModelError(res?.error ? String(res.error) : '');
        if (!llmModel && res?.default && models.includes(res.default)) {
          setLlmModel(res.default);
        } else if (llmModel && !models.includes(llmModel) && res?.default && models.includes(res.default)) {
          setLlmModel(res.default);
        } else if (!llmModel && models.length > 0) {
          setLlmModel(models[0]);
        } else if (models.length === 0) {
          setLlmModel('');
        }
      })
      .catch((e: any) => {
        if (!mounted) return;
        setProviders([{ key: 'ollama', label: 'Ollama', configured: true }, { key: 'gemini', label: 'Gemini', configured: false }]);
        setLlmModels([]);
        setLlmModel('');
        setLlmModelError(e?.message || 'Failed to load models');
      });
    return () => { mounted = false; };
  }, [llmProvider]);

  React.useEffect(() => {
    try {
      localStorage.setItem(CHAT_PROVIDER_KEY, llmProvider);
    } catch {
      // ignore
    }
  }, [llmProvider]);

  React.useEffect(() => {
    try {
      if (llmModel) localStorage.setItem(CHAT_MODEL_KEY, llmModel);
      else localStorage.removeItem(CHAT_MODEL_KEY);
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
    listV2StudioStrategyFiles()
      .then((files) => {
        if (!mounted) return;
        const list = Array.isArray(files) ? files : [];
        setAvailableSaved(list);
        if (!savedStrategy && list.length > 0) {
          setSavedStrategy(list.includes('smc') ? 'smc' : list[0]);
        }
      })
      .catch(() => undefined);
    return () => { mounted = false; };
  }, []);

  const send = async (message: string) => {
    const text = (message || '').trim();
    if (!text || isLoading) return;
    const selectedModel = llmModels.includes(llmModel) ? llmModel : '';

    const history = messages
      .slice(-10)
      .map((m) => ({ role: m.role, content: String(m.content || '').slice(0, 400) }));

    const req: V2StudioTaskRequest = {
      task_type: 'chat',
      goal: text,
      params: {
        history,
        symbol,
        timeframe,
        num_bars: numBars,
        strategy_name: savedStrategy || 'draft',
        llm_provider: llmProvider,
        llm_model: selectedModel || undefined,
        current_code: draftCode || undefined,
      },
    };

    setIsLoading(true);
    setMessages((prev) => [...prev, { role: 'user', content: text }]);
    setView('auto');
    try {
      const res: V2StudioTaskResponse = await executeV2StudioTask(req);
      if (res.status === 'success') {
        const nextCode = typeof res.result?.stdout === 'string' ? res.result.stdout : '';
        if (nextCode) {
          setDraftCode(nextCode);
          setLastResult({ stdout: nextCode, provider: res.result?.provider, model: res.result?.model });
        }
        setMessages((prev) => [...prev, {
          role: 'assistant',
          type: nextCode ? 'code' : 'text',
          content: res.message || (nextCode ? 'Strategy draft updated.' : 'Task completed.'),
        }]);
      } else {
        setMessages((prev) => [...prev, { role: 'assistant', type: 'error', content: res.message || 'Task failed.' }]);
      }
    } catch (e: any) {
      setMessages((prev) => [...prev, { role: 'assistant', type: 'error', content: e?.message || 'Request failed.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  const runBacktest = async () => {
    if (isLoading) return;
    setIsLoading(true);
    setMessages((prev) => [...prev, {
      role: 'user',
      content: draftCode
        ? `Run backtest on current draft for ${symbol} ${timeframe} (${numBars} bars)`
        : `Run backtest on saved strategy ${savedStrategy || 'sma'} for ${symbol} ${timeframe} (${numBars} bars)`,
    }]);

    try {
      let result: any;
      const meta = { strategy: savedStrategy || 'draft', symbol, timeframe, numBars };

      if (draftCode) {
        const req: V2StudioTaskRequest = {
          task_type: 'backtest_strategy',
          goal: 'backtest current draft',
          params: {
            symbol,
            timeframe,
            num_bars: numBars,
            fee_bps: feeBps,
            slippage_bps: slippageBps,
            strategy_name: savedStrategy || 'draft',
            code: draftCode,
          },
        };
        const res = await executeV2StudioTask(req);
        if (res.status !== 'success') {
          throw new Error(res.message || 'Backtest failed.');
        }
        result = res.result;
      } else {
        result = await backtestV2SavedStrategy(
          savedStrategy || 'sma',
          symbol,
          timeframe,
          numBars,
          feeBps,
          slippageBps,
        );
      }

      setLastResult(result);
      persistBacktestResult(result, meta);
      setMessages((prev) => [...prev, { role: 'assistant', content: 'Backtest complete.' }]);
    } catch (e: any) {
      setMessages((prev) => [...prev, { role: 'assistant', type: 'error', content: e?.message || 'Backtest failed.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  const saveStrategy = async () => {
    const code = draftCode;
    if (!code || isLoading) return;
    const name = prompt('Strategy name to save as (filename)');
    if (!name) return;
    setIsLoading(true);
    try {
      const req: V2StudioTaskRequest = {
        task_type: 'save_strategy',
        goal: `save strategy ${name}`,
        params: { strategy_name: name, code },
      };
      const res: V2StudioTaskResponse = await executeV2StudioTask(req);
      if (res.status === 'success') {
        setMessages((prev) => [...prev, { role: 'assistant', content: res.message || `Saved as ${name}` }]);
        const files = await listV2StudioStrategyFiles();
        const list = Array.isArray(files) ? files : [];
        setAvailableSaved(list);
        if (list.includes(name)) setSavedStrategy(name);
      } else {
        setMessages((prev) => [...prev, { role: 'assistant', type: 'error', content: res.message || 'Save failed.' }]);
      }
    } catch (e: any) {
      setMessages((prev) => [...prev, { role: 'assistant', type: 'error', content: e?.message || 'Request failed.' }]);
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
          ...(layoutMode === 'stack' ? { height: 'min(560px, 48vh)', minHeight: 380 } : {}),
        }}
      >
        <StrategyChat
          messages={messages}
          isLoading={isLoading}
          onSendMessage={send}
          placeholder='Create or improve a strategy. Example: "Create an XAUUSD M5 strategy using FVG and market structure."'
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
              <span className="muted" style={{ fontSize: 12 }}>Provider</span>
              <select
                value={llmProvider}
                onChange={(e) => setLlmProvider(e.target.value)}
                disabled={isLoading}
                title="LLM provider"
                style={{ maxWidth: 160 }}
              >
                {(providers.length ? providers : [{ key: 'ollama', label: 'Ollama', configured: true }]).map((provider) => (
                  <option key={provider.key} value={provider.key}>
                    {provider.label}{provider.configured ? '' : ' (setup)'}
                  </option>
                ))}
              </select>
              <span className="muted" style={{ fontSize: 12 }}>Model</span>
              <select
                value={llmModel}
                onChange={(e) => setLlmModel(e.target.value)}
                disabled={isLoading || llmModels.length === 0}
                title={llmModelError ? `Studio model error: ${llmModelError}` : 'Studio model'}
                style={{ maxWidth: 220 }}
              >
                {llmModels.length === 0 ? (
                  <option value="">{llmModelError || '(no models)'}</option>
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
            <div style={{ fontWeight: 600 }}>{draftCode ? 'Draft / Result' : 'Result'}</div>
            <button className="btn primary" type="button" onClick={openResults} title="Open the latest backtest results in a new tab">
              Open Results
            </button>
            <button className="btn" type="button" onClick={resetStudio} disabled={isLoading} title="Clear the current draft and the last backtest result">
              Reset
            </button>
          </div>
          <div className="stack">
            <input value={symbol} onChange={e => setSymbol(e.target.value.toUpperCase())} style={{ width: 90 }} title="Symbol" />
            <select value={timeframe} onChange={e => setTimeframe(e.target.value)} title="Timeframe">
              {['M1', 'M5', 'M15', 'H1', 'H4', 'D1'].map(tf => (<option key={tf} value={tf}>{tf}</option>))}
            </select>
            <input
              type="number"
              min={200}
              step={100}
              value={numBars}
              onChange={e => setNumBars(Math.max(200, parseInt(e.target.value || '1500', 10)))}
              style={{ width: 110 }}
              title="Bars"
            />
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
            <button className="btn" type="button" onClick={runBacktest} disabled={isLoading || (!draftCode && !savedStrategy)}>
              Run Backtest
            </button>
            <select value={savedStrategy} onChange={e => setSavedStrategy(e.target.value)} title="Saved Strategy">
              <option value="">Select saved strategy…</option>
              {availableSaved.map(name => (<option key={name} value={name}>{name}</option>))}
            </select>
            <button className="btn" type="button" onClick={saveStrategy} disabled={isLoading || !draftCode}>Save Strategy</button>
            <button className="btn" type="button" onClick={() => setView('auto')} disabled={view === 'auto'}>Formatted</button>
            <button className="btn" type="button" onClick={() => setView('raw')} disabled={view === 'raw'}>Raw JSON</button>
          </div>
        </div>
        <div style={{ flex: 1, minHeight: 0 }}>
          {renderResult(lastResult, view, draftCode)}
        </div>
      </div>
    </div>
  );
}

function renderResult(lastResult: any, view: 'auto' | 'raw', draftCode: string) {
  if (!lastResult && draftCode) return <CodeDisplay code={draftCode} />;
  if (!lastResult) return <div className="muted">No result yet.</div>;
  if (view === 'raw') return <CodeDisplay code={JSON.stringify(lastResult, null, 2)} />;
  if (lastResult.stdout) return <CodeDisplay code={lastResult.stdout} />;

  if (lastResult.metrics && (lastResult.equity || lastResult.optimization_results)) {
    return (
      <Suspense fallback={<div className="muted">Loading backtest dashboard...</div>}>
        <BacktestDashboard data={lastResult} />
      </Suspense>
    );
  }

  if (typeof lastResult === 'object' && lastResult && lastResult['Total Return [%]'] !== undefined) {
    return <BacktestResult metrics={lastResult} />;
  }

  return <CodeDisplay code={JSON.stringify(lastResult, null, 2)} />;
}
