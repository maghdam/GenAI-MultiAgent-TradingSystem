import React, { Suspense, lazy, useState } from 'react';
import {
  backtestV2SavedStrategy,
  executeV2StudioTask,
  getV2StudioModels,
  getV2StudioStrategySource,
  getV2StrategyLifecycle,
  listV2StudioStrategyFiles,
  promoteV2StrategyLifecycle,
  recordV2PaperEvidence,
  updateV2StrategyHypothesis,
  type V2StrategyValidationMode,
  type V2StrategyLifecycle,
  type V2StudioTaskRequest,
  type V2StudioTaskResponse,
  type V2StudioProviderInfo,
} from '../../services/api';
import {
  STUDIO_LIFECYCLE_STAGES,
  STUDIO_VALIDATION_OPTIONS,
  lifecycleStageState,
} from '../../services/strategyStudioPresentation';
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
  const [feeBps, setFeeBps] = useState<number>(0);
  const [slippageBps, setSlippageBps] = useState<number>(0);
  const [spreadBps, setSpreadBps] = useState<number>(0);
  const [positionSizePct, setPositionSizePct] = useState<number>(100);
  const [validationKind, setValidationKind] = useState<V2StrategyValidationMode>('development_backtest');
  const [lifecycle, setLifecycle] = useState<V2StrategyLifecycle | null>(null);
  const [hypothesis, setHypothesis] = useState('');
  const [lifecycleError, setLifecycleError] = useState('');
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

  const isBacktestLikeResult = (result: any) => {
    if (!result) return false;
    if (result.metrics && (result.equity || result.optimization_results || result.plots)) return true;
    if (typeof result === 'object' && result['Total Return [%]'] !== undefined) return true;
    return false;
  };

  const persistBacktestResult = (
    result: any,
    meta?: { strategy: string; symbol: string; timeframe: string; numBars: number }
  ) => {
    try {
      localStorage.setItem(RESULT_KEY, JSON.stringify(result));
      const nextMeta = meta ?? { strategy: savedStrategy || 'draft', symbol, timeframe, numBars };
      localStorage.setItem(META_KEY, JSON.stringify({
        ts: Date.now(),
        ...nextMeta,
      }));
    } catch {
      // ignore
    }
  };

  const openResults = () => {
    if (lastResult && isBacktestLikeResult(lastResult)) {
      persistBacktestResult(lastResult);
      window.open('/build-test/results', '_blank', 'noopener,noreferrer');
    }
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
      .catch((error: any) => {
        if (!mounted) return;
        setProviders([
          { key: 'ollama', label: 'Ollama', configured: true },
          { key: 'gemini', label: 'Gemini', configured: false },
        ]);
        setLlmModels([]);
        setLlmModel('');
        setLlmModelError(error?.message || 'Failed to load models');
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
    let mounted = true;
    listV2StudioStrategyFiles()
      .then((files) => {
        if (!mounted) return;
        const list = Array.isArray(files) ? files : [];
        setAvailableSaved(list);
      })
      .catch(() => undefined);
    return () => { mounted = false; };
  }, []);

  React.useEffect(() => {
    let mounted = true;
    if (!savedStrategy) {
      setLifecycle(null);
      setHypothesis('');
      setLifecycleError('');
      return () => { mounted = false; };
    }
    getV2StrategyLifecycle(savedStrategy)
      .then((record) => {
        if (!mounted) return;
        setLifecycle(record);
        setHypothesis(record.hypothesis || '');
        setLifecycleError('');
      })
      .catch((error: any) => {
        if (!mounted) return;
        setLifecycle(null);
        setHypothesis('');
        setLifecycleError(error?.message || 'Lifecycle is not registered yet. Save or backtest the strategy to register it.');
      });
    return () => { mounted = false; };
  }, [savedStrategy]);

  const send = async (message: string) => {
    const text = (message || '').trim();
    if (!text || isLoading) return;
    const selectedModel = llmModels.includes(llmModel) ? llmModel : '';

    const history = messages
      .slice(-10)
      .map((item) => ({ role: item.role, content: String(item.content || '').slice(0, 400) }));

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
        fee_bps: feeBps,
        slippage_bps: slippageBps,
        spread_bps: spreadBps,
        position_size_pct: positionSizePct,
      },
    };

    setIsLoading(true);
    setMessages((previous) => [...previous, { role: 'user', content: text }]);
    setView('auto');
    try {
      const res: V2StudioTaskResponse = await executeV2StudioTask(req);
      if (res.status === 'success') {
        const nextCode = typeof res.result?.stdout === 'string' ? res.result.stdout : '';
        if (nextCode) {
          setDraftCode(nextCode);
          setLastResult({ stdout: nextCode, provider: res.result?.provider, model: res.result?.model });
        }
        setMessages((previous) => [...previous, {
          role: 'assistant',
          type: nextCode ? 'code' : 'text',
          content: res.message || (nextCode ? 'Strategy draft updated.' : 'Task completed.'),
        }]);
      } else {
        setMessages((previous) => [...previous, {
          role: 'assistant',
          type: 'error',
          content: res.message || 'Task failed.',
        }]);
      }
    } catch (error: any) {
      setMessages((previous) => [...previous, {
        role: 'assistant',
        type: 'error',
        content: error?.message || 'Request failed.',
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const runBacktest = async () => {
    if (isLoading) return;
    setIsLoading(true);
    setMessages((previous) => [...previous, {
      role: 'user',
      content: draftCode
        ? `Run ${validationKind} on current draft for ${symbol} ${timeframe} (${numBars} bars)`
        : `Run ${validationKind} on saved strategy ${savedStrategy || 'sma'} for ${symbol} ${timeframe} (${numBars} bars)`,
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
            spread_bps: spreadBps,
            position_size_pct: positionSizePct,
            strategy_name: savedStrategy || 'draft',
            validation_kind: validationKind,
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
          validationKind,
          spreadBps,
          positionSizePct,
        );
      }

      setLastResult(result);
      persistBacktestResult(result, meta);
      setView('auto');
      if (result?.Lifecycle) {
        setLifecycle(result.Lifecycle);
        setHypothesis(result.Lifecycle.hypothesis || '');
        setLifecycleError('');
      }
      const evidence = result?.Lifecycle?.evidence?.[0];
      setMessages((previous) => [...previous, {
        role: 'assistant',
        content: evidence
          ? `${evidence.summary} Gate ${evidence.passed ? 'passed' : 'failed'}.`
          : 'Backtest complete.',
      }]);
    } catch (error: any) {
      setMessages((previous) => [...previous, {
        role: 'assistant',
        type: 'error',
        content: error?.message || 'Backtest failed.',
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const loadSavedStrategy = async () => {
    if (!savedStrategy || isLoading) return;
    setIsLoading(true);
    try {
      const loaded = await getV2StudioStrategySource(savedStrategy);
      setDraftCode(loaded.source);
      setLastResult({ stdout: loaded.source, source: 'saved', strategy: loaded.strategy });
      setView('auto');
      setMessages((previous) => [...previous, {
        role: 'assistant',
        content: `Loaded saved strategy ${loaded.strategy} into the editable draft.`,
      }]);
    } catch (error: any) {
      setMessages((previous) => [...previous, {
        role: 'assistant',
        type: 'error',
        content: error?.message || 'Failed to load saved strategy.',
      }]);
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
        setMessages((previous) => [...previous, {
          role: 'assistant',
          content: res.message || `Saved as ${name}`,
        }]);
        const files = await listV2StudioStrategyFiles();
        const list = Array.isArray(files) ? files : [];
        setAvailableSaved(list);
        const savedName = res.result?.lifecycle?.strategy || name.toLowerCase();
        if (list.includes(savedName)) setSavedStrategy(savedName);
        if (res.result?.lifecycle) {
          setLifecycle(res.result.lifecycle);
          setHypothesis(res.result.lifecycle.hypothesis || '');
          setLifecycleError('');
        }
      } else {
        setMessages((previous) => [...previous, {
          role: 'assistant',
          type: 'error',
          content: res.message || 'Save failed.',
        }]);
      }
    } catch (error: any) {
      setMessages((previous) => [...previous, {
        role: 'assistant',
        type: 'error',
        content: error?.message || 'Request failed.',
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const saveLifecycleHypothesis = async () => {
    if (!savedStrategy || isLoading) return;
    setIsLoading(true);
    try {
      const record = await updateV2StrategyHypothesis(savedStrategy, hypothesis);
      setLifecycle(record);
      setLifecycleError('');
    } catch (error: any) {
      setLifecycleError(error?.message || 'Failed to save hypothesis.');
    } finally {
      setIsLoading(false);
    }
  };

  const promoteLifecycle = async () => {
    if (!savedStrategy || !lifecycle?.next_stage || isLoading) return;
    const operator = prompt('Operator name for the audit trail');
    if (!operator?.trim()) return;
    const reason = prompt(`Reason for promotion to ${lifecycle.next_stage}`);
    if (!reason?.trim()) {
      setLifecycleError('Promotion requires an audit reason.');
      return;
    }
    setIsLoading(true);
    try {
      const record = await promoteV2StrategyLifecycle(savedStrategy, operator.trim(), reason.trim());
      setLifecycle(record);
      setLifecycleError('');
    } catch (error: any) {
      setLifecycleError(error?.message || 'Promotion failed.');
    } finally {
      setIsLoading(false);
    }
  };

  const collectPaperEvidence = async () => {
    if (!savedStrategy || isLoading) return;
    setIsLoading(true);
    try {
      const record = await recordV2PaperEvidence(savedStrategy);
      setLifecycle(record);
      setLifecycleError('');
    } catch (error: any) {
      setLifecycleError(error?.message || 'Paper evidence collection failed.');
    } finally {
      setIsLoading(false);
    }
  };

  const hasBacktestResult = isBacktestLikeResult(lastResult);

  return (
    <div className="studio-console">
      <section className="ta-panel studio-lifecycle">
        <div className="studio-lifecycle__top">
          <div>
            <span className="studio-eyebrow">Strategy lifecycle</span>
            <strong>{lifecycle ? lifecycle.strategy : savedStrategy || 'New draft'}</strong>
            <small>
              {lifecycle
                ? `Version ${lifecycle.version} · ${lifecycle.version_hash.slice(0, 10)}`
                : lifecycleError || 'Save or backtest a strategy to register its governed lifecycle.'}
            </small>
          </div>
          <div className="studio-lifecycle__track" aria-label="Strategy lifecycle progression">
            {STUDIO_LIFECYCLE_STAGES.map((stage) => (
              <span
                className={`studio-stage studio-stage--${lifecycleStageState(lifecycle?.stage, stage)}`}
                key={stage}
              >
                {stage}
              </span>
            ))}
          </div>
        </div>

        {lifecycle && (
          <div className="studio-lifecycle__detail">
            <label>
              <span>Measurable hypothesis</span>
              <textarea
                className="ta-input"
                value={hypothesis}
                onChange={(event) => setHypothesis(event.target.value)}
                placeholder="Market, setup, entry/exit rule, expected behavior, and invalidation condition."
                rows={2}
              />
            </label>
            <div className="studio-lifecycle__actions">
              <button className="ta-btn" type="button" onClick={saveLifecycleHypothesis} disabled={isLoading}>
                Save hypothesis
              </button>
              {lifecycle.stage === 'paper' && (
                <button className="ta-btn" type="button" onClick={collectPaperEvidence} disabled={isLoading}>
                  Collect paper evidence
                </button>
              )}
              {lifecycle.next_stage && (
                <button
                  className="ta-btn ta-btn--primary"
                  type="button"
                  onClick={promoteLifecycle}
                  disabled={isLoading || !lifecycle.promotion_ready}
                >
                  Promote to {lifecycle.next_stage}
                </button>
              )}
            </div>
          </div>
        )}

        {lifecycle?.blockers.length ? (
          <div className="studio-lifecycle__notice">
            <strong>Blocked</strong>
            <span>{lifecycle.blockers.join(' ')}</span>
          </div>
        ) : null}

        {lifecycle?.evidence.length ? (
          <div className="studio-evidence-strip">
            {lifecycle.evidence.slice(0, 3).map((item) => (
              <span className={item.passed ? 'pass' : 'fail'} key={item.id}>
                {item.passed ? 'PASS' : 'FAIL'} · {item.summary}
              </span>
            ))}
          </div>
        ) : null}
      </section>

      <section className="studio-work-grid">
        <article className="ta-panel studio-work-card">
          <StrategyChat
            messages={messages}
            isLoading={isLoading}
            onSendMessage={send}
            placeholder='Example: "Create an XAUUSD M5 strategy using FVG and market structure."'
            headerRight={(
              <details className="studio-popover">
                <summary className="ta-btn ta-btn--ghost ta-btn--sm">AI settings</summary>
                <div className="studio-popover__menu studio-ai-settings">
                  <label>
                    Provider
                    <select
                      className="ta-input ta-input--sm"
                      value={llmProvider}
                      onChange={(event) => setLlmProvider(event.target.value)}
                      disabled={isLoading}
                    >
                      {(providers.length ? providers : [{ key: 'ollama', label: 'Ollama', configured: true }]).map((provider) => (
                        <option key={provider.key} value={provider.key}>
                          {provider.label}{provider.configured ? '' : ' (setup)'}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Model
                    <select
                      className="ta-input ta-input--sm"
                      value={llmModel}
                      onChange={(event) => setLlmModel(event.target.value)}
                      disabled={isLoading || llmModels.length === 0}
                      title={llmModelError ? `Studio model error: ${llmModelError}` : 'Studio model'}
                    >
                      {llmModels.length === 0 ? (
                        <option value="">{llmModelError || '(no models)'}</option>
                      ) : (
                        llmModels.map((model) => <option key={model} value={model}>{model}</option>)
                      )}
                    </select>
                  </label>
                </div>
              </details>
            )}
          />
        </article>

        <article className="ta-panel studio-work-card">
          <div className="studio-card-head">
            <div>
              <span className="studio-eyebrow">Strategy workspace</span>
              <strong>{draftCode ? 'Editable draft' : 'No draft loaded'}</strong>
            </div>
            <div className="studio-strategy-actions">
              <select
                className="ta-input ta-input--sm"
                value={savedStrategy}
                onChange={(event) => setSavedStrategy(event.target.value)}
                title="Saved strategy"
              >
                <option value="">Saved strategies…</option>
                {availableSaved.map((name) => <option key={name} value={name}>{name}</option>)}
              </select>
              <button className="ta-btn ta-btn--sm" type="button" onClick={loadSavedStrategy} disabled={isLoading || !savedStrategy}>
                Load
              </button>
              <button className="ta-btn ta-btn--primary ta-btn--sm" type="button" onClick={saveStrategy} disabled={isLoading || !draftCode}>
                Save
              </button>
            </div>
          </div>

          <div className="studio-draft">
            {draftCode
              ? <CodeDisplay code={draftCode} />
              : <div className="studio-empty">Ask the assistant to draft rules or load a saved strategy.</div>}
          </div>

          <div className="studio-card-footer">
            <span>{draftCode ? 'Draft is editable research code until lifecycle gates are satisfied.' : 'No strategy code is active in this workspace.'}</span>
            <button className="ta-btn ta-btn--ghost ta-btn--sm" type="button" onClick={resetStudio} disabled={isLoading}>
              Clear workspace
            </button>
          </div>
        </article>
      </section>

      <section className="ta-panel studio-backtest">
        <div className="studio-card-head">
          <div>
            <span className="studio-eyebrow">Backtest</span>
            <strong>Test the current draft or selected saved strategy</strong>
          </div>
          <button
            className="ta-btn ta-btn--primary"
            type="button"
            onClick={runBacktest}
            disabled={isLoading || (!draftCode && !savedStrategy)}
          >
            {isLoading ? 'Working…' : 'Run backtest'}
          </button>
        </div>

        <div className="studio-backtest__controls">
          <label>
            <span>Symbol</span>
            <input className="ta-input ta-input--sm" value={symbol} onChange={(event) => setSymbol(event.target.value.toUpperCase())} />
          </label>
          <label>
            <span>Timeframe</span>
            <select className="ta-input ta-input--sm" value={timeframe} onChange={(event) => setTimeframe(event.target.value)}>
              {['M1', 'M5', 'M15', 'H1', 'H4', 'D1'].map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </label>
          <label>
            <span>Bars</span>
            <input
              className="ta-input ta-input--sm"
              type="number"
              min={200}
              step={100}
              value={numBars}
              onChange={(event) => setNumBars(Math.max(200, parseInt(event.target.value || '1500', 10)))}
            />
          </label>
          <label className="studio-validation-control">
            <span>Validation</span>
            <select
              className="ta-input ta-input--sm"
              value={validationKind}
              onChange={(event) => setValidationKind(event.target.value as V2StrategyValidationMode)}
            >
              {STUDIO_VALIDATION_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </label>

          <details className="studio-assumptions">
            <summary className="ta-btn ta-btn--ghost ta-btn--sm">Assumptions</summary>
            <div className="studio-assumptions__menu">
              <label>Fees (bps)<input className="ta-input ta-input--sm" type="number" min={0} step={0.1} value={feeBps} onChange={(event) => setFeeBps(Math.max(0, Number.parseFloat(event.target.value || '0')))} /></label>
              <label>Slippage (bps)<input className="ta-input ta-input--sm" type="number" min={0} step={0.1} value={slippageBps} onChange={(event) => setSlippageBps(Math.max(0, Number.parseFloat(event.target.value || '0')))} /></label>
              <label>Spread (bps)<input className="ta-input ta-input--sm" type="number" min={0} step={0.1} value={spreadBps} onChange={(event) => setSpreadBps(Math.max(0, Number.parseFloat(event.target.value || '0')))} /></label>
              <label>Position size (%)<input className="ta-input ta-input--sm" type="number" min={1} max={100} step={1} value={positionSizePct} onChange={(event) => setPositionSizePct(Math.min(100, Math.max(1, Number.parseFloat(event.target.value || '100'))))} /></label>
            </div>
          </details>
        </div>
      </section>

      <section className="ta-panel studio-results">
        <div className="studio-card-head">
          <div>
            <span className="studio-eyebrow">Latest result</span>
            <strong>{hasBacktestResult ? `${symbol} · ${timeframe} · ${numBars} bars` : 'No backtest result yet'}</strong>
          </div>
          <div className="studio-result-actions">
            <button
              className="ta-btn ta-btn--sm"
              type="button"
              onClick={openResults}
              disabled={!hasBacktestResult}
            >
              Open results
            </button>
            {hasBacktestResult && (
              <details className="studio-popover">
                <summary className="ta-btn ta-btn--ghost ta-btn--sm">More</summary>
                <div className="studio-popover__menu">
                  <button className="ta-btn ta-btn--ghost ta-btn--sm" type="button" onClick={() => setView('auto')}>Formatted view</button>
                  <button className="ta-btn ta-btn--ghost ta-btn--sm" type="button" onClick={() => setView('raw')}>Raw JSON</button>
                </div>
              </details>
            )}
          </div>
        </div>
        <div className="studio-results__body">
          {hasBacktestResult
            ? renderBacktestResult(lastResult, view)
            : <div className="studio-empty">Run a development, holdout, regime, or walk-forward test to populate this area.</div>}
        </div>
      </section>
    </div>
  );
}

function renderBacktestResult(lastResult: any, view: 'auto' | 'raw') {
  if (view === 'raw') return <CodeDisplay code={JSON.stringify(lastResult, null, 2)} />;

  if (lastResult.metrics && (lastResult.equity || lastResult.optimization_results)) {
    return (
      <Suspense fallback={<div className="studio-empty">Loading backtest dashboard…</div>}>
        <BacktestDashboard data={lastResult} />
      </Suspense>
    );
  }

  if (typeof lastResult === 'object' && lastResult && lastResult['Total Return [%]'] !== undefined) {
    return <BacktestResult metrics={lastResult} />;
  }

  return <CodeDisplay code={JSON.stringify(lastResult, null, 2)} />;
}
