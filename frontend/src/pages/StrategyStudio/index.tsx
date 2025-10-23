import React, { useState } from 'react';
import { executeTask, type TaskRequest, type TaskResponse } from '../../services/api';
import StrategyChat, { type ChatMessage } from '../../components/StrategyChat';
import { CodeDisplay } from '../../components/CodeDisplay';
import { BacktestResult } from '../../components/BacktestResult';

export default function StrategyStudioPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [lastResult, setLastResult] = useState<any>(null);
  const [view, setView] = useState<'auto'|'raw'>('auto');
  const [symbol, setSymbol] = useState('XAUUSD');
  const [timeframe, setTimeframe] = useState('M5');
  const [numBars, setNumBars] = useState(1500);

  const parseMessage = (message: string): TaskRequest => {
    const lower = message.toLowerCase();
    if (lower.startsWith('save this strategy as')) {
      const name = message.substring('save this strategy as'.length).trim();
      return { task_type: 'save_strategy', goal: message, params: { strategy_name: name } };
    }
    if (lower.startsWith('backtest')) {
      return { task_type: 'backtest_strategy', goal: message };
    }
    if (lower.includes('create') && lower.includes('strategy')) {
      return { task_type: 'create_strategy', goal: message };
    }
    return { task_type: 'calculate_indicator', goal: message };
  };

  const send = async (text?: string) => {
    const message = (text ?? input).trim();
    if (!message || isLoading) return;
    setIsLoading(true);
    setMessages(prev => [...prev, { role: 'user', content: message }]);
    setLastResult(null);
    setInput('');
    try {
      const req = parseMessage(message);
      const res: TaskResponse = await executeTask(req);
      if (res.status === 'success') {
        setLastResult(res.result);
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
    setIsLoading(true);
    setMessages(prev => [...prev, { role: 'user', content: `Backtest ${symbol} ${timeframe} (${numBars} bars)` }]);
    setLastResult(null);
    try {
      const req: TaskRequest = {
        task_type: 'backtest_strategy',
        goal: `backtest ${symbol} on ${timeframe}`,
        params: { symbol, timeframe, num_bars: numBars },
      };
      const res: TaskResponse = await executeTask(req);
      if (res.status === 'success') {
        setLastResult(res.result);
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
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', padding: '16px' }}>
      <div className="box" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <StrategyChat messages={messages} isLoading={isLoading} onSendMessage={send} />
      </div>
      <div className="box">
        <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom: 8, gap: 8 }}>
          <div style={{ fontWeight: 600 }}>Result</div>
          <div className="stack">
            <input value={symbol} onChange={e => setSymbol(e.target.value.toUpperCase())} style={{ width: 90 }} title="Symbol" />
            <select value={timeframe} onChange={e => setTimeframe(e.target.value)} title="Timeframe">
              {['M1','M5','M15','H1','H4','D1'].map(tf => (<option key={tf} value={tf}>{tf}</option>))}
            </select>
            <input type="number" min={200} max={5000} step={100} value={numBars}
              onChange={e => setNumBars(Math.max(200, Math.min(5000, parseInt(e.target.value || '1500', 10))))}
              style={{ width: 110 }} title="Bars" />
            <button className="btn" type="button" onClick={runBacktest} disabled={isLoading}>Run Backtest</button>
            <button className="btn" type="button" onClick={saveStrategy} disabled={isLoading || !lastResult?.stdout}>Save Strategy</button>
            <button className="btn" type="button" onClick={() => setView('auto')} disabled={view==='auto'}>Formatted</button>
            <button className="btn" type="button" onClick={() => setView('raw')} disabled={view==='raw'}>Raw JSON</button>
          </div>
        </div>
        {renderResult(lastResult, view)}
      </div>
    </div>
  );
}

function renderResult(lastResult: any, view: 'auto'|'raw') {
  if (!lastResult) return <div className="muted">No result yet.</div>;
  if (view === 'raw') return <CodeDisplay code={JSON.stringify(lastResult, null, 2)} />;
  if (lastResult.stdout) return <CodeDisplay code={lastResult.stdout} />;
  if (typeof lastResult === 'object' && lastResult && lastResult['Total Return [%]'] !== undefined) {
    return <BacktestResult metrics={lastResult} />;
  }
  return <CodeDisplay code={JSON.stringify(lastResult, null, 2)} />;
}
