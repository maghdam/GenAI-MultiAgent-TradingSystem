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

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', padding: '16px' }}>
      <div className="box" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <StrategyChat messages={messages} isLoading={isLoading} onSendMessage={send} />
      </div>
      <div className="box">
        <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom: 8 }}>
          <div style={{ fontWeight: 600 }}>Result</div>
          <div className="stack">
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
