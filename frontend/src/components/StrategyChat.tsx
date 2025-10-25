import React, { useState } from 'react';

export interface ChatMessage {
  role: 'user' | 'assistant';
  type?: 'text' | 'code' | 'error';
  content: string;
}

interface StrategyChatProps {
  messages: ChatMessage[];
  isLoading?: boolean;
  placeholder?: string;
  onSendMessage: (message: string) => void;
}

export default function StrategyChat({ messages, isLoading, placeholder, onSendMessage }: StrategyChatProps) {
  const [input, setInput] = useState('');

  const send = () => {
    const text = input.trim();
    if (!text || isLoading) return;
    onSendMessage(text);
    setInput('');
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, height: '100%' }}>
      <div style={{ fontWeight: 600 }}>Strategy Chat</div>
      <div style={{ flex: 1, minHeight: 280, overflow: 'auto', background: '#0b0b0f', padding: 8, borderRadius: 6 }}>
        {messages.length === 0 ? (
          <div className="muted">Describe a task, e.g. "create an SMA crossover strategy" or "backtest RSI on XAUUSD".</div>
        ) : (
          messages.map((m, i) => (
            <div key={i} style={{ marginBottom: 10, display:'flex', justifyContent: m.role==='user'?'flex-end':'flex-start' }}>
              <div style={{ maxWidth: '80%', background: m.role==='user'? '#1b2437' : '#121a2b', border: '1px solid #1a2030', borderRadius: 10, padding: 8 }}>
                <div className="muted" style={{ marginBottom: 2 }}>{m.role === 'user' ? 'You' : 'Assistant'}</div>
                <div style={{ whiteSpace: 'pre-wrap' }}>{m.content}</div>
              </div>
            </div>
          ))
        )}
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') send(); }}
          placeholder={placeholder || 'Type your request...'}
          style={{ flex: 1 }}
        />
        <button className="btn primary" type="button" onClick={send} disabled={!!isLoading}>Send</button>
      </div>
    </div>
  );
}
