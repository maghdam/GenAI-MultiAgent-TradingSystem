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
  headerRight?: React.ReactNode;
}

export default function StrategyChat({ messages, isLoading, placeholder, onSendMessage, headerRight }: StrategyChatProps) {
  const [input, setInput] = useState('');

  const send = () => {
    const text = input.trim();
    if (!text || isLoading) return;
    onSendMessage(text);
    setInput('');
  };

  return (
    <div className="studio-chat">
      <div className="studio-chat__header">
        <div>
          <strong>Strategy assistant</strong>
          <span>Research and refine the strategy without changing deployment state.</span>
        </div>
        {headerRight}
      </div>

      <div className="studio-chat__messages">
        {messages.length === 0 ? (
          <div className="studio-empty">
            Describe the market idea, setup, entry/exit logic, or improvement you want to test.
          </div>
        ) : (
          messages.map((message, index) => (
            <div
              className={`studio-chat__message studio-chat__message--${message.role}`}
              key={`${message.role}-${index}`}
            >
              <div className={`studio-chat__bubble ${message.type === 'error' ? 'studio-chat__bubble--error' : ''}`}>
                <small>{message.role === 'user' ? 'You' : 'Assistant'}</small>
                <div>{message.content}</div>
              </div>
            </div>
          ))
        )}
      </div>

      <div className="studio-chat__composer">
        <input
          className="ta-input"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') send();
          }}
          placeholder={placeholder || 'Type your request...'}
        />
        <button className="ta-btn ta-btn--primary" type="button" onClick={send} disabled={!!isLoading}>
          {isLoading ? 'Working…' : 'Send'}
        </button>
      </div>
    </div>
  );
}
