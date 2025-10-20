import { useEffect, useRef, useState } from 'react';

type Role = 'user' | 'assistant';
interface Msg { role: Role; content: string }

export default function ChatWidget() {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Msg[]>([]);
  const [loading, setLoading] = useState(false);
  const clientIdRef = useRef<string>('');
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const key = 'chatClientId';
    let cid = localStorage.getItem(key);
    if (!cid) {
      cid = crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2);
      localStorage.setItem(key, cid);
    }
    clientIdRef.current = cid;
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, open]);

  const send = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput('');
    setMessages(m => [...m, { role: 'user', content: text }]);

    setLoading(true);
    try {
      const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, client_id: clientIdRef.current }),
      });
      if (!res.body) throw new Error('No response body');
      // Append empty assistant message to stream into
      setMessages(m => [...m, { role: 'assistant', content: '' }]);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        setMessages(m => {
          const last = m[m.length - 1];
          if (!last || last.role !== 'assistant') return [...m, { role: 'assistant', content: chunk }];
          const next = m.slice(0, -1);
          next.push({ role: 'assistant', content: last.content + chunk });
          return next;
        });
      }
    } catch (e: any) {
      setMessages(m => [...m, { role: 'assistant', content: 'Failed to get reply. Please try again.' }]);
    } finally {
      setLoading(false);
    }
  };

  const onSubmit: React.FormEventHandler<HTMLFormElement> = (e) => {
    e.preventDefault();
    send();
  };

  return (
    <div>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          position: 'fixed', bottom: 20, right: 20, width: 56, height: 56,
          borderRadius: '50%', border: 'none', background: 'var(--accent)',
          color: 'var(--bg)', fontSize: 18, cursor: 'pointer', zIndex: 1000,
          boxShadow: '0 4px 8px rgba(0,0,0,0.2)'
        }}
        aria-expanded={open}
        aria-label={open ? 'Close chat' : 'Open chat'}
      >
        {open ? '×' : 'Chat'}
      </button>

      {open && (
        <div style={{
          position: 'fixed', bottom: 90, right: 20, width: 420, maxWidth: '92vw',
          height: 520, maxHeight: '75vh', background: 'var(--panel)', color: '#e5e9f0',
          border: '1px solid #171a24', borderRadius: 12, display: 'flex', flexDirection: 'column',
          zIndex: 1000, boxShadow: '0 5px 15px rgba(0,0,0,0.3)'
        }}>
          <div style={{ padding: '10px 12px', borderBottom: '1px solid #171a24', display: 'flex' }}>
            <div style={{ fontWeight: 600 }}>Assistant</div>
            <div style={{ marginLeft: 'auto', fontSize: 12, opacity: 0.8 }}>{loading ? 'Responding…' : 'Ready'}</div>
          </div>
          <div style={{ flex: 1, overflowY: 'auto', padding: 12, display: 'flex', flexDirection: 'column', gap: 10 }}>
            {messages.map((m, i) => (
              <div key={i} style={{ textAlign: m.role === 'user' ? 'right' as const : 'left' as const }}>
                <div style={{
                  display: 'inline-block', borderRadius: 12, padding: '8px 10px',
                  background: m.role === 'user' ? '#1b2437' : '#0e1117',
                  border: m.role === 'assistant' ? '1px solid #1a2030' : 'none'
                }}>{m.content}</div>
              </div>
            ))}
            <div ref={scrollRef} />
          </div>
          <form onSubmit={onSubmit} style={{ display: 'flex', gap: 8, padding: 10, borderTop: '1px solid #171a24' }}>
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder={loading ? 'Sending…' : 'Type a message…'}
              disabled={loading}
              style={{ flex: 1, borderRadius: 8, border: '1px solid #22283a', background: '#0e1117', color: '#e5e9f0', padding: '8px 10px' }}
            />
            <button type="submit" disabled={loading} style={{
              border: 'none', borderRadius: 8, padding: '8px 12px', background: 'var(--accent)', color: 'var(--bg)', cursor: 'pointer'
            }}>Send</button>
          </form>
        </div>
      )}
    </div>
  );
}

