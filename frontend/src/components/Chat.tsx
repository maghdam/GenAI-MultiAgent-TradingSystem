import { useEffect, useRef, useState } from 'react';

type Sender = 'user' | 'bot';

interface ChatMessage {
  id: number;
  text: string;
  sender: Sender;
}

function createSocketUrl(): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/api/chat/ws/chat`;
}

export default function Chat() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>(() => ([
    { id: 0, text: 'Hello! Click the chat icon to connect.', sender: 'bot' },
  ]));
  const [input, setInput] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const nextIdRef = useRef(1);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const appendMessage = (text: string, sender: Sender) => {
    setMessages(prev => [...prev, { id: nextIdRef.current++, text, sender }]);
  };

  const disconnect = () => {
    if (socketRef.current) {
      socketRef.current.removeEventListener('open', handleOpen);
      socketRef.current.removeEventListener('message', handleMessage);
      socketRef.current.removeEventListener('close', handleClose);
      socketRef.current.removeEventListener('error', handleError);
      socketRef.current.close();
      socketRef.current = null;
    }
    setIsConnected(false);
  };

  const scheduleReconnect = () => {
    if (reconnectTimerRef.current) {
      return;
    }
    reconnectTimerRef.current = setTimeout(() => {
      reconnectTimerRef.current = null;
      if (isOpen) {
        connect();
      }
    }, 3000);
  };

  const handleOpen = () => {
    setIsConnected(true);
    appendMessage('Connected to AI assistant.', 'bot');
  };

  const handleMessage = (event: MessageEvent) => {
    appendMessage(event.data, 'bot');
  };

  const handleClose = () => {
    setIsConnected(false);
    appendMessage('Connection lost. Reconnecting in 3 seconds…', 'bot');
    scheduleReconnect();
  };

  const handleError = () => {
    appendMessage('An error occurred with the chat connection.', 'bot');
  };

  const connect = () => {
    disconnect();
    try {
      const socket = new WebSocket(createSocketUrl());
      socketRef.current = socket;
      socket.addEventListener('open', handleOpen);
      socket.addEventListener('message', handleMessage);
      socket.addEventListener('close', handleClose);
      socket.addEventListener('error', handleError);
    } catch (error) {
      console.error('Failed to establish chat websocket', error);
      appendMessage('Failed to connect. Retrying…', 'bot');
      scheduleReconnect();
    }
  };

  useEffect(() => {
    if (isOpen && !socketRef.current) {
      connect();
    }

    if (!isOpen) {
      disconnect();
    }

    return () => {
      if (!isOpen) {
        disconnect();
      }
    };
  }, [isOpen]);

  useEffect(() => {
    return () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      disconnect();
    };
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = () => {
    const trimmed = input.trim();
    if (!trimmed || !socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      return;
    }
    appendMessage(trimmed, 'user');
    socketRef.current.send(trimmed);
    setInput('');
  };

  const handleKeyPress: React.KeyboardEventHandler<HTMLInputElement> = event => {
    if (event.key === 'Enter') {
      event.preventDefault();
      sendMessage();
    }
  };

  return (
    <>
      <button
        id="chat-toggle"
        type="button"
        aria-expanded={isOpen}
        onClick={() => setIsOpen(prev => !prev)}
      >
        💬
      </button>

      <div id="chat-window" style={{ display: isOpen ? 'flex' : 'none' }}>
        <div id="chat-messages">
          {messages.map(message => (
            <div key={message.id} className={`chat-message ${message.sender}`}>
              {message.text}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
        <div id="chat-input-area">
          <input
            id="chat-input"
            type="text"
            value={input}
            onChange={event => setInput(event.target.value)}
            onKeyDown={handleKeyPress}
            placeholder={isConnected ? 'Type a message…' : 'Connecting…'}
            disabled={!isConnected}
          />
          <button
            id="chat-send"
            type="button"
            onClick={sendMessage}
            disabled={!isConnected}
          >
            Send
          </button>
        </div>
      </div>
    </>
  );
}