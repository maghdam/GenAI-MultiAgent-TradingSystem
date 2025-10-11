document.addEventListener('DOMContentLoaded', () => {
    const chatWindow = document.getElementById('chat-window');
    const chatToggle = document.getElementById('chat-toggle');
    const chatInput = document.getElementById('chat-input');
    const chatSend = document.getElementById('chat-send');
    const messagesContainer = document.getElementById('chat-messages');

    let socket = null;

    function connect() {
        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        socket = new WebSocket(`${proto}//${host}/api/chat/ws/chat`);

        socket.onopen = () => {
            console.log('Chat WebSocket connected.');
            addMessage('Connected to AI assistant.', 'bot');
        };

        socket.onmessage = (event) => {
            addMessage(event.data, 'bot');
        };

        socket.onclose = () => {
            console.log('Chat WebSocket disconnected. Reconnecting...');
            addMessage('Connection lost. Reconnecting in 3 seconds...', 'bot');
            setTimeout(connect, 3000);
        };

        socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            addMessage('An error occurred with the connection.', 'bot');
        };
    }

    if (chatToggle && chatWindow) {
        chatToggle.addEventListener('click', () => {
            const isHidden = chatWindow.style.display === 'none' || chatWindow.style.display === '';
            chatWindow.style.display = isHidden ? 'flex' : 'none';
            if (isHidden) {
                chatInput.focus();
                if (!socket || socket.readyState === WebSocket.CLOSED) {
                    connect();
                }
            }
        });
    }

    function addMessage(text, sender) {
        const messageEl = document.createElement('div');
        messageEl.className = `chat-message ${sender}`;
        messageEl.textContent = text;
        messagesContainer.appendChild(messageEl);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function sendMessage() {
        const message = chatInput.value.trim();
        if (message && socket && socket.readyState === WebSocket.OPEN) {
            addMessage(message, 'user');
            socket.send(message);
            chatInput.value = '';
        }
    }

    chatSend.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // Initial welcome message
    addMessage('Hello! Click the chat icon to connect.', 'bot');
});