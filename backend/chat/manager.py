from fastapi import WebSocket
import logging

class ConnectionManager:
    """Manages active WebSocket connections."""
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept a new connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logging.info(f"New chat connection: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        """Disconnect a websocket."""
        self.active_connections.remove(websocket)
        logging.info(f"Chat connection closed: {websocket.client}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a single websocket connection."""
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        """Send a message to all connected clients."""
        for connection in self.active_connections:
            await connection.send_text(message)

# Create a single instance of the manager to be used by the router
manager = ConnectionManager()