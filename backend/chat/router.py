from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from .manager import manager
from .service import process_message

router = APIRouter()

@router.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            response = await process_message(data, websocket)
            await manager.send_personal_message(response, websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client #{websocket.client} left the chat")