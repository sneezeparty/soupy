from __future__ import annotations

import asyncio
from typing import Set

from fastapi import WebSocket


class WebsocketManager:
    """Tracks connected WebSocket clients and broadcasts messages to all."""

    def __init__(self) -> None:
        self._connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast_text(self, message: str) -> None:
        # Snapshot connections to avoid holding the lock while sending
        async with self._lock:
            connections = list(self._connections)
        if not connections:
            return
        to_remove = []
        for ws in connections:
            try:
                await ws.send_text(message)
            except Exception:
                to_remove.append(ws)
        if to_remove:
            async with self._lock:
                for ws in to_remove:
                    self._connections.discard(ws)


