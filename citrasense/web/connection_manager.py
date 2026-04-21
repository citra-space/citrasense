"""WebSocket connection management and streaming helpers for the web layer."""

import asyncio
import time

from fastapi import WebSocket

from citrasense.logging import CITRASENSE_LOGGER


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    _SEND_TIMEOUT = 5.0
    _STALE_THRESHOLD = 60.0

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._last_heard: dict[WebSocket, float] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self._last_heard[websocket] = time.time()
        CITRASENSE_LOGGER.info(f"WebSocket client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        self._last_heard.pop(websocket, None)
        CITRASENSE_LOGGER.info(f"WebSocket client disconnected. Total: {len(self.active_connections)}")

    async def disconnect_and_close(self, websocket: WebSocket) -> None:
        """Remove a client from the active set and close the underlying socket."""
        self.disconnect(websocket)
        try:
            await websocket.close()
        except Exception:
            pass

    def record_heard(self, websocket: WebSocket) -> None:
        """Update last-heard timestamp when a client sends a message."""
        self._last_heard[websocket] = time.time()

    async def prune_stale(self) -> None:
        """Disconnect clients that haven't sent a heartbeat recently."""
        now = time.time()
        stale = [ws for ws, ts in self._last_heard.items() if now - ts > self._STALE_THRESHOLD]
        for ws in stale:
            age = now - self._last_heard.get(ws, 0)
            CITRASENSE_LOGGER.info("Pruning stale WebSocket client (no heartbeat for %.0fs)", age)
            await self.disconnect_and_close(ws)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients with per-client timeout."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await asyncio.wait_for(connection.send_json(message), timeout=self._SEND_TIMEOUT)
            except (asyncio.TimeoutError, Exception) as e:
                CITRASENSE_LOGGER.warning(f"Failed to send to WebSocket client: {e}")
                disconnected.append(connection)

        for connection in disconnected:
            await self.disconnect_and_close(connection)


class _TarStreamBuffer:
    """Write-only file-like object that accumulates bytes for streaming tar output."""

    def __init__(self) -> None:
        self._chunks: list[bytes] = []

    def write(self, data: bytes) -> int:
        self._chunks.append(data)
        return len(data)

    def flush(self) -> None:
        pass

    def drain(self) -> bytes:
        out = b"".join(self._chunks)
        self._chunks.clear()
        return out
