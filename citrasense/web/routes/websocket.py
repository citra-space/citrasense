"""WebSocket router for real-time updates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from citrasense.logging import CITRASENSE_LOGGER

if TYPE_CHECKING:
    from citrasense.web.app import CitraSenseWebApp


def build_websocket_router(ctx: CitraSenseWebApp) -> APIRouter:
    """WebSocket endpoint for real-time updates."""
    router = APIRouter()

    @router.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await ctx.connection_manager.connect(websocket)
        try:
            # Send initial status
            if ctx.daemon:
                ctx._update_status_from_daemon()
            await websocket.send_json({"type": "status", "data": ctx.status.model_dump()})

            while True:
                data = await websocket.receive_text()
                ctx.connection_manager.record_heard(websocket)
                if data != "ping":
                    await websocket.send_json({"type": "pong", "data": data})

        except WebSocketDisconnect:
            ctx.connection_manager.disconnect(websocket)
        except Exception as e:
            CITRASENSE_LOGGER.exception("WebSocket error: %s", e)
            await ctx.connection_manager.disconnect_and_close(websocket)

    return router
