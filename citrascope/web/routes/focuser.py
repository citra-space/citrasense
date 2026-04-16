"""Focuser movement and abort endpoints."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from citrascope.logging import CITRASCOPE_LOGGER

if TYPE_CHECKING:
    from citrascope.web.app import CitraScopeWebApp


def build_focuser_router(ctx: CitraScopeWebApp) -> APIRouter:
    """Focuser move (absolute/relative) and abort."""
    router = APIRouter(prefix="/api", tags=["focuser"])

    @router.post("/focuser/move")
    async def focuser_move(body: dict[str, Any]):
        """Move the focuser to an absolute position or by relative steps.

        Jog moves are fire-and-forget: the command is issued and the
        endpoint returns immediately.  The UI tracks position via the
        status poll.  Issuing a move while the focuser is already moving
        stops the previous move first.
        """
        if busy := ctx._require_system_idle():
            return busy
        if not ctx.daemon or not ctx.daemon.hardware_adapter:
            return JSONResponse({"error": "Hardware adapter not initialized"}, status_code=503)

        focuser = ctx.daemon.hardware_adapter.focuser
        if focuser is None or not focuser.is_connected():
            return JSONResponse({"error": "No focuser connected"}, status_code=404)

        absolute = body.get("position")
        relative = body.get("relative")

        try:
            if await asyncio.to_thread(focuser.is_moving):
                await asyncio.to_thread(focuser.abort_move)
                await asyncio.sleep(0.1)

            if absolute is not None:
                if not isinstance(absolute, int):
                    return JSONResponse({"error": "position must be an integer"}, status_code=400)
                if not await asyncio.to_thread(focuser.move_absolute, absolute):
                    return JSONResponse({"error": "Move failed"}, status_code=500)
                pos = await asyncio.to_thread(focuser.get_position)
                return {"success": True, "position": pos}

            if relative is not None:
                if not isinstance(relative, int):
                    return JSONResponse({"error": "relative must be an integer"}, status_code=400)
                if not await asyncio.to_thread(focuser.move_relative, relative):
                    return JSONResponse({"error": "Move failed"}, status_code=500)
                pos = await asyncio.to_thread(focuser.get_position)
                return {"success": True, "position": pos}

        except Exception as e:
            CITRASCOPE_LOGGER.error(f"Focuser move error: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

        return JSONResponse({"error": "Provide 'position' (absolute) or 'relative' (steps)"}, status_code=400)

    @router.post("/focuser/abort")
    async def focuser_abort():
        """Stop focuser movement."""
        if not ctx.daemon or not ctx.daemon.hardware_adapter:
            return JSONResponse({"error": "Hardware adapter not initialized"}, status_code=503)

        focuser = ctx.daemon.hardware_adapter.focuser
        if focuser is None or not focuser.is_connected():
            return JSONResponse({"error": "No focuser connected"}, status_code=404)

        try:
            focuser.abort_move()
            return {"success": True, "position": focuser.get_position()}
        except Exception as e:
            CITRASCOPE_LOGGER.error(f"Focuser abort error: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    return router
