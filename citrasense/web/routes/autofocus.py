"""Autofocus trigger, cancel, and preset endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from citrasense.constants import AUTOFOCUS_TARGET_PRESETS
from citrasense.logging import CITRASENSE_LOGGER

if TYPE_CHECKING:
    from citrasense.web.app import CitraSenseWebApp


def build_autofocus_router(ctx: CitraSenseWebApp) -> APIRouter:
    """Autofocus trigger, cancel, and preset listing."""
    router = APIRouter(prefix="/api", tags=["autofocus"])

    @router.post("/adapter/autofocus")
    async def trigger_autofocus():
        """Request autofocus to run between tasks."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)

        if not ctx.daemon.hardware_adapter or not ctx.daemon.hardware_adapter.supports_filter_management():
            return JSONResponse({"error": "Filter management not supported"}, status_code=404)

        try:
            success, error = ctx.daemon.trigger_autofocus()
            if success:
                return {"success": True, "message": "Autofocus queued - will run between tasks"}
            return JSONResponse({"error": error}, status_code=500)
        except Exception as e:
            CITRASENSE_LOGGER.error(f"Error queueing autofocus: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.post("/adapter/autofocus/cancel")
    async def cancel_autofocus():
        """Cancel autofocus — works whether queued or actively running."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)

        try:
            was_cancelled = ctx.daemon.cancel_autofocus()
            return {"success": was_cancelled}
        except Exception as e:
            CITRASENSE_LOGGER.error(f"Error cancelling autofocus: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.get("/adapter/autofocus/presets")
    async def get_autofocus_presets():
        """Return available autofocus target star presets."""
        presets = [{"key": key, **preset} for key, preset in AUTOFOCUS_TARGET_PRESETS.items()]
        return {"presets": presets}

    return router
