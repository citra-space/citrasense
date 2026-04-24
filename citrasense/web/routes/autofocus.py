"""Autofocus trigger, cancel, and preset endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from citrasense.constants import AUTOFOCUS_TARGET_PRESETS
from citrasense.logging import CITRASENSE_LOGGER
from citrasense.web.helpers import get_sensor_context

if TYPE_CHECKING:
    from citrasense.web.app import CitraSenseWebApp


def build_autofocus_router(ctx: CitraSenseWebApp) -> APIRouter:
    """Autofocus trigger, cancel, and preset listing."""
    router = APIRouter(prefix="/api/sensors/{sensor_id}", tags=["autofocus"])

    @router.post("/autofocus")
    async def trigger_autofocus(sensor_id: str):
        """Request autofocus to run between tasks."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        sensor, _runtime = get_sensor_context(ctx, sensor_id)
        if not sensor.adapter.supports_filter_management():
            return JSONResponse({"error": "Filter management not supported"}, status_code=404)

        try:
            success, error = ctx.daemon.trigger_autofocus(sensor_id=sensor_id)
            if success:
                return {"success": True, "message": "Autofocus queued - will run between tasks"}
            return JSONResponse({"error": error}, status_code=500)
        except Exception as e:
            CITRASENSE_LOGGER.error(f"Error queueing autofocus: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.post("/autofocus/cancel")
    async def cancel_autofocus(sensor_id: str):
        """Cancel autofocus — works whether queued or actively running."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        get_sensor_context(ctx, sensor_id)
        try:
            was_cancelled = ctx.daemon.cancel_autofocus(sensor_id=sensor_id)
            return {"success": was_cancelled}
        except Exception as e:
            CITRASENSE_LOGGER.error(f"Error cancelling autofocus: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.get("/autofocus/presets")
    async def get_autofocus_presets(sensor_id: str):
        """Return available autofocus target star presets."""
        get_sensor_context(ctx, sensor_id)
        presets = [{"key": key, **preset} for key, preset in AUTOFOCUS_TARGET_PRESETS.items()]
        return {"presets": presets}

    return router
