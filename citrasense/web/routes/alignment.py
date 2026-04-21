"""Alignment, pointing-model calibration, and manual sync endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from citrasense.logging import CITRASENSE_LOGGER

if TYPE_CHECKING:
    from citrasense.web.app import CitraSenseWebApp


def build_alignment_router(ctx: CitraSenseWebApp) -> APIRouter:
    """Alignment, pointing model, and sync endpoints."""
    router = APIRouter(prefix="/api", tags=["alignment"])

    @router.post("/adapter/alignment")
    async def trigger_alignment():
        """Request plate-solve alignment to run between tasks."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        if not ctx.daemon.task_manager:
            return JSONResponse({"error": "Task manager not available"}, status_code=503)

        try:
            ctx.daemon.task_manager.alignment_manager.request()
            return {"success": True, "message": "Alignment queued — will run between tasks"}
        except Exception as e:
            CITRASENSE_LOGGER.error(f"Error queueing alignment: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.post("/adapter/alignment/cancel")
    async def cancel_alignment():
        """Cancel pending alignment request."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        if not ctx.daemon.task_manager:
            return JSONResponse({"error": "Task manager not available"}, status_code=503)

        try:
            was_cancelled = ctx.daemon.task_manager.alignment_manager.cancel()
            return {"success": was_cancelled}
        except Exception as e:
            CITRASENSE_LOGGER.error(f"Error cancelling alignment: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.post("/mount/pointing-model/calibrate")
    async def calibrate_pointing_model():
        """Trigger a full pointing model calibration run."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        if not ctx.daemon.task_manager:
            return JSONResponse({"error": "Task manager not available"}, status_code=503)

        alignment_mgr = ctx.daemon.task_manager.alignment_manager
        if alignment_mgr.is_calibrating():
            return JSONResponse({"error": "Calibration already running"}, status_code=409)

        try:
            ok = alignment_mgr.request_calibration()
            if ok:
                return {"success": True, "message": "Pointing calibration queued"}
            return JSONResponse({"error": "Calibration request rejected"}, status_code=409)
        except Exception as e:
            CITRASENSE_LOGGER.error(f"Error starting pointing calibration: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.post("/mount/pointing-model/reset")
    async def reset_pointing_model():
        """Clear the pointing model and persisted state."""
        if busy := ctx._require_system_idle():
            return busy
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)

        adapter = ctx.daemon.hardware_adapter
        if adapter.pointing_model:
            adapter.pointing_model.reset()
            return {"success": True, "message": "Pointing model reset"}
        return JSONResponse({"error": "Pointing model not available"}, status_code=404)

    @router.post("/mount/pointing-model/calibrate/cancel")
    async def cancel_pointing_calibration():
        """Cancel an in-progress or pending pointing calibration."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        if not ctx.daemon.task_manager:
            return JSONResponse({"error": "Task manager not available"}, status_code=503)

        try:
            ctx.daemon.task_manager.alignment_manager.cancel_calibration()
            return {"success": True}
        except Exception as e:
            CITRASENSE_LOGGER.error(f"Error cancelling pointing calibration: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.post("/adapter/sync")
    async def manual_sync(request: dict[str, Any]):
        """Manually sync the mount to given RA/Dec coordinates."""
        if busy := ctx._require_system_idle():
            return busy
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        if not ctx.daemon.hardware_adapter:
            return JSONResponse({"error": "Hardware adapter not available"}, status_code=503)

        ra = request.get("ra")
        dec = request.get("dec")
        if ra is None or dec is None:
            return JSONResponse({"error": "Both 'ra' and 'dec' are required (degrees)"}, status_code=400)

        try:
            ra_f = float(ra)
            dec_f = float(dec)
        except (TypeError, ValueError):
            return JSONResponse({"error": "RA and Dec must be numeric (degrees)"}, status_code=400)

        mount = ctx.daemon.hardware_adapter.mount
        if not mount:
            return JSONResponse({"error": "No mount connected"}, status_code=404)

        try:
            success = mount.sync_to_radec(ra_f, dec_f)
            if success:
                return {"success": True, "message": f"Mount synced to RA={ra_f:.4f}°, Dec={dec_f:.4f}°"}
            return JSONResponse({"error": "Mount sync returned failure"}, status_code=500)
        except NotImplementedError:
            return JSONResponse({"error": "Mount does not support sync"}, status_code=404)
        except Exception as e:
            CITRASENSE_LOGGER.error(f"Manual sync failed: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    return router
