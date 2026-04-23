"""Mount movement, homing, limits, goto, and tracking endpoints."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from citrasense.logging import CITRASENSE_LOGGER

if TYPE_CHECKING:
    from citrasense.web.app import CitraSenseWebApp


def build_mount_router(ctx: CitraSenseWebApp) -> APIRouter:
    """Mount home, limits, unwind, move, goto, tracking."""
    router = APIRouter(prefix="/api", tags=["mount"])

    @router.post("/mount/home")
    async def trigger_mount_home():
        """Request mount homing — queued to run when imaging is idle."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        if not ctx.daemon.task_manager:
            return JSONResponse({"error": "Task manager not available"}, status_code=503)

        try:
            success = ctx.daemon.task_manager.homing_manager.request()
            if success:
                return {"success": True, "message": "Mount homing queued — will run when imaging is idle"}
            return JSONResponse({"error": "Homing already in progress"}, status_code=409)
        except Exception as e:
            CITRASENSE_LOGGER.error(f"Error requesting mount homing: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.get("/mount/limits")
    async def get_mount_limits():
        """Get the mount's altitude limits."""
        if not ctx.daemon or not ctx.daemon.hardware_adapter:
            return JSONResponse({"error": "Hardware not available"}, status_code=503)
        try:
            h_limit, o_limit = ctx.daemon.hardware_adapter.get_mount_limits()
            return {"horizon_limit": h_limit, "overhead_limit": o_limit}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.post("/mount/limits")
    async def set_mount_limits(request: dict[str, Any]):
        """Set the mount's altitude limits."""
        if busy := ctx._require_system_idle():
            return busy
        if not ctx.daemon or not ctx.daemon.hardware_adapter:
            return JSONResponse({"error": "Hardware not available"}, status_code=503)
        adapter = ctx.daemon.hardware_adapter
        results: dict[str, Any] = {}
        try:
            if "horizon_limit" in request:
                results["horizon_ok"] = adapter.set_mount_horizon_limit(int(request["horizon_limit"]))
            if "overhead_limit" in request:
                results["overhead_ok"] = adapter.set_mount_overhead_limit(int(request["overhead_limit"]))
            return {"success": True, **results}
        except Exception as e:
            CITRASENSE_LOGGER.error("Error setting mount limits: %s", e, exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.post("/mount/unwind")
    async def trigger_cable_unwind():
        """Manually trigger cable unwind in a background thread."""
        if not ctx.daemon or not ctx.daemon.safety_monitor:
            return JSONResponse({"error": "Safety monitor not available"}, status_code=503)

        import threading

        from citrasense.sensors.telescope.safety.cable_wrap_check import CableWrapCheck

        chk = ctx.daemon.safety_monitor.get_check("cable_wrap")
        if not isinstance(chk, CableWrapCheck):
            return JSONResponse({"error": "No cable wrap check configured"}, status_code=404)
        if chk.is_unwinding:
            return JSONResponse({"error": "Unwind already in progress"}, status_code=409)
        threading.Thread(target=chk.execute_action, daemon=True, name="cable-unwind").start()
        return JSONResponse({"success": True, "message": "Cable unwind started"}, status_code=202)

    @router.post("/mount/move")
    async def mount_move(body: dict[str, Any]):
        """Start or stop directional mount motion (jog control).

        In alt-az mode: north=up, south=down, east=right, west=left.
        """
        if busy := ctx._require_system_idle():
            return busy
        if not ctx.daemon or not ctx.daemon.hardware_adapter:
            return JSONResponse({"error": "Hardware adapter not initialized"}, status_code=503)

        mount = ctx.daemon.hardware_adapter.mount
        if mount is None or not ctx.daemon.hardware_adapter.is_telescope_connected():
            return JSONResponse({"error": "No mount connected"}, status_code=404)

        action = body.get("action")
        direction = body.get("direction")
        valid_directions = ("north", "south", "east", "west")

        if direction not in valid_directions:
            return JSONResponse({"error": f"direction must be one of {valid_directions}"}, status_code=400)

        try:
            if action == "start":
                rate = body.get("rate")
                if rate is not None:
                    if isinstance(rate, bool) or not isinstance(rate, int) or not 0 <= rate <= mount.max_move_rate:
                        return JSONResponse(
                            {"error": f"rate must be an integer 0-{mount.max_move_rate}"}, status_code=400
                        )
                ok = await asyncio.to_thread(mount.start_move, direction, rate)
                return {"success": ok}
            if action == "stop":
                ok = await asyncio.to_thread(mount.stop_move, direction)
                return {"success": ok}
            return JSONResponse({"error": "action must be 'start' or 'stop'"}, status_code=400)
        except Exception as e:
            CITRASENSE_LOGGER.error(f"Mount move error: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.post("/mount/goto")
    async def mount_goto(body: dict[str, Any]):
        """Slew the mount to arbitrary RA/Dec coordinates (degrees)."""
        if busy := ctx._require_system_idle():
            return busy
        if not ctx.daemon or not ctx.daemon.hardware_adapter:
            return JSONResponse({"error": "Hardware adapter not initialized"}, status_code=503)

        mount = ctx.daemon.hardware_adapter.mount
        if mount is None or not ctx.daemon.hardware_adapter.is_telescope_connected():
            return JSONResponse({"error": "No mount connected"}, status_code=404)

        ra = body.get("ra")
        dec = body.get("dec")
        if not isinstance(ra, (int, float)) or not isinstance(dec, (int, float)):
            return JSONResponse({"error": "ra and dec must be numbers (degrees)"}, status_code=400)
        if not 0 <= float(ra) <= 360:
            return JSONResponse({"error": "ra must be 0-360"}, status_code=400)
        if not -90 <= float(dec) <= 90:
            return JSONResponse({"error": "dec must be -90 to 90"}, status_code=400)

        try:
            ok = await asyncio.to_thread(mount.slew_to_radec, float(ra), float(dec))
            if not ok:
                return JSONResponse({"error": "Mount rejected slew command"}, status_code=500)
            return {"success": True, "message": f"Slewing to RA={ra:.4f}, Dec={dec:.4f}"}
        except Exception as e:
            CITRASENSE_LOGGER.error(f"Mount goto error: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.post("/mount/tracking")
    async def mount_tracking(body: dict[str, Any]):
        """Start or stop sidereal tracking."""
        if busy := ctx._require_system_idle():
            return busy
        if not ctx.daemon or not ctx.daemon.hardware_adapter:
            return JSONResponse({"error": "Hardware adapter not initialized"}, status_code=503)

        mount = ctx.daemon.hardware_adapter.mount
        if mount is None or not ctx.daemon.hardware_adapter.is_telescope_connected():
            return JSONResponse({"error": "No mount connected"}, status_code=404)

        enabled = body.get("enabled")
        if not isinstance(enabled, bool):
            return JSONResponse({"error": "enabled must be a boolean"}, status_code=400)

        try:
            if enabled:
                ok = await asyncio.to_thread(mount.start_tracking)
            else:
                ok = await asyncio.to_thread(mount.stop_tracking)
            return {"success": ok, "tracking": enabled}
        except Exception as e:
            CITRASENSE_LOGGER.error(f"Mount tracking error: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    return router
