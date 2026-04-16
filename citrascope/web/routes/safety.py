"""Safety monitor, emergency stop, and operator-stop endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from citrascope.logging import CITRASCOPE_LOGGER

if TYPE_CHECKING:
    from citrascope.web.app import CitraScopeWebApp


def build_safety_router(ctx: CitraScopeWebApp) -> APIRouter:
    """Safety status, cable-wrap reset, operator stop, and emergency stop."""
    router = APIRouter(prefix="/api", tags=["safety"])

    @router.get("/safety/status")
    async def get_safety_status():
        """Return status of all safety checks."""
        if not ctx.daemon or not ctx.daemon.safety_monitor:
            return {"checks": [], "watchdog_alive": False, "watchdog_last_heartbeat": 0}
        try:
            return ctx.daemon.safety_monitor.get_status()
        except Exception as e:
            CITRASCOPE_LOGGER.error(f"Error getting safety status: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.post("/safety/cable-wrap/reset")
    async def reset_cable_wrap():
        """Reset cable wrap counter to zero (operator confirms cables are straight)."""
        if not ctx.daemon or not ctx.daemon.safety_monitor:
            return JSONResponse({"error": "Safety monitor not available"}, status_code=503)

        from citrascope.safety.cable_wrap_check import CableWrapCheck

        chk = ctx.daemon.safety_monitor.get_check("cable_wrap")
        if not isinstance(chk, CableWrapCheck):
            return JSONResponse({"error": "No cable wrap check configured"}, status_code=404)
        if chk.is_unwinding:
            return JSONResponse({"error": "Cannot reset during unwind"}, status_code=409)
        chk.reset()
        CITRASCOPE_LOGGER.info("Cable wrap counter reset by operator")
        return {"success": True, "message": "Cable wrap counter reset to 0°"}

    @router.post("/emergency-stop")
    async def emergency_stop():
        """Stop mount, pause task processing, cancel in-flight imaging."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)

        import threading

        if ctx.daemon.safety_monitor:
            ctx.daemon.safety_monitor.activate_operator_stop()

        cancelled = 0
        tm = ctx.daemon.task_manager
        if tm:
            tm.pause()
            cancelled = tm.clear_pending_tasks()
        if ctx.daemon.settings:
            ctx.daemon.settings.task_processing_paused = True
            ctx.daemon.settings.save()

        # Immediate mount halt in background thread (serial I/O can't
        # run on the async event loop).  The watchdog provides ongoing
        # enforcement at 1 Hz; this gives sub-second first response.
        daemon = ctx.daemon

        def _halt_mount():
            mount = daemon.hardware_adapter.mount if daemon.hardware_adapter else None
            if not mount:
                return
            try:
                mount.abort_slew()
                mount.stop_tracking()
                for d in ("north", "south", "east", "west"):
                    mount.stop_move(d)
            except Exception:
                CITRASCOPE_LOGGER.error("Error halting mount during emergency stop", exc_info=True)

        threading.Thread(target=_halt_mount, daemon=True, name="emergency-stop").start()

        CITRASCOPE_LOGGER.warning(
            "EMERGENCY STOP by operator — processing paused, %d imaging task(s) cancelled, mount halt issued",
            cancelled,
        )
        return JSONResponse(
            {
                "success": True,
                "message": (f"Emergency stop: mount halted, processing paused, {cancelled} imaging task(s) cancelled"),
            },
            status_code=202,
        )

    @router.post("/safety/operator-stop/clear")
    async def clear_operator_stop():
        """Clear the operator stop — allows motion to resume.

        Processing stays paused; the operator must manually re-enable it.
        """
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        if not ctx.daemon.safety_monitor:
            return JSONResponse({"error": "Safety monitor not available"}, status_code=503)
        ctx.daemon.safety_monitor.clear_operator_stop()

        CITRASCOPE_LOGGER.info("Operator stop cleared — processing remains paused, re-enable manually")
        return {"success": True, "message": "Operator stop cleared — re-enable processing when ready"}

    return router
