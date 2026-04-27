"""Safety monitor, emergency stop, and operator-stop endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from citrasense.logging import CITRASENSE_LOGGER
from citrasense.web.helpers import get_sensor_context

if TYPE_CHECKING:
    from citrasense.web.app import CitraSenseWebApp


def build_safety_router(ctx: CitraSenseWebApp) -> APIRouter:
    """Safety status, cable-wrap reset, operator stop, and emergency stop."""
    router = APIRouter(tags=["safety"])

    # ── Site-level endpoints ──────────────────────────────────────────

    @router.get("/api/safety/status")
    async def get_safety_status():
        """Return status of all safety checks."""
        if not ctx.daemon or not ctx.daemon.safety_monitor:
            return {"checks": [], "watchdog_alive": False, "watchdog_last_heartbeat": 0}
        try:
            return ctx.daemon.safety_monitor.get_status()
        except Exception as e:
            CITRASENSE_LOGGER.error(f"Error getting safety status: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.post("/api/emergency-stop")
    async def emergency_stop():
        """Stop mount, pause task processing, cancel in-flight imaging."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)

        import threading

        if ctx.daemon.safety_monitor:
            ctx.daemon.safety_monitor.activate_operator_stop()

        cancelled = 0
        tm = ctx.daemon.task_dispatcher
        if tm:
            tm.pause_sensor(None)
            cancelled = tm.clear_pending_tasks()
        elif ctx.daemon.settings:
            for sc in ctx.daemon.settings.sensors:
                sc.task_processing_paused = True
            ctx.daemon.settings.save()

        daemon = ctx.daemon

        def _halt_mount():
            if not daemon.sensor_manager:
                return
            for sensor in daemon.sensor_manager:
                mount = getattr(getattr(sensor, "adapter", None), "mount", None)
                if not mount:
                    continue
                try:
                    mount.abort_slew()
                    mount.stop_tracking()
                    for d in ("north", "south", "east", "west"):
                        mount.stop_move(d)
                except Exception:
                    CITRASENSE_LOGGER.error(
                        "Error halting mount on %s during emergency stop", sensor.sensor_id, exc_info=True
                    )

        threading.Thread(target=_halt_mount, daemon=True, name="emergency-stop").start()

        CITRASENSE_LOGGER.warning(
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

    @router.post("/api/safety/operator-stop/clear")
    async def clear_operator_stop():
        """Clear the operator stop — allows motion to resume."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        if not ctx.daemon.safety_monitor:
            return JSONResponse({"error": "Safety monitor not available"}, status_code=503)
        ctx.daemon.safety_monitor.clear_operator_stop()

        CITRASENSE_LOGGER.info("Operator stop cleared — processing remains paused, re-enable manually")
        return {"success": True, "message": "Operator stop cleared — re-enable processing when ready"}

    # ── Per-sensor safety endpoints ───────────────────────────────────

    @router.post("/api/sensors/{sensor_id}/safety/cable-wrap/reset")
    async def reset_cable_wrap(sensor_id: str):
        """Reset cable wrap counter to zero (operator confirms cables are straight)."""
        if not ctx.daemon or not ctx.daemon.safety_monitor:
            return JSONResponse({"error": "Safety monitor not available"}, status_code=503)
        get_sensor_context(ctx, sensor_id)

        from citrasense.sensors.telescope.safety.cable_wrap_check import CableWrapCheck

        checks = ctx.daemon.safety_monitor.get_sensor_checks(sensor_id)
        chk = next((c for c in checks if isinstance(c, CableWrapCheck)), None)
        if chk is None:
            return JSONResponse({"error": "No cable wrap check configured"}, status_code=404)
        if chk.is_unwinding:
            return JSONResponse({"error": "Cannot reset during unwind"}, status_code=409)
        chk.reset()
        CITRASENSE_LOGGER.info("Cable wrap counter reset by operator (sensor %s)", sensor_id)
        return {"success": True, "message": "Cable wrap counter reset to 0°"}

    return router
