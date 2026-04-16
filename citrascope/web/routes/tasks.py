"""Task management endpoints: listing, pause/resume, scheduling toggles, self-tasking."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from citrascope.logging import CITRASCOPE_LOGGER
from citrascope.web.helpers import _task_to_dict

if TYPE_CHECKING:
    from citrascope.web.app import CitraScopeWebApp


def build_tasks_router(ctx: CitraScopeWebApp) -> APIRouter:
    """Task queue, pause/resume, scheduling, and self-tasking endpoints."""
    router = APIRouter(prefix="/api", tags=["tasks"])

    @router.get("/tasks")
    async def get_tasks():
        """Get scheduled task queue (not yet started or waiting to retry)."""
        if not ctx.daemon or not hasattr(ctx.daemon, "task_manager") or ctx.daemon.task_manager is None:
            return []

        task_manager = ctx.daemon.task_manager
        return [_task_to_dict(t) for t in task_manager.get_tasks_snapshot(exclude_active=True)]

    @router.get("/tasks/active")
    async def get_active_tasks():
        """Get currently executing tasks (all stages)."""
        if not ctx.daemon or not hasattr(ctx.daemon, "task_manager") or ctx.daemon.task_manager is None:
            return []

        tasks_by_stage = ctx.daemon.task_manager.get_tasks_by_stage()

        active = []
        for stage, tasks in tasks_by_stage.items():
            for task_info in tasks:
                active.append(
                    {
                        "id": task_info["task_id"],
                        "target": task_info.get("target_name", "unknown"),
                        "stage": stage,
                        "elapsed": task_info["elapsed"],
                        "status_msg": task_info.get("status_msg"),
                        "retry_scheduled_time": task_info.get("retry_scheduled_time"),
                        "is_being_executed": task_info.get("is_being_executed", False),
                    }
                )

        return active

    @router.post("/tasks/pause")
    async def pause_tasks():
        """Pause task processing."""
        if not ctx.daemon or not ctx.daemon.task_manager:
            return JSONResponse({"error": "Task manager not available"}, status_code=503)

        ctx.daemon.task_manager.pause()
        ctx.daemon.settings.task_processing_paused = True
        ctx.daemon.settings.save()
        await ctx.broadcast_status()

        return {"status": "paused", "message": "Task processing paused"}

    @router.post("/tasks/resume")
    async def resume_tasks():
        """Resume task processing."""
        if not ctx.daemon or not ctx.daemon.task_manager:
            return JSONResponse({"error": "Task manager not available"}, status_code=503)

        ctx.daemon.task_manager.resume()
        ctx.daemon.settings.task_processing_paused = False
        ctx.daemon.settings.save()
        await ctx.broadcast_status()

        return {"status": "active", "message": "Task processing resumed"}

    @router.patch("/telescope/automated-scheduling")
    async def update_automated_scheduling(request: dict[str, bool]):
        """Toggle automated scheduling on/off."""
        if not ctx.daemon or not ctx.daemon.task_manager:
            return JSONResponse({"error": "Task manager not available"}, status_code=503)

        if not ctx.daemon.api_client:
            return JSONResponse({"error": "API client not available"}, status_code=503)

        enabled = request.get("enabled")
        if enabled is None:
            return JSONResponse({"error": "Missing 'enabled' field in request body"}, status_code=400)

        try:
            telescope_id = ctx.daemon.telescope_record["id"]
            success = ctx.daemon.api_client.update_telescope_automated_scheduling(telescope_id, enabled)

            if success:
                ctx.daemon.task_manager.automated_scheduling = enabled
                CITRASCOPE_LOGGER.info(f"Automated scheduling set to {'enabled' if enabled else 'disabled'}")
                await ctx.broadcast_status()
                return {"status": "success", "enabled": enabled}
            return JSONResponse({"error": "Failed to update telescope on server"}, status_code=500)

        except Exception as e:
            CITRASCOPE_LOGGER.error(f"Error updating automated scheduling: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.patch("/observing-session")
    async def toggle_observing_session(request: dict[str, bool]):
        """Toggle observing session on/off."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)

        enabled = request.get("enabled")
        if enabled is None:
            return JSONResponse({"error": "Missing 'enabled' field in request body"}, status_code=400)

        ctx.daemon.settings.observing_session_enabled = enabled
        ctx.daemon.settings.save()
        CITRASCOPE_LOGGER.info(f"Observing session set to {'enabled' if enabled else 'disabled'}")
        await ctx.broadcast_status()
        return {"status": "success", "enabled": enabled}

    @router.patch("/self-tasking")
    async def toggle_self_tasking(request: dict[str, bool]):
        """Toggle self-tasking on/off.

        When enabling, also enables Observing Session, Scheduling
        (server-side), and Processing (local) so the autonomous
        pipeline is fully active.
        """
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)

        enabled = request.get("enabled")
        if enabled is None:
            return JSONResponse({"error": "Missing 'enabled' field in request body"}, status_code=400)

        ctx.daemon.settings.self_tasking_enabled = enabled
        ctx.daemon.settings.save()

        if enabled:
            if not ctx.daemon.settings.observing_session_enabled:
                ctx.daemon.settings.observing_session_enabled = True
                ctx.daemon.settings.save()
                CITRASCOPE_LOGGER.info("Self-tasking: auto-enabled observing session")

        if enabled and ctx.daemon.task_manager:
            if not ctx.daemon.task_manager.is_processing_active():
                ctx.daemon.task_manager.resume()
                ctx.daemon.settings.task_processing_paused = False
                ctx.daemon.settings.save()
                CITRASCOPE_LOGGER.info("Self-tasking: auto-enabled processing")

            if not ctx.daemon.task_manager.automated_scheduling:
                try:
                    telescope_id = ctx.daemon.telescope_record["id"]
                    success = ctx.daemon.api_client.update_telescope_automated_scheduling(telescope_id, True)
                    if success:
                        ctx.daemon.task_manager.automated_scheduling = True
                        CITRASCOPE_LOGGER.info("Self-tasking: auto-enabled scheduling")
                    else:
                        CITRASCOPE_LOGGER.warning("Self-tasking: failed to enable scheduling on server")
                except Exception as e:
                    CITRASCOPE_LOGGER.warning(f"Self-tasking: could not enable scheduling: {e}")

        CITRASCOPE_LOGGER.info(f"Self-tasking set to {'enabled' if enabled else 'disabled'}")
        await ctx.broadcast_status()
        return {"status": "success", "enabled": enabled}

    @router.post("/self-tasking/request-now")
    async def request_batch_now():
        """Fire a single batch collection request, bypassing session-state and timer gating."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)

        gs = getattr(ctx.daemon, "ground_station", None)
        tr = getattr(ctx.daemon, "telescope_record", None)
        if not gs or not tr:
            return JSONResponse({"error": "Ground station or telescope not configured"}, status_code=503)

        settings = ctx.daemon.settings
        ground_station_id = gs["id"]
        sensor_id = tr["id"]

        group_ids = settings.self_tasking_satellite_group_ids or None
        exclude_types = settings.self_tasking_exclude_object_types or None
        orbit_regimes = settings.self_tasking_include_orbit_regimes or None
        collection_type = settings.self_tasking_collection_type or "Track"

        now = datetime.now(timezone.utc)
        window_start = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        window_stop = (now + timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")

        CITRASCOPE_LOGGER.info(
            "Manual batch request: type=%s, window %s → %s, gs=%s, sensor=%s",
            collection_type,
            window_start,
            window_stop,
            ground_station_id,
            sensor_id,
        )

        try:
            result = await asyncio.to_thread(
                ctx.daemon.api_client.create_batch_collection_requests,
                window_start=window_start,
                window_stop=window_stop,
                ground_station_id=ground_station_id,
                sensor_id=sensor_id,
                discover_visible=not bool(group_ids),
                satellite_group_ids=group_ids,
                request_type=collection_type,
                exclude_types=exclude_types,
                include_orbit_regimes=orbit_regimes,
            )
        except Exception as e:
            CITRASCOPE_LOGGER.error("Manual batch request failed", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

        if result is None:
            return JSONResponse({"error": "API request failed"}, status_code=502)

        created = result.get("created", 0)
        CITRASCOPE_LOGGER.info("Manual batch request succeeded (created=%s)", created)
        return {"status": "ok", "created": created}

    return router
