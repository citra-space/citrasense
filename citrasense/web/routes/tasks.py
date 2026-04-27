"""Task management endpoints: listing, pause/resume, scheduling toggles, self-tasking."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from citrasense.logging import CITRASENSE_LOGGER
from citrasense.web.sky_enrichment import get_web_tasks

if TYPE_CHECKING:
    from citrasense.web.app import CitraSenseWebApp


def build_tasks_router(ctx: CitraSenseWebApp) -> APIRouter:
    """Task queue, pause/resume, scheduling, and self-tasking endpoints."""
    router = APIRouter(prefix="/api", tags=["tasks"])

    @router.get("/tasks")
    async def get_tasks():
        """Get scheduled task queue (not yet started or waiting to retry).

        Delegates to :func:`get_web_tasks` so this route and the WebSocket
        broadcaster in :mod:`citrasense.web.app` produce identical wire
        formats.  Sky enrichment (alt/az/compass/trend/peak) and any future
        derived fields live there.
        """
        return get_web_tasks(ctx.daemon, exclude_active=True)

    @router.get("/tasks/active")
    async def get_active_tasks():
        """Get currently executing tasks (all stages)."""
        if not ctx.daemon or not hasattr(ctx.daemon, "task_dispatcher") or ctx.daemon.task_dispatcher is None:
            return []

        tasks_by_stage = ctx.daemon.task_dispatcher.get_tasks_by_stage()

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
                        # Include sensor_id so the monitoring UI can filter
                        # per-sensor pipeline stages.  ``task_dispatcher``
                        # already enriches this in ``get_tasks_by_stage``.
                        "sensor_id": task_info.get("sensor_id"),
                        "sensor_type": task_info.get("sensor_type"),
                    }
                )

        return active

    @router.post("/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str):
        """Cancel a queued task.

        PUTs status=Canceled to the Citra API and removes the task from the
        local queue so the UI updates immediately. Refuses to cancel the
        currently-executing task (cancelling mid-imaging is not supported).
        """
        if not ctx.daemon or not ctx.daemon.api_client:
            return JSONResponse({"error": "API client not available"}, status_code=503)

        tm = getattr(ctx.daemon, "task_dispatcher", None)
        if tm is None:
            return JSONResponse({"error": "Task manager not available"}, status_code=503)

        active_ids = set(getattr(tm, "current_task_ids", {}).values())
        if task_id in active_ids:
            return JSONResponse(
                {"error": "Cannot cancel the currently executing task"},
                status_code=409,
            )

        try:
            success = await asyncio.to_thread(ctx.daemon.api_client.cancel_task, task_id)
        except Exception as e:
            CITRASENSE_LOGGER.error(f"Error cancelling task {task_id}: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

        if not success:
            return JSONResponse(
                {"error": "Server refused cancel (task not found or already terminal)"},
                status_code=409,
            )

        tm.drop_scheduled_task(task_id)
        CITRASENSE_LOGGER.info(f"Cancelled task {task_id} via web UI")
        await ctx.broadcast_tasks()
        return {"status": "ok", "task_id": task_id}

    @router.post("/tasks/pause")
    async def pause_tasks(request: dict | None = None):
        """Pause task processing for a specific sensor.

        ``sensor_id`` is required; there is no site-wide broadcast pause.
        Use the emergency-stop endpoint for a full site halt.
        """
        if not ctx.daemon or not ctx.daemon.task_dispatcher:
            return JSONResponse({"error": "Task manager not available"}, status_code=503)

        sensor_id = (request or {}).get("sensor_id")
        if not sensor_id:
            return JSONResponse({"error": "sensor_id is required"}, status_code=400)
        if ctx.daemon.task_dispatcher.get_runtime(sensor_id) is None:
            return JSONResponse({"error": f"Unknown sensor_id: {sensor_id}"}, status_code=400)

        ctx.daemon.task_dispatcher.pause_sensor(sensor_id)
        await ctx.broadcast_status()

        return {"status": "paused", "message": f"Task processing paused for {sensor_id}"}

    @router.post("/tasks/resume")
    async def resume_tasks(request: dict | None = None):
        """Resume task processing for a specific sensor.

        ``sensor_id`` is required.
        """
        if not ctx.daemon or not ctx.daemon.task_dispatcher:
            return JSONResponse({"error": "Task manager not available"}, status_code=503)

        sensor_id = (request or {}).get("sensor_id")
        if not sensor_id:
            return JSONResponse({"error": "sensor_id is required"}, status_code=400)
        if ctx.daemon.task_dispatcher.get_runtime(sensor_id) is None:
            return JSONResponse({"error": f"Unknown sensor_id: {sensor_id}"}, status_code=400)

        ctx.daemon.task_dispatcher.resume_sensor(sensor_id)
        await ctx.broadcast_status()

        return {"status": "active", "message": f"Task processing resumed for {sensor_id}"}

    @router.patch("/telescope/automated-scheduling")
    async def update_automated_scheduling(request: dict[str, Any]):
        """Toggle automated scheduling on/off for a specific telescope sensor.

        ``sensor_id`` is required; broadcast semantics were removed to avoid
        accidental site-wide flips in multi-sensor deployments.
        """
        if not ctx.daemon or not ctx.daemon.task_dispatcher:
            return JSONResponse({"error": "Task manager not available"}, status_code=503)

        if not ctx.daemon.api_client:
            return JSONResponse({"error": "API client not available"}, status_code=503)

        enabled = request.get("enabled")
        if enabled is None:
            return JSONResponse({"error": "Missing 'enabled' field in request body"}, status_code=400)

        sensor_id = request.get("sensor_id")
        if not sensor_id:
            return JSONResponse({"error": "sensor_id is required"}, status_code=400)

        try:
            updated = 0
            for trt in ctx.daemon.task_dispatcher._telescope_runtimes():
                ts = trt.sensor
                if not getattr(ts, "citra_record", None):
                    continue
                if ts.sensor_id != sensor_id:
                    continue
                telescope_api_id = ts.citra_record["id"]
                success = ctx.daemon.api_client.update_telescope_automated_scheduling(telescope_api_id, enabled)
                if success:
                    ts.citra_record["automatedScheduling"] = enabled
                    updated += 1

            if updated == 0:
                return JSONResponse({"error": f"No matching telescope for sensor_id={sensor_id}"}, status_code=404)

            CITRASENSE_LOGGER.info(
                "Automated scheduling set to %s for sensor %s",
                "enabled" if enabled else "disabled",
                sensor_id,
            )
            await ctx.broadcast_status()
            return {"status": "success", "enabled": enabled}

        except Exception as e:
            CITRASENSE_LOGGER.error(f"Error updating automated scheduling: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.patch("/observing-session")
    async def toggle_observing_session(request: dict[str, Any]):
        """Toggle observing session on/off for a specific sensor.

        ``sensor_id`` is required; there is no broadcast form.
        """
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)

        enabled = request.get("enabled")
        if enabled is None:
            return JSONResponse({"error": "Missing 'enabled' field in request body"}, status_code=400)

        sensor_id = request.get("sensor_id")
        if not sensor_id:
            return JSONResponse({"error": "sensor_id is required"}, status_code=400)

        sc = ctx.daemon.settings.get_sensor_config(sensor_id)
        if sc is None:
            return JSONResponse({"error": f"Unknown sensor_id: {sensor_id}"}, status_code=400)

        sc.observing_session_enabled = enabled
        ctx.daemon.settings.save()
        CITRASENSE_LOGGER.info(
            "Observing session set to %s for sensor %s",
            "enabled" if enabled else "disabled",
            sensor_id,
        )
        await ctx.broadcast_status()
        return {"status": "success", "enabled": enabled}

    @router.patch("/self-tasking")
    async def toggle_self_tasking(request: dict[str, Any]):
        """Toggle self-tasking on/off for a specific sensor.

        ``sensor_id`` is required. When enabling, also enables Observing
        Session, Scheduling (server-side), and Processing (local) for
        that sensor so the autonomous pipeline is fully active.
        """
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)

        enabled = request.get("enabled")
        if enabled is None:
            return JSONResponse({"error": "Missing 'enabled' field in request body"}, status_code=400)

        sensor_id = request.get("sensor_id")
        if not sensor_id:
            return JSONResponse({"error": "sensor_id is required"}, status_code=400)

        sc = ctx.daemon.settings.get_sensor_config(sensor_id)
        if sc is None:
            return JSONResponse({"error": f"Unknown sensor_id: {sensor_id}"}, status_code=400)

        sc.self_tasking_enabled = enabled
        if enabled and not sc.observing_session_enabled:
            sc.observing_session_enabled = True
            CITRASENSE_LOGGER.info("Self-tasking: auto-enabled observing session for %s", sensor_id)
        ctx.daemon.settings.save()

        if enabled and ctx.daemon.task_dispatcher:
            rt = ctx.daemon.task_dispatcher.get_runtime(sensor_id)
            if rt is not None and rt.paused:
                ctx.daemon.task_dispatcher.resume_sensor(sensor_id)
                CITRASENSE_LOGGER.info("Self-tasking: auto-enabled processing for %s", sensor_id)

            try:
                for trt in ctx.daemon.task_dispatcher._telescope_runtimes():
                    ts = trt.sensor
                    if ts.sensor_id != sensor_id or not getattr(ts, "citra_record", None):
                        continue
                    if ts.citra_record.get("automatedScheduling"):
                        continue
                    tid = ts.citra_record["id"]
                    ok = ctx.daemon.api_client.update_telescope_automated_scheduling(tid, True)
                    if ok:
                        ts.citra_record["automatedScheduling"] = True
                        CITRASENSE_LOGGER.info("Self-tasking: auto-enabled scheduling for %s", ts.sensor_id)
                    else:
                        CITRASENSE_LOGGER.warning(
                            "Self-tasking: failed to enable scheduling on server for %s", ts.sensor_id
                        )
            except Exception as e:
                CITRASENSE_LOGGER.warning(f"Self-tasking: could not enable scheduling: {e}")

        CITRASENSE_LOGGER.info(
            "Self-tasking set to %s for sensor %s",
            "enabled" if enabled else "disabled",
            sensor_id,
        )
        await ctx.broadcast_status()
        return {"status": "success", "enabled": enabled}

    @router.post("/self-tasking/request-now")
    async def request_batch_now(request: dict[str, Any] | None = None):
        """Fire a single batch collection request for a specific telescope.

        ``sensor_id`` is required; there is no implicit "first telescope"
        fallback.
        """
        if not ctx.daemon or not ctx.daemon.task_dispatcher:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)

        gs = getattr(ctx.daemon, "ground_station", None)
        if not gs:
            return JSONResponse({"error": "Ground station not configured"}, status_code=503)

        req_sensor_id = (request or {}).get("sensor_id")
        if not req_sensor_id:
            return JSONResponse({"error": "sensor_id is required"}, status_code=400)

        target_ts = None
        for trt in ctx.daemon.task_dispatcher._telescope_runtimes():
            ts = trt.sensor
            if not getattr(ts, "citra_record", None):
                continue
            if ts.sensor_id == req_sensor_id:
                target_ts = ts
                break

        if not target_ts or not target_ts.citra_record:
            return JSONResponse({"error": f"No matching telescope for sensor_id={req_sensor_id}"}, status_code=404)

        sc = ctx.daemon.settings.get_sensor_config(target_ts.sensor_id)
        ground_station_id = gs["id"]
        api_sensor_id = target_ts.citra_record["id"]

        group_ids = (sc.self_tasking_satellite_group_ids if sc else []) or None
        exclude_types = (sc.self_tasking_exclude_object_types if sc else []) or None
        orbit_regimes = (sc.self_tasking_include_orbit_regimes if sc else []) or None
        collection_type = (sc.self_tasking_collection_type if sc else "Track") or "Track"

        now = datetime.now(timezone.utc)
        window_start = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        window_stop = (now + timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")

        CITRASENSE_LOGGER.info(
            "Manual batch request: type=%s, window %s → %s, gs=%s, sensor=%s",
            collection_type,
            window_start,
            window_stop,
            ground_station_id,
            api_sensor_id,
        )

        try:
            result = await asyncio.to_thread(
                ctx.daemon.api_client.create_batch_collection_requests,
                window_start=window_start,
                window_stop=window_stop,
                ground_station_id=ground_station_id,
                sensor_id=api_sensor_id,
                discover_visible=not bool(group_ids),
                satellite_group_ids=group_ids,
                request_type=collection_type,
                exclude_types=exclude_types,
                include_orbit_regimes=orbit_regimes,
            )
        except Exception as e:
            CITRASENSE_LOGGER.error("Manual batch request failed", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

        if result is None:
            return JSONResponse({"error": "API request failed"}, status_code=502)

        created = result.get("created", 0)
        CITRASENSE_LOGGER.info("Manual batch request succeeded (created=%s)", created)
        return {"status": "ok", "created": created}

    return router
