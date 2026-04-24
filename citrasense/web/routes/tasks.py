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

        if getattr(tm, "current_task_id", None) == task_id:
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
        """Pause task processing for a sensor (or all sensors)."""
        if not ctx.daemon or not ctx.daemon.task_dispatcher:
            return JSONResponse({"error": "Task manager not available"}, status_code=503)

        sensor_id = (request or {}).get("sensor_id")
        ctx.daemon.task_dispatcher.pause()
        for sc in ctx.daemon.settings.sensors:
            if sensor_id is None or sc.id == sensor_id:
                sc.task_processing_paused = True
        ctx.daemon.settings.save()
        await ctx.broadcast_status()

        return {"status": "paused", "message": "Task processing paused"}

    @router.post("/tasks/resume")
    async def resume_tasks(request: dict | None = None):
        """Resume task processing for a sensor (or all sensors)."""
        if not ctx.daemon or not ctx.daemon.task_dispatcher:
            return JSONResponse({"error": "Task manager not available"}, status_code=503)

        sensor_id = (request or {}).get("sensor_id")
        ctx.daemon.task_dispatcher.resume()
        for sc in ctx.daemon.settings.sensors:
            if sensor_id is None or sc.id == sensor_id:
                sc.task_processing_paused = False
        ctx.daemon.settings.save()
        await ctx.broadcast_status()

        return {"status": "active", "message": "Task processing resumed"}

    @router.patch("/telescope/automated-scheduling")
    async def update_automated_scheduling(request: dict[str, Any]):
        """Toggle automated scheduling on/off for a sensor (or all sensors)."""
        if not ctx.daemon or not ctx.daemon.task_dispatcher:
            return JSONResponse({"error": "Task manager not available"}, status_code=503)

        if not ctx.daemon.api_client:
            return JSONResponse({"error": "API client not available"}, status_code=503)

        enabled = request.get("enabled")
        if enabled is None:
            return JSONResponse({"error": "Missing 'enabled' field in request body"}, status_code=400)

        sensor_id = request.get("sensor_id")

        try:
            updated = 0
            for trt in ctx.daemon.task_dispatcher._telescope_runtimes():
                ts = trt.sensor
                if not getattr(ts, "citra_record", None):
                    continue
                if sensor_id is not None and ts.sensor_id != sensor_id:
                    continue
                telescope_api_id = ts.citra_record["id"]
                success = ctx.daemon.api_client.update_telescope_automated_scheduling(telescope_api_id, enabled)
                if success:
                    ts.citra_record["automated_scheduling"] = enabled
                    updated += 1

            if updated == 0:
                return JSONResponse({"error": "No matching telescope found"}, status_code=404)

            CITRASENSE_LOGGER.info(
                "Automated scheduling set to %s for %d telescope(s)",
                "enabled" if enabled else "disabled",
                updated,
            )
            await ctx.broadcast_status()
            return {"status": "success", "enabled": enabled}

        except Exception as e:
            CITRASENSE_LOGGER.error(f"Error updating automated scheduling: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.patch("/observing-session")
    async def toggle_observing_session(request: dict[str, Any]):
        """Toggle observing session on/off for a sensor."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)

        enabled = request.get("enabled")
        if enabled is None:
            return JSONResponse({"error": "Missing 'enabled' field in request body"}, status_code=400)

        sensor_id = request.get("sensor_id")
        for sc in ctx.daemon.settings.sensors:
            if sensor_id is None or sc.id == sensor_id:
                sc.observing_session_enabled = enabled
        ctx.daemon.settings.save()
        CITRASENSE_LOGGER.info(f"Observing session set to {'enabled' if enabled else 'disabled'}")
        await ctx.broadcast_status()
        return {"status": "success", "enabled": enabled}

    @router.patch("/self-tasking")
    async def toggle_self_tasking(request: dict[str, Any]):
        """Toggle self-tasking on/off for a sensor.

        When enabling, also enables Observing Session, Scheduling
        (server-side), and Processing (local) so the autonomous
        pipeline is fully active.
        """
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)

        enabled = request.get("enabled")
        if enabled is None:
            return JSONResponse({"error": "Missing 'enabled' field in request body"}, status_code=400)

        sensor_id = request.get("sensor_id")
        target_configs = [sc for sc in ctx.daemon.settings.sensors if sensor_id is None or sc.id == sensor_id]
        for sc in target_configs:
            sc.self_tasking_enabled = enabled
        ctx.daemon.settings.save()

        if enabled:
            for sc in target_configs:
                if not sc.observing_session_enabled:
                    sc.observing_session_enabled = True
                    CITRASENSE_LOGGER.info("Self-tasking: auto-enabled observing session for %s", sc.id)
            ctx.daemon.settings.save()

        if enabled and ctx.daemon.task_dispatcher:
            if not ctx.daemon.task_dispatcher.is_processing_active():
                ctx.daemon.task_dispatcher.resume()
                for sc in target_configs:
                    sc.task_processing_paused = False
                ctx.daemon.settings.save()
                CITRASENSE_LOGGER.info("Self-tasking: auto-enabled processing")

            if not ctx.daemon.task_dispatcher.automated_scheduling:
                try:
                    for trt in ctx.daemon.task_dispatcher._telescope_runtimes():
                        ts = trt.sensor
                        if not getattr(ts, "citra_record", None):
                            continue
                        if sensor_id is not None and ts.sensor_id != sensor_id:
                            continue
                        tid = ts.citra_record["id"]
                        ok = ctx.daemon.api_client.update_telescope_automated_scheduling(tid, True)
                        if ok:
                            ts.citra_record["automated_scheduling"] = True
                            CITRASENSE_LOGGER.info("Self-tasking: auto-enabled scheduling for %s", ts.sensor_id)
                        else:
                            CITRASENSE_LOGGER.warning(
                                "Self-tasking: failed to enable scheduling on server for %s", ts.sensor_id
                            )
                except Exception as e:
                    CITRASENSE_LOGGER.warning(f"Self-tasking: could not enable scheduling: {e}")

        CITRASENSE_LOGGER.info(f"Self-tasking set to {'enabled' if enabled else 'disabled'}")
        await ctx.broadcast_status()
        return {"status": "success", "enabled": enabled}

    @router.post("/self-tasking/request-now")
    async def request_batch_now(request: dict[str, Any] | None = None):
        """Fire a single batch collection request, bypassing session-state and timer gating."""
        if not ctx.daemon or not ctx.daemon.task_dispatcher:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)

        gs = getattr(ctx.daemon, "ground_station", None)
        if not gs:
            return JSONResponse({"error": "Ground station not configured"}, status_code=503)

        req_sensor_id = (request or {}).get("sensor_id")

        target_ts = None
        for trt in ctx.daemon.task_dispatcher._telescope_runtimes():
            ts = trt.sensor
            if not getattr(ts, "citra_record", None):
                continue
            if req_sensor_id is None or ts.sensor_id == req_sensor_id:
                target_ts = ts
                break

        if not target_ts or not target_ts.citra_record:
            return JSONResponse({"error": "No matching telescope found"}, status_code=404)

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
