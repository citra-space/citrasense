"""Calibration library, capture, and master frame management endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from citrasense.logging import CITRASENSE_LOGGER
from citrasense.web.helpers import get_sensor_context

if TYPE_CHECKING:
    from citrasense.web.app import CitraSenseWebApp


def _sensor_calibration_count(ctx: CitraSenseWebApp, sensor_id: str, field: str, default: int) -> int:
    """Read ``field`` from the per-sensor config, falling back to ``default``."""
    if not ctx.daemon or not ctx.daemon.settings:
        return default
    sc = ctx.daemon.settings.get_sensor_config(sensor_id)
    if sc is None:
        return default
    return int(getattr(sc, field, default) or default)


def build_calibration_router(ctx: CitraSenseWebApp) -> APIRouter:
    """Calibration status, capture, suite, and master management."""
    router = APIRouter(prefix="/api/sensors/{sensor_id}", tags=["calibration"])

    @router.get("/calibration/status")
    async def get_calibration_status(sensor_id: str):
        """Return calibration library status for the connected camera."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        sensor, runtime = get_sensor_context(ctx, sensor_id)
        adapter = sensor.adapter
        lib = getattr(runtime, "calibration_library", None)
        if not lib or not adapter.supports_direct_camera_control():
            return {"available": False}

        camera = adapter.camera
        if not camera:
            return {"available": False}

        profile = camera.get_calibration_profile()
        if not profile.calibration_applicable:
            return {"available": False}

        library_status = lib.get_library_status(profile.camera_id)
        cal_mgr = runtime.calibration_manager

        filters: list[dict[str, Any]] = []
        if adapter.supports_filter_management():
            filters = [
                {"name": f["name"], "position": int(pos)}
                for pos, f in adapter.get_filter_config().items()
                if f.get("enabled", True) and f.get("name")
            ]

        return {
            "available": True,
            "camera_id": profile.camera_id,
            "model": profile.model,
            "has_mechanical_shutter": profile.has_mechanical_shutter,
            "has_cooling": profile.has_cooling,
            "current_gain": profile.current_gain,
            "current_binning": profile.current_binning,
            "current_temperature": profile.current_temperature,
            "target_temperature": profile.target_temperature,
            "read_mode": profile.read_mode or "default",
            "gain_range": list(profile.gain_range) if profile.gain_range else None,
            "supported_binning": profile.supported_binning,
            "filters": filters,
            "library": library_status,
            "masters_dir": str(lib.masters_dir),
            "capture_running": cal_mgr.is_running() if cal_mgr else False,
            "capture_requested": cal_mgr.is_requested() if cal_mgr else False,
            "capture_progress": cal_mgr.get_progress() if cal_mgr else {},
            "frame_count_setting": _sensor_calibration_count(ctx, sensor_id, "calibration_frame_count", 30),
            "flat_frame_count_setting": _sensor_calibration_count(ctx, sensor_id, "flat_frame_count", 15),
        }

    @router.post("/calibration/capture")
    async def trigger_calibration_capture(sensor_id: str, request: dict[str, Any]):
        """Queue a calibration capture job."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        sensor, _runtime = get_sensor_context(ctx, sensor_id)
        try:
            if request.get("frame_type") == "flat":
                fp = request.get("filter_position")
                if fp is None:
                    return JSONResponse({"error": "filter_position is required for flat frames"}, status_code=400)
                fp_int = int(fp)
                fm = sensor.adapter.filter_map if sensor.adapter else {}
                if fp_int not in fm:
                    return JSONResponse({"error": f"Unknown filter position: {fp_int}"}, status_code=400)
                request["filter_position"] = fp_int
                request["filter_name"] = fm[fp_int].get("name", f"Filter {fp_int}")

            ok, err = ctx.daemon.trigger_calibration(request, sensor_id=sensor_id)
            if not ok:
                return JSONResponse({"error": err}, status_code=400)
            return {"success": True, "message": "Calibration queued"}
        except Exception as e:
            CITRASENSE_LOGGER.error("Error triggering calibration: %s", e, exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.post("/calibration/cancel")
    async def cancel_calibration(sensor_id: str):
        """Cancel pending or active calibration capture."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        get_sensor_context(ctx, sensor_id)
        try:
            was_cancelled = ctx.daemon.cancel_calibration(sensor_id=sensor_id)
            return {"success": was_cancelled}
        except Exception as e:
            CITRASENSE_LOGGER.error("Error cancelling calibration: %s", e, exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.post("/calibration/capture-suite")
    async def trigger_calibration_suite(sensor_id: str, request: dict[str, Any]):
        """Queue a calibration suite (bias_and_dark or all_flats)."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        sensor, _runtime = get_sensor_context(ctx, sensor_id)
        adapter = sensor.adapter
        try:
            from citrasense.calibration import FilterSlot
            from citrasense.calibration.calibration_suites import all_flats_suite, bias_and_dark_suite

            suite_name = request.get("suite", "")
            if not adapter.supports_direct_camera_control():
                return JSONResponse({"error": "No direct camera control"}, status_code=400)

            camera = adapter.camera
            if not camera:
                return JSONResponse({"error": "Camera not connected"}, status_code=400)

            profile = camera.get_calibration_profile()
            if not profile.calibration_applicable:
                return JSONResponse({"error": "Camera does not support calibration"}, status_code=400)

            frame_count = _sensor_calibration_count(ctx, sensor_id, "calibration_frame_count", 30)
            flat_count = _sensor_calibration_count(ctx, sensor_id, "flat_frame_count", 15)

            if suite_name == "bias_and_dark":
                jobs = bias_and_dark_suite(profile, frame_count)
            elif suite_name == "all_flats":
                filters: list[FilterSlot] = []
                if adapter.supports_filter_management():
                    filters = [
                        FilterSlot(position=int(pos), name=f["name"])
                        for pos, f in adapter.get_filter_config().items()
                        if f.get("enabled", True) and f.get("name")
                    ]
                if not filters:
                    return JSONResponse({"error": "No filters configured"}, status_code=400)
                jobs = all_flats_suite(profile, filters, flat_count)
            else:
                return JSONResponse({"error": f"Unknown suite: {suite_name}"}, status_code=400)

            if not jobs:
                return JSONResponse({"error": "Suite generated no jobs"}, status_code=400)

            ok, err = ctx.daemon.trigger_calibration_suite(jobs, sensor_id=sensor_id)
            if not ok:
                return JSONResponse({"error": err}, status_code=400)
            return {"success": True, "message": f"Suite queued: {len(jobs)} jobs", "job_count": len(jobs)}
        except Exception as e:
            CITRASENSE_LOGGER.error("Error triggering calibration suite: %s", e, exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.delete("/calibration/master")
    async def delete_calibration_master(sensor_id: str, request: dict[str, Any]):
        """Delete a specific master calibration frame."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        _sensor, runtime = get_sensor_context(ctx, sensor_id)
        lib = getattr(runtime, "calibration_library", None)
        if not lib:
            return JSONResponse({"error": "Calibration not available"}, status_code=400)
        try:
            deleted = lib.delete_master(
                frame_type=request.get("frame_type", ""),
                camera_id=request.get("camera_id", ""),
                gain=int(request.get("gain", 0)),
                binning=int(request.get("binning", 1)),
                exposure_time=float(request.get("exposure_time", 0)),
                temperature=request.get("temperature"),
                filter_name=request.get("filter_name", ""),
                read_mode=request.get("read_mode", ""),
            )
            return {"success": deleted}
        except Exception as e:
            CITRASENSE_LOGGER.error("Error deleting calibration master: %s", e, exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.get("/calibration/master/download")
    async def download_calibration_master(sensor_id: str, filename: str):
        """Download a master calibration FITS file by filename."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        _sensor, runtime = get_sensor_context(ctx, sensor_id)
        lib = getattr(runtime, "calibration_library", None)
        if not lib:
            return JSONResponse({"error": "Calibration not available"}, status_code=400)
        safe_name = Path(filename).name
        if not safe_name or safe_name != filename:
            return JSONResponse({"error": "Invalid filename"}, status_code=400)
        path = lib.masters_dir / safe_name
        if not path.exists():
            return JSONResponse({"error": "File not found"}, status_code=404)
        if path.is_symlink() or not path.resolve().is_relative_to(lib.masters_dir.resolve()):
            return JSONResponse({"error": "Invalid filename"}, status_code=400)
        return FileResponse(path, filename=safe_name, media_type="application/fits")

    return router
