"""Calibration library, capture, and master frame management endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from citrasense.logging import CITRASENSE_LOGGER

if TYPE_CHECKING:
    from citrasense.web.app import CitraSenseWebApp


def build_calibration_router(ctx: CitraSenseWebApp) -> APIRouter:
    """Calibration status, capture, suite, and master management."""
    router = APIRouter(prefix="/api", tags=["calibration"])

    @router.get("/calibration/status")
    async def get_calibration_status():
        """Return calibration library status for the connected camera."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        lib = getattr(ctx.daemon, "calibration_library", None)
        hw = ctx.daemon.hardware_adapter
        if not lib or not hw or not hw.supports_direct_camera_control():
            return {"available": False}

        camera = hw.camera
        if not camera:
            return {"available": False}

        profile = camera.get_calibration_profile()
        if not profile.calibration_applicable:
            return {"available": False}

        library_status = lib.get_library_status(profile.camera_id)
        tm = ctx.daemon.task_manager
        cal_mgr = tm.calibration_manager if tm else None

        filters: list[dict[str, Any]] = []
        if hw.supports_filter_management():
            filters = [
                {"name": f["name"], "position": int(pos)}
                for pos, f in hw.get_filter_config().items()
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
            "frame_count_setting": ctx.daemon.settings.calibration_frame_count if ctx.daemon.settings else 30,
            "flat_frame_count_setting": ctx.daemon.settings.flat_frame_count if ctx.daemon.settings else 15,
        }

    @router.post("/calibration/capture")
    async def trigger_calibration_capture(request: dict[str, Any]):
        """Queue a calibration capture job."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        try:
            if request.get("frame_type") == "flat":
                fp = request.get("filter_position")
                if fp is None:
                    return JSONResponse({"error": "filter_position is required for flat frames"}, status_code=400)
                fp_int = int(fp)
                fm = ctx.daemon.hardware_adapter.filter_map if ctx.daemon.hardware_adapter else {}
                if fp_int not in fm:
                    return JSONResponse({"error": f"Unknown filter position: {fp_int}"}, status_code=400)
                request["filter_position"] = fp_int
                request["filter_name"] = fm[fp_int].get("name", f"Filter {fp_int}")

            ok, err = ctx.daemon.trigger_calibration(request)
            if not ok:
                return JSONResponse({"error": err}, status_code=400)
            return {"success": True, "message": "Calibration queued"}
        except Exception as e:
            CITRASENSE_LOGGER.error("Error triggering calibration: %s", e, exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.post("/calibration/cancel")
    async def cancel_calibration():
        """Cancel pending or active calibration capture."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        try:
            was_cancelled = ctx.daemon.cancel_calibration()
            return {"success": was_cancelled}
        except Exception as e:
            CITRASENSE_LOGGER.error("Error cancelling calibration: %s", e, exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.post("/calibration/capture-suite")
    async def trigger_calibration_suite(request: dict[str, Any]):
        """Queue a calibration suite (bias_and_dark or all_flats)."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        try:
            from citrasense.calibration import FilterSlot
            from citrasense.calibration.calibration_suites import all_flats_suite, bias_and_dark_suite

            suite_name = request.get("suite", "")
            hw = ctx.daemon.hardware_adapter
            if not hw or not hw.supports_direct_camera_control():
                return JSONResponse({"error": "No direct camera control"}, status_code=400)

            camera = hw.camera
            if not camera:
                return JSONResponse({"error": "Camera not connected"}, status_code=400)

            profile = camera.get_calibration_profile()
            if not profile.calibration_applicable:
                return JSONResponse({"error": "Camera does not support calibration"}, status_code=400)

            frame_count = ctx.daemon.settings.calibration_frame_count if ctx.daemon.settings else 30
            flat_count = ctx.daemon.settings.flat_frame_count if ctx.daemon.settings else 15

            if suite_name == "bias_and_dark":
                jobs = bias_and_dark_suite(profile, frame_count)
            elif suite_name == "all_flats":
                filters: list[FilterSlot] = []
                if hw.supports_filter_management():
                    filters = [
                        FilterSlot(position=int(pos), name=f["name"])
                        for pos, f in hw.get_filter_config().items()
                        if f.get("enabled", True) and f.get("name")
                    ]
                if not filters:
                    return JSONResponse({"error": "No filters configured"}, status_code=400)
                jobs = all_flats_suite(profile, filters, flat_count)
            else:
                return JSONResponse({"error": f"Unknown suite: {suite_name}"}, status_code=400)

            if not jobs:
                return JSONResponse({"error": "Suite generated no jobs"}, status_code=400)

            ok, err = ctx.daemon.trigger_calibration_suite(jobs)
            if not ok:
                return JSONResponse({"error": err}, status_code=400)
            return {"success": True, "message": f"Suite queued: {len(jobs)} jobs", "job_count": len(jobs)}
        except Exception as e:
            CITRASENSE_LOGGER.error("Error triggering calibration suite: %s", e, exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.delete("/calibration/master")
    async def delete_calibration_master(request: dict[str, Any]):
        """Delete a specific master calibration frame."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        lib = getattr(ctx.daemon, "calibration_library", None)
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
    async def download_calibration_master(filename: str):
        """Download a master calibration FITS file by filename."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        lib = getattr(ctx.daemon, "calibration_library", None)
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
