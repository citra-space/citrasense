"""Calibration library, capture, and master frame management endpoints."""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from astropy.io import fits  # type: ignore[attr-defined]
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

from citrasense.calibration.calibration_library import resolve_camera_id
from citrasense.logging import CITRASENSE_LOGGER
from citrasense.web.helpers import get_sensor_context

if TYPE_CHECKING:
    from citrasense.web.app import CitraSenseWebApp

# Uploaded master FITS files are rejected above this size.  128 MB covers
# a 16 MP float32 master plus headroom for common read/bin configurations.
_UPLOAD_MAX_BYTES = 128 * 1024 * 1024

_VALID_FRAME_TYPES = {"bias", "dark", "flat"}


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
        """Return calibration library status for the connected camera.

        Two capture modes are supported:

        * ``direct`` — adapter exposes an :class:`AbstractCamera` that
          CitraSense drives for shutter-closed captures.  All library
          metadata comes from ``camera.get_calibration_profile()``.
        * ``upload_only`` — orchestrator adapters (NINA) that own their
          own capture path.  Masters are ingested via the upload
          endpoint; the adapter only needs to provide a ``camera_id``
          matching what it writes into science FITS headers.
        """
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        sensor, runtime = get_sensor_context(ctx, sensor_id)
        adapter = sensor.adapter
        lib = getattr(runtime, "calibration_library", None)
        if not lib:
            return {"available": False}

        filters: list[dict[str, Any]] = []
        if adapter.supports_filter_management():
            filters = [
                {"name": f["name"], "position": int(pos)}
                for pos, f in adapter.get_filter_config().items()
                if f.get("enabled", True) and f.get("name")
            ]

        # ``flat_automation_available`` is true when something other
        # than the direct camera is wired to capture flats — today that
        # means NINA's trained-flat wizard.  The frontend uses it to
        # swap the per-frame exposure/count controls for a single "Run
        # Flat Wizard" button, because NINA's trained exposure table is
        # the source of truth for those parameters.
        cal_mgr = runtime.calibration_manager
        flat_automation_available = False
        if cal_mgr is not None:
            backend = cal_mgr.flat_backend
            if backend is not None:
                from citrasense.calibration.flat_capture_backend import DirectCameraFlatBackend

                flat_automation_available = not isinstance(backend, DirectCameraFlatBackend)

        sensor_cfg = ctx.daemon.settings.get_sensor_config(sensor_id) if ctx.daemon.settings else None
        auto_capture_flats_enabled = bool(getattr(sensor_cfg, "auto_capture_flats_enabled", False))
        last_flats_capture_iso = getattr(sensor_cfg, "last_flats_capture_iso", None)

        # Prefer the direct-camera profile when available.  Adapters
        # that return ``None`` here fall through to the upload-only path.
        camera = adapter.camera
        if camera is not None:
            profile = camera.get_calibration_profile()
            if not profile.calibration_applicable:
                return {"available": False}

            return {
                "available": True,
                "capture_mode": "direct",
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
                "library": lib.get_library_status(profile.camera_id),
                "masters_dir": str(lib.masters_dir),
                "capture_running": cal_mgr.is_running() if cal_mgr else False,
                "capture_requested": cal_mgr.is_requested() if cal_mgr else False,
                "capture_progress": cal_mgr.get_progress() if cal_mgr else {},
                "frame_count_setting": _sensor_calibration_count(ctx, sensor_id, "calibration_frame_count", 30),
                "flat_frame_count_setting": _sensor_calibration_count(ctx, sensor_id, "flat_frame_count", 15),
                "flat_automation_available": flat_automation_available,
                "auto_capture_flats_enabled": auto_capture_flats_enabled,
                "last_flats_capture_iso": last_flats_capture_iso,
            }

        summary_fn = getattr(adapter, "get_calibration_profile_summary", None)
        raw_summary = summary_fn() if callable(summary_fn) else None
        if not isinstance(raw_summary, dict) or not raw_summary.get("camera_id"):
            return {"available": False}
        summary: dict[str, Any] = raw_summary

        return {
            "available": True,
            "capture_mode": "upload_only",
            "camera_id": summary["camera_id"],
            "model": summary.get("model", summary["camera_id"]),
            "has_mechanical_shutter": bool(summary.get("has_mechanical_shutter", False)),
            "has_cooling": bool(summary.get("has_cooling", False)),
            "current_gain": summary.get("current_gain"),
            "current_binning": summary.get("current_binning"),
            "current_temperature": summary.get("current_temperature"),
            "target_temperature": summary.get("target_temperature"),
            "read_mode": summary.get("read_mode") or "default",
            "gain_range": None,
            "supported_binning": None,
            "filters": filters,
            "library": lib.get_library_status(summary["camera_id"]),
            "masters_dir": str(lib.masters_dir),
            "capture_running": cal_mgr.is_running() if cal_mgr else False,
            "capture_requested": cal_mgr.is_requested() if cal_mgr else False,
            "capture_progress": cal_mgr.get_progress() if cal_mgr else {},
            "frame_count_setting": _sensor_calibration_count(ctx, sensor_id, "calibration_frame_count", 30),
            "flat_frame_count_setting": _sensor_calibration_count(ctx, sensor_id, "flat_frame_count", 15),
            "flat_automation_available": flat_automation_available,
            "auto_capture_flats_enabled": auto_capture_flats_enabled,
            "last_flats_capture_iso": last_flats_capture_iso,
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
        """Queue a calibration suite (``bias_and_dark`` or ``all_flats``).

        ``all_flats`` works for both direct hardware and any adapter
        with a wired :class:`FlatCaptureBackend` (e.g. NINA trained
        flats).  ``bias_and_dark`` still requires a direct camera
        because shutter-closed captures have no orchestrator path.
        """
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        sensor, runtime = get_sensor_context(ctx, sensor_id)
        adapter = sensor.adapter
        try:
            from citrasense.calibration import FilterSlot
            from citrasense.calibration.calibration_suites import all_flats_suite, bias_and_dark_suite
            from citrasense.hardware.devices.camera.abstract_camera import CalibrationProfile

            suite_name = request.get("suite", "")
            frame_count = _sensor_calibration_count(ctx, sensor_id, "calibration_frame_count", 30)
            flat_count = _sensor_calibration_count(ctx, sensor_id, "flat_frame_count", 15)

            if suite_name == "bias_and_dark":
                if not adapter.supports_direct_camera_control():
                    return JSONResponse({"error": "No direct camera control"}, status_code=400)
                camera = adapter.camera
                if not camera:
                    return JSONResponse({"error": "Camera not connected"}, status_code=400)
                profile = camera.get_calibration_profile()
                if not profile.calibration_applicable:
                    return JSONResponse({"error": "Camera does not support calibration"}, status_code=400)
                jobs = bias_and_dark_suite(profile, frame_count)
            elif suite_name == "all_flats":
                cal_mgr = runtime.calibration_manager
                if cal_mgr is None or not cal_mgr.supports_frame_type("flat"):
                    return JSONResponse({"error": "Flat capture not available on this sensor"}, status_code=400)

                camera = adapter.camera
                if camera is not None:
                    profile = camera.get_calibration_profile()
                    if not profile.calibration_applicable:
                        return JSONResponse({"error": "Camera does not support calibration"}, status_code=400)
                else:
                    # Synthesise a profile from the adapter summary so
                    # all_flats_suite has a consistent input shape.
                    summary_fn = getattr(adapter, "get_calibration_profile_summary", None)
                    raw = summary_fn() if callable(summary_fn) else None
                    if not isinstance(raw, dict) or not raw.get("camera_id"):
                        return JSONResponse({"error": "No calibration profile available"}, status_code=400)
                    profile = CalibrationProfile(
                        calibration_applicable=True,
                        camera_id=str(raw["camera_id"]),
                        model=str(raw.get("model") or raw["camera_id"]),
                        has_mechanical_shutter=bool(raw.get("has_mechanical_shutter", False)),
                        has_cooling=bool(raw.get("has_cooling", False)),
                        current_gain=raw.get("current_gain"),
                        current_binning=int(raw.get("current_binning") or 1),
                        current_temperature=raw.get("current_temperature"),
                        target_temperature=raw.get("target_temperature"),
                        read_mode=str(raw.get("read_mode") or ""),
                    )

                filters: list[FilterSlot] = []
                if adapter.supports_filter_management():
                    filters = [
                        FilterSlot(position=int(pos), name=f["name"])
                        for pos, f in adapter.get_filter_config().items()
                        if f.get("enabled", True) and f.get("name")
                    ]
                if not filters:
                    return JSONResponse({"error": "No filters configured"}, status_code=400)
                if camera is not None:
                    jobs = all_flats_suite(profile, filters, flat_count)
                else:
                    # NINA trained flats are fundamentally per-filter — there
                    # is no interleaved capture primitive on the REST API — so
                    # we generate N separate flat jobs that the manager runs
                    # sequentially through the backend.
                    jobs = [
                        {
                            "frame_type": "flat",
                            "count": flat_count,
                            "gain": profile.current_gain or 0,
                            "binning": profile.current_binning,
                            "filter_position": slot.position,
                            "filter_name": slot.name,
                        }
                        for slot in filters
                    ]
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

    @router.post("/calibration/upload")
    async def upload_calibration_master(
        sensor_id: str,
        request: Request,
        frame_type: str,
        normalize_flat: bool = True,
        override_filter: str = "",
    ):
        """Ingest a pre-built master FITS file into the calibration library.

        The FITS payload is read from the raw request body (no
        multipart) and metadata is extracted from the file's header.
        Query parameters select the frame type and override / disable
        the default flat normalization.

        Primary path for orchestrator adapters (NINA) whose capture
        pipeline is external to CitraSense.  The user builds masters in
        NINA's Flat Wizard / dark library and uploads them here; the
        :class:`CalibrationProcessor` then applies them to incoming
        light frames using the same header-keyed lookup as direct
        hardware.  Phase 2 (future PR) can add a "Run NINA flat wizard"
        action that drives NINA's ``flats/trained-flat`` endpoint and
        funnels the resulting frames through the same
        :meth:`CalibrationLibrary.save_master` call below — keeping one
        naming/storage contract across capture sources.
        """
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        _sensor, runtime = get_sensor_context(ctx, sensor_id)
        lib = getattr(runtime, "calibration_library", None)
        if not lib:
            return JSONResponse({"error": "Calibration not available"}, status_code=400)

        frame_type = frame_type.strip().lower()
        if frame_type not in _VALID_FRAME_TYPES:
            return JSONResponse(
                {"error": f"frame_type must be one of {sorted(_VALID_FRAME_TYPES)}"},
                status_code=400,
            )

        try:
            payload = await request.body()
        except Exception as e:
            return JSONResponse({"error": f"Could not read upload: {e}"}, status_code=400)

        if not payload:
            return JSONResponse({"error": "Uploaded file is empty"}, status_code=400)
        if len(payload) > _UPLOAD_MAX_BYTES:
            return JSONResponse(
                {"error": f"Uploaded file exceeds {_UPLOAD_MAX_BYTES // (1024 * 1024)} MB"},
                status_code=413,
            )

        warnings: list[str] = []
        try:
            with fits.open(io.BytesIO(payload)) as hdul:
                primary = hdul[0]
                assert isinstance(primary, fits.PrimaryHDU)
                header = primary.header
                data = primary.data
                if data is None:
                    return JSONResponse({"error": "FITS file has no image data"}, status_code=400)
                image = np.asarray(data, dtype=np.float32)
                hdr_items: dict[str, Any] = {
                    "camera_id": resolve_camera_id(header),
                    "gain": int(header.get("GAIN", 0) or 0),  # type: ignore[arg-type]
                    "binning": int(header.get("XBINNING", 1) or 1),  # type: ignore[arg-type]
                    "exposure_time": float(header.get("EXPTIME", 0) or 0),  # type: ignore[arg-type]
                    "temperature": header.get("CCD-TEMP"),
                    "filter_name": str(header.get("FILTER", "") or "").strip(),
                    # NINA/ASCOM frequently emits READOUTM rather than READMODE.
                    "read_mode": str(header.get("READMODE", "") or header.get("READOUTM", "") or "").strip(),
                    "instrument": str(header.get("INSTRUME", "") or "").strip(),
                    "bias_subtracted": header.get("BIASSUB"),
                }
        except OSError as e:
            return JSONResponse({"error": f"Not a valid FITS file: {e}"}, status_code=400)
        except Exception as e:
            CITRASENSE_LOGGER.error("Upload FITS parse error: %s", e, exc_info=True)
            return JSONResponse({"error": f"FITS parse error: {e}"}, status_code=400)

        if image.ndim != 2:
            return JSONResponse(
                {"error": f"Expected 2D image data, got shape {tuple(image.shape)}"},
                status_code=400,
            )

        camera_id = hdr_items["camera_id"] or "unknown"
        if camera_id == "unknown":
            warnings.append("FITS header missing CAMSER/INSTRUME — camera_id defaulted to 'unknown'")

        override = override_filter.strip()
        if override:
            hdr_items["filter_name"] = override

        temperature = hdr_items["temperature"]
        try:
            temperature = float(temperature) if temperature is not None else None
        except (TypeError, ValueError):
            temperature = None

        if frame_type == "dark":
            if hdr_items["exposure_time"] <= 0:
                return JSONResponse(
                    {"error": "Dark frame requires a positive EXPTIME header"},
                    status_code=400,
                )
            if temperature is None:
                warnings.append("CCD-TEMP missing — dark will match temperature-agnostically")
        elif frame_type == "flat":
            if not hdr_items["filter_name"]:
                return JSONResponse(
                    {"error": "Flat frame requires a FILTER header or override_filter form field"},
                    status_code=400,
                )

        if frame_type == "flat" and normalize_flat:
            # Maintain the library invariant from MasterBuilder.build_flat:
            # flats are divided by their median so pipeline division
            # preserves absolute photometry.
            median = float(np.median(image))
            if median > 0 and not (0.95 <= median <= 1.05):
                image = image / median
                warnings.append(f"Flat normalized to median=1.0 (input median was {median:.3f})")
            elif median <= 0:
                return JSONResponse(
                    {"error": f"Flat has non-positive median ({median:.3f}); refusing to normalize"},
                    status_code=400,
                )

        # Darks uploaded from NINA/third parties usually are not
        # bias-subtracted.  The processor treats ``BIASSUB`` absent /
        # False as "subtract bias at apply time if available", which is
        # the safe default for ingest.  Respect the header if the user
        # built a bias-subtracted master themselves.
        if frame_type == "dark":
            raw_biassub = hdr_items["bias_subtracted"]
            bias_subtracted: bool | None
            if isinstance(raw_biassub, bool):
                bias_subtracted = raw_biassub
            elif isinstance(raw_biassub, str):
                bias_subtracted = raw_biassub.strip().upper() in ("T", "TRUE", "1", "Y", "YES")
            else:
                bias_subtracted = False
        else:
            bias_subtracted = None

        # ``header`` is bound above inside the ``with fits.open(...)`` block.
        # astropy cards are cheap POD so reading them here after the file
        # handle has been closed is safe.  astropy's .get() typing is
        # overly broad (union of Header subclasses) so we force-coerce
        # through ``str`` to keep pyright happy.
        ncombine_raw = header.get("NCOMBINE")
        try:
            ncombine = int(str(ncombine_raw)) if ncombine_raw is not None else 0
        except (TypeError, ValueError):
            ncombine = 0

        try:
            saved_path = lib.save_master(
                frame_type=frame_type,
                camera_id=camera_id,
                data=image,
                gain=hdr_items["gain"],
                binning=hdr_items["binning"],
                exposure_time=hdr_items["exposure_time"] if frame_type == "dark" else 0.0,
                temperature=temperature if frame_type == "dark" else None,
                filter_name=hdr_items["filter_name"] if frame_type == "flat" else "",
                ncombine=ncombine,
                camera_model=hdr_items["instrument"] or camera_id,
                read_mode=hdr_items["read_mode"],
                bias_subtracted=bias_subtracted,
            )
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception as e:
            CITRASENSE_LOGGER.error("Error saving uploaded master: %s", e, exc_info=True)
            return JSONResponse({"error": f"Failed to save master: {e}"}, status_code=500)

        CITRASENSE_LOGGER.info(
            "Ingested uploaded %s master → %s (sensor=%s)",
            frame_type,
            saved_path.name,
            sensor_id,
        )
        return {
            "success": True,
            "saved_as": saved_path.name,
            "camera_id": camera_id,
            "frame_type": frame_type,
            "warnings": warnings,
        }

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
