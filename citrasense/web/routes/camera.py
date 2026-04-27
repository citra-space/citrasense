"""Camera capture and preview endpoints."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from citrasense.logging import CITRASENSE_LOGGER
from citrasense.web.helpers import get_sensor_context

if TYPE_CHECKING:
    from citrasense.web.app import CitraSenseWebApp


def build_camera_router(ctx: CitraSenseWebApp) -> APIRouter:
    """Camera test capture and preview endpoints."""
    router = APIRouter(prefix="/api/sensors/{sensor_id}", tags=["camera"])

    @router.post("/camera/capture")
    async def camera_capture(sensor_id: str, request: dict[str, Any]):
        """Trigger a test camera capture."""
        sensor, _runtime = get_sensor_context(ctx, sensor_id)
        if busy := ctx.require_sensor_idle(_runtime):
            return busy
        adapter = sensor.adapter

        if not adapter.supports_direct_camera_control():
            return JSONResponse({"error": "Hardware adapter does not support direct camera control"}, status_code=400)

        try:
            duration = request.get("duration", 0.1)

            if duration <= 0:
                return JSONResponse({"error": "Exposure duration must be positive"}, status_code=400)
            if duration > 300:
                return JSONResponse({"error": "Exposure duration must be 300 seconds or less"}, status_code=400)

            CITRASENSE_LOGGER.info(f"Test capture requested: {duration}s exposure")

            filepath = adapter.expose_camera(exposure_time=duration, gain=None, offset=None, count=1)

            file_path = Path(filepath)
            if not file_path.exists():
                return JSONResponse({"error": "Capture completed but file not found"}, status_code=500)

            filename = file_path.name
            file_format = file_path.suffix.upper().lstrip(".")

            CITRASENSE_LOGGER.info(f"Test capture complete: {filename}")

            return {"success": True, "filename": filename, "filepath": str(file_path), "format": file_format}

        except Exception as e:
            CITRASENSE_LOGGER.error(f"Error during test capture: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.post("/camera/preview")
    async def camera_preview(sensor_id: str, request: dict[str, Any]):
        """Take an ephemeral preview exposure and return a JPEG data URL."""
        sensor, _runtime = get_sensor_context(ctx, sensor_id)
        adapter = sensor.adapter

        if not adapter.supports_direct_camera_control():
            return JSONResponse({"error": "Hardware adapter does not support direct camera control"}, status_code=400)

        if _runtime and not _runtime.paused:
            return JSONResponse(
                {"error": "Camera unavailable — task processing is active for this sensor"}, status_code=503
            )

        try:
            duration = request.get("duration", 1.0)
            if not isinstance(duration, (int, float)):
                return JSONResponse({"error": "duration must be a number"}, status_code=400)
            duration = float(duration)
            if duration <= 0:
                return JSONResponse({"error": "Exposure duration must be positive"}, status_code=400)
            if duration > 30:
                return JSONResponse({"error": "Preview exposure must be 30 seconds or less"}, status_code=400)

            flip_h = bool(request.get("flip_horizontal", False))
            image_data = await asyncio.to_thread(adapter.capture_preview, duration, flip_h)
            return {"image_data": image_data}

        except NotImplementedError:
            return JSONResponse({"error": "Preview not supported by this adapter"}, status_code=400)
        except RuntimeError as e:
            if "already in progress" in str(e):
                return JSONResponse({"error": "busy"}, status_code=409)
            CITRASENSE_LOGGER.error(f"Error during preview capture: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)
        except Exception as e:
            CITRASENSE_LOGGER.error(f"Error during preview capture: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    return router
