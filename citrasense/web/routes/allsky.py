"""Allsky camera control and preview endpoints.

Per-sensor routes under ``/api/sensors/{sensor_id}/allsky/*`` that surface
:class:`~citrasense.sensors.allsky.allsky_camera_sensor.AllskyCameraSensor`
state to the web UI:

- ``GET  /allsky/status``      → cached status snapshot (also embedded
  per-tick into ``StatusCollector`` as ``sensor.allsky``).
- ``POST /allsky/capture``     → trigger one off-cycle capture.
- ``GET  /allsky/latest.jpg``  → latest JPEG frame, served as a normal
  image response so the detail page can use it as a stable
  ``<img src=...>`` URL with cache-busting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from citrasense.logging import CITRASENSE_LOGGER
from citrasense.web.helpers import get_sensor_context

if TYPE_CHECKING:
    from citrasense.sensors.allsky.allsky_camera_sensor import AllskyCameraSensor
    from citrasense.web.app import CitraSenseWebApp


def build_allsky_router(ctx: CitraSenseWebApp) -> APIRouter:
    """Routes for the allsky sensor type.

    Returns 404 when the sensor id is unknown and 409 when the sensor
    exists but is not an ``allsky`` — matches the radar/autofocus router
    convention.
    """
    router = APIRouter(prefix="/api/sensors/{sensor_id}", tags=["allsky"])

    def _get_allsky_sensor(sensor_id: str) -> AllskyCameraSensor | JSONResponse:
        sensor, _runtime = get_sensor_context(ctx, sensor_id)
        if sensor.sensor_type != "allsky":
            return JSONResponse(
                {"error": f"Sensor {sensor_id!r} is not an allsky sensor"},
                status_code=409,
            )
        return sensor

    @router.get("/allsky/status")
    async def allsky_status(sensor_id: str):
        """Return the sensor's cached live-status snapshot."""
        sensor = _get_allsky_sensor(sensor_id)
        if isinstance(sensor, JSONResponse):
            return sensor
        return sensor.get_live_status()

    @router.post("/allsky/capture")
    async def allsky_capture(sensor_id: str):
        """Trigger one off-cycle capture.

        Returns ``{ok, captured_at, error}`` so the UI can surface either
        a success toast or the underlying camera failure verbatim.  A 409
        is returned when the sensor isn't connected — operators get a
        clearer signal than a generic 500.
        """
        sensor = _get_allsky_sensor(sensor_id)
        if isinstance(sensor, JSONResponse):
            return sensor
        try:
            return sensor.capture_now()
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        except Exception as exc:
            CITRASENSE_LOGGER.warning("allsky_capture(%s) failed: %s", sensor_id, exc, exc_info=True)
            return JSONResponse({"error": str(exc)}, status_code=500)

    @router.post("/allsky/streaming")
    async def set_allsky_streaming(sensor_id: str, body: dict[str, Any]):
        """Toggle the operator pause flag for the allsky capture loop.

        Body: ``{"enabled": bool}``.  Persists to
        :attr:`SensorConfig.streaming_enabled` and immediately
        starts/stops the producer via
        :meth:`SensorRuntime.set_streaming_enabled`.  No-op if the
        camera is currently disconnected — the flag still persists and
        will be honoured on the next reconnect.

        Returns ``{"success": True, "streaming_enabled": bool}`` on
        success.  404 for unknown sensors and 409 for non-allsky
        sensors are inherited from :func:`_get_allsky_sensor`.
        """
        sensor = _get_allsky_sensor(sensor_id)
        if isinstance(sensor, JSONResponse):
            return sensor
        enabled = body.get("enabled")
        if not isinstance(enabled, bool):
            return JSONResponse({"error": "enabled must be a boolean"}, status_code=400)
        _, runtime = get_sensor_context(ctx, sensor_id)
        runtime.set_streaming_enabled(enabled)
        return {"success": True, "streaming_enabled": enabled}

    @router.get("/allsky/latest.jpg")
    async def allsky_latest_jpg(sensor_id: str):
        """Return the latest captured frame as a raw JPEG.

        404 until the first capture lands.  ``Cache-Control: no-store``
        keeps the browser from serving a stale frame on the auto-refresh
        path; the page typically uses a cache-buster query string anyway.
        """
        sensor = _get_allsky_sensor(sensor_id)
        if isinstance(sensor, JSONResponse):
            return sensor
        jpeg = sensor.latest_jpeg_bytes
        if not jpeg:
            return JSONResponse(
                {"error": "No frame captured yet"},
                status_code=404,
            )
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    return router
