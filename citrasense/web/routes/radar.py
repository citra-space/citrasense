"""Radar control and status endpoints.

Per-sensor routes under ``/api/sensors/{sensor_id}/radar/*`` that proxy
through :class:`~citrasense.sensors.radar.passive_radar_sensor.PassiveRadarSensor`
into the NATS detection source.  Request-reply commands (``start``,
``stop``, ``ping``, ``config.set``) are issued synchronously with a
short timeout; the caller sees the raw ``{"ok": ..., ...}`` reply from
``pr_sensor`` so the UI can surface ``error`` strings verbatim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from citrasense.logging import CITRASENSE_LOGGER
from citrasense.web.helpers import get_sensor_context

if TYPE_CHECKING:
    from citrasense.sensors.radar.passive_radar_sensor import PassiveRadarSensor
    from citrasense.web.app import CitraSenseWebApp


def build_radar_router(ctx: CitraSenseWebApp) -> APIRouter:
    """Routes for the passive-radar sensor type.

    Returns 404 if the sensor id is unknown and 409 if the sensor
    exists but is not a ``passive_radar`` — matches the existing
    autofocus/alignment router convention.
    """
    router = APIRouter(prefix="/api/sensors/{sensor_id}", tags=["radar"])

    def _get_radar_sensor(sensor_id: str) -> PassiveRadarSensor | JSONResponse:
        sensor, _runtime = get_sensor_context(ctx, sensor_id)
        if sensor.sensor_type != "passive_radar":
            return JSONResponse(
                {"error": f"Sensor {sensor_id!r} is not a passive_radar sensor"},
                status_code=409,
            )
        return sensor

    @router.get("/radar/status")
    async def radar_status(sensor_id: str):
        """Return the sensor's cached status / health / stations snapshot."""
        sensor = _get_radar_sensor(sensor_id)
        if isinstance(sensor, JSONResponse):
            return sensor
        return sensor.get_live_status()

    @router.post("/radar/start")
    async def radar_start(sensor_id: str, body: dict | None = None):
        """Request pr_sensor to transition to state=running."""
        sensor = _get_radar_sensor(sensor_id)
        if isinstance(sensor, JSONResponse):
            return sensor
        mock = bool((body or {}).get("mock", False))
        try:
            reply = sensor.send_start_command(mock=mock)
            return reply
        except Exception as exc:
            CITRASENSE_LOGGER.warning("radar_start(%s) failed: %s", sensor_id, exc)
            return JSONResponse({"error": str(exc)}, status_code=502)

    @router.post("/radar/stop")
    async def radar_stop(sensor_id: str):
        sensor = _get_radar_sensor(sensor_id)
        if isinstance(sensor, JSONResponse):
            return sensor
        try:
            reply = sensor.send_stop_command()
            return reply
        except Exception as exc:
            CITRASENSE_LOGGER.warning("radar_stop(%s) failed: %s", sensor_id, exc)
            return JSONResponse({"error": str(exc)}, status_code=502)

    @router.post("/radar/config")
    async def radar_push_config(sensor_id: str, body: dict | None = None):
        """Push the sensor's configured RadarConfig to pr_sensor."""
        sensor = _get_radar_sensor(sensor_id)
        if isinstance(sensor, JSONResponse):
            return sensor
        persist = bool((body or {}).get("persist", False))
        try:
            reply = sensor.push_radar_config(persist=persist)
            return reply
        except Exception as exc:
            CITRASENSE_LOGGER.warning("radar_push_config(%s) failed: %s", sensor_id, exc)
            return JSONResponse({"error": str(exc)}, status_code=502)

    @router.post("/radar/ping")
    async def radar_ping(sensor_id: str):
        sensor = _get_radar_sensor(sensor_id)
        if isinstance(sensor, JSONResponse):
            return sensor
        try:
            reply = sensor.send_ping()
            return reply
        except Exception as exc:
            CITRASENSE_LOGGER.warning("radar_ping(%s) failed: %s", sensor_id, exc)
            return JSONResponse({"error": str(exc)}, status_code=502)

    @router.get("/radar/detections")
    async def radar_detections(sensor_id: str, since: float | None = None):
        """Return the recent slim-dict detection snapshot.

        ``since`` cursor semantics match
        :meth:`DetectionRingBuffer.snapshot_since`:

        - unset / ``None`` → whole buffer (≤ ring-buffer cap)
        - non-negative → absolute Unix epoch (return newer than that)
        - negative → "last |since| seconds" (convenience for hydration)

        Used by the monitoring and detail pages on page entry to
        back-fill the range-Doppler plot before live WebSocket
        broadcasts take over.
        """
        sensor = _get_radar_sensor(sensor_id)
        if isinstance(sensor, JSONResponse):
            return sensor
        return {"detections": sensor.get_recent_detections(since)}

    return router
