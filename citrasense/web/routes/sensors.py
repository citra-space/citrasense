"""Sensor enumeration, detail, connect/disconnect, and config CRUD endpoints."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from citrasense.logging import CITRASENSE_LOGGER
from citrasense.settings.citrasense_settings import SensorConfig

if TYPE_CHECKING:
    from citrasense.web.app import CitraSenseWebApp

_SENSOR_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,62}$")


def build_sensors_router(ctx: CitraSenseWebApp) -> APIRouter:
    """Endpoints under ``/api/sensors`` and ``/api/config/sensors``."""
    router = APIRouter(tags=["sensors"])

    # ── Live sensor queries (runtime) ───────────────────────────────

    @router.get("/api/sensors")
    async def list_sensors():
        """Return all registered sensors with basic metadata."""
        if not ctx.daemon or not ctx.daemon.sensor_manager:
            return JSONResponse({"error": "Sensor manager not available"}, status_code=503)
        sensors = []
        for s in ctx.daemon.sensor_manager:
            sensors.append(
                {
                    "id": s.sensor_id,
                    "type": s.sensor_type,
                    "connected": s.is_connected(),
                    "name": getattr(s, "name", s.sensor_id),
                }
            )
        return {"sensors": sensors}

    @router.get("/api/sensors/{sensor_id}")
    async def sensor_detail(sensor_id: str):
        """Return detailed status for a single sensor."""
        if not ctx.daemon or not ctx.daemon.sensor_manager:
            return JSONResponse({"error": "Sensor manager not available"}, status_code=503)
        sensor = ctx.daemon.sensor_manager.get_sensor(sensor_id)
        if sensor is None:
            return JSONResponse({"error": f"Unknown sensor: {sensor_id}"}, status_code=404)
        detail: dict = {
            "id": sensor.sensor_id,
            "type": sensor.sensor_type,
            "connected": sensor.is_connected(),
            "name": getattr(sensor, "name", sensor.sensor_id),
        }
        if hasattr(sensor, "adapter") and sensor.adapter:
            detail["adapter_type"] = type(sensor.adapter).__name__
        return detail

    @router.post("/api/sensors/{sensor_id}/connect")
    async def connect_sensor(sensor_id: str):
        """Connect a sensor's hardware adapter."""
        if not ctx.daemon or not ctx.daemon.sensor_manager:
            return JSONResponse({"error": "Sensor manager not available"}, status_code=503)
        sensor = ctx.daemon.sensor_manager.get_sensor(sensor_id)
        if sensor is None:
            return JSONResponse({"error": f"Unknown sensor: {sensor_id}"}, status_code=404)
        try:
            ok = sensor.connect()
            if ok:
                return {"success": True, "message": f"Sensor {sensor_id} connected"}
            return JSONResponse({"error": "Connection failed"}, status_code=500)
        except Exception as e:
            CITRASENSE_LOGGER.error("Sensor %s connect error: %s", sensor_id, e, exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.post("/api/sensors/{sensor_id}/disconnect")
    async def disconnect_sensor(sensor_id: str):
        """Disconnect a sensor's hardware adapter."""
        if not ctx.daemon or not ctx.daemon.sensor_manager:
            return JSONResponse({"error": "Sensor manager not available"}, status_code=503)
        sensor = ctx.daemon.sensor_manager.get_sensor(sensor_id)
        if sensor is None:
            return JSONResponse({"error": f"Unknown sensor: {sensor_id}"}, status_code=404)
        try:
            sensor.disconnect()
            return {"success": True, "message": f"Sensor {sensor_id} disconnected"}
        except Exception as e:
            CITRASENSE_LOGGER.error("Sensor %s disconnect error: %s", sensor_id, e, exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    # ── Config-level sensor CRUD (persists to settings file) ──────

    @router.get("/api/config/sensors")
    async def list_sensor_configs():
        """Return all sensor config entries from settings."""
        if not ctx.daemon or not ctx.daemon.settings:
            return JSONResponse({"error": "Settings not available"}, status_code=503)
        return {
            "sensors": [s.model_dump() for s in ctx.daemon.settings.sensors],
        }

    @router.post("/api/config/sensors")
    async def add_sensor_config(body: dict[str, Any]):
        """Add a new sensor config entry and persist.

        Required body fields: ``id``, ``type``, ``adapter``.
        Optional: ``adapter_settings``, ``citra_sensor_id``.
        Returns the created config and signals the UI to reload.
        """
        if not ctx.daemon or not ctx.daemon.settings:
            return JSONResponse({"error": "Settings not available"}, status_code=503)

        sensor_id = (body.get("id") or "").strip()
        sensor_type = (body.get("type") or "telescope").strip()
        adapter = (body.get("adapter") or "").strip()

        if not sensor_id:
            return JSONResponse({"error": "Sensor id is required"}, status_code=400)
        if not _SENSOR_ID_RE.match(sensor_id):
            return JSONResponse(
                {"error": "Invalid sensor id — use alphanumeric, dots, hyphens, underscores (max 63 chars)"},
                status_code=400,
            )
        if ctx.daemon.settings.get_sensor_config(sensor_id):
            return JSONResponse({"error": f"Sensor '{sensor_id}' already exists"}, status_code=409)

        citra_sensor_id = (body.get("citra_sensor_id") or "").strip()
        if citra_sensor_id:
            clash = next(
                (s.id for s in ctx.daemon.settings.sensors if s.citra_sensor_id == citra_sensor_id),
                None,
            )
            if clash:
                return JSONResponse(
                    {
                        "error": (
                            f"citra_sensor_id {citra_sensor_id!r} is already used by sensor "
                            f"{clash!r}. Each local sensor must map to a distinct Citra telescope id."
                        )
                    },
                    status_code=409,
                )

        new_cfg = SensorConfig(
            id=sensor_id,
            type=sensor_type,
            adapter=adapter,
            adapter_settings=body.get("adapter_settings", {}),
            citra_sensor_id=citra_sensor_id,
        )
        ctx.daemon.settings.sensors.append(new_cfg)
        ctx.daemon.settings.save()

        CITRASENSE_LOGGER.info("Added sensor config: %s (type=%s, adapter=%s)", sensor_id, sensor_type, adapter)
        return {
            "success": True,
            "sensor": new_cfg.model_dump(),
            "message": f"Sensor '{sensor_id}' added — save & reload to activate.",
        }

    @router.delete("/api/config/sensors/{sensor_id}")
    async def remove_sensor_config(sensor_id: str):
        """Remove a sensor config entry and persist.

        At least one sensor must remain after removal.
        """
        if not ctx.daemon or not ctx.daemon.settings:
            return JSONResponse({"error": "Settings not available"}, status_code=503)

        settings = ctx.daemon.settings
        idx = next((i for i, s in enumerate(settings.sensors) if s.id == sensor_id), None)
        if idx is None:
            return JSONResponse({"error": f"Sensor '{sensor_id}' not found"}, status_code=404)
        if len(settings.sensors) <= 1:
            return JSONResponse({"error": "Cannot remove the last sensor"}, status_code=400)

        settings.sensors.pop(idx)
        settings.save()

        CITRASENSE_LOGGER.info("Removed sensor config: %s", sensor_id)
        remaining_ids = [s.id for s in settings.sensors]
        return {
            "success": True,
            "message": f"Sensor '{sensor_id}' removed — save & reload to finalize.",
            "remaining_sensors": remaining_ids,
        }

    return router
