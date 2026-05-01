"""Sensor enumeration, detail, connect/disconnect, and config CRUD endpoints."""

from __future__ import annotations

import inspect
import json
import re
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from citrasense.logging import CITRASENSE_LOGGER
from citrasense.sensors.sensor_registry import get_sensor_class
from citrasense.sensors.sensor_registry import list_sensors as _list_sensor_registry
from citrasense.settings.citrasense_settings import SensorConfig

#: Friendly labels for sensor-type dropdowns. Keys must match
#: :mod:`citrasense.sensors.sensor_registry`. Unknown sensor types fall
#: back to their registry key.
_SENSOR_TYPE_LABELS: dict[str, str] = {
    "telescope": "Telescope",
    "passive_radar": "Passive radar",
    "allsky": "Allsky camera",
}

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
        """Queue a sensor's hardware ``connect()`` on the async init worker.

        Returns ``202 Accepted`` immediately — the operator watches the
        toast / monitoring badge for the result.  This shape stops a
        hung adapter from blocking the HTTP request itself, which is
        the core of issue #339.
        """
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        ok, err = ctx.daemon.request_sensor_reconnect(sensor_id)
        if not ok:
            # ``Unknown sensor`` -> 404; in-flight reconnect -> 409;
            # any other diagnostic message stays at 400.
            if err and err.startswith("Unknown sensor"):
                return JSONResponse({"error": err}, status_code=404)
            if err and "already in flight" in err:
                return JSONResponse({"error": err}, status_code=409)
            return JSONResponse({"error": err or "Could not queue connect"}, status_code=400)
        return JSONResponse(
            {
                "success": True,
                "message": f"Sensor {sensor_id} connect queued",
                "init_state": "connecting",
            },
            status_code=202,
        )

    @router.post("/api/sensors/{sensor_id}/reconnect")
    async def reconnect_sensor(sensor_id: str):
        """Queue a per-sensor disconnect + connect cycle on the async init worker.

        Same shape as ``/connect`` — returns ``202`` and lets the toast
        / status badge carry the result.  ``request_sensor_reconnect``
        runs the disconnect inline (cheap) and submits the connect to
        the executor with the per-sensor watchdog timeout, so a hung
        adapter blows the deadline cleanly without blocking the
        request.
        """
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        ok, err = ctx.daemon.request_sensor_reconnect(sensor_id)
        if not ok:
            if err and err.startswith("Unknown sensor"):
                return JSONResponse({"error": err}, status_code=404)
            if err and "already in flight" in err:
                return JSONResponse({"error": err}, status_code=409)
            return JSONResponse({"error": err or "Could not queue reconnect"}, status_code=400)
        return JSONResponse(
            {
                "success": True,
                "message": f"Sensor {sensor_id} reconnect queued",
                "init_state": "connecting",
            },
            status_code=202,
        )

    @router.post("/api/sensors/{sensor_id}/disconnect")
    async def disconnect_sensor(sensor_id: str):
        """Disconnect a sensor's hardware adapter and reset its init_state."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        ok, err = ctx.daemon.request_sensor_disconnect(sensor_id)
        if not ok:
            if err and err.startswith("Unknown sensor"):
                return JSONResponse({"error": err}, status_code=404)
            return JSONResponse({"error": err or "Disconnect failed"}, status_code=500)
        return {"success": True, "message": f"Sensor {sensor_id} disconnected"}

    # ── Sensor-type discovery (registry metadata, class-level schema) ─

    @router.get("/api/sensor-types")
    async def list_sensor_types():
        """Return all registered sensor types for the Add Sensor picker.

        Shape: ``{"types": [{value, label, description}, ...]}``. ``value``
        is the registry key that the frontend sends back in
        ``POST /api/config/sensors``.
        """
        types = []
        for key, info in _list_sensor_registry().items():
            types.append(
                {
                    "value": key,
                    "label": _SENSOR_TYPE_LABELS.get(key, key),
                    "description": info.get("description", ""),
                }
            )
        return {"types": types}

    @router.get("/api/sensor-types/{sensor_type}/schema")
    async def get_sensor_type_schema(sensor_type: str, current_settings: str = ""):
        """Class-level settings schema for a sensor type.

        Used by the Hardware config tab when ``sensor.type`` is
        something other than ``telescope`` (telescope schemas are
        adapter-specific and live behind
        ``/api/hardware-adapters/{adapter}/schema``). The class must
        expose a ``build_settings_schema()`` classmethod returning the
        same :class:`SettingSchemaEntry` shape the hardware-adapter
        schema endpoint uses, so the existing form renderer can drive
        the returned fields unchanged.

        ``current_settings`` is an optional JSON-encoded query param —
        the same shape :func:`_fetch_adapter_schema` accepts on the
        telescope-adapter route — so sensor types that build a
        conditional schema (e.g. allsky reloading the form when the user
        picks a different camera) can react to live form state.  Unknown
        keys are passed through verbatim; invalid JSON is silently
        ignored to mirror the hardware-adapter route's tolerance.
        """
        try:
            cls = get_sensor_class(sensor_type)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        except ImportError as exc:
            CITRASENSE_LOGGER.error("Failed to import sensor type %s: %s", sensor_type, exc)
            return JSONResponse(
                {"error": f"Sensor type {sensor_type!r} is not installed: {exc}"},
                status_code=500,
            )

        schema_fn = getattr(cls, "build_settings_schema", None)
        if not callable(schema_fn):
            return JSONResponse(
                {
                    "error": (
                        f"Sensor type {sensor_type!r} does not expose a class-level schema — "
                        "use the per-adapter schema endpoint instead."
                    )
                },
                status_code=400,
            )

        settings_kwargs: dict[str, Any] = {}
        if current_settings:
            try:
                parsed = json.loads(current_settings)
                if isinstance(parsed, dict):
                    settings_kwargs = parsed
            except json.JSONDecodeError:
                # Match the hardware-adapter route's tolerance — bad JSON
                # falls back to a plain (no-kwarg) schema rather than 400ing.
                pass

        # Inspect the signature once and dispatch deterministically: only
        # forward ``**settings_kwargs`` when the callable advertises
        # ``**kwargs``. The previous ``except TypeError`` fallback would
        # silently swallow real bugs raised inside ``build_settings_schema``
        # (e.g. an int/str mismatch deep in conditional schema logic) and
        # serve a stale static schema instead of 500ing.
        try:
            sig = inspect.signature(schema_fn)
        except (TypeError, ValueError):
            # Builtin or C-implemented callables can refuse introspection.
            # Fall back to no-kwarg dispatch — same effective behavior the
            # narrow ``except TypeError`` used to provide here.
            sig = None
        accepts_kwargs = (
            bool(settings_kwargs)
            and sig is not None
            and any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        )
        try:
            schema = schema_fn(**settings_kwargs) if accepts_kwargs else schema_fn()
        except Exception as exc:
            CITRASENSE_LOGGER.error("Error building schema for %s: %s", sensor_type, exc, exc_info=True)
            return JSONResponse({"error": str(exc)}, status_code=500)
        return {"schema": schema}

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
        known_types = set(_list_sensor_registry().keys())
        if sensor_type not in known_types:
            valid = ", ".join(sorted(known_types))
            return JSONResponse(
                {"error": f"Unknown sensor type {sensor_type!r}. Valid options: {valid}"},
                status_code=400,
            )
        # Only telescopes use the hardware-adapter registry today —
        # streaming sensors (passive_radar) carry all their per-sensor
        # config in ``adapter_settings`` and keep ``adapter`` empty.
        if sensor_type == "telescope" and not adapter:
            return JSONResponse({"error": "Telescope sensors require an adapter"}, status_code=400)
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
