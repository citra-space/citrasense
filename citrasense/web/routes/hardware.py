"""Hardware adapter discovery, reconnect, scan, and configuration update."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from citrasense.hardware.adapter_registry import get_adapter_schema as get_schema
from citrasense.hardware.adapter_registry import list_adapters
from citrasense.hardware.devices.abstract_hardware_device import AbstractHardwareDevice
from citrasense.logging import CITRASENSE_LOGGER
from citrasense.sensors.sensor_registry import get_sensor_class
from citrasense.settings.citrasense_settings import CitraSenseSettings

if TYPE_CHECKING:
    from citrasense.web.app import CitraSenseWebApp


def _fetch_sensor_type_schema(sensor_type: str) -> list | JSONResponse:
    """Return the class-level settings schema for a non-telescope sensor type.

    Mirrors :func:`_fetch_adapter_schema` for the hardware-adapter path —
    runs synchronously (callers wrap in :func:`asyncio.to_thread`) and
    converts registry misses into a ``JSONResponse`` for easy inline
    propagation.
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
            {"error": f"Sensor type {sensor_type!r} does not expose a class-level schema"},
            status_code=400,
        )
    try:
        schema = schema_fn()
    except Exception as exc:
        CITRASENSE_LOGGER.error("Error building schema for %s: %s", sensor_type, exc, exc_info=True)
        return JSONResponse({"error": str(exc)}, status_code=500)
    # ``schema_fn`` was fetched via ``getattr`` so pyright widens the
    # return to ``object``; narrow it back to ``list`` for the caller.
    if not isinstance(schema, list):
        return JSONResponse(
            {"error": f"Sensor type {sensor_type!r} schema must be a list (got {type(schema).__name__})"},
            status_code=500,
        )
    return schema


async def _fetch_adapter_schema(adapter_name: str, current_settings: str = "") -> Any:
    """Shared implementation for adapter schema retrieval (used by GET /schema and POST /config)."""
    try:
        settings_kwargs = {}
        if current_settings:
            try:
                settings_kwargs = json.loads(current_settings)
            except json.JSONDecodeError:
                pass

        schema = await asyncio.wait_for(
            asyncio.to_thread(get_schema, adapter_name, **settings_kwargs),
            timeout=15.0,
        )
        return {"schema": schema}
    except asyncio.TimeoutError:
        CITRASENSE_LOGGER.warning(
            "Schema generation for %s timed out — hardware probe may be hung",
            adapter_name,
        )
        return JSONResponse(
            {"error": "Schema generation timed out — hardware may need a power cycle"},
            status_code=504,
        )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        CITRASENSE_LOGGER.error(f"Error getting schema for {adapter_name}: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


def build_hardware_router(ctx: CitraSenseWebApp) -> APIRouter:
    """Routes for hardware adapter listing, schema, reconnect, scan, and config update."""
    router = APIRouter(prefix="/api", tags=["hardware"])

    @router.get("/hardware-adapters")
    async def get_hardware_adapters():
        """Get list of available hardware adapters."""
        adapters_info = list_adapters()
        return {
            "adapters": list(adapters_info.keys()),
            "descriptions": {name: info["description"] for name, info in adapters_info.items()},
        }

    @router.get("/hardware-adapters/{adapter_name}/schema")
    async def get_adapter_schema(adapter_name: str, current_settings: str = ""):
        """Get configuration schema for a specific hardware adapter."""
        return await _fetch_adapter_schema(adapter_name, current_settings)

    @router.post("/hardware/scan")
    async def scan_hardware(body: dict[str, Any]):
        """Clear hardware probe caches and return a fresh adapter schema.

        .. note::

           This endpoint clears the **process-wide**
           :attr:`AbstractHardwareDevice._hardware_probe_cache`, plus a few
           device-class-specific caches (``ZwoAmMount._port_cache``,
           ``MoravianCamera._read_mode_cache``).  In a multi-sensor
           deployment this invalidates probe results for *every* sensor's
           devices, not just the one whose adapter schema is being
           refreshed — subsequent ``get_settings_schema`` calls for other
           sensors will re-enumerate their hardware.  That is safe (probes
           are idempotent) and inexpensive (probes run in a subprocess
           with a timeout), but it is worth being aware of if you are
           debugging why a different sensor's USB scan happens to run at
           the same time.

           Narrower, per-adapter invalidation would require a mapping from
           adapter → probed device classes; we intentionally keep the
           scan broad until that inventory is needed.
        """
        adapter_name = body.get("adapter_name", "")
        if not adapter_name:
            return JSONResponse({"error": "adapter_name is required"}, status_code=400)

        current_settings = body.get("current_settings", {})
        if not isinstance(current_settings, dict):
            return JSONResponse({"error": "current_settings must be a JSON object"}, status_code=400)

        def _scan() -> list:
            # Process-wide cache clear — see the endpoint docstring for
            # the multi-sensor implication.
            AbstractHardwareDevice._hardware_probe_cache.clear()
            try:
                from citrasense.hardware.devices.mount.zwo_am_mount import ZwoAmMount

                ZwoAmMount._port_cache = None
                ZwoAmMount._port_cache_timestamp = 0
            except ImportError:
                pass
            try:
                from citrasense.hardware.devices.camera.moravian_camera import MoravianCamera

                MoravianCamera._read_mode_cache = None
            except ImportError:
                pass
            return get_schema(adapter_name, **current_settings)

        try:
            schema = await asyncio.wait_for(asyncio.to_thread(_scan), timeout=30.0)
            return {"schema": schema}
        except asyncio.TimeoutError:
            CITRASENSE_LOGGER.warning("Hardware scan for %s timed out", adapter_name)
            return JSONResponse(
                {"error": "Hardware scan timed out — a device may be unresponsive"},
                status_code=504,
            )
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        except Exception as e:
            CITRASENSE_LOGGER.error(f"Error scanning hardware for {adapter_name}: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.post("/config")
    async def update_config(config: dict[str, Any]):
        """Update configuration and trigger hot-reload."""
        try:
            if not ctx.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)

            if not config.get("personal_access_token"):
                return JSONResponse({"error": "Missing required field: personal_access_token"}, status_code=400)

            sensors = config.get("sensors", [])
            if not sensors:
                return JSONResponse({"error": "At least one sensor is required"}, status_code=400)

            # Reject duplicate citra_sensor_id up front — two local sensors
            # cannot share one backend telescope record without silently
            # starving one of them (TaskDispatcher routes API ids to the
            # first matching runtime; the second scope would sit idle).
            dupes = CitraSenseSettings.find_duplicate_citra_sensor_ids(sensors)
            if dupes:
                detail = "; ".join(
                    f"{api_id!r} claimed by {', '.join(local_ids)}" for api_id, local_ids in dupes.items()
                )
                return JSONResponse(
                    {
                        "error": (
                            "Duplicate citra_sensor_id — each local sensor must map to a distinct "
                            f"Citra telescope id. Collisions: {detail}."
                        )
                    },
                    status_code=400,
                )

            for sensor_cfg in sensors:
                sensor_type = sensor_cfg.get("type") or "telescope"
                adapter_name = sensor_cfg.get("adapter") or ""
                # ``adapter`` is required for the telescope modality (it
                # selects the N.I.N.A./KStars/INDI/Direct backend) but
                # intentionally empty for streaming sensors like
                # ``passive_radar`` whose per-sensor config lives in
                # ``adapter_settings`` instead.
                if sensor_type == "telescope" and not adapter_name:
                    return JSONResponse(
                        {"error": f"Sensor '{sensor_cfg.get('id', '?')}' missing adapter"},
                        status_code=400,
                    )
                if not sensor_cfg.get("citra_sensor_id"):
                    return JSONResponse(
                        {"error": f"Sensor '{sensor_cfg.get('id', '?')}' missing citra_sensor_id"},
                        status_code=400,
                    )

                adapter_settings = sensor_cfg.get("adapter_settings", {})
                # Fetch the relevant schema: per-adapter for telescopes,
                # class-level for modality-native streaming sensors.
                if sensor_type == "telescope":
                    schema_response = await _fetch_adapter_schema(adapter_name)
                    if isinstance(schema_response, JSONResponse):
                        return schema_response
                    schema = schema_response.get("schema", [])
                else:
                    schema = await asyncio.to_thread(_fetch_sensor_type_schema, sensor_type)
                    if isinstance(schema, JSONResponse):
                        return schema

                for field_schema in schema:
                    field_name = field_schema.get("name")
                    is_required = field_schema.get("required", False)
                    if is_required and field_name not in adapter_settings:
                        return JSONResponse(
                            {"error": f"Missing required adapter setting: {field_name}"},
                            status_code=400,
                        )
                    if field_name in adapter_settings:
                        value = adapter_settings[field_name]
                        field_type = field_schema.get("type")
                        if field_type == "int":
                            try:
                                value = int(value)
                                adapter_settings[field_name] = value
                            except (ValueError, TypeError):
                                return JSONResponse(
                                    {"error": f"Field '{field_name}' must be an integer"},
                                    status_code=400,
                                )
                            if "min" in field_schema and value < field_schema["min"]:
                                return JSONResponse(
                                    {"error": f"Field '{field_name}' must be >= {field_schema['min']}"},
                                    status_code=400,
                                )
                            if "max" in field_schema and value > field_schema["max"]:
                                return JSONResponse(
                                    {"error": f"Field '{field_name}' must be <= {field_schema['max']}"},
                                    status_code=400,
                                )
                        elif field_type == "float":
                            try:
                                value = float(value)
                                adapter_settings[field_name] = value
                            except (ValueError, TypeError):
                                return JSONResponse(
                                    {"error": f"Field '{field_name}' must be a number"},
                                    status_code=400,
                                )

            for dir_field in ("custom_data_dir", "custom_log_dir"):
                dir_value = config.get(dir_field, "")
                if dir_value:
                    dir_path = Path(dir_value)
                    if not dir_path.is_absolute():
                        return JSONResponse(
                            {"error": f"{dir_field} must be an absolute path"},
                            status_code=400,
                        )
                    try:
                        dir_path.mkdir(parents=True, exist_ok=True)
                    except OSError as e:
                        return JSONResponse(
                            {"error": f"Cannot create {dir_field} '{dir_value}': {e}"},
                            status_code=400,
                        )

            ctx.daemon.settings.update_and_save(config)

            success, error = await asyncio.to_thread(ctx.daemon.reload_configuration)

            if success:
                return {
                    "status": "success",
                    "message": "Configuration updated and reloaded successfully",
                }
            return JSONResponse(
                {
                    "status": "error",
                    "message": f"Configuration saved but reload failed: {error}",
                    "error": error,
                },
                status_code=500,
            )

        except Exception as e:
            CITRASENSE_LOGGER.error(f"Error updating config: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    return router
