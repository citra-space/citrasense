"""Hardware adapter discovery, reconnect, scan, and configuration update."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from citrascope.hardware.adapter_registry import get_adapter_schema as get_schema
from citrascope.hardware.adapter_registry import list_adapters
from citrascope.hardware.devices.abstract_hardware_device import AbstractHardwareDevice
from citrascope.logging import CITRASCOPE_LOGGER

if TYPE_CHECKING:
    from citrascope.web.app import CitraScopeWebApp


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
        CITRASCOPE_LOGGER.warning(
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
        CITRASCOPE_LOGGER.error(f"Error getting schema for {adapter_name}: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


def build_hardware_router(ctx: CitraScopeWebApp) -> APIRouter:
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

    @router.post("/hardware/reconnect")
    async def reconnect_hardware():
        """Retry hardware connection using current in-memory settings."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        if not ctx.daemon.settings.is_configured():
            return JSONResponse(
                {"error": "Configuration incomplete — configure hardware adapter first"},
                status_code=400,
            )

        success, error = await asyncio.to_thread(ctx.daemon.retry_connection)

        if success:
            return {"status": "success", "message": "Hardware reconnected successfully"}
        return JSONResponse(
            {"status": "error", "message": f"Reconnect failed: {error}", "error": error},
            status_code=500,
        )

    @router.post("/hardware/scan")
    async def scan_hardware(body: dict[str, Any]):
        """Clear hardware probe caches and return a fresh adapter schema."""
        adapter_name = body.get("adapter_name", "")
        if not adapter_name:
            return JSONResponse({"error": "adapter_name is required"}, status_code=400)

        current_settings = body.get("current_settings", {})
        if not isinstance(current_settings, dict):
            return JSONResponse({"error": "current_settings must be a JSON object"}, status_code=400)

        def _scan() -> list:
            AbstractHardwareDevice._hardware_probe_cache.clear()
            try:
                from citrascope.hardware.devices.mount.zwo_am_mount import ZwoAmMount

                ZwoAmMount._port_cache = None
                ZwoAmMount._port_cache_timestamp = 0
            except ImportError:
                pass
            try:
                from citrascope.hardware.devices.camera.moravian_camera import MoravianCamera

                MoravianCamera._read_mode_cache = None
            except ImportError:
                pass
            return get_schema(adapter_name, **current_settings)

        try:
            schema = await asyncio.wait_for(asyncio.to_thread(_scan), timeout=30.0)
            return {"schema": schema}
        except asyncio.TimeoutError:
            CITRASCOPE_LOGGER.warning("Hardware scan for %s timed out", adapter_name)
            return JSONResponse(
                {"error": "Hardware scan timed out — a device may be unresponsive"},
                status_code=504,
            )
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        except Exception as e:
            CITRASCOPE_LOGGER.error(f"Error scanning hardware for {adapter_name}: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.post("/config")
    async def update_config(config: dict[str, Any]):
        """Update configuration and trigger hot-reload."""
        try:
            if not ctx.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)

            required_fields = ["personal_access_token", "telescope_id", "hardware_adapter"]
            for field in required_fields:
                if field not in config or not config[field]:
                    return JSONResponse(
                        {"error": f"Missing required field: {field}"},
                        status_code=400,
                    )

            adapter_name = config.get("hardware_adapter")
            adapter_settings = config.get("adapter_settings", {})

            if adapter_name:
                schema_response = await _fetch_adapter_schema(adapter_name)
                if isinstance(schema_response, JSONResponse):
                    return schema_response

                schema = schema_response.get("schema", [])

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
            CITRASCOPE_LOGGER.error(f"Error updating config: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    return router
