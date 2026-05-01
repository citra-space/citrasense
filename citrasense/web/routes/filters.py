"""Filter wheel management endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from citrasense.logging import CITRASENSE_LOGGER
from citrasense.web.helpers import FILTER_NAME_OPTIONS, get_sensor_context

if TYPE_CHECKING:
    from citrasense.sensors.abstract_sensor import AbstractSensor
    from citrasense.web.app import CitraSenseWebApp


def _resolve_filter_adapter(sensor: AbstractSensor) -> Any | JSONResponse:
    """Return the sensor's hardware adapter or a 404 ``JSONResponse``.

    Filter management is a telescope concept — :class:`AllskyCameraSensor`
    and :class:`PassiveRadarSensor` have no ``.adapter`` attribute, so a
    naive ``sensor.adapter`` access raises ``AttributeError`` and the
    request bubbles a 500 (issue #342).  The frontend already treats 404
    as "this sensor does not have filters" and hides the panel cleanly,
    so 404 is the right code here.
    """
    adapter = getattr(sensor, "adapter", None)
    if adapter is None:
        return JSONResponse(
            {"error": "Sensor does not support filter management"},
            status_code=404,
        )
    return adapter


def build_filters_router(ctx: CitraSenseWebApp) -> APIRouter:
    """Filter configuration, batch updates, sync, and manual position change."""
    router = APIRouter(prefix="/api/sensors/{sensor_id}", tags=["filters"])

    @router.get("/filters")
    async def get_filters(sensor_id: str):
        """Get current filter configuration."""
        sensor, _runtime = get_sensor_context(ctx, sensor_id)
        adapter = _resolve_filter_adapter(sensor)
        if isinstance(adapter, JSONResponse):
            return adapter

        if not adapter.supports_filter_management():
            return JSONResponse({"error": "Adapter does not support filter management"}, status_code=404)

        try:
            filter_config = adapter.get_filter_config()
            names_editable = adapter.supports_filter_rename()
            response: dict = {"filters": filter_config, "names_editable": names_editable}
            if names_editable:
                response["filter_name_options"] = FILTER_NAME_OPTIONS
            return response
        except Exception as e:
            CITRASENSE_LOGGER.error(f"Error getting filter config: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.post("/filters/batch")
    async def update_filters_batch(sensor_id: str, updates: list[dict[str, Any]]):
        """Update multiple filters atomically with single disk write."""
        sensor, _runtime = get_sensor_context(ctx, sensor_id)
        adapter = _resolve_filter_adapter(sensor)
        if isinstance(adapter, JSONResponse):
            return adapter

        if not updates or not isinstance(updates, list):
            return JSONResponse({"error": "Updates must be a non-empty array"}, status_code=400)

        try:
            filter_config = adapter.filter_map

            validated_updates = []
            for idx, update in enumerate(updates):
                if not isinstance(update, dict):
                    return JSONResponse({"error": f"Update at index {idx} must be an object"}, status_code=400)

                if "filter_id" not in update:
                    return JSONResponse({"error": f"Update at index {idx} missing filter_id"}, status_code=400)

                filter_id = update["filter_id"]
                try:
                    filter_id_int = int(filter_id)
                except (ValueError, TypeError):
                    return JSONResponse({"error": f"Invalid filter_id at index {idx}: {filter_id}"}, status_code=400)

                if filter_id_int not in filter_config:
                    return JSONResponse({"error": f"Filter ID {filter_id} not found"}, status_code=404)

                validated_update: dict[str, int | str | bool | None] = {"filter_id_int": filter_id_int}

                if "focus_position" in update:
                    focus_position = update["focus_position"]
                    if focus_position is None:
                        validated_update["focus_position"] = None
                    elif not isinstance(focus_position, int):
                        return JSONResponse(
                            {"error": f"focus_position at index {idx} must be an integer or null"}, status_code=400
                        )
                    elif focus_position < 0 or focus_position > 65535:
                        return JSONResponse(
                            {"error": f"focus_position at index {idx} must be between 0 and 65535"},
                            status_code=400,
                        )
                    else:
                        validated_update["focus_position"] = focus_position

                if "enabled" in update:
                    enabled = update["enabled"]
                    if not isinstance(enabled, bool):
                        return JSONResponse({"error": f"enabled at index {idx} must be a boolean"}, status_code=400)
                    validated_update["enabled"] = enabled

                if "name" in update:
                    name = update["name"]
                    if not isinstance(name, str) or not name.strip():
                        return JSONResponse(
                            {"error": f"name at index {idx} must be a non-empty string"}, status_code=400
                        )
                    validated_update["name"] = name.strip()

                validated_updates.append(validated_update)

            current_enabled = {fid for fid, fdata in filter_config.items() if fdata.get("enabled", True)}
            for validated in validated_updates:
                if "enabled" in validated:
                    if validated["enabled"]:
                        current_enabled.add(validated["filter_id_int"])
                    else:
                        current_enabled.discard(validated["filter_id_int"])

            if not current_enabled:
                return JSONResponse(
                    {"error": "Cannot disable all filters. At least one filter must remain enabled."},
                    status_code=400,
                )

            for validated in validated_updates:
                filter_id_int = validated["filter_id_int"]

                if "focus_position" in validated:
                    if not adapter.update_filter_focus(str(filter_id_int), validated["focus_position"]):
                        return JSONResponse(
                            {"error": f"Failed to update filter {filter_id_int} focus"}, status_code=500
                        )

                if "enabled" in validated:
                    if not adapter.update_filter_enabled(str(filter_id_int), validated["enabled"]):
                        return JSONResponse(
                            {"error": f"Failed to update filter {filter_id_int} enabled state"}, status_code=500
                        )

                if "name" in validated and adapter.supports_filter_rename():
                    if not adapter.update_filter_name(str(filter_id_int), validated["name"]):
                        return JSONResponse({"error": f"Failed to update filter {filter_id_int} name"}, status_code=500)

            if ctx.daemon:
                ctx.daemon.save_filter_config(sensor)

            return {"success": True, "updated_count": len(validated_updates)}

        except Exception as e:
            CITRASENSE_LOGGER.error(f"Error in batch filter update: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.post("/filters/sync")
    async def sync_filters_to_backend(sensor_id: str):
        """Explicitly sync filter configuration to backend API."""
        sensor, _runtime = get_sensor_context(ctx, sensor_id)
        adapter = _resolve_filter_adapter(sensor)
        if isinstance(adapter, JSONResponse):
            return adapter
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        try:
            ctx.daemon.sync_filters_to_backend(sensor)
            return {"success": True, "message": "Filters synced to backend"}
        except Exception as e:
            CITRASENSE_LOGGER.error(f"Error syncing filters to backend: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.post("/filter/set")
    async def set_filter_position(sensor_id: str, body: dict[str, Any]):
        """Command the filter wheel to move to a specific position."""
        sensor, _runtime = get_sensor_context(ctx, sensor_id)
        adapter = _resolve_filter_adapter(sensor)
        if isinstance(adapter, JSONResponse):
            return adapter
        if busy := ctx.require_sensor_idle(_runtime):
            return busy

        if not adapter.filter_map:
            return JSONResponse({"error": "No filter wheel available"}, status_code=404)

        position = body.get("position")
        if position is None or not isinstance(position, int):
            return JSONResponse({"error": "position must be an integer"}, status_code=400)

        if position not in adapter.filter_map:
            return JSONResponse({"error": f"Invalid filter position: {position}"}, status_code=400)

        try:
            success = adapter.set_filter(position)
            if success:
                name = adapter.filter_map[position].get("name", f"Filter {position}")
                return {"success": True, "position": position, "name": name}
            return JSONResponse({"error": "Filter change failed"}, status_code=500)
        except Exception as e:
            CITRASENSE_LOGGER.error(f"Error setting filter position: {e}", exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)

    return router
