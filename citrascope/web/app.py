"""FastAPI web application for CitraScope monitoring and configuration."""

import json
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from citrascope.constants import (
    AUTOFOCUS_TARGET_PRESETS,
    DEV_API_HOST,
    DEV_APP_URL,
    PROD_APP_URL,
)
from citrascope.hardware.adapter_registry import get_adapter_schema as get_schema
from citrascope.hardware.adapter_registry import list_adapters
from citrascope.logging import CITRASCOPE_LOGGER


class SystemStatus(BaseModel):
    """Current system status."""

    telescope_connected: bool = False
    camera_connected: bool = False
    supports_direct_camera_control: bool = False
    current_task: str | None = None
    tasks_pending: int = 0
    processing_active: bool = True
    automated_scheduling: bool = False
    hardware_adapter: str = "unknown"
    telescope_ra: float | None = None
    telescope_dec: float | None = None
    telescope_az: float | None = None
    telescope_alt: float | None = None
    ground_station_id: str | None = None
    ground_station_name: str | None = None
    ground_station_url: str | None = None
    autofocus_requested: bool = False
    autofocus_running: bool = False
    autofocus_progress: str = ""
    last_autofocus_timestamp: int | None = None
    next_autofocus_minutes: int | None = None
    time_health: dict[str, Any] | None = None
    gps_location: dict[str, Any] | None = None
    last_update: str = ""
    missing_dependencies: list[dict[str, str]] = []  # List of {device, packages, install_cmd}
    active_processors: list[str] = []  # Names of enabled image processors
    tasks_by_stage: dict[str, list[dict]] | None = None  # Tasks in each pipeline stage
    pipeline_stats: dict[str, Any] | None = None  # Lifetime counters for queues, processors, and tasks
    supports_alignment: bool = False
    supports_autofocus: bool = False
    supports_manual_sync: bool = False
    mount_at_home: bool = False
    mount_homing: bool = False
    mount_horizon_limit: int | None = None
    mount_overhead_limit: int | None = None
    alignment_requested: bool = False
    alignment_running: bool = False
    alignment_progress: str = ""
    last_alignment_timestamp: int | None = None
    safety_status: dict[str, Any] | None = None


class HardwareConfig(BaseModel):
    """Hardware configuration settings."""

    adapter: str
    indi_server_url: str | None = None
    indi_server_port: int | None = None
    indi_telescope_name: str | None = None
    indi_camera_name: str | None = None
    nina_url_prefix: str | None = None


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        CITRASCOPE_LOGGER.info(f"WebSocket client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        CITRASCOPE_LOGGER.info(f"WebSocket client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                CITRASCOPE_LOGGER.warning(f"Failed to send to WebSocket client: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def broadcast_text(self, message: str):
        """Broadcast text message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                CITRASCOPE_LOGGER.warning(f"Failed to send to WebSocket client: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)


class CitraScopeWebApp:
    """Web application for CitraScope."""

    def __init__(self, daemon=None, web_log_handler=None):
        self.app = FastAPI(title="CitraScope", description="Telescope Control and Monitoring")
        self.daemon = daemon
        self.connection_manager = ConnectionManager()
        self.status = SystemStatus()
        self.web_log_handler = web_log_handler

        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Mount static files with no-cache headers so browsers always
        # pick up changes during development.  In production behind a
        # reverse proxy, the proxy can add its own long-lived cache headers.
        static_dir = Path(__file__).parent / "static"
        if static_dir.exists():
            self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

            @self.app.middleware("http")
            async def _no_cache_static(request: Request, call_next):
                response = await call_next(request)
                if request.url.path.startswith("/static/"):
                    response.headers["Cache-Control"] = "no-cache, must-revalidate"
                return response

        # Mount images directory for camera captures (read-only access)
        if daemon and hasattr(daemon, "settings"):
            images_dir = daemon.settings.get_images_dir()
            if images_dir.exists():
                self.app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")

        # Initialize Jinja2 templates with a cache-buster so browsers pick up
        # new static files after each daemon restart.
        templates_dir = Path(__file__).parent / "templates"
        self.templates = Jinja2Templates(directory=str(templates_dir))
        self._cache_bust = str(int(time.time()))
        self.templates.env.globals["cache_bust"] = self._cache_bust

        # Register routes
        self._setup_routes()

    def set_daemon(self, daemon):
        """Set the daemon instance after initialization."""
        self.daemon = daemon

    def _setup_routes(self):
        """Setup all API routes."""

        @self.app.get("/", response_class=HTMLResponse)
        async def root(request: Request):
            """Serve the main dashboard page."""
            return self.templates.TemplateResponse("dashboard.html", {"request": request})

        @self.app.get("/api/status")
        async def get_status():
            """Get current system status."""
            if self.daemon:
                self._update_status_from_daemon()
            return self.status

        @self.app.get("/api/config")
        async def get_config():
            """Get current configuration."""
            if not self.daemon or not self.daemon.settings:
                return JSONResponse({"error": "Configuration not available"}, status_code=503)

            settings = self.daemon.settings
            # Determine app URL based on API host
            app_url = DEV_APP_URL if settings.host == DEV_API_HOST else PROD_APP_URL

            # Get config file path
            config_path = str(settings.config_manager.get_config_path())

            # Get current log file path
            log_file_path = (
                str(settings.config_manager.get_current_log_path()) if settings.file_logging_enabled else None
            )

            # Get images directory path
            images_dir_path = str(settings.get_images_dir())

            # Get processing working directory path (sibling to images directory)
            processing_dir_path = str(settings.get_images_dir().parent / "processing")

            return {
                "host": settings.host,
                "port": settings.port,
                "use_ssl": settings.use_ssl,
                "personal_access_token": settings.personal_access_token,
                "telescope_id": settings.telescope_id,
                "use_dummy_api": settings.use_dummy_api,
                "hardware_adapter": settings.hardware_adapter,
                "adapter_settings": settings._all_adapter_settings,
                "log_level": settings.log_level,
                "keep_images": settings.keep_images,
                "file_logging_enabled": settings.file_logging_enabled,
                "log_retention_days": settings.log_retention_days,
                "max_task_retries": settings.max_task_retries,
                "initial_retry_delay_seconds": settings.initial_retry_delay_seconds,
                "max_retry_delay_seconds": settings.max_retry_delay_seconds,
                "scheduled_autofocus_enabled": settings.scheduled_autofocus_enabled,
                "autofocus_interval_minutes": settings.autofocus_interval_minutes,
                "last_autofocus_timestamp": settings.last_autofocus_timestamp,
                "autofocus_target_preset": settings.autofocus_target_preset,
                "autofocus_target_custom_ra": settings.autofocus_target_custom_ra,
                "autofocus_target_custom_dec": settings.autofocus_target_custom_dec,
                "alignment_exposure_seconds": settings.alignment_exposure_seconds,
                "align_on_startup": settings.align_on_startup,
                "last_alignment_timestamp": settings.last_alignment_timestamp,
                "time_check_interval_minutes": settings.time_check_interval_minutes,
                "time_offset_pause_ms": settings.time_offset_pause_ms,
                "gps_location_updates_enabled": settings.gps_location_updates_enabled,
                "gps_update_interval_minutes": settings.gps_update_interval_minutes,
                "task_processing_paused": settings.task_processing_paused,
                "processors_enabled": settings.processors_enabled,
                "enabled_processors": settings.enabled_processors,
                "app_url": app_url,
                "config_file_path": config_path,
                "log_file_path": log_file_path,
                "images_dir_path": images_dir_path,
                "processing_dir_path": processing_dir_path,
            }

        @self.app.get("/api/config/status")
        async def get_config_status():
            """Get configuration status."""
            if not self.daemon or not self.daemon.settings:
                return {"configured": False, "error": "Settings not available"}

            return {
                "configured": self.daemon.settings.is_configured(),
                "error": getattr(self.daemon, "configuration_error", None),
            }

        @self.app.get("/api/version")
        async def get_version():
            """Get CitraScope version."""
            try:
                pkg_version = version("citrascope")
                return {"version": pkg_version}
            except PackageNotFoundError:
                return {"version": "development"}

        @self.app.get("/api/hardware-adapters")
        async def get_hardware_adapters():
            """Get list of available hardware adapters."""
            adapters_info = list_adapters()
            return {
                "adapters": list(adapters_info.keys()),
                "descriptions": {name: info["description"] for name, info in adapters_info.items()},
            }

        @self.app.get("/api/hardware-adapters/{adapter_name}/schema")
        async def get_adapter_schema(adapter_name: str, current_settings: str = ""):
            """Get configuration schema for a specific hardware adapter.

            Args:
                adapter_name: Name of the adapter
                current_settings: JSON string of current adapter_settings (for dynamic schemas)
            """
            try:
                # Parse current settings if provided
                settings_kwargs = {}
                if current_settings:
                    try:
                        settings_kwargs = json.loads(current_settings)
                    except json.JSONDecodeError:
                        pass  # Ignore invalid JSON, use empty kwargs

                schema = get_schema(adapter_name, **settings_kwargs)
                return {"schema": schema}
            except ValueError as e:
                # Invalid adapter name
                return JSONResponse({"error": str(e)}, status_code=404)
            except Exception as e:
                CITRASCOPE_LOGGER.error(f"Error getting schema for {adapter_name}: {e}", exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.post("/api/config")
        async def update_config(config: dict[str, Any]):
            """Update configuration and trigger hot-reload."""
            try:
                if not self.daemon:
                    return JSONResponse({"error": "Daemon not available"}, status_code=503)

                # Validate required fields
                required_fields = ["personal_access_token", "telescope_id", "hardware_adapter"]
                for field in required_fields:
                    if field not in config or not config[field]:
                        return JSONResponse(
                            {"error": f"Missing required field: {field}"},
                            status_code=400,
                        )

                # Validate adapter_settings against schema if adapter is specified
                adapter_name = config.get("hardware_adapter")
                adapter_settings = config.get("adapter_settings", {})

                if adapter_name:
                    # Get schema for validation
                    schema_response = await get_adapter_schema(adapter_name)
                    if isinstance(schema_response, JSONResponse):
                        return schema_response  # Error getting schema

                    schema = schema_response.get("schema", [])

                    # Validate required fields in adapter settings
                    for field_schema in schema:
                        field_name = field_schema.get("name")
                        is_required = field_schema.get("required", False)

                        if is_required and field_name not in adapter_settings:
                            return JSONResponse(
                                {"error": f"Missing required adapter setting: {field_name}"},
                                status_code=400,
                            )

                        # Validate type and constraints if present
                        if field_name in adapter_settings:
                            value = adapter_settings[field_name]
                            field_type = field_schema.get("type")

                            # Type validation
                            if field_type == "int":
                                try:
                                    value = int(value)
                                    adapter_settings[field_name] = value
                                except (ValueError, TypeError):
                                    return JSONResponse(
                                        {"error": f"Field '{field_name}' must be an integer"},
                                        status_code=400,
                                    )

                                # Range validation
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

                self.daemon.settings.update_and_save(config)

                # Trigger hot-reload
                success, error = self.daemon.reload_configuration()

                if success:
                    return {
                        "status": "success",
                        "message": "Configuration updated and reloaded successfully",
                    }
                else:
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

        @self.app.get("/api/processors")
        async def get_processors():
            """Get list of all available processors with metadata."""
            if not self.daemon or not hasattr(self.daemon, "processor_registry") or not self.daemon.processor_registry:
                return []

            return self.daemon.processor_registry.get_all_processors()

        @self.app.get("/api/tasks")
        async def get_tasks():
            """Get scheduled task queue (not yet started or waiting to retry)."""
            if not self.daemon or not hasattr(self.daemon, "task_manager") or self.daemon.task_manager is None:
                return []

            task_manager = self.daemon.task_manager
            tasks = []

            # Get IDs of tasks that are actively in a stage
            with task_manager._stage_lock:
                active_ids = set()
                active_ids.update(task_manager.imaging_tasks.keys())
                active_ids.update(task_manager.processing_tasks.keys())
                active_ids.update(task_manager.uploading_tasks.keys())

            with task_manager.heap_lock:
                for start_time, stop_time, task_id, task in task_manager.task_heap:
                    # Only include tasks not currently in a stage (scheduled future work)
                    if task_id not in active_ids:
                        tasks.append(
                            {
                                "id": task_id,
                                "start_time": datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat(),
                                "stop_time": (
                                    datetime.fromtimestamp(stop_time, tz=timezone.utc).isoformat()
                                    if stop_time
                                    else None
                                ),
                                "status": task.status,
                                "target": getattr(task, "satelliteName", getattr(task, "target", "unknown")),
                            }
                        )

            return tasks

        @self.app.get("/api/tasks/active")
        async def get_active_tasks():
            """Get currently executing tasks (all stages)."""
            if not self.daemon or not hasattr(self.daemon, "task_manager") or self.daemon.task_manager is None:
                return []

            # Use get_tasks_by_stage which returns enriched task info
            tasks_by_stage = self.daemon.task_manager.get_tasks_by_stage()

            # Flatten into single list with stage information
            active = []
            for stage, tasks in tasks_by_stage.items():
                for task_info in tasks:
                    active.append(
                        {
                            "id": task_info["task_id"],
                            "target": task_info.get("target_name", "unknown"),
                            "stage": stage,
                            "elapsed": task_info["elapsed"],
                            "status_msg": task_info.get("status_msg"),
                            "retry_scheduled_time": task_info.get("retry_scheduled_time"),
                            "is_being_executed": task_info.get("is_being_executed", False),
                        }
                    )

            return active

        @self.app.get("/api/logs")
        async def get_logs(limit: int = 100):
            """Get recent log entries."""
            if self.web_log_handler:
                logs = self.web_log_handler.get_recent_logs(limit)
                return {"logs": logs}
            return {"logs": []}

        @self.app.post("/api/tasks/pause")
        async def pause_tasks():
            """Pause task processing."""
            if not self.daemon or not self.daemon.task_manager:
                return JSONResponse({"error": "Task manager not available"}, status_code=503)

            self.daemon.task_manager.pause()
            self.daemon.settings.task_processing_paused = True
            self.daemon.settings.save()
            await self.broadcast_status()

            return {"status": "paused", "message": "Task processing paused"}

        @self.app.post("/api/tasks/resume")
        async def resume_tasks():
            """Resume task processing."""
            if not self.daemon or not self.daemon.task_manager:
                return JSONResponse({"error": "Task manager not available"}, status_code=503)

            self.daemon.task_manager.resume()
            self.daemon.settings.task_processing_paused = False
            self.daemon.settings.save()
            await self.broadcast_status()

            return {"status": "active", "message": "Task processing resumed"}

        @self.app.patch("/api/telescope/automated-scheduling")
        async def update_automated_scheduling(request: dict[str, bool]):
            """Toggle automated scheduling on/off."""
            if not self.daemon or not self.daemon.task_manager:
                return JSONResponse({"error": "Task manager not available"}, status_code=503)

            if not self.daemon.api_client:
                return JSONResponse({"error": "API client not available"}, status_code=503)

            enabled = request.get("enabled")
            if enabled is None:
                return JSONResponse({"error": "Missing 'enabled' field in request body"}, status_code=400)

            try:
                telescope_id = self.daemon.telescope_record["id"]
                success = self.daemon.api_client.update_telescope_automated_scheduling(telescope_id, enabled)

                if success:
                    self.daemon.task_manager._automated_scheduling = enabled
                    CITRASCOPE_LOGGER.info(f"Automated scheduling set to {'enabled' if enabled else 'disabled'}")
                    await self.broadcast_status()
                    return {"status": "success", "enabled": enabled}
                else:
                    return JSONResponse({"error": "Failed to update telescope on server"}, status_code=500)

            except Exception as e:
                CITRASCOPE_LOGGER.error(f"Error updating automated scheduling: {e}", exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.get("/api/adapter/filters")
        async def get_filters():
            """Get current filter configuration."""
            if not self.daemon or not self.daemon.hardware_adapter:
                return JSONResponse({"error": "Hardware adapter not available"}, status_code=503)

            if not self.daemon.hardware_adapter.supports_filter_management():
                return JSONResponse({"error": "Adapter does not support filter management"}, status_code=404)

            try:
                filter_config = self.daemon.hardware_adapter.get_filter_config()
                return {"filters": filter_config}
            except Exception as e:
                CITRASCOPE_LOGGER.error(f"Error getting filter config: {e}", exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.post("/api/adapter/filters/batch")
        async def update_filters_batch(updates: list[dict[str, Any]]):
            """Update multiple filters atomically with single disk write.

            Args:
                updates: Array of filter updates, each containing:
                    - filter_id (str): Filter ID
                    - focus_position (int, optional): Focus position in steps
                    - enabled (bool, optional): Whether filter is enabled

            Returns:
                {"success": true, "updated_count": N} on success
                {"error": "..."} on validation failure
            """
            if not self.daemon or not self.daemon.hardware_adapter:
                return JSONResponse({"error": "Hardware adapter not initialized"}, status_code=503)

            if not updates or not isinstance(updates, list):
                return JSONResponse({"error": "Updates must be a non-empty array"}, status_code=400)

            try:
                filter_config = self.daemon.hardware_adapter.filter_map

                # Phase 1: Validate ALL updates before applying ANY changes
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
                        return JSONResponse(
                            {"error": f"Invalid filter_id at index {idx}: {filter_id}"}, status_code=400
                        )

                    if filter_id_int not in filter_config:
                        return JSONResponse({"error": f"Filter ID {filter_id} not found"}, status_code=404)

                    validated_update = {"filter_id_int": filter_id_int}

                    # Validate focus_position if provided
                    if "focus_position" in update:
                        focus_position = update["focus_position"]
                        if not isinstance(focus_position, int):
                            return JSONResponse(
                                {"error": f"focus_position at index {idx} must be an integer"}, status_code=400
                            )
                        if focus_position < 0 or focus_position > 65535:
                            return JSONResponse(
                                {"error": f"focus_position at index {idx} must be between 0 and 65535"}, status_code=400
                            )
                        validated_update["focus_position"] = focus_position

                    # Validate enabled if provided
                    if "enabled" in update:
                        enabled = update["enabled"]
                        if not isinstance(enabled, bool):
                            return JSONResponse({"error": f"enabled at index {idx} must be a boolean"}, status_code=400)
                        validated_update["enabled"] = enabled

                    validated_updates.append(validated_update)

                # Validate at least one filter remains enabled
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

                # Phase 2: Apply all validated updates
                for validated in validated_updates:
                    filter_id_int = validated["filter_id_int"]

                    if "focus_position" in validated:
                        if not self.daemon.hardware_adapter.update_filter_focus(
                            str(filter_id_int), validated["focus_position"]
                        ):
                            return JSONResponse(
                                {"error": f"Failed to update filter {filter_id_int} focus"}, status_code=500
                            )

                    if "enabled" in validated:
                        if not self.daemon.hardware_adapter.update_filter_enabled(
                            str(filter_id_int), validated["enabled"]
                        ):
                            return JSONResponse(
                                {"error": f"Failed to update filter {filter_id_int} enabled state"}, status_code=500
                            )

                # Phase 3: Save once after all updates
                self.daemon._save_filter_config()

                return {"success": True, "updated_count": len(validated_updates)}

            except Exception as e:
                CITRASCOPE_LOGGER.error(f"Error in batch filter update: {e}", exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.post("/api/adapter/filters/sync")
        async def sync_filters_to_backend():
            """Explicitly sync filter configuration to backend API.

            Call this after batch filter updates to sync enabled filters to backend.
            """
            if not self.daemon or not self.daemon.hardware_adapter:
                return JSONResponse({"error": "Hardware adapter not initialized"}, status_code=503)

            try:
                self.daemon._sync_filters_to_backend()
                return {"success": True, "message": "Filters synced to backend"}
            except Exception as e:
                CITRASCOPE_LOGGER.error(f"Error syncing filters to backend: {e}", exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.post("/api/adapter/autofocus")
        async def trigger_autofocus():
            """Request autofocus to run between tasks."""
            if not self.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)

            if not self.daemon.hardware_adapter or not self.daemon.hardware_adapter.supports_filter_management():
                return JSONResponse({"error": "Filter management not supported"}, status_code=404)

            try:
                success, error = self.daemon.trigger_autofocus()
                if success:
                    return {"success": True, "message": "Autofocus queued - will run between tasks"}
                else:
                    return JSONResponse({"error": error}, status_code=500)
            except Exception as e:
                CITRASCOPE_LOGGER.error(f"Error queueing autofocus: {e}", exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.post("/api/adapter/autofocus/cancel")
        async def cancel_autofocus():
            """Cancel pending autofocus request."""
            if not self.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)

            try:
                was_cancelled = self.daemon.cancel_autofocus()
                return {"success": was_cancelled}
            except Exception as e:
                CITRASCOPE_LOGGER.error(f"Error cancelling autofocus: {e}", exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.get("/api/adapter/autofocus/presets")
        async def get_autofocus_presets():
            """Return available autofocus target star presets."""
            presets = [{"key": key, **preset} for key, preset in AUTOFOCUS_TARGET_PRESETS.items()]
            return {"presets": presets}

        @self.app.post("/api/adapter/alignment")
        async def trigger_alignment():
            """Request plate-solve alignment to run between tasks."""
            if not self.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)
            if not self.daemon.task_manager:
                return JSONResponse({"error": "Task manager not available"}, status_code=503)

            try:
                self.daemon.task_manager.alignment_manager.request()
                return {"success": True, "message": "Alignment queued — will run between tasks"}
            except Exception as e:
                CITRASCOPE_LOGGER.error(f"Error queueing alignment: {e}", exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.post("/api/adapter/alignment/cancel")
        async def cancel_alignment():
            """Cancel pending alignment request."""
            if not self.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)
            if not self.daemon.task_manager:
                return JSONResponse({"error": "Task manager not available"}, status_code=503)

            try:
                was_cancelled = self.daemon.task_manager.alignment_manager.cancel()
                return {"success": was_cancelled}
            except Exception as e:
                CITRASCOPE_LOGGER.error(f"Error cancelling alignment: {e}", exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.post("/api/adapter/sync")
        async def manual_sync(request: dict[str, Any]):
            """Manually sync the mount to given RA/Dec coordinates."""
            if not self.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)
            if not self.daemon.hardware_adapter:
                return JSONResponse({"error": "Hardware adapter not available"}, status_code=503)

            ra = request.get("ra")
            dec = request.get("dec")
            if ra is None or dec is None:
                return JSONResponse({"error": "Both 'ra' and 'dec' are required (degrees)"}, status_code=400)

            try:
                ra_f = float(ra)
                dec_f = float(dec)
            except (TypeError, ValueError):
                return JSONResponse({"error": "RA and Dec must be numeric (degrees)"}, status_code=400)

            mount = getattr(self.daemon.hardware_adapter, "mount", None)
            if not mount:
                return JSONResponse({"error": "No mount connected"}, status_code=404)

            try:
                success = mount.sync_to_radec(ra_f, dec_f)
                if success:
                    return {"success": True, "message": f"Mount synced to RA={ra_f:.4f}°, Dec={dec_f:.4f}°"}
                else:
                    return JSONResponse({"error": "Mount sync returned failure"}, status_code=500)
            except NotImplementedError:
                return JSONResponse({"error": "Mount does not support sync"}, status_code=404)
            except Exception as e:
                CITRASCOPE_LOGGER.error(f"Manual sync failed: {e}", exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.post("/api/mount/home")
        async def trigger_mount_home():
            """Request mount homing — queued to run when imaging is idle."""
            if not self.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)
            if not self.daemon.task_manager:
                return JSONResponse({"error": "Task manager not available"}, status_code=503)

            try:
                success = self.daemon.task_manager.homing_manager.request()
                if success:
                    return {"success": True, "message": "Mount homing queued — will run when imaging is idle"}
                else:
                    return JSONResponse({"error": "Homing already in progress"}, status_code=409)
            except Exception as e:
                CITRASCOPE_LOGGER.error(f"Error requesting mount homing: {e}", exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.get("/api/mount/limits")
        async def get_mount_limits():
            """Get the mount's altitude limits."""
            if not self.daemon or not self.daemon.hardware_adapter:
                return JSONResponse({"error": "Hardware not available"}, status_code=503)
            try:
                h_limit, o_limit = self.daemon.hardware_adapter.get_mount_limits()
                return {"horizon_limit": h_limit, "overhead_limit": o_limit}
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.get("/api/safety/status")
        async def get_safety_status():
            """Return status of all safety checks."""
            if not self.daemon or not self.daemon.safety_monitor:
                return {"checks": [], "watchdog_alive": False, "watchdog_last_heartbeat": 0}
            try:
                return self.daemon.safety_monitor.get_status()
            except Exception as e:
                CITRASCOPE_LOGGER.error(f"Error getting safety status: {e}", exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.post("/api/mount/unwind")
        async def trigger_cable_unwind():
            """Manually trigger cable unwind in a background thread."""
            if not self.daemon or not self.daemon.safety_monitor:
                return JSONResponse({"error": "Safety monitor not available"}, status_code=503)

            import threading

            from citrascope.safety.cable_wrap_check import CableWrapCheck

            chk = self.daemon.safety_monitor.get_check("cable_wrap")
            if not isinstance(chk, CableWrapCheck):
                return JSONResponse({"error": "No cable wrap check configured"}, status_code=404)
            if chk.is_unwinding:
                return JSONResponse({"error": "Unwind already in progress"}, status_code=409)
            threading.Thread(target=chk.execute_action, daemon=True, name="cable-unwind").start()
            return JSONResponse({"success": True, "message": "Cable unwind started"}, status_code=202)

        @self.app.post("/api/safety/cable-wrap/reset")
        async def reset_cable_wrap():
            """Reset cable wrap counter to zero (operator confirms cables are straight)."""
            if not self.daemon or not self.daemon.safety_monitor:
                return JSONResponse({"error": "Safety monitor not available"}, status_code=503)

            from citrascope.safety.cable_wrap_check import CableWrapCheck

            chk = self.daemon.safety_monitor.get_check("cable_wrap")
            if not isinstance(chk, CableWrapCheck):
                return JSONResponse({"error": "No cable wrap check configured"}, status_code=404)
            if chk.is_unwinding:
                return JSONResponse({"error": "Cannot reset during unwind"}, status_code=409)
            chk.reset()
            CITRASCOPE_LOGGER.info("Cable wrap counter reset by operator")
            return {"success": True, "message": "Cable wrap counter reset to 0°"}

        @self.app.post("/api/emergency-stop")
        async def emergency_stop():
            """Stop mount, pause task processing, and drain imaging queue."""
            if not self.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)

            import threading

            # Activate safety check — watchdog will enforce continuously
            if self.daemon.safety_monitor:
                self.daemon.safety_monitor.activate_operator_stop()

            cleared = 0
            tm = self.daemon.task_manager
            if tm:
                tm.pause()
                cleared = tm.clear_pending_tasks()
            if self.daemon.settings:
                self.daemon.settings.task_processing_paused = True
                self.daemon.settings.save()

            # Immediate mount halt in background thread (serial I/O can't
            # run on the async event loop).  The watchdog provides ongoing
            # enforcement at 1 Hz; this gives sub-second first response.
            daemon = self.daemon

            def _halt_mount():
                mount = getattr(daemon.hardware_adapter, "mount", None) if daemon.hardware_adapter else None
                if not mount:
                    return
                try:
                    mount.abort_slew()
                    mount.stop_tracking()
                except Exception:
                    CITRASCOPE_LOGGER.error("Error halting mount during emergency stop", exc_info=True)

            threading.Thread(target=_halt_mount, daemon=True, name="emergency-stop").start()

            CITRASCOPE_LOGGER.warning(
                "EMERGENCY STOP by operator — processing paused, %d imaging tasks cleared, mount halt issued",
                cleared,
            )
            return JSONResponse(
                {
                    "success": True,
                    "message": f"Emergency stop: mount halt issued, {cleared} queued task(s) cleared",
                },
                status_code=202,
            )

        @self.app.post("/api/safety/operator-stop/clear")
        async def clear_operator_stop():
            """Clear the operator stop — allows motion to resume.

            Also reverses the pause that emergency_stop applied so
            task processing can pick up where it left off.
            """
            if not self.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)
            if not self.daemon.safety_monitor:
                return JSONResponse({"error": "Safety monitor not available"}, status_code=503)
            self.daemon.safety_monitor.clear_operator_stop()

            tm = self.daemon.task_manager
            if tm:
                tm.resume()
            if self.daemon.settings:
                self.daemon.settings.task_processing_paused = False
                self.daemon.settings.save()

            CITRASCOPE_LOGGER.info("Operator stop cleared via web UI — task processing resumed")
            return {"success": True, "message": "Operator stop cleared — motion may resume"}

        @self.app.post("/api/mount/limits")
        async def set_mount_limits(request: dict[str, Any]):
            """Set the mount's altitude limits."""
            if not self.daemon or not self.daemon.hardware_adapter:
                return JSONResponse({"error": "Hardware not available"}, status_code=503)
            adapter = self.daemon.hardware_adapter
            results: dict[str, Any] = {}
            try:
                if "horizon_limit" in request:
                    results["horizon_ok"] = adapter.set_mount_horizon_limit(int(request["horizon_limit"]))
                if "overhead_limit" in request:
                    results["overhead_ok"] = adapter.set_mount_overhead_limit(int(request["overhead_limit"]))
                return {"success": True, **results}
            except Exception as e:
                CITRASCOPE_LOGGER.error("Error setting mount limits: %s", e, exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.post("/api/camera/capture")
        async def camera_capture(request: dict[str, Any]):
            """Trigger a test camera capture."""
            if not self.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)

            if not self.daemon.hardware_adapter:
                return JSONResponse({"error": "Hardware adapter not available"}, status_code=503)

            # Check if adapter supports direct camera control
            if not self.daemon.hardware_adapter.supports_direct_camera_control():
                return JSONResponse(
                    {"error": "Hardware adapter does not support direct camera control"}, status_code=400
                )

            try:
                duration = request.get("duration", 0.1)

                # Validate exposure duration
                if duration <= 0:
                    return JSONResponse({"error": "Exposure duration must be positive"}, status_code=400)
                if duration > 300:
                    return JSONResponse({"error": "Exposure duration must be 300 seconds or less"}, status_code=400)

                CITRASCOPE_LOGGER.info(f"Test capture requested: {duration}s exposure")

                # Take exposure using hardware adapter
                filepath = self.daemon.hardware_adapter.expose_camera(
                    exposure_time=duration, gain=None, offset=None, count=1
                )

                # Get file info
                file_path = Path(filepath)
                if not file_path.exists():
                    return JSONResponse({"error": "Capture completed but file not found"}, status_code=500)

                filename = file_path.name
                file_format = file_path.suffix.upper().lstrip(".")

                CITRASCOPE_LOGGER.info(f"Test capture complete: {filename}")

                return {"success": True, "filename": filename, "filepath": str(file_path), "format": file_format}

            except Exception as e:
                CITRASCOPE_LOGGER.error(f"Error during test capture: {e}", exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await self.connection_manager.connect(websocket)
            try:
                # Send initial status
                if self.daemon:
                    self._update_status_from_daemon()
                await websocket.send_json({"type": "status", "data": self.status.dict()})

                # Keep connection alive and listen for client messages
                while True:
                    data = await websocket.receive_text()
                    # Handle client requests if needed
                    await websocket.send_json({"type": "pong", "data": data})

            except WebSocketDisconnect:
                self.connection_manager.disconnect(websocket)
            except Exception as e:
                CITRASCOPE_LOGGER.error(f"WebSocket error: {e}")
                self.connection_manager.disconnect(websocket)

    def _update_status_from_daemon(self):
        """Update status from daemon state."""
        if not self.daemon:
            return

        try:
            self.status.hardware_adapter = self.daemon.settings.hardware_adapter

            if hasattr(self.daemon, "hardware_adapter") and self.daemon.hardware_adapter:
                adapter = self.daemon.hardware_adapter
                mount = getattr(adapter, "mount", None)

                # Check telescope connection status
                try:
                    self.status.telescope_connected = adapter.is_telescope_connected()
                    if self.status.telescope_connected:
                        snap = mount.cached_state if mount is not None else None
                        if snap is not None:
                            self.status.telescope_ra = snap.ra_deg
                            self.status.telescope_dec = snap.dec_deg
                            self.status.telescope_az = snap.az_deg
                            self.status.telescope_alt = snap.alt_deg
                        else:
                            ra, dec = adapter.get_telescope_direction()
                            self.status.telescope_ra = ra
                            self.status.telescope_dec = dec
                except Exception:
                    self.status.telescope_connected = False

                # Check camera connection status
                try:
                    self.status.camera_connected = adapter.is_camera_connected()
                except Exception:
                    self.status.camera_connected = False

                # Check adapter capabilities
                try:
                    self.status.supports_direct_camera_control = adapter.supports_direct_camera_control()
                except Exception:
                    self.status.supports_direct_camera_control = False

                self.status.supports_autofocus = adapter.supports_autofocus()

                has_camera = getattr(adapter, "camera", None) is not None
                self.status.supports_alignment = has_camera and mount is not None

                if mount is not None and mount.cached_state is not None:
                    self.status.supports_manual_sync = mount.cached_mount_info.get("supports_sync", False)
                    self.status.mount_at_home = mount.cached_state.is_at_home
                    h_limit, o_limit = mount.cached_limits
                    self.status.mount_horizon_limit = h_limit
                    self.status.mount_overhead_limit = o_limit

            if hasattr(self.daemon, "task_manager") and self.daemon.task_manager:
                task_manager = self.daemon.task_manager
                self.status.current_task = task_manager.current_task_id
                self.status.mount_homing = (
                    task_manager.homing_manager.is_running() or task_manager.homing_manager.is_requested()
                )
                self.status.autofocus_requested = task_manager.autofocus_manager.is_requested()
                self.status.autofocus_running = task_manager.autofocus_manager.is_running()
                self.status.autofocus_progress = task_manager.autofocus_manager.progress
                self.status.alignment_requested = task_manager.alignment_manager.is_requested()
                self.status.alignment_running = task_manager.alignment_manager.is_running()
                self.status.alignment_progress = task_manager.alignment_manager.progress
                with task_manager.heap_lock:
                    self.status.tasks_pending = len(task_manager.task_heap)

            # Get autofocus timing information
            if self.daemon.settings:
                settings = self.daemon.settings
                self.status.last_autofocus_timestamp = settings.last_autofocus_timestamp
                self.status.last_alignment_timestamp = settings.last_alignment_timestamp

                # Calculate next autofocus time if scheduled is enabled
                if settings.scheduled_autofocus_enabled:
                    last_ts = settings.last_autofocus_timestamp
                    interval_minutes = settings.autofocus_interval_minutes
                    if last_ts is not None:
                        elapsed_minutes = (int(time.time()) - last_ts) / 60
                        remaining = max(0, interval_minutes - elapsed_minutes)
                        self.status.next_autofocus_minutes = int(remaining)
                    else:
                        # Never run - will trigger immediately
                        self.status.next_autofocus_minutes = 0
                else:
                    self.status.next_autofocus_minutes = None

            # Get time sync status from time monitor
            if hasattr(self.daemon, "time_monitor") and self.daemon.time_monitor:
                health = self.daemon.time_monitor.get_current_health()
                self.status.time_health = health.to_dict() if health else None
            else:
                # Time monitoring not initialized yet
                self.status.time_health = None

            # Get GPS location status from location service
            # Use allow_blocking=False to prevent blocking the async event loop
            if (
                hasattr(self.daemon, "location_service")
                and self.daemon.location_service
                and self.daemon.location_service.gps_monitor
            ):
                gps_fix = self.daemon.location_service.gps_monitor.get_current_fix(allow_blocking=False)
                if gps_fix:
                    self.status.gps_location = {
                        "latitude": gps_fix.latitude,
                        "longitude": gps_fix.longitude,
                        "altitude": gps_fix.altitude,
                        "fix_mode": gps_fix.fix_mode,
                        "satellites": gps_fix.satellites,
                        "is_strong": gps_fix.is_strong_fix,
                        "eph": gps_fix.eph,
                        "sep": gps_fix.sep,
                        "source": "gps",
                    }
                else:
                    self.status.gps_location = None
            else:
                self.status.gps_location = None

            # Get ground station information from daemon (available after API validation)
            if hasattr(self.daemon, "ground_station") and self.daemon.ground_station:
                gs_record = self.daemon.ground_station
                gs_id = gs_record.get("id")
                gs_name = gs_record.get("name", "Unknown")

                # Build the URL based on the API host (dev vs prod)
                api_host = self.daemon.settings.host
                base_url = DEV_APP_URL if "dev." in api_host else PROD_APP_URL

                self.status.ground_station_id = gs_id
                self.status.ground_station_name = gs_name
                self.status.ground_station_url = f"{base_url}/ground-stations/{gs_id}" if gs_id else None

            # Update task processing state
            if hasattr(self.daemon, "task_manager") and self.daemon.task_manager:
                self.status.processing_active = self.daemon.task_manager.is_processing_active()
                self.status.automated_scheduling = self.daemon.task_manager._automated_scheduling or False

            # Check for missing dependencies from adapter
            self.status.missing_dependencies = []
            if hasattr(self.daemon, "hardware_adapter") and self.daemon.hardware_adapter:
                if hasattr(self.daemon.hardware_adapter, "get_missing_dependencies"):
                    try:
                        self.status.missing_dependencies = self.daemon.hardware_adapter.get_missing_dependencies()
                    except Exception as e:
                        CITRASCOPE_LOGGER.debug(f"Could not check missing dependencies: {e}")

            # Get list of active processors
            if hasattr(self.daemon, "processor_registry") and self.daemon.processor_registry:
                self.status.active_processors = [p.name for p in self.daemon.processor_registry.processors]
            else:
                self.status.active_processors = []

            # Get tasks by pipeline stage
            if hasattr(self.daemon, "task_manager") and self.daemon.task_manager:
                self.status.tasks_by_stage = self.daemon.task_manager.get_tasks_by_stage()
            else:
                self.status.tasks_by_stage = None

            # Collect lifetime pipeline stats from queues, processor registry, and task manager
            if hasattr(self.daemon, "task_manager") and self.daemon.task_manager:
                tm = self.daemon.task_manager
                self.status.pipeline_stats = {
                    "imaging": tm.imaging_queue.get_stats(),
                    "processing": tm.processing_queue.get_stats(),
                    "uploading": tm.upload_queue.get_stats(),
                    "tasks": tm.get_task_stats(),
                }
            else:
                self.status.pipeline_stats = None
            if hasattr(self.daemon, "processor_registry") and self.daemon.processor_registry:
                if self.status.pipeline_stats is None:
                    self.status.pipeline_stats = {}
                self.status.pipeline_stats["processors"] = self.daemon.processor_registry.get_processor_stats()

            # Safety monitor status
            if hasattr(self.daemon, "safety_monitor") and self.daemon.safety_monitor:
                try:
                    self.status.safety_status = self.daemon.safety_monitor.get_status()
                except Exception:
                    self.status.safety_status = None
            else:
                self.status.safety_status = None

            self.status.last_update = datetime.now().isoformat()

        except Exception as e:
            CITRASCOPE_LOGGER.error(f"Error updating status: {e}")

    async def broadcast_status(self):
        """Broadcast current status to all connected clients."""
        if self.daemon:
            self._update_status_from_daemon()
        await self.connection_manager.broadcast({"type": "status", "data": self.status.dict()})

    async def broadcast_tasks(self):
        """Broadcast current task queue to all connected clients."""
        if not self.daemon or not hasattr(self.daemon, "task_manager") or self.daemon.task_manager is None:
            return

        task_manager = self.daemon.task_manager
        tasks = []

        with task_manager.heap_lock:
            for start_time, stop_time, task_id, task in task_manager.task_heap:
                tasks.append(
                    {
                        "id": task_id,
                        "start_time": datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat(),
                        "stop_time": (
                            datetime.fromtimestamp(stop_time, tz=timezone.utc).isoformat() if stop_time else None
                        ),
                        "status": task.status,
                        "target": getattr(task, "satelliteName", getattr(task, "target", "unknown")),
                    }
                )

        await self.connection_manager.broadcast({"type": "tasks", "data": tasks})

    async def broadcast_log(self, log_entry: dict):
        """Broadcast log entry to all connected clients."""
        await self.connection_manager.broadcast({"type": "log", "data": log_entry})
