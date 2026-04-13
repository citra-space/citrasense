"""FastAPI web application for CitraScope monitoring and configuration."""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
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
from citrascope.hardware.devices.abstract_hardware_device import AbstractHardwareDevice
from citrascope.location.twilight import compute_twilight
from citrascope.logging import CITRASCOPE_LOGGER
from citrascope.settings.directory_manager import DirectoryManager

# Standard astronomical filter names for the editable filter name dropdown.
# Mirrors the canonical names from the Citra API's filter library so that
# task assignment matching works without typos.
FILTER_NAME_OPTIONS = [
    {"group": "Broadband", "names": ["Luminance", "Red", "Green", "Blue", "Clear"]},
    {"group": "Johnson-Cousins", "names": ["U", "B", "V", "R", "I"]},
    {"group": "Sloan", "names": ["sloan_u", "sloan_g", "sloan_r", "sloan_i", "sloan_z"]},
    {"group": "Narrowband", "names": ["Ha", "Hb", "OIII", "SII"]},
]


def _task_to_dict(task: Any) -> dict:
    """Format a Task object into the dict shape the web layer expects."""
    return {
        "id": task.id,
        "start_time": task.taskStart,
        "stop_time": task.taskStop or None,
        "status": task.status,
        "target": task.satelliteName,
        "filter": task.assigned_filter_name,
    }


def _resolve_autofocus_target_name(settings: Any) -> str:
    """Return a human-readable name for the active autofocus target."""
    preset_key = settings.autofocus_target_preset or "mirach"

    if preset_key == "current":
        return "Current position"

    if preset_key == "custom":
        ra = settings.autofocus_target_custom_ra
        dec = settings.autofocus_target_custom_dec
        if ra is not None and dec is not None:
            return f"Custom (RA={ra:.4f}, Dec={dec:.4f})"
        return "Mirach (custom missing coords)"

    preset = AUTOFOCUS_TARGET_PRESETS.get(preset_key)
    if not preset:
        return f"Mirach (unknown preset '{preset_key}')"

    return f"{preset['name']} ({preset['designation']})"


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
    ground_station_latitude: float | None = None
    ground_station_longitude: float | None = None
    ground_station_altitude: float | None = None
    location_source: str | None = None  # "gps" or "ground_station"
    location_latitude: float | None = None
    location_longitude: float | None = None
    location_altitude: float | None = None
    autofocus_requested: bool = False
    autofocus_running: bool = False
    autofocus_progress: str = ""
    autofocus_points: list[dict[str, int | float | str]] = []
    autofocus_filter_results: list[dict[str, str | int | float]] = []
    autofocus_last_result: str = ""
    autofocus_target_name: str = ""
    last_autofocus_timestamp: int | None = None
    next_autofocus_minutes: int | None = None
    hfr_history: list[dict[str, int | float | str]] = []
    last_hfr_median: float | None = None
    hfr_baseline: float | None = None
    hfr_increase_percent: int = 30
    hfr_refocus_enabled: bool = False
    hfr_sample_window: int = 5
    time_health: dict[str, Any] | None = None
    gps_location: dict[str, Any] | None = None
    last_update: str = ""
    missing_dependencies: list[dict[str, str]] = []  # List of {device, packages, install_cmd}
    active_processors: list[str] = []  # Names of enabled image processors
    tasks_by_stage: dict[str, list[dict]] | None = None  # Tasks in each pipeline stage
    pipeline_stats: dict[str, Any] | None = None  # Lifetime counters for queues, processors, and tasks
    supports_alignment: bool = False
    supports_autofocus: bool = False
    supports_hardware_safety_monitor: bool = False
    supports_manual_sync: bool = False
    mount_at_home: bool = False
    mount_homing: bool = False
    mount_horizon_limit: int | None = None
    mount_overhead_limit: int | None = None
    alignment_requested: bool = False
    alignment_running: bool = False
    alignment_progress: str = ""
    last_alignment_timestamp: int | None = None
    pointing_model: dict[str, Any] | None = None
    pointing_calibration_running: bool = False
    pointing_calibration_progress: dict[str, int] | None = None
    fov_short_deg: float | None = None
    camera_temperature: float | None = None
    current_filter_position: int | None = None
    current_filter_name: str | None = None
    focuser_connected: bool = False
    focuser_position: int | None = None
    focuser_max_position: int | None = None
    focuser_temperature: float | None = None
    focuser_moving: bool = False
    mount_tracking: bool = False
    mount_slewing: bool = False
    supports_direct_mount_control: bool = False
    safety_status: dict[str, Any] | None = None
    elset_health: dict[str, Any] | None = None
    latest_task_image_url: str | None = None
    calibration_status: dict[str, Any] | None = None
    config_health: dict[str, Any] | None = None
    status_collection_ms: float | None = None
    status_collection_breakdown: dict[str, float] | None = None
    # System busy guard — true when automated hardware operations are in progress
    system_busy: bool = False
    system_busy_reason: str = ""
    # Observing session / self-tasking
    observing_session_enabled: bool = False
    self_tasking_enabled: bool = False
    observing_session_state: str = "daytime"
    session_activity: str | None = None
    observing_session_threshold: float = -12.0
    sun_altitude: float | None = None
    dark_window_start: str | None = None
    dark_window_end: str | None = None
    last_batch_request: float | None = None
    last_batch_created: int | None = None
    next_request_seconds: float | None = None


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
            images_dir = daemon.settings.directories.images_dir
            if images_dir.exists():
                self.app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")

        # Initialize Jinja2 templates with a cache-buster so browsers pick up
        # new static files after each daemon restart.
        templates_dir = Path(__file__).parent / "templates"
        self.templates = Jinja2Templates(directory=str(templates_dir))
        self._cache_bust = str(int(time.time()))
        self.templates.env.globals["cache_bust"] = self._cache_bust
        self.templates.env.globals["default_data_dir"] = str(DirectoryManager.default_data_dir())
        self.templates.env.globals["default_log_dir"] = str(DirectoryManager.default_log_dir())

        # Register routes
        self._setup_routes()

    def set_daemon(self, daemon):
        """Set the daemon instance after initialization."""
        self.daemon = daemon

    def _require_system_idle(self) -> JSONResponse | None:
        """Return a 409 response if the system is busy with automated operations, else None."""
        if self.status.system_busy:
            return JSONResponse(
                {"error": f"System busy ({self.status.system_busy_reason})", "system_busy": True},
                status_code=409,
            )
        return None

    def _setup_routes(self):
        """Setup all API routes."""

        @self.app.get("/", response_class=HTMLResponse)
        async def root(request: Request):
            """Serve the main dashboard page."""
            return self.templates.TemplateResponse(request, "dashboard.html")

        @self.app.get("/api/status")
        async def get_status():
            """Get current system status."""
            if self.daemon:
                self._update_status_from_daemon()
            return self.status

        @self.app.get("/api/task-preview/latest")
        async def get_latest_task_preview():
            """Serve the latest annotated task image."""
            ann_path = getattr(self.daemon, "latest_annotated_image_path", None)
            if not ann_path or not Path(ann_path).exists():
                return JSONResponse({"error": "No preview available"}, status_code=404)
            mime = "image/jpeg" if Path(ann_path).suffix.lower() in (".jpg", ".jpeg") else "image/png"
            return FileResponse(ann_path, media_type=mime)

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
            log_file_path = str(settings.directories.current_log_path()) if settings.file_logging_enabled else None

            # Get images directory path
            images_dir_path = str(settings.directories.images_dir)

            processing_dir_path = str(settings.directories.processing_dir)

            return {
                **settings.to_dict(),
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
            """Get CitraScope version and install metadata."""
            from citrascope.version import get_version_info

            return get_version_info()

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
                # Invalid adapter name
                return JSONResponse({"error": str(e)}, status_code=404)
            except Exception as e:
                CITRASCOPE_LOGGER.error(f"Error getting schema for {adapter_name}: {e}", exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.post("/api/hardware/reconnect")
        async def reconnect_hardware():
            """Retry hardware connection using current in-memory settings."""
            if not self.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)
            if not self.daemon.settings.is_configured():
                return JSONResponse(
                    {"error": "Configuration incomplete — configure hardware adapter first"},
                    status_code=400,
                )

            success, error = await asyncio.to_thread(self.daemon.retry_connection)

            if success:
                return {"status": "success", "message": "Hardware reconnected successfully"}
            else:
                return JSONResponse(
                    {"status": "error", "message": f"Reconnect failed: {error}", "error": error},
                    status_code=500,
                )

        @self.app.post("/api/hardware/scan")
        async def scan_hardware(body: dict[str, Any]):
            """Clear hardware probe caches and return a fresh adapter schema.

            The "Scan Hardware" button in the UI calls this so operators
            can explicitly re-enumerate USB devices, serial ports, etc.
            """
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

                self.daemon.settings.update_and_save(config)

                # Trigger hot-reload in a worker thread so slow adapter I/O
                # doesn't block the event loop.  GIL-holding native calls
                # (e.g. dlopen) are already subprocess-isolated inside probes.
                success, error = await asyncio.to_thread(self.daemon.reload_configuration)

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
            return [_task_to_dict(t) for t in task_manager.get_tasks_snapshot(exclude_active=True)]

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
                    self.daemon.task_manager.automated_scheduling = enabled
                    CITRASCOPE_LOGGER.info(f"Automated scheduling set to {'enabled' if enabled else 'disabled'}")
                    await self.broadcast_status()
                    return {"status": "success", "enabled": enabled}
                else:
                    return JSONResponse({"error": "Failed to update telescope on server"}, status_code=500)

            except Exception as e:
                CITRASCOPE_LOGGER.error(f"Error updating automated scheduling: {e}", exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.patch("/api/observing-session")
        async def toggle_observing_session(request: dict[str, bool]):
            """Toggle observing session on/off."""
            if not self.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)

            enabled = request.get("enabled")
            if enabled is None:
                return JSONResponse({"error": "Missing 'enabled' field in request body"}, status_code=400)

            self.daemon.settings.observing_session_enabled = enabled
            self.daemon.settings.save()
            CITRASCOPE_LOGGER.info(f"Observing session set to {'enabled' if enabled else 'disabled'}")
            await self.broadcast_status()
            return {"status": "success", "enabled": enabled}

        @self.app.patch("/api/self-tasking")
        async def toggle_self_tasking(request: dict[str, bool]):
            """Toggle self-tasking on/off.

            When enabling, also enables Observing Session, Scheduling
            (server-side), and Processing (local) so the autonomous
            pipeline is fully active.
            """
            if not self.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)

            enabled = request.get("enabled")
            if enabled is None:
                return JSONResponse({"error": "Missing 'enabled' field in request body"}, status_code=400)

            self.daemon.settings.self_tasking_enabled = enabled
            self.daemon.settings.save()

            if enabled:
                # Ensure observing session is active (prerequisite)
                if not self.daemon.settings.observing_session_enabled:
                    self.daemon.settings.observing_session_enabled = True
                    self.daemon.settings.save()
                    CITRASCOPE_LOGGER.info("Self-tasking: auto-enabled observing session")

            if enabled and self.daemon.task_manager:
                # Ensure processing is active
                if not self.daemon.task_manager.is_processing_active():
                    self.daemon.task_manager.resume()
                    self.daemon.settings.task_processing_paused = False
                    self.daemon.settings.save()
                    CITRASCOPE_LOGGER.info("Self-tasking: auto-enabled processing")

                # Ensure scheduling is active on the server
                if not self.daemon.task_manager.automated_scheduling:
                    try:
                        telescope_id = self.daemon.telescope_record["id"]
                        success = self.daemon.api_client.update_telescope_automated_scheduling(telescope_id, True)
                        if success:
                            self.daemon.task_manager.automated_scheduling = True
                            CITRASCOPE_LOGGER.info("Self-tasking: auto-enabled scheduling")
                        else:
                            CITRASCOPE_LOGGER.warning("Self-tasking: failed to enable scheduling on server")
                    except Exception as e:
                        CITRASCOPE_LOGGER.warning(f"Self-tasking: could not enable scheduling: {e}")

            CITRASCOPE_LOGGER.info(f"Self-tasking set to {'enabled' if enabled else 'disabled'}")
            await self.broadcast_status()
            return {"status": "success", "enabled": enabled}

        @self.app.post("/api/self-tasking/request-now")
        async def request_batch_now():
            """Fire a single batch collection request, bypassing session-state and timer gating."""
            if not self.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)

            gs = getattr(self.daemon, "ground_station", None)
            tr = getattr(self.daemon, "telescope_record", None)
            if not gs or not tr:
                return JSONResponse({"error": "Ground station or telescope not configured"}, status_code=503)

            settings = self.daemon.settings
            ground_station_id = gs["id"]
            sensor_id = tr["id"]

            group_ids = settings.self_tasking_satellite_group_ids or None
            exclude_types = settings.self_tasking_exclude_object_types or None
            orbit_regimes = settings.self_tasking_include_orbit_regimes or None
            collection_type = settings.self_tasking_collection_type or "Track"

            from datetime import timedelta, timezone

            now = datetime.now(timezone.utc)
            window_start = now.strftime("%Y-%m-%dT%H:%M:%SZ")
            window_stop = (now + timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")

            CITRASCOPE_LOGGER.info(
                "Manual batch request: type=%s, window %s → %s, gs=%s, sensor=%s",
                collection_type,
                window_start,
                window_stop,
                ground_station_id,
                sensor_id,
            )

            try:
                result = await asyncio.to_thread(
                    self.daemon.api_client.create_batch_collection_requests,
                    window_start=window_start,
                    window_stop=window_stop,
                    ground_station_id=ground_station_id,
                    sensor_id=sensor_id,
                    discover_visible=not bool(group_ids),
                    satellite_group_ids=group_ids,
                    request_type=collection_type,
                    exclude_types=exclude_types,
                    include_orbit_regimes=orbit_regimes,
                )
            except Exception as e:
                CITRASCOPE_LOGGER.error("Manual batch request failed", exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

            if result is None:
                return JSONResponse({"error": "API request failed"}, status_code=502)

            created = result.get("created", 0)
            CITRASCOPE_LOGGER.info("Manual batch request succeeded (created=%s)", created)
            return {"status": "ok", "created": created}

        @self.app.get("/api/adapter/filters")
        async def get_filters():
            """Get current filter configuration."""
            if not self.daemon or not self.daemon.hardware_adapter:
                return JSONResponse({"error": "Hardware adapter not available"}, status_code=503)

            if not self.daemon.hardware_adapter.supports_filter_management():
                return JSONResponse({"error": "Adapter does not support filter management"}, status_code=404)

            try:
                filter_config = self.daemon.hardware_adapter.get_filter_config()
                names_editable = self.daemon.hardware_adapter.supports_filter_rename()
                response: dict = {"filters": filter_config, "names_editable": names_editable}
                if names_editable:
                    response["filter_name_options"] = FILTER_NAME_OPTIONS
                return response
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

                    validated_update: dict[str, int | str | bool | None] = {"filter_id_int": filter_id_int}

                    # Validate focus_position if provided (null clears it)
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
                                {"error": f"focus_position at index {idx} must be between 0 and 65535"}, status_code=400
                            )
                        else:
                            validated_update["focus_position"] = focus_position

                    # Validate enabled if provided
                    if "enabled" in update:
                        enabled = update["enabled"]
                        if not isinstance(enabled, bool):
                            return JSONResponse({"error": f"enabled at index {idx} must be a boolean"}, status_code=400)
                        validated_update["enabled"] = enabled

                    # Validate name if provided
                    if "name" in update:
                        name = update["name"]
                        if not isinstance(name, str) or not name.strip():
                            return JSONResponse(
                                {"error": f"name at index {idx} must be a non-empty string"}, status_code=400
                            )
                        validated_update["name"] = name.strip()

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

                    if "name" in validated and self.daemon.hardware_adapter.supports_filter_rename():
                        if not self.daemon.hardware_adapter.update_filter_name(str(filter_id_int), validated["name"]):
                            return JSONResponse(
                                {"error": f"Failed to update filter {filter_id_int} name"}, status_code=500
                            )

                # Phase 3: Save once after all updates
                self.daemon.save_filter_config()

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
                self.daemon.sync_filters_to_backend()
                return {"success": True, "message": "Filters synced to backend"}
            except Exception as e:
                CITRASCOPE_LOGGER.error(f"Error syncing filters to backend: {e}", exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.post("/api/adapter/filter/set")
        async def set_filter_position(body: dict[str, Any]):
            """Command the filter wheel to move to a specific position."""
            if busy := self._require_system_idle():
                return busy
            if not self.daemon or not self.daemon.hardware_adapter:
                return JSONResponse({"error": "Hardware adapter not initialized"}, status_code=503)

            adapter = self.daemon.hardware_adapter
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
                CITRASCOPE_LOGGER.error(f"Error setting filter position: {e}", exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.post("/api/focuser/move")
        async def focuser_move(body: dict[str, Any]):
            """Move the focuser to an absolute position or by relative steps.

            Jog moves are fire-and-forget: the command is issued and the
            endpoint returns immediately.  The UI tracks position via the
            status poll.  Issuing a move while the focuser is already moving
            stops the previous move first.
            """
            if busy := self._require_system_idle():
                return busy
            if not self.daemon or not self.daemon.hardware_adapter:
                return JSONResponse({"error": "Hardware adapter not initialized"}, status_code=503)

            focuser = self.daemon.hardware_adapter.focuser
            if focuser is None or not focuser.is_connected():
                return JSONResponse({"error": "No focuser connected"}, status_code=404)

            absolute = body.get("position")
            relative = body.get("relative")

            try:
                if await asyncio.to_thread(focuser.is_moving):
                    await asyncio.to_thread(focuser.abort_move)
                    await asyncio.sleep(0.1)

                if absolute is not None:
                    if not isinstance(absolute, int):
                        return JSONResponse({"error": "position must be an integer"}, status_code=400)
                    if not await asyncio.to_thread(focuser.move_absolute, absolute):
                        return JSONResponse({"error": "Move failed"}, status_code=500)
                    pos = await asyncio.to_thread(focuser.get_position)
                    return {"success": True, "position": pos}

                if relative is not None:
                    if not isinstance(relative, int):
                        return JSONResponse({"error": "relative must be an integer"}, status_code=400)
                    if not await asyncio.to_thread(focuser.move_relative, relative):
                        return JSONResponse({"error": "Move failed"}, status_code=500)
                    pos = await asyncio.to_thread(focuser.get_position)
                    return {"success": True, "position": pos}

            except Exception as e:
                CITRASCOPE_LOGGER.error(f"Focuser move error: {e}", exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

            return JSONResponse({"error": "Provide 'position' (absolute) or 'relative' (steps)"}, status_code=400)

        @self.app.post("/api/focuser/abort")
        async def focuser_abort():
            """Stop focuser movement."""
            if not self.daemon or not self.daemon.hardware_adapter:
                return JSONResponse({"error": "Hardware adapter not initialized"}, status_code=503)

            focuser = self.daemon.hardware_adapter.focuser
            if focuser is None or not focuser.is_connected():
                return JSONResponse({"error": "No focuser connected"}, status_code=404)

            try:
                focuser.abort_move()
                return {"success": True, "position": focuser.get_position()}
            except Exception as e:
                CITRASCOPE_LOGGER.error(f"Focuser abort error: {e}", exc_info=True)
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
            """Cancel autofocus — works whether queued or actively running."""
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

        # ── Pointing model endpoints ───────────────────────────────────

        @self.app.post("/api/mount/pointing-model/calibrate")
        async def calibrate_pointing_model():
            """Trigger a full pointing model calibration run."""
            if not self.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)
            if not self.daemon.task_manager:
                return JSONResponse({"error": "Task manager not available"}, status_code=503)

            alignment_mgr = self.daemon.task_manager.alignment_manager
            if alignment_mgr.is_calibrating():
                return JSONResponse({"error": "Calibration already running"}, status_code=409)

            try:
                ok = alignment_mgr.request_calibration()
                if ok:
                    return {"success": True, "message": "Pointing calibration queued"}
                return JSONResponse({"error": "Calibration request rejected"}, status_code=409)
            except Exception as e:
                CITRASCOPE_LOGGER.error(f"Error starting pointing calibration: {e}", exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.post("/api/mount/pointing-model/reset")
        async def reset_pointing_model():
            """Clear the pointing model and persisted state."""
            if busy := self._require_system_idle():
                return busy
            if not self.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)

            adapter = self.daemon.hardware_adapter
            if adapter.pointing_model:
                adapter.pointing_model.reset()
                return {"success": True, "message": "Pointing model reset"}
            return JSONResponse({"error": "Pointing model not available"}, status_code=404)

        @self.app.post("/api/mount/pointing-model/calibrate/cancel")
        async def cancel_pointing_calibration():
            """Cancel an in-progress or pending pointing calibration."""
            if not self.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)
            if not self.daemon.task_manager:
                return JSONResponse({"error": "Task manager not available"}, status_code=503)

            try:
                self.daemon.task_manager.alignment_manager.cancel_calibration()
                return {"success": True}
            except Exception as e:
                CITRASCOPE_LOGGER.error(f"Error cancelling pointing calibration: {e}", exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        # ── Calibration endpoints ─────────────────────────────────────

        @self.app.get("/api/calibration/status")
        async def get_calibration_status():
            """Return calibration library status for the connected camera."""
            if not self.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)
            lib = getattr(self.daemon, "calibration_library", None)
            hw = self.daemon.hardware_adapter
            if not lib or not hw or not hw.supports_direct_camera_control():
                return {"available": False}

            camera = hw.camera
            if not camera:
                return {"available": False}

            profile = camera.get_calibration_profile()
            if not profile.calibration_applicable:
                return {"available": False}

            library_status = lib.get_library_status(profile.camera_id)
            tm = self.daemon.task_manager
            cal_mgr = tm.calibration_manager if tm else None

            filters: list[dict[str, Any]] = []
            if hw.supports_filter_management():
                filters = [
                    {"name": f["name"], "position": int(pos)}
                    for pos, f in hw.get_filter_config().items()
                    if f.get("enabled", True) and f.get("name")
                ]

            return {
                "available": True,
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
                "library": library_status,
                "masters_dir": str(lib.masters_dir),
                "capture_running": cal_mgr.is_running() if cal_mgr else False,
                "capture_requested": cal_mgr.is_requested() if cal_mgr else False,
                "capture_progress": cal_mgr.get_progress() if cal_mgr else {},
                "frame_count_setting": self.daemon.settings.calibration_frame_count if self.daemon.settings else 30,
                "flat_frame_count_setting": self.daemon.settings.flat_frame_count if self.daemon.settings else 15,
            }

        @self.app.post("/api/calibration/capture")
        async def trigger_calibration_capture(request: dict[str, Any]):
            """Queue a calibration capture job."""
            if not self.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)
            try:
                if request.get("frame_type") == "flat":
                    fp = request.get("filter_position")
                    if fp is None:
                        return JSONResponse({"error": "filter_position is required for flat frames"}, status_code=400)
                    fp_int = int(fp)
                    fm = self.daemon.hardware_adapter.filter_map if self.daemon.hardware_adapter else {}
                    if fp_int not in fm:
                        return JSONResponse({"error": f"Unknown filter position: {fp_int}"}, status_code=400)
                    request["filter_position"] = fp_int
                    request["filter_name"] = fm[fp_int].get("name", f"Filter {fp_int}")

                ok, err = self.daemon.trigger_calibration(request)
                if not ok:
                    return JSONResponse({"error": err}, status_code=400)
                return {"success": True, "message": "Calibration queued"}
            except Exception as e:
                CITRASCOPE_LOGGER.error("Error triggering calibration: %s", e, exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.post("/api/calibration/cancel")
        async def cancel_calibration():
            """Cancel pending or active calibration capture."""
            if not self.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)
            try:
                was_cancelled = self.daemon.cancel_calibration()
                return {"success": was_cancelled}
            except Exception as e:
                CITRASCOPE_LOGGER.error("Error cancelling calibration: %s", e, exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.post("/api/calibration/capture-suite")
        async def trigger_calibration_suite(request: dict[str, Any]):
            """Queue a calibration suite (bias_and_dark or all_flats)."""
            if not self.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)
            try:
                from citrascope.calibration import FilterSlot
                from citrascope.calibration.calibration_suites import all_flats_suite, bias_and_dark_suite

                suite_name = request.get("suite", "")
                hw = self.daemon.hardware_adapter
                if not hw or not hw.supports_direct_camera_control():
                    return JSONResponse({"error": "No direct camera control"}, status_code=400)

                camera = hw.camera
                if not camera:
                    return JSONResponse({"error": "Camera not connected"}, status_code=400)

                profile = camera.get_calibration_profile()
                if not profile.calibration_applicable:
                    return JSONResponse({"error": "Camera does not support calibration"}, status_code=400)

                frame_count = self.daemon.settings.calibration_frame_count if self.daemon.settings else 30
                flat_count = self.daemon.settings.flat_frame_count if self.daemon.settings else 15

                if suite_name == "bias_and_dark":
                    jobs = bias_and_dark_suite(profile, frame_count)
                elif suite_name == "all_flats":
                    filters: list[FilterSlot] = []
                    if hw.supports_filter_management():
                        filters = [
                            FilterSlot(position=int(pos), name=f["name"])
                            for pos, f in hw.get_filter_config().items()
                            if f.get("enabled", True) and f.get("name")
                        ]
                    if not filters:
                        return JSONResponse({"error": "No filters configured"}, status_code=400)
                    jobs = all_flats_suite(profile, filters, flat_count)
                else:
                    return JSONResponse({"error": f"Unknown suite: {suite_name}"}, status_code=400)

                if not jobs:
                    return JSONResponse({"error": "Suite generated no jobs"}, status_code=400)

                ok, err = self.daemon.trigger_calibration_suite(jobs)
                if not ok:
                    return JSONResponse({"error": err}, status_code=400)
                return {"success": True, "message": f"Suite queued: {len(jobs)} jobs", "job_count": len(jobs)}
            except Exception as e:
                CITRASCOPE_LOGGER.error("Error triggering calibration suite: %s", e, exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.delete("/api/calibration/master")
        async def delete_calibration_master(request: dict[str, Any]):
            """Delete a specific master calibration frame."""
            if not self.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)
            lib = getattr(self.daemon, "calibration_library", None)
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
                CITRASCOPE_LOGGER.error("Error deleting calibration master: %s", e, exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.get("/api/calibration/master/download")
        async def download_calibration_master(filename: str):
            """Download a master calibration FITS file by filename."""
            if not self.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)
            lib = getattr(self.daemon, "calibration_library", None)
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

        @self.app.get("/api/twilight")
        async def get_twilight_info():
            """Return current/next nautical twilight flat window for the observatory.

            The "flat window" is the nautical twilight band where the Sun
            is between -6 deg (civil) and -12 deg (nautical) below the
            horizon — bright enough for uniform sky illumination, dark
            enough to avoid saturation.
            """
            if not self.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)

            loc_svc = getattr(self.daemon, "location_service", None)
            if not loc_svc:
                return {"location_available": False}

            location = loc_svc.get_current_location()
            if not location:
                return {"location_available": False}

            try:
                info = await asyncio.to_thread(compute_twilight, location["latitude"], location["longitude"])
                return info.to_dict()
            except Exception as e:
                CITRASCOPE_LOGGER.error("Error computing twilight info: %s", e, exc_info=True)
                return {"location_available": False, "error": str(e)}

        @self.app.post("/api/adapter/sync")
        async def manual_sync(request: dict[str, Any]):
            """Manually sync the mount to given RA/Dec coordinates."""
            if busy := self._require_system_idle():
                return busy
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

            mount = self.daemon.hardware_adapter.mount
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

        @self.app.post("/api/mount/move")
        async def mount_move(body: dict[str, Any]):
            """Start or stop directional mount motion (jog control).

            In alt-az mode: north=up, south=down, east=right, west=left.
            """
            if busy := self._require_system_idle():
                return busy
            if not self.daemon or not self.daemon.hardware_adapter:
                return JSONResponse({"error": "Hardware adapter not initialized"}, status_code=503)

            mount = self.daemon.hardware_adapter.mount
            if mount is None or not self.daemon.hardware_adapter.is_telescope_connected():
                return JSONResponse({"error": "No mount connected"}, status_code=404)

            action = body.get("action")
            direction = body.get("direction")
            valid_directions = ("north", "south", "east", "west")

            if direction not in valid_directions:
                return JSONResponse({"error": f"direction must be one of {valid_directions}"}, status_code=400)

            try:
                if action == "start":
                    rate = body.get("rate")
                    if rate is not None:
                        if isinstance(rate, bool) or not isinstance(rate, int) or not 0 <= rate <= mount.max_move_rate:
                            return JSONResponse(
                                {"error": f"rate must be an integer 0-{mount.max_move_rate}"}, status_code=400
                            )
                    ok = await asyncio.to_thread(mount.start_move, direction, rate)
                    return {"success": ok}
                elif action == "stop":
                    ok = await asyncio.to_thread(mount.stop_move, direction)
                    return {"success": ok}
                else:
                    return JSONResponse({"error": "action must be 'start' or 'stop'"}, status_code=400)
            except Exception as e:
                CITRASCOPE_LOGGER.error(f"Mount move error: {e}", exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.post("/api/mount/goto")
        async def mount_goto(body: dict[str, Any]):
            """Slew the mount to arbitrary RA/Dec coordinates (degrees).

            Fire-and-forget: initiates the slew and returns immediately.
            The UI tracks slew progress via mount_slewing in the status poll.
            """
            if busy := self._require_system_idle():
                return busy
            if not self.daemon or not self.daemon.hardware_adapter:
                return JSONResponse({"error": "Hardware adapter not initialized"}, status_code=503)

            mount = self.daemon.hardware_adapter.mount
            if mount is None or not self.daemon.hardware_adapter.is_telescope_connected():
                return JSONResponse({"error": "No mount connected"}, status_code=404)

            ra = body.get("ra")
            dec = body.get("dec")
            if not isinstance(ra, (int, float)) or not isinstance(dec, (int, float)):
                return JSONResponse({"error": "ra and dec must be numbers (degrees)"}, status_code=400)
            if not 0 <= float(ra) <= 360:
                return JSONResponse({"error": "ra must be 0-360"}, status_code=400)
            if not -90 <= float(dec) <= 90:
                return JSONResponse({"error": "dec must be -90 to 90"}, status_code=400)

            try:
                ok = await asyncio.to_thread(mount.slew_to_radec, float(ra), float(dec))
                if not ok:
                    return JSONResponse({"error": "Mount rejected slew command"}, status_code=500)
                return {"success": True, "message": f"Slewing to RA={ra:.4f}, Dec={dec:.4f}"}
            except Exception as e:
                CITRASCOPE_LOGGER.error(f"Mount goto error: {e}", exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.post("/api/mount/tracking")
        async def mount_tracking(body: dict[str, Any]):
            """Start or stop sidereal tracking."""
            if busy := self._require_system_idle():
                return busy
            if not self.daemon or not self.daemon.hardware_adapter:
                return JSONResponse({"error": "Hardware adapter not initialized"}, status_code=503)

            mount = self.daemon.hardware_adapter.mount
            if mount is None or not self.daemon.hardware_adapter.is_telescope_connected():
                return JSONResponse({"error": "No mount connected"}, status_code=404)

            enabled = body.get("enabled")
            if not isinstance(enabled, bool):
                return JSONResponse({"error": "enabled must be a boolean"}, status_code=400)

            try:
                if enabled:
                    ok = await asyncio.to_thread(mount.start_tracking)
                else:
                    ok = await asyncio.to_thread(mount.stop_tracking)
                return {"success": ok, "tracking": enabled}
            except Exception as e:
                CITRASCOPE_LOGGER.error(f"Mount tracking error: {e}", exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

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
            """Stop mount, pause task processing, cancel in-flight imaging."""
            if not self.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)

            import threading

            # Activate safety check — watchdog will enforce continuously
            if self.daemon.safety_monitor:
                self.daemon.safety_monitor.activate_operator_stop()

            cancelled = 0
            tm = self.daemon.task_manager
            if tm:
                tm.pause()
                cancelled = tm.clear_pending_tasks()
            if self.daemon.settings:
                self.daemon.settings.task_processing_paused = True
                self.daemon.settings.save()

            # Immediate mount halt in background thread (serial I/O can't
            # run on the async event loop).  The watchdog provides ongoing
            # enforcement at 1 Hz; this gives sub-second first response.
            daemon = self.daemon

            def _halt_mount():
                mount = daemon.hardware_adapter.mount if daemon.hardware_adapter else None
                if not mount:
                    return
                try:
                    mount.abort_slew()
                    mount.stop_tracking()
                    for d in ("north", "south", "east", "west"):
                        mount.stop_move(d)
                except Exception:
                    CITRASCOPE_LOGGER.error("Error halting mount during emergency stop", exc_info=True)

            threading.Thread(target=_halt_mount, daemon=True, name="emergency-stop").start()

            CITRASCOPE_LOGGER.warning(
                "EMERGENCY STOP by operator — processing paused, %d imaging task(s) cancelled, mount halt issued",
                cancelled,
            )
            return JSONResponse(
                {
                    "success": True,
                    "message": (
                        f"Emergency stop: mount halted, processing paused," f" {cancelled} imaging task(s) cancelled"
                    ),
                },
                status_code=202,
            )

        @self.app.post("/api/safety/operator-stop/clear")
        async def clear_operator_stop():
            """Clear the operator stop — allows motion to resume.

            Processing stays paused; the operator must manually re-enable it.
            """
            if not self.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)
            if not self.daemon.safety_monitor:
                return JSONResponse({"error": "Safety monitor not available"}, status_code=503)
            self.daemon.safety_monitor.clear_operator_stop()

            CITRASCOPE_LOGGER.info("Operator stop cleared — processing remains paused, re-enable manually")
            return {"success": True, "message": "Operator stop cleared — re-enable processing when ready"}

        @self.app.post("/api/mount/limits")
        async def set_mount_limits(request: dict[str, Any]):
            """Set the mount's altitude limits."""
            if busy := self._require_system_idle():
                return busy
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
            if busy := self._require_system_idle():
                return busy
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

        @self.app.post("/api/camera/preview")
        async def camera_preview(request: dict[str, Any]):
            """Take an ephemeral preview exposure and return a JPEG data URL."""
            if not self.daemon:
                return JSONResponse({"error": "Daemon not available"}, status_code=503)
            if not self.daemon.hardware_adapter:
                return JSONResponse({"error": "Hardware adapter not available"}, status_code=503)
            if not self.daemon.hardware_adapter.supports_direct_camera_control():
                return JSONResponse(
                    {"error": "Hardware adapter does not support direct camera control"}, status_code=400
                )

            task_manager = self.daemon.task_manager if hasattr(self.daemon, "task_manager") else None
            if task_manager and task_manager.is_processing_active():
                return JSONResponse({"error": "Camera unavailable — task processing is active"}, status_code=503)

            try:
                duration = request.get("duration", 1.0)
                if not isinstance(duration, (int, float)):
                    return JSONResponse({"error": "duration must be a number"}, status_code=400)
                duration = float(duration)
                if duration <= 0:
                    return JSONResponse({"error": "Exposure duration must be positive"}, status_code=400)
                if duration > 30:
                    return JSONResponse({"error": "Preview exposure must be 30 seconds or less"}, status_code=400)

                adapter = self.daemon.hardware_adapter
                flip_h = bool(request.get("flip_horizontal", False))
                image_data = await asyncio.to_thread(adapter.capture_preview, duration, flip_h)
                return {"image_data": image_data}

            except NotImplementedError:
                return JSONResponse({"error": "Preview not supported by this adapter"}, status_code=400)
            except RuntimeError as e:
                if "already in progress" in str(e):
                    return JSONResponse({"error": "busy"}, status_code=409)
                CITRASCOPE_LOGGER.error(f"Error during preview capture: {e}", exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)
            except Exception as e:
                CITRASCOPE_LOGGER.error(f"Error during preview capture: {e}", exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        # ── Analysis endpoints ──────────────────────────────────────────

        @self.app.get("/api/analysis/tasks")
        async def analysis_tasks(
            limit: int = 50,
            offset: int = 0,
            sort: str = "completed_at",
            order: str = "desc",
            target_name: str | None = None,
            plate_solved: bool | None = None,
            target_matched: bool | None = None,
            missed_window: bool | None = None,
            date_from: str | None = None,
            date_to: str | None = None,
        ):
            """Paginated, filterable list of completed tasks."""
            if not self.daemon or not self.daemon.task_index:
                return JSONResponse({"tasks": [], "total": 0})
            return self.daemon.task_index.query_tasks(
                limit=min(limit, 200),
                offset=offset,
                sort=sort,
                order=order,
                target_name=target_name,
                plate_solved=plate_solved,
                target_matched=target_matched,
                missed_window=missed_window,
                date_from=date_from,
                date_to=date_to,
            )

        @self.app.get("/api/analysis/tasks/{task_id}")
        async def analysis_task_detail(task_id: str):
            """Single task detail with all fields."""
            if not self.daemon or not self.daemon.task_index:
                return JSONResponse({"error": "Analysis not available"}, status_code=503)
            task = self.daemon.task_index.get_task(task_id)
            if task is None:
                return JSONResponse({"error": "Task not found"}, status_code=404)
            task["artifacts_available"] = (self.daemon.settings.directories.processing_dir / task_id).is_dir()
            return task

        @self.app.get("/api/analysis/tasks/{task_id}/image")
        async def analysis_task_image(task_id: str):
            """Serve annotated preview image for a task."""
            if not self.daemon:
                return JSONResponse({"error": "Not available"}, status_code=503)
            preview = self.daemon.settings.directories.analysis_previews_dir / f"{task_id}.jpg"
            if not preview.is_file():
                return JSONResponse({"error": "Image not available"}, status_code=404)
            return FileResponse(str(preview), media_type="image/jpeg")

        @self.app.get("/api/analysis/tasks/{task_id}/artifacts/{filename}")
        async def analysis_task_artifact(task_id: str, filename: str):
            """Serve an artifact file from the processing directory."""
            if not self.daemon:
                return JSONResponse({"error": "Not available"}, status_code=503)
            # Prevent directory traversal
            safe_name = Path(filename).name
            artifact = self.daemon.settings.directories.processing_dir / task_id / safe_name
            if not artifact.is_file():
                return JSONResponse({"error": "Artifact not found or expired"}, status_code=404)
            return FileResponse(str(artifact))

        @self.app.get("/api/analysis/tasks/{task_id}/bundle")
        async def analysis_task_bundle(task_id: str):
            """Stream a tar.gz bundle of a task's processing directory."""
            import io
            import tarfile

            if not self.daemon:
                return JSONResponse({"error": "Not available"}, status_code=503)
            safe_id = Path(task_id).name
            task_dir = self.daemon.settings.directories.processing_dir / safe_id
            if not task_dir.is_dir():
                return JSONResponse({"error": "Task artifacts not found or expired"}, status_code=404)

            def _generate():
                buf = io.BytesIO()
                with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                    for file_path in sorted(task_dir.rglob("*")):
                        if not file_path.is_file():
                            continue
                        arcname = f"{safe_id}/{file_path.relative_to(task_dir)}"
                        tar.add(str(file_path), arcname=arcname)
                buf.seek(0)
                yield from iter(lambda: buf.read(65536), b"")

            filename = f"{safe_id}.tar.gz"
            return StreamingResponse(
                _generate(),
                media_type="application/gzip",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )

        @self.app.get("/api/analysis/stats")
        async def analysis_stats(hours: int = 24):
            """Aggregate statistics over the given time window."""
            if not self.daemon or not self.daemon.task_index:
                from citrascope.analysis.task_index import _empty_stats

                return _empty_stats()
            return self.daemon.task_index.get_stats(hours=max(1, min(hours, 8760)))

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

    def _build_calibration_status(self) -> dict[str, Any] | None:
        """Build calibration status dict for SystemStatus."""
        if not self.daemon:
            return None
        lib = getattr(self.daemon, "calibration_library", None)
        hw = self.daemon.hardware_adapter
        if not lib or not hw or not hw.supports_direct_camera_control():
            return None

        camera = hw.camera
        if not camera:
            return None

        profile = camera.get_calibration_profile()
        if not profile.calibration_applicable:
            return None

        cam_id = profile.camera_id
        gain = profile.current_gain or 0
        binning = profile.current_binning
        temperature = profile.current_temperature
        read_mode = profile.read_mode

        filter_name = ""
        filter_pos = hw.get_filter_position() if hasattr(hw, "get_filter_position") else None
        if filter_pos is not None and hasattr(hw, "filter_map"):
            fdata = hw.filter_map.get(filter_pos, {})
            filter_name = fdata.get("name", "")

        has_bias = lib.get_master_bias(cam_id, gain, binning, read_mode) is not None
        has_dark = (
            lib.get_master_dark(cam_id, gain, binning, temperature or 0.0, read_mode) is not None
            if temperature is not None
            else False
        )
        has_flat = (
            lib.get_master_flat(cam_id, gain, binning, filter_name, read_mode) is not None if filter_name else True
        )

        missing: list[str] = []
        if not has_bias:
            missing.append(f"bias (gain {gain}, bin {binning})")
        if not has_dark:
            temp_str = f"{temperature:.1f}°C" if temperature is not None else "unknown"
            missing.append(f"dark (at {temp_str})")
        if filter_name and not has_flat:
            missing.append(f"flat ({filter_name})")

        # CalibrationManager state
        tm = self.daemon.task_manager
        cal_mgr = tm.calibration_manager if tm else None
        capture_running = cal_mgr.is_running() if cal_mgr else False
        capture_requested = cal_mgr.is_requested() if cal_mgr else False
        capture_progress = cal_mgr.get_progress() if cal_mgr else {}

        return {
            "has_bias": has_bias,
            "has_dark": has_dark,
            "has_flat": has_flat,
            "missing": missing,
            "missing_summary": ", ".join(missing) if missing else "",
            "capture_running": capture_running,
            "capture_requested": capture_requested,
            "capture_progress": capture_progress,
            "calibration_applicable": True,
            "has_mechanical_shutter": profile.has_mechanical_shutter,
            "has_cooling": profile.has_cooling,
            "camera_id": cam_id,
            "current_gain": gain,
            "current_binning": binning,
            "current_temperature": temperature,
        }

    def _update_status_from_daemon(self):
        """Update status from daemon state."""
        if not self.daemon:
            return

        import time as _time

        _t0 = _time.perf_counter()
        _prev = _t0
        _breakdown: dict[str, float] = {}

        def _mark(name: str) -> None:
            nonlocal _prev
            now = _time.perf_counter()
            _breakdown[name] = round((now - _prev) * 1000, 2)
            _prev = now

        try:
            self.status.hardware_adapter = self.daemon.settings.hardware_adapter

            if hasattr(self.daemon, "hardware_adapter") and self.daemon.hardware_adapter:
                adapter = self.daemon.hardware_adapter
                mount = adapter.mount

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
                            self.status.mount_tracking = snap.is_tracking
                            self.status.mount_slewing = snap.is_slewing
                        else:
                            ra, dec = adapter.get_telescope_direction()
                            self.status.telescope_ra = ra
                            self.status.telescope_dec = dec
                            self.status.mount_tracking = False
                            self.status.mount_slewing = False
                    else:
                        self.status.mount_tracking = False
                        self.status.mount_slewing = False
                except Exception:
                    self.status.telescope_connected = False
                    self.status.mount_tracking = False
                    self.status.mount_slewing = False
                _mark("hw.telescope")

                # Check camera connection status
                try:
                    self.status.camera_connected = adapter.is_camera_connected()
                    camera = adapter.camera
                    if self.status.camera_connected and camera is not None:
                        self.status.camera_temperature = camera.get_temperature()
                    else:
                        self.status.camera_temperature = None
                except Exception:
                    self.status.camera_connected = False
                    self.status.camera_temperature = None
                _mark("hw.camera")

                # Check adapter capabilities
                try:
                    self.status.supports_direct_camera_control = adapter.supports_direct_camera_control()
                except Exception:
                    self.status.supports_direct_camera_control = False

                self.status.supports_direct_mount_control = mount is not None and self.status.telescope_connected

                self.status.supports_autofocus = adapter.supports_autofocus()
                self.status.supports_hardware_safety_monitor = adapter.supports_hardware_safety_monitor
                _mark("hw.capabilities")

                try:
                    pos = adapter.get_filter_position()
                    self.status.current_filter_position = pos
                    if pos is not None and pos in adapter.filter_map:
                        self.status.current_filter_name = adapter.filter_map[pos].get("name")
                    else:
                        self.status.current_filter_name = None
                except Exception:
                    self.status.current_filter_position = None
                    self.status.current_filter_name = None
                _mark("hw.filter")

                # Check focuser status
                focuser = adapter.focuser
                if focuser is not None and focuser.is_connected():
                    self.status.focuser_connected = True
                    try:
                        self.status.focuser_position = focuser.get_position()
                    except Exception:
                        self.status.focuser_position = None
                    try:
                        self.status.focuser_max_position = focuser.get_max_position()
                    except Exception:
                        self.status.focuser_max_position = None
                    try:
                        self.status.focuser_temperature = focuser.get_temperature()
                    except Exception:
                        self.status.focuser_temperature = None
                    try:
                        self.status.focuser_moving = focuser.is_moving()
                    except Exception:
                        self.status.focuser_moving = False
                else:
                    self.status.focuser_connected = False
                    self.status.focuser_position = None
                    self.status.focuser_max_position = None
                    self.status.focuser_temperature = None
                    self.status.focuser_moving = False
                _mark("hw.focuser")

                self.status.supports_alignment = self.status.camera_connected and mount is not None

                if mount is not None and mount.cached_state is not None:
                    self.status.supports_manual_sync = mount.cached_mount_info.get("supports_sync", False)
                    self.status.mount_at_home = mount.cached_state.is_at_home
                    h_limit, o_limit = mount.cached_limits
                    self.status.mount_horizon_limit = h_limit
                    self.status.mount_overhead_limit = o_limit
                _mark("hw.mount_state")

            if hasattr(self.daemon, "task_manager") and self.daemon.task_manager:
                task_manager = self.daemon.task_manager
                self.status.current_task = task_manager.current_task_id
                self.status.mount_homing = (
                    task_manager.homing_manager.is_running() or task_manager.homing_manager.is_requested()
                )
                self.status.autofocus_requested = task_manager.autofocus_manager.is_requested()
                self.status.autofocus_running = task_manager.autofocus_manager.is_running()
                self.status.autofocus_progress = task_manager.autofocus_manager.progress
                self.status.autofocus_points = [
                    {"pos": p, "hfr": h, "filter": f} for p, h, f in task_manager.autofocus_manager.points
                ]
                self.status.autofocus_filter_results = task_manager.autofocus_manager.filter_results
                self.status.autofocus_last_result = task_manager.autofocus_manager.last_result
                hfr_hist = task_manager.autofocus_manager.hfr_history
                self.status.hfr_history = [{"hfr": h, "ts": t, "filter": f} for h, t, f in hfr_hist]
                self.status.last_hfr_median = hfr_hist[-1][0] if hfr_hist else None
                if self.daemon.settings:
                    self.status.hfr_baseline = self.daemon.settings.adapter_settings.get("hfr_baseline")
                    self.status.hfr_increase_percent = self.daemon.settings.autofocus_hfr_increase_percent
                    self.status.hfr_refocus_enabled = self.daemon.settings.autofocus_on_hfr_increase_enabled
                    self.status.hfr_sample_window = self.daemon.settings.autofocus_hfr_sample_window
                self.status.alignment_requested = task_manager.alignment_manager.is_requested()
                self.status.alignment_running = task_manager.alignment_manager.is_running()
                self.status.alignment_progress = task_manager.alignment_manager.progress
                self.status.pointing_calibration_running = task_manager.alignment_manager.is_calibrating()
                self.status.pointing_calibration_progress = task_manager.alignment_manager.calibration_progress
                self.status.tasks_pending = task_manager.pending_task_count

                busy_reasons: list[str] = []
                if task_manager.is_processing_active():
                    busy_reasons.append("processing")
                if not task_manager.imaging_queue.is_idle():
                    busy_reasons.append("imaging")
                if self.status.alignment_running:
                    busy_reasons.append("alignment")
                if self.status.autofocus_running:
                    busy_reasons.append("autofocus")
                if self.status.pointing_calibration_running:
                    busy_reasons.append("calibration")
                if self.status.mount_homing:
                    busy_reasons.append("homing")
                self.status.system_busy = bool(busy_reasons)
                self.status.system_busy_reason = ", ".join(busy_reasons)

            _mark("task_manager")

            # Get autofocus timing information
            if self.daemon.settings:
                settings = self.daemon.settings
                self.status.last_autofocus_timestamp = settings.last_autofocus_timestamp
                self.status.last_alignment_timestamp = settings.last_alignment_timestamp
                self.status.autofocus_target_name = _resolve_autofocus_target_name(settings)

                # Calculate next autofocus time (delegates mode-aware logic to AutofocusManager)
                if (
                    hasattr(self.daemon, "task_manager")
                    and self.daemon.task_manager
                    and hasattr(self.daemon.task_manager, "autofocus_manager")
                ):
                    self.status.next_autofocus_minutes = (
                        self.daemon.task_manager.autofocus_manager.get_next_autofocus_minutes()
                    )
                elif settings.scheduled_autofocus_enabled:
                    last_ts = settings.last_autofocus_timestamp
                    interval_minutes = settings.autofocus_interval_minutes
                    if last_ts is not None:
                        elapsed_minutes = (int(time.time()) - last_ts) / 60
                        remaining = max(0, interval_minutes - elapsed_minutes)
                        self.status.next_autofocus_minutes = int(remaining)
                    else:
                        self.status.next_autofocus_minutes = 0
                else:
                    self.status.next_autofocus_minutes = None

            _mark("autofocus")

            # Get time sync status from time monitor
            if hasattr(self.daemon, "time_monitor") and self.daemon.time_monitor:
                health = self.daemon.time_monitor.get_current_health()
                self.status.time_health = health.to_dict() if health else None
            else:
                # Time monitoring not initialized yet
                self.status.time_health = None

            # Get GPS location status from location service (gpsd or camera GPS)
            # Use allow_blocking=False to prevent blocking the async event loop
            if hasattr(self.daemon, "location_service") and self.daemon.location_service:
                gps_fix = self.daemon.location_service.get_best_gps_fix(allow_blocking=False)
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
                        "gpsd_version": gps_fix.gpsd_version,
                        "device_path": gps_fix.device_path,
                        "device_driver": gps_fix.device_driver,
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
                self.status.ground_station_latitude = gs_record.get("latitude")
                self.status.ground_station_longitude = gs_record.get("longitude")
                self.status.ground_station_altitude = gs_record.get("altitude")

            # Resolve active operating location from data already fetched above
            # (avoids calling get_current_location() which can block on subprocess)
            gps = self.status.gps_location
            if gps and gps.get("is_strong") and gps.get("latitude") is not None and gps.get("longitude") is not None:
                self.status.location_source = "gps"
                self.status.location_latitude = gps["latitude"]
                self.status.location_longitude = gps["longitude"]
                self.status.location_altitude = gps.get("altitude")
            elif self.status.ground_station_latitude is not None and self.status.ground_station_longitude is not None:
                self.status.location_source = "ground_station"
                self.status.location_latitude = self.status.ground_station_latitude
                self.status.location_longitude = self.status.ground_station_longitude
                self.status.location_altitude = self.status.ground_station_altitude
            else:
                self.status.location_source = None
                self.status.location_latitude = None
                self.status.location_longitude = None
                self.status.location_altitude = None

            _mark("time_gps_location")

            # Update task processing state
            if hasattr(self.daemon, "task_manager") and self.daemon.task_manager:
                self.status.processing_active = self.daemon.task_manager.is_processing_active()
                self.status.automated_scheduling = self.daemon.task_manager.automated_scheduling

                # Observing session / self-tasking status
                osm = self.daemon.task_manager.observing_session_manager
                stm = self.daemon.task_manager.self_tasking_manager
                if osm:
                    sd = osm.status_dict()
                    self.status.observing_session_state = sd.get("observing_session_state", "daytime")
                    self.status.session_activity = sd.get("session_activity")
                    self.status.observing_session_threshold = sd.get("observing_session_threshold", -12.0)
                    self.status.sun_altitude = sd.get("sun_altitude")
                    self.status.dark_window_start = sd.get("dark_window_start")
                    self.status.dark_window_end = sd.get("dark_window_end")
                if stm:
                    st = stm.status_dict()
                    self.status.last_batch_request = st.get("last_batch_request")
                    self.status.last_batch_created = st.get("last_batch_created")
                    self.status.next_request_seconds = st.get("next_request_seconds")

            self.status.observing_session_enabled = self.daemon.settings.observing_session_enabled
            self.status.self_tasking_enabled = self.daemon.settings.self_tasking_enabled

            _mark("session")

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

            _mark("pipeline")

            # Safety monitor status
            if hasattr(self.daemon, "safety_monitor") and self.daemon.safety_monitor:
                try:
                    self.status.safety_status = self.daemon.safety_monitor.get_status()
                except Exception:
                    self.status.safety_status = None
            else:
                self.status.safety_status = None

            # Elset cache health for satellite matching status
            if hasattr(self.daemon, "elset_cache") and self.daemon.elset_cache:
                self.status.elset_health = self.daemon.elset_cache.get_health()
            else:
                self.status.elset_health = None

            _mark("safety_elset")

            # Latest annotated task image for the Optics pane
            ann_path = getattr(self.daemon, "latest_annotated_image_path", None)
            if ann_path and Path(ann_path).exists():
                mtime_ns = Path(ann_path).stat().st_mtime_ns
                self.status.latest_task_image_url = f"/api/task-preview/latest?t={mtime_ns}"
            else:
                self.status.latest_task_image_url = None

            # Calibration status
            self.status.calibration_status = self._build_calibration_status()

            # Pointing model status
            adapter = self.daemon.hardware_adapter
            self.status.pointing_model = adapter.get_pointing_model_status() if adapter else None
            self.status.fov_short_deg = adapter.observed_fov_short_deg if adapter else None

            # Config health: compare server telescope record vs hardware + plate solve
            if self.daemon.telescope_record and adapter:
                from citrascope.hardware.config_health import assess_config_health

                camera_info = adapter.get_camera_info()
                health = assess_config_health(
                    telescope_record=self.daemon.telescope_record,
                    camera_info=camera_info,
                    observed_pixel_scale=adapter.observed_pixel_scale_arcsec,
                    observed_fov_w=adapter.observed_fov_w_deg,
                    observed_fov_h=adapter.observed_fov_h_deg,
                    observed_slew_rate=adapter.observed_slew_rate_deg_per_s,
                )
                self.status.config_health = health.to_dict()
            else:
                self.status.config_health = None

            _mark("optics_calibration")

            self.status.last_update = datetime.now().isoformat()
            self.status.status_collection_ms = round((_time.perf_counter() - _t0) * 1000, 2)
            self.status.status_collection_breakdown = _breakdown

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
        tasks = [_task_to_dict(t) for t in task_manager.get_tasks_snapshot()]
        await self.connection_manager.broadcast({"type": "tasks", "data": tasks})

    async def broadcast_preview(self):
        """Pop a frame from the PreviewBus and broadcast it to all clients."""
        if not self.daemon:
            return
        bus = getattr(self.daemon, "preview_bus", None)
        if not bus:
            return
        frame = bus.pop()
        if frame:
            data_url, source = frame
            await self.connection_manager.broadcast({"type": "preview", "data": data_url, "source": source})

    async def broadcast_log(self, log_entry: dict):
        """Broadcast log entry to all connected clients."""
        await self.connection_manager.broadcast({"type": "log", "data": log_entry})

    async def broadcast_toast(self, message: str, toast_type: str = "info", toast_id: str | None = None):
        """Broadcast a toast notification to all connected web clients."""
        await self.connection_manager.broadcast(
            {"type": "toast", "data": {"message": message, "toast_type": toast_type, "id": toast_id}}
        )
