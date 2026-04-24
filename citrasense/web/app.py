"""FastAPI web application for CitraSense monitoring and configuration."""

import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from citrasense.settings.directory_manager import DirectoryManager
from citrasense.web.connection_manager import ConnectionManager
from citrasense.web.helpers import (
    FILTER_NAME_OPTIONS,
    _gps_fix_to_dict,
    _resolve_autofocus_target_name,
)
from citrasense.web.models import HardwareConfig, SystemStatus
from citrasense.web.routes import build_all_routers
from citrasense.web.sky_enrichment import get_web_tasks
from citrasense.web.status_collector import StatusCollector

__all__ = [
    "FILTER_NAME_OPTIONS",
    "CitraSenseWebApp",
    "ConnectionManager",
    "HardwareConfig",
    "SystemStatus",
    "_gps_fix_to_dict",
    "_resolve_autofocus_target_name",
]


class CitraSenseWebApp:
    """Web application for CitraSense."""

    def __init__(self, daemon=None, web_log_handler=None):
        self.app = FastAPI(title="CitraSense", description="Telescope Control and Monitoring")
        self.daemon = daemon
        self.connection_manager = ConnectionManager()
        self.status = SystemStatus()
        self._status_collector = StatusCollector(daemon)
        self.web_log_handler = web_log_handler

        from citrasense.jobs import BackgroundJobRunner

        self.job_runner = BackgroundJobRunner()

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
        self._status_collector.daemon = daemon

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

        for build in build_all_routers():
            self.app.include_router(build(self))

    def _update_status_from_daemon(self):
        """Update status from daemon state."""
        if not self.daemon:
            return
        self._status_collector.collect(self.status)

    async def broadcast_status(self):
        """Broadcast current status to all connected clients."""
        if self.daemon:
            self._update_status_from_daemon()
        await self.connection_manager.broadcast({"type": "status", "data": self.status.model_dump()})

    async def broadcast_tasks(self):
        """Broadcast current task queue to all connected clients.

        Goes through :func:`get_web_tasks` so the wire format matches
        ``GET /api/tasks`` exactly -- previously the two emitters built
        their own dicts and one of them shipped without the sky fields.
        Always broadcasts when the daemon is ready (even an empty list) so
        the client can clear its table when the queue drains.
        """
        if not self.daemon or not getattr(self.daemon, "task_dispatcher", None):
            return
        tasks = get_web_tasks(self.daemon)
        await self.connection_manager.broadcast({"type": "tasks", "data": tasks})

    async def broadcast_preview(self):
        """Pop all pending preview frames and broadcast them to all clients."""
        if not self.daemon:
            return
        bus = getattr(self.daemon, "preview_bus", None)
        if not bus:
            return
        for payload, source, kind, sensor_id in bus.pop_all():
            msg = {"source": source, "sensor_id": sensor_id}
            if kind == "url":
                msg["type"] = "preview_url"
                msg["url"] = payload
            else:
                msg["type"] = "preview"
                msg["data"] = payload
            await self.connection_manager.broadcast(msg)

    async def broadcast_log(self, log_entry: dict):
        """Broadcast log entry to all connected clients."""
        await self.connection_manager.broadcast({"type": "log", "data": log_entry})

    async def broadcast_toast(self, message: str, toast_type: str = "info", toast_id: str | None = None):
        """Broadcast a toast notification to all connected web clients."""
        await self.connection_manager.broadcast(
            {"type": "toast", "data": {"message": message, "toast_type": toast_type, "id": toast_id}}
        )
