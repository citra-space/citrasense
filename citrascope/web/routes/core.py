"""Read-only info endpoints and the dashboard root template."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from citrascope.constants import DEV_API_HOST, DEV_APP_URL, PROD_APP_URL
from citrascope.location.twilight import compute_twilight
from citrascope.logging import CITRASCOPE_LOGGER

if TYPE_CHECKING:
    from citrascope.web.app import CitraScopeWebApp


def build_core_router(ctx: CitraScopeWebApp) -> APIRouter:
    """Core informational endpoints: dashboard root, status, version, config, logs, twilight."""
    router = APIRouter(tags=["core"])

    @router.get("/", response_class=HTMLResponse)
    async def root(request: Request):
        """Serve the main dashboard page."""
        return ctx.templates.TemplateResponse(request, "dashboard.html")

    @router.get("/api/status")
    async def get_status():
        """Get current system status."""
        if ctx.daemon:
            ctx._update_status_from_daemon()
        return ctx.status

    @router.get("/api/task-preview/latest")
    async def get_latest_task_preview():
        """Serve the latest annotated task image."""
        ann_path = getattr(ctx.daemon, "latest_annotated_image_path", None)
        if not ann_path or not Path(ann_path).exists():
            return JSONResponse({"error": "No preview available"}, status_code=404)
        mime = "image/jpeg" if Path(ann_path).suffix.lower() in (".jpg", ".jpeg") else "image/png"
        return FileResponse(ann_path, media_type=mime)

    @router.get("/api/config")
    async def get_config():
        """Get current configuration."""
        if not ctx.daemon or not ctx.daemon.settings:
            return JSONResponse({"error": "Configuration not available"}, status_code=503)

        settings = ctx.daemon.settings
        app_url = DEV_APP_URL if settings.host == DEV_API_HOST else PROD_APP_URL

        config_path = str(settings.config_manager.get_config_path())
        log_file_path = str(settings.directories.current_log_path()) if settings.file_logging_enabled else None
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

    @router.get("/api/config/status")
    async def get_config_status():
        """Get configuration status."""
        if not ctx.daemon or not ctx.daemon.settings:
            return {"configured": False, "error": "Settings not available"}

        return {
            "configured": ctx.daemon.settings.is_configured(),
            "error": getattr(ctx.daemon, "configuration_error", None),
        }

    @router.get("/api/version")
    async def get_version():
        """Get CitraScope version and install metadata."""
        from citrascope.version import get_version_info

        return get_version_info()

    @router.get("/api/processors")
    async def get_processors():
        """Get list of all available processors with metadata."""
        if not ctx.daemon or not hasattr(ctx.daemon, "processor_registry") or not ctx.daemon.processor_registry:
            return []

        return ctx.daemon.processor_registry.get_all_processors()

    @router.get("/api/logs")
    async def get_logs(limit: int = 100):
        """Get recent log entries."""
        if ctx.web_log_handler:
            logs = ctx.web_log_handler.get_recent_logs(limit)
            return {"logs": logs}
        return {"logs": []}

    @router.get("/api/logs/download")
    async def download_log():
        """Download the current log file."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        settings = ctx.daemon.settings
        if not settings.file_logging_enabled:
            return JSONResponse({"error": "File logging is disabled"}, status_code=404)
        log_path = settings.directories.current_log_path()
        if not log_path.exists():
            return JSONResponse({"error": "Log file not found"}, status_code=404)
        return FileResponse(
            str(log_path),
            filename=log_path.name,
            media_type="text/plain",
        )

    @router.get("/api/twilight")
    async def get_twilight_info():
        """Return current/next nautical twilight flat window for the observatory.

        The "flat window" is the nautical twilight band where the Sun
        is between -6 deg (civil) and -12 deg (nautical) below the
        horizon — bright enough for uniform sky illumination, dark
        enough to avoid saturation.
        """
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)

        loc_svc = getattr(ctx.daemon, "location_service", None)
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

    return router
