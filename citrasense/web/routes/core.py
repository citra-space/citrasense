"""Read-only info endpoints and the dashboard root template."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from citrasense.constants import DEV_API_HOST, DEV_APP_URL, PROD_APP_URL
from citrasense.location.twilight import compute_twilight
from citrasense.logging import CITRASENSE_LOGGER

if TYPE_CHECKING:
    from citrasense.web.app import CitraSenseWebApp


def build_core_router(ctx: CitraSenseWebApp) -> APIRouter:
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
    async def get_latest_task_preview(sensor_id: str | None = None):
        """Serve the latest annotated task image for *sensor_id*.

        Previously this endpoint fell back to "newest preview across all
        sensors" when ``sensor_id`` was missing or unrecognised, which
        in multi-sensor deployments silently aliased two telescopes to
        the same preview slot.  Now the caller must name the sensor —
        other callers should reach for the per-sensor runtime directly.
        """
        paths = getattr(ctx.daemon, "latest_annotated_image_paths", {})
        if not sensor_id:
            return JSONResponse({"error": "sensor_id is required"}, status_code=400)
        ann_path = paths.get(sensor_id)
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
        """Get CitraSense version and install metadata."""
        from citrasense.version import get_version_info

        return get_version_info()

    _update_check_cache: dict = {"result": None, "fetched_at": 0.0}

    @router.get("/api/version/check-updates")
    async def check_for_updates():
        """Server-side GitHub version check with 1-hour TTL cache.

        Replaces the client-side GitHub API calls that were rate-limited
        (60 req/hr/IP unauthenticated) and failed when the telescope
        could reach the daemon but not the internet.
        """
        import time

        import requests

        from citrasense.version import get_version_info

        TTL_SECONDS = 3600
        now = time.time()

        if _update_check_cache["result"] and (now - _update_check_cache["fetched_at"]) < TTL_SECONDS:
            return _update_check_cache["result"]

        info = get_version_info()
        current_version = info["version"]
        install_type = info["install_type"]
        git_hash = info.get("git_hash")
        git_branch = info.get("git_branch")
        git_dirty = info.get("git_dirty", False)
        base = {
            "current_version": current_version,
            "install_type": install_type,
            "git_hash": git_hash,
            "git_branch": git_branch,
            "git_dirty": git_dirty,
        }

        if current_version in ("development", "unknown"):
            result = {"status": "up-to-date", **base}
            _update_check_cache.update(result=result, fetched_at=now)
            return result

        try:
            if install_type != "pypi" and git_hash:
                resp = await asyncio.to_thread(
                    requests.get,
                    f"https://api.github.com/repos/citra-space/citrasense/compare/{git_hash}...main",
                    timeout=10,
                )
                if resp.status_code != 200:
                    result = {"status": "error", **base}
                    _update_check_cache.update(result=result, fetched_at=now)
                    return result
                behind_by = resp.json().get("ahead_by", 0)
                if behind_by > 0:
                    result = {"status": "update-available", "behind_by": behind_by, **base}
                else:
                    result = {"status": "up-to-date", **base}
                _update_check_cache.update(result=result, fetched_at=now)
                return result

            resp = await asyncio.to_thread(
                requests.get,
                "https://api.github.com/repos/citra-space/citrasense/releases/latest",
                timeout=10,
            )
            if resp.status_code != 200:
                result = {"status": "error", **base}
                _update_check_cache.update(result=result, fetched_at=now)
                return result

            release_data = resp.json()
            latest_version = release_data.get("tag_name", "").lstrip("v")
            release_url = release_data.get("html_url", "")

            def _version_tuple(v: str) -> tuple:
                return tuple(int(x) for x in v.split(".") if x.isdigit())

            if _version_tuple(latest_version) > _version_tuple(current_version):
                result = {
                    "status": "update-available",
                    "latest_version": latest_version,
                    "release_url": release_url,
                    **base,
                }
            else:
                result = {"status": "up-to-date", **base}
            _update_check_cache.update(result=result, fetched_at=now)
            return result
        except Exception as e:
            CITRASENSE_LOGGER.debug("Update check failed: %s", e)
            result = {"status": "error", "error": str(e), **base}
            _update_check_cache.update(result=result, fetched_at=now)
            return result

    @router.get("/api/processors")
    async def get_processors(sensor_id: str):
        """Get list of all available processors with metadata.

        ``sensor_id`` selects which sensor's ``enabled_processors`` map to
        apply to the returned ``enabled`` flags. It is required — there is
        no "default" sensor in a multi-sensor deployment.
        """
        if not ctx.daemon or not hasattr(ctx.daemon, "processor_registry") or not ctx.daemon.processor_registry:
            return []

        settings = getattr(ctx.daemon, "settings", None)
        if not settings or not settings.sensors:
            return JSONResponse({"error": "No sensors configured"}, status_code=503)

        sensor_config = settings.get_sensor_config(sensor_id)
        if sensor_config is None:
            return JSONResponse({"error": f"Unknown sensor_id: {sensor_id}"}, status_code=400)

        return ctx.daemon.processor_registry.get_all_processors(sensor_config=sensor_config)

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
            CITRASENSE_LOGGER.error("Error computing twilight info: %s", e, exc_info=True)
            return {"location_available": False, "error": str(e)}

    return router
