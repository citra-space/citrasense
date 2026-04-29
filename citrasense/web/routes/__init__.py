"""FastAPI router factories for the CitraSense web layer.

Each `build_*_router(ctx)` function returns an `APIRouter` whose handlers close
over the `CitraSenseWebApp` instance (`ctx`). This preserves the exact closure
semantics of the original inline route definitions in `app.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from fastapi import APIRouter

from citrasense.web.routes.alignment import build_alignment_router
from citrasense.web.routes.analysis import build_analysis_router
from citrasense.web.routes.autofocus import build_autofocus_router
from citrasense.web.routes.calibration import build_calibration_router
from citrasense.web.routes.camera import build_camera_router
from citrasense.web.routes.core import build_core_router
from citrasense.web.routes.filters import build_filters_router
from citrasense.web.routes.focuser import build_focuser_router
from citrasense.web.routes.hardware import build_hardware_router
from citrasense.web.routes.jobs import build_jobs_router
from citrasense.web.routes.mount import build_mount_router
from citrasense.web.routes.radar import build_radar_router
from citrasense.web.routes.safety import build_safety_router
from citrasense.web.routes.sensors import build_sensors_router
from citrasense.web.routes.spa_fallback import build_spa_fallback_router
from citrasense.web.routes.tasks import build_tasks_router
from citrasense.web.routes.websocket import build_websocket_router

if TYPE_CHECKING:
    from citrasense.web.app import CitraSenseWebApp


def build_all_routers() -> list[Callable[[CitraSenseWebApp], APIRouter]]:
    """Return all router factories in the order they should be included.

    Order is load-bearing: FastAPI is first-match-wins, and the SPA
    fallback catchall (``build_spa_fallback_router``) must sit at the end
    so it only receives requests that didn't match any real backend
    route.  Don't reorder unless you know what you're doing.
    """
    return [
        build_websocket_router,
        build_core_router,
        build_hardware_router,
        build_tasks_router,
        build_filters_router,
        build_focuser_router,
        build_autofocus_router,
        build_alignment_router,
        build_calibration_router,
        build_mount_router,
        build_radar_router,
        build_sensors_router,
        build_safety_router,
        build_camera_router,
        build_analysis_router,
        build_jobs_router,
        # Keep LAST — catchall for SPA client-side routes.
        build_spa_fallback_router,
    ]


__all__ = [
    "build_alignment_router",
    "build_all_routers",
    "build_analysis_router",
    "build_autofocus_router",
    "build_calibration_router",
    "build_camera_router",
    "build_core_router",
    "build_filters_router",
    "build_focuser_router",
    "build_hardware_router",
    "build_jobs_router",
    "build_mount_router",
    "build_radar_router",
    "build_safety_router",
    "build_sensors_router",
    "build_spa_fallback_router",
    "build_tasks_router",
    "build_websocket_router",
]
